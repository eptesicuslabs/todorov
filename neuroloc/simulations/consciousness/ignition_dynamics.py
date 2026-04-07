import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from brian2 import *

prefs.codegen.target = 'numpy'

seed(42)
np.random.seed(42)

N_processors = 5
N_per_proc = 50
N_total = N_processors * N_per_proc

tau_m = 20 * ms
v_rest = -65 * mV
v_thresh = -50 * mV
v_reset = -65 * mV
tau_ref = 2 * ms
tau_syn = 5 * ms

noise_sigma = 2.0 * mV

eqs = '''
dv/dt = (v_rest - v + I_syn + I_stim) / tau_m + noise_sigma / sqrt(tau_m) * xi : volt (unless refractory)
dI_syn/dt = -I_syn / tau_syn : volt
I_stim : volt
'''

g_local = 0.8 * mV
p_local = 0.15

g_longrange = 0.3 * mV
p_longrange = 0.03

stim_processor = 0
stim_onset = 200 * ms
stim_offset = 350 * ms

weak_stim = 8.0 * mV
strong_stim = 18.0 * mV

sim_duration = 600 * ms
bin_width = 20 * ms

processor_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
processor_labels = ['processor 0 (stimulated)', 'processor 1', 'processor 2',
                    'processor 3', 'processor 4']


def compute_rates(spike_mon, proc_idx, n_neurons, duration, bin_w):
    bins = np.arange(0, float(duration), float(bin_w))
    rates = np.zeros(len(bins) - 1)
    proc_neurons = set(range(proc_idx * n_neurons, (proc_idx + 1) * n_neurons))
    for i in range(len(bins) - 1):
        mask = (spike_mon.t / second >= bins[i]) & (spike_mon.t / second < bins[i + 1])
        spikes_in_bin = np.sum([1 for n in spike_mon.i[mask] if int(n) in proc_neurons])
        rates[i] = spikes_in_bin / (float(bin_w) * n_neurons)
    centers = (bins[:-1] + bins[1:]) / 2
    return centers, rates


def run_simulation(stim_strength, label):
    start_scope()

    neurons = NeuronGroup(N_total, eqs, threshold='v > v_thresh',
                          reset='v = v_reset', refractory=tau_ref, method='euler')
    neurons.v = v_rest + np.random.uniform(-5, 5, N_total) * mV

    local_synapses = []
    for p in range(N_processors):
        start_idx = p * N_per_proc
        end_idx = (p + 1) * N_per_proc
        source = neurons[start_idx:end_idx]
        target = neurons[start_idx:end_idx]
        syn = Synapses(source, target, on_pre='I_syn_post += g_local', name=f'local_{p}')
        syn.connect(p=p_local)
        local_synapses.append(syn)

    longrange_synapses = []
    for p_src in range(N_processors):
        for p_tgt in range(N_processors):
            if p_src == p_tgt:
                continue
            src_start = p_src * N_per_proc
            src_end = (p_src + 1) * N_per_proc
            tgt_start = p_tgt * N_per_proc
            tgt_end = (p_tgt + 1) * N_per_proc
            source = neurons[src_start:src_end]
            target = neurons[tgt_start:tgt_end]
            syn = Synapses(source, target, on_pre='I_syn_post += g_longrange',
                           name=f'lr_{p_src}_{p_tgt}')
            syn.connect(p=p_longrange)
            longrange_synapses.append(syn)

    spike_mon = SpikeMonitor(neurons)

    @network_operation(dt=1 * ms)
    def apply_stimulus(t):
        idx_start = stim_processor * N_per_proc
        idx_end = (stim_processor + 1) * N_per_proc
        if stim_onset <= t < stim_offset:
            neurons.I_stim[idx_start:idx_end] = stim_strength
        else:
            neurons.I_stim[idx_start:idx_end] = 0 * mV

    net = Network(neurons, local_synapses, longrange_synapses, spike_mon, apply_stimulus)
    net.run(sim_duration, report=None)

    return spike_mon


fig, axes = plt.subplots(2, 2, figsize=(16, 12))

conditions = [
    (weak_stim, 'weak stimulus (8 mV) -- no ignition'),
    (strong_stim, 'strong stimulus (18 mV) -- ignition'),
]

for col_idx, (stim_val, condition_label) in enumerate(conditions):
    spike_mon = run_simulation(stim_val, condition_label)

    ax_rate = axes[0, col_idx]
    for p in range(N_processors):
        centers, rates = compute_rates(spike_mon, p, N_per_proc,
                                        float(sim_duration / second),
                                        float(bin_width / second))
        ax_rate.plot(centers * 1000, rates, color=processor_colors[p],
                     linewidth=2, label=processor_labels[p])

    ax_rate.axvspan(float(stim_onset / ms), float(stim_offset / ms),
                    alpha=0.15, color='yellow')
    ax_rate.set_title(condition_label, fontsize=12, fontweight='bold')
    ax_rate.set_xlabel('time (ms)')
    ax_rate.set_ylabel('firing rate (Hz)')
    ax_rate.set_xlim(0, float(sim_duration / ms))
    ax_rate.legend(fontsize=8, loc='upper right')
    ax_rate.set_ylim(0, None)

    ax_raster = axes[1, col_idx]
    for p in range(N_processors):
        proc_mask = (spike_mon.i >= p * N_per_proc) & (spike_mon.i < (p + 1) * N_per_proc)
        ax_raster.scatter(spike_mon.t[proc_mask] / ms,
                          spike_mon.i[proc_mask],
                          s=0.5, c=processor_colors[p], alpha=0.6)

    for p in range(1, N_processors):
        ax_raster.axhline(y=p * N_per_proc, color='gray', linewidth=0.5, alpha=0.5)

    ax_raster.axvspan(float(stim_onset / ms), float(stim_offset / ms),
                      alpha=0.15, color='yellow')
    ax_raster.set_xlabel('time (ms)')
    ax_raster.set_ylabel('neuron index')
    ax_raster.set_xlim(0, float(sim_duration / ms))
    ax_raster.set_ylim(0, N_total)

fig.suptitle('ignition dynamics: weak vs strong stimulus\n'
             '5 processors (50 LIF neurons each), sparse long-range connections',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('C:/Users/deyan/Projects/todorov/neuroloc/simulations/consciousness/ignition_dynamics.png',
            dpi=150, bbox_inches='tight')
plt.close()

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

stim_levels = np.linspace(4.0, 25.0, 15)
mean_nonlocal_rates = []
stimulated_rates = []

for stim_val in stim_levels:
    spike_mon = run_simulation(stim_val * mV, f'sweep_{stim_val}')

    measure_start = float(stim_onset / second) + 0.05
    measure_end = float(stim_offset / second)

    nonlocal_spikes = 0
    nonlocal_neurons = 0
    stim_spikes = 0
    for p in range(N_processors):
        proc_neurons = set(range(p * N_per_proc, (p + 1) * N_per_proc))
        mask = (spike_mon.t / second >= measure_start) & (spike_mon.t / second < measure_end)
        count = np.sum([1 for n in spike_mon.i[mask] if int(n) in proc_neurons])
        if p == stim_processor:
            stim_spikes = count
        else:
            nonlocal_spikes += count
            nonlocal_neurons += N_per_proc

    duration_window = measure_end - measure_start
    if nonlocal_neurons > 0 and duration_window > 0:
        mean_nonlocal_rates.append(nonlocal_spikes / (duration_window * nonlocal_neurons))
    else:
        mean_nonlocal_rates.append(0)
    stimulated_rates.append(stim_spikes / (duration_window * N_per_proc))

ax_sweep = axes2[0]
ax_sweep.plot(stim_levels, mean_nonlocal_rates, 'o-', color='#d62728',
              linewidth=2, markersize=6, label='non-stimulated processors (mean)')
ax_sweep.plot(stim_levels, stimulated_rates, 's-', color='#1f77b4',
              linewidth=2, markersize=6, label='stimulated processor')
ax_sweep.set_xlabel('stimulus strength (mV)')
ax_sweep.set_ylabel('firing rate (Hz)')
ax_sweep.set_title('ignition threshold sweep', fontsize=12, fontweight='bold')
ax_sweep.legend(fontsize=9)
ax_sweep.axvline(x=12.0, color='gray', linestyle='--', alpha=0.5, label='approx threshold')

ax_ratio = axes2[1]
ratio = np.array(mean_nonlocal_rates) / (np.array(stimulated_rates) + 1e-6)
ax_ratio.plot(stim_levels, ratio, 'o-', color='#2ca02c', linewidth=2, markersize=6)
ax_ratio.set_xlabel('stimulus strength (mV)')
ax_ratio.set_ylabel('broadcast ratio (non-local / local)')
ax_ratio.set_title('broadcast ratio vs stimulus strength', fontsize=12, fontweight='bold')
ax_ratio.axvline(x=12.0, color='gray', linestyle='--', alpha=0.5)

fig2.suptitle('ignition threshold analysis\n'
              'nonlinear transition from local to global activation',
              fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig('C:/Users/deyan/Projects/todorov/neuroloc/simulations/consciousness/ignition_threshold.png',
            dpi=150, bbox_inches='tight')
plt.close()

print('saved: ignition_dynamics.png, ignition_threshold.png')
