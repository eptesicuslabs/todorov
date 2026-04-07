from brian2 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

defaultclock.dt = 0.1*ms

np.random.seed(42)

n_input = 20
n_output = 5
duration_per_phase = 5*second
n_phases = 4

tau_m = 20*ms
v_rest = -70*mV
v_thresh = -50*mV
v_reset = -65*mV

tau_stdp = 20*ms
a_plus_base = 0.015
a_minus_base = 0.018
w_max = 2.0
w_min = 0.0

plasticity_schedule = [1.0, 0.7, 0.2, 0.05]

input_patterns = {}
input_patterns['A'] = np.zeros(n_input)
input_patterns['A'][:10] = 1.0
input_patterns['B'] = np.zeros(n_input)
input_patterns['B'][10:] = 1.0

base_rate = 5*Hz
stim_rate = 40*Hz

all_weights_A = []
all_weights_B = []
all_times = []
all_plasticity = []

for phase_idx in range(n_phases):
    start_scope()

    plast = plasticity_schedule[phase_idx]
    a_plus = a_plus_base * plast
    a_minus = a_minus_base * plast

    output_eqs = '''
    dv/dt = (v_rest - v) / tau_m : volt
    '''

    output_neurons = NeuronGroup(n_output, output_eqs,
                                  threshold='v > v_thresh',
                                  reset='v = v_reset',
                                  method='exact')
    output_neurons.v = v_rest

    half = int(duration_per_phase / (0.5*second))
    spike_indices = []
    spike_times_list = []

    for seg in range(half):
        t_start = seg * 0.5
        if seg % 2 == 0:
            pattern = input_patterns['A']
        else:
            pattern = input_patterns['B']

        for neuron_idx in range(n_input):
            rate = float(stim_rate) if pattern[neuron_idx] > 0 else float(base_rate)
            n_spikes_expected = int(rate * 0.5)
            times = np.sort(np.random.uniform(t_start, t_start + 0.5, n_spikes_expected))
            for t in times:
                spike_indices.append(neuron_idx)
                spike_times_list.append(t)

    spike_indices = np.array(spike_indices, dtype=int)
    spike_times_arr = np.array(spike_times_list) * second

    sort_idx = np.argsort(spike_times_arr)
    spike_indices = spike_indices[sort_idx]
    spike_times_arr = spike_times_arr[sort_idx]

    input_group = SpikeGeneratorGroup(n_input, spike_indices, spike_times_arr)

    synapse_eqs = '''
    w : 1
    dapre/dt = -apre / tau_stdp : 1 (event-driven)
    dapost/dt = -apost / tau_stdp : 1 (event-driven)
    '''

    synapse_pre = '''
    v_post += w * 4.0 * mV
    apre += a_plus
    w = clip(w + apost, w_min, w_max)
    '''

    synapse_post = '''
    apost -= a_minus
    w = clip(w + apre, w_min, w_max)
    '''

    synapses = Synapses(input_group, output_neurons,
                         synapse_eqs,
                         on_pre=synapse_pre,
                         on_post=synapse_post)
    synapses.connect()

    if phase_idx == 0:
        synapses.w = '0.3 + 0.1 * rand()'
    else:
        prev_w = all_weights_A[-1][:n_input * n_output]
        flat_prev = []
        for i in range(n_input):
            for j in range(n_output):
                flat_prev.append(prev_w[i * n_output + j] if i * n_output + j < len(prev_w) else 0.3)
        synapses.w = flat_prev[:len(synapses.w)]

    output_mon = SpikeMonitor(output_neurons)

    w_snapshots_A = []
    w_snapshots_B = []
    snapshot_times = []

    n_snapshots = 20
    snapshot_interval = duration_per_phase / n_snapshots

    for snap in range(n_snapshots):
        run(snapshot_interval)
        w_matrix = np.array(synapses.w).reshape(n_input, n_output)
        w_snapshots_A.append(np.mean(w_matrix[:10, :], axis=1).mean())
        w_snapshots_B.append(np.mean(w_matrix[10:, :], axis=1).mean())
        t_global = phase_idx * float(duration_per_phase) + (snap + 1) * float(snapshot_interval)
        snapshot_times.append(t_global)

    all_weights_A.extend(w_snapshots_A)
    all_weights_B.extend(w_snapshots_B)
    all_times.extend(snapshot_times)
    all_plasticity.extend([plast] * n_snapshots)

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

ax1 = axes[0]
ax1.plot(all_times, all_weights_A, color='#2980b9', linewidth=2, label='pattern A weights (trained early)')
ax1.plot(all_times, all_weights_B, color='#e74c3c', linewidth=2, label='pattern B weights (trained early)')

for i in range(1, n_phases):
    t_boundary = i * float(duration_per_phase)
    ax1.axvline(x=t_boundary, color='#7f8c8d', linestyle='--', alpha=0.7)

ax1.set_xlabel('time (s)')
ax1.set_ylabel('mean synaptic weight')
ax1.set_title('weight evolution across developmental phases')
ax1.legend(loc='upper left')
ax1.set_xlim(0, n_phases * float(duration_per_phase))

ax2 = axes[1]
ax2.fill_between(all_times, all_plasticity, color='#27ae60', alpha=0.3)
ax2.plot(all_times, all_plasticity, color='#27ae60', linewidth=2)

for i in range(1, n_phases):
    t_boundary = i * float(duration_per_phase)
    ax2.axvline(x=t_boundary, color='#7f8c8d', linestyle='--', alpha=0.7)

phase_labels = ['phase 1\nhigh plasticity\n(immature)', 'phase 2\nmoderate\nplasticity',
                'phase 3\nlow plasticity\n(maturing)', 'phase 4\nminimal plasticity\n(mature)']
for i in range(n_phases):
    t_center = (i + 0.5) * float(duration_per_phase)
    ax2.text(t_center, 0.85, phase_labels[i], ha='center', va='top', fontsize=8)

ax2.set_xlabel('time (s)')
ax2.set_ylabel('plasticity level')
ax2.set_title('plasticity modulation (mimicking PV+ interneuron maturation)')
ax2.set_ylim(0, 1.1)

start_scope()

late_output_eqs = '''
dv/dt = (v_rest - v) / tau_m : volt
'''

late_output = NeuronGroup(n_output, late_output_eqs,
                           threshold='v > v_thresh',
                           reset='v = v_reset',
                           method='exact')
late_output.v = v_rest

late_spike_indices = []
late_spike_times_list = []
late_duration = duration_per_phase

half_late = int(late_duration / (0.5*second))
for seg in range(half_late):
    t_start = seg * 0.5
    if seg % 2 == 0:
        pattern = input_patterns['A']
    else:
        pattern = input_patterns['B']

    for neuron_idx in range(n_input):
        rate = float(stim_rate) if pattern[neuron_idx] > 0 else float(base_rate)
        n_spikes_expected = int(rate * 0.5)
        times = np.sort(np.random.uniform(t_start, t_start + 0.5, n_spikes_expected))
        for t in times:
            late_spike_indices.append(neuron_idx)
            late_spike_times_list.append(t)

late_spike_indices = np.array(late_spike_indices, dtype=int)
late_spike_times_arr = np.array(late_spike_times_list) * second

sort_idx = np.argsort(late_spike_times_arr)
late_spike_indices = late_spike_indices[sort_idx]
late_spike_times_arr = late_spike_times_arr[sort_idx]

late_input = SpikeGeneratorGroup(n_input, late_spike_indices, late_spike_times_arr)

late_a_plus = a_plus_base * 0.05
late_a_minus = a_minus_base * 0.05

late_syn_eqs = '''
w : 1
dapre/dt = -apre / tau_stdp : 1 (event-driven)
dapost/dt = -apost / tau_stdp : 1 (event-driven)
'''

late_syn_pre = f'''
v_post += w * 4.0 * mV
apre += {late_a_plus}
w = clip(w + apost, {w_min}, {w_max})
'''

late_syn_post = f'''
apost -= {late_a_minus}
w = clip(w + apre, {w_min}, {w_max})
'''

late_synapses = Synapses(late_input, late_output,
                          late_syn_eqs,
                          on_pre=late_syn_pre,
                          on_post=late_syn_post)
late_synapses.connect()
late_synapses.w = '0.3 + 0.1 * rand()'

late_weights_A = []
late_weights_B = []
late_snap_times = []

for snap in range(n_snapshots):
    run(snapshot_interval)
    w_matrix = np.array(late_synapses.w).reshape(n_input, n_output)
    late_weights_A.append(np.mean(w_matrix[:10, :], axis=1).mean())
    late_weights_B.append(np.mean(w_matrix[10:, :], axis=1).mean())
    late_snap_times.append((snap + 1) * float(snapshot_interval))

ax3 = axes[2]

early_selectivity = [abs(a - b) for a, b in zip(all_weights_A[:n_snapshots], all_weights_B[:n_snapshots])]
late_selectivity = [abs(a - b) for a, b in zip(late_weights_A, late_weights_B)]

ax3.plot(range(1, n_snapshots + 1), early_selectivity, color='#2980b9', linewidth=2,
         marker='o', markersize=4, label='early training (high plasticity)')
ax3.plot(range(1, n_snapshots + 1), late_selectivity, color='#e74c3c', linewidth=2,
         marker='s', markersize=4, label='late training (low plasticity, same stimuli)')
ax3.set_xlabel('training snapshot')
ax3.set_ylabel('weight selectivity (|pattern A - pattern B|)')
ax3.set_title('critical period effect: same stimuli, different plasticity')
ax3.legend()

plt.tight_layout()

output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, 'critical_period.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f'saved: {output_path}')
print(f'early training final selectivity: {early_selectivity[-1]:.4f}')
print(f'late training final selectivity: {late_selectivity[-1]:.4f}')
print(f'ratio (early/late): {early_selectivity[-1] / max(late_selectivity[-1], 1e-6):.1f}x')
