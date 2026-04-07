import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from brian2 import *
import os

prefs.codegen.target = 'numpy'

seed(42)
np.random.seed(42)

script_dir = os.path.dirname(os.path.abspath(__file__))

N_exc = 400
N_inh = 100
N_thal = 50
N_total = N_exc + N_inh

duration_stim = 500 * ms
duration_spont = 200 * ms
duration_total = duration_spont + duration_stim + duration_spont

eqs_lif = '''
dv/dt = (-v + V_rest + I_syn + I_ext) / tau_m : volt
I_syn : volt
I_ext : volt
tau_m : second
V_rest : volt
'''

exc_group = NeuronGroup(N_exc, eqs_lif,
                        threshold='v > -50*mV',
                        reset='v = -65*mV',
                        refractory=2*ms,
                        method='euler')
exc_group.v = -65 * mV
exc_group.tau_m = 20 * ms
exc_group.V_rest = -65 * mV
exc_group.I_ext = 0 * mV

inh_group = NeuronGroup(N_inh, eqs_lif,
                        threshold='v > -50*mV',
                        reset='v = -65*mV',
                        refractory=1*ms,
                        method='euler')
inh_group.v = -65 * mV
inh_group.tau_m = 10 * ms
inh_group.V_rest = -65 * mV
inh_group.I_ext = 0 * mV

thal_group = PoissonGroup(N_thal, rates=0*Hz)

preferred_orientation = np.linspace(0, np.pi, N_exc, endpoint=False)
inh_orientation = np.linspace(0, np.pi, N_inh, endpoint=False)

syn_ee = Synapses(exc_group, exc_group, 'w : volt', on_pre='I_syn_post += w')
syn_ee.connect(p=0.15)
for s_idx in range(len(syn_ee)):
    i_pre = syn_ee.i[s_idx]
    i_post = syn_ee.j[s_idx]
    ori_diff = abs(preferred_orientation[i_pre] - preferred_orientation[i_post])
    ori_diff = min(ori_diff, np.pi - ori_diff)
    tuning_factor = np.exp(-ori_diff**2 / (2 * (np.pi/6)**2))
    syn_ee.w[s_idx] = 0.8 * mV * tuning_factor

syn_ei = Synapses(exc_group, inh_group, 'w : volt', on_pre='I_syn_post += w')
syn_ei.connect(p=0.4)
syn_ei.w = 1.2 * mV

syn_ie = Synapses(inh_group, exc_group, 'w : volt', on_pre='I_syn_post -= w')
syn_ie.connect(p=0.4)
syn_ie.w = 2.0 * mV

syn_ii = Synapses(inh_group, inh_group, 'w : volt', on_pre='I_syn_post -= w')
syn_ii.connect(p=0.3)
syn_ii.w = 1.5 * mV

thal_orientation = np.pi / 4
thal_tuning = np.zeros(N_exc)
for i in range(N_exc):
    ori_diff = abs(preferred_orientation[i] - thal_orientation)
    ori_diff = min(ori_diff, np.pi - ori_diff)
    thal_tuning[i] = np.exp(-ori_diff**2 / (2 * (np.pi/4)**2))

syn_thal = Synapses(thal_group, exc_group, 'w : volt', on_pre='I_syn_post += w')
syn_thal.connect()
for s_idx in range(len(syn_thal)):
    i_post = syn_thal.j[s_idx]
    syn_thal.w[s_idx] = 3.0 * mV * thal_tuning[i_post]

spike_mon_exc = SpikeMonitor(exc_group)
spike_mon_inh = SpikeMonitor(inh_group)

rate_mon_exc = PopulationRateMonitor(exc_group)
rate_mon_inh = PopulationRateMonitor(inh_group)

exc_group.I_ext = '2.0*mV * rand()'
inh_group.I_ext = '1.0*mV * rand()'

run(duration_spont)

thal_group.rates = 200 * Hz
exc_group.I_ext = '2.0*mV * rand()'

run(duration_stim)

thal_group.rates = 0 * Hz

run(duration_spont)

fig, axes = plt.subplots(4, 1, figsize=(14, 16))
fig.suptitle('canonical cortical microcircuit simulation\n'
             '400 excitatory + 100 inhibitory neurons, orientation-tuned recurrent connections',
             fontsize=13, y=0.98)

ax = axes[0]
exc_times = spike_mon_exc.t / ms
exc_ids = spike_mon_exc.i
inh_times = spike_mon_inh.t / ms
inh_ids = spike_mon_inh.i

ax.scatter(exc_times, exc_ids, s=0.3, c='#2166ac', alpha=0.5, rasterized=True)
ax.scatter(inh_times, inh_ids + N_exc, s=0.3, c='#b2182b', alpha=0.5, rasterized=True)
ax.axvline(x=duration_spont/ms, color='green', linewidth=1.5, linestyle='--', label='stimulus onset')
ax.axvline(x=(duration_spont + duration_stim)/ms, color='red', linewidth=1.5, linestyle='--', label='stimulus offset')
ax.set_ylabel('neuron index')
ax.set_xlabel('time (ms)')
ax.set_title('raster plot (blue: excitatory, red: inhibitory)')
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim(0, duration_total/ms)

ax = axes[1]
window_ms = 10
rate_times = rate_mon_exc.t / ms
dt_ms = float(defaultclock.dt / ms)
window_bins = max(1, int(window_ms / dt_ms))
kernel = np.ones(window_bins) / window_bins
rate_exc_raw = rate_mon_exc.rate / Hz
rate_inh_raw = rate_mon_inh.rate / Hz
rate_exc_smooth = np.convolve(rate_exc_raw, kernel, mode='same')
rate_inh_smooth = np.convolve(rate_inh_raw, kernel, mode='same')
ax.plot(rate_times, rate_exc_smooth, color='#2166ac', linewidth=1.5, label='excitatory')
ax.plot(rate_times, rate_inh_smooth, color='#b2182b', linewidth=1.5, label='inhibitory')
ax.axvline(x=duration_spont/ms, color='green', linewidth=1.5, linestyle='--')
ax.axvline(x=(duration_spont + duration_stim)/ms, color='red', linewidth=1.5, linestyle='--')
ax.set_ylabel('firing rate (Hz)')
ax.set_xlabel('time (ms)')
ax.set_title('population firing rates (10 ms smoothing)')
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim(0, duration_total/ms)

ax = axes[2]
stim_start = duration_spont / ms
stim_end = (duration_spont + duration_stim) / ms

stim_mask = (exc_times >= stim_start) & (exc_times < stim_end)
stim_spike_ids = exc_ids[stim_mask]

n_ori_bins = 36
ori_bins = np.linspace(0, np.pi, n_ori_bins + 1)
ori_centers = (ori_bins[:-1] + ori_bins[1:]) / 2

firing_rates_stim = np.zeros(n_ori_bins)
for b in range(n_ori_bins):
    neurons_in_bin = np.where((preferred_orientation >= ori_bins[b]) &
                               (preferred_orientation < ori_bins[b+1]))[0]
    if len(neurons_in_bin) > 0:
        spike_count = np.sum(np.isin(stim_spike_ids, neurons_in_bin))
        firing_rates_stim[b] = spike_count / (len(neurons_in_bin) * float(duration_stim/second))

spont_mask = exc_times < stim_start
spont_spike_ids = exc_ids[spont_mask]
firing_rates_spont = np.zeros(n_ori_bins)
for b in range(n_ori_bins):
    neurons_in_bin = np.where((preferred_orientation >= ori_bins[b]) &
                               (preferred_orientation < ori_bins[b+1]))[0]
    if len(neurons_in_bin) > 0:
        spike_count = np.sum(np.isin(spont_spike_ids, neurons_in_bin))
        firing_rates_spont[b] = spike_count / (len(neurons_in_bin) * float(duration_spont/second))

ax.bar(ori_centers * 180 / np.pi, firing_rates_stim, width=180/n_ori_bins * 0.8,
       color='#2166ac', alpha=0.8, label='during stimulus')
ax.bar(ori_centers * 180 / np.pi, firing_rates_spont, width=180/n_ori_bins * 0.5,
       color='#999999', alpha=0.6, label='spontaneous')
ax.axvline(x=thal_orientation * 180 / np.pi, color='green', linewidth=2,
           linestyle='--', label=f'stimulus orientation ({thal_orientation*180/np.pi:.0f} deg)')
ax.set_xlabel('preferred orientation (degrees)')
ax.set_ylabel('firing rate (Hz)')
ax.set_title('orientation tuning: recurrent amplification of thalamic input')
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim(0, 180)

ax = axes[3]

thal_input_strength = thal_tuning.copy()
thal_input_binned = np.zeros(n_ori_bins)
for b in range(n_ori_bins):
    neurons_in_bin = np.where((preferred_orientation >= ori_bins[b]) &
                               (preferred_orientation < ori_bins[b+1]))[0]
    if len(neurons_in_bin) > 0:
        thal_input_binned[b] = np.mean(thal_input_strength[neurons_in_bin])

thal_input_normalized = thal_input_binned / (np.max(thal_input_binned) + 1e-10)
cortical_response_normalized = firing_rates_stim / (np.max(firing_rates_stim) + 1e-10)

ax.plot(ori_centers * 180 / np.pi, thal_input_normalized, color='#66c2a5',
        linewidth=2.5, label='thalamic input (normalized)', linestyle='--')
ax.plot(ori_centers * 180 / np.pi, cortical_response_normalized, color='#2166ac',
        linewidth=2.5, label='cortical response (normalized)')

peak_input = np.max(thal_input_binned)
peak_response = np.max(firing_rates_stim)
baseline_response = np.mean(firing_rates_spont)
if peak_input > 0 and baseline_response >= 0:
    amplification = peak_response / (peak_input * 3.0 + 1e-10)

input_above_half = np.sum(thal_input_normalized > 0.5) * (180.0 / n_ori_bins)
response_above_half = np.sum(cortical_response_normalized > 0.5) * (180.0 / n_ori_bins)

ax.set_xlabel('preferred orientation (degrees)')
ax.set_ylabel('normalized response')
ax.set_title(f'recurrent amplification: thalamic input vs cortical response\n'
             f'input width at half-max: {input_above_half:.0f} deg, '
             f'response width at half-max: {response_above_half:.0f} deg')
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim(0, 180)
ax.set_ylim(0, 1.15)

plt.tight_layout(rect=[0, 0, 1, 0.96])
output_path = os.path.join(script_dir, 'canonical_circuit.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f'saved: {output_path}')
print(f'excitatory spikes: {spike_mon_exc.num_spikes}')
print(f'inhibitory spikes: {spike_mon_inh.num_spikes}')
print(f'peak firing rate (stimulus): {np.max(firing_rates_stim):.1f} Hz')
print(f'mean spontaneous rate: {np.mean(firing_rates_spont):.1f} Hz')
print(f'thalamic input tuning width: {input_above_half:.0f} deg')
print(f'cortical response tuning width: {response_above_half:.0f} deg')
