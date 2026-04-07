import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from brian2 import *

prefs.codegen.target = 'numpy'
defaultclock.dt = 0.5*ms

N_sensory = 50
N_error = 50
N_representation = 20

tau_sensory = 10*ms
tau_error = 5*ms
tau_rep = 20*ms
v_rest = -65*mV
v_thresh = -50*mV
v_reset = -70*mV
tau_ref = 2*ms

n_patterns = 4
patterns = np.zeros((n_patterns, N_sensory))
for i in range(n_patterns):
    active = np.random.choice(N_sensory, size=N_sensory//4, replace=False)
    patterns[i, active] = 1.0

pattern_duration = 200*ms
n_presentations = 40
total_time = n_presentations * pattern_duration

pattern_sequence = np.tile(np.arange(n_patterns), n_presentations // n_patterns + 1)[:n_presentations]

sensory_eqs = '''
dv/dt = (v_rest - v + I_ext * 20*mV) / tau_sensory : volt (unless refractory)
I_ext : 1
'''

error_eqs = '''
dv/dt = (v_rest - v + (I_bottom_up - I_top_down) * 20*mV) / tau_error : volt (unless refractory)
I_bottom_up : 1
I_top_down : 1
'''

rep_eqs = '''
dv/dt = (v_rest - v + I_error * 20*mV) / tau_rep : volt (unless refractory)
I_error : 1
'''

sensory_group = NeuronGroup(N_sensory, sensory_eqs, threshold='v > v_thresh',
                            reset='v = v_reset', refractory=tau_ref, method='euler')
sensory_group.v = v_rest

error_group = NeuronGroup(N_error, error_eqs, threshold='v > v_thresh',
                          reset='v = v_reset', refractory=tau_ref, method='euler')
error_group.v = v_rest

rep_group = NeuronGroup(N_representation, rep_eqs, threshold='v > v_thresh',
                        reset='v = v_reset', refractory=tau_ref, method='euler')
rep_group.v = v_rest

W_sensory_to_error = np.eye(N_sensory, N_error) * 0.5

syn_s2e = Synapses(sensory_group, error_group, 'w : 1', on_pre='I_bottom_up_post += w')
syn_s2e.connect()
syn_s2e.w = W_sensory_to_error.flatten()

W_rep_to_error = np.random.randn(N_representation, N_error) * 0.1

syn_r2e = Synapses(rep_group, error_group, 'w : 1', on_pre='I_top_down_post += w')
syn_r2e.connect()
syn_r2e.w = W_rep_to_error.flatten()

W_error_to_rep = np.random.randn(N_error, N_representation) * 0.1

syn_e2r = Synapses(error_group, rep_group, 'w : 1', on_pre='I_error_post += w')
syn_e2r.connect()
syn_e2r.w = W_error_to_rep.flatten()

sensory_mon = SpikeMonitor(sensory_group)
error_mon = SpikeMonitor(error_group)
rep_mon = SpikeMonitor(rep_group)

error_rate_mon = PopulationRateMonitor(error_group)
rep_rate_mon = PopulationRateMonitor(rep_group)

error_rates_over_time = []
rep_rates_over_time = []
presentation_times = []

for p_idx in range(n_presentations):
    pat = patterns[pattern_sequence[p_idx]]

    sensory_group.I_ext = pat * 2.0

    error_group.I_bottom_up = 0
    error_group.I_top_down = 0
    rep_group.I_error = 0

    run(pattern_duration)

    t_start = p_idx * float(pattern_duration)
    t_end = (p_idx + 1) * float(pattern_duration)

    e_mask = (np.array(error_mon.t/ms) >= t_start*1000) & (np.array(error_mon.t/ms) < t_end*1000)
    e_count = np.sum(e_mask)
    e_rate = e_count / (N_error * float(pattern_duration/second))

    r_mask = (np.array(rep_mon.t/ms) >= t_start*1000) & (np.array(rep_mon.t/ms) < t_end*1000)
    r_count = np.sum(r_mask)
    r_rate = r_count / (N_representation * float(pattern_duration/second))

    error_rates_over_time.append(e_rate)
    rep_rates_over_time.append(r_rate)
    presentation_times.append(p_idx)

    if p_idx > 0 and p_idx % 5 == 0:
        lr = 0.002
        e_spikes_idx = np.array(error_mon.i)[e_mask]
        r_spikes_idx = np.array(rep_mon.i)[r_mask]

        e_activity = np.zeros(N_error)
        for idx in e_spikes_idx:
            e_activity[idx] += 1
        e_activity = e_activity / (float(pattern_duration/second) + 1e-10)

        r_activity = np.zeros(N_representation)
        for idx in r_spikes_idx:
            r_activity[idx] += 1
        r_activity = r_activity / (float(pattern_duration/second) + 1e-10)

        dW_e2r = lr * np.outer(e_activity, r_activity)
        W_error_to_rep = W_error_to_rep + dW_e2r
        W_error_to_rep = np.clip(W_error_to_rep, -2.0, 2.0)
        syn_e2r.w = W_error_to_rep.flatten()

        dW_r2e = lr * np.outer(r_activity, e_activity)
        W_rep_to_error = W_rep_to_error + dW_r2e
        W_rep_to_error = np.clip(W_rep_to_error, -2.0, 2.0)
        syn_r2e.w = W_rep_to_error.flatten()

fig, axes = plt.subplots(4, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 2, 2, 1.5]})

ax = axes[0]
if len(sensory_mon.t) > 0:
    ax.scatter(sensory_mon.t/ms, sensory_mon.i, s=0.5, c='black', alpha=0.3)
ax.set_ylabel('sensory neuron')
ax.set_title('level 1: sensory input layer')
ax.set_xlim(0, float(total_time/ms))

for p_idx in range(min(n_presentations, 40)):
    t_start = p_idx * float(pattern_duration/ms)
    pat_id = pattern_sequence[p_idx]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    ax.axvspan(t_start, t_start + float(pattern_duration/ms), alpha=0.05, color=colors[pat_id])

ax = axes[1]
if len(error_mon.t) > 0:
    ax.scatter(error_mon.t/ms, error_mon.i, s=0.5, c='red', alpha=0.3)
ax.set_ylabel('error neuron')
ax.set_title('prediction error units (superficial pyramidal analogs)')
ax.set_xlim(0, float(total_time/ms))

ax = axes[2]
if len(rep_mon.t) > 0:
    ax.scatter(rep_mon.t/ms, rep_mon.i, s=0.5, c='blue', alpha=0.3)
ax.set_ylabel('representation neuron')
ax.set_title('representation units (deep pyramidal analogs)')
ax.set_xlim(0, float(total_time/ms))

ax = axes[3]
ax.plot(presentation_times, error_rates_over_time, 'r-o', markersize=3, label='error unit rate', linewidth=1.5)
ax.plot(presentation_times, rep_rates_over_time, 'b-s', markersize=3, label='representation unit rate', linewidth=1.5)
ax.set_xlabel('presentation number')
ax.set_ylabel('firing rate (Hz)')
ax.set_title('population firing rates across presentations')
ax.legend(loc='upper right', fontsize=8)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('predictive_coding_demo.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"simulation complete: {n_presentations} presentations of {n_patterns} patterns")
print(f"sensory spikes: {sensory_mon.num_spikes}")
print(f"error spikes: {error_mon.num_spikes}")
print(f"representation spikes: {rep_mon.num_spikes}")
print(f"final error rate: {error_rates_over_time[-1]:.1f} Hz")
print(f"final representation rate: {rep_rates_over_time[-1]:.1f} Hz")
if len(error_rates_over_time) > 10:
    early_error = np.mean(error_rates_over_time[:5])
    late_error = np.mean(error_rates_over_time[-5:])
    print(f"error rate change: {early_error:.1f} -> {late_error:.1f} Hz ({(late_error-early_error)/early_error*100:.1f}%)")
