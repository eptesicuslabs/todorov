import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from brian2 import *

prefs.codegen.target = 'numpy'

seed(42)
np.random.seed(42)

N_pop = 200
N_per_pool = 100
pool_A_idx = slice(0, N_per_pool)
pool_B_idx = slice(N_per_pool, N_pop)

stim_drive_A = 3.5 * nA
stim_drive_B = 3.5 * nA
bias_strength = 2.0 * nA
noise_sigma = 0.5 * nA
g_cross_inhibit = 0.15
g_within_inhibit = 0.05
sim_duration = 500 * ms
bias_onset = 200 * ms
bias_offset = 500 * ms

eqs = '''
dv/dt = (-v + I_stim + I_bias - I_inh_cross - I_inh_within) / (10*ms) + noise_sigma/sqrt(10*ms) * xi : volt
I_stim : volt
I_bias : volt
I_inh_cross : volt
I_inh_within : volt
'''

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

bias_levels = [0.0, 1.0, 2.0]
bias_labels = ['no bias', 'moderate bias (1.0 nA)', 'strong bias (2.0 nA)']

for col_idx, (bias_val, bias_label) in enumerate(zip(bias_levels, bias_labels)):
    start_scope()

    neurons = NeuronGroup(N_pop, eqs, threshold='v > 1*volt',
                          reset='v = 0*volt', refractory=2*ms, method='euler')
    neurons.v = 0 * volt

    neurons.I_stim[pool_A_idx] = stim_drive_A / nA * volt / 1000
    neurons.I_stim[pool_B_idx] = stim_drive_B / nA * volt / 1000
    neurons.I_bias = 0 * volt
    neurons.I_inh_cross = 0 * volt
    neurons.I_inh_within = 0 * volt

    input_A = np.ones(N_per_pool) * stim_drive_A / nA
    input_B = np.ones(N_per_pool) * stim_drive_B / nA
    noise_A = noise_sigma / nA * np.random.randn(N_per_pool)
    noise_B = noise_sigma / nA * np.random.randn(N_per_pool)

    neurons.I_stim[pool_A_idx] = (input_A + noise_A) * volt / 1000
    neurons.I_stim[pool_B_idx] = (input_B + noise_B) * volt / 1000

    spike_mon = SpikeMonitor(neurons)
    rate_A = PopulationRateMonitor(neurons[pool_A_idx])
    rate_B = PopulationRateMonitor(neurons[pool_B_idx])

    run(bias_onset)

    neurons.I_bias[pool_A_idx] = bias_val * volt / 1000

    inh_eq_cross = '''
    I_inh_cross_post = g_cross * rate_pre : volt (summed)
    g_cross : volt/Hz
    rate_pre : Hz
    '''

    run(bias_offset - bias_onset)

    t_A = rate_A.t / ms
    r_A = rate_A.smooth_rate(window='flat', width=20*ms) / Hz
    t_B = rate_B.t / ms
    r_B = rate_B.smooth_rate(window='flat', width=20*ms) / Hz

    axes[0, col_idx].plot(t_A, r_A, color='#2196F3', linewidth=2, label='pool A (biased)')
    axes[0, col_idx].plot(t_B, r_B, color='#F44336', linewidth=2, label='pool B (unbiased)')
    axes[0, col_idx].axvline(x=bias_onset/ms, color='gray', linestyle='--', linewidth=1)
    axes[0, col_idx].set_title(bias_label, fontsize=12)
    axes[0, col_idx].set_ylabel('firing rate (Hz)', fontsize=10)
    axes[0, col_idx].legend(fontsize=9, loc='upper left')
    axes[0, col_idx].set_xlim(0, sim_duration/ms)
    axes[0, col_idx].set_ylim(0, None)

    spike_trains_A = spike_mon.spike_trains()
    spike_trains_B = spike_mon.spike_trains()

    for i in range(N_per_pool):
        spikes_i = spike_trains_A[i] / ms
        post_bias = spikes_i[spikes_i > bias_onset/ms]
        if len(post_bias) > 0:
            axes[1, col_idx].plot(post_bias, np.ones_like(post_bias) * i,
                                  '.', color='#2196F3', markersize=1)
    for i in range(N_per_pool, N_pop):
        spikes_i = spike_trains_B[i] / ms
        post_bias = spikes_i[spikes_i > bias_onset/ms]
        if len(post_bias) > 0:
            axes[1, col_idx].plot(post_bias, np.ones_like(post_bias) * i,
                                  '.', color='#F44336', markersize=1)

    axes[1, col_idx].axhline(y=N_per_pool, color='black', linestyle='-', linewidth=0.5)
    axes[1, col_idx].axvline(x=bias_onset/ms, color='gray', linestyle='--', linewidth=1)
    axes[1, col_idx].set_xlabel('time (ms)', fontsize=10)
    axes[1, col_idx].set_ylabel('neuron index', fontsize=10)
    axes[1, col_idx].set_xlim(0, sim_duration/ms)
    axes[1, col_idx].set_ylim(0, N_pop)

fig.suptitle('biased competition: two populations competing for representation',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('biased_competition.png', dpi=150, bbox_inches='tight')
plt.close()

print('saved biased_competition.png')

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

bias_sweep = np.linspace(0, 4.0, 20)
rate_diff = []
selectivity = []

for bias_val in bias_sweep:
    start_scope()

    neurons = NeuronGroup(N_pop, eqs, threshold='v > 1*volt',
                          reset='v = 0*volt', refractory=2*ms, method='euler')
    neurons.v = 0 * volt

    input_A = np.ones(N_per_pool) * stim_drive_A / nA
    input_B = np.ones(N_per_pool) * stim_drive_B / nA
    noise_A = noise_sigma / nA * np.random.randn(N_per_pool)
    noise_B = noise_sigma / nA * np.random.randn(N_per_pool)

    neurons.I_stim[pool_A_idx] = (input_A + noise_A) * volt / 1000
    neurons.I_stim[pool_B_idx] = (input_B + noise_B) * volt / 1000
    neurons.I_bias = 0 * volt
    neurons.I_inh_cross = 0 * volt
    neurons.I_inh_within = 0 * volt
    neurons.I_bias[pool_A_idx] = bias_val * volt / 1000

    spike_mon = SpikeMonitor(neurons)

    run(300 * ms)

    spikes_A = sum(1 for i in range(N_per_pool) for _ in spike_mon.spike_trains()[i])
    spikes_B = sum(1 for i in range(N_per_pool, N_pop) for _ in spike_mon.spike_trains()[i])

    r_A_mean = spikes_A / (N_per_pool * 0.3)
    r_B_mean = spikes_B / (N_per_pool * 0.3)

    rate_diff.append(r_A_mean - r_B_mean)
    if r_A_mean + r_B_mean > 0:
        selectivity.append((r_A_mean - r_B_mean) / (r_A_mean + r_B_mean))
    else:
        selectivity.append(0)

axes2[0].plot(bias_sweep, rate_diff, 'o-', color='#2196F3', linewidth=2)
axes2[0].set_xlabel('bias strength (nA)', fontsize=11)
axes2[0].set_ylabel('rate difference (A - B, Hz)', fontsize=11)
axes2[0].set_title('rate difference vs bias', fontsize=12)
axes2[0].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

axes2[1].plot(bias_sweep, selectivity, 's-', color='#4CAF50', linewidth=2)
axes2[1].set_xlabel('bias strength (nA)', fontsize=11)
axes2[1].set_ylabel('selectivity index (A-B)/(A+B)', fontsize=11)
axes2[1].set_title('selectivity vs bias', fontsize=12)
axes2[1].set_ylim(-0.1, 1.0)

stim_ratios = [0.5, 0.75, 1.0, 1.5, 2.0]
winner_rates = []
loser_rates = []

for ratio in stim_ratios:
    start_scope()

    neurons = NeuronGroup(N_pop, eqs, threshold='v > 1*volt',
                          reset='v = 0*volt', refractory=2*ms, method='euler')
    neurons.v = 0 * volt

    drive_A = stim_drive_A / nA * ratio
    drive_B = stim_drive_B / nA
    noise_A = noise_sigma / nA * np.random.randn(N_per_pool)
    noise_B = noise_sigma / nA * np.random.randn(N_per_pool)

    neurons.I_stim[pool_A_idx] = (np.ones(N_per_pool) * drive_A + noise_A) * volt / 1000
    neurons.I_stim[pool_B_idx] = (np.ones(N_per_pool) * drive_B + noise_B) * volt / 1000
    neurons.I_bias = 0 * volt
    neurons.I_inh_cross = 0 * volt
    neurons.I_inh_within = 0 * volt
    neurons.I_bias[pool_A_idx] = 1.5 * volt / 1000

    spike_mon = SpikeMonitor(neurons)
    run(300 * ms)

    spikes_A = sum(1 for i in range(N_per_pool) for _ in spike_mon.spike_trains()[i])
    spikes_B = sum(1 for i in range(N_per_pool, N_pop) for _ in spike_mon.spike_trains()[i])

    winner_rates.append(spikes_A / (N_per_pool * 0.3))
    loser_rates.append(spikes_B / (N_per_pool * 0.3))

axes2[2].plot(stim_ratios, winner_rates, 'o-', color='#2196F3', linewidth=2, label='pool A (biased)')
axes2[2].plot(stim_ratios, loser_rates, 's-', color='#F44336', linewidth=2, label='pool B (unbiased)')
axes2[2].set_xlabel('stimulus ratio (A/B)', fontsize=11)
axes2[2].set_ylabel('firing rate (Hz)', fontsize=11)
axes2[2].set_title('competition outcome vs stimulus strength', fontsize=12)
axes2[2].legend(fontsize=9)

fig2.suptitle('biased competition: parametric analysis',
              fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('biased_competition_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print('saved biased_competition_analysis.png')
