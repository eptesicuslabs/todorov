from brian2 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

defaultclock.dt = 0.1*ms

tau_pre = 20*ms
tau_post = 20*ms
A_plus = 0.01
A_minus = 0.012
w_max = 1.0
w_init = 0.5

duration = 10*second
input_rate = 20*Hz

correlation_strengths = [0.0, 0.3, 0.6, 0.9]
colors = ['#2c3e50', '#e74c3c', '#2980b9', '#27ae60']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, corr in enumerate(correlation_strengths):
    start_scope()

    eqs_neuron = '''
    dv/dt = -v / (10*ms) : 1
    '''

    pre_neuron = NeuronGroup(1, eqs_neuron, threshold='v > 1', reset='v = 0', method='exact')
    post_neuron = NeuronGroup(1, eqs_neuron, threshold='v > 1', reset='v = 0', method='exact')

    np.random.seed(42 + idx)

    n_spikes = int(float(input_rate) * float(duration))
    pre_spike_times = np.sort(np.random.uniform(0, float(duration), n_spikes))

    correlated_spikes = []
    independent_spikes = np.sort(np.random.uniform(0, float(duration),
                                                    int(n_spikes * (1 - corr))))

    for t_pre in pre_spike_times:
        if np.random.random() < corr:
            delay = np.random.normal(5e-3, 2e-3)
            correlated_spikes.append(t_pre + delay)

    correlated_spikes = np.array(correlated_spikes)
    post_spike_times = np.sort(np.concatenate([correlated_spikes, independent_spikes]))
    post_spike_times = post_spike_times[(post_spike_times > 0) &
                                         (post_spike_times < float(duration))]

    pre_input = SpikeGeneratorGroup(1, np.zeros(len(pre_spike_times), dtype=int),
                                     pre_spike_times * second)
    post_input = SpikeGeneratorGroup(1, np.zeros(len(post_spike_times), dtype=int),
                                      post_spike_times * second)

    pre_drive = Synapses(pre_input, pre_neuron, on_pre='v += 1.5')
    pre_drive.connect()
    post_drive = Synapses(post_input, post_neuron, on_pre='v += 1.5')
    post_drive.connect()

    stdp_eqs = '''
    w : 1
    dapre/dt = -apre / tau_pre : 1 (event-driven)
    dapost/dt = -apost / tau_post : 1 (event-driven)
    '''

    stdp_pre = '''
    apre += A_plus
    w = clip(w + apost, 0, w_max)
    '''

    stdp_post = '''
    apost -= A_minus
    w = clip(w + apre, 0, w_max)
    '''

    S = Synapses(pre_neuron, post_neuron, model=stdp_eqs,
                  on_pre=stdp_pre, on_post=stdp_post)
    S.connect()
    S.w = w_init

    w_mon = StateMonitor(S, 'w', record=True, dt=50*ms)
    pre_spk = SpikeMonitor(pre_neuron)
    post_spk = SpikeMonitor(post_neuron)

    run(duration)

    ax = axes[idx]
    t_plot = w_mon.t / second
    w_plot = w_mon.w[0]
    ax.plot(t_plot, w_plot, color=colors[idx], linewidth=2)
    ax.axhline(y=w_init, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('synaptic weight')
    ax.set_title(f'correlation = {corr}')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, float(duration))

    pre_count = pre_spk.num_spikes
    post_count = post_spk.num_spikes
    final_w = float(w_plot[-1])
    ax.text(0.02, 0.95, f'final w = {final_w:.3f}\npre: {pre_count} spk\npost: {post_count} spk',
            transform=ax.transAxes, verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.suptitle('STDP weight evolution under varying input correlation', fontsize=14, fontweight='bold')
plt.tight_layout()

output_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(output_dir, 'stdp_weight_evolution.png'), dpi=150, bbox_inches='tight')
plt.close()
