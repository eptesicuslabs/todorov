from brian2 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

defaultclock.dt = 0.1*ms

N = 100
N_exc = 80
N_inh = 20

tau_m = 20*ms
V_rest = -65*mV
V_th = -50*mV
V_reset = -65*mV
tau_e = 5*ms
tau_i = 10*ms

tau_homeo = 1*second
target_rate = 10*Hz
eta_homeo = 0.001
eta_hebb = 0.0001
w_max_exc = 0.5*mV
w_init_exc = 0.15*mV

duration = 30*second

eqs = '''
dv/dt = (V_rest - v + ge*(0*mV - v)/(10*mV) + gi*(-80*mV - v)/(10*mV)) / tau_m : volt
dge/dt = -ge / tau_e : volt
dgi/dt = -gi / tau_i : volt
rate_avg : Hz
'''

start_scope()

neurons = NeuronGroup(N, eqs, threshold='v > V_th', reset='v = V_reset',
                       refractory=2*ms, method='euler')
neurons.v = V_rest
neurons.rate_avg = target_rate

exc_neurons = neurons[:N_exc]
inh_neurons = neurons[N_exc:]

noise_input = PoissonInput(neurons, 'ge', N=50, rate=5*Hz, weight=0.3*mV)

hebb_only_eqs = '''
w : volt
dapre/dt = -apre / (20*ms) : 1 (event-driven)
dapost/dt = -apost / (20*ms) : 1 (event-driven)
'''

hebb_pre = '''
ge_post += w
apre += 1
w = clip(w + apost * eta_hebb * mV, 0*mV, 10*mV)
'''

hebb_post = '''
apost += 1
w = clip(w + apre * eta_hebb * mV, 0*mV, 10*mV)
'''

S_hebb = Synapses(exc_neurons, exc_neurons, model=hebb_only_eqs,
                   on_pre=hebb_pre, on_post=hebb_post)
S_hebb.connect(p=0.2)
np.random.seed(42)
S_hebb.w = 'rand() * w_init_exc'

initial_weights_hebb = np.array(S_hebb.w / mV).copy()

homeo_eqs = '''
w : volt
dapre/dt = -apre / (20*ms) : 1 (event-driven)
dapost/dt = -apost / (20*ms) : 1 (event-driven)
scaling_factor : 1
'''

homeo_pre = '''
ge_post += w * scaling_factor
apre += 1
w = clip(w + apost * eta_hebb * mV, 0*mV, 10*mV)
'''

homeo_post = '''
apost += 1
w = clip(w + apre * eta_hebb * mV, 0*mV, 10*mV)
'''

S_inh = Synapses(inh_neurons, neurons, on_pre='gi_post += 0.3*mV')
S_inh.connect(p=0.3)

spike_mon = SpikeMonitor(neurons)
rate_mon = PopulationRateMonitor(neurons)

run(duration, report='text')

spike_trains = spike_mon.spike_trains()

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

bin_size = 500*ms
bins = np.arange(0, float(duration) + float(bin_size), float(bin_size))
bin_centers = (bins[:-1] + bins[1:]) / 2

exc_rates = np.zeros(len(bins) - 1)
for i in range(N_exc):
    if i in spike_trains:
        counts, _ = np.histogram(spike_trains[i] / second, bins=bins)
        exc_rates += counts / float(bin_size)
exc_rates /= N_exc

axes[0, 0].plot(bin_centers, exc_rates, color='#2980b9', linewidth=1.5)
axes[0, 0].axhline(y=float(target_rate), color='red', linestyle='--', alpha=0.7, linewidth=1)
axes[0, 0].set_xlabel('time (s)')
axes[0, 0].set_ylabel('firing rate (Hz)')
axes[0, 0].set_title('excitatory population firing rate')

final_weights = np.array(S_hebb.w / mV)

axes[0, 1].hist(initial_weights_hebb, bins=30, alpha=0.6, color='#2c3e50',
                 label='initial', density=True)
axes[0, 1].hist(final_weights, bins=30, alpha=0.6, color='#e74c3c',
                 label='final (Hebbian only)', density=True)
axes[0, 1].set_xlabel('weight (mV)')
axes[0, 1].set_ylabel('density')
axes[0, 1].set_title('weight distribution: Hebbian only')
axes[0, 1].legend(fontsize=8)

weight_evolution = []
check_times = np.linspace(0, float(duration), 20)

axes[1, 0].scatter(initial_weights_hebb, final_weights, s=5, alpha=0.3, color='#2980b9')
max_w = max(initial_weights_hebb.max(), final_weights.max()) * 1.1
axes[1, 0].plot([0, max_w], [0, max_w], 'k--', alpha=0.3, linewidth=1)
axes[1, 0].set_xlabel('initial weight (mV)')
axes[1, 0].set_ylabel('final weight (mV)')
axes[1, 0].set_title('weight correlation: initial vs final')

w_mean_over_time = []
w_std_over_time = []
time_points = np.linspace(0, float(duration), 50)

for t_check in time_points:
    spike_counts = np.zeros(N_exc)
    window = 1.0
    for i in range(N_exc):
        if i in spike_trains:
            times = np.array(spike_trains[i] / second)
            spike_counts[i] = np.sum((times > max(0, t_check - window)) &
                                      (times <= t_check)) / window

    w_mean_over_time.append(np.mean(final_weights))
    w_std_over_time.append(np.std(final_weights))

axes[1, 1].plot(bin_centers, exc_rates, color='#2980b9', linewidth=1.5)
axes[1, 1].set_xlabel('time (s)')
axes[1, 1].set_ylabel('mean firing rate (Hz)')
axes[1, 1].set_title('activity dynamics')
axes[1, 1].axhline(y=float(target_rate), color='red', linestyle='--', alpha=0.7)

weight_cv = np.std(final_weights) / (np.mean(final_weights) + 1e-10)
initial_cv = np.std(initial_weights_hebb) / (np.mean(initial_weights_hebb) + 1e-10)

sorted_initial = np.argsort(initial_weights_hebb)
sorted_final = np.argsort(final_weights)

from scipy.stats import spearmanr
if len(initial_weights_hebb) > 2:
    rho, p_val = spearmanr(initial_weights_hebb, final_weights)
else:
    rho, p_val = 0.0, 1.0

axes[2, 0].bar(['initial', 'final'], [initial_cv, weight_cv],
                color=['#2c3e50', '#e74c3c'], alpha=0.7)
axes[2, 0].set_ylabel('coefficient of variation')
axes[2, 0].set_title('weight heterogeneity preservation')
axes[2, 0].text(0.5, 0.9, f'Spearman rho = {rho:.3f}\np = {p_val:.2e}',
                transform=axes[2, 0].transAxes, ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

total_spikes = spike_mon.num_spikes
mean_rate = total_spikes / (N * float(duration))
summary_text = (
    f'network: {N_exc} exc + {N_inh} inh neurons\n'
    f'mean rate: {mean_rate:.1f} Hz (target: {float(target_rate):.0f} Hz)\n'
    f'initial mean w: {np.mean(initial_weights_hebb):.4f} mV\n'
    f'final mean w: {np.mean(final_weights):.4f} mV\n'
    f'weight growth: {np.mean(final_weights)/np.mean(initial_weights_hebb):.2f}x\n'
    f'rank preservation (rho): {rho:.3f}'
)
axes[2, 1].text(0.1, 0.5, summary_text, transform=axes[2, 1].transAxes,
                fontsize=11, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
axes[2, 1].set_xlim(0, 1)
axes[2, 1].set_ylim(0, 1)
axes[2, 1].axis('off')
axes[2, 1].set_title('summary statistics')

fig.suptitle('Hebbian plasticity with inhibitory stabilization (100 neurons)',
             fontsize=14, fontweight='bold')
plt.tight_layout()

output_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(output_dir, 'homeostatic_scaling.png'), dpi=150, bbox_inches='tight')
plt.close()
