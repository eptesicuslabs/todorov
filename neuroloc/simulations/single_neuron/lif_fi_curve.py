from brian2 import *
import matplotlib.pyplot as plt
import numpy as np

defaultclock.dt = 0.1 * ms

tau_m = 20 * ms
V_rest = -65 * mV
V_th = -50 * mV
V_reset = -65 * mV
R = 100 * Mohm
t_ref = 2 * ms

eqs = '''
dv/dt = (-(v - V_rest) + R * I_ext) / tau_m : volt (unless refractory)
I_ext : amp
'''

n_neurons = 50
I_min = 0.0 * nA
I_max = 1.0 * nA
I_values = np.linspace(I_min / nA, I_max / nA, n_neurons) * nA

G = NeuronGroup(
    n_neurons,
    eqs,
    threshold='v > V_th',
    reset='v = V_reset',
    refractory=t_ref,
    method='euler',
)
G.v = V_rest
G.I_ext = I_values

spike_mon = SpikeMonitor(G)
state_mon = StateMonitor(G, 'v', record=[0, n_neurons // 4, n_neurons // 2, 3 * n_neurons // 4, n_neurons - 1])

duration = 1.0 * second
run(duration)

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

firing_rates = np.zeros(n_neurons)
for i in range(n_neurons):
    spike_times = spike_mon.t[spike_mon.i == i]
    if len(spike_times) > 1:
        firing_rates[i] = len(spike_times) / float(duration / second)

I_rheo = (V_th - V_rest) / R
I_plot = I_values / nA

axes[0].plot(I_plot, firing_rates, 'k-', linewidth=2)
axes[0].axvline(x=float(I_rheo / nA), color='gray', linestyle='--', linewidth=1, label=f'rheobase = {I_rheo/nA:.3f} nA')
axes[0].set_xlabel('input current (nA)')
axes[0].set_ylabel('firing rate (Hz)')
axes[0].set_title('LIF f-I curve')
axes[0].legend()
axes[0].set_xlim(0, I_max / nA)

I_theory = np.linspace(float(I_rheo / nA) + 0.001, float(I_max / nA), 200) * nA
f_theory = np.zeros(len(I_theory))
for i, I_val in enumerate(I_theory):
    numerator = R * I_val + V_rest - V_reset
    denominator = R * I_val + V_rest - V_th
    if numerator > 0 * mV and denominator > 0 * mV:
        f_theory[i] = 1.0 / (float(t_ref / second) + float(tau_m / second) * np.log(float(numerator / denominator)))

axes[0].plot(I_theory / nA, f_theory, 'r--', linewidth=1.5, label='analytical')
axes[0].legend()

trace_indices = [0, n_neurons // 4, n_neurons // 2, 3 * n_neurons // 4, n_neurons - 1]
colors = ['#2c3e50', '#2980b9', '#27ae60', '#e67e22', '#c0392b']
t_window = 200 * ms
t_mask = state_mon.t < t_window

for idx, (rec_idx, color) in enumerate(zip(range(len(trace_indices)), colors)):
    I_label = f'{I_values[trace_indices[idx]]/nA:.2f} nA'
    axes[1].plot(
        state_mon.t[t_mask] / ms,
        state_mon.v[rec_idx][t_mask] / mV,
        color=color,
        linewidth=1,
        label=I_label,
    )

axes[1].axhline(y=V_th / mV, color='gray', linestyle=':', linewidth=1, label=f'threshold = {V_th/mV:.0f} mV')
axes[1].set_xlabel('time (ms)')
axes[1].set_ylabel('membrane potential (mV)')
axes[1].set_title('LIF membrane potential traces (first 200 ms)')
axes[1].legend(loc='upper right', fontsize=8)
axes[1].set_xlim(0, t_window / ms)

plt.tight_layout()
plt.savefig('neuroloc/simulations/single_neuron/fi_curve.png', dpi=150, bbox_inches='tight')
plt.close()

print(f'rheobase current: {I_rheo/nA:.4f} nA')
print(f'max firing rate: {max(firing_rates):.1f} Hz')
print(f'theoretical max (1/t_ref): {1.0/float(t_ref/second):.1f} Hz')
print(f'saved: fi_curve.png')
