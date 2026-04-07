from brian2 import *
import matplotlib.pyplot as plt
import numpy as np

defaultclock.dt = 0.05 * ms

C_s = 1.0 * uF / cm**2
C_d = 0.8 * uF / cm**2
area_s = 1500 * um**2
area_d = 800 * um**2
g_L = 0.05 * mS / cm**2
V_rest = -65 * mV
V_th_na = -50 * mV
V_reset = -65 * mV
E_Na = 50 * mV
E_K = -80 * mV
E_Ca = 120 * mV
g_c = 0.5 * mS / cm**2

eqs = '''
dV_s/dt = (-g_L * (V_s - V_rest) - I_Na_s - I_K_s + g_c * (V_d - V_s) + I_basal / area_s) / C_s : volt
dV_d/dt = (-g_L * (V_d - V_rest) - I_Ca_d + g_c * (V_s - V_d) + I_apical / area_d) / C_d : volt

dm_Na/dt = (m_Na_inf - m_Na) / tau_m_Na : 1
dh_Na/dt = (h_Na_inf - h_Na) / tau_h_Na : 1
dn_K/dt = (n_K_inf - n_K) / tau_n_K : 1

ds_Ca/dt = (s_Ca_inf - s_Ca) / tau_s_Ca : 1
dr_Ca/dt = (r_Ca_inf - r_Ca) / tau_r_Ca : 1

m_Na_inf = 1.0 / (1.0 + exp(-(V_s/mV + 40.0) / 6.0)) : 1
h_Na_inf = 1.0 / (1.0 + exp((V_s/mV + 45.0) / 4.0)) : 1
n_K_inf = 1.0 / (1.0 + exp(-(V_s/mV + 40.0) / 10.0)) : 1
tau_m_Na = 0.15 * ms : second
tau_h_Na = 2.0 * ms : second
tau_n_K = 3.0 * ms : second

s_Ca_inf = 1.0 / (1.0 + exp(-(V_d/mV + 20.0) / 5.0)) : 1
r_Ca_inf = 1.0 / (1.0 + exp((V_d/mV + 40.0) / 5.0)) : 1
tau_s_Ca = 5.0 * ms : second
tau_r_Ca = 50.0 * ms : second

I_Na_s = 50.0 * mS/cm**2 * m_Na**3 * h_Na * (V_s - E_Na) : amp/meter**2
I_K_s = 10.0 * mS/cm**2 * n_K**4 * (V_s - E_K) : amp/meter**2
I_Ca_d = 3.0 * mS/cm**2 * s_Ca**2 * r_Ca * (V_d - E_Ca) : amp/meter**2

I_basal : amp
I_apical : amp
'''

N = 3
G = NeuronGroup(
    N,
    eqs,
    threshold='V_s > V_th_na',
    reset='V_s = V_reset',
    refractory=3 * ms,
    method='euler',
)

G.V_s = V_rest
G.V_d = V_rest
G.m_Na = 0.0
G.h_Na = 1.0
G.n_K = 0.0
G.s_Ca = 0.0
G.r_Ca = 1.0
G.I_basal = 0.0 * nA
G.I_apical = 0.0 * nA

state_mon = StateMonitor(G, ['V_s', 'V_d'], record=True)
spike_mon = SpikeMonitor(G)

t_basal_start = 50 * ms
t_basal_end = 55 * ms
t_apical_start = 52 * ms
t_apical_end = 72 * ms

I_basal_amp = 2.5 * nA
I_apical_amp = 1.8 * nA

@network_operation(dt=defaultclock.dt)
def update_inputs(t):
    G.I_basal[0] = I_basal_amp if t_basal_start <= t < t_basal_end else 0.0 * nA
    G.I_apical[0] = 0.0 * nA

    G.I_basal[1] = 0.0 * nA
    G.I_apical[1] = I_apical_amp if t_apical_start <= t < t_apical_end else 0.0 * nA

    G.I_basal[2] = I_basal_amp if t_basal_start <= t < t_basal_end else 0.0 * nA
    G.I_apical[2] = I_apical_amp if t_apical_start <= t < t_apical_end else 0.0 * nA

net = Network(G, state_mon, spike_mon, update_inputs)
net.run(150 * ms)

fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)

titles = [
    'bottom-up only (basal input, no apical)',
    'top-down only (apical input, no basal)',
    'coincident (basal + apical = BAC firing)',
]

colors_soma = '#c0392b'
colors_dend = '#2980b9'

for row in range(3):
    t_plot = state_mon.t / ms

    ax_s = axes[row, 0]
    ax_s.plot(t_plot, state_mon.V_s[row] / mV, color=colors_soma, linewidth=1.5)
    ax_s.set_ylabel('V_soma (mV)')
    ax_s.set_title(f'{titles[row]} -- soma')
    ax_s.set_ylim(-80, 60)
    ax_s.axhline(y=V_th_na / mV, color='gray', linestyle=':', linewidth=0.8)

    spike_times = spike_mon.t[spike_mon.i == row] / ms
    n_spikes = len(spike_times)
    ax_s.text(
        0.02, 0.95, f'{n_spikes} spike{"s" if n_spikes != 1 else ""}',
        transform=ax_s.transAxes, fontsize=10, verticalalignment='top',
        fontweight='bold',
    )

    if t_basal_start / ms < 150:
        if row in [0, 2]:
            ax_s.axvspan(t_basal_start / ms, t_basal_end / ms, alpha=0.15, color='red', label='basal input')

    ax_d = axes[row, 1]
    ax_d.plot(t_plot, state_mon.V_d[row] / mV, color=colors_dend, linewidth=1.5)
    ax_d.set_ylabel('V_dendrite (mV)')
    ax_d.set_title(f'{titles[row]} -- apical dendrite')
    ax_d.set_ylim(-80, 60)

    if row in [1, 2]:
        ax_d.axvspan(t_apical_start / ms, t_apical_end / ms, alpha=0.15, color='blue', label='apical input')

    if row == 0:
        ax_s.legend(loc='upper right', fontsize=8)
    if row == 1:
        ax_d.legend(loc='upper right', fontsize=8)
    if row == 2:
        ax_s.legend(loc='upper right', fontsize=8)
        ax_d.legend(loc='upper right', fontsize=8)

axes[2, 0].set_xlabel('time (ms)')
axes[2, 1].set_xlabel('time (ms)')

plt.suptitle(
    'two-compartment neuron: BAC firing requires coincident basal + apical input',
    fontsize=13, fontweight='bold', y=0.98,
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('neuroloc/simulations/dendritic/bac_firing.png', dpi=150, bbox_inches='tight')
plt.close()

for i, label in enumerate(['bottom-up only', 'top-down only', 'coincident (BAC)']):
    spikes = spike_mon.t[spike_mon.i == i]
    print(f'{label}: {len(spikes)} somatic spike(s)')
    if len(spikes) > 0:
        print(f'  spike times: {[f"{t/ms:.1f} ms" for t in spikes]}')

print(f'saved: bac_firing.png')
