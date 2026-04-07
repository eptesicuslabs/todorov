from brian2 import *
import matplotlib.pyplot as plt

defaultclock.dt = 0.1 * ms

eqs = '''
dv/dt = (-g_L * (v - E_L) + g_L * delta_T * exp((v - V_T) / delta_T) - w + I_ext) / C : volt
dw/dt = (a * (v - E_L) - w) / tau_w : amp
C : farad
g_L : siemens
E_L : volt
delta_T : volt
V_T : volt
a : siemens
tau_w : second
b : amp
V_r : volt
I_ext : amp
'''

patterns = {
    'regular spiking': dict(
        C=200*pF, g_L=10*nS, E_L=-70*mV, V_T=-50*mV, delta_T=2*mV,
        a=2*nS, tau_w=30*ms, b=0*pA, V_r=-58*mV, I_ext=500*pA,
    ),
    'adaptation': dict(
        C=200*pF, g_L=12*nS, E_L=-70*mV, V_T=-50*mV, delta_T=2*mV,
        a=2*nS, tau_w=300*ms, b=60*pA, V_r=-58*mV, I_ext=500*pA,
    ),
    'initial bursting': dict(
        C=130*pF, g_L=18*nS, E_L=-58*mV, V_T=-50*mV, delta_T=2*mV,
        a=4*nS, tau_w=150*ms, b=120*pA, V_r=-50*mV, I_ext=400*pA,
    ),
    'regular bursting': dict(
        C=200*pF, g_L=10*nS, E_L=-58*mV, V_T=-50*mV, delta_T=2*mV,
        a=2*nS, tau_w=120*ms, b=100*pA, V_r=-46*mV, I_ext=210*pA,
    ),
}

n_patterns = len(patterns)
duration = 500 * ms

fig, axes = plt.subplots(n_patterns, 1, figsize=(12, 3 * n_patterns), sharex=True)

for idx, (name, params) in enumerate(patterns.items()):
    start_scope()
    defaultclock.dt = 0.1 * ms

    G = NeuronGroup(
        1,
        eqs,
        threshold='v > 0*mV',
        reset='v = V_r; w += b',
        method='euler',
    )

    G.v = params['E_L']
    G.w = 0 * pA
    G.C = params['C']
    G.g_L = params['g_L']
    G.E_L = params['E_L']
    G.delta_T = params['delta_T']
    G.V_T = params['V_T']
    G.a = params['a']
    G.tau_w = params['tau_w']
    G.b = params['b']
    G.V_r = params['V_r']
    G.I_ext = params['I_ext']

    state_mon = StateMonitor(G, ['v', 'w'], record=0)
    spike_mon = SpikeMonitor(G)

    run(duration)

    ax = axes[idx]
    ax.plot(state_mon.t / ms, state_mon.v[0] / mV, 'k-', linewidth=0.8)
    ax.set_ylabel('V (mV)')
    ax.set_title(f'{name} (b={params["b"]/pA:.0f} pA, tau_w={params["tau_w"]/ms:.0f} ms, a={params["a"]/nS:.0f} nS)')
    ax.set_ylim(-80, 10)

    n_spikes = spike_mon.num_spikes
    if n_spikes > 0:
        rate = n_spikes / float(duration / second)
        ax.text(0.98, 0.95, f'{rate:.0f} Hz', transform=ax.transAxes,
                ha='right', va='top', fontsize=9, color='gray')

axes[-1].set_xlabel('time (ms)')
plt.tight_layout()
plt.savefig('neuroloc/simulations/single_neuron/adex_patterns.png', dpi=150, bbox_inches='tight')
plt.close()

print(f'saved: adex_patterns.png')
for name, params in patterns.items():
    print(f'  {name}: b={params["b"]/pA:.0f} pA, tau_w={params["tau_w"]/ms:.0f} ms')
