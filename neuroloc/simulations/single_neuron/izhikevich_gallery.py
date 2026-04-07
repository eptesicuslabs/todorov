import numpy as np
import matplotlib.pyplot as plt


def simulate_izhikevich(a, b, c, d, I_func, T=500, dt=0.5):
    steps = int(T / dt)
    v = np.zeros(steps)
    u = np.zeros(steps)
    v[0] = c
    u[0] = b * c
    t = np.arange(steps) * dt

    for i in range(1, steps):
        I = I_func(t[i])
        if v[i - 1] >= 30.0:
            v[i - 1] = 30.0
            v[i] = c
            u[i] = u[i - 1] + d
        else:
            dv = 0.04 * v[i - 1] ** 2 + 5.0 * v[i - 1] + 140.0 - u[i - 1] + I
            du = a * (b * v[i - 1] - u[i - 1])
            v[i] = v[i - 1] + dv * dt
            u[i] = u[i - 1] + du * dt

    return t, v, u


patterns = {
    'regular spiking (RS)': dict(a=0.02, b=0.2, c=-65, d=8, I=10),
    'intrinsically bursting (IB)': dict(a=0.02, b=0.2, c=-55, d=4, I=10),
    'chattering (CH)': dict(a=0.02, b=0.2, c=-50, d=2, I=10),
    'fast spiking (FS)': dict(a=0.1, b=0.2, c=-65, d=2, I=10),
    'low-threshold spiking (LTS)': dict(a=0.02, b=0.25, c=-65, d=2, I=10),
    'resonator (RZ)': dict(a=0.1, b=0.26, c=-65, d=2, I=10),
}

n_patterns = len(patterns)
fig, axes = plt.subplots(n_patterns, 1, figsize=(12, 2.5 * n_patterns), sharex=True)

T = 500
dt = 0.25

for idx, (name, params) in enumerate(patterns.items()):
    I_val = params['I']
    I_func = lambda t, I=I_val: I if t > 50 else 0

    t, v, u = simulate_izhikevich(
        params['a'], params['b'], params['c'], params['d'],
        I_func, T=T, dt=dt,
    )

    ax = axes[idx]
    ax.plot(t, v, 'k-', linewidth=0.7)
    ax.set_ylabel('v (mV)')
    ax.set_title(f'{name} (a={params["a"]}, b={params["b"]}, c={params["c"]}, d={params["d"]})')
    ax.set_ylim(-80, 35)

    n_spikes = np.sum(v >= 29.5)
    firing_time = (T - 50) / 1000.0
    if n_spikes > 0 and firing_time > 0:
        rate = n_spikes / firing_time
        ax.text(0.98, 0.95, f'{rate:.0f} Hz', transform=ax.transAxes,
                ha='right', va='top', fontsize=9, color='gray')

axes[-1].set_xlabel('time (ms)')
plt.tight_layout()
plt.savefig('neuroloc/simulations/single_neuron/izhikevich_gallery.png', dpi=150, bbox_inches='tight')
plt.close()

print(f'saved: izhikevich_gallery.png')
for name, params in patterns.items():
    print(f'  {name}: a={params["a"]}, b={params["b"]}, c={params["c"]}, d={params["d"]}')
