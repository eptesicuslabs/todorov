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

N_exc = 800
N_inh = 200
N_total = N_exc + N_inh

duration_total = 1000 * ms

eqs_exc = '''
dv/dt = (-v + V_rest + I_syn_e - I_syn_i + I_drive) / tau_m : volt
dI_syn_e/dt = -I_syn_e / tau_e : volt
dI_syn_i/dt = -I_syn_i / tau_i : volt
I_drive : volt
tau_m : second
V_rest : volt
tau_e : second
tau_i : second
'''

eqs_inh = '''
dv/dt = (-v + V_rest + I_syn_e - I_syn_i + I_drive) / tau_m : volt
dI_syn_e/dt = -I_syn_e / tau_e : volt
dI_syn_i/dt = -I_syn_i / tau_i : volt
I_drive : volt
tau_m : second
V_rest : volt
tau_e : second
tau_i : second
'''

exc_group = NeuronGroup(N_exc, eqs_exc,
                        threshold='v > -50*mV',
                        reset='v = -60*mV',
                        refractory=2*ms,
                        method='euler')
exc_group.v = '-65*mV + 5*mV * rand()'
exc_group.tau_m = 20 * ms
exc_group.V_rest = -65 * mV
exc_group.tau_e = 2 * ms
exc_group.tau_i = 10 * ms
exc_group.I_drive = '18*mV + 4*mV * randn()'

inh_group = NeuronGroup(N_inh, eqs_inh,
                        threshold='v > -50*mV',
                        reset='v = -60*mV',
                        refractory=1*ms,
                        method='euler')
inh_group.v = '-65*mV + 5*mV * rand()'
inh_group.tau_m = 10 * ms
inh_group.V_rest = -65 * mV
inh_group.tau_e = 2 * ms
inh_group.tau_i = 10 * ms
inh_group.I_drive = '5*mV + 2*mV * randn()'

syn_ee = Synapses(exc_group, exc_group, on_pre='I_syn_e_post += 0.3*mV')
syn_ee.connect(p=0.05)

syn_ei = Synapses(exc_group, inh_group, on_pre='I_syn_e_post += 1.5*mV')
syn_ei.connect(p=0.2)

syn_ie = Synapses(inh_group, exc_group, on_pre='I_syn_i_post += 2.5*mV')
syn_ie.connect(p=0.2)

syn_ii = Synapses(inh_group, inh_group, on_pre='I_syn_i_post += 1.5*mV')
syn_ii.connect(p=0.2)

spike_mon_exc = SpikeMonitor(exc_group)
spike_mon_inh = SpikeMonitor(inh_group)
rate_mon_exc = PopulationRateMonitor(exc_group)
rate_mon_inh = PopulationRateMonitor(inh_group)

run(duration_total)

fig, axes = plt.subplots(3, 1, figsize=(14, 14))
fig.suptitle('PING gamma oscillation network\n'
             '800 excitatory + 200 inhibitory neurons, E-I reciprocal coupling',
             fontsize=13, y=0.98)

ax = axes[0]
exc_times = spike_mon_exc.t / ms
exc_ids = spike_mon_exc.i
inh_times = spike_mon_inh.t / ms
inh_ids = spike_mon_inh.i

plot_start = 400
plot_end = 700
exc_mask = (exc_times >= plot_start) & (exc_times < plot_end)
inh_mask = (inh_times >= plot_start) & (inh_times < plot_end)

ax.scatter(exc_times[exc_mask], exc_ids[exc_mask], s=0.3, c='#2166ac', alpha=0.4, rasterized=True)
ax.scatter(inh_times[inh_mask], inh_ids[inh_mask] + N_exc, s=0.3, c='#b2182b', alpha=0.4, rasterized=True)
ax.set_ylabel('neuron index')
ax.set_xlabel('time (ms)')
ax.set_title('spike raster (300 ms window, blue: excitatory, red: inhibitory)')
ax.set_xlim(plot_start, plot_end)

ax = axes[1]
window_ms = 5
rate_times = rate_mon_exc.t / ms
dt_ms = float(defaultclock.dt / ms)
window_bins = max(1, int(window_ms / dt_ms))
kernel = np.ones(window_bins) / window_bins
rate_exc_raw = rate_mon_exc.rate / Hz
rate_inh_raw = rate_mon_inh.rate / Hz
rate_exc_smooth = np.convolve(rate_exc_raw, kernel, mode='same')
rate_inh_smooth = np.convolve(rate_inh_raw, kernel, mode='same')

rate_mask = (rate_times >= plot_start) & (rate_times < plot_end)
ax.plot(rate_times[rate_mask], rate_exc_smooth[rate_mask], color='#2166ac', linewidth=1.2, label='excitatory (LFP proxy)')
ax.plot(rate_times[rate_mask], rate_inh_smooth[rate_mask], color='#b2182b', linewidth=1.2, label='inhibitory')
ax.set_ylabel('population rate (Hz)')
ax.set_xlabel('time (ms)')
ax.set_title('population firing rates (5 ms smoothing) -- gamma oscillation visible')
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim(plot_start, plot_end)

ax = axes[2]
analysis_start = 200
analysis_end = 1000
analysis_mask = (rate_times >= analysis_start) & (rate_times < analysis_end)
lfp_signal = rate_exc_smooth[analysis_mask]

if len(lfp_signal) > 0:
    lfp_detrended = lfp_signal - np.mean(lfp_signal)
    n_fft = len(lfp_detrended)
    sampling_rate = 1.0 / (dt_ms * 1e-3)
    freqs = np.fft.rfftfreq(n_fft, d=1.0/sampling_rate)
    fft_vals = np.fft.rfft(lfp_detrended)
    power_spectrum = np.abs(fft_vals)**2 / n_fft

    freq_mask = (freqs >= 5) & (freqs <= 150)
    ax.plot(freqs[freq_mask], power_spectrum[freq_mask], color='#2166ac', linewidth=1.5)

    gamma_mask = (freqs >= 25) & (freqs <= 100)
    if np.any(gamma_mask):
        gamma_freqs = freqs[gamma_mask]
        gamma_power = power_spectrum[gamma_mask]
        peak_idx = np.argmax(gamma_power)
        peak_freq = gamma_freqs[peak_idx]
        ax.axvline(x=peak_freq, color='#b2182b', linewidth=1.5, linestyle='--',
                   label=f'gamma peak: {peak_freq:.1f} Hz')

    ax.axvspan(30, 100, alpha=0.1, color='orange', label='gamma band (30-100 Hz)')
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('power (a.u.)')
    ax.set_title('power spectrum of excitatory population rate')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(5, 150)

plt.tight_layout(rect=[0, 0, 1, 0.96])
output_path = os.path.join(script_dir, 'gamma_ping.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f'saved: {output_path}')
print(f'excitatory spikes: {spike_mon_exc.num_spikes}')
print(f'inhibitory spikes: {spike_mon_inh.num_spikes}')
print(f'excitatory mean rate: {spike_mon_exc.num_spikes / (N_exc * float(duration_total/second)):.1f} Hz')
print(f'inhibitory mean rate: {spike_mon_inh.num_spikes / (N_inh * float(duration_total/second)):.1f} Hz')
if len(lfp_signal) > 0 and np.any(gamma_mask):
    print(f'gamma peak frequency: {peak_freq:.1f} Hz')

coupling_strengths = [0.5, 1.0, 2.0, 4.0]
peak_freqs = []
peak_powers = []

for w_ie_factor in coupling_strengths:
    start_scope()

    seed(42)
    np.random.seed(42)

    exc_g = NeuronGroup(N_exc, eqs_exc,
                        threshold='v > -50*mV',
                        reset='v = -60*mV',
                        refractory=2*ms,
                        method='euler')
    exc_g.v = '-65*mV + 5*mV * rand()'
    exc_g.tau_m = 20 * ms
    exc_g.V_rest = -65 * mV
    exc_g.tau_e = 2 * ms
    exc_g.tau_i = 10 * ms
    exc_g.I_drive = '18*mV + 4*mV * randn()'

    inh_g = NeuronGroup(N_inh, eqs_inh,
                        threshold='v > -50*mV',
                        reset='v = -60*mV',
                        refractory=1*ms,
                        method='euler')
    inh_g.v = '-65*mV + 5*mV * rand()'
    inh_g.tau_m = 10 * ms
    inh_g.V_rest = -65 * mV
    inh_g.tau_e = 2 * ms
    inh_g.tau_i = 10 * ms
    inh_g.I_drive = '5*mV + 2*mV * randn()'

    s_ee = Synapses(exc_g, exc_g, on_pre='I_syn_e_post += 0.3*mV')
    s_ee.connect(p=0.05)

    s_ei = Synapses(exc_g, inh_g, on_pre='I_syn_e_post += 1.5*mV')
    s_ei.connect(p=0.2)

    w_ie_val = 2.5 * w_ie_factor
    s_ie = Synapses(inh_g, exc_g, on_pre=f'I_syn_i_post += {w_ie_val}*mV')
    s_ie.connect(p=0.2)

    s_ii = Synapses(inh_g, inh_g, on_pre='I_syn_i_post += 1.5*mV')
    s_ii.connect(p=0.2)

    rate_mon = PopulationRateMonitor(exc_g)

    run(duration_total)

    rate_t = rate_mon.t / ms
    rate_r = rate_mon.rate / Hz
    r_smooth = np.convolve(rate_r, kernel, mode='same')

    a_mask = (rate_t >= analysis_start) & (rate_t < analysis_end)
    sig = r_smooth[a_mask]
    sig_d = sig - np.mean(sig)
    n = len(sig_d)
    f = np.fft.rfftfreq(n, d=1.0/sampling_rate)
    p = np.abs(np.fft.rfft(sig_d))**2 / n
    g_mask = (f >= 25) & (f <= 100)
    if np.any(g_mask):
        g_f = f[g_mask]
        g_p = p[g_mask]
        pidx = np.argmax(g_p)
        peak_freqs.append(g_f[pidx])
        peak_powers.append(g_p[pidx])
    else:
        peak_freqs.append(0)
        peak_powers.append(0)

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle('effect of E-I coupling strength on gamma oscillations', fontsize=13, y=1.02)

ax = axes2[0]
ax.plot(coupling_strengths, peak_freqs, 'o-', color='#2166ac', linewidth=2, markersize=8)
ax.set_xlabel('I->E coupling strength (x baseline)')
ax.set_ylabel('gamma peak frequency (Hz)')
ax.set_title('gamma frequency vs inhibitory coupling')

ax = axes2[1]
ax.plot(coupling_strengths, peak_powers, 's-', color='#b2182b', linewidth=2, markersize=8)
ax.set_xlabel('I->E coupling strength (x baseline)')
ax.set_ylabel('gamma peak power (a.u.)')
ax.set_title('gamma power vs inhibitory coupling')

plt.tight_layout()
output_path2 = os.path.join(script_dir, 'gamma_coupling_sweep.png')
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
plt.close()

print(f'saved: {output_path2}')
for i, w in enumerate(coupling_strengths):
    print(f'  w_ie={w:.1f}x: peak freq={peak_freqs[i]:.1f} Hz, peak power={peak_powers[i]:.1f}')
