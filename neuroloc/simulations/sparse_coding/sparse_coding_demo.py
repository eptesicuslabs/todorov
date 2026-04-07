import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from brian2 import *

prefs.codegen.target = 'numpy'

seed(42)
np.random.seed(42)

N_input = 50
N_patterns = 8
N_encoding = 200
N_readout = N_input
duration_per_pattern = 200 * ms
input_rate_active = 100 * Hz
input_rate_silent = 5 * Hz

patterns = np.random.binomial(1, 0.3, size=(N_patterns, N_input)).astype(float)

sparsity_levels = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80]
results_reconstruction = []
results_information = []

for target_sparsity in sparsity_levels:
    v_th_enc = -50 * mV + (1.0 - target_sparsity) * 20 * mV

    eqs_input = '''
    dv/dt = (-v + I_ext) / (10*ms) : volt
    I_ext : volt
    '''

    eqs_encoding = '''
    dv/dt = (-v + I_syn) / (20*ms) : volt
    I_syn = w_sum * volt : volt
    w_sum : 1
    '''

    eqs_readout = '''
    dv/dt = (-v + I_syn_rd) / (10*ms) : volt
    I_syn_rd = w_sum_rd * volt : volt
    w_sum_rd : 1
    '''

    input_group = NeuronGroup(N_input, eqs_input,
                              threshold='v > -50*mV',
                              reset='v = -70*mV',
                              refractory=2*ms,
                              method='euler')
    input_group.v = -70 * mV

    encoding_group = NeuronGroup(N_encoding, eqs_encoding,
                                 threshold=f'v > {v_th_enc/mV}*mV',
                                 reset='v = -70*mV',
                                 refractory=3*ms,
                                 method='euler')
    encoding_group.v = -70 * mV

    readout_group = NeuronGroup(N_readout, eqs_readout,
                                threshold='v > -50*mV',
                                reset='v = -70*mV',
                                refractory=2*ms,
                                method='euler')
    readout_group.v = -70 * mV

    W_enc = np.random.randn(N_encoding, N_input) * 0.5
    W_dec = np.random.randn(N_readout, N_encoding) * 0.3

    input_mon = SpikeMonitor(input_group)
    enc_mon = SpikeMonitor(encoding_group)
    readout_mon = SpikeMonitor(readout_group)

    net = Network(input_group, encoding_group, readout_group,
                  input_mon, enc_mon, readout_mon)

    reconstruction_errors = []
    mi_values = []

    for p_idx in range(N_patterns):
        pattern = patterns[p_idx]

        for j in range(N_input):
            if pattern[j] > 0.5:
                input_group.I_ext[j] = -40 * mV
            else:
                input_group.I_ext[j] = -68 * mV

        enc_drive = W_enc @ pattern
        for j in range(N_encoding):
            encoding_group.w_sum[j] = float(enc_drive[j]) * 0.5

        t_start = net.t / ms

        net.run(duration_per_pattern)

        t_end = net.t / ms

        enc_spikes_per_neuron = np.zeros(N_encoding)
        for idx, t in zip(enc_mon.i, enc_mon.t / ms):
            if t_start <= t < t_end:
                enc_spikes_per_neuron[idx] += 1

        enc_rates = enc_spikes_per_neuron / (float(duration_per_pattern / ms) / 1000.0)
        enc_active = (enc_rates > 5.0).astype(float)
        actual_sparsity = np.mean(enc_active)

        dec_output = W_dec @ enc_active
        dec_output_norm = 1.0 / (1.0 + np.exp(-dec_output))

        recon_error = np.mean((pattern - dec_output_norm) ** 2)
        reconstruction_errors.append(recon_error)

        p_enc = np.clip(np.mean(enc_active), 1e-10, 1 - 1e-10)
        h_enc = -p_enc * np.log2(p_enc) - (1 - p_enc) * np.log2(1 - p_enc)

        p_input = np.clip(np.mean(pattern), 1e-10, 1 - 1e-10)
        h_input = -p_input * np.log2(p_input) - (1 - p_input) * np.log2(1 - p_input)

        corr_val = np.abs(np.corrcoef(pattern, dec_output_norm)[0, 1])
        if np.isnan(corr_val):
            corr_val = 0.0

        mi_approx = h_input * corr_val
        mi_values.append(mi_approx)

        input_group.v = -70 * mV
        encoding_group.v = -70 * mV
        readout_group.v = -70 * mV

    mean_recon = np.mean(reconstruction_errors)
    mean_mi = np.mean(mi_values)
    results_reconstruction.append(mean_recon)
    results_information.append(mean_mi)

    print(f"sparsity target={target_sparsity:.2f} | recon_error={mean_recon:.4f} | MI_approx={mean_mi:.4f}")

fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot([s * 100 for s in sparsity_levels], results_reconstruction,
         'o-', color='#2c3e50', linewidth=2, markersize=8)
ax1.set_xlabel('target sparsity level (%)', fontsize=12)
ax1.set_ylabel('mean reconstruction error (MSE)', fontsize=12)
ax1.set_title('sparsity vs reconstruction quality', fontsize=14)
ax1.axvline(x=41, color='#e74c3c', linestyle='--', linewidth=1.5,
            label='todorov firing rate (41%)')
ax1.axvspan(1, 10, alpha=0.15, color='#27ae60', label='cortical range (1-10%)')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 85)
plt.tight_layout()
fig1.savefig('neuroloc/simulations/sparse_coding/sparsity_vs_reconstruction.png', dpi=150)
plt.close(fig1)

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot([s * 100 for s in sparsity_levels], results_information,
         's-', color='#8e44ad', linewidth=2, markersize=8)
ax2.set_xlabel('target sparsity level (%)', fontsize=12)
ax2.set_ylabel('information preservation (approx MI, bits)', fontsize=12)
ax2.set_title('sparsity vs information content', fontsize=14)
ax2.axvline(x=41, color='#e74c3c', linestyle='--', linewidth=1.5,
            label='todorov firing rate (41%)')
ax2.axvspan(1, 10, alpha=0.15, color='#27ae60', label='cortical range (1-10%)')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 85)
plt.tight_layout()
fig2.savefig('neuroloc/simulations/sparse_coding/sparsity_vs_information.png', dpi=150)
plt.close(fig2)

print("saved: sparsity_vs_reconstruction.png, sparsity_vs_information.png")
