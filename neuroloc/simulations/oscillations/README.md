# oscillations simulations

## gamma_ping.py

PING (pyramidal-interneuron network gamma) simulation demonstrating gamma oscillation emergence from E-I reciprocal coupling.

### network

- 800 excitatory LIF neurons (tau_m = 20 ms, V_thresh = -50 mV, V_reset = -60 mV, refractory = 2 ms)
- 200 inhibitory LIF neurons (tau_m = 10 ms, V_thresh = -50 mV, V_reset = -60 mV, refractory = 1 ms)
- E->E: p=0.05, w=0.3 mV (weak recurrent excitation)
- E->I: p=0.2, w=1.5 mV (strong feedforward excitation to interneurons)
- I->E: p=0.2, w=2.5 mV (strong feedback inhibition, perisomatic)
- I->I: p=0.2, w=1.5 mV (mutual interneuron inhibition)
- synaptic time constants: AMPA tau_e = 2 ms, GABAA tau_i = 10 ms

### protocol

1. 1000 ms continuous simulation with tonic excitatory drive (18 +/- 4 mV to excitatory, 5 +/- 2 mV to inhibitory)
2. coupling strength sweep: I->E weight multiplied by [0.5, 1.0, 2.0, 4.0] to demonstrate frequency/power dependence on inhibitory strength

### output

- gamma_ping.png: three-panel figure (spike raster, population rates showing gamma oscillation, power spectrum with gamma peak)
- gamma_coupling_sweep.png: two-panel figure (gamma peak frequency and power vs I->E coupling strength)

### dependencies

- brian2
- numpy
- matplotlib

### run

    python gamma_ping.py

### expected results

- clear gamma-band oscillation (~30-50 Hz) visible in population firing rate
- spectral peak in the gamma band (30-100 Hz) in the power spectrum
- increasing I->E coupling strength should increase gamma frequency (shorter inhibitory recovery = faster cycle) and power (stronger synchronization)

### relevance to todorov

the PING mechanism demonstrates how gamma oscillations emerge from E-I interaction -- a network property, not a parameter. this is the key difference from Mamba3's data-dependent rotation, which applies a parametric oscillation to the state without any network dynamics generating or modulating the frequency. see [[oscillations_vs_recurrence]] and [[oscillations_to_mamba3_rotation]] for analysis.
