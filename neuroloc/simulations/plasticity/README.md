# plasticity simulations

## contents

### stdp_weight_evolution.py

simulates two neurons connected by a synapse with pair-based STDP. the presynaptic neuron is driven by a Poisson spike train, and the postsynaptic neuron receives correlated input at varying correlation strengths (0.0, 0.3, 0.6, 0.9). demonstrates that STDP weight evolution depends on the temporal correlation between pre- and postsynaptic firing: higher correlation (pre consistently fires before post) drives synaptic weight toward the maximum, while uncorrelated firing drives it toward zero (because A_- > A_+, ensuring net depression for random timing).

parameters:
- tau_pre = tau_post = 20 ms
- A_+ = 0.01, A_- = 0.012
- w_max = 1.0, w_init = 0.5
- input rate = 20 Hz
- duration = 10 s

output: stdp_weight_evolution.png (four panels, one per correlation level)

### homeostatic_scaling.py

simulates a small network (80 excitatory + 20 inhibitory neurons) with Hebbian STDP at excitatory-to-excitatory synapses and fixed inhibitory connections. demonstrates the interaction between Hebbian potentiation and inhibitory stabilization. measures whether the relative weight structure (rank order) is preserved as weights evolve under Hebbian learning.

parameters:
- N = 100 (80 exc, 20 inh)
- connection probability = 0.2 (exc-exc), 0.3 (inh-all)
- tau_m = 20 ms, V_th = -50 mV
- target rate = 10 Hz
- duration = 30 s

output: homeostatic_scaling.png (six panels: firing rate, weight distributions, weight correlation, activity dynamics, heterogeneity preservation, summary statistics)

## dependencies

- brian2 >= 2.5
- matplotlib >= 3.5
- numpy >= 1.21
- scipy >= 1.7

## running

    pip install brian2 matplotlib scipy
    python stdp_weight_evolution.py
    python homeostatic_scaling.py
