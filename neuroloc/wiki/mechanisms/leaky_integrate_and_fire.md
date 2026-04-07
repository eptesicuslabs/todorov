# leaky integrate-and-fire model

**why this matters**: the LIF is the default neuron model in spiking neural networks and the direct biological ancestor of every spike-based activation function used in neuromorphic ML, including the ATMN neuron in todorov. understanding its leak-threshold-reset loop clarifies which biological features are essential for sparse recurrent computation and which are safely omitted.

## status
[DRAFT]
last updated: 2026-04-06
sources: 7 papers, 2 textbooks

## biological description

the leaky integrate-and-fire (LIF) model is the simplest neuron model that captures the essential behavior of real neurons: subthreshold integration of inputs, a threshold for spike initiation, and a post-spike reset. it was introduced by Louis Lapicque in 1907 and formalized as an **RC circuit** (resistor-capacitor circuit modeling passive membrane electrical properties) analogy.

the model treats the neuron membrane as a resistor (ion channels) in parallel with a capacitor (**lipid bilayer**: the thin fatty membrane surrounding every cell). in the absence of input, the **membrane potential** (voltage difference across the cell membrane) decays exponentially back to a resting value -- this is the "leak." when the membrane potential reaches a threshold, a spike is emitted (instantaneously -- the spike shape is not modeled) and the potential is reset to a value below threshold. ML analog: the leak is mathematically identical to the exponential decay in a gated recurrent unit -- it is a forgetting mechanism that discounts old inputs.

physically, the leak corresponds to the **passive ion channels** (channels that are always open, primarily K+ and Cl-) that drive the membrane toward its resting potential. the capacitance is the charge-storing capacity of the lipid bilayer (~1 uF/cm^2 for all biological membranes). the threshold represents the voltage at which the positive-feedback Na+ channel activation of the [[hodgkin_huxley]] model becomes self-sustaining.

for an ML researcher: the LIF is the neuron equivalent of a leaky ReLU with a threshold. it integrates inputs over time (like an RNN hidden state), has a nonlinear output (spike or no spike), and forgets old inputs exponentially (the leak). ML analog: the threshold-and-reset creates a sparse binary output, equivalent to a hard activation function with built-in sparsity enforcement. it is the default model in computational neuroscience because it is analytically tractable, cheap to simulate, and captures the input-output relationship of real neurons with reasonable accuracy.

## mathematical formulation

the membrane equation:

    C_m * dV/dt = -g_L * (V - V_rest) + I(t)

equivalently, dividing by g_L and defining tau_m = C_m / g_L and R = 1 / g_L:

    tau_m * dV/dt = -(V - V_rest) + R * I(t)

where:
- V = membrane potential (mV)
- C_m = membrane capacitance (nF, typical: 0.2-1.0 nF for cortical neurons)
- g_L = leak conductance (nS, typical: 10-50 nS)
- V_rest = resting potential (mV, typical: -65 to -70 mV)
- tau_m = membrane time constant (ms, typical: 10-30 ms for cortical neurons)
- R = membrane resistance (MOhm, typical: 20-100 MOhm)
- I(t) = input current (nA)

threshold and reset:

    when V(t) >= V_th: emit spike, set V -> V_reset, enforce refractory period t_ref

where:
- V_th = spike threshold (mV, typical: -50 to -55 mV)
- V_reset = reset potential (mV, typical: -65 to -70 mV, often = V_rest)
- t_ref = absolute refractory period (ms, typical: 1-5 ms)

during the refractory period, the membrane potential is clamped at V_reset and no spikes can be emitted.

the **f-I curve** (firing rate as function of constant input current, the neuron's transfer function):

for constant input I > I_rheo (**rheobase current**: the minimum sustained current that produces firing), the steady-state firing rate is:

    f(I) = 1 / (t_ref + tau_m * ln((R*I + V_rest - V_reset) / (R*I + V_rest - V_th)))

where I_rheo = g_L * (V_th - V_rest) is the minimum current for sustained firing.

key properties of the f-I curve:
- discontinuous onset: firing rate jumps from 0 to 1/(t_ref + tau_m * ln((V_th - V_reset)/(V_th - V_th))) at I = I_rheo (type I excitability)
- saturates at f_max = 1/t_ref for large currents
- the shape is entirely determined by tau_m, t_ref, V_th, V_reset

the analytical solution for constant current (between spikes):

    V(t) = V_rest + R*I * (1 - exp(-t/tau_m)) + (V_0 - V_rest) * exp(-t/tau_m)

where V_0 is the initial voltage (V_reset after a spike).

## evidence strength

STRONG. the LIF model is the most widely used neuron model in computational neuroscience. its predictions have been validated against:

- intracellular recordings from cortical neurons (Rauch et al. 2003)
- population firing statistics in cortical networks (Brunel 2000)
- mean-field theory of large networks (Amit and Brunel 1997)
- spike timing precision under fluctuating inputs (Mainen and Sejnowski 1995)

the LIF accounts for ~70-80% of the variance in spike timing for cortical neurons driven by fluctuating currents (Jolivet et al. 2004, Gerstner and Naud 2009). it fails primarily for bursting neurons and for predicting precise subthreshold dynamics.

## challenges and counter-arguments

1. **no spike-frequency adaptation.** real cortical neurons reduce their firing rate over sustained stimulation (adaptation). the LIF fires at a constant rate for constant input. this is a major omission for any model that processes sequences: adaptation acts as a high-pass filter, emphasizing changes over steady state. the [[adaptive_exponential]] model fixes this.

2. **fixed threshold is biologically wrong.** in real neurons, the threshold voltage depends on recent firing history (threshold fatigue), rate of depolarization (accommodation), and neuromodulatory state. the LIF's fixed threshold cannot capture these effects. evidence: Azouz and Gray (2000) showed that cortical neuron thresholds vary by 5-10 mV depending on input dynamics.

3. **the leak timescale may be wrong for artificial architectures.** biological tau_m is 10-30 ms, set by physical membrane properties. in an artificial system processing discrete tokens, the "natural" timescale is 1 token. mapping biological leak constants to token-level processing is not principled. too much leak destroys memory; too little leak eliminates the noise-rejection benefit.

4. **the f-I curve shape is wrong.** the LIF produces a type I f-I curve (square-root onset). many cortical neurons show type II excitability (discontinuous jump to finite frequency), which requires resonant subthreshold dynamics absent in the LIF. type II neurons are better modeled by the [[izhikevich_model]] with appropriate parameters.

5. **the refractory period is ad hoc.** the LIF imposes an absolute refractory period as a hard constraint, but real refractory periods arise from Na+ channel inactivation (absolute) and K+ channel deactivation (relative). the ad hoc implementation misses the relative refractory period, during which the neuron can fire but requires stronger input.

6. **one-dimensional dynamics cannot produce bursting.** the LIF is a 1D dynamical system (plus reset). bursting requires at least 2D dynamics (membrane potential + slow variable). since bursting is common in cortical neurons and may carry distinct information, the LIF systematically misrepresents a significant fraction of neural activity.

## simulation

see [[lif_fi_curve]] (neuroloc/simulations/single_neuron/lif_fi_curve.py).

the simulation demonstrates:
- membrane potential trajectory under constant current injection
- the f-I curve showing firing rate vs. input current
- the effect of the refractory period on maximum firing rate

key parameters to vary:
- tau_m: controls integration timescale (10-30 ms range)
- V_th - V_reset: controls the "gain" of the f-I curve
- t_ref: controls maximum firing rate
- I_ext: controls operating point on the f-I curve

## computational implications

the LIF establishes the minimum viable neuron model for artificial architectures:

1. **the leak is a forgetting mechanism.** the term -g_L*(V - V_rest) drives the membrane back to rest exponentially. this is mathematically equivalent to an exponential moving average: it smooths noisy inputs and prevents unbounded accumulation. in an RNN, this is the decay term in a gated recurrent unit.

2. **the threshold creates sparse binary/ternary output.** only inputs strong enough to cross threshold produce output. this enforces a sparse code -- most neurons are silent most of the time. in todorov, the ternary spike {-1, 0, +1} achieves the same sparsity.

3. **the reset creates a nonlinear feedback loop.** after spiking, the reset drives V below threshold, creating a "refractory" period proportional to V_th - V_reset. this prevents runaway activation and creates temporal structure in the spike train.

4. **the f-I curve is a natural activation function.** the mapping from input current to firing rate is a smooth, saturating nonlinearity -- biologically grounded unlike ReLU, sigmoid, or other ad hoc choices.

the minimum set of LIF features needed for a computationally useful spike neuron: leak (exponential decay), threshold (sparse output), reset (refractory behavior). ATMN currently has threshold and reset but not leak.

## bridge to todorov

the LIF model is the closest biological reference for ATMN. the specific mapping:

| LIF component | ATMN equivalent | match quality |
|--------------|-----------------|---------------|
| C_m dV/dt = -g_L(V-V_rest) + I | h = x + (1/tau)*u | partial: no leak term |
| V_th (fixed) | exp(threshold_log) (learnable) | different: ATMN is per-neuron, learnable |
| V -> V_reset | u = h - spikes*V_th | similar: reset by subtraction |
| t_ref (refractory) | none | missing |
| tau_m = C_m/g_L | tau = 2.0 | different: tau in ATMN is a mixing coefficient, not a leak time constant |

critical gaps identified in [[neuron_models_to_atmn]]:
- ATMN has no leak. the membrane potential accumulates without decay.
- ATMN resets to zero each batch during training, destroying temporal state.
- ATMN has no refractory period.

## related mechanisms

- [[hodgkin_huxley]] -- the biophysical model that LIF approximates
- [[adaptive_exponential]] -- LIF + exponential spike initiation + adaptation
- [[izhikevich_model]] -- alternative 2D simplification of HH
- [[neuron_model_comparison]] -- side-by-side comparison of all models

## open questions

1. what is the optimal leak time constant for a spike neuron in a language model? biological values (10-30 ms, ~10-30 tokens at typical rates) are probably wrong. the right value depends on the timescale of relevant features in the input sequence.

2. does the lack of leak in ATMN actually hurt performance? the membrane potential reset-to-zero at each batch boundary might implicitly compensate (by preventing runaway accumulation). but within a batch, there is no decay. experiments needed.

3. the LIF's f-I curve is a smooth nonlinearity. ATMN's input-output function is a hard ternary quantization. which is better for gradient flow in a deep network? the straight-through estimator in ATMN may lose gradient information that a smooth f-I curve would preserve.

4. can the LIF's analytically tractable dynamics (exponential solution, closed-form f-I curve) be exploited for more efficient training? e.g., computing the spike times analytically rather than simulating step-by-step.

## source bibliography

- Lapicque, L. (1907). Recherches quantitatives sur l'excitation electrique des nerfs traitee comme une polarisation. Journal de Physiologie et de Pathologie Generale, 9, 620-635.
- Abbott, L.F. (1999). Lapicque's introduction of the integrate-and-fire model neuron (1907). Brain Research Bulletin, 50(5-6), 303-304.
- Gerstner, W., Kistler, W.M., Naud, R., and Paninski, L. (2014). Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition. Cambridge University Press. Chapter 1.
- Brunel, N. (2000). Dynamics of sparsely connected networks of excitatory and inhibitory spiking neurons. Journal of Computational Neuroscience, 8(3), 183-208.
- Jolivet, R., Lewis, T.J., and Bhatt, D.K. (2004). Period-to-period variability of neural activity in the visual cortex. Journal of Neurophysiology, 92, 959-976.
- Rauch, A., La Camera, G., Luscher, H.R., Senn, W., and Fusi, S. (2003). Neocortical pyramidal cells respond as integrate-and-fire neurons to in vivo-like input currents. Journal of Neurophysiology, 90(3), 1598-1612.
- Azouz, R. and Gray, C.M. (2000). Dynamic spike threshold reveals a mechanism for synaptic coincidence detection in cortical neurons in vivo. Proceedings of the National Academy of Sciences, 97(14), 8110-8115.
- Burkitt, A.N. (2006). A review of the integrate-and-fire neuron model: I. Homogeneous synaptic input. Biological Cybernetics, 95(1), 1-19.
