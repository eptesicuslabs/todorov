# izhikevich model

**why this matters**: the Izhikevich model demonstrates that 4 learnable parameters per neuron suffice to produce 20+ qualitatively distinct temporal dynamics -- a powerful design lesson for ML architectures seeking to add biologically inspired recurrent behavior without per-neuron parameter explosion.

## status
[DRAFT]
last updated: 2026-04-06
sources: 4 papers, 1 textbook

## biological description

the Izhikevich model (2003) is a two-variable neuron model designed to combine the biological plausibility of the [[hodgkin_huxley]] model with the computational efficiency of the [[leaky_integrate_and_fire]] model. it achieves this by using a quadratic voltage equation (a polynomial approximation to the HH dynamics) coupled with a linear recovery variable.

the model can reproduce over 20 known firing patterns of cortical and thalamic neurons by varying just 4 parameters (a, b, c, d). these include regular spiking, intrinsically bursting, chattering, fast spiking, low-threshold spiking, thalamocortical, and resonator patterns. this makes it uniquely versatile among computationally efficient models.

the key biological insight is that the quadratic term (0.04v^2 + 5v + 140) approximates the interaction between Na+ activation and K+ activation. this interaction drives the **subthreshold dynamics** (voltage behavior below the spike threshold) of cortical neurons. the **recovery variable** u (a slow negative feedback signal combining K+ channel activation and Na+ channel inactivation) plays the role of both channel types, combined into a single slow variable. ML analog: the quadratic self-interaction v^2 is a bilinear operation where the state interacts with itself, equivalent to the B (bilinear) family in the CRBR framework.

for an ML researcher: think of this as a 2D recurrent cell where the state update is a quadratic function of the previous state plus a linear recovery term. the 4 parameters (a, b, c, d) control the qualitative behavior of the cell: whether it fires regularly, bursts, oscillates, or adapts. the model costs ~13 FLOPS per timestep (compared to ~1200 for HH and ~5 for LIF), making it feasible for medium-scale networks.

## mathematical formulation

the model consists of two coupled differential equations:

    dv/dt = 0.04 * v^2 + 5 * v + 140 - u + I
    du/dt = a * (b * v - u)

with reset conditions (when v >= 30 mV):

    v <- c
    u <- u + d

where:
- v = membrane potential (mV). the variable v represents the membrane voltage of the neuron.
- u = membrane recovery variable (dimensionless). combines the effects of K+ channel activation and Na+ channel inactivation. provides negative feedback to v.
- I = input current (arbitrary units, can be scaled)
- a, b, c, d = dimensionless parameters

parameter definitions:

- a: time scale of the recovery variable u (typical range: 0.02-0.1). smaller values result in slower recovery. biologically, this corresponds to the speed of K+ channel deactivation.

- b: sensitivity of u to subthreshold fluctuations of v (typical range: 0.2-0.26). larger values couple v and u more strongly, producing more subthreshold oscillations and low-threshold spiking. when b > 0, u acts as a negative feedback (recovery); the value b = 0.2 gives a standard sensitivity.

- c: after-spike reset value of v (typical range: -65 to -50 mV). controls the depth of the post-spike hyperpolarization. deeper resets (more negative c) produce regular spiking; shallower resets produce bursting because the membrane recovers to threshold faster.

- d: after-spike increment of u (typical range: 0.5-8). controls spike-triggered adaptation. larger values produce stronger adaptation (more slowing of firing rate over time). biologically, this corresponds to the activation of slow K+ currents by the spike.

the coefficients 0.04, 5, 140, and the spike cutoff at 30 mV were chosen by Izhikevich to match the subthreshold dynamics and spike initiation of cortical neurons. they are not free parameters.

parameter values for known neuron types:

regular spiking (RS):          a=0.02, b=0.2,  c=-65, d=8
intrinsically bursting (IB):   a=0.02, b=0.2,  c=-55, d=4
chattering (CH):               a=0.02, b=0.2,  c=-50, d=2
fast spiking (FS):             a=0.1,  b=0.2,  c=-65, d=2
low-threshold spiking (LTS):   a=0.02, b=0.25, c=-65, d=2
thalamocortical (TC):          a=0.02, b=0.25, c=-65, d=0.05
resonator (RZ):                a=0.1,  b=0.26, c=-65, d=2

note: RS, IB, CH are cortical excitatory neurons. FS, LTS are cortical inhibitory interneurons. TC and RZ are thalamic neurons.

computational cost comparison (from Izhikevich 2004):

    model                  FLOPS/timestep    # patterns    biological fidelity
    hodgkin-huxley         ~1200             all           high
    izhikevich             ~13               20+           good
    adaptive exponential   ~20               8+            good
    leaky integrate-fire   ~5                1-2           low
    binary threshold       ~1                1             minimal

the Izhikevich model achieves 10,000 spiking neurons with 1,000,000 synaptic connections at 1 ms resolution on a 1 GHz PC (circa 2003).

## evidence strength

MODERATE-STRONG. the model is widely used and validated:

- Izhikevich (2003) demonstrated reproduction of 20+ firing patterns from cortical recordings.
- Izhikevich (2004, "Which Model to Use for Cortical Spiking Neurons?") provided a systematic comparison with other models on computational cost and biological fidelity.
- the model has been used in large-scale cortical simulations (Izhikevich and Edelman 2008, 10^11 synapses).
- implemented in NEST, Brian2, NEURON, and other major simulators.

however, the model has not been as rigorously benchmarked against intracellular recordings as the [[adaptive_exponential]] (which achieves 96% spike prediction accuracy). the fixed coefficients (0.04, 5, 140) limit the model's ability to fit individual neurons precisely.

## challenges and counter-arguments

1. **the fixed coefficients are a weakness, not a feature.** the values 0.04, 5, 140 in the voltage equation were chosen to roughly match cortical neurons. for any specific neuron type, the fit is approximate. unlike the [[adaptive_exponential]], where all parameters have physical units and can be measured electrophysiologically, the Izhikevich parameters are phenomenological. this means you cannot use electrophysiological data to systematically constrain the model.

2. **the dimensionless formulation obscures physical meaning.** the variable u has no clear physical units. the parameters a, b, c, d are dimensionless. this makes it difficult to relate the model to specific ion channel properties or to transfer parameters between species or brain regions. the AdEx, by contrast, uses conductances (nS), currents (pA), and voltages (mV) throughout.

3. **the quadratic approximation fails for strongly driven neurons.** the quadratic v equation produces dynamics that diverge from HH when the neuron is driven far from its resting state. in particular, the model does not correctly capture the action potential shape (the spike waveform is a mathematical artifact capped at 30 mV, not a physical prediction). this matters less for rate coding but matters for temporal coding schemes.

4. **spike timing accuracy is lower than AdEx.** Jolivet et al. (2008) showed that the AdEx predicts spike timing better than the Izhikevich model for cortical pyramidal cells. for applications where precise spike timing matters (coincidence detection, temporal coding), the Izhikevich model is suboptimal.

5. **the 4-parameter space is not well organized.** unlike the AdEx, where Naud et al. (2008) systematically classified the parameter space into firing pattern regions, the Izhikevich parameter space has been explored only by example. the boundaries between firing pattern regimes are not analytically characterized. this makes systematic exploration difficult.

6. **the model requires euler integration with dt <= 0.5 ms.** the quadratic nonlinearity makes the system numerically stiff for large inputs. izhikevich recommends solving each timestep as two 0.5 ms half-steps for stability. this doubles the effective computational cost.

## simulation

see [[izhikevich_gallery]] (neuroloc/simulations/single_neuron/izhikevich_gallery.py).

the simulation demonstrates:
- regular spiking (RS): most common cortical excitatory pattern
- intrinsically bursting (IB): initial burst followed by regular spiking
- chattering (CH): rhythmic bursting at high frequency
- fast spiking (FS): high-frequency firing without adaptation
- low-threshold spiking (LTS): burst from hyperpolarized state
- thalamocortical (TC): dual-mode firing (rest vs. burst)

key parameters to vary:
- c and d together: control the transition from regular spiking to bursting
- a: controls recovery speed (fast = FS-like, slow = RS-like)
- b: controls subthreshold oscillations (higher = more resonant)

## computational implications

the Izhikevich model demonstrates three principles relevant to artificial architectures:

1. **4 parameters suffice for 20+ behaviors.** this is a powerful result for neural architecture design: you do not need many parameters per neuron to capture diverse temporal dynamics. a 2D recurrent cell with 4 learnable parameters per unit could, in principle, learn the optimal firing pattern for each position in the network.

2. **the quadratic nonlinearity is key.** the 0.04v^2 term creates a positive-feedback loop analogous to the HH Na+ channel activation. this is what gives the model its biological realism beyond the LIF. in CRBR terms, the quadratic self-interaction is a B (bilinear) operation with v interacting with itself. this is qualitatively different from the linear dynamics of the LIF.

3. **the recovery variable is a slow negative feedback.** the variable u with small a (slow recovery) acts as a low-pass filtered version of the output that provides negative feedback. this is functionally identical to spike-frequency adaptation. in a language model, this could help regulate the sparsity of spike activations over sequence positions.

4. **the reset mechanism creates discrete state transitions.** the jump (v->c, u->u+d) at each spike is a discontinuous map. the composition of continuous dynamics (between spikes) and discrete resets (at spikes) creates a **hybrid dynamical system** (a system mixing continuous evolution with discrete jumps). ML analog: this is structurally similar to the delta-rule layers in todorov, which accumulate state continuously and then apply discrete spike quantization via the straight-through estimator.

## bridge to todorov

mapping to ATMN:

| Izhikevich component | ATMN equivalent | match quality |
|---------------------|-----------------|---------------|
| 0.04v^2+5v+140 (quadratic dynamics) | linear: h = x + (1/tau)*u | poor: ATMN is linear |
| u recovery variable | none | missing |
| v -> c (voltage reset) | u = h - spikes*V_th (subtraction reset) | different reset mechanism |
| u -> u+d (recovery increment) | none | missing |
| 4 parameters (a,b,c,d) | 2 parameters (tau, threshold_log) | fewer parameters |

the Izhikevich model suggests two possible improvements to ATMN:
1. add a quadratic self-interaction term to the membrane dynamics (bilinear in CRBR terms)
2. add a recovery variable for spike-frequency adaptation

however, both additions would increase computational cost. the quadratic term requires one additional multiply per neuron per timestep; the recovery variable doubles the state size.

## related mechanisms

- [[leaky_integrate_and_fire]] -- the model that Izhikevich improves upon
- [[hodgkin_huxley]] -- the biophysical model that Izhikevich approximates
- [[adaptive_exponential]] -- alternative 2D model with similar capabilities
- [[neuron_model_comparison]] -- side-by-side comparison

## open questions

1. could the Izhikevich quadratic dynamics be used directly in an artificial spike neuron? v' = 0.04v^2 + 5v + 140 - u + I is cheap to compute and provides a natural nonlinearity without explicit activation functions. but the fixed coefficients would need to be made learnable, and the numerical stability concerns would need to be addressed.

2. the model's ability to switch between firing patterns by changing 4 parameters suggests a "meta-learning" approach: learn the parameters (a, b, c, d) during training so that each neuron finds its optimal firing pattern. has this been tried in spiking neural networks?

3. the thalamocortical (TC) mode shows dual-mode behavior: regular spiking from depolarized rest, rebound bursting from hyperpolarized state. could a similar dual-mode mechanism be useful in language models for distinguishing "attend" (tonic) and "reset" (burst) operations?

4. the recovery variable u provides a natural mechanism for implementing relative refractory periods. the d parameter controls how much harder it is to fire immediately after a spike. could ATMN benefit from this type of graded refractoriness?

## source bibliography

- Izhikevich, E.M. (2003). Simple model of spiking neurons. IEEE Transactions on Neural Networks, 14(6), 1569-1572.
- Izhikevich, E.M. (2004). Which model to use for cortical spiking neurons? IEEE Transactions on Neural Networks, 15(5), 1063-1070.
- Izhikevich, E.M. (2007). Dynamical Systems in Neuroscience: The Geometry of Excitability and Bursting. MIT Press.
- Izhikevich, E.M. and Edelman, G.M. (2008). Large-scale model of mammalian thalamocortical systems. Proceedings of the National Academy of Sciences, 105(9), 3593-3598.
- Gerstner, W., Kistler, W.M., Naud, R., and Paninski, L. (2014). Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition. Cambridge University Press.
