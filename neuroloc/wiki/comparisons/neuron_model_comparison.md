# neuron model comparison

status: current (as of 2026-04-16).

## document status

draft promoted to current on 2026-04-16 as part of the wiki refactor.
last content update: 2026-04-06. sources: 8 papers, 2 textbooks. the
core analysis (which neuron model features inform ATMN design) remains
directly applicable — the architecture pivoted to slot memory but the
spike-neuron question below is orthogonal to the memory substrate
choice. future content updates should use the `## correction
(YYYY-MM-DD)` format per `OPERATING_DIRECTIVE.md`.

## overview

this document compares four neuron models -- [[hodgkin_huxley]], [[leaky_integrate_and_fire]], [[adaptive_exponential]], and [[izhikevich_model]] -- on dimensions relevant to building spike neurons for a 300M-parameter language model (todorov).

the comparison is structured around a practical question: which model (or which features of which model) should inform the design of ATMN, todorov's spike neuron?

## dimension 1: biological fidelity

    model              state variables    ion channels    spike shape    adaptation    bursting
    hodgkin-huxley     4 (V, m, h, n)     3 (Na, K, L)   realistic      via channels  yes (multi-mechanism)
    LIF                1 (V)              0 (implicit)    not modeled    no            no
    AdEx               2 (V, w)           0 (implicit)    upstroke only  yes (w)       yes (via reset/w)
    Izhikevich         2 (v, u)           0 (implicit)    not modeled    yes (u)       yes (via c, d)

the HH model is the gold standard for single-neuron fidelity. the LIF is the worst. the AdEx and Izhikevich models are comparable, with the AdEx having a slight edge because its exponential spike initiation more accurately captures the dynamics of Na+ channel activation.

for todorov, biological fidelity per se is irrelevant. what matters is whether biological features correspond to useful computational properties. the relevant question is not "how realistic is the spike shape?" but "does the model capture the input-output relationship that makes biological neural computation effective?"

## dimension 2: computational cost

    model              FLOPS/timestep    state size per neuron    numerical stability
    hodgkin-huxley     ~1200             4 floats                 stiff (requires implicit methods, dt <= 0.01 ms)
    LIF                ~5                1 float                  stable (explicit euler, dt <= 1.0 ms)
    AdEx               ~20               2 floats                 moderate (exponential term needs care, dt <= 0.1 ms)
    Izhikevich         ~13               2 floats                 moderate (quadratic term, dt <= 0.5 ms)

for a 300M-parameter model with ~768 spike neurons per layer and ~24 layers, the per-forward-pass cost of the neuron model itself is:

    HH:    768 * 24 * 1200 = 22.1M FLOPS (per token, per timestep)
    LIF:   768 * 24 * 5    = 92K FLOPS
    AdEx:  768 * 24 * 20   = 369K FLOPS
    Izh:   768 * 24 * 13   = 240K FLOPS

for reference, the attention and MLP computations in a 300M model are ~600M FLOPS per token. even the HH model is only ~4% overhead. the neuron model cost is not the bottleneck.

however, this analysis assumes one timestep per token. if we simulate multiple timesteps per token (e.g., 10 substeps for numerical stability with HH), the cost scales linearly. the LIF with 1 substep is unambiguously cheapest.

the real cost concern is not FLOPS but memory: each additional state variable per neuron adds to the activation memory that must be stored for backpropagation. at 300M scale with 2048 tokens, doubling the neuron state from 1 to 2 floats per neuron adds ~150MB of activation memory across all layers. this is meaningful but manageable.

## dimension 3: number of parameters

    model              parameters per neuron    learnable in ATMN context
    hodgkin-huxley     ~20                      impractical (too many, physically constrained)
    LIF                3-5 (tau, V_th, V_r, t_ref, optional V_rest)    feasible
    AdEx               9 (C, g_L, E_L, V_T, delta_T, a, tau_w, b, V_r)    expensive (9 per neuron)
    Izhikevich         4 (a, b, c, d)           feasible
    ATMN (current)     2 (tau, threshold_log)   current implementation

for 768-dimensional spike layers at 24 layers, the total learnable neuron parameters:

    LIF (3 params):    768 * 24 * 3 = 55K parameters (0.02% of 300M)
    Izh (4 params):    768 * 24 * 4 = 74K parameters (0.02% of 300M)
    AdEx (9 params):   768 * 24 * 9 = 166K parameters (0.06% of 300M)
    ATMN (2 params):   768 * 24 * 2 = 37K parameters (0.01% of 300M)

all are negligible relative to the total model size. parameter count per neuron is not a binding constraint.

## dimension 4: firing patterns captured

    model              patterns    adaptation    bursting    resonance    rebound    bistability
    hodgkin-huxley     all         yes           yes         yes          yes        yes
    LIF                1-2         no            no          no           no         no
    AdEx               8+          yes           yes         yes (a<0)    no         no
    Izhikevich         20+         yes           yes         yes          yes        yes

the Izhikevich model wins on pattern diversity. the AdEx wins on quantitative accuracy for the patterns it does capture. the LIF can only produce tonic spiking (regular firing at a rate determined by input current).

for todorov, the relevant question is: which firing patterns are useful for language modeling?

- tonic spiking: basic rate coding. essential. all models support this.
- adaptation: high-pass temporal filtering. potentially useful for emphasizing novel tokens. AdEx and Izhikevich support this.
- bursting: transmitting high-priority information in short windows. potentially useful for attention-like mechanisms. AdEx and Izhikevich support this.
- resonance: frequency-selective response. unclear utility for language.
- rebound: firing after inhibition release. unclear utility for language.

verdict: adaptation is probably useful. bursting is possibly useful. resonance and rebound are unlikely to help.

## dimension 5: suitability for 300M-param language model

evaluation criteria:
1. computational cost per forward pass
2. memory overhead for backpropagation
3. gradient flow through the neuron model
4. number of learnable parameters
5. numerical stability on GPU
6. potential to improve BPB over current ATMN

    model              cost    memory    gradients    parameters    stability    potential
    hodgkin-huxley     poor    poor      poor         poor          poor         low
    LIF                best    best      good         good          best         moderate
    AdEx               good    good      moderate     moderate      moderate     moderate-high
    Izhikevich         good    good      moderate     good          moderate     moderate
    ATMN (current)     best    best      good (STE)   best          best         baseline

assessment:

**HH is ruled out.** too expensive, too many parameters, too stiff. the full biophysical model provides no computational advantage over simpler approximations at this scale.

**LIF is the natural next step.** adding a leak term to ATMN is the minimum change with the highest expected impact. it prevents unbounded membrane potential accumulation, provides a natural forgetting mechanism, and adds only one parameter (leak rate) and one multiply-add per neuron per timestep. this is what [[neuron_models_to_atmn]] recommends as the first intervention.

**Izhikevich is the best value model.** 4 parameters, ~13 FLOPS, 20+ firing patterns. if experiments show that adaptation or bursting improves BPB, the Izhikevich model provides these at lower cost than the AdEx. the fixed coefficients (0.04, 5, 140) would need to be validated for the discrete-token domain.

**AdEx is the most principled model.** all parameters have physical meaning, the exponential spike initiation is biologically accurate, and the firing pattern classification is well-characterized. but 9 parameters per neuron is expensive, and the exponential term creates numerical difficulties.

## verdict

**recommendation: start with LIF (add leak to ATMN), then test Izhikevich-style adaptation.**

the reasoning:
1. the single most important missing feature in ATMN is the leak term. without it, membrane potentials accumulate without bound within a sequence. the LIF's leak is the simplest fix: one multiply-add, one parameter.

2. if leak alone does not improve BPB, the next feature to test is adaptation. the Izhikevich recovery variable u provides adaptation at the lowest cost (4 parameters, ~13 FLOPS).

3. the AdEx should be tested only if Izhikevich-style adaptation proves useful but numerically unstable, or if the exponential spike initiation shows gradient-flow benefits over the hard ternary threshold.

4. the HH model should never be implemented directly. its insights are already captured by the simpler models.

## dissenting argument: why this verdict might be wrong

the verdict assumes that biological neuron dynamics are useful for language modeling. this assumption could be wrong in several ways:

1. **the leak might hurt, not help.** ATMN's lack of leak means it can maintain membrane state indefinitely (within a batch). this is analogous to an LSTM's constant error carousel -- a feature, not a bug. adding leak creates an exponential forgetting curve that destroys long-range information. the success of transformers (no forgetting) and LSTMs (selective forgetting via gates) suggests that the LIF's unconditional leak is the wrong inductive bias. the right approach might be a gated leak (like LSTM forget gates), not a constant leak.

2. **adaptation might be counterproductive for language.** spike-frequency adaptation is designed for novelty detection in sensory processing: respond strongly to changes, ignore steady state. but language modeling requires maintaining context, not discarding it. a neuron that adapts (fires less over time) would progressively lose information about the beginning of a sequence -- exactly the failure mode that transformers were designed to avoid.

3. **the whole comparison is irrelevant because ATMN is not a temporal integrator.** ATMN processes each token independently (membrane potential is reset to zero each batch during training). the temporal dynamics of all four models (leak, adaptation, recovery, gating) assume continuous-time integration across many timesteps. if ATMN is fundamentally a threshold function applied per-token, then the right comparison is not "which neuron model?" but "which activation function?" -- and the answer might be that the ternary spike is already optimal for gradient-efficient sparse coding.

4. **the computational cost analysis is misleading.** the FLOPS comparison shows that all models are cheap relative to attention. but the real constraint is not FLOPS but parallelism. the LIF, AdEx, and Izhikevich models are all sequential in time (each timestep depends on the previous). on a GPU optimized for parallel computation, any temporal recurrence is expensive. the current ATMN avoids this by resetting each batch. adding true temporal dynamics would create a sequential bottleneck.

5. **biological fidelity may be anti-correlated with ML performance.** the most biologically realistic model (HH) is the worst for ML. the least biologically realistic (LIF/ATMN) performs best. perhaps the useful insight from neuroscience is not the specific dynamics but the sparse ternary coding, which ATMN already captures. pursuing greater biological fidelity might be a dead end.

these are genuine concerns, not strawmen. the strongest counterargument is #1 (gated leak > constant leak) combined with #3 (ATMN is not really temporal). if true, the right intervention is not "add LIF dynamics" but "fix the batch reset so ATMN can maintain state across batches, then add a learnable gate to control information retention."

## related mechanisms

- [[hodgkin_huxley]]
- [[leaky_integrate_and_fire]]
- [[adaptive_exponential]]
- [[izhikevich_model]]
- [[neuron_models_to_atmn]]

## source bibliography

- Izhikevich, E.M. (2004). Which model to use for cortical spiking neurons? IEEE Transactions on Neural Networks, 15(5), 1063-1070.
- Brette, R. and Gerstner, W. (2005). Adaptive exponential integrate-and-fire model as an effective description of neuronal activity. Journal of Neurophysiology, 94(5), 3637-3642.
- Jolivet, R., Kobayashi, R., Rauch, A., Naud, R., Shinomoto, S., and Gerstner, W. (2008). A benchmark test for a quantitative assessment of simple neuron models. Journal of Neuroscience Methods, 169(2), 417-424.
- Naud, R., Marcille, N., Clopath, C., and Gerstner, W. (2008). Firing patterns in the adaptive exponential integrate-and-fire model. Biological Cybernetics, 99(4-5), 335-347.
- Gerstner, W., Kistler, W.M., Naud, R., and Paninski, L. (2014). Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition. Cambridge University Press.
- Burkitt, A.N. (2006). A review of the integrate-and-fire neuron model: I. Homogeneous synaptic input. Biological Cybernetics, 95(1), 1-19.
- Izhikevich, E.M. (2003). Simple model of spiking neurons. IEEE Transactions on Neural Networks, 14(6), 1569-1572.
- Abbott, L.F. (1999). Lapicque's introduction of the integrate-and-fire model neuron (1907). Brain Research Bulletin, 50(5-6), 303-304.
