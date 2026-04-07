# spike-timing-dependent plasticity (STDP)

**why this matters**: STDP provides a biologically grounded mechanism for learning temporal sequences and causal structure from spike timing. its asymmetric learning window directly informs the design of temporal credit assignment in recurrent architectures, and the trace-based implementation maps naturally to the exponentially decaying state in delta-rule recurrence.

## summary

STDP is a temporally asymmetric form of [[hebbian_learning]] in which the sign and magnitude of synaptic modification depend on the precise relative timing of pre- and postsynaptic spikes. when the presynaptic spike precedes the postsynaptic spike (pre-before-post), the synapse is strengthened (**LTP**: long-term potentiation, a lasting increase in synaptic strength). when the order is reversed (post-before-pre), the synapse is weakened (**LTD**: long-term depression, a lasting decrease in synaptic strength). this asymmetric learning window was systematically characterized by Bi and Poo (1998) in cultured hippocampal neurons. it provides a biophysically grounded mechanism for temporal sequence learning and causal inference at the synaptic level.

## mechanism

### the asymmetric learning window

the classical pair-based STDP rule defines the weight change as a function of the spike timing difference Delta_t = t_post - t_pre:

for pre-before-post (Delta_t > 0, LTP):

    Delta_w = A_+ * exp(-Delta_t / tau_+)

for post-before-pre (Delta_t < 0, LTD):

    Delta_w = -A_- * exp(Delta_t / tau_-)

where:
- A_+ > 0: maximum amplitude of potentiation (typically 0.005 to 0.01 for normalized weights)
- A_- > 0: maximum amplitude of depression (typically 1.05 * A_+ to ensure net depression for uncorrelated inputs)
- tau_+ ~ 16-20 ms: time constant of the LTP window
- tau_- ~ 20-34 ms: time constant of the LTD window
- Delta_t = t_post - t_pre: timing difference between postsynaptic and presynaptic spikes

the asymmetry A_- > A_+ is functionally important: it ensures that random, uncorrelated firing produces net synaptic depression, preventing the unbounded weight growth that plagues pure [[hebbian_learning]].

### implementation via spike traces

in practice, STDP is implemented using exponentially decaying **spike traces** (running variables that track recent spiking history, equivalent to eligibility traces in RL). each neuron maintains a trace that increases by 1 at each spike and decays exponentially between spikes. ML analog: the spike trace is mathematically identical to the exponentially decaying state in a linear recurrence h_t = alpha * h_{t-1} + x_t, which is the core mechanism of the KDA delta rule's state accumulation.

    dx_pre/dt = -x_pre / tau_+    (jump by +1 at each presynaptic spike)
    dx_post/dt = -x_post / tau_-   (jump by +1 at each postsynaptic spike)

the weight update occurs at each spike:

    at a postsynaptic spike: Delta_w = A_+ * x_pre   (LTP)
    at a presynaptic spike: Delta_w = -A_- * x_post   (LTD)

this trace-based formulation is equivalent to the exponential window for isolated spike pairs and naturally generalizes to spike trains.

### molecular basis

the timing dependence arises from the biophysics of NMDA receptors and calcium dynamics:

1. NMDA receptors require both presynaptic glutamate release AND postsynaptic depolarization to open (voltage-dependent magnesium block removal)
2. pre-before-post: the postsynaptic spike depolarizes the membrane while glutamate is still bound, producing large calcium influx through NMDA channels. high [Ca2+] activates CaMKII, triggering AMPA receptor insertion (LTP)
3. post-before-pre: the postsynaptic spike occurs before glutamate arrives, producing only modest calcium elevation. moderate [Ca2+] preferentially activates calcineurin (protein phosphatase 2B), triggering AMPA receptor internalization (LTD)

the calcium hypothesis (Shouval et al. 2002) unifies this: there exist two thresholds theta_d < theta_p such that:
- [Ca2+] < theta_d: no change
- theta_d < [Ca2+] < theta_p: LTD
- [Ca2+] > theta_p: LTP

## the Bi and Poo experiment (1998)

Bi and Poo recorded from pairs of connected neurons in dissociated hippocampal cultures and systematically varied the timing between pre- and postsynaptic action potentials. key findings:

- the critical window is approximately +/- 20 ms
- LTP magnitude decays exponentially with positive Delta_t (pre-before-post)
- LTD magnitude decays exponentially with negative Delta_t (post-before-pre)
- LTP occurred only at synapses with relatively low initial strength (weight dependence)
- LTD did not show obvious dependence on initial synaptic strength
- 60 repetitions at 1 Hz were sufficient to induce lasting changes

this data established the canonical asymmetric STDP window and demonstrated that synaptic modification encodes the temporal order of neural activity, not just its co-occurrence.

## variants

### symmetric STDP

some synapses show symmetric learning windows where both pre-before-post and post-before-pre produce LTP (or both produce LTD). symmetric LTP windows have been observed at inhibitory synapses in the cerebellum and at some excitatory synapses in the striatum. the functional role of symmetric windows is debated, but they may implement coincidence detection without directional selectivity.

### anti-Hebbian STDP

in some interneurons and at certain cortical synapses, the learning window is inverted: pre-before-post produces LTD and post-before-pre produces LTP. this may serve to decorrelate neural activity and prevent runaway excitation.

### triplet STDP

the pair-based model fails to capture frequency dependence observed experimentally: LTP at 50 Hz is much larger than predicted by summing pair-based contributions. the triplet model (Pfister and Gerstner, 2006) adds interactions between spike triplets:

    at a postsynaptic spike: Delta_w = x_pre * (A_2+ + A_3+ * x_post_slow)
    at a presynaptic spike: Delta_w = -x_post * (A_2- + A_3- * x_pre_slow)

where x_post_slow and x_pre_slow are slow spike traces with time constants tau_y ~ 100-120 ms and tau_x ~ 100-120 ms. the triplet terms capture the nonlinear dependence of plasticity on recent postsynaptic activity and recover the BCM-like frequency dependence.

### voltage-dependent STDP

the Clopath et al. (2010) model replaces spike timing with postsynaptic voltage:

    dw/dt = A_LTD * x_pre * f_-(u_-) + A_LTP * x_pre * f_+(u) * f_-(u_-)

where u is the postsynaptic membrane potential and u_- is a low-pass filtered version. this formulation naturally accounts for dendritic voltage attenuation and explains why STDP learning windows differ between proximal and distal synapses.

## weight dependence

### additive (hard bounds) model

A_+ and A_- are constants independent of the current weight, with hard bounds [0, w_max]:

    if w + Delta_w > w_max: w = w_max
    if w + Delta_w < 0: w = 0

this produces a bimodal weight distribution (synapses cluster near 0 or w_max), which is not observed experimentally.

### multiplicative (soft bounds) model

A_+ and A_- depend on the current weight:

    A_+(w) = gamma * (w_max - w)^mu
    A_-(w) = gamma * w^mu

with mu = 1 for the fully multiplicative case. this produces a unimodal weight distribution consistent with experimental observations (van Rossum et al. 2000, Turrigiano et al. 1998). the multiplicative model also naturally implements [[homeostatic_plasticity]] because strong synapses are harder to potentiate and easier to depress.

## computational role

### temporal sequence learning

STDP strengthens synapses from neurons that fire before a postsynaptic spike and weakens those that fire after. in a chain A -> B -> C, repeated sequential activation strengthens A->B and B->C while weakening reverse connections. this creates directional flow and can learn temporal sequences.

### receptive field development

STDP with lateral inhibition can explain the development of orientation selectivity in visual cortex (Song and Abbott, 2001). neurons exposed to natural visual input develop localized, oriented receptive fields through STDP-driven competition.

### causal inference

the asymmetric window implements a primitive form of causal detection: if A consistently fires before B, the A->B synapse strengthens, reflecting the statistical regularity that A may cause B. this is the temporal analog of Hebb's original intuition about causality.

## relationship to todorov

the KDA delta rule uses k_t * v_t^T, which is a Hebbian outer product computed from the SAME timestep, not from different timesteps. there is no explicit timing dependence in the association. however, the recurrence S_t = diag(alpha) * S_{t-1} + beta_t * k_t * v_t^T creates implicit temporal structure: the state S_t carries information about past key-value associations, exponentially decayed. the readout q_t^T * S_t compares the current query against historical associations, which introduces a form of temporal sensitivity through the decay structure rather than through the learning rule itself. see [[plasticity_to_kda_delta_rule]] for the full analysis of whether this constitutes a form of STDP.

## challenges

1. **pair-based STDP is insufficient.** the simple exponential learning window fails to predict frequency-dependent plasticity. LTP at 50 Hz is much larger than the sum of pair-based contributions would predict. the triplet model (Pfister and Gerstner 2006) fixes this but adds parameters. more complex phenomena (burst-timing-dependent plasticity, dendritic compartmentalization effects) require still richer models.

2. **the learning window varies across synapse types.** the canonical asymmetric window (pre-before-post = LTP, post-before-pre = LTD) is not universal. some synapses show symmetric windows, inverted windows (anti-Hebbian), or timing-independent plasticity. the simple exponential model cannot capture this diversity without per-synapse parameterization.

3. **neuromodulatory gating is poorly understood.** dopamine, acetylcholine, and noradrenaline all modulate STDP, but the interactions are complex and context-dependent. the calcium-based models (Shouval et al. 2002, Graupner and Brunel 2012) offer a more mechanistic account but introduce many parameters that are difficult to constrain experimentally.

4. **interaction with homeostatic mechanisms is unresolved.** how STDP interacts with [[homeostatic_plasticity]] and [[bcm_theory]] to produce stable learning in recurrent networks over long timescales remains the most pressing open question. fast STDP modifications could be erased by slow homeostatic rescaling, or the two could synergize -- the answer likely depends on their relative timescales and molecular substrates.

## key references

- Bi, G.-q. & Poo, M.-m. (1998). Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type. Journal of Neuroscience, 18(24), 10464-10472.
- Markram, H., Lubke, J., Frotscher, M. & Sakmann, B. (1997). Regulation of synaptic efficacy by coincidence of postsynaptic APs and EPSPs. Science, 275(5297), 213-215.
- Gerstner, W., Kistler, W. M., Naud, R. & Paninski, L. (2014). Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition. Cambridge University Press. Ch. 19.
- Pfister, J.-P. & Gerstner, W. (2006). Triplets of spikes in a model of spike timing-dependent plasticity. Journal of Neuroscience, 26(38), 9673-9682.
- Clopath, C., Busing, L., Vasilaki, E. & Gerstner, W. (2010). Connectivity reflects coding: a model of voltage-based STDP with homeostasis. Nature Neuroscience, 13(3), 344-352.
- Shouval, H. Z., Bear, M. F. & Cooper, L. N. (2002). A unified model of NMDA receptor-dependent bidirectional synaptic plasticity. PNAS, 99(16), 10831-10836.
- Song, S. & Abbott, L. F. (2001). Cortical development and remapping through spike timing-dependent plasticity. Neuron, 32(2), 339-350.
- Abbott, L. F. & Nelson, S. B. (2000). Synaptic plasticity: taming the beast. Nature Neuroscience, 3, 1178-1183.
