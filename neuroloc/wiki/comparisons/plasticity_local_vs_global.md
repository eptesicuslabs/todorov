# local vs. global plasticity rules

status: current (as of 2026-04-16).

## summary

synaptic plasticity rules can be classified by their information requirements. a **local** rule uses only information available at the synapse itself: presynaptic activity, postsynaptic activity, and the current synaptic weight. a **global** rule requires information that is not locally available: an error signal computed elsewhere in the network, a reward signal from the environment, or a neuromodulatory broadcast from a distant brain region. this distinction is the central constraint on biologically plausible learning: the brain has no backpropagation bus, so any learning rule it implements must either be local or use a biologically realistic mechanism for delivering global signals.

## classification

### purely local rules (two-factor)

these rules depend only on presynaptic and postsynaptic activity.

| rule | formulation | information used | limitations |
|---|---|---|---|
| [[hebbian_learning]] | Delta_w = eta * x_pre * x_post | pre rate, post rate | unstable, no LTD |
| [[stdp]] | Delta_w = f(t_post - t_pre) | pre spike time, post spike time | no error signal, no reward |
| [[bcm_theory]] | dw/dt = phi(c, theta_M) * d | pre rate, post rate, post history | theta_M requires slow temporal averaging |
| Oja's rule | dw/dt = eta * (x*y - y^2*w) | pre rate, post rate, weight | extracts only first PC |
| anti-Hebbian | Delta_w = -eta * x_pre * x_post | pre rate, post rate | decorrelation only |

locality is their strength and their weakness. they can learn input correlations, develop feature selectivity, and stabilize activity, but they cannot solve credit assignment: they have no way to know whether the features they extract are useful for a downstream task.

### neuromodulated rules (three-factor)

three-factor rules multiply the local Hebbian term by a global neuromodulatory signal:

    Delta_w = M * H(pre, post)

where M is the neuromodulatory "third factor" and H is a Hebbian coincidence detector.

| rule | third factor M | biological substrate | computational analog |
|---|---|---|---|
| reward-modulated STDP | reward prediction error | dopamine from VTA/SNc | REINFORCE / policy gradient |
| surprise-modulated STDP | novelty signal | acetylcholine from basal forebrain | curiosity-driven learning |
| attention-modulated STDP | salience signal | noradrenaline from locus coeruleus | gated learning |
| eligibility trace + reward | delayed reward | dopamine + synaptic tag | temporal difference learning |

three-factor rules are semi-local: the Hebbian component H(pre, post) is computed locally, but the neuromodulator M is a global broadcast signal. the neuromodulator does not carry synapse-specific error information -- it modulates ALL synapses in the target area simultaneously. synapse-specificity comes from the eligibility trace: only synapses that were recently co-activated (and therefore have a nonzero eligibility trace) are modified.

### global rules (error-driven)

these rules require synapse-specific error information that is computed by a global process.

| rule | information required | biological plausibility |
|---|---|---|
| backpropagation | per-synapse error gradient dL/dw | implausible (requires symmetric feedback weights, layer-by-layer backward pass) |
| BPTT | per-synapse error gradient across time | implausible (requires storing full forward trajectory, backward pass through time) |
| feedback alignment | random feedback weights | partially plausible (removes weight symmetry requirement, but still needs distinct backward pass) |
| predictive coding | local prediction errors at each layer | plausible (errors computed locally, but requires specific circuit architecture) |
| target propagation | per-layer targets | partially plausible (requires top-down connections carrying target values) |

backpropagation and BPTT are the gold standards for training artificial neural networks but face the "weight transport problem": computing the gradient at a synapse requires knowing the weights of downstream connections, which is not locally available.

## analysis: todorov's delta rule

the KDA delta rule state update is:

    S_t = diag(alpha) * S_{t-1} + beta_t * k_t * v_t^T

this runs in the FORWARD pass. but its parameters (alpha_log, beta_proj weights, q/k/v projection weights) are trained by backpropagation through time.

**the state update itself is a local operation.** at each timestep, the update to S_t depends on:
- S_{t-1} (the current state, locally available)
- alpha (a learned parameter, locally stored)
- beta_t (computed from the current input, locally available given the projection weights)
- k_t, v_t (computed from the current input, locally available given the projection weights)

no information from future timesteps or from other layers is needed to COMPUTE S_t. in this sense, the forward dynamics are biologically plausible: a neuron implementing the KDA update would only need access to its own state, its own parameters, and the current input.

**the parameter learning is global.** alpha_log, beta_proj, and the q/k/v projections are trained by gradient descent, which requires backpropagation through the full computational graph. this is not biologically plausible.

the critical distinction: the state update rule (S_t = diag(alpha) * S_{t-1} + beta_t * k_t * v_t^T) is local and could, in principle, be implemented by biological hardware. the TRAINING of the parameters that govern this update rule is global and requires backpropagation.

## analysis: BPTT

BPTT unrolls a recurrent network through time and computes gradients by backpropagating errors through the unrolled graph. this requires:

1. storing the full forward state trajectory (all S_t for t = 1, ..., T)
2. computing per-timestep loss gradients dL/dS_t
3. propagating gradients backward through time: dL/dS_{t-1} = diag(alpha) * dL/dS_t

this is maximally non-local in both space and time: the gradient at timestep t depends on the loss at all future timesteps.

no biological neuron has access to this information. the brain does not have a "backward pass."

## verdict

| system | forward dynamics | parameter learning | biological plausibility |
|---|---|---|---|
| pure [[hebbian_learning]] / [[stdp]] | local | local | high |
| three-factor (reward-modulated STDP) | local | semi-local (needs neuromodulator) | moderate |
| [[bcm_theory]] | local (needs slow averaging) | local | moderate-high |
| [[homeostatic_plasticity]] (scaling) | local (needs firing rate sensor) | local | high |
| KDA delta rule (forward) | local | N/A (not a learning rule) | high |
| KDA parameter training (BPTT) | N/A | global (backprop) | low |
| standard transformer (attention) | global (attends to all positions) | global (backprop) | low |

the KDA delta rule occupies an interesting middle ground. its forward dynamics are more biologically plausible than standard attention (which requires global comparison against all positions), but its training still relies on backpropagation. this is exactly the position occupied by most computational neuroscience models: biologically plausible dynamics trained by biologically implausible optimization.

## dissenting argument

one could argue that the distinction between "local forward dynamics" and "global parameter learning" is misleading. what matters for biological plausibility is not whether the forward pass is local, but whether the system can LEARN without backpropagation.

from this perspective, KDA's locality in the forward pass is irrelevant: the parameters alpha, beta_proj, and the projection matrices were all shaped by BPTT. the resulting system may LOOK like a biological plasticity rule, but it was found by an optimization algorithm that no biological system could implement. the analogy to synaptic plasticity is post-hoc: we are projecting biological intuitions onto a system that was designed by gradient descent, not by evolution.

the counter-counter-argument: evolution also performs optimization, just on a much longer timescale. the fact that backpropagation found these parameters does not mean that evolution (or developmental plasticity, or reward-modulated learning) could not find similar ones. the structure of the KDA update rule -- outer-product association with exponential forgetting and gated writing -- may be a convergent solution that both gradient descent and biological optimization discover independently. the test is not how the parameters were found, but whether the resulting dynamics match biological observations. and on that test, the KDA state update is structurally closer to a biological plasticity rule than standard attention is.

## key references

- Gerstner, W., Lehmann, M., Liakoni, V., Corneil, D. & Brea, J. (2018). Eligibility traces and plasticity on behavioral time scales: experimental support of neoHebbian three-factor learning rules. Frontiers in Neural Circuits, 12, 53.
- Lillicrap, T. P., Santoro, A., Marris, L., Akerman, C. J. & Hinton, G. (2020). Backpropagation and the brain. Nature Reviews Neuroscience, 21(6), 335-346.
- Whittington, J. C. R. & Bogacz, R. (2019). Theories of error back-propagation in the brain. Trends in Cognitive Sciences, 23(3), 235-250.
- Fremaux, N. & Gerstner, W. (2016). Neuromodulated spike-timing-dependent plasticity, and theory of three-factor learning rules. Frontiers in Neural Circuits, 9, 85.
- Rao, R. P. N. & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. Nature Neuroscience, 2(1), 79-87.
