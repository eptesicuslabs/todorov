# bridge: biological normalization to RMSNorm

## the biological mechanism

four biological normalization mechanisms operate across different timescales and spatial extents in cortex:

### 1. divisive normalization (Carandini & Heeger 2012)

the "canonical neural computation." each neuron's response is divided by the pooled activity of a normalization pool:

    R_i = gamma * D_i^n / (sigma^n + sum_j(w_j * D_j^n))

the pool weights w_j define which neurons contribute to the denominator. in V1, the pool typically includes neurons tuned to all orientations at one spatial location (cross-orientation suppression) or neurons at nearby locations (surround suppression). the exponent n (typically 1.0-3.5) controls the nonlinearity. the semi-saturation constant sigma sets the operating range and adapts to stimulus history.

key property: the denominator depends on NEIGHBORING neurons' activity. each neuron is normalized by a structured pool of other neurons, not by its own activity alone. the pool structure encodes computational assumptions about which neurons should compete. see [[divisive_normalization]] for full treatment.

### 2. homeostatic synaptic scaling (Turrigiano et al. 1998)

multiplicative scaling of ALL excitatory synapses on a single neuron:

    w_i -> s * w_i    for all synapses i on neuron j

the scaling factor s is driven by the difference between the neuron's actual firing rate and its target firing rate: ds/dt = (1/tau_h) * (r_target - r_actual). operates on hours-to-days timescale (tau_h ~ 12-48 hours). the multiplicative nature preserves relative weight structure: if synapse A is twice as strong as synapse B before scaling, this ratio is maintained after scaling. see [[homeostatic_plasticity]].

key property: this is per-neuron multiplicative rescaling that preserves relative synaptic strengths. it is a stabilizing (negative feedback) mechanism, not a competitive one.

### 3. neuromodulatory gain control

neuromodulators -- acetylcholine (ACh), norepinephrine (NE), dopamine (DA) -- modulate overall neural gain multiplicatively. ACh enhances thalamocortical transmission and increases the signal-to-noise ratio in sensory cortex. NE modulates the gain of cortical neurons globally, shifting between explorative (low gain, broad tuning) and exploitative (high gain, sharp tuning) modes. operates on seconds-to-minutes timescale. see [[norepinephrine_system]], [[neuromodulation_to_learning_and_gating]].

key property: gain is modulated by behavioral state (arousal, attention, surprise), not by local neural activity. this is a top-down, context-dependent multiplicative rescaling.

### 4. population-level normalization

in V1 and MT, individual neural responses are normalized by the aggregate activity of the local population, not just a small pool. Busse et al. (2009) showed that V1 responses to plaids are consistent with normalization pools that span all orientations and spatial frequencies at a given location. this is broader than the original cross-orientation model and approaches global normalization within a cortical column.

key property: the normalization pool extends across the full local population, approaching the "uniform pool" assumption of RMSNorm but restricted to a spatial neighborhood.

## the current todorov implementation

### RMSNorm (src/model/decode_head.py)

```
class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(Tensor(d_model).fill_(1.0))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = (x.float().pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight
```

the operation:

    RMSNorm(x)_i = gamma_i * x_i / sqrt((1/d) * sum_j x_j^2 + eps)

where gamma_i (self.weight) is a learnable per-dimension scale parameter initialized to 1.0.

### placement and count

RMSNorm is applied in a pre-norm architecture:
- before every attention sublayer (self.n1 in Block)
- before every MLP sublayer (self.n2 in Block)
- before the output projection (self.norm in LM)

at 267m scale: d_model=1024, 24 layers, each with 2 RMSNorm applications = 48 within-block applications + 1 final = 49 total RMSNorm applications per forward pass.

the pre-norm pattern: the residual stream x is normalized BEFORE being read by each sublayer, but the sublayer output is added back to the UNNORMALIZED residual stream. this means the residual stream accumulates unnormalized contributions from all layers, and each layer sees a normalized view of the accumulated state.

### interaction with ternary spikes

within KDA layers, ternary spike quantization occurs AFTER the RMSNorm-normalized input has been projected to queries and keys. the spike threshold is alpha * mean(|x|), computed on the projected values, not on the raw residual stream. so the normalization chain is: residual stream -> RMSNorm -> linear projection -> ternary spike threshold. RMSNorm controls the scale of the input to the projection, and the ternary spike operates on the projected output. see [[sparse_coding_to_ternary_spikes]], [[lateral_inhibition_to_adaptive_threshold]].

## adversarial analysis

### is RMSNorm a form of divisive normalization?

the mathematical case for yes:

RMSNorm divides each feature by a quantity derived from pooled activity across all features. rewrite the divisive normalization equation with n=2, uniform pool weights w_j = 1/d, sigma = sqrt(eps), and a square root over the denominator:

    DN(x)_i = gamma * x_i / sqrt(sigma^2 + (1/d) * sum_j x_j^2)

this is identical to RMSNorm(x)_i = gamma_i * x_i / sqrt(eps + (1/d) * sum_j x_j^2). the mathematical forms are the same. RMSNorm IS divisive normalization with a specific parameter setting.

the mathematical case for no:

mathematical identity does not imply functional equivalence. five structural differences undermine the analogy:

**1. denominator source (severity: HIGH).** biological divisive normalization divides by the activity of NEIGHBORING neurons -- other units in a structured pool. RMSNorm divides by the activity of the SAME vector's dimensions. in biological terms, RMSNorm is a neuron normalizing itself by its own activation pattern, not by its neighbors' activity. this is the difference between social competition (my response depends on what others are doing) and self-regulation (my response depends on what i am doing). divisive normalization is competitive. RMSNorm is not.

**2. pool structure (severity: HIGH).** the computational power of divisive normalization comes from structured pools: cross-orientation pools in V1 remove orientation-dependent gain; surround pools enhance spatial contrast; attention-modulated pools implement selective enhancement. RMSNorm's pool is uniform: every feature normalizes every other feature equally. this is the weakest possible pool structure. it cannot implement feature-selective competition because it treats all features identically in the denominator.

**3. per-token independence (severity: MODERATE).** RMSNorm normalizes each token independently. there is no temporal context -- the normalization of token t is unaffected by token t-1 or t+1. biological normalization has spatial extent (neighboring neurons) and temporal extent (adaptation timescales from milliseconds to minutes). RMSNorm has neither spatial nor temporal extent. it is an instantaneous, pointwise operation.

**4. the exponent is fixed (severity: LOW-MODERATE).** RMSNorm's effective exponent is n=2 (sum of squares, square root). biological systems use n ranging from 1.0 to 3.5 depending on the brain region and computation. the exponent controls the shape of the contrast response function: n=1 gives hyperbolic (Michaelis-Menten), n=2 gives sigmoidal (Naka-Rushton), larger n approaches WTA. RMSNorm cannot adjust its nonlinearity.

**5. sigma is vestigial (severity: LOW).** in biological normalization, sigma (the semi-saturation constant) sets the operating range and adapts to stimulus statistics. in RMSNorm, eps = 1e-6 is a numerical stability constant with no computational role. RMSNorm cannot adapt its operating range.

### is learnable gamma analogous to homeostatic scaling?

the case for yes: both are per-neuron (per-feature) multiplicative rescaling. gamma_i scales the normalized output of feature i, just as synaptic scaling factor s scales all synapses on neuron j. both preserve relative structure within their scope. gamma_i is learned during training (via backpropagation), and synaptic scaling is learned during development (via calcium-dependent feedback). both serve to maintain activations in a useful dynamic range.

the case for no: the mechanisms differ in every important respect. synaptic scaling is activity-dependent negative feedback -- it monitors the neuron's actual firing rate and adjusts to maintain a target. gamma_i is a fixed parameter learned by gradient descent -- it does not monitor the feature's activation statistics at inference time. synaptic scaling operates on hours-to-days timescale; gamma is fixed after training. synaptic scaling responds to perturbations (silence the network, and scaling factors increase); gamma does not respond to distribution shift. calling gamma "homeostatic" confuses a learned parameter with a dynamic control system.

gamma_i is more precisely analogous to neuromodulatory gain control: a per-feature multiplicative factor that adjusts the output scale. but neuromodulatory gain is dynamically modulated by behavioral state, while gamma is static.

### does pre-norm vs post-norm matter for the biological analogy?

biological divisive normalization operates on neural OUTPUTS: the firing rate R_i is the normalized response. the normalization is part of the output computation, not a preprocessing step.

todorov uses pre-norm: RMSNorm is applied to the INPUT of each sublayer, not to its output. the sublayer's output is added to the residual stream without normalization. this means the residual stream accumulates unnormalized contributions, and normalization occurs only when the stream is read by the next layer.

functionally, this is equivalent to normalizing each layer's VIEW of the accumulated state, not the state itself. the biological analog would be a neuron that normalizes its input sensitivity (gain control on the dendrite) rather than its output firing rate. dendritic gain control does exist (see [[dendritic_computation_to_swiglu]]), but it is distinct from divisive normalization, which operates on the output.

the honest assessment: pre-norm RMSNorm is input gain control, not output normalization. the biological analog is closer to dendritic gain adjustment than to divisive normalization of firing rates. this does not make it worse -- it makes it a different computation mapped to a different biological mechanism.

## the proposed change

### option A: no change (status quo)

RMSNorm with learnable gamma is a well-validated component in transformer architectures. it provides training stability (preventing gradient explosion/vanishing from scale drift), implicit gain control (each sublayer sees unit-RMS inputs), and learnable per-feature rescaling. the biological correspondence is limited but the computational value is established. changing RMSNorm to something more biologically faithful risks destabilizing training for uncertain representational benefit.

recommendation: RMSNorm stays. the question is whether anything should be ADDED.

### option B: adaptive sigma (semi-saturation dynamics)

replace the fixed eps with a learnable per-layer semi-saturation constant:

    RMSNorm_adaptive(x)_i = gamma_i * x_i / sqrt(sigma^2 + (1/d) * sum_j x_j^2)

where sigma is a learnable scalar per RMSNorm instance, initialized to a value like 0.1 (not 1e-6). this gives RMSNorm an adjustable operating range: at low input magnitudes (RMS << sigma), the normalization approaches a linear scaling (x / sigma); at high input magnitudes (RMS >> sigma), it approaches standard RMSNorm.

this adds one learnable parameter per RMSNorm instance (49 parameters total at 267m scale -- negligible). the biological motivation is that sigma in divisive normalization adapts to stimulus statistics, providing dynamic range adaptation. the learnable sigma allows each layer to set its own operating point.

risk: low. sigma = 1e-6 is already a special case. making it learnable can only add flexibility.

### option C: group-structured normalization pools

replace uniform-pool RMSNorm with grouped normalization:

    GroupRMSNorm(x)_i = gamma_i * x_i / sqrt(eps + (1/|G_i|) * sum_{j in G_i} x_j^2)

where G_i is the normalization group for feature i. this introduces pool structure: features within the same group normalize each other, but features in different groups are normalized independently. this is closer to biological divisive normalization with structured pools.

implementation: partition d_model features into G groups of size d_model/G. normalize within each group independently. G is a hyperparameter. G=1 recovers standard RMSNorm; G=d_model normalizes each feature by itself only (instance-level).

risk: moderate. group normalization is standard in vision (ResNet, U-Net) but rare in language models. the choice of G is arbitrary without biological guidance on what features should form normalization pools. incorrect G could hurt representation quality.

### option D: exponential moving average normalization statistics

maintain a running estimate of the RMS across tokens, blending current-token statistics with historical statistics:

    rms_t = (1 - alpha) * rms_{t-1} + alpha * sqrt((1/d) * sum_j x_t_j^2)
    output_i = gamma_i * x_i / (rms_t + eps)

this adds temporal extent to the normalization, analogous to the temporal adaptation in biological divisive normalization. the blending factor alpha controls the timescale: alpha=1 recovers standard per-token RMSNorm; alpha near 0 gives slow adaptation across many tokens.

risk: high. this creates a dependency between tokens that complicates parallel training (current-token normalization depends on all previous tokens). it would require sequential processing or an approximation. it also breaks the stationarity assumption that underlies standard training procedures. the recurrent layers (KDA, Mamba3) already provide temporal context in their state; adding it to normalization may be redundant.

## expected impact / risk assessment

### option A (no change)
- expected BPB impact: none (baseline)
- risk: none
- biological fidelity gain: none
- recommendation: maintain as default

### option B (learnable sigma)
- expected BPB impact: negligible to slightly positive. the network gains one degree of freedom per normalization point. unlikely to be measurable at 267m scale.
- risk: very low. worst case: sigma collapses to eps and recovers standard RMSNorm
- biological fidelity gain: small. adds one property (adjustable operating range) out of the five missing properties
- recommendation: safe to include but low priority. testable as a minor ablation

### option C (group-structured pools)
- expected BPB impact: uncertain. positive in vision, unproven in language
- risk: moderate. wrong G hurts performance. adds a hyperparameter that requires tuning
- biological fidelity gain: moderate. pool structure is the most important missing property
- recommendation: defer to phase 6+. requires systematic ablation over G values

### option D (EMA statistics)
- expected BPB impact: uncertain. could help or hurt
- risk: high. breaks parallel training. may conflict with recurrent state
- biological fidelity gain: high. temporal adaptation is a fundamental property of biological normalization
- recommendation: defer indefinitely. the recurrent layers already provide temporal context. adding temporal normalization may be redundant and is architecturally costly

## challenges and counter-arguments

**1. RMSNorm exists for optimization, not computation.** the original motivation for layer normalization (Ba et al. 2016) was to stabilize training by preventing internal covariate shift. RMSNorm simplifies this further by dropping mean centering. the biological normalization mechanisms exist for completely different reasons: efficient dynamic range usage, invariant representations, and competitive selection. mapping an optimization technique to a computational mechanism confuses the level of explanation. RMSNorm may produce normalization-like effects as a side effect of its optimization role, but this does not mean it IMPLEMENTS biological normalization. the convergence of mathematical form is accidental, not principled.

**2. the "uniform pool" objection is fatal to strong analogy claims.** the entire explanatory power of divisive normalization comes from pool structure. cross-orientation pools explain orientation-invariant contrast coding. surround pools explain contextual modulation. attention-modulated pools explain selective enhancement. RMSNorm has none of this structure. claiming RMSNorm is biological normalization because it divides by a pooled quantity is like claiming a thermostat is a nervous system because both use negative feedback. the mathematical operation (division by aggregate activity) is the least interesting part of divisive normalization. the interesting part -- what is in the pool and why -- is entirely absent from RMSNorm.

**3. the analogy obscures what todorov actually does well.** todorov's genuine normalization story is the combination of RMSNorm (scale control) + ternary spikes (population-relative sparse selection) + adaptive threshold (gain adaptation). this three-stage pipeline achieves gain control, sparsification, and population-relative coding -- three of the five computational effects of divisive normalization -- through sequential composition rather than a single divisive operation. forcing the analogy onto RMSNorm alone misses the distributed nature of todorov's normalization, which is spread across multiple mechanisms rather than concentrated in one equation. see [[lateral_inhibition_to_adaptive_threshold]] for the full analysis of how the pipeline maps to biological competition.

**4. per-feature gamma overstates the homeostatic analogy.** gamma_i is a static parameter fixed after training. homeostatic synaptic scaling is a dynamic feedback controller that monitors activity and adjusts in real time. a thermostat analogy: gamma_i is a fixed temperature setting written on a piece of paper. synaptic scaling is a thermostat that measures temperature and adjusts the furnace. both result in a specific temperature, but only the thermostat adapts to perturbation. calling gamma "homeostatic" because it produces a target scale is misleading.

**5. pre-norm placement weakens the output-normalization analogy.** biological divisive normalization normalizes a neuron's OUTPUT firing rate. todorov's RMSNorm normalizes a sublayer's INPUT. if we take the layer-as-neuron analogy seriously, RMSNorm is adjusting the sensitivity of the dendrite, not the response of the axon. dendritic gain control exists biologically but is a different mechanism from divisive normalization, with different computational implications (input gain controls what drives the neuron; output normalization controls what the neuron communicates to others).

## see also

- [[divisive_normalization]]
- [[homeostatic_plasticity]]
- [[norepinephrine_system]]
- [[neuromodulation_to_learning_and_gating]]
- [[lateral_inhibition_to_adaptive_threshold]]
- [[sparse_coding_to_ternary_spikes]]
- [[normalization_layernorm_vs_divisive]]
- [[dendritic_computation_to_swiglu]]
- [[population_coding_to_spike_health]]
- [[carandini_heeger]]
- [[turrigiano]]
