# bridge: lateral inhibition to adaptive threshold

status: current (as of 2026-04-16).

## the biological mechanism

in cortex, competition between neurons is implemented through three interacting mechanisms:

1. **[[lateral_inhibition]]:** neighboring neurons suppress each other via inhibitory interneurons. the Hartline-Ratliff equations describe the simplest form: r_i = e_i - sum_j K_ij * r_j. the effect is subtractive: the mean activity is removed, leaving only the neurons with above-average drive.

2. **[[divisive_normalization]]:** each neuron's response is divided by the pooled activity of a normalization pool: R_i = D_i^n / (sigma^n + sum_j w_j * D_j^n). this is the canonical cortical computation (Carandini and Heeger, 2012). the effect is divisive: the response is scaled by the population activity, providing contrast gain control, attention modulation, and cross-feature suppression.

3. **[[winner_take_all]]:** lateral inhibition selects the k most active neurons and suppresses the rest, enforcing sparse population codes. PV+ basket cells ([[inhibitory_interneurons]]) implement WTA via fast perisomatic inhibition, potentially structured by gamma oscillations (~30-80 Hz).

these three mechanisms are not independent. the stabilized supralinear network (Rubin et al., 2015) shows that subtractive lateral inhibition in circuits with supralinear neurons produces emergent divisive normalization. and strong divisive normalization approaches WTA in the limit (as the exponent n increases, only the largest input survives normalization).

## the current todorov implementation

### ternary spike threshold (src/spikes/ternary_spike.py)

the AdaptiveTernarySpike module computes:

    threshold = alpha * mean(|x|)
    output_i = sign(x_i) * [|x_i| > threshold]

where alpha is a learnable scalar (initialized 1.0). the output is ternary: {-1, 0, +1}.

this threshold depends on the population mean absolute activation -- the mean of |x| across all d features. neurons fire (+1 or -1) only if their absolute activation exceeds the threshold. approximately 41% of neurons fire at alpha=1.0.

the threshold is a GLOBAL scalar: one value for all neurons in the layer. it is computed from the population activity (mean(|x|)), so it adapts to the scale of the input. but there is no spatial, semantic, or feature-based structure in the thresholding.

### RMSNorm (src/model/decode_head.py)

applied before each attention and MLP sublayer:

    RMSNorm(x)_i = gamma_i * x_i / sqrt((1/d) * sum_j x_j^2 + eps)

this divides each feature by the root mean square of all features. it is a form of global divisive normalization with uniform pool weights and no nonlinearity (see [[normalization_layernorm_vs_divisive]]).

### ATMN per-neuron thresholds (src/spikes/atmn_spike.py)

ATMN has per-neuron learnable thresholds V_th_i = exp(threshold_log_i). each neuron has its own firing threshold, allowing different sparsity levels across features. but there is no inter-neuron competition -- each neuron's threshold is independent of its neighbors' activity.

### what todorov does NOT have

- **no lateral inhibition between neurons.** there is no mechanism by which one neuron's activity suppresses another neuron's activity within the same layer. the ternary spike threshold depends on the MEAN activation, not on the relative ranking of activations.
- **no competitive dynamics between attention heads.** each attention head operates independently. there is no normalization across heads that would implement head-level competition or selection.
- **no structured normalization pools.** RMSNorm normalizes across all features equally. there is no grouping of features into pools that would implement feature-specific gain control.
- **no WTA selection.** the ternary spike is a threshold function, not a competition. all neurons above threshold fire; all neurons below threshold are silent. the identity of the winners is determined by the threshold alone, not by competition.

## is the ternary spike threshold divisive normalization?

this is the central adversarial question. the answer is nuanced.

### the mathematical comparison

divisive normalization:

    R_i = D_i^n / (sigma^n + sum_j w_j * D_j^n)

todorov ternary spike:

    output_i = sign(x_i) * [|x_i| > alpha * mean(|x|)]

rewrite the ternary spike as a two-step process:

1. compute a normalized activation: z_i = |x_i| / mean(|x|)
2. threshold: output_i = sign(x_i) * [z_i > alpha]

step 1 is division by the population mean absolute activation. this is a form of divisive normalization with:
- n = 1 (no power law)
- sigma = 0 (no semi-saturation)
- w_j = 1/d for all j (uniform pool weights)
- the numerator is |x_i|, not x_i (absolute value)

so the ternary spike threshold DOES involve a normalization-like division: each activation is implicitly compared to the population mean. but the output is binary (fire or not), not graded. and the "normalization" is in the threshold computation, not in the output value.

### what it IS

the ternary spike threshold is a **population-relative threshold function**. it asks: is this neuron's activation above or below the population average, scaled by alpha? this is a meaningful form of population-dependent coding: the firing set adapts to the input statistics.

the effect is similar to normalization: in a layer with one large activation and many small ones, the large activation will exceed the threshold (fires) and the small ones will not (silence). in a layer with uniformly large activations, the threshold rises and fewer neurons fire. in a layer with uniformly small activations, the threshold drops and more neurons fire. this is gain control.

### what it is NOT

1. **not per-neuron.** biological divisive normalization computes a different normalization for each neuron, depending on its pool. the ternary spike threshold is one scalar for the entire layer. this means the threshold cannot distinguish between neurons that SHOULD fire (high-information features) and neurons that should not (noise).

2. **not divisive in the output.** divisive normalization produces a graded, rescaled output: R_i is a continuous value between 0 and gamma. the ternary spike produces {-1, 0, +1}. the quantization discards all magnitude information. the "normalization" affects only whether a neuron fires, not its output value.

3. **not structured.** there is no pool structure. RMSNorm's uniform pool at least has learnable per-feature gain (gamma_i). the ternary threshold has a single learnable parameter (alpha) shared across all features. per-neuron adaptation requires ATMN.

4. **not competitive.** in divisive normalization, increasing one neuron's response DECREASES other neurons' responses (through the shared denominator). in the ternary spike, increasing one neuron's activation INCREASES the threshold for all neurons (through the mean), which is competitive in a global sense but without neuron-specific dynamics. the competition is mediated entirely by the mean, losing all information about which specific neurons are competing.

5. **not adaptive across time.** the threshold is computed from the current input only. biological normalization adapts across multiple timescales (milliseconds via synaptic depression, seconds via contrast adaptation, minutes via light adaptation).

### the honest assessment

todorov's ternary spike threshold is a crude, global, population-dependent gating function. it achieves a similar high-level effect to divisive normalization (strong activations survive, weak ones are suppressed) through a different mechanism (thresholding vs division). calling it "divisive normalization" overstates the biological correspondence. calling it "unrelated to normalization" understates the functional similarity.

the more precise statement: the ternary spike threshold implements population-relative sparse selection, which is one of the computational effects that divisive normalization achieves in biology. but it implements this effect without the pool structure, nonlinearity, temporal adaptation, or graded output that make biological divisive normalization computationally powerful.

### does RMSNorm fill the gap?

RMSNorm is applied before each sublayer and provides genuine divisive normalization (with uniform pool weights). the combination of RMSNorm + ternary spikes provides:

- RMSNorm: divisive scaling by population activity (graded, continuous)
- ternary spike: population-relative selection (binary, discrete)

together, these provide normalization (RMSNorm) followed by sparse selection (spikes). this is loosely analogous to divisive normalization (gain control) followed by WTA (competition). but the normalization and selection are sequential, not interleaved as in biological circuits where normalization and competition occur simultaneously through recurrent dynamics.

## the proposed change

### option A: structured normalization pools

replace RMSNorm's uniform pool with structured pools that group features by learned affinity:

    GroupNorm: normalize within groups of features rather than across all features
    PooledNorm: learn normalization pool weights w_j per feature group

this would move RMSNorm from global uniform normalization to pool-specific normalization, closer to biological divisive normalization.

implementation: PyTorch's GroupNorm already exists. the number of groups (G) is a hyperparameter. G=1 is LayerNorm; G=d is InstanceNorm; intermediate G provides structured pools.

### option B: competitive ternary spikes (k-WTA)

replace the threshold-based spike selection with k-WTA: select the top-k activations by absolute value and suppress the rest.

    top_k_indices = topk(|x|, k)
    output_i = sign(x_i) if i in top_k_indices, else 0

this replaces population-relative thresholding with explicit competition. the number of winners k directly controls the firing rate (f = k/d). this is closer to biological WTA via lateral inhibition.

implementation cost: topk is O(d * log k) on GPU. for d=1024 and k=400 (~41%), this is fast. the gradient issue is the same as for the current ternary spike (STE for the selection, zero gradient for the suppressed neurons).

### option C: divisive normalization before ternary spikes

add explicit divisive normalization before the ternary spike:

    z_i = x_i^n / (sigma^n + (1/d) * sum_j x_j^n)
    output_i = sign(z_i) * [|z_i| > alpha * mean(|z|)]

this applies divisive normalization (with learnable n and sigma) BEFORE the ternary quantization. the normalization reshapes the activation distribution, potentially improving the quality of the subsequent selection.

implementation cost: one power operation, one reduction, and one division per neuron. the exponent n and sigma are learnable parameters.

## implementation spec

the recommended first experiment is option B (k-WTA), because it most directly implements the biological mechanism and requires minimal code change:

```python
class KWTATernarySpike(nn.Module):
    def __init__(self, k_fraction=0.41):
        super().__init__()
        self.k_fraction = k_fraction

    def forward(self, x):
        k = max(1, int(self.k_fraction * x.shape[-1]))
        abs_x = torch.abs(x)
        topk_vals, topk_idx = torch.topk(abs_x, k, dim=-1)
        mask = torch.zeros_like(x)
        mask.scatter_(-1, topk_idx, 1.0)
        spikes = torch.sign(x) * mask
        return spikes
```

the STE backward pass is unchanged (gradient passes through the selection as identity).

## expected impact

**k-WTA vs current threshold:**
- the firing rate becomes exactly controllable (f = k/d) instead of approximately controlled by alpha
- the selection is competitive (relative ranking) instead of absolute (above/below threshold)
- the number of active neurons is fixed, which may stabilize training (no dead neuron pathologies)
- but the fixed k removes the ability of the network to modulate sparsity per token (some tokens may need more or fewer active features)

**structured normalization vs RMSNorm:**
- GroupNorm adds pool structure but introduces the hyperparameter G
- the benefit depends on whether features naturally cluster into groups with distinct normalization needs
- for language models, the evidence for structured normalization is weak (GroupNorm is standard in vision, not in NLP)

**divisive normalization before spikes:**
- adds nonlinearity (power law) before quantization
- the power law compresses the dynamic range, potentially improving the quality of ternary quantization
- but adds computational cost and hyperparameters (n, sigma)

## risk assessment

1. **k-WTA breaks differentiability.** the topk operation is not differentiable at boundaries (when the k-th and (k+1)-th largest values are equal). STE handles this for the ternary quantization, but the selection itself introduces a new non-differentiable step. risk: gradient pathologies at boundaries. mitigation: add small noise to break ties.

2. **fixed k may be suboptimal.** the current threshold allows the network to learn different firing rates at different layers (via alpha). k-WTA with fixed k/d removes this flexibility. mitigation: make k_fraction a per-layer learnable parameter.

3. **GroupNorm may not help for language.** feature groups in language models do not have the spatial structure that makes GroupNorm useful in vision. the features at position 37 and position 38 in a d=1024 vector have no natural spatial relationship. mitigation: learn the group assignments, but this adds substantial complexity.

4. **overhead.** topk is fast on GPU but adds a kernel launch per spike layer. for a 300M-parameter model with ~24 spike layers, this is ~48 additional kernel launches per forward pass. negligible for total training time, but non-zero.

## see also

- [[lateral_inhibition]]
- [[divisive_normalization]]
- [[winner_take_all]]
- [[inhibitory_interneurons]]
- [[sparse_coding_to_ternary_spikes]]
- [[normalization_layernorm_vs_divisive]]
- [[population_coding_to_spike_health]]
