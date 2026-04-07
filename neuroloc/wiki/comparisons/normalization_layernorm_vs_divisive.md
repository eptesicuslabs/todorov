# LayerNorm vs divisive normalization

## the two normalizations

### LayerNorm (Ba, Kiros, Hinton 2016)

LayerNorm normalizes activations across features within each sample:

    LayerNorm(x)_i = gamma_i * (x_i - mu) / sqrt(sigma^2 + eps) + beta_i

where:
- mu = (1/d) * sum_j x_j is the mean across features
- sigma^2 = (1/d) * sum_j (x_j - mu)^2 is the variance across features
- gamma_i, beta_i are learnable per-feature scale and shift parameters
- eps is a small constant for numerical stability

LayerNorm is applied identically to every sample, independent of other samples in the batch. it centers the activation vector (subtracts the mean) and rescales it (divides by standard deviation). the result is a vector with zero mean and unit variance, then linearly transformed by gamma and beta.

todorov uses RMSNorm, a simplification that omits the mean subtraction:

    RMSNorm(x)_i = gamma_i * x_i / sqrt((1/d) * sum_j x_j^2 + eps)

RMSNorm divides by the root mean square instead of the standard deviation. it is computationally cheaper (no mean subtraction step) and empirically equivalent or superior to LayerNorm in transformer architectures (Zhang and Sennrich, 2019).

both LayerNorm and RMSNorm are applied before each attention and MLP sublayer in todorov (pre-norm architecture).

### divisive normalization (Heeger 1992; Carandini & Heeger 2012)

divisive normalization divides each neuron's response by a weighted sum of a pool of neurons:

    R_i = gamma * D_i^n / (sigma^n + sum_j(w_j * D_j^n))

see [[divisive_normalization]] for full treatment. the key features:
- **per-neuron output:** each neuron i has its own numerator D_i
- **pool-specific denominator:** the weights w_j define which neurons contribute to normalization. different neurons can have different pools.
- **nonlinear:** the exponent n introduces a power-law nonlinearity before normalization
- **activity-dependent:** the denominator is the actual pooled neural activity, not a batch or layer statistic

## dimension-by-dimension comparison

### what is normalized

- **LayerNorm/RMSNorm:** normalizes the entire feature vector (all d dimensions) of a single token. every feature contributes equally to the normalization statistics (mean, variance, or RMS). this is a GLOBAL operation across the feature dimension.
- **divisive normalization:** normalizes each neuron by a SPECIFIC pool of neurons, defined by the weights w_j. the pool can be local (nearby neurons), feature-specific (neurons tuned to similar features), or global (all neurons). the pool structure encodes computational assumptions about which neurons should compete.

### the normalization operation

- **LayerNorm:** subtractive (mean removal) + divisive (variance scaling). the subtraction removes the DC component; the division makes the vector unit-norm in the variance sense.
- **RMSNorm:** divisive only (RMS scaling). no mean removal.
- **divisive normalization:** divisive only (power-weighted pool). no explicit mean removal. the semi-saturation constant sigma plays a role analogous to the eps in LayerNorm (preventing division by zero) but also sets the operating point of the gain control.

### linearity vs nonlinearity

- **LayerNorm/RMSNorm:** linear in the input, modulo the normalization. the variance (or RMS) depends on the input, making the full operation nonlinear, but the nonlinearity is mild -- it is a smooth rescaling.
- **divisive normalization:** nonlinear due to the exponent n. when n > 1, the response is a compressive power function, producing sigmoidal contrast response curves. the nonlinearity is fundamental to the computation, not incidental.

### pool structure

- **LayerNorm/RMSNorm:** the "pool" is the entire feature vector. all features contribute equally to the normalization statistic. there is no spatial, semantic, or feature-based structure in the pool.
- **divisive normalization:** the pool is structured. in V1, the pool typically includes neurons tuned to all orientations at one location (cross-orientation normalization) or neurons at nearby locations (surround normalization). the pool structure determines what is factored out: same-location all-orientation pools produce orientation-invariant contrast encoding; surround pools produce spatial contrast enhancement.

### adaptivity

- **LayerNorm/RMSNorm:** the normalization statistics (mean, variance, RMS) are computed deterministically from the current input. there is no temporal adaptation -- each token is normalized independently. the learnable gamma and beta parameters are fixed after training.
- **divisive normalization:** the denominator is the current neural activity, which reflects stimulus history (through synaptic depression, short-term plasticity), behavioral state (through neuromodulation), and attention (through top-down gain). the normalization adapts on multiple timescales.

## does LayerNorm approximate divisive normalization?

### the argument for yes

consider RMSNorm applied to a feature vector x:

    RMSNorm(x)_i = gamma_i * x_i / sqrt((1/d) * sum_j x_j^2 + eps)

rewrite:

    RMSNorm(x)_i = gamma_i * x_i / (eps' + (1/d) * sum_j x_j^2)^(1/2)

compare to divisive normalization with n=1, uniform weights w_j = 1/d, and sigma = sqrt(eps):

    DN(x)_i = gamma * x_i / (sigma + (1/d) * sum_j x_j)

these have the same structure: numerator is x_i, denominator is a constant plus a pooled sum of the inputs. the differences are:

1. RMSNorm sums squares (x_j^2) and takes the square root; DN sums the inputs directly (or raises to power n). for n=2 and a square root in the denominator, RMSNorm IS divisive normalization with uniform pool weights.
2. RMSNorm has per-feature learnable scale (gamma_i); DN has a single global scale (gamma). the per-feature scale allows RMSNorm to learn feature-specific gain, partially compensating for the lack of pool structure.

so: RMSNorm is a SPECIAL CASE of divisive normalization where:
- the pool includes all features equally (w_j = 1/d)
- the exponent is n=2 in the sum, with a square root in the denominator
- sigma is a fixed small constant (eps)

this special case corresponds to divisive normalization with the weakest possible pool structure: every feature normalizes every other feature equally.

### the argument for no (the dissent)

the claim that RMSNorm approximates divisive normalization is misleading for five reasons:

**1. pool structure is the point.** the computational power of divisive normalization comes from its structured pools: cross-orientation suppression, surround suppression, and attention-modulated pools each encode different computational assumptions. RMSNorm's uniform pool (every feature normalizes every other feature equally) discards this structure entirely. it is like saying "averaging is a special case of weighted averaging" -- technically true but computationally vacuous.

**2. the exponent matters.** divisive normalization with n=2 vs n=1 produces qualitatively different contrast response functions (sigmoidal vs hyperbolic). biological systems use n ~ 1-3.5 depending on the computation. RMSNorm's effective n=2 is not tunable and may not be optimal for any particular biological analog.

**3. no semi-saturation dynamics.** in biological divisive normalization, sigma sets the operating point: it determines the contrast at which the response is half-maximal. sigma adapts across conditions (light adaptation, contrast adaptation) on timescales of seconds to minutes. RMSNorm's eps (typically 1e-6) is a numerical hack, not a computational parameter. it plays no functional role.

**4. no temporal adaptation.** biological divisive normalization adapts to stimulus history through synaptic depression, short-term plasticity, and slow changes in inhibitory tone. RMSNorm is stateless: each token is normalized independently, with no memory of previous inputs. the temporal dynamics of normalization are essential for phenomena like contrast adaptation and light adaptation.

**5. LayerNorm/RMSNorm exists for training stability, not computation.** the original motivation for LayerNorm was to stabilize gradient flow in deep networks by preventing internal covariate shift. this is an optimization concern, not a computational one. divisive normalization exists because it improves the REPRESENTATION, not the OPTIMIZATION. confusing these two purposes obscures the biology.

### the honest verdict

RMSNorm is a degenerate special case of divisive normalization with uniform pool weights, fixed exponent, no semi-saturation dynamics, and no temporal adaptation. the mathematical form is similar. the computational substance is different.

RMSNorm does something normalization-like: it prevents features from dominating by scale alone, and it creates a form of competition (features with large values increase the denominator for all features, suppressing the relative magnitude of small features). but this competition is global and unstructured, unlike the local, pool-specific, adapted competition in biological divisive normalization.

the more interesting question is whether todorov NEEDS structured normalization, or whether RMSNorm + ternary spikes + attention already provide sufficient competition. see [[lateral_inhibition_to_adaptive_threshold]] for this analysis.

## what todorov could gain from true divisive normalization

if divisive normalization were added to todorov, the expected benefits would be:

1. **robustness to input scale variation.** RMSNorm already provides this, but divisive normalization with adaptive sigma would provide dynamic range adaptation that persists across tokens (through the recurrent state).
2. **structured feature competition.** pool-specific normalization (e.g., normalizing attention heads against each other, or normalizing within feature groups) could improve representation quality by removing within-group redundancy.
3. **attention enhancement.** Reynolds and Heeger (2009) showed that attention is well-modeled as a gain field that interacts with normalization. adding explicit normalization to the attention mechanism could improve the selectivity of attention.

the expected costs would be:
1. additional hyperparameters (pool structure, n, sigma)
2. computational overhead (the normalization pool sum is O(d) per neuron per layer)
3. potential interactions with existing RMSNorm and ternary spike quantization

## challenges

- the comparison between LayerNorm/RMSNorm and divisive normalization conflates two levels of description: the mathematical operation (which is similar) and the computational purpose (which is different). mathematical similarity does not imply functional equivalence.
- the claim that RMSNorm is a special case of divisive normalization is technically correct but practically misleading. every linear operation is a special case of some nonlinear operation; this does not mean linear operations approximate nonlinear ones.
- testing whether structured divisive normalization improves todorov requires careful ablation: add divisive normalization while keeping RMSNorm (to maintain training stability), and measure whether the additional normalization provides computational benefit beyond what RMSNorm already gives. this has not been done.
- the literature on divisive normalization in deep learning is small and mostly focused on vision (image classification, image compression). whether divisive normalization benefits language models specifically is an open question with limited empirical evidence.

## key references

- ba, j. l., kiros, j. r. & hinton, g. e. (2016). layer normalization. arXiv:1607.06450.
- zhang, b. & sennrich, r. (2019). root mean square layer normalization. advances in neural information processing systems, 32.
- carandini, m. & heeger, d. j. (2012). normalization as a canonical neural computation. nature reviews neuroscience, 13(1), 51-62.
- ren, m., liao, r., urtasun, r., sinz, f. h. & zemel, r. s. (2017). normalizing the normalizers: comparing and extending network normalization schemes. international conference on learning representations.
- balle, j., laparra, v. & simoncelli, e. p. (2016). density modeling of images using a generalized normalization transformation. international conference on learning representations.
- pinto, t. et al. (2024). unconditional stability of a recurrent neural circuit implementing divisive normalization. NeurIPS 2024.

## see also

- [[divisive_normalization]]
- [[lateral_inhibition]]
- [[lateral_inhibition_to_adaptive_threshold]]
- [[sparse_vs_dense_representations]]
