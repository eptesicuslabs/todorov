# sparse vs dense representations

status: current (as of 2026-04-16).

## the question

biological neural networks use sparse representations: only 1-10% of neurons in a cortical area fire at any given time. transformer-based language models use dense representations: every dimension of a 768- or 4096-dimensional embedding vector is a continuous floating-point value, all participating simultaneously.

todorov's architecture uses ternary spikes: {-1, 0, +1}. at alpha=1.0, approximately 41% of dimensions are active (nonzero). this is neither cortically sparse nor transformer-dense. what are the consequences?

## dimension 1: information capacity per bit

**dense (float32)**: each dimension carries up to 32 bits. a vector of dimension d carries up to 32d bits. a 768-dim embedding: ~24,576 bits.

**binary SDR**: each dimension carries 1 bit. but the information content of the vector is concentrated in the pattern of active bits. for N=2048, f=0.02 (40 active bits), the entropy is log2(C(2048,40)) ~ 247 bits. efficiency: 247/2048 ~ 0.12 bits per dimension.

**ternary spike**: each dimension carries log2(3) ~ 1.58 bits. for d=384 (todorov's 6m scale), the maximum entropy at 41% firing rate with balanced +1/-1 among the active units is approximately:

    H = -0.59*log2(0.59) - 0.205*log2(0.205) - 0.205*log2(0.205) ~ 1.50 bits/dim

total: ~576 bits for 384 dimensions. this is between the SDR (~247 bits for 2048 dims) and dense float32 (~24,576 bits for 768 dims).

**verdict**: dense representations carry far more information per dimension. ternary spikes sacrifice ~95% of per-dimension capacity relative to float32. but this is the wrong metric -- what matters is information per unit energy.

## dimension 2: information per unit energy

**dense (GPU float32)**: a multiply-accumulate (MAC) operation on float32 values costs ~4 pJ on modern hardware. a 768x768 matrix multiply: ~768^2 * 4 pJ ~ 2.4 microjoules.

**ternary spike**: multiplication by {-1, 0, +1} replaces MACs with conditional additions and sign flips. at 41% sparsity, 59% of operations are skipped (multiply by zero). the remaining 41% are additions or negations (~0.1 pJ each). effective cost: ~768^2 * 0.41 * 0.1 pJ ~ 24 nanojoules. this is ~100x cheaper than float32.

**biological sparse code**: a cortical spike costs ~10^4 ATP molecules ~ 5 * 10^-13 J per spike (attwell and laughlin, 2001). at 2% population sparsity and 1000 neurons, ~20 active neurons per timestep. total: ~10 picojoules per representation update. this is ~2400x cheaper than ternary spikes on silicon.

**verdict**: ternary spikes achieve substantial energy savings over dense representations but remain far more expensive than biological sparse codes. the 41% firing rate is a key factor -- reducing to 10% would save an additional 4x.

## dimension 3: noise robustness

**dense**: noise robustness depends entirely on the learned representation geometry. transformers learn representations where semantically similar inputs are nearby in cosine distance, but there is no structural guarantee of noise tolerance. adversarial perturbations can move representations far in semantic space with small changes in the input vector.

**binary SDR**: structurally noise-robust. the overlap metric degrades gracefully with bit flips. a 10% corruption of active bits reduces overlap by 10% but the pattern remains identifiable with high probability. this is a mathematical guarantee, not a learned property.

**ternary spike**: intermediate. the quantization itself provides some noise robustness: small perturbations to the pre-spike activations do not change the ternary output unless they cross the threshold. the threshold acts as a denoising filter. but the robustness is localized at each dimension independently -- there is no population-level structural guarantee like SDRs provide.

**verdict**: SDRs have the strongest structural noise guarantees. ternary spikes inherit some robustness from quantization but lack the combinatorial guarantees of high-dimensional sparse binary codes.

## dimension 4: compositionality

**dense**: compositionality in transformers is learned through the attention mechanism. subword embeddings compose into word meanings, word meanings compose into sentence meanings, all through learned nonlinear operations. the composition rules are implicit in the weights.

**binary SDR**: compositionality is structural. union (OR) of two SDRs produces a representation that is recognizably related to both inputs. intersection (AND) extracts shared structure. these operations are defined by the representation format, not learned.

**ternary spike**: ternary values permit a limited form of structural composition. the sign carries directional information (positive vs negative association), which binary SDRs lack. but the composition rules for ternary vectors are less well-defined than for binary SDRs -- addition of two ternary vectors produces values in {-2, -1, 0, 1, 2}, requiring re-quantization.

**verdict**: dense representations offer the most flexible compositionality (learned via attention and MLPs). SDRs offer the most principled compositionality (set-theoretic operations). ternary spikes are awkwardly between the two.

## dimension 5: gradient flow

**dense**: gradients flow through continuous activations without obstruction. the chain rule applies directly. this enables deep learning via backpropagation.

**ternary spike**: the quantization function has zero gradient almost everywhere (the derivative of a step function is zero or undefined). todorov uses the straight-through estimator (STE): during the backward pass, the gradient passes through the quantization as if it were the identity function. this is mathematically unprincipled but empirically effective. the STE introduces gradient bias -- the gradient the optimizer sees is not the true gradient of the loss with respect to the pre-spike activations.

the 41% firing rate is not arbitrary in this context. if only 5% of dimensions were nonzero, 95% of gradient paths through the spike layer would carry zero information (even with STE, the gradient magnitude correlates with the pre-spike activation magnitude, which is small for near-zero activations). the 41% rate may be the minimum needed to maintain sufficient gradient flow through the STE for stable optimization of a model with ~300M parameters.

**binary SDR**: no gradient flow. SDR systems use hebbian learning rules, not backpropagation. this makes them incompatible with deep learning architectures.

**verdict**: the need for gradient flow through the STE may be the primary reason todorov's firing rate is ~41% rather than the biologically observed ~2-10%. this is an engineering constraint, not a biological one.

## dimension 6: the 41% question

is todorov's 41% firing rate "sparse coding"?

**by the cortical standard**: no. cortical population sparseness is typically 1-10%. V1 lifetime sparseness is ~0.88. hippocampal place cells have ~1-2% population sparseness. 41% is dense by any cortical measure.

**by the information-theoretic standard**: it depends on the definition. the treves-rolls sparseness measure for ternary values with 41% active units gives:

    S = 1 - (E[|x|])^2 / E[x^2] = 1 - (0.41)^2 / (0.41) = 1 - 0.41 = 0.59

this is "moderately sparse" by the treves-rolls metric (0 = dense, 1 = maximally sparse).

**by the machine learning standard**: 41% sparsity (59% of values are zero) is sparse. most activation functions in transformers (ReLU, GELU) produce 30-50% zero activations. ternary quantization on top of this is adding a different kind of sparsity -- not just amplitude sparsity but value-set restriction.

**the honest assessment**: todorov's ternary spikes are inspired by biological sparse coding but operate in a fundamentally different regime. the biological claim is not that the firing rate matches cortex. the claim is that the mechanism -- adaptive thresholding producing a sparse, quantized code -- is borrowed from biology, even if the operating point (41% vs 2%) is dictated by the engineering constraints of gradient-based optimization.

## the dissenting argument

the biological analogy may be misleading rather than useful. consider:

1. **biological sparse codes are shaped by metabolic constraints**. neurons fire sparsely because spikes are expensive. in a GPU, all operations cost roughly the same -- there is no metabolic gradient favoring silence. the pressure that produces biological sparsity is absent.

2. **biological sparse codes operate over temporal sequences of spikes**. a cortical neuron's "activity" is a spike train, not a single scalar value. temporal sparsity (long silent periods punctuated by brief bursts) is as important as population sparsity. todorov's ternary spikes are instantaneous -- each input produces one ternary vector with no temporal dynamics (in the basic mode; ATMN spikes add membrane potential, but this is reset each batch during training).

3. **the STE gradient has no biological analog**. biological learning rules (hebbian, anti-hebbian, STDP) operate on local information. the STE passes a global error signal backward through a quantization boundary, which no known biological mechanism does. calling this "biologically inspired" while using a non-biological learning rule is selectively borrowing the parts of biology that suit the engineering and ignoring the parts that don't.

4. **41% active dimensions is closer to a lossy compression scheme than to a sparse code**. ternary quantization with 41% nonzero outputs is better understood as 1.58-bit quantization (information compression) than as sparse coding (representational sparsity). the biological sparse coding literature assumes activity levels where the active set is informative precisely because it is small. at 41%, the inactive set (59%) is barely smaller than the active set -- neither set is "special."

5. **the alpha=1.0 threshold is not divisive normalization**. divisive normalization (carandini and heeger, 2012) divides each neuron's response by the summed activity of a pool of neurons, implementing gain control. todorov's threshold is alpha * mean(|x|) -- a single scalar threshold applied uniformly. this is closer to batch normalization than to divisive normalization. true divisive normalization would compute a different threshold for each neuron based on its local population context.

these criticisms do not invalidate the architecture. they sharpen the claim: todorov's ternary spikes are a quantization mechanism inspired by biological thresholding, operating at a firing rate dictated by gradient flow requirements rather than metabolic constraints. the biological analogy is a source of design ideas, not a source of theoretical guarantees.

## key references

- ahmad, s. & hawkins, j. (2016). how do neurons operate on sparse distributed representations? arXiv:1601.00720.
- attwell, d. & laughlin, s. b. (2001). an energy budget for signaling in the grey matter of the brain. journal of cerebral blood flow and metabolism, 21(10), 1133-1145.
- carandini, m. & heeger, d. j. (2012). normalization as a canonical neural computation. nature reviews neuroscience, 13(1), 51-62.
- bengio, y., leonard, n. & courville, a. (2013). estimating or propagating gradients through stochastic neurons for conditional computation. arXiv:1308.3432.
- guo, y. et al. (2024). ternary spike: learning ternary spikes for spiking neural networks. AAAI 2024.

## see also

- [[sparse_coding]]
- [[sparse_distributed_representations]]
- [[efficient_coding]]
- [[population_coding]]
- [[sparse_coding_to_ternary_spikes]]
