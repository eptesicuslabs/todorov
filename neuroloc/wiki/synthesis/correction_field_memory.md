# correction-field memory

## the core idea

the matrix memory in its current form stores raw key-value outer products. every token writes its full representation into the state, regardless of whether that representation contains new information or could have been predicted from context. the state fills up, older content decays, and the per-head capacity ceiling at d_head=64 means only ~8 patterns cross the 0.5 cosine threshold under the harshest tested retention setting (simulation round b, 2026-04-12, decay=0.40, dense-key/top-k-value encoding); the 0.14*d theoretical formula gives ~9 for the symmetric no-decay case.

the correction-field reframing changes WHAT gets stored. instead of raw representations, the state accumulates the difference between what the model's own prediction function expects and what actually arrives. the matrix memory becomes a correction field: for any query direction q, the state returns a correction to the model's prediction. the model reconstructs the full representation by combining its prediction with the stored correction.

## empirical status (2026-04-14)

the cpu simulation `simulations/memory/correction_field_capacity.py` ran the full sweep (17,568 trials across d_head {32,64,128}, pattern_counts {4,8,16,32,64}, prediction_qualities {0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99}, decays {0.5, 0.8, 0.9, 0.95, 1.0}, 32 trials per cell) using a synthetic prediction of the form `pred = q * v + sqrt(1 - q^2) * noise`. headline findings at d_head=64, decay=0.9:

- **memory_capacity_gain = 0 across every prediction quality.** storing residuals in the state instead of raw values does NOT increase the matrix memory's own effective pattern count. raw_effective_capacity and correction_field_memory_effective_capacity are both 32.
- **reconstruction_effective_capacity = 64 at prediction_quality >= 0.5**, vs raw_effective_capacity = 32. the combined system (prediction + memory correction) doubles effective capacity, but the gain comes from the prediction, not the memory.
- **memory_threshold_shift_by_decay = 0.** the hypothesized decay-tolerance shift is falsified under this test: the memory still needs decay=0.9 for 32 patterns regardless of whether values are residuals or raw.
- **surprise gating works as designed.** at predictable_fraction=0.05 and tau=0.99, write_fraction_mean drops to 0.016 (1.6% of tokens write). the filter rate scales controllably with tau.

these results partially falsify the article's original claims. the design's value is not a memory-side capacity increase and not a decay-tolerance shift. its value is the predict+reconstruct mechanism: a small prediction head provides most of the content, and the matrix memory only needs to carry what the prediction failed to anticipate. the architecture is a learned-decoder-plus-residual-cache, not a self-compressing memory.

an important caveat: the synthetic prediction `q * v + sqrt(1-q^2) * noise` has no structure beyond the variance ratio. a real trained prediction head would have correlated structure (predicts some key directions better than others, exhibits systematic biases) that the memory could complement non-trivially. the trained-prediction sim (pending) is the next test; if it shows the same flat memory_capacity_gain, the design ships only as a prediction-head-with-residual-cache. if it shows non-trivial memory gain, the self-compression thesis survives.

## what the design actually provides (post-simulation)

1. **predictive filtering** (mechanism 1): confirmed in simulation. surprise-gated writes reduce write frequency in proportion to prediction quality. a well-predicted sequence writes fewer entries, leaving the state less crowded.
2. **generative replacement** (mechanism 2): confirmed. the prediction head supplies most of the content at reconstruction time. the memory provides only the correction residual.
3. **capacity via prediction, not via memory** (revised claim): the reconstruction-level effective capacity doubles over the raw memory when prediction quality >= 0.5, but this gain is from the prediction head, not from a property of the matrix memory itself. the memory's intrinsic capacity ceiling stays at 32 patterns under decay=0.9.

the remaining three mechanisms in the six-mechanism list (content-addressable indexing, manifold abstraction, reconsolidation) are orthogonal and composable with this one. the correction-field design does not prevent later tiers from compressing corrections further via codebook indexing or schema-delta encoding. it IS a valid first tier of the compound compression stack — but for different reasons than the pre-simulation article claimed.

## mathematical formulation

all notation follows the project glossary in `PROJECT_PLAN.md`. per layer, per head:

### write path

```
h_t        = layer input at timestep t (d-dimensional, from residual stream)
p_t        = f(h_{t-1})                prediction of h_t from previous hidden state
r_t        = h_t - p_t                 prediction residual
s_t        = ||r_t||^2 / (||h_t||^2 + eps)    surprise ratio (0 = perfectly predicted, 1 = fully novel)

k_t        = W_k @ h_t                key projection (from RAW hidden, dense, for addressability)
v_t        = W_v @ r_t                value projection (from RESIDUAL, not raw hidden)
b_t        = sigma(W_b @ h_t) * g(s_t)   write gate (learned component * surprise modulation)

S_t        = decay * S_{t-1} + b_t * k_t @ v_t^T
```

where f is a lightweight prediction function (one linear map W_pred of shape d x d, or a low-rank factorization W_down @ W_up of shape d x r x d for parameter efficiency), sigma is the sigmoid function, and g is a monotone increasing function of surprise that controls how strongly surprising tokens write. the simplest choice for g is a clamped linear ramp: `g(s) = min(1, s / tau)` where tau is a learned or fixed threshold.

### read path

```
q_t        = W_q @ h_t                query projection (from raw hidden)
correction = q_t^T @ S_t              retrieved correction (in d_head space)
projected  = W_o @ correction         output projection back to d_model space (same W_o as current design)
output_t   = p_t + projected          reconstruction = prediction + projected correction
```

p_t is computed once per timestep and reused for both the write residual (r_t = h_t - p_t) and the output reconstruction. in the autoregressive setting this is valid because h_{t-1} is always available when processing token t. the output projection W_o maps from (num_heads * d_head) back to d_model, the same projection the current architecture already uses. the addition with p_t happens in d_model space after that projection. the result is added back to the residual stream in the same location as the current matrix memory's output. no architectural change to the surrounding model is needed beyond adding W_pred.

### key properties

1. **keys are dense.** keys are projected from the raw hidden state h_t, not from the residual r_t. this preserves the address space. the project's hard rule ("keys must stay dense") is satisfied.

2. **value norms are smaller when prediction is good.** when prediction explains 90% of the hidden-state variance, ||r_t|| is ~32% of ||h_t|| (since variance is squared norm, ||r||^2 = 0.1 * ||h||^2 implies ||r|| = sqrt(0.1) * ||h|| ~ 0.316 ||h||). the information per value vector drops proportionally. note: the synthetic-prediction sim (2026-04-14) showed this does NOT translate to an increased per-head pattern count in the memory itself — memory_capacity_gain was zero. the smaller-norm property still helps by enabling a prediction head to dominate reconstruction without the memory needing to carry the full raw content.

3. **write frequency adapts to content.** the surprise modulation g(s_t) means predictable tokens produce near-zero writes. in natural language, the distribution of surprise is heavy-tailed: most tokens are predictable (function words, common continuations), and a few are novel (rare words, topic shifts, factual content). the state preferentially stores the rare, novel tokens.

4. **retrieval is generative.** the model does not look up raw stored content. it first predicts what it expects, then adds the stored correction. if the correction is missing (decayed or never stored because it was predictable), the model falls back to its prediction alone. this is the same graceful degradation that biological memory exhibits: old memories drift toward the generative prior.

5. **self-compressing during training (partially falsified 2026-04-14; hypothesis pending trained-prediction test).** at initialization, f is random and r_t ~ h_t (no prediction quality). the state stores approximately raw values, behaving like the current design. as training progresses, f learns to predict and r_t shrinks. the state writes less per token (confirmed via surprise-gated write fraction falling to 1.6% at high predictable-fraction + tau). but in the synthetic-prediction sim, writing less per token did NOT increase the pattern count the memory could hold at matched decay. the self-compression-as-capacity-gain thesis is pending the trained-prediction sim; if that also shows zero memory-side gain, the property reduces to "writes less per token" without a capacity consequence.

6. **no new architectural components.** the prediction function f is a single linear map (or low-rank pair) per layer. the surprise ratio s_t is a scalar computed from two norms already available. the write gate modification g(s_t) is one multiply. total parameter overhead is d^2 per layer for f (or 2 * d * r for the low-rank variant with r << d), which at d=1024 and r=64 is 131k per layer (~3.1m total for 24 layers, ~1% of the 283m model).

## why this is novel

no published architecture implements the specific combination of:
(a) prediction-residual value storage in an explicit outer-product matrix memory with linear readout,
(b) surprise-modulated write gating that reduces effective write frequency based on a separate prediction head,
(c) prediction-based reconstruction at read time by adding retrieved corrections to predictions.

titans comes closest (its gradient update has a structurally similar prediction-error outer product), but differs in memory substrate, read path, and prediction source (see below).

the individual components exist in isolation:
- titans (behrouz et al. 2025) uses surprise-gated writes whose gradient-based update, when expanded for a linear memory, produces a term of the form `(W_0 x_t - x_t) x_t^T` which is structurally a prediction-error outer product. the key differences from the correction-field design are: the memory substrate is MLP weights (not an explicit state matrix), retrieval is a nonlinear forward pass (not a linear matrix-vector product), and the prediction function is implicit in the gradient (not a separate learned head). the overlap is real but the mechanisms differ in substrate, read path, and prediction source.
- deltakv (arxiv 2602.08005) stores residuals relative to historical tokens for kv cache compression, but operates at inference time on a frozen model, not as a training-time mechanism.
- predictive coding (rao and ballard 1999) filters by prediction error in biological cortex, but no published deep learning architecture applies this to the state update of a recurrent outer-product memory.
- dreamer v3 (hafner et al. 2023) stores small latent codes with a generative decoder, but for rl environments, not for autoregressive language memory.

the correction-field design stacks all three as one integrated write-read mechanism in the existing matrix memory. the state matrix becomes a learned map from query directions to prediction corrections. the map writes fewer entries per token as prediction improves (confirmed in simulation), but whether this translates to additional retrievable capacity in the memory itself is still open — the synthetic-q sim said no, trained-prediction sim is pending.

## the self-compression property

this is the deepest novel claim: the state's information content is bounded above by the model's prediction error. the argument below is informal and relies on stated assumptions; it is not a formal proof.

let I(S_t) denote the mutual information between the state S_t and the sequence of hidden states h_1, ..., h_t that produced it. for the current raw-value design, I(S_t) is bounded below by the rank of S_t times the per-dimension entropy of the stored values. for the correction-field design, the following informal bound applies under these assumptions: (a) the state update is deterministic given the residuals and write gates, (b) the write gate b_t is bounded in [0, 1], (c) r_i is conditionally independent of r_j for j > i given S_{i-1}:

```
I(S_t) <= sum_{i=1}^{t} H(r_i | S_{i-1})
```

the inequality follows from the data processing inequality applied to the chain rule for mutual information: S_t is a deterministic function of (r_1, ..., r_t, b_1, ..., b_t, decay), so I(S_t; h_{1..t}) = I(S_t; r_{1..t}) <= H(r_{1..t}) <= sum H(r_i | S_{i-1}) by the chain rule for entropy. note that H(r_i | S_{i-1}) is itself a function of all prior residuals via the state, so the bound is not a closed-form function of prediction quality alone. as the prediction function f improves, each H(r_i | S_{i-1}) decreases (because r_i = h_i - f(h_{i-1}) gets smaller and more concentrated). therefore I(S_t) decreases — the state carries less total information even though it accumulates over more timesteps.

the practical consequence: a well-trained model's matrix memory writes less per token. the information in S is dominated by the genuinely surprising tokens. the predictable majority contributes near-zero to the state's frobenius norm. this is the opposite of the current design, where every token adds its full representation regardless of novelty. caveat: the synthetic-prediction sim (2026-04-14) confirmed the "writes less per token" half of this claim but showed that smaller per-token writes did NOT translate to an increased retrievable pattern count at matched decay. the state carries less information, but the information-to-capacity mapping is weaker than the bound above suggests. whether a trained prediction head with correlated structure recovers a capacity effect is the subject of the pending trained-prediction simulation.

## the cross-layer depth gradient

a second testable prediction (hypothesis, not an established consequence): **compression may increase with depth.**

the prediction function f at layer L predicts layer L's hidden state from the previous timestep's state at the same layer. one hypothesis is that hidden states at deeper layers are more predictable because they have been processed by more layers of context integration. under this hypothesis:

- at layer 1, the prediction quality is low (raw embeddings are hard to predict). the residuals are large. the state stores a lot.
- at layer 12, the prediction quality is moderate. the residuals are medium. the state stores less.
- at layer 24, the prediction quality is high (near the output distribution). the residuals are small. the state stores very little.

a competing hypothesis is that intermediate layers have the largest representational dynamics and therefore the largest residuals, while the very shallow and very deep layers are more predictable. the direction of the depth gradient is an empirical question.

the testable form: at convergence, measure the frobenius norm of S_L at each layer L, normalized by the frobenius norm of the corresponding raw-value state. if the ratio decreases monotonically with depth, the first hypothesis holds and a naturally emergent compression hierarchy exists within the existing layer stack. if the ratio is non-monotonic (e.g., peaks at intermediate layers), the compression hierarchy must be imposed externally rather than emerging from the correction-field mechanism.

if the first hypothesis holds, the shallow layers act as tier 0 (verbatim), middle layers act as tier 1-2 (partially compressed), and deep layers act as tier 3-4 (heavily compressed, near-generative). the 5-tier architecture from `bridge/memory_compression_to_tiered_architecture.md` may emerge from the correction-field mechanism alone.

## the multi-resolution head split

a practical extension: split the attention heads within each layer into groups with different temporal roles.

```
fast heads (e.g., heads 1-8):
    decay = 0.5 (fast forgetting)
    write gate: b_t = sigma(W_b @ h_t), no surprise modulation
    stores: raw residuals for immediate context (last ~4-8 tokens effective window)

medium heads (e.g., heads 9-14):
    decay = 0.95 (slow forgetting)
    write gate: b_t = sigma(W_b @ h_t) * g(s_t, tau=0.1)
    stores: moderately surprising residuals for recent context (~32 token window)

slow heads (e.g., heads 15-16):
    decay = 0.995 (very slow forgetting)
    write gate: b_t = sigma(W_b @ h_t) * g(s_t, tau=0.5)
    stores: only highly surprising residuals (~200 token window)
```

total memory budget is unchanged (same number of heads, same head dimension). the split allocates different heads to different temporal resolutions and compression levels. fast heads behave like the current raw-value design (immediate context). slow heads behave like a heavily filtered long-term store. this implements the temporal hierarchy from the 5-tier proposal without adding any new data structures.

the head split is not a mandatory component. the core design (prediction-residual storage with surprise modulation) works with uniform heads. the split is an optimization for when the core mechanism is validated.

## interaction with existing mechanisms

### decay coefficient

the correction-field design changes the interpretation of decay. in the current design, decay controls retention: higher decay means older raw representations persist longer. in the correction-field design, decay controls how long CORRECTIONS persist. the pre-simulation hypothesis was that since corrections are smaller than raw values, the same decay rate would give a longer effective retention time and the decay sweep's 0.90 threshold would shift downward. **this hypothesis was falsified in the synthetic-prediction sim (2026-04-14).** memory_threshold_shift_by_decay was zero across all tested conditions; the memory still needs decay=0.9 for 32 patterns regardless of whether values are residuals or raw. see testable prediction 5. the trained-prediction sim may change this if real predictions exhibit direction-dependent accuracy that lowers the effective load on specific key directions, but under the synthetic test the decay-tolerance shift does not happen.

### erasure

the correction-field design does not require erasure. erasure subtracts the projection of the state onto the incoming key direction before writing. with raw-value storage, this prevents stacking in a key direction. with residual storage, stacking is already reduced because the residuals in a given key direction get smaller as the prediction improves for that key direction. erasure may still help for keys that map to volatile content (frequently changing corrections), but the hypothesis is that it is less necessary. this is an untested prediction; the overwrite sweep (2026-04-12) tested erasure on raw-value storage and found it harmful at the 32-pattern knee. whether the correction-field design changes that result requires separate simulation.

### dense keys

the design enforces dense keys by construction: keys come from the raw hidden state, not from the residual. the residual only appears in the value projection. this is architecturally baked in, not a convention.

### compressed attention layers

the 4 compressed attention layers in the architecture are unchanged. they provide exact per-token retrieval over a low-rank cache. the correction-field design affects only the 24 matrix-memory layers. the two mechanisms are complementary: compressed attention handles verbatim retrieval for recent context, the correction-field memory handles compressed retrieval for older context.

## parameter budget at 283m scale

```
component                         params per layer    total (24 layers)
W_pred (d x d, or W_down d x r + W_up r x d)  131,072 (low-rank)  3,145,728
surprise threshold tau             1 (if learned)     24
total new parameters (24 matrix-memory layers per PROJECT_PLAN.md glossary)  ~3.15m
percentage of 283m model                              ~1.1%
```

the W_v projection already exists (it projects h_t to get values). the change is that its input switches from h_t to r_t. no new projection matrix is needed for the value side. the only genuinely new parameter is W_pred (the prediction head) and optionally the surprise threshold.

## what this does NOT do

this design does not address:
- **codebook compression** (tier 2). the corrections stored in the state are still continuous vectors, not discrete codebook indices. codebook compression of corrections is a later tier, composable with this one.
- **schema learning** (tier 3). the corrections do not cluster into reusable templates automatically. explicit schema extraction from the correction distribution is a later tier.
- **reconsolidation** (mechanism 6). retrieved corrections are not rewritten in compressed form after retrieval. adding a re-encoding step after each read is a later extension.
- **multi-modal input**. the prediction function f operates on the residual stream, which is modality-agnostic. vision or audio tokens that flow through the residual stream are filtered by the same prediction mechanism.

## testable predictions

1. **capacity increase in memory alone**: FALSIFIED (2026-04-14, synthetic prediction). storing residuals instead of raw values did not increase the matrix memory's own effective pattern count. memory_capacity_gain is zero across all tested prediction qualities. open question: does a trained prediction head with correlated structure change the result?

2. **reconstruction capacity gain**: CONFIRMED (2026-04-14, synthetic prediction). the combined prediction + memory correction has double the effective capacity of the raw memory at prediction_quality >= 0.5. the gain is attributable to the prediction head, not the memory.

3. **write sparsity**: PARTIALLY CONFIRMED (2026-04-14, synthetic). the surprise-gated write fraction drops to 1.6% at predictable_fraction=0.05 and tau=0.99, confirming the filter mechanism works. on real natural language the distribution of surprise needs to be measured before a concrete write-fraction prediction can be made.

4. **depth gradient**: UNTESTED. at convergence, the frobenius norm of S_L (correction-field state at layer L, normalized by the frobenius norm of the raw-value state at the same layer) is hypothesized to decrease with L. requires training a correction-field preset to convergence and measuring per-layer state norms. no synthetic equivalent.

5. **decay tolerance shift**: FALSIFIED (2026-04-14, synthetic prediction). memory_threshold_shift_by_decay is zero. the memory still needs decay=0.9 for 32 patterns regardless of whether values are residuals or raw. open question: does the reconstruction-level threshold (not the memory-alone threshold) shift? the sim's current summary computes both; the reconstruction-level shift is also zero in the synthetic regime, but trained-prediction data may differ.

6. **graceful degradation**: UNTESTED. as the prediction function f is weakened (e.g., by reducing its rank), the correction-field design is hypothesized to smoothly converge to the raw-value design. the synthetic sim at q=0 shows the memory-alone numbers ARE identical (capacity 32 at both), which is consistent with graceful degradation but not a strong test because q=0 is not the same as a weakened trained head.

empirical artifact: `neuroloc/output/simulation_suites/correction_field_capacity/correction_field_capacity_metrics.json` (full sweep, 471s, 17568 trials).

## implementation path (revised post-simulation)

the correction-field design slots into the run sequence from `blueprint.md`, but the rationale for each slot is updated to reflect what the simulation showed:

- **run 1 (baseline dense, pending benchmark)**: no correction field. validates that the base matrix memory with dense keys and no erasure can hold retrievable content at all. this is the precondition for the correction-field to matter: if the memory cannot retrieve anything, adding a prediction head won't help.
- **run 2 (value-side compression, conditional)**: the correction-field design is a candidate IF the trained-prediction cpu sim shows non-trivial memory gain. the synthetic-q sim showed zero memory gain; the trained version is pending. if the trained sim also shows zero memory gain, the design ships as a prediction-head-with-residual-cache (same mechanism, but framed as "the prediction head is the compressor, the memory is a corrector" rather than "the memory self-compresses"). fallback for run 2 if both sims are negative: rate-coded 50% selection on values only, dense keys preserved.
- **run 3 (if run 2 passes)**: add the multi-resolution head split. fast heads with raw values, slow heads with correction-field residuals. the split is itself untested and needs its own cpu sim before this runs.
- **run 4 (if run 3 passes)**: add codebook compression of the corrections stored in the slow heads (tier 2 from the 5-tier proposal). this is where the stored content becomes discrete indices and the per-memory storage cost drops sharply.

the correction-field mechanism and the rate-coded approach are not mutually exclusive. rate-coded truncation can be applied to residuals (compresses what the memory has to hold). if both pass simulation, run 2 can combine them; if only rate-coded passes, run 2 uses rate-coded on values only.

## see also

- `compression_beyond_quantization.md` (the six biological mechanisms; this article implements mechanisms 1-3)
- `bridge/memory_compression_to_tiered_architecture.md` (the 5-tier proposal; this article is the mechanism for tiers 0-1)
- `knowledge/generative_memory_research.md` (curated research on generative replacement; provides the theoretical ceiling)
- `PROJECT_PLAN.md` (canonical project state; must be updated when this design enters the experimental method)
- `spec/blueprint.md` (run sequencing; correction-field design replaces the value-only rate-coding slot)
- `spec/next_gen.md` (roadmap; correction-field is the concrete mechanism for phases 6b-6c)
