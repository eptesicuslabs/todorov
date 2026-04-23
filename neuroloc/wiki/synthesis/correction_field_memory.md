# correction-field memory

status: current (as of 2026-04-23).

## the core idea

the matrix memory in its current form stores raw key-value outer products. every token writes its full representation into the state, regardless of whether that representation contains new information or could have been predicted from context. the state fills up, older content decays, and the per-head capacity ceiling at d_head=64 means only ~8 patterns cross the 0.5 cosine threshold under the harshest tested retention setting (simulation round b, 2026-04-12, decay=0.40, dense-key/top-k-value encoding); the 0.14*d theoretical formula gives ~9 for the symmetric no-decay case.

the correction-field reframing changes WHAT gets stored. instead of raw representations, the state accumulates the difference between what the model's own prediction function expects and what actually arrives. the matrix memory becomes a correction field: for any query direction q, the state returns a correction to the model's prediction. the model reconstructs the full representation by combining its prediction with the stored correction.

## empirical status (2026-04-14)

the cpu simulation `simulations/memory/correction_field_capacity.py` ran the capacity sweep (16,800 records across d_head {32,64,128}, pattern_counts {4,8,16,32,64}, prediction_qualities {0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99}, decays {0.5, 0.8, 0.9, 0.95, 1.0}, 32 trials per cell) using a synthetic prediction of the form `pred = q * v + sqrt(1 - q^2) * noise`. the same artifact also includes 768 surprise-gating records, for 17,568 records total. headline findings at d_head=64, decay=0.9:

- **memory_capacity_gain = 0 across every prediction quality.** storing residuals in the state instead of raw values does not increase the matrix memory's own effective pattern count. raw_effective_capacity and correction_field_memory_effective_capacity are both 32.
- **reconstruction_effective_capacity = 64 at prediction_quality >= 0.5**, vs raw_effective_capacity = 32. the combined system (prediction + memory correction) doubles effective capacity, but the gain comes from the prediction, not the memory.
- **memory_threshold_shift_by_decay = 0.** the hypothesized decay-tolerance shift is falsified under this test: the memory still needs decay=0.9 for 32 patterns regardless of whether values are residuals or raw.
- **surprise gating works as designed.** at predictable_fraction=0.05 and tau=0.99, write_fraction_mean drops to 0.016 (1.6% of tokens write). the filter rate scales controllably with tau.

these results partially falsify the article's original claims. the design's value is not a memory-side capacity increase and not a decay-tolerance shift. its value is the predict+reconstruct mechanism: a small prediction head provides most of the content, and the matrix memory only needs to carry what the prediction failed to anticipate. the architecture is a learned-decoder-plus-residual-cache, not a self-compressing memory.

the trained-prediction follow-up has now run. it also returned zero memory-side capacity gain. the remaining value of this design is reconstruction-side: a prediction head can provide a baseline reconstruction and the memory can provide a key-conditioned residual correction, but the matrix memory itself does not become a higher-capacity store.

## what the design actually provides (post-simulation)

1. **predictive filtering** (mechanism 1): confirmed in simulation. surprise-gated writes reduce write frequency in proportion to prediction quality. a well-predicted sequence writes fewer entries, leaving the state less crowded.
2. **generative replacement** (mechanism 2): supported at the reconstruction level. in the synthetic sweep, high prediction quality lets the prediction term supply most reconstructed content. in the trained-prediction follow-up, the prediction term supplies a partial baseline and the memory still supplies the key-conditioned residual correction.
3. **capacity via prediction, not via memory** (revised claim): the reconstruction-level effective capacity doubles over the raw memory when prediction quality >= 0.5, but this gain is from the prediction head, not from a property of the matrix memory itself. the memory's intrinsic capacity ceiling stays at 32 patterns under decay=0.9.

the remaining three mechanisms in the six-mechanism list (content-addressable indexing, manifold abstraction, reconsolidation) are orthogonal and composable with this one. the correction-field design does not prevent later tiers from compressing corrections further via codebook indexing or schema-delta encoding. it is a valid first tier of the compound compression stack, but for different reasons than the pre-simulation article claimed.

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

2. **value norms are smaller when prediction is good.** when prediction explains 90% of the hidden-state variance, ||r_t|| is ~32% of ||h_t|| (since variance is squared norm, ||r||^2 = 0.1 * ||h||^2 implies ||r|| = sqrt(0.1) * ||h|| ~ 0.316 ||h||). the information per value vector drops proportionally. note: the synthetic-prediction sim (2026-04-14) showed this does not translate to an increased per-head pattern count in the memory itself — memory_capacity_gain was zero. the smaller-norm property still helps by enabling a prediction head to dominate reconstruction without the memory needing to carry the full raw content.

3. **write frequency adapts to content.** the surprise modulation g(s_t) means predictable tokens produce near-zero writes. in natural language, the distribution of surprise is heavy-tailed: most tokens are predictable (function words, common continuations), and a few are novel (rare words, topic shifts, factual content). the state preferentially stores the rare, novel tokens.

4. **retrieval is generative.** the model does not look up raw stored content. it first predicts what it expects, then adds the stored correction. if the correction is missing (decayed or never stored because it was predictable), the model falls back to its prediction alone. this is the same graceful degradation that biological memory exhibits: old memories drift toward the generative prior.

5. **write filtering, not memory-side self-compression.** at initialization, f is random and r_t ~ h_t (no prediction quality). as f improves, residuals can shrink and surprise-gated writes can become rarer. the synthetic simulation confirmed the write-filter half of this claim, but both synthetic and trained-prediction simulations found no increased retrievable pattern count in the memory itself. the property is therefore narrowed to "writes less or reconstructs better," not "the matrix memory stores more associations."

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

the correction-field design stacks all three as one integrated write-read mechanism in the existing matrix memory. the state matrix becomes a learned map from query directions to prediction corrections. the map can write fewer entries per token as prediction improves, but the tested memory-side capacity result is negative in both synthetic and trained-prediction regimes.

## the self-compression property

this is the deepest hypothesis, not a proved property: the value payload is bounded by the model's prediction residual, but the full state is not, because the update also stores raw-hidden-derived keys and raw-hidden-derived write gates.

let K_t denote the key sequence and B_t the write gates. under deterministic updates, the defensible informal upper bound is:

```
I(S_t; h_{1..t}) <= H(K_{1..t}, B_{1..t}, r_{1..t})
```

the current raw-value design has a finite state capacity controlled by rank, precision, decay, and interference; this article does not claim a general lower bound from rank alone. the correction-field design can lower the value payload when residuals become smaller and more concentrated, but the raw key channel and write-gate channel still carry information about the hidden sequence. shrinking residuals therefore does not prove that the full state information decreases. the evidence supports write filtering and reconstruction bias, not a formal self-compression theorem.

the practical consequence: a well-trained model's matrix memory may write less per token. the information in S should be dominated by genuinely surprising tokens. the predictable majority should contribute less to the state's frobenius norm. this is the opposite of the current design, where every token adds its full representation regardless of novelty. caveat: the synthetic-prediction sim (2026-04-14) confirmed the "writes less per token" half of this claim but showed that smaller per-token writes did not translate to an increased retrievable pattern count at matched decay. the trained-prediction follow-up did not recover that capacity effect.

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

the correction-field design changes the interpretation of decay. in the current design, decay controls retention: higher decay means older raw representations persist longer. in the correction-field design, decay controls how long corrections persist. the pre-simulation hypothesis was that since corrections are smaller than raw values, the same decay rate would give a longer effective retention time and the decay sweep's 0.90 threshold would shift downward. this hypothesis was falsified in the synthetic-prediction sim and not rescued by the trained-prediction follow-up.

### erasure

the correction-field design does not require erasure. erasure subtracts the projection of the state onto the incoming key direction before writing. with raw-value storage, this prevents stacking in a key direction. with residual storage, stacking is already reduced because the residuals in a given key direction get smaller as the prediction improves for that key direction. erasure may still help for keys that map to volatile content (frequently changing corrections), but the hypothesis is that it is less necessary. this is an untested prediction; the overwrite sweep (2026-04-12) tested erasure on raw-value storage and found it harmful at the 32-pattern knee. whether the correction-field design changes that result requires separate simulation.

### dense keys

the design enforces dense keys by construction: keys come from the raw hidden state, not from the residual. the residual only appears in the value projection. this is architecturally baked in, not a convention.

### compressed attention layers

the 4 compressed attention layers in the architecture are unchanged. they provide exact lookup over compressed per-token representations. the correction-field design affects only the 24 matrix-memory layers. the two mechanisms are intended to be complementary: compressed attention supplies recent-context lookup over cached representations, while a validated correction-field path would add prediction-biased residual corrections for older context.

## parameter budget at 283m scale

```
component                         params per layer    total (24 layers)
W_pred (d x d, or W_down d x r + W_up r x d)  131,072 (low-rank)  3,145,728
surprise threshold tau             1 (if learned)     24
total new parameters (24 matrix-memory layers per PROJECT_PLAN.md glossary)  ~3.15m
percentage of 283m model                              ~1.1%
```

the W_v projection already exists (it projects h_t to get values). the change is that its input switches from h_t to r_t. no new projection matrix is needed for the value side. the only genuinely new parameter is W_pred (the prediction head) and optionally the surprise threshold.

## what this does not do

this design does not address:
- **codebook compression** (tier 2). the corrections stored in the state are still continuous vectors, not discrete codebook indices. codebook compression of corrections is a later tier, composable with this one.
- **schema learning** (tier 3). the corrections do not cluster into reusable templates automatically. explicit schema extraction from the correction distribution is a later tier.
- **reconsolidation** (mechanism 6). retrieved corrections are not rewritten in compressed form after retrieval. adding a re-encoding step after each read is a later extension.
- **multi-modal input**. the prediction function f operates on the residual stream, which is modality-agnostic. vision or audio tokens that flow through the residual stream are filtered by the same prediction mechanism.

## testable predictions

1. **capacity increase in memory alone**: falsified in both synthetic and trained-prediction simulations. storing residuals instead of raw values did not increase the matrix memory's own effective pattern count. memory_capacity_gain is zero across all tested prediction qualities.

2. **reconstruction capacity gain**: confirmed (2026-04-14, synthetic prediction). the combined prediction + memory correction has double the effective capacity of the raw memory at prediction_quality >= 0.5. the gain is attributable to the prediction head, not the memory.

3. **write sparsity**: partially confirmed (2026-04-14, synthetic). the surprise-gated write fraction drops to 1.6% at predictable_fraction=0.05 and tau=0.99, confirming the filter mechanism works. on real natural language the distribution of surprise needs to be measured before a concrete write-fraction prediction can be made.

4. **depth gradient**: untested. at convergence, the frobenius norm of S_L (correction-field state at layer L, normalized by the frobenius norm of the raw-value state at the same layer) is hypothesized to decrease with L. requires training a correction-field preset to convergence and measuring per-layer state norms. no synthetic equivalent.

5. **decay tolerance shift**: falsified in the tested regimes. memory_threshold_shift_by_decay is zero. the memory still needs decay=0.9 for 32 patterns regardless of whether values are residuals or raw. the reconstruction-level interpretation remains separate from the memory-side capacity claim.

6. **graceful degradation**: untested. as the prediction function f is weakened (e.g., by reducing its rank), the correction-field design is hypothesized to smoothly converge to the raw-value design. the synthetic sim at q=0 shows the memory-alone numbers are identical (capacity 32 at both), which is consistent with graceful degradation but not a strong test because q=0 is not the same as a weakened trained head.

empirical artifacts: `neuroloc/output/simulation_suites/correction_field_capacity/correction_field_capacity_metrics.json` (synthetic sweep, 471s, 17,568 records) and `neuroloc/output/simulation_suites/correction_field_trained_prediction_full/correction_field_trained_prediction/correction_field_trained_prediction_metrics.json` (trained-prediction follow-up, 2,880 paired recall records).

## current implementation status

this article is no longer a launch plan. the trained-prediction follow-up closed the memory-side capacity question negatively. the correction-field mechanism may still be useful as a reconstruction-side bias or as one component of an indexed reconstruction system, but it should not be treated as the next memory-capacity intervention for the current matrix memory.

the follow-up direction is now the compact-handle plus reconstruction frame in `indexed_reconstruction_compression.md`: store addresses, schema or latent ids, compact residuals, and provenance; reconstruct through a shared decoder; and evaluate compression under an explicit stored-bit budget.

## see also

- [[compression_beyond_quantization]] (the six biological mechanisms; this article now supplies reconstruction-side evidence only)
- [[memory_compression_to_tiered_architecture]] (the five-tier proposal; this article is evidence for why residual values alone are insufficient)
- [[generative_memory_research]] (curated research on generative replacement; provides the theoretical ceiling)
- [[indexed_reconstruction_compression]] (current compact-handle plus reconstruction direction)
- [[PROJECT_PLAN]] (canonical project state)
- [[tests/correction_field_trained_prediction_results]] (trained-prediction follow-up confirming memory-side null result)

## references

- [behrouz et al. 2025, titans: learning to memorize at test time](https://arxiv.org/abs/2501.00663)
- [deltakv, residual kv-cache compression](https://arxiv.org/abs/2602.08005)
- [rao and ballard 1999, predictive coding in visual cortex](https://www.nature.com/articles/nn0199_79)
- [hafner et al. 2023, mastering diverse domains through world models](https://arxiv.org/abs/2301.04104)
