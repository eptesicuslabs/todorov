# next gen spec

see also: `neuroloc/wiki/PROJECT_PLAN.md` is the canonical persistent project state. this next_gen.md covers the post-baseline architectural direction (tiered compression, thinking loop, vision); PROJECT_PLAN.md covers what is being run right now. read PROJECT_PLAN.md first when picking up the project.

## scope

this is the roadmap for what comes after god_run_v2. god_run v1 produced val_bpb 1.3950 with
0% passkey; god_run_v2 tests whether the F1 torch.exp math correction recovers retrieval,
but regardless of that outcome the next generation of the architecture moves in this
direction. blueprint.md covers sequential feature isolation on the current single-tier
design; this file covers the next architectural leap.

## direction (set by deyan 2026-04-11, compression elevated 2026-04-12)

1. **compression** (primary): the correction-field memory design from
   `wiki/synthesis/correction_field_memory.md` is the concrete mechanism for the first
   three of the six biological compression mechanisms. the 5-tier architecture from
   `wiki/bridge/memory_compression_to_tiered_architecture.md` is the longer-term target;
   the correction-field is the first tier validated by simulation before gpu spend.
   compression is the lead research direction because the matrix memory's capacity
   ceiling at d_head=64 (~8 patterns crossing 0.5 cosine in simulation round b at
   decay=0.40; ~9 from the 0.14*d theoretical formula) is the binding constraint on
   everything else. nothing else matters until the memory can hold retrievable content.
2. **thinking**: intermediate representations generated during reasoning and fed back into
   the forward path.
3. **imagination**: promoted from "gated residual probe" to core computational primitive
   on every forward pass.
4. **vision**: first-class modality, not a bolt-on encoder.
5. **dual logging**: the existing metrics jsonl stream continues; a new aesthetic artifact
   stream captures the creative/generative side of the model as images and video.

## hard rules

- no weight growth. the architecture must absorb new capacity through the tiered compression
  (fixed codebooks, fixed schemas, fixed prediction heads) and not by adding parameters at
  runtime.
- provenance audit log for every memory write across all tiers. enables targeted unlearning
  via delta-rule erasure when a source is flagged malicious.
- no feature bundling without per-feature validation. god_run and god_run_v2 both
  demonstrated that activating five novel features simultaneously produces an uninterpretable
  result; v2 proved the failure persists even with the F1 math fix. every tier, every
  modality, every feedback loop must be added one at a time with isolation gates.
- logging must round-trip to disk. no in-memory assertions without a jsonl (or image file)
  read-back verification. same rule that god_run v1 violated and god_run_v2 fixes.
- **keys must stay dense.** this is the hardest rule learned from god_run_v2: competitive
  top-k selection at 20% on keys drops the per-head content-addressable capacity from ~9
  interference-free patterns to ~2, and delta erasure with sparse keys leaves ghost content
  in the zeroed dimensions
  that accumulates across steps. any form of sparsity, compression, or address-space
  truncation must be applied to VALUES only, never to KEYS. a memory tier that compresses
  keys is no longer content-addressable. this invariant applies to all five tiers,
  including the predictive residual buffer (tier 1) and the latent codebook (tier 2):
  the codebook compresses what is stored, not how it is addressed.
- **imagination is read-only in phase 6-9.** any feedback mechanism that reads from the
  delta state must not have its own trainable weights in a way that creates an
  alternative gradient path to the output. the god_run_v2 imagination probe had
  `imag_filter_down/up` as trainable MLPs whose output was added to the residual stream;
  the model learned to use them as a bypass instead of training the delta memory. if a
  feedback loop has trainable parameters, those parameters must only receive gradient
  through the memory, not around it.

## phase ordering

### phase 6a: dual-stream logging (lowest risk, independent of architecture)

- extend `MetricsLogger` or add a sibling `AestheticLogger` that writes image artifacts to
  `output/<run_name>/aesthetic/step_<N>/`.
- minimum viable artifacts at each val_interval:
  - per-layer delta state frobenius norm heatmap (24 delta layers x steps)
  - imag_ratio_mean and alpha_eff_mean time series plot
  - kwta_k_rate_per_layer and pc_error_l2_per_layer distribution plots
  - top-k per-token predictions on a fixed sample prompt (show "what the model thinks
    comes next")
- write as PNG via matplotlib, one image per artifact per val step.
- total cost: a few hundred lines of code, no math, no architectural change.
- validates: the logging infrastructure for everything that follows.

### phase 6b: correction-field validation (compression primary)

- **cpu simulation first**: run `simulations/memory/correction_field_capacity.py` to test
  whether storing prediction residuals increases effective capacity at matched parameters.
  the simulation uses synthetic prediction quality to control the compression ratio
  without needing a trained model. the checked-in harness now lives at
  `simulations/memory/correction_field_capacity.py`; treat its results as evidence only
  after the required prosecutor review and a recorded run artifact.
- **if simulation confirms capacity increase**: implement the correction-field write path
  in `god_machine.py` as a new config option. the prediction head is a low-rank linear map
  W_down @ W_up (d x r x d, r=64) per layer, ~131k params per layer, ~3.1m total (~1%
  overhead). the value projection's input switches from h_t to r_t = h_t - f(h_{t-1}).
  keys remain dense (from raw h_t). the write gate gains a surprise modulation factor.
- **validation on h200**: train a correction-field preset at 283m scale. gate: passkey
  at 256 tokens with at least 4/100 successes (wilson 95% lower bound > 0%) AND measurable
  capacity increase over the raw-value baseline (measured via the structure ratio probe).
- **cross-layer depth gradient measurement**: at convergence, measure frobenius norm of
  S_L at each layer L, normalized by the raw-value baseline. the prediction is monotonic
  decrease with depth.
- replaces the prior phase 6b plan (inter-layer prediction head) because the correction-
  field design IS the prediction head, applied within each layer's matrix memory rather
  than between layers. the self-compression property subsumes the "90% variance explained"
  gate: if the prediction head explains 90% of variance, the state stores ~3x less
  information per token, which is directly measurable as capacity increase.

### phase 6c: codebook compression of corrections (tier 2 prototype)

- if phase 6b confirms the correction-field design, apply discrete codebook indexing to
  the corrections stored in the slow heads. train a learned codebook encoder/decoder on
  correction vectors (prediction residuals), not on raw hidden-state snapshots. 8192-entry
  codebook, 64-dim latents. measure reconstruction quality of corrections and downstream
  retrieval retention.
- this replaces the prior plan to apply discrete codebook compression to raw hidden states.
  corrections are smaller and more structured than raw states, so the codebook should
  achieve better coverage with fewer entries.
- standalone, takes ~1-2 weeks of compute on a single gpu.
- validates: whether prediction residuals from the correction-field layer are amenable to
  discrete codebook representation and whether codebook-compressed corrections preserve
  retrieval quality.

### phase 6d: schema learning (tier 3 prototype)

- online schema extraction: find clusters of similar hidden states across the training
  corpus, each cluster = a schema template. store per-memory (schema_id, delta) pairs.
- measure: schema coverage (what fraction of hidden states are close to a schema),
  delta size distribution (how small is the per-memory residual), retrieval quality under
  schema+delta reconstruction.

### phase 6e: 5-tier integration (god_machine_v3 / tiered_machine)

- full integration of tiers 0-4 plus provenance audit log plus delta-rule erasure as a
  safety primitive. this is the big architectural change.
- new file: `neuroloc/model/tiered_machine.py`. based on god_machine but with the tier
  hierarchy replacing the single-tier delta state.
- retains god_machine_v2 as a reference baseline for ablation.
- decision point: does the tiered version achieve the brain-like compression ratio (several
  orders of magnitude beyond single-tier) without catastrophic forgetting?

### phase 7: thinking loop

- intermediate-representation feedback during generation. the model produces a hidden state,
  feeds it back as a reasoning input, produces a refined hidden state, and so on, for a
  variable number of "think steps" before emitting the next token.
- text-only first. measure reasoning benchmarks (GSM8K, math, reading comprehension) against
  the non-thinking baseline.
- this is the intermediate-representation feedback mechanism reframed for text.

### phase 8: vision modality

- patch-based image encoder that maps image tokens into the residual stream.
- joint training on text + vision. the delta-rule memory is modality-agnostic by design,
  so vision tokens flow through the same compression and thinking machinery.
- measure: image classification, image description, visual question answering.

### phase 9: embodied / sensor

- add proprioception, motion, continuous sensor streams. the tiered memory and
  predictive-filter mechanism should absorb any continuous modality without architectural
  change.

## what is NOT in this spec

- this spec does not commit to a specific schedule. phases 6-9 could take weeks to months
  each. commit to one phase at a time, validate, then commit to the next.
- this spec does not replace `neuroloc/spec/blueprint.md`. blueprint covers the feature
  isolation protocol on the current single-tier design. next_gen.md covers the leap beyond
  single-tier. both are valid; god_run_v2's eval decides which is near-term.

## next action (post god_run_v2, 2026-04-12)

god_run_v2 returned. passkey 0/100 at 256/1024/4096, copy 0/100 at all tested lengths,
val_bpb 1.4453 (vs v1's 1.3950). bundle-is-broken branch of the decision rule. the F1
torch.exp math fix was not the root cause; the bundle intrinsically destroys verbatim
memory via the eight root causes documented in `memory/project_v2_diagnosis.md`.

the concrete forward path is dual-tracked:

**track 1 (single-tier fix, corrected sequential isolation, near-term):**

follow the revised blueprint.md ordering: run 1 is the `god_machine.py` preset
`run1_baseline_noerasure` at 283m (dense keys, dense values, no erasure, non-FLA path,
no auxiliary mechanisms), run 2 is value-side compression with dense keys (leading
candidate: correction-field design from `wiki/synthesis/correction_field_memory.md`),
run 3 is activity-adaptive decay with faster EMA, run 4 is erasure ablation after the
dense-key baseline is validated. each run gates on retrieval (passkey > 0 at 256 tokens
is a HARD gate, a run that drops passkey to 0 is rejected regardless of BPB improvement).
do NOT add imagination, multi-compartment, or PC diagnostic until retrieval is validated
at each step. see blueprint.md for the full run-by-run detail.

**track 2 (compression + aesthetic logging, phase 6+):**

- phase 6a (aesthetic logging): start immediately after god_run_v2 artifact pull. independent
  of track 1. matplotlib PNG artifacts per val_interval showing the creative side of the
  model — delta state frobenius heatmap, imag_ratio/alpha_eff_mean time series, kwta rate
  distributions, top-k predictions on a fixed prompt. low risk, lightweight, validates
  the logging infrastructure for phases 7+.
- phase 6b (correction-field validation): cpu simulation first
  (`simulations/memory/correction_field_capacity.py`), then implementation in god_machine.py,
  then h200 validation. this is the primary compression milestone. the correction-field
  design compounds three of the six biological compression mechanisms (predictive filtering,
  generative reconstruction, residual coding) in a single write-read modification to the
  existing matrix memory. see `wiki/synthesis/correction_field_memory.md` for the full design.
- phase 6c (codebook compression of corrections): if phase 6b succeeds, apply discrete
  codebook indexing to the corrections stored in the slow heads. this compounds mechanism 4
  (content-addressable indexing) with the first three. replaces the prior "vq-vae on hidden
  states" plan because it compresses corrections (smaller, structured) not raw states.
- phase 6d (schema learning): extract reusable correction templates from the codebook. each
  new correction is stored as (template_id, small delta). adds schema-delta encoding
  (mechanism 3 from the six-mechanism list) on top of the correction-field (mechanisms 1-2)
  and codebook (mechanism 4). standalone.
- phase 6e (5-tier integration): new file `tiered_machine.py`, retains god_machine as
  ablation baseline. dense-key invariant enforced at every tier. the correction-field is tier
  0-1, codebook is tier 2, schema is tier 3, generative-only is tier 4.
- phase 7 (thinking): intermediate-rep feedback during generation. only after phase 6e.
- phase 8 (vision): multimodal encoder feeding the tiered stream.

track 1 runs in parallel with track 2 phase 6a-6b. track 2 phase 6c+ happens only after
track 1 gives us a working baseline with nonzero retrieval AND track 2 phase 6b confirms
the correction-field capacity increase in simulation. if track 1 run 1 itself fails
(i.e., baseline dense also produces 0% passkey), then the entire delta-rule mechanism is
suspect and we re-examine the base mechanism before building compression tiers on top of it.

immediate next action: finish pulling god_run_v2 artifacts, write the run card, commit
and push. then start phase 6a aesthetic logging prototype as a standalone skill, writing
to `output/<run_name>/aesthetic/step_<N>/*.png`. low risk, valuable for all subsequent
phases.

## see also

- `wiki/bridge/memory_compression_to_tiered_architecture.md` (5-tier architectural detail)
- `wiki/synthesis/compression_beyond_quantization.md` (thesis: lossy mechanisms preserve fit,
  destroy retrieval)
- `wiki/tests/god_run_findings.md` (empirical evidence from god_run v1)
- `neuroloc/spec/blueprint.md` (current single-tier design and sequential isolation protocol)
- `neuroloc/spec/implementation_plan.md` (5-run sequential protocol)
- `memory/project_next_gen_direction.md` (persistent memory entry with user's direct quote)
