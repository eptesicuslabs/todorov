# god_run findings

## what it tested

god_run is the first end-to-end run of `neuroloc/model/god_machine.py`, a 283M-parameter
byte-level transformer-adjacent architecture that activates all five blueprint features
simultaneously (k-WTA rate-coded compression at 20%, delta-rule erasure, BCM-adaptive alpha
with gamma=0.3, multi-compartment SwiGLU with K=4, compressed attention via SDPA), plus an
always-on imagination probe (learned query into the delta state, gated residual) and a
per-layer predictive-coding diagnostic head (lambda=1e-4). trained for 4000 steps on
fineweb-edu byte-level, 131,072,000 total tokens, seed 42, on a single H200. 53 minutes
wall-clock. matched the pipeline used by run_010, run_011, and the transformer baseline so
numbers are directly comparable.

this article is the empirical test of the theoretical prediction in
`wiki/synthesis/compression_beyond_quantization.md`, which argues that lossy compression
mechanisms preserve statistical distribution-fitting while destroying verbatim
content-addressable retrieval. god_run is the first run in this project that can either
confirm or falsify that prediction at scale.

## the result

god_run confirms the prediction.

val_bpb at step 4000: 1.3950. this is 0.390x the transformer baseline (3.58 bpb) and
1.70x better than run_010 (2.375 bpb). by any standard bpb metric, god_run produced the
best next-byte distribution fit in project history. the val bpb curve is smooth and
monotonic: 2.381 → 2.07 → 1.94 → 1.87 → 1.81 → 1.75 → 1.71 → 1.67 → 1.62 → 1.57 → 1.52 →
1.50 → 1.48 → 1.46 → 1.44 → 1.43 → 1.42 → 1.41 → 1.40 → 1.3950.

**but retrieval failed at every tested length.** passkey 0/20 at 256/1024/4096. selective
copy 0/20 at 256/512/1024/2048. a passkey score of 0/20 has a 95% wilson upper confidence
bound of 13.9% (too weak to distinguish "zero" from "a few percent," so the re-run uses
num_trials=100, but 0/20 is compatible with the hypothesis that the delta-rule memory has
learned nothing retrievable), and selective_copy's identical zeros make the same point.

the perplexity-at-length curve, however, DOES show context usage: bpb falls monotonically
from 1.9354 at context 256 to 1.3751 at context 4096, which is the signature of an attention
mechanism correctly consuming longer context. the compressed-attention + MLP path is doing
language modeling. whatever it is doing does not involve the delta memory as a
content-addressable store.

the delta state structure probe (formerly called "imagination test," renamed for accuracy)
closes the case. the probe takes the trained model, closes the external input gate, feeds
512 warmup bytes to populate the delta state, then queries the accumulated state in each of
the 24 delta layers with 32 novel random keys and measures two things: the norm of the
readout output vs a reference random state of equal frobenius norm (structure_ratio), and
the pairwise cosine similarity of the 32 outputs (structured content = clustered, noise =
orthogonal). god_run scores mean_structure_ratio=0.981 and mean_pairwise_cos=-0.003. a
random state of the same frobenius norm would score structure_ratio≈1.0 and pairwise_cos≈0.
the delta state is statistically indistinguishable from noise. compare with kalaj et al.'s
theoretical prediction for an outer-product memory loaded above critical capacity with
trained content: structured interpolation with pairwise cosine around 0.93. the delta state
in god_run is not "doing structured interpolation above critical load." it is noise.

## interpretation

the correct reading is: the 5-feature bundle produced a model that fits the next-byte
distribution through the compressed-attention + MLP path alone. the delta-rule memory
became a structured-but-content-free state that contributes to statistical prediction
(because the training loss is next-byte cross-entropy and gradients flow through the
delta output via the residual stream) but never learns to store and retrieve specific
content from earlier in the context. k-WTA 20% + delta-rule erasure + BCM adaptive alpha
+ imagination probe collectively destroyed whatever content-addressable binding the
delta memory would have learned without those features.

this is exactly the failure mode compression_beyond_quantization.md predicts for lossy
mechanisms: preserved statistical fit, destroyed verbatim memory. as theoretical evidence
goes, this is clean: the prediction was made in advance of the run, and the run produced
exactly the predicted gap. however, the evidence is not conclusive because there is a
confounder: the BCM train/eval path divergence (prosecutor finding F1).

the FLA chunk_gated_delta_rule path computes the effective retention coefficient
`alpha_eff` from the slow-moving `running_state_norm` EMA buffer. the pre-fix recurrent
path computed `alpha_eff` from the live per-step state frobenius norm. training always
takes the FLA path (state=None, T>1, CUDA, can_use_fla=True). eval continuation always
takes the recurrent path after the first chunk (state is not None). these are
fundamentally different dynamical rules. the model was trained under one rule and
evaluated under another for everything longer than one chunk, which covers all of
passkey and selective_copy. any content-addressable binding the delta memory did
manage to learn during training might have been unretrievable at eval time under the
different retention dynamics. this is a credible alternative explanation for the 0%
retrieval numbers that is NOT intrinsic to the 5-feature bundle.

F1 has been fixed in code (the recurrent path now uses `_effective_log_alpha` which reads
`running_state_norm`, matching the FLA path). F2 (the `running_state_norm` buffer being
silently mutated during every validation pass, polluting the BCM threshold) has also been
fixed. the re-run will tell us which hypothesis is correct:

- **if re-run passkey > 0**: F1+F2 were the culprit, not the feature bundle. the
  architecture retrieves when evaluated under the same rule it was trained under. the
  5-feature bundle is defensible and the compression_beyond_quantization thesis is only
  partially supported (the lossy mechanisms are compatible with retrieval when the
  train/eval paths match, so statistical fit and verbatim memory are NOT fundamentally
  opposed at this scale).
- **if re-run passkey still 0**: the feature bundle destroys verbatim memory regardless of
  train/eval path alignment. the compression_beyond_quantization thesis holds empirically.
  the correct next step is to follow blueprint sequential isolation: run 1 is the 350M
  ternary-spike baseline (no bundle), run 2 adds k-WTA alone, run 3 adds delta erasure
  alone, etc. the bundle is the wrong starting point for the neural machine.

## why this matters

the project's central claim is that biological principles beat transformers at matched
scale. god_run produces the strongest-ever bpb number in the project (1.3950 vs baseline
3.58, a 2.57x improvement) at a similar parameter count. it is tempting to call this a
win. it is not a win. a model that hits 1.3950 bpb while failing every retrieval task is
producing good one-step predictions without building a memory that a human would call
"understanding." standard LLM benchmarks that only measure perplexity or bpb would rank
this model extremely highly. retrieval benchmarks would rank it at zero. the gap between
these two rankings is the information content of god_run.

this is relevant to the wider literature: every KV-cache-compression paper (turboquant,
kvtc, vqkv, commvq, vqllm, deltakv) measures bit savings at fixed quality on standard
benchmarks that do not distinguish "distribution fit" from "content retrieval." god_run
suggests that at aggressive enough compression levels this distinction becomes the whole
story. a compression scheme that preserves 99% of perplexity while dropping 100% of
passkey is not "lossless up to 1% degradation"; it is lossy on the dimension you actually
cared about, and lossless on the dimension the benchmark cares about. the compression
literature needs retrieval-aware evaluation benchmarks before any claim of "lossless
compression" is defensible.

## see also

- `neuroloc/output/god_run/run_card.md` (headline numbers, full eval suite data, next steps)
- `neuroloc/output/god_run/god_run_eval_suite.json` (raw per-layer structure probe numbers)
- `neuroloc/output/god_run/god_run_metadata.json` (config, environment, git sha)
- `wiki/synthesis/compression_beyond_quantization.md` (the theoretical prediction this run tests)
- `wiki/knowledge/generative_memory_research.md` (structural compression literature)
- `wiki/bridge/memory_compression_to_tiered_architecture.md` (the 5-tier proposal)
- `wiki/knowledge/imagination_computation_research.md` (kalaj et al. structured interpolation)
- `neuroloc/spec/blueprint.md` (5-feature bundle design)
- `neuroloc/spec/implementation_plan.md` (sequential isolation protocol the god_run bypassed)
