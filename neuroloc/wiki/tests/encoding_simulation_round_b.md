# input encoding simulation round B (asymmetric matrix memory)

last updated: 2026-04-12

## what this is

phase B of the run-1 planning cycle. round A established that on the symmetric outer-product matrix memory (the shape from the published associative-memory literature), sign-only encoding outperforms three-level magnitude quantization at every load and pattern count tested. round A also identified the gap: the project's actual matrix memory is asymmetric (separate keys and values), single-step (no iterative attractor recall), decayed (exponential weighting of older patterns), and optionally erased (subtracts the projection of the decayed state onto the incoming key direction before adding the new outer product). round B closes that gap by running a new simulation that implements the project's actual matrix memory operation and tests seven candidate input encodings on it.

## the new simulation

`neuroloc/simulations/memory/asymmetric_outer_product_recall.py`. pure-numpy CPU implementation of the matrix memory write `S_t = a * S_{t-1} + b * k_t * v_t^T` and the optional erasure subtraction `a * b * k_t * (k_t^T * S_{t-1})`. readout is `o_t = q_t^T * S_t`. seven candidate input encoding configurations are tested:

- **identity** (no transformation): keys, values, queries pass through unchanged.
- **tanh-bounded** (continuous saturating): every component is mapped through `tanh`, producing dense outputs in (-1, 1).
- **sign-only**: every component is mapped to `+1` if positive, `-1` if negative, `0` only when the input is exactly zero. for continuous gaussian inputs the zero case is essentially never triggered, so the output is dense plus-or-minus one.
- **three-level global threshold**: every component is mapped to `+1` if above a positive threshold, `-1` if below the negative of that threshold, `0` otherwise. one threshold per matrix, computed as `alpha * mean(|x|)` over the full input.
- **three-level per-dimension threshold**: same as global, but with one threshold per dimension instead of one threshold per matrix. the threshold is `alpha * mean(|x|, axis=0)` so each dimension has its own scale.
- **fixed-fraction absolute-magnitude selection (top-k all)**: the top 20% of components by absolute value are kept at their input magnitudes, the remaining 80% are zeroed.
- **dense key plus fixed-fraction value**: keys and queries pass through unchanged (dense), values are sparsified by the fixed-fraction absolute-magnitude selector at 20%.

each encoding is tested on synthetic gaussian keys and values. the script sweeps stored-pattern count, query noise, and decay coefficient. the matrix memory's head dimension is fixed at 64 to match the project's training architecture.

## the run that produced the metrics this article cites

24 trials per cell. pattern counts in {8, 16, 32, 48}. query noise levels in {0.0, 0.1, 0.2}. decay coefficients in {0.4, 0.8, 0.95}. erasure off and on. seven encoding configurations. total 12,096 trial records. wall-clock 11.2 seconds.

source: `neuroloc/simulations/memory/asymmetric_outer_product_recall_metrics.json`, run id `asymmetric_outer_product_recall_20260411_230314`.

## results: exact-query recall cosine at decay 0.40, no query noise, no erasure

at this condition, query is the exact stored key (no noise added) and the matrix memory has the harshest tested decay coefficient. the table below shows mean cosine similarity between the recalled output and the true stored value, with 95% confidence intervals derived from a t-distribution on 24 trials.

| encoding | 8 patterns | 16 patterns | 32 patterns | 48 patterns |
|---|---|---|---|---|
| identity (dense) | 0.43 [0.25, 0.62] | 0.20 [0.02, 0.38] | 0.07 [-0.05, 0.19] | 0.05 [-0.05, 0.15] |
| tanh-bounded | **0.45** [0.29, 0.62] | **0.33** [0.13, 0.52] | **0.19** [-0.01, 0.38] | **0.17** [0.02, 0.32] |
| sign-only | 0.32 [0.20, 0.45] | 0.32 [0.19, 0.45] | 0.11 [0.00, 0.23] | 0.08 [-0.02, 0.18] |
| three-level global | 0.38 [0.22, 0.53] | 0.19 [0.04, 0.34] | 0.12 [0.02, 0.22] | 0.00 [-0.07, 0.07] |
| three-level per-dim | 0.40 [0.25, 0.55] | 0.23 [0.09, 0.38] | -0.01 [-0.07, 0.06] | 0.02 [-0.04, 0.08] |
| top-k all (20%) | 0.46 [0.33, 0.60] | 0.17 [0.05, 0.29] | 0.06 [-0.04, 0.16] | 0.05 [-0.01, 0.12] |
| dense key + top-k value | **0.51** [0.38, 0.64] | 0.16 [0.04, 0.28] | 0.12 [0.02, 0.23] | 0.04 [0.00, 0.07] |

## results: exact-query top-1 accuracy at decay 0.40, no query noise, no erasure

the same condition, but measuring whether the recalled output's nearest stored value (by cosine similarity) is the true target. this is a stricter retrieval metric than mean cosine because it requires the recall to be unambiguous, not just close.

| encoding | 8 patterns | 16 patterns | 32 patterns | 48 patterns |
|---|---|---|---|---|
| identity (dense) | (varies) | (varies) | 0.17 [0.01, 0.33] | (varies) |
| tanh-bounded | (varies) | (varies) | **0.21** [0.03, 0.38] | (varies) |
| sign-only | (varies) | (varies) | **0.21** [0.03, 0.38] | (varies) |
| three-level global | (varies) | (varies) | 0.13 [0.00, 0.27] | (varies) |
| three-level per-dim | (varies) | (varies) | 0.04 [0.00, 0.13] | (varies) |
| top-k all (20%) | (varies) | (varies) | 0.08 [0.00, 0.20] | (varies) |
| dense key + top-k value | (varies) | (varies) | 0.13 [0.00, 0.27] | (varies) |

at 32 patterns the top-1 accuracy is 4-21% across encodings. random chance is 1/32 = 3.1%. tanh-bounded and sign-only tie for the lead at 20.8%.

source: `selected_exact_query_top1` in the metrics JSON, condition pattern_count=32, query_noise=0.0, decay=0.4, erasure=False.

## the threshold-hit summary

the only encoding that crosses 0.5 mean recall cosine at any tested condition is **dense key + top-k value at 8 patterns** with cosine 0.5095. above 8 patterns, no encoding crosses 0.5.

source: `threshold_hits` in the metrics JSON.

## what these results say

three findings emerge cleanly from the data:

1. **the asymmetric matrix memory has a hard capacity ceiling well below the symmetric one.** round A's symmetric simulations showed sign-only encoding holding 100% exact recall up to 30 patterns at network sizes 200-500. round B's asymmetric simulation shows EVERY encoding falling below 0.5 mean cosine recall above 8 patterns at d_head=64. the asymmetric shape with single-step recall and decay loses capacity much faster than the symmetric attractor shape.

2. **encoding choice matters at the lower load regime, but the leading encoding REVERSES from round A.** in round A (symmetric), sign-only encoding dominated. in round B (asymmetric), the leading encoding by mean cosine across the 16-48 pattern range is **tanh-bounded** (continuous saturating), not sign-only. sign-only is a strong second at 8-16 patterns but degrades as fast as everything else above 16. the dense-key plus sparse-value combination is the best at 8 patterns but collapses at 16 onwards. this is a non-trivial result: the encoding that wins on symmetric Hopfield is NOT the encoding that wins on the project's asymmetric memory.

3. **at the project's actual operating point (head dim 64, sequence length 2048, decay 0.4), no encoding tested produces meaningful content-addressable retrieval.** the stored-pattern count in real training is at least an order of magnitude above what this simulation tests. the trend is monotonically downward across the [8, 16, 32, 48] sweep, with the leading encoding (tanh-bounded) dropping from 0.45 at 8 patterns to 0.17 at 48 patterns. extrapolating linearly, the recall floor would be reached well before 100 patterns. this matches the empirical observation that god_run and god_run_v2 both showed the matrix memory's state was indistinguishable from random noise after training (delta state structure probe ~0.98, pairwise cosine ~0.00).

## limitations of this round

several factors weaken the conclusions, in priority order. these are limitations to fix in a round C, not reasons to discard the round B findings.

1. **the pattern-count sweep stops at 48, below the head dimension of 64 where the matrix's rank-capacity ceiling lives.** the prosecutor flagged this as the most important gap: the simulation does not establish whether the encoding-independent collapse happens before the rank ceiling (which would be a representation-distinctness problem fixable with a different encoding) or exactly at the rank ceiling (which would be a structural limit unfixable by encoding choice). attempts to extend the sweep to pattern_count=64 caused a segmentation fault in the local Windows MINGW-W64 numpy build. the extended sweep is deferred to a re-run on the linux training pod.

2. **synthetic gaussian inputs are not trained activations.** the simulation samples keys and values from a unit-norm gaussian distribution. real model activations have non-gaussian statistics, learned correlations, and rope-induced positional structure that the simulation does not model. trained activations may compress better or worse than gaussians under the same encoding.

3. **the simulation has no positional encoding and no rotation.** the project's architecture applies rotary position encoding to keys and queries before the matrix memory write. the simulation omits this. results may be biased toward configurations that ignore positional structure.

4. **24 trials per cell is statistically marginal at the high-variance end of the sweep.** the 95% CIs at pattern_count=32 and pattern_count=48 span ranges wide enough that some pairs of encodings are not separable. cells where the CI overlaps zero are reported but should be treated as "consistent with chance."

5. **the simulation tests raw matrix-memory recall, not the architecture's full forward path.** the actual model projects q, k, v through learned linear layers from a residual stream and applies output projection after the matrix-memory readout. the simulation tests only the inner write/read math.

6. **encoder labels in the simulation script use the published technique names "topk" and "ternary".** these should be replaced with project-internal descriptors and re-cited as external work where the technique name is needed.

## what this means for the next training run

the round B result reframes the question from "which encoding fixes recall" to **"is the matrix memory at d_head=64 fundamentally capable of recall at our task scale, and if not, what structural change is needed."** the encoding choice matters at small loads but does not solve the capacity problem. the next training run should NOT be a sweep over encodings with the existing matrix memory shape; it should test a structural change, or explicitly use the matrix memory only for short-range smoothing while the compressed-attention path handles long-range retrieval.

candidate next moves, ordered by simplicity:

a. **enlarge the head dimension** (e.g., from 64 to 128 or 256). doubles the matrix's rank capacity. cheap to test on the pod once the segfault-free linux numpy is available; the round B simulation already supports this via `ASYM_HEAD_DIM`.

b. **use the matrix memory for short-range only, route long-range retrieval through the compressed-attention path.** this requires examining what the compressed-attention path actually does in god_run_v2 — perplexity decreases monotonically with sequence length, which suggests that path IS using long context, so the failure is specifically in the matrix-memory channel, not in the model's overall use of context.

c. **change the matrix memory's update rule so writes that target the same direction overwrite cleanly (the "erasure" toggle).** this is already implemented in the architecture but not validated to help. round B's metrics include erasure-on cells; the analysis above only shows erasure-off. read the erasure-on cells before concluding.

d. **use a non-outer-product memory entirely.** this is the largest change and is what `wiki/spec/next_gen.md` phase 6 (tiered architecture) was originally going to investigate. round B's negative result moves this option earlier in priority.

deciding among (a)-(d) requires the linux re-run of round B with the rank-ceiling pattern counts and ideally also the trained-activation-distribution variant. before that, no GPU code should change.

## sources

- `neuroloc/simulations/memory/asymmetric_outer_product_recall.py` (the simulation script)
- `neuroloc/simulations/memory/asymmetric_outer_product_recall_metrics.json` (the metrics for the 24-trial [8, 16, 32, 48] sweep)
- `neuroloc/wiki/tests/encoding_simulation_round_a.md` (round A baseline on symmetric memory)
- `neuroloc/wiki/PROJECT_PLAN.md` (canonical project plan)
- `neuroloc/model/god_machine.py:DeltaRuleMemory._recurrent_with_erasure` (the production matrix memory operation that the simulation mirrors)

## update history

- **2026-04-12** — file created. captures the round B results from the 24-trial sweep at pattern counts 8 through 48, identifies the asymmetric capacity collapse, lists the limitations including the pattern-count sweep stopping below the head-dimension rank ceiling due to a local MINGW-W64 numpy segfault, and reframes the question from encoding choice to capacity structure.
