# project_plan

this file is the canonical, persistent project state. any agent picking up the project starts here. updating this file is a hard rule with prosecutor enforcement (see `CLAUDE.md` and the rule section at the bottom of this file).

last updated: 2026-04-12 (head-dimension sweep complete, asymmetric capacity ceiling still holds, retention and overwrite analysis now prioritized)

## status

**current run**: none. previous run (`god_run_v2`) finished 2026-04-12 with the long-distance-recall task at 0/100 across every tested distance and the marker-gated copy task at 0/100 across every tested length. the architecture's measured behaviour fits the next-byte distribution well (final val 1.4453 bpb) but does not perform verbatim recall at any distance. the five-feature bundle that produced this result has been ruled broken. the next scheduled run is run 1 baseline dense on h200, and phase 6a aesthetic logging starts in parallel from the frozen `god_run_v2` artifacts.

**current question**: does the baseline dense matrix-memory architecture recover nonzero passkey at 256 tokens when keys and values stay dense, erasure stays on, and the auxiliary mechanisms that previously confounded retrieval are removed? if the answer is still no, the base delta-rule mechanism itself is suspect and long-range retrieval likely has to move elsewhere.

**current obstacle**: the last full bundle fit the distribution while destroying verbatim retrieval, so the project has to isolate the base mechanism before adding sparsification or adaptive decay again. the simulation program has already ruled out an encoding-only rescue at the old operating point, but it has not yet answered whether the dense baseline is sufficient.

**immediate next action**: run 1 baseline dense launches next on h200 with dense keys, dense values, erasure on, and the extra mechanisms removed. the hard gate is nonzero passkey at 256 tokens. the completed simulation sequence already narrows what this run should and should not test: `neuroloc/wiki/tests/encoding_simulation_round_a.md` showed that the symmetric toy problem prefers sign-only encoding, `neuroloc/wiki/tests/encoding_simulation_round_b.md` showed that the project's asymmetric memory does not recover meaningful recall above 8 stored patterns at d_head=64, and `neuroloc/wiki/tests/head_dim_sweep_results.md` showed that widening alone is sub-linear with $p^*(32)=16$, $p^*(64)=16$, $p^*(128)=16$, $p^*(192)=24$, and $p^*(256)=32`. every threshold-crossing cell in that width sweep came from the slower-retention setting `decay=0.8`; at `decay=0.4`, only the 32-dimensional run crosses 0.5, and only at 8 patterns. the next paid run is therefore baseline isolation, not a head-dimension-only tweak. in parallel, keep the simulation program active on retention and overwrite behavior and start phase 6a aesthetic logging from the frozen `god_run_v2` artifacts so the first follow-up run is chosen by evidence rather than bundle intuition.

## hypothesis under test

**hypothesis**: the baseline dense matrix-memory architecture can recover measurable verbatim retrieval once dense keys and values are restored and the auxiliary mechanisms that corrupted the address space are removed.

**falsifiable claim**: run 1 baseline dense produces nonzero passkey at 256 tokens. if it still scores 0/100, the base delta-rule mechanism is not carrying retrievable content strongly enough for the task and feature add-ons should stop until the mechanism is rethought.

**status**: pending. the narrower encoding-only claim is already closed negative: the d_head=64 sweep explicitly includes pattern_count=32, decay=0.4, and 32 trials per cell, and every evaluated encoding stays below 0.205 mean raw cosine there. that failure is why the project is no longer scheduling an encoding-only paid run.

## experimental method

we have two testing environments in this project:

1. **simulation directory** at `neuroloc/simulations/` — pure-python and brian2 scripts that test biological and mathematical hypotheses on cpu in seconds to minutes. used to validate candidate ideas before paid compute.
2. **paid compute** at h200 scale via runpod — used to validate candidates that survive simulation.

the experimental method for the next run is:

1. **keep the simulation track active** under `neuroloc/simulations/`, but only for retention, overwrite, and retrieval-role questions that affect the post-baseline decision. the encoding-only branch is already closed negative at the old operating point.
2. **start phase 6a aesthetic logging in parallel** from the frozen `god_run_v2` artifacts. this is a standalone low-risk track that should not block run 1 baseline dense.
3. **launch run 1 baseline dense** on h200 with dense keys, dense values, erasure on, no activity-adaptive decay, no compartmental feedforward split, no imagination branch, and no predictive-coding diagnostic head.
4. **smoke test before launch** and halt on any smoke failure.
5. **treat nonzero passkey at 256 as the hard gate**. if the baseline dense run cannot clear that gate, stop adding features and re-evaluate the base mechanism.
6. **if run 1 passes, launch run 2** with value-side rate-coded compression only while keys remain dense.
7. **if run 2 passes, launch run 3** with activity-adaptive decay using the revised slower-moving state tracker.
8. **do not reintroduce key-side sparsification, imagination-side readout branches, compartmental feedforward splits, or predictive-coding heads** until each earlier run has preserved the retrieval gate.
9. **analyse every completed run** and update this file, the run cards, and the relevant wiki notes before the next launch.

each phase has a wiki article deliverable that captures what we learned. each phase has a prosecutor pass on its deliverables. no phase is complete until prosecutor returns clean.

## decision rules

- **if run 1 baseline dense produces nonzero passkey at 256 tokens**: the base mechanism survives isolation. lock dense keys in place, record the result here, and proceed to run 2 with value-side rate-coded compression only.
- **if run 1 baseline dense produces zero passkey at every tested distance**: the base delta-rule mechanism is suspect even without the extra confounds. document the negative result, update this file, and shift the next planning round toward a structural rethink rather than another bundle tweak.
- **if the smoke test on pod fails before training launches**: halt. do not launch training. ask the user. fix locally first.
- **if a supporting simulation contradicts a completed run artifact**: treat the run artifact as the launch gate, document the contradiction, and use the simulation only to target the next low-cost hypothesis test.

## prior runs index

- **`god_run`**: 2026-04-11. 283m params. five-feature bundle (20% rate-coded compression on keys and values, delta erasure, activity-adaptive decay, compartment-split feedforward block, compressed attention) plus always-on imagination probe and predictive-coding diagnostic head. trained successfully (final val 1.3950 bpb). long-distance recall: 0/20 at every tested distance. marker-gated copy: 0/20 at every tested length. probe metric stream lost approximately 35 keys due to a step-logger bug. run card at `neuroloc/output/god_run/run_card.md`.
- **`god_run_v2`**: 2026-04-12. same architecture as `god_run`, with all 17 prosecutor findings F1-F17 plus 14 follow-up findings G1-G14 fixed before launch. critical fix: F1 corrected the recurrent path's effective decay coefficient from `sigmoid(logsigmoid(pre))` to `exp(logsigmoid(pre))`, eliminating a ~28% train/eval divergence. probe metric stream now records 74 keys per step. trained (final val 1.4453 bpb, ~0.05 worse than v1). long-distance recall: 0/100 at distances 256, 1024, 4096 (Wilson 95% upper bound 3.7%). marker-gated copy: 0/100 at lengths 256, 512, 1024, 2048. perplexity at length series, all five points: 1.9254 at 256 tokens, 1.8941 at 512, 1.5373 at 1024, 1.4776 at 2048, 1.4192 at 4096 (monotonic decrease, confirming the attention path uses context). delta state structure probe: structure_ratio 0.977, pairwise cosine 0.003 (near-orthogonal, statistically indistinguishable from random state of equal Frobenius norm). **the F1 math fix did not recover recall.** run card at `neuroloc/output/god_run_v2/run_card.md`.
- **prior runs `run_001` through `run_011`**: see `state/program_status.yaml` and `docs/EXPERIMENT_LOG.md`. none of these runs measured the long-distance-recall task or the marker-gated copy task. they are blind to the retrieval question.

## open research questions

1. does the architecture's asymmetric outer-product matrix memory support content-addressable retrieval at any encoding?
2. if yes, which encoding family preserves the most retrievable structure?
3. does the optimal encoding for the key half differ from the optimal encoding for the value half? biological cortex pattern-separates keys aggressively (~5% sparse) and stores values densely. does the project's architecture want the same asymmetry?
4. does the erasure subtraction term help or hurt at the chosen encoding?
5. does the decay coefficient matter — is a larger decay coefficient (slower forgetting, more retention) better, equal, or worse than a smaller one for retrieval at long distance?
6. is there a structural change to the architecture that would allow a different memory operation to handle retrieval, while the matrix memory handles a different role (e.g., short-term smoothing)?
7. once we have a validated encoding from simulation, does it scale to the h200 training environment without losing its property — i.e., does standard gradient descent at 283m params find the encoding's predicted operating point?

## glossary

this is the project's own vocabulary. every term here describes a thing the project is building, not a thing imported from a published paper. each entry maps a project term to its mathematical definition. adding a new term to the glossary is part of the prosecutor cycle.

- **architecture**: the neural model the project is building. has no external name. it is a sequence-processing model with two layer kinds (the matrix memory and the compressed attention) interleaved at a 6 to 1 ratio. byte-level vocabulary, 1024-dimensional residual stream, 28 layers (24 matrix-memory layers and 4 compressed-attention layers).
- **matrix memory**: the per-layer state-keeping operation `S_t = a * S_{t-1} + b * k_t * v_t^T`, optionally with erasure `S_t -= a * b * k_t * (k_t^T * S_{t-1})`. shape `(batch, num_heads, head_dim, head_dim)`. read via `o_t = q_t^T * S_t`. exists in 24 of the 28 layers.
- **compressed attention**: the per-layer operation that projects the residual stream to a low-rank cached representation per token, then computes scaled dot-product attention over the cache with rotary positional encoding concatenated onto the keys. exists in 4 of the 28 layers.
- **decay coefficient `a`**: the per-step retention factor in the matrix memory's update. computed at the start of each forward pass as `exp(logsigmoid(alpha_log_base))` for the static case, or with an optional adaptive subtraction term for the activity-dependent case. range (0, 1).
- **write gate `b`**: the per-step write strength in the matrix memory's update. computed per token as `sigmoid(beta_proj(x_t))`. range (0, 1).
- **erasure**: the optional subtraction term `a * b * k_t * (k_t^T * S_{t-1})` in the matrix memory's update. when present, every write subtracts the projection of the existing state onto the incoming key direction before adding the new key-value outer product, so writes targeting the same direction overwrite cleanly instead of stacking. when absent, the matrix memory is a pure accumulator: every write adds, nothing subtracts, and writes targeting the same direction stack indefinitely until decay removes them. the subtraction is applied in both execution paths when the toggle is true; which path runs depends on backend availability and sequence shape.
- **input encoding**: the function applied to keys, values, and queries before they enter the matrix memory's update. it remains a tunable part of the mechanism, but the current paid-run branch is not an encoding-only sweep because the earlier simulation work already closed that branch negative at the old operating point. the candidate space currently includes: identity (no transformation), three-level magnitude quantization with one global threshold per layer, three-level magnitude quantization with one threshold per dimension, fixed-fraction absolute-magnitude selection (keep the largest k of d entries, zero the rest), sign-only encoding (map to plus-or-minus one based on sign of the input, zero only when input is exactly zero), continuous bounded encoding (saturating nonlinearity with no zeros), and any new encoding that simulation work justifies adding.
- **long-distance recall task**: the evaluation in which a five-byte sequence is hidden inside a longer noise sequence and the architecture is asked to reproduce it after the noise. the architecture passes if it produces the correct sequence. tested at distances 256, 1024, 4096 tokens. used as the hard gate for retrieval.
- **marker-gated copy task**: the evaluation in which the architecture is asked to copy a marker-prefixed substring from earlier in the input. tested at lengths 256, 512, 1024, 2048 tokens. used as a secondary retrieval gate.
- **structure ratio**: the metric `mean ||probe_output|| / mean ||random_state_output||` measured over 32 random query vectors against the matrix memory's accumulated state after a fixed warmup. structure ratio near 1.0 means the matrix memory's state is statistically indistinguishable from random noise of the same Frobenius norm; structure ratio significantly below 1.0 (or above 1.0 with high pairwise cosine) means the state has stored content. used to diagnose whether the matrix memory has accumulated meaningful structure during training.
- **firing rate**: the fraction of dimensions in an encoded vector that are nonzero after the encoding is applied. depends on the chosen encoding and on the input distribution. measured per layer per step.
- **prosecutor pass**: a code review carried out by the `feature-dev:code-reviewer` agent (the non-claude-suffix variant). the review identifies findings ranked by priority. the rule is that every finding must be either fixed or explicitly waived in writing by the user before the related code is allowed to run on paid compute or to be published as a wiki article.

## update history

every entry below is an append-only record of a change to this file. format: `date — author — one-line description of what changed and why`. the project's sole author is the human who runs the project; no other authorship credit is given here.

- **2026-04-12** — deyan todorov — file created. captures god_run_v2 result, the bundle-is-broken diagnosis, the immediate next action of running three existing simulations followed by writing a new one, and the project glossary's first entries. supersedes the implicit "what happens next" knowledge that previously lived only in conversation context.
- **2026-04-12** — deyan todorov — phase A simulation round complete. three existing simulations (`neuroloc/simulations/memory/capacity_scaling.py`, `neuroloc/simulations/prototypes/rate_coded_spike.py`, `neuroloc/simulations/sparse_coding/hierarchical_ternary.py`) ran on cpu, producing metrics that consistently rank sign-only encoding above three-level magnitude quantization on symmetric outer-product memory recall. continuous bounded encoding produces high cosine similarity but zero exact recall. zeros in the encoded vector destroy the address space. the gap to the project's architecture (asymmetric, single-step, decayed, optionally-erased) is identified and documented in `neuroloc/wiki/tests/encoding_simulation_round_a.md`. round B (asymmetric simulation) is the next deliverable.
- **2026-04-12** — deyan todorov — phase B simulation round complete on partial pattern-count sweep [8, 16, 32, 48]. the new simulation `neuroloc/simulations/memory/asymmetric_outer_product_recall.py` implements the project's actual matrix memory operation (asymmetric, single-step, decayed, optional erasure) at head dimension 64 and tests seven candidate input encoding configurations. 24 trials per cell, 12,096 trial records. **leading finding: at head dim 64, no encoding tested produces meaningful recall above 8 patterns. the only crossing of 0.5 mean cosine recall is dense-key plus sparse-value at 8 patterns (0.51).** at 32 patterns the leader is tanh-bounded continuous encoding at cosine 0.19 and top-1 accuracy 0.21 (random chance is 0.031). the encoding leader reversed from round A: round A's sign-only winner is now the second-best at low loads and degrades as fast as the others above 16 patterns. the result is a capacity-ceiling finding, not an encoding finding. at the time, the rank-ceiling test (pattern_count = 64 = head_dim) was deferred because a local windows mingw-w64 numpy build segfaulted at that pattern count. four candidate next moves are listed in `neuroloc/wiki/tests/encoding_simulation_round_b.md`: enlarge head dim, route long-range through compressed attention, test the erasure-on cells, replace the matrix memory entirely. user direction was needed before any paid-compute code changes.
- **2026-04-12** — deyan todorov — head-dimension sweep complete for the asymmetric matrix-memory simulation at head dims 32, 64, 128, 192, and 256. the threshold curve is sub-linear: $p^*(d) = {16, 16, 16, 24, 32}$. every threshold-crossing cell comes from the slower-retention setting `decay=0.8`; at `decay=0.4`, only the 32-dimensional run crosses 0.5 mean raw cosine, and only at 8 patterns. this partially answers candidate move (a): wider state helps, but it is not a standalone fix at the current faster-forgetting operating point. the next priority is retention and overwrite analysis, not a pure head-dimension paid run.
- **2026-04-12** — deyan todorov — canonical plan updated to the post-`god_run_v2` dense-baseline branch. the next scheduled paid run is run 1 baseline dense on h200 with dense keys, dense values, and erasure on; the simulation track stays focused on retention and overwrite questions; and phase 6a aesthetic logging starts in parallel from the frozen `god_run_v2` artifacts. this entry records the shift away from the older encoding-first launch plan and aligns the plan with `state/program_status.yaml`, `docs/STATUS_BOARD.md`, and the revised sequencing in `neuroloc/spec/blueprint.md`.

## the rule

`neuroloc/wiki/PROJECT_PLAN.md` is the canonical project state. it MUST be updated whenever any of the following happens:

- a training run starts
- a training run completes (with a one-paragraph entry in the prior runs index pointing to its run card)
- a hypothesis is added or falsified
- the experimental method changes
- a new project term enters the glossary
- a prior research question is answered (move it from open research questions to a paragraph somewhere relevant, with the answer)

failing to update this file is the same class of error as failing to run the prosecutor: both are blockers for paid compute launches. an outdated `PROJECT_PLAN.md` is a bug to fix, not noise to ignore.

the update history at the bottom of this file is append-only. every change to this file gets a new entry with date, author, and a one-line description of what changed and why.
