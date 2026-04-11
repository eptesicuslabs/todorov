# PROJECT_PLAN

This file is the canonical, persistent project state. Any agent picking up the project starts here. Updating this file is a hard rule with prosecutor enforcement (see `CLAUDE.md` and the rule section at the bottom of this file).

last updated: 2026-04-12 (post-god_run_v2 collapse, planning the next run from first principles)

## status

**current run**: none. previous run (`god_run_v2`) finished 2026-04-12 with the long-distance-recall task at 0/100 across every tested distance and the marker-gated copy task at 0/100 across every tested length. the architecture's measured behaviour fits the next-byte distribution well (val cross-entropy 1.4453 nats per byte) but does not perform verbatim recall at any distance. the 5-feature bundle that produced this result has been ruled broken.

**current question**: what input encoding for the architecture's outer-product matrix memory preserves the most retrievable structure while remaining trainable under standard gradient descent? we do not yet know this. the published literature (citing external technique names: Hopfield 1982, the k-winners-take-all family, the magnitude-quantization-with-adaptive-threshold family, the softmax-as-content-addressable-memory result of Ramsauer et al. 2021) gives partial answers for nearby problem shapes but does not directly answer for our architecture's exact memory operation.

**current obstacle**: we have run two large training experiments without first validating the encoding choice on the cheap testing environment. both burned compute. the new plan does the empirical validation first and only spends paid compute on configurations that have measurable retrieval in the testing environment.

**immediate next action**: run three existing testing-environment experiments in `neuroloc/simulations/` to baseline what we already know about magnitude-quantized vs continuous encodings on the symmetric outer-product memory. then write a new testing-environment experiment for our actual asymmetric memory operation, derive the encoding from the result, and only then touch GPU code.

## hypothesis under test

**hypothesis**: there exists an input encoding for the architecture's matrix memory write `S_t = a * S_{t-1} + b * k_t * v_t^T` (with optional erasure subtraction `a * b * k_t * (k_t^T * S_{t-1})`) such that recall via `o_t = q_t^T * S_t` produces verbatim retrieval of stored values when queried with their corresponding (or near-corresponding) keys.

**falsifiable claim**: at the parameters of the planned training run (head dimension 64, sequence length 2048, decay coefficient ~0.4 per step), some encoding from the candidate set produces recall fidelity above 0.5 cosine similarity at a stored-pattern count of at least 32. this is the threshold below which the architecture cannot in principle support the long-distance-recall evaluation task at 256 tokens.

**status**: untested. previous training runs assumed the answer existed without measuring it.

## experimental method

we have two testing environments in this project:

1. **simulation directory** at `neuroloc/simulations/` — pure-python and brian2 scripts that test biological and mathematical hypotheses on CPU in seconds to minutes. used to validate candidate ideas before paid compute.
2. **paid GPU compute** at H200 scale via runpod — used to validate candidates that survive simulation.

the experimental method for the next run is:

1. **run existing simulations** that test magnitude quantization on outer-product memory: `simulations/memory/capacity_scaling.py`, `simulations/prototypes/rate_coded_spike.py`, `simulations/sparse_coding/hierarchical_ternary.py`. record what each says about the candidate encoding family.
2. **identify the gap**: existing simulations test the symmetric, iterative attractor memory shape from classical literature. the architecture we use is asymmetric (write k onto v, read with q), single step, and decays with optional erasure. existing simulations do not directly test our shape.
3. **build a new simulation** at `simulations/memory/asymmetric_outer_product_recall.py` that tests our actual memory operation under candidate encodings. prosecutor pass before running. run it. read every output metric.
4. **derive the math** for the encoding that wins in the new simulation. write it to `wiki/synthesis/run1_encoding_math.md`. prosecutor pass.
5. **implement the chosen encoding** in `neuroloc/model/god_machine.py` as one logical commit per code change. prosecutor pass on every commit. fix every finding to zero before the next commit.
6. **smoke test on pod** when the user provides a new pod link. halt on any smoke failure.
7. **launch one training run** of ~50 minutes wall-clock at H200 scale. cron-monitor at off-minute cadence.
8. **analyse results** and update this file plus the wiki run history.

each phase has a wiki article deliverable that captures what we learned. each phase has a prosecutor pass on its deliverables. no phase is complete until prosecutor returns clean.

## decision rules

- **if the chosen encoding produces nonzero long-distance-recall at 256 tokens after training**: the encoding is validated for run 1. proceed to run 2 with the next variable changed (likely the decay coefficient or erasure flag), keeping the encoding fixed. document the validated encoding in this file and in `wiki/synthesis/compression_beyond_quantization.md`.
- **if the chosen encoding produces zero long-distance-recall at every distance**: the architecture's outer-product memory may be fundamentally incapable of content-addressable retrieval at our task scale, regardless of encoding. document the negative result, update this file, ask the user how to proceed. the next planning round investigates whether retrieval has to come from a structurally different component (e.g., the compressed-attention path, or a new component the project does not yet have).
- **if the smoke test on pod fails before training launches**: halt. do not launch training. ask the user. fix locally first.
- **if the new simulation in phase B produces results that contradict the existing simulations from phase A**: do not pick a winner. write up the contradiction in the round-A and round-B wiki articles. ask the user how to interpret.

## prior runs index

- **`god_run`**: 2026-04-11. 283M params. five-feature bundle (k-WTA at 20%, delta erasure, BCM-adaptive decay, multi-compartment FFN, compressed attention) plus always-on imagination probe and predictive-coding diagnostic head. trained successfully (val 1.3950 nats/byte). long-distance recall: 0/20 at every tested distance. marker-gated copy: 0/20 at every tested length. probe metric stream lost approximately 35 keys due to a step-logger bug. run card at `neuroloc/output/god_run/run_card.md`.
- **`god_run_v2`**: 2026-04-12. same architecture as `god_run`, with all 17 prosecutor findings F1-F17 plus 14 follow-up findings G1-G14 fixed before launch. critical fix: F1 corrected the recurrent path's effective decay coefficient from `sigmoid(logsigmoid(pre))` to `exp(logsigmoid(pre))`, eliminating a ~28% train/eval divergence. probe metric stream now records 74 keys per step. trained (val 1.4453 nats/byte, ~0.05 worse than v1). long-distance recall: 0/100 at distances 256, 1024, 4096 (Wilson 95% upper bound 3.7%). marker-gated copy: 0/100 at lengths 256, 512, 1024, 2048. perplexity at length series, all five points: 1.9254 at 256 tokens, 1.8941 at 512, 1.5373 at 1024, 1.4776 at 2048, 1.4192 at 4096 (monotonic decrease, confirming the attention path uses context). delta state structure probe: structure_ratio 0.977, pairwise cosine 0.003 (near-orthogonal, statistically indistinguishable from random state of equal Frobenius norm). **the F1 math fix did not recover recall.** run card at `neuroloc/output/god_run_v2/run_card.md`.
- **prior runs `run_001` through `run_011`**: see `state/program_status.yaml` and `docs/EXPERIMENT_LOG.md`. none of these runs measured the long-distance-recall task or the marker-gated copy task. they are blind to the retrieval question.

## open research questions

1. does the architecture's asymmetric outer-product matrix memory support content-addressable retrieval at any encoding?
2. if yes, which encoding family preserves the most retrievable structure?
3. does the optimal encoding for the key half differ from the optimal encoding for the value half? biological cortex pattern-separates keys aggressively (~5% sparse) and stores values densely. does the project's architecture want the same asymmetry?
4. does the erasure subtraction term help or hurt at the chosen encoding?
5. does the decay coefficient matter — is a larger decay (faster forgetting) better, equal, or worse than a smaller one for retrieval at long distance?
6. is there a structural change to the architecture that would allow a different memory operation to handle retrieval, while the matrix memory handles a different role (e.g., short-term smoothing)?
7. once we have a validated encoding from simulation, does it scale to the H200 training environment without losing its property — i.e., does standard gradient descent at 283M params find the encoding's predicted operating point?

## glossary

This is the project's own vocabulary. Every term here describes a thing the project is building, not a thing imported from a published paper. Each entry maps a project term to its mathematical definition. Adding a new term to the glossary is part of the prosecutor cycle.

- **architecture**: the neural model the project is building. has no external name. it is a sequence-processing model with two layer kinds (the matrix memory and the compressed attention) interleaved at a 6 to 1 ratio. byte-level vocabulary, 1024-dimensional residual stream, 28 layers (24 matrix-memory layers and 4 compressed-attention layers).
- **matrix memory**: the per-layer state-keeping operation `S_t = a * S_{t-1} + b * k_t * v_t^T`, optionally with erasure `S_t -= a * b * k_t * (k_t^T * S_{t-1})`. shape `(batch, num_heads, head_dim, head_dim)`. read via `o_t = q_t^T * S_t`. exists in 24 of the 28 layers.
- **compressed attention**: the per-layer operation that projects the residual stream to a low-rank cached representation per token, then computes scaled dot-product attention over the cache with rotary positional encoding concatenated onto the keys. exists in 4 of the 28 layers.
- **decay coefficient `a`**: the per-step retention factor in the matrix memory's update. computed at the start of each forward pass as `exp(logsigmoid(alpha_log_base))` for the static case, or with an optional adaptive subtraction term for the activity-dependent case. range (0, 1).
- **write gate `b`**: the per-step write strength in the matrix memory's update. computed per token as `sigmoid(beta_proj(x_t))`. range (0, 1).
- **erasure**: the optional subtraction term `a * b * k_t * (k_t^T * S_{t-1})` in the matrix memory's update. when present, every write subtracts the projection of the existing state onto the incoming key direction before adding the new key-value outer product, so writes targeting the same direction overwrite cleanly instead of stacking. when absent, the matrix memory is a pure accumulator: every write adds, nothing subtracts, and writes targeting the same direction stack indefinitely until decay removes them. the subtraction is only honoured when the fast-path kernel is used; the python recurrent path also applies it when its toggle is true.
- **input encoding**: the function applied to keys, values, and queries before they enter the matrix memory's update. the choice of input encoding is the question this run cycle is testing. the candidate space currently includes: identity (no transformation), three-level magnitude quantization with one global threshold per layer, three-level magnitude quantization with one threshold per dimension, fixed-fraction absolute-magnitude selection (keep the largest k of d entries, zero the rest), sign-only encoding (map to plus-or-minus one based on sign of the input, zero only when input is exactly zero), continuous bounded encoding (saturating nonlinearity with no zeros), and any new encoding that simulation work justifies adding.
- **long-distance recall task**: the evaluation in which a five-byte sequence is hidden inside a longer noise sequence and the architecture is asked to reproduce it after the noise. the architecture passes if it produces the correct sequence. tested at distances 256, 1024, 4096 tokens. used as the hard gate for retrieval.
- **marker-gated copy task**: the evaluation in which the architecture is asked to copy a marker-prefixed substring from earlier in the input. tested at lengths 256, 512, 1024, 2048 tokens. used as a secondary retrieval gate.
- **structure ratio**: the metric `mean ||probe_output|| / mean ||random_state_output||` measured over 32 random query vectors against the matrix memory's accumulated state after a fixed warmup. structure ratio near 1.0 means the matrix memory's state is statistically indistinguishable from random noise of the same Frobenius norm; structure ratio significantly below 1.0 (or above 1.0 with high pairwise cosine) means the state has stored content. used to diagnose whether the matrix memory has accumulated meaningful structure during training.
- **firing rate**: the fraction of dimensions in an encoded vector that are nonzero after the encoding is applied. depends on the chosen encoding and on the input distribution. measured per layer per step.
- **prosecutor pass**: a code review carried out by the `feature-dev:code-reviewer` agent (the non-claude-suffix variant). the review identifies findings ranked by priority. the rule is that every finding must be either fixed or explicitly waived in writing by the user before the related code is allowed to run on paid compute or to be published as a wiki article.

## update history

every entry below is an append-only record of a change to this file. format: `date — one-line description of what changed and why`. the project's sole author is the human who runs the project; no other authorship credit is given here.

- **2026-04-12** — file created. captures god_run_v2 result, the bundle-is-broken diagnosis, the immediate next action of running three existing simulations followed by writing a new one, and the project glossary's first entries. supersedes the implicit "what happens next" knowledge that previously lived only in conversation context.

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
