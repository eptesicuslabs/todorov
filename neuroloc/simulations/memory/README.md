# memory simulations

## pattern_completion.py

ca3-like attractor baseline for pattern completion with an explicit shuffled-weight control.

network: 200 binary neurons ({-1, +1}), hebbian outer-product weights $w_{ij} = \frac{1}{n}\sum_p \xi_i^p\xi_j^p$, true in-place asynchronous updates, and a matched shuffled-weight control that preserves the weight histogram while destroying the learned structure.

experiments:
1. convergence example at 30% corruption for hebbian weights and the shuffled control
2. corruption sweep from 0% to 50% with trial-level overlap, bit error rate, and exact retrieval probability
3. load sweep at $n=200$ with exact retrieval probability under 10% corruption
4. scaling sweep across $n \in \{100, 200, 300\}$ with load-fraction comparisons around the $0.138n$ reference point
5. visual comparison of target, cue, hebbian retrieval, and shuffled-control retrieval

dependencies: numpy, scipy, matplotlib

output:
- pattern_completion.png
- pattern_completion_metrics.json

run: `python pattern_completion.py`

quantitative verdict: this script is a controlled attractor-memory baseline, not a validation of hippocampal memory as a whole. it measures how much structured recurrent weights improve recovery relative to a matched control and how retrieval degrades with corruption and load. the $0.138n$ line is included as theoretical context, but this run does not claim to prove the asymptotic hopfield limit.

relevance: this simulation isolates the part of biological memory that todorov's current kda path does not implement: nonlinear attractor cleanup through recurrent dynamics. it is useful as a baseline for future associative-memory circuit work, but it should not be read as evidence that kda already matches ca3 or that hippocampal memory has been captured without pattern separation.

## asymmetric_outer_product_recall.py

asymmetric outer-product recall baseline for the actual write/read geometry the project is using: sequential writes into a matrix state followed by content-addressable readout through a query vector.

memory rule: `s_t = a * s_{t-1} + b * k_t * v_t^T`, with optional overwrite subtraction `s_t -= b * k_t * (k_t^T * s_t)` applied after decay and before the new write. readout is `o_t = q_t^T * s_t`.

experiments:
1. exact-query recall versus stored-pattern count for multiple encoding families
2. noisy-query recall versus query corruption level
3. decay sensitivity for the same state geometry
4. dense-key versus sparse-key comparison at matched sparse value encoding
5. optional erasure-on versus erasure-off comparison at the same operating point

encoding families included:
- identity on queries, keys, and values
- bounded continuous (`tanh`)
- sign-only
- global ternary threshold
- per-dimension ternary threshold
- top-k on all paths
- dense keys with top-k values

dependencies: numpy, scipy, matplotlib

output:
- asymmetric_outer_product_recall.png
- asymmetric_outer_product_recall_metrics.json

run: `python asymmetric_outer_product_recall.py`

quantitative verdict: this script does not claim to model a full trained sequence processor. it isolates the geometry of the write/read rule itself, using synthetic gaussian keys and values, so the project can answer whether a candidate encoding family can support nontrivial recall before any paid compute run.

relevance: this is the blocker-clearing simulation named in `neuroloc/wiki/PROJECT_PLAN.md`. it tests the project's actual asymmetric matrix-memory operation rather than the symmetric attractor baseline from classical hopfield-style literature. it is the right place to compare dense addressing against sparse addressing and to decide whether the current memory substrate is viable before scaling the neural model.

## slot_key_interference_sweep.py

cpu sweep for one of the missing failure classes in the current battery:
correlated keys. the script compares slot softmax readout against the plain
matrix baseline as key correlation increases from near-orthogonal to nearly
collapsed.

output:

- slot_key_interference_sweep.png
- slot_key_interference_sweep_metrics.json

run: `python slot_key_interference_sweep.py`

relevance: this is the direct interference test recommended by the external
phase-1 synthesis. the point is not just "can the mechanism retrieve" but "how
gracefully does retrieval degrade when addresses stop being clean."

## multi_association_recall.py

synthetic one-cue-many-values recall harness. each cue is bound to several
values, and evaluation asks whether the mechanism recovers the whole set rather
than a single item.

output:

- multi_association_recall.png
- multi_association_recall_metrics.json

run: `python multi_association_recall.py`

relevance: this is the small non-language analogue of mqar-style evaluation.
the target is bundle retrieval and set recovery, not single-needle lookup.

## delayed_cue_world.py

small partially observable delayed-cue harness with explicit baseline policies:
no memory, last cue, majority vote, bayes belief, and oracle. the environment
itself is lightweight, but it gives the project a clean delayed-use test
surface before any paid run.

output:

- delayed_cue_world.png
- delayed_cue_world_metrics.json

run: `python delayed_cue_world.py`

relevance: this is the first bridge from pure associative recall to belief-state
tasks. it does not train a neural model by itself; it establishes a cheap
partial-observability harness and baseline gap that later model-side evals can
plug into.

## episodic_separation_completion.py

state-and-action-first episodic indexing harness for separation, completion,
and novelty. each trial stores similar episodes with controllable overlap,
queries one with partial cue, and separately probes whether a near-miss should
be treated as novel rather than collapsed into the wrong stored episode.

output:

- episodic_separation_completion.png
- episodic_separation_completion_metrics.json

run: `python episodic_separation_completion.py`

relevance: this is the first biology-grounded phase-1 family. it is the cheap
cpu analogue of pattern separation, pattern completion, and comparator-style
novelty detection, with explicit oracle, shuffled-mapping, and no-memory
controls plus both state and action metrics.

## episodic_replay_reuse.py

offline replay and reuse harness. a target latent task is seen once, followed
by distractor episodes, then queried again under targeted replay, no replay,
random replay, and recency-only controls.

output:

- episodic_replay_reuse.png
- episodic_replay_reuse_metrics.json

run: `python episodic_replay_reuse.py`

relevance: this is the consolidation/reuse family for phase 1. it does not ask
whether a model can predict future outputs. it asks whether replay helps later
state recovery and action choice after interference.

## contextual_gate_routing.py

bottom-up content plus top-down context routing harness. some trials require
context to disambiguate the same cue; others should ignore context. the
controls are correct context, bottom-up only, shuffled context, forced gate
open, and forced gate closed.

output:

- contextual_gate_routing.png
- contextual_gate_routing_metrics.json

run: `python contextual_gate_routing.py`

relevance: this is the phase-1 context-gating family. it turns the dendritic
and interneuron story into a concrete state-and-action surface with explicit
gate-dependency and false-bind metrics.
