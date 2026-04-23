# project research recovery dossier

status: current read-only synthesis (as of 2026-04-23).

scope: this document rebuilds the todorov / neuroloc research state from first principles after the six paid retrieval failures. it records mathematical structure, implemented surfaces, stale documents, and proof obligations. it does not authorize compute, launch experiments, change wiki state, or supersede `neuroloc/wiki/PROJECT_PLAN.md`.

## source hierarchy

canonical project state remains:

- `neuroloc/wiki/PROJECT_PLAN.md`
- `C:\Users\deyan\.claude\plans\compressed-dancing-haven.md`
- `state/program_status.yaml`
- `docs/STATUS_BOARD.md`

raw claude and codex chat history is searchable evidence only. it is useful for reconstructing why a decision was made, but it is not authoritative. do not quote it as project state. do not treat it as clean data. some history surfaces include sensitive pasted material, so search only narrow terms and move the result into canonical files before relying on it.

## current scientific state

the project has a clean negative result: six paid runs returned zero long-distance retrieval at 256 tokens across two memory substrates, two retention regimes, and two corpora. the last run used a synthetic cognition corpus where retrieval was an explicit part of the training distribution and still plateaued at the alphabet prior. that closes the weaker hypothesis that natural text alone was the bottleneck.

the live diagnosis is now architectural trainability. the substrate can retrieve in symbolic or hand-placed settings, but the model path does not learn to route useful information through it under the current initialization, loss, and evaluation surface. the next problem is not "which run next." the next problem is: what mathematical and evaluation structure makes a future intervention scientifically interpretable?

while the curriculum remains the active workstream, paid compute stays paused. when architecture work resumes, one candidate intervention must be selected, cpu-validated, and measured against a broader neural-model battery before any paid run is considered.

## the mathematical spine

the unified theory still gives the right high-level decomposition. each layer is a compressed rotational bilinear recurrence:

```text
z_t = Q(R(B(C(x_t), C(h_{t-1}))))
```

the important part is not the acronym. it is the decomposition:

- `C` chooses which information survives into the operation.
- `B` performs the two-input interaction: outer product, inner product, element-wise product, or geometric product.
- `R` supplies positional or rotational structure.
- `Q` discretizes or bounds the output.

this decomposition says the present failure can come from at least four places. the compression may destroy addressability. the bilinear write/read may have too little capacity. the rotational structure may not preserve the relation the read needs. the output path may suppress a useful memory signal before it can train.

the mistake would be to treat the failed passkey as one scalar verdict. the math says the loop has separate formation, storage, addressing, readout, gating, and loss-coupling stages. each stage needs its own control.

## matrix memory

the matrix memory is a fixed-size outer-product state:

```text
S_t = a S_{t-1} + b k_t v_t^T
o_t = q_t^T S_t
```

with optional erase-before-write variants in historical runs. this is a linear associative store. its capacity is controlled by dimension, key correlation, value correlation, decay, write gate, and readout noise. if keys are orthogonal and values are well-separated, the state can retrieve a small number of patterns. if keys are correlated, every read becomes a mixture. if decay is too low, the state forgets before the query arrives. if the state is added to the residual stream while carrying noise, it can train the rest of the model to ignore or route around it.

the pure-math issue is interference. a query read expands as:

```text
q^T S = sum_i a^{t-i} b_i (q^T k_i) v_i
```

the desired term is the one whose key matches the query. every other term is interference. the project needs a bound or controlled measurement for the ratio between the target coefficient and the sum of non-target coefficients under the actual trained key distribution, not under an ideal orthogonal-key assumption.

the correction-field work did not solve this. storing prediction residuals instead of raw values did not raise the memory-side pattern capacity in either synthetic-prediction or trained-prediction simulations. it survives only as a reconstruction-side bias: a prediction head can improve the final reconstruction, but the outer-product memory itself still collides in the same state space.

## slot memory

slot memory replaces the matrix state with a fixed bank of per-head slots. reads use softmax addressing over slot keys:

```text
w = softmax(q K^T / sqrt(d))
o = w V
```

mathematically, this attacks the right failure. linear readout sums all stored content with weights that can remain nonzero under correlation. softmax can suppress non-matching slots exponentially when the correct slot logit separates from the rest.

but the paid runs show that "can retrieve when addresses are placed" is not the same as "will learn addresses under the training loop." the present slot substrate has at least three trainability hazards:

- the output gate was initialized near closed.
- the prototype keys may collapse to low-entropy slot usage early.
- the retrieval-bearing target positions are a small fraction of byte-level loss, even in a retrieval-shaped corpus.

that means the key question is not only capacity. it is whether gradient descent can discover the addressing convention and open the output path before the rest of the network learns to ignore memory.

## output-gate fixed point

the gate pathology has a simple derivative shape. if a memory path contributes

```text
y = x + sigmoid(g) m_theta(x)
```

then

```text
dL/dg = <dL/dy, m_theta(x)> sigmoid(g)(1 - sigmoid(g))
dL/dtheta is scaled by sigmoid(g)
```

when `g = -4`, `sigmoid(g)` is about `0.018`. the gradient into the memory parameters is scaled down by the same factor. the gradient into the gate depends on `m_theta(x)` already being useful and aligned with the loss gradient. at initialization the memory output is usually noise. so the gate needs useful memory to open, and the memory needs an open gate to become useful. that is a plausible fixed point, and it matches the observed flat gate telemetry in the fixed-retention slot run.

the proof obligation is to split the loop:

- oracle write / learned read
- learned write / oracle read
- open-gate sweep
- frozen open gate with measured slot entropy
- auxiliary target-position loss with and without open gate

without those controls, another paid run only says "the loop failed again."

## softmax addressing and collapse

softmax addressing gives high selectivity only when logits separate cleanly. at initialization, random prototype keys with small variance may produce weak differences. training can then concentrate writes into one or a few slots because early random advantages are reinforced. the symptom is low slot-usage entropy and rank collapse. the result is a slot bank behaving like a noisy accumulator instead of a set of addressable records.

the required math is a logit-margin story. for a query `q`, correct retrieval needs:

```text
q k_target - max_j q k_j >= margin
```

where the margin depends on tolerated leakage. if no training signal creates this margin, softmax capacity is only theoretical. orthogonal prototype initialization is attractive because it gives the addressing geometry a non-collapsed starting point, but it is still only a candidate until write and read controls localize the failure.

## recurrent stability

the project already found two stability classes that matter:

- retention coefficients compound across time, so a harmless-looking per-step value can erase memory over 256 tokens.
- train and eval paths must implement the same recurrence, or chunked evaluation measures a different system from training.

the recovery rule is therefore mathematical parity before performance:

- derive the effective retention product over the tested delay lengths.
- verify recurrent and accelerated paths agree on the same recurrence.
- prevent validation from mutating training-state buffers.
- log the actual retention, gate, slot entropy, and memory-output norm during the run.

if those quantities are missing, the run cannot answer the scientific question.

## implemented surface

the codebase now has a meaningful cpu-first surface:

- `neuroloc/data/nm_worlds.py` defines latent-world episodes with hidden state, object identity, occlusion, motion, feature dropout, and labels.
- `neuroloc/data/cognition_corpus.py` defines the synthetic byte-level cognition corpus used by the last paid run.
- `neuroloc/simulations/memory/contextual_recall_world.py` tests contextual recall.
- `neuroloc/simulations/memory/slot_key_interference_sweep.py` tests interference as key correlation rises.
- `neuroloc/simulations/memory/multi_association_recall.py` tests one-to-many association.
- `neuroloc/simulations/memory/delayed_cue_world.py` tests delayed use.
- `neuroloc/simulations/memory/episodic_separation_completion.py` tests separation and completion.
- `neuroloc/simulations/memory/episodic_replay_reuse.py` tests reuse after distraction.
- `neuroloc/simulations/memory/contextual_gate_routing.py` tests context-gated routing.
- `neuroloc/simulations/suite_registry.py` exposes the `biology_phase1` and `phase1_nm` suites.

the main gap is model-side parity. the symbolic battery exists. the model-side harness that probes the same state, action, and joint-success properties is not yet the center of the implementation. `neuroloc/model/god_machine.py` contains the paid-run model path and evaluation suite, but its current eval surface is still dominated by passkey, copy, length perplexity, and state-structure probes. those remain useful smoke tests, not a full phase-1 neural-model evaluation.

## stale and contradictory surfaces

these are recorded for later wiki hygiene. no wiki/state files are edited by this dossier.

- downstream references to `neuroloc/wiki/synthesis/correction_field_memory.md` should be checked for older trained-prediction-open language. the synthesis article itself now records the completed negative memory-side result, and `neuroloc/wiki/synthesis/compression_beyond_quantization.md` now narrows its correction-field role to reconstruction-side evidence.
- `neuroloc/wiki/synthesis/slot_memory_design.md` contains historical launch-plan wording and optimistic claims about passkey, gate opening, and paid-run expectations that are superseded by `run3_cognition_phase1`.
- `neuroloc/wiki/synthesis/linear_attention_retrieval_wall.md` is correctly marked superseded, but older top-level references still overstate it as live diagnosis. it is evidence, not the current theory.
- `neuroloc/spec/blueprint.md` is a backlog design target, not the active workstream. some active-language and old sequencing claims should be read as historical until cleaned.
- `docs/STATUS_BOARD.md` preserves older phase summaries and an A-C bundling line. the current method in `PROJECT_PLAN.md` says one intervention at a time after cpu validation.
- `state/program_status.yaml` and `docs/STATUS_BOARD.md` still carry 2026-04-22 status language even though the 2026-04-23 curriculum snapshot was pushed elsewhere.

if any of these files are edited later, use `neuroloc/wiki/OPERATING_DIRECTIVE.md`: banner discipline, bidirectional references, append-only update history where required, and a clean prosecutor pass.

## proof obligations before architecture resumes

the project should not resume by picking a tempting paid intervention. it should resume by proving the loop components in isolation.

first, prove the memory math:

- target-to-interference ratio under trained or realistic key correlations
- effective retention over the exact delay lengths tested
- slot-usage entropy and address margin under initialization and early training
- memory-output norm relative to residual stream norm

second, prove trainability:

- open-gate sweep
- orthogonal prototype-key initialization control
- target-position auxiliary-loss sweep
- oracle write / learned read
- learned write / oracle read
- hand-placed address warm start

third, prove evaluation parity:

- each symbolic phase-1 task gets a model-side analogue
- each task reports memory and no-memory baselines
- passkey remains a smoke test, not the sole gate
- success requires state, action, and joint metrics where the task has a behavior component

fourth, preserve compute discipline:

- no paid run without cpu validation
- no paid run without telemetry for gates, slots, retention, and memory-output norms
- no paid run without a prosecutor-clean plan and state update
- no bundled interventions unless the user explicitly waives the one-change discipline in writing

## immediate recovery posture

chapter 1's narrow math correction pass is complete, and chapter 1 remains review-ready. chapter 2 remains paused until outline approval. the research track should use this dossier as a map for cleanup and proof work, not as permission to restart architecture execution.

the next intellectually honest research step is a model-side evaluation design, not a new run. make the model prove that it can form state, retrieve associations, resist interference, use delayed information, reuse episodes, route by context, and benefit from extra internal compute. only then does an intervention result mean anything.
