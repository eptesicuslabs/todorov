# run card: run3_cognition_phase1 (paid)

status: historical context only. frozen as of 2026-04-17. do not edit.

## one-line result

sixth paid run. first run on a corpus explicitly designed to require
memory routing — synthetic cognition with 50% passkey / 30% kv recall /
20% copy blocks. 355M params, slot substrate, retention fixed, FLA
active, 4000 steps, 72 min, partial eval before the user stopped the
pod. **val_bpb 6.3519 (plateaued at the alphabet prior from step 150;
no further descent). passkey @ 256 = 0/100, @ 1024 = 0/100.** the
substrate did not learn retrieval on a task that explicitly required it.

## headline numbers

- preset: `run3_cognition_phase1`
- git sha: `f27b81c`
- config hash: `4e3f02be249cc069`
- params: 355,281,024
- layer pattern: (SLOT, SLOT, SLOT, SLOT, SLOT, SLOT, ATTN) × 4 — 24 SLOT + 4 ATTN
- alpha_log_mean: 5.0 (alpha_eff = 0.9933, retention 0.18 at 256 tokens)
- FLA: enabled and active (`fused_recurrent_simple_gla`)
- compute: H200 144GB, bf16 AMP, gradient checkpointing on
- training corpus: synthetic cognition via `neuroloc/data/cognition_corpus.py`
  - 200,000,000 bytes total; 190M train / 10M val
  - block_seq_len = 512
  - mix: passkey 0.5 / kv recall 0.3 / copy 0.2
  - passkey distances sampled from {16, 64, 128, 256, 512}
  - hex-digit ASCII alphabet (0-9, a-f); printable filler (32-126); distinct marker triples per task
- training: 4000 steps, 131,072,000 tokens, 4322 sec wall-clock = 72 min
- throughput: steady 32,800-32,900 tok/sec

## training trajectory

- step 0: loss 5.74, val_bpb not yet measured
- step 50: loss 4.51
- step 150: first val, val_bpb 6.3524
- steps 150-4000: loss in a narrow band [4.34, 4.42], val_bpb in a narrow band [6.35, 6.37]
- best val_bpb 6.3519 (recorded at step 2150; never improved past it)
- final val_bpb 6.3627

the model trained to the alphabet-prior floor at step ~150 and stayed
there for the remaining 3850 steps.

## eval (partial, interrupted)

- passkey @ 256 tokens: 0/100 (0.0%, Wilson 95% CI upper 3.7%)
- passkey @ 1024 tokens: 0/100 (0.0%, Wilson 95% CI upper 3.7%)
- passkey @ 4096: not reached
- selective_copy at 256/512/1024/2048: not reached
- perplexity at length: not reached
- delta_state_structure_probe: not reached (would have returned the
  expected error "no DELTA layers had populated state" for a SLOT-only
  preset)
- retrieval_gate: not persisted (pod stopped before eval completed)

## how this compares to the five prior paid runs

| # | run | val_bpb | passkey@256 | substrate | retention | corpus |
|---|---|---|---|---|---|---|
| 1 | god_run | 1.3950 | 0/20 | matrix | inherited -0.5 | fineweb-edu |
| 2 | god_run_v2 | 1.4453 | 0/100 | matrix | inherited -0.5 | fineweb-edu |
| 3 | run1_baseline_noerasure | 1.4499 | 0/100 | matrix | inherited -0.5 | fineweb-edu |
| 4 | run2_slot_memory (first) | 1.5107 | 0/100 | slot | inherited -0.5 | fineweb-edu |
| 5 | run2_slot_memory_retention_fixed | 1.4777 | 0/100 | slot | explicit 5.0 | fineweb-edu |
| 6 | run3_cognition_phase1 (this) | 6.3519 | 0/100 | slot | explicit 5.0 | synthetic cognition |

run 3 is the only paid run on a corpus designed to require memory. its
val_bpb sits 4.9 bpb higher than runs 1-5 because the cognition corpus
is intentionally higher-entropy (hex digits + filler + markers with no
natural-language redundancy to exploit). the numeric comparison is not
val_bpb — it is passkey accuracy, and across all six runs and every
axis varied (substrate, retention, corpus, retention init) the answer is
0 / 100 at 256 tokens.

## what the run validates

- the FLA startup guard (`_assert_fla_available_if_requested`) correctly
  admitted the preset when the package was present on the pod; the
  previous failure class from `run2_slot_memory_fla_silent_fall_through`
  cannot recur silently.
- the cognition corpus generator and the NM_DATASET=cognition routing
  work correctly end-to-end: 200M bytes generated in ~1 min, training
  loop consumed them at 32,800 tok/sec steady, markers visible in every
  batch.
- training ran without NaN, without gradient explosion, without eval-
  time crash, without throughput degradation.
- the retention fix (alpha_log_mean=5.0) is stable over 4000 steps on
  cognition data as well as on fineweb-edu.

## what the run does NOT validate

- whether the slot substrate is trainable by SGD on natural text. that
  question was the one run 2 tried to answer; run 2 returned 0% and
  this run narrows the interpretation further: even on data that
  explicitly requires memory, the substrate does not learn retrieval
  within 4000 steps at 355M params.
- whether longer training would change the outcome. the val_bpb
  plateaued at step 150 and stayed there for 3850 more steps; there is
  no evidence of any slow-to-emerge descent. an 8000- or 16000-step
  run is predicted to be 0% passkey as well.

## diagnosis

see `wiki/synthesis/substrate_requires_architectural_change.md` for the
full analysis. summary: the prior diagnosis
(`training_objective_vs_architectural_goal.md`, 2026-04-16) proposed
that cognition-shaped data would discriminate between "LM loss is the
problem" and "the architecture cannot be trained." this run is the
discriminant. the architecture cannot be trained on retrieval within
the configuration the project has been using, even when the loss
explicitly requires retrieval. the next step is architectural: a
different init, a different auxiliary loss, a warm start from hand-
placed addresses, or a different substrate entirely.

## cost

- training: ~$3.50 of paid h200 compute (72 min at ~$3/hr)
- eval (partial before kill): ~$0.25
- total: ~$4

## see also

- `wiki/synthesis/phase1_evaluation_surface_for_neural_models.md` — the later external-synthesis article that broadens phase-1 evaluation beyond passkey-only gating
- `wiki/synthesis/synthetic_shared_world_bridge.md` — the later bridge article for extending the same latent-world tests into phase 2
- `wiki/synthesis/substrate_requires_architectural_change.md` — the post-run-3 analysis
- `wiki/synthesis/training_objective_vs_architectural_goal.md` — the prior analysis that this run was designed to discriminate
- `wiki/synthesis/slot_memory_design.md` — the substrate under test
- `wiki/tests/run2_slot_memory_retention_fixed_results.md` — the previous paid run on fineweb with same substrate + retention
- `neuroloc/data/cognition_corpus.py` — the corpus generator this run used
- `wiki/PROJECT_PLAN.md` — the canonical project state
- `wiki/INDEX.md` — full wiki navigation map
- `wiki/OPERATING_DIRECTIVE.md` — rules governing this article
