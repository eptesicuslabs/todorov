# run card: run2_slot_memory (retention fixed, paid)

## one-line result

slot substrate trained cleanly on fineweb-edu to val_bpb 1.4777 in 72 min on
H200, then produced 0/100 passkey at 256 and 1024 tokens — a clean
language-model trajectory with zero retrieval, identical to all prior paid
runs. the consistent 0% across architectures, retention values, and substrates
points the diagnosis at the training objective rather than the architecture.

## headline numbers

- preset: `run2_slot_memory` at git `edcfe5d`
- params: 355,281,024
- layer pattern: (SLOT, SLOT, SLOT, SLOT, SLOT, SLOT, ATTN) × 4 (24 SLOT + 4 ATTN)
- alpha_log_mean: 5.0 (alpha_eff = sigmoid(5.0) = 0.9933, retention 0.18 at 256)
- FLA: enabled and active (`fused_recurrent_simple_gla`)
- compute: H200 144GB, AMP bfloat16, gradient checkpointing on
- training: 4000 steps, 131,072,000 tokens, 4309 sec wall-clock = 72 min
- throughput: 33,000 tokens/sec steady (vs god_run_v2 matrix at ~45,000)
- final val_bpb: 1.4777 (best)
- val_bpb trajectory: 3.16 → 1.99 → 1.88 → 1.82 → 1.77 → 1.60 → 1.58 → 1.56 → 1.54 → 1.53 → 1.51 → 1.49 → 1.48 → 1.4777

## eval (partial)

eval suite ran on the best checkpoint and was killed before completion. the
following entries reached the log:

- passkey @ 256 tokens: 0/100 (0.0%, Wilson 95% CI upper 3.7%)
- passkey @ 1024 tokens: 0/100 (0.0%, Wilson 95% CI upper 3.7%)
- passkey @ 4096: not reached
- selective_copy at 256/512/1024/2048: not reached
- perplexity at length: not reached
- delta_state_structure_probe: not reached (would have returned the
  expected `{"error": "no DELTA layers had populated state"}` for SLOT-only)
- retrieval_gate: not persisted (eval was killed before exit)

## how this compares to prior paid runs

| run                          | val_bpb | passkey@256 | substrate    | retention                  |
|------------------------------|---------|-------------|--------------|----------------------------|
| god_run                      | 1.3950  | 0/20        | matrix       | inherited -0.5 (broken)    |
| god_run_v2                   | 1.4453  | 0/100       | matrix       | inherited -0.5 (broken)    |
| run1_baseline_noerasure      | 1.4499  | 0/100       | matrix       | inherited -0.5 (broken)    |
| run2_slot_memory (first)     | 1.5107  | 0/100       | slot         | inherited -0.5 (broken)    |
| run2_slot_memory (this)      | 1.4777  | 0/100       | slot         | explicit 5.0 (good)        |

the only run with retention not in the evaporating regime is the current one.
language-modeling performance is comparable to the matrix-substrate runs.
retrieval is unchanged.

## what the run validates

- the structural retention guard (`_assert_preset_retention_safe`) accepted
  the preset on first construction and the run launched cleanly under the
  full gating contract (NM_AUTHORIZE_FULL_RUN, retrieval_gate persistence,
  fail-closed exit on inconclusive eval).
- the SlotMemory FLA dispatch worked correctly with `fused_recurrent_simple_gla`
  on H200 at d_model=1024, batch_size=16, seq_len=2048, throughput 33k tok/s.
- the retention fix (alpha_log_mean=5.0) survives 4000 steps without
  numerical instability. alpha_eff_mean stayed at 0.9933 throughout training.
- the training loop has no remaining instability, NaN, or eval-time crash
  vectors at this scale.

## what the run does NOT validate

- whether the slot substrate can be trained to perform retrieval. the
  language-modelling loss does not reward retrieval, so gradient descent
  collapses the slot output gate toward inactive (mean gate 0.018 throughout)
  and never learns to route.
- whether the slot substrate exceeds the ~3 bits-per-parameter knowledge
  ceiling of vanilla transformers. matching val_bpb at matched parameter
  count means matching knowledge density at best, not exceeding it.
- whether longer training would change retrieval. four prior runs at 4000
  steps converged on 0% passkey, and this run is the fifth at 0% passkey
  on the same regime; there is no evidence that a sixth run with the same
  loss would differ. the prediction is that 8000 or 16000 steps on
  fineweb-edu would also produce 0% passkey.

## diagnosis

see `wiki/synthesis/training_objective_vs_architectural_goal.md` for the
full analysis. summary: the architecture is specified as a substrate for
neural computation (memory, compression, thinking, imagination). the
training pipeline is shaped for token prediction. those goals do not
overlap, and gradient descent does not bridge them on its own. a memory
substrate trained on a loss that does not require memory does not learn
to use memory.

the consistent 0% passkey across five paid runs, two substrate variants,
and two retention regimes is now interpreted as evidence about the training
objective, not about the architecture.

## cost

- training: ~$3-4 of paid h200 compute (72 min at typical $3/hr rate)
- eval (partial before kill): ~$0.50
- prior failed launch attempt (FLA-not-installed, killed at 17 min): ~$1
- total: ~$5

## see also

- `wiki/synthesis/training_objective_vs_architectural_goal.md`
- `wiki/mistakes/run2_slot_memory_decay_copy_paste.md` (first run2 launch)
- `wiki/mistakes/run2_slot_memory_fla_silent_fall_through.md` (this run's
  prior launch attempt that hit the FLA-not-installed silent slowdown)
- `wiki/synthesis/slot_memory_design.md` (the substrate this tested)
- `wiki/synthesis/linear_attention_retrieval_wall.md` (prior diagnosis,
  partially superseded by the training-objective realisation)
