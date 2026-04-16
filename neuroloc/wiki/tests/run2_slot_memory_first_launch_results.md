# run card: run2_slot_memory (first launch with broken retention)

status: historical context only. frozen as of 2026-04-15. do not edit.

## one-line result

first paid launch of the slot-memory substrate. h200, 355M params, 4000
steps on fineweb-edu. final val_bpb 1.5107. passkey 0/100 at every
tested length. the preset inherited `alpha_log_mean=-0.5` from Config
defaults, reproducing the state-evaporation failure mode the project
had documented four days earlier. this run measured the retention bug,
not the substrate. the substrate was relaunched with retention fixed
as `run2_slot_memory_retention_fixed` on the same day.

## headline numbers

- preset: `run2_slot_memory` (before commit `7abb781`)
- params: 355M
- layer pattern: 24 SLOT + 4 ATTN (SLOT×6 + ATTN×1 × 4)
- substrate change: slot memory with softmax addressing over 64 prototype
  keys per head, surprise-gated lru writes, FLA-accelerated via
  `fused_recurrent_simple_gla`
- retention: `alpha_log_mean=-0.5` (inherited from Config defaults —
  this is the bug). `alpha_eff=sigmoid(-0.5)=0.3775`. over 256 tokens:
  `0.3775^256 ≈ 10^-109`, below float32 epsilon. state evaporates
  before the passkey query can read it.
- compute: h200, fineweb-edu byte-level, 131M tokens
- training: 4000 steps, ~72 min wall clock
- best val_bpb: 1.5107

## eval

- passkey @ 256/1024/4096: 0/100 at every length
- selective_copy @ 256/512/1024/2048: 0/100 at every length
- delta_state_structure_probe: returned
  `{"error": "no DELTA layers had populated state"}` because the
  run2_slot_memory preset replaces every DELTA block with SLOT and the
  probe only reads DELTA blocks. the `0.000` visible in the step log is
  the logger's `.get(..., 0)` default for a missing key, not a
  measurement.

## diagnosis

self-inflicted config bug. the slot preset was written to replace the
matrix substrate but did not override `alpha_log_mean`, so it inherited
the broken default that the project had documented four days earlier
in `wiki/synthesis/linear_attention_retrieval_wall.md` evidence line 4
(now superseded by `wiki/synthesis/training_objective_vs_architectural_goal.md`).
the result was architecturally predetermined.

fix in commit `7abb781` sets `alpha_log_mean=5.0` in the preset
(`alpha_eff=0.9933`, `0.9933^256 ≈ 0.18`, well above float epsilon).
subsequent audit found the same inherited-default was present in
`run1_baseline_noerasure` and `run4_erasure_ablation`; those presets
were also fixed, and a structural guard `_assert_preset_retention_safe`
was added to god_machine.py to prevent the class of bug from recurring.

relaunched as `run2_slot_memory_retention_fixed` on 2026-04-15 with
retention fixed and FLA actually active (the initial relaunch attempt
lost 17 min to FLA silently falling through to the python recurrent
loop because `flash-linear-attention` was not pinned in
`requirements.txt`; fix in commit `edcfe5d`).

## see also

- `wiki/tests/run2_slot_memory_retention_fixed_results.md` — the
  relaunch with retention fixed
- `wiki/mistakes/run2_slot_memory_decay_copy_paste.md` — the mistake
  doc for the retention bug
- `wiki/mistakes/run2_slot_memory_fla_silent_fall_through.md` — the
  mistake doc for the FLA-not-installed silent slowdown on the relaunch
- `wiki/synthesis/slot_memory_design.md` — the substrate being tested
- `wiki/synthesis/training_objective_vs_architectural_goal.md` — the
  post-five-runs realisation
- `state/program_status.yaml` → `run_history.run2_slot_memory`
