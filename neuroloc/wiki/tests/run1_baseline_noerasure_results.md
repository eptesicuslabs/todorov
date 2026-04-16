# run card: run1_baseline_noerasure (all five bundle features off)

status: historical context only. frozen as of 2026-04-14. do not edit.

## one-line result

god_machine.py with all five bundle features off. h200, 353M params,
fineweb-edu 131M tokens, 4000 steps, 42 min. final val_bpb 1.4499 —
same as god_run_v2 within noise. passkey 0/100 at 256, copy 0/100 at
every length. confirmed that the base matrix memory substrate itself
does not retrieve, independent of the bundle features.

## headline numbers

- preset: `run1_baseline_noerasure`
- params: 353M (note: baseline dense config without multi-compartment
  SwiGLU has a different parameter count from the god_run bundle's 283M)
- features: all off
  - k-wta disabled
  - delta erasure off
  - bcm alpha off
  - multi-compartment feedforward off
  - imagination probe off
  - pc diagnostic head off
- FLA: on (chunk_simple_gla kernel; F1 math fix aligned FLA and recurrent
  paths)
- compute: h200, fineweb-edu byte-level, 131M tokens
- training: 4000 steps, 42 min wall clock (from god_run_v2's 59 min, the
  feature bundle's removal saved ~17 min)
- best val_bpb: 1.4499 (vs god_run_v2's 1.4453; within noise)

## eval

- passkey @ 256: 0/100 (Wilson 95% upper bound 3.7%)
- selective_copy @ every length: 0/100
- val_bpb behaves identically to god_run_v2, indicating the five bundle
  features contributed no compression or generalization value on fineweb-edu

## diagnosis at the time

bundle-broken confirmed at the substrate level: with every feature
stripped, the base matrix memory still produces 0% passkey. the problem
is deeper than the bundle, which motivated the pivot to slot memory as
a substrate change (run2_slot_memory, 2026-04-15).

## diagnosis today (2026-04-16, five paid runs later)

this run also inherited `alpha_log_mean=-0.5` from Config defaults and
was therefore also affected by state evaporation. the pivot to slot
memory likewise inherited the same default until commit `7abb781`. the
fifth paid run fixed retention and still produced 0/100, establishing
that neither substrate change nor retention fix alone resolves retrieval
— the training objective on fineweb-edu does not exercise the memory
substrate at all. see
`wiki/synthesis/training_objective_vs_architectural_goal.md`.

## see also

- `wiki/tests/god_run_v2_results.md` — the prior paid run
- `wiki/tests/run2_slot_memory_first_launch_results.md` — the next paid
  run, first launch with broken retention
- `wiki/synthesis/training_objective_vs_architectural_goal.md` — the
  post-five-runs realisation
- `state/program_status.yaml` → run_history
