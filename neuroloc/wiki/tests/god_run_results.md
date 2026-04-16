# run card: god_run (first paid neural-machine run)

status: historical context only. frozen as of 2026-04-11. do not edit.

## one-line result

first end-to-end run of the five-feature bundle at 283M parameters on
h200. trained cleanly to val_bpb 1.3950 (0.390x transformer baseline)
in 53 minutes, then produced 0/20 passkey at 256/1024/4096 and 0/20
selective_copy at 256/512/1024/2048. compressed attention + mlp path
fit the next-byte distribution; delta-rule memory was noise.

## headline numbers

- preset: `god` (default config)
- params: 283M
- layer pattern: bundle of all five experimental features
  - 20% rate-coded compression on keys and values
  - delta erasure
  - activity-adaptive decay (bcm alpha gamma=0.3)
  - multi-compartment feedforward block (k=4)
  - compressed attention via sdpa
  - always-on imagination probe
  - predictive-coding diagnostic head
- compute: h200, fineweb-edu byte-level, 131M tokens
- training: 4000 steps, 53 min wall clock
- best val_bpb: 1.3950 (0.390x vs transformer baseline of 3.580)
- firing rate: 0.200 (k-wta target met)
- no dead neurons

## eval

- passkey @ 256/1024/4096: 0/20, 0/20, 0/20
- selective_copy @ 256/512/1024/2048: 0/20 at every length
- perplexity at length: 1.9354 → 1.8437 → 1.4909 → 1.4110 → 1.3751
  (monotonic decrease — attention path uses context)
- delta_state_structure_probe: mean_structure_ratio 0.981, pairwise_cos
  -0.003, random_pairwise_cos 0.000 (state is high-dimensional noise,
  not content-addressable)

## diagnosis at the time

the compressed-attention and mlp paths drove val_bpb to 0.390x the
transformer baseline while the delta-rule memory contributed noise. the
feature bundle as a whole was deemed unvalidated; 17 prosecutor findings
f1-f17 were opened and fixed before the follow-up run. the telemetry
logger also dropped 40+ probe metrics from its stream due to a hardcoded
dict at the step-log site.

## diagnosis today (2026-04-16, five paid runs later)

the real cause of 0% passkey was not the five-feature bundle itself. it
was the inherited `alpha_log_mean=-0.5` retention default that caused
state evaporation over 256 tokens, combined with the training objective
(next-byte cross-entropy on natural text) not exercising the memory
substrate at all. see
`wiki/synthesis/training_objective_vs_architectural_goal.md` for the
post-five-runs analysis that supersedes the per-run diagnoses.

## see also

- `wiki/tests/god_run_findings.md` — the original long-form synthesis of
  this run's findings, written at the time
- `wiki/synthesis/training_objective_vs_architectural_goal.md` — the
  post-five-runs realisation that reframes this run's 0% passkey
- `wiki/mistakes/run2_slot_memory_decay_copy_paste.md` — the audit that
  identified the retention bug was present in every paid run including
  this one
- `state/program_status.yaml` → `run_history.god_run` for the raw entry
- `CLAUDE.md` → results summary for the line-item record
