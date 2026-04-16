# run card: god_run_v2 (17 F + 14 G prosecutor fixes)

status: historical context only. frozen as of 2026-04-12. do not edit.

## one-line result

re-run of the five-feature bundle with 31 prosecutor findings fixed,
critical among them the F1 `torch.exp` math correction. trained to
val_bpb 1.4453 (+0.050 vs v1) in 59 minutes on h200, then produced 0/100
passkey at 256/1024/4096 and 0/100 selective_copy at every length. the
F1 math fix did not recover passkey. bundle-is-broken confirmed.

## headline numbers

- preset: `god` (same config as v1 with 31 fixes applied)
- params: 283M
- critical F1 fix: replaced `sigmoid(logsigmoid(pre))` with
  `exp(logsigmoid(pre))` in the recurrent path so alpha_eff matches the
  FLA `exp(g)` gate convention
- other fixes: F2 gated running_state_norm EMA on self.training; F3/F5
  merged spike_stats and val_result into jsonl via a loop (fixing the
  telemetry regression); F10 low-rank factorised imag_filter/pc_head;
  F11 uint8 ByteDataset; 12 more covering smoke tests, resume
  correctness, load errors, dead code
- compute: h200, fineweb-edu byte-level, 131M tokens
- training: 4000 steps, 59 min wall clock
- best val_bpb: 1.4453 (0.404x vs transformer baseline of 3.580)
- probe metric stream now records 74 keys per step (up from the 5 that
  survived v1's bug)

## eval

- passkey @ 256/1024/4096: 0/100 at every length (Wilson 95% upper bound 3.7%)
- selective_copy @ 256/512/1024/2048: 0/100 at every length
- perplexity at length: 1.9254 → 1.8941 → 1.5373 → 1.4776 → 1.4192
- delta_state_structure_probe: structure_ratio 0.977, pairwise_cos 0.003,
  random_pairwise_cos 0.000 (statistically indistinguishable from random
  state of equal frobenius norm)

## diagnosis at the time

the F1 math fix did NOT recover passkey — the alpha_eff divergence between
the recurrent and FLA paths was not the cause of retrieval failure.
external review identified eight candidate contributing mechanisms:
k-WTA on keys destroying address space, erasure with sparse keys leaving
ghost content, imagination probe gradient bypass, slow BCM EMA,
multi-compartment effective-width loss, PC head loss drag, FLA/recurrent
numerical drift, q/k normalisation timing mismatch. theory was provisional
pending controls.

## diagnosis today (2026-04-16, five paid runs later)

all four theorised contributing mechanisms share a single confound: the
inherited `alpha_log_mean=-0.5` retention default caused state evaporation
below float32 epsilon over 256 tokens. the fifth paid run
(`run2_slot_memory_retention_fixed`) fixed retention to 5.0 and still
produced 0/100, which combined with the other four runs rules out the
architectural variables as sole cause and points the diagnosis at the
training objective. see `wiki/synthesis/training_objective_vs_architectural_goal.md`.

## see also

- `wiki/tests/god_run_results.md` — the first-run card
- `wiki/synthesis/training_objective_vs_architectural_goal.md` — the
  post-five-runs realisation
- `wiki/mistakes/run2_slot_memory_decay_copy_paste.md` — the audit
- `state/program_status.yaml` → `run_history.god_run_v2`
- `CLAUDE.md` → results summary
