# mistake: run2_slot_memory inherited the broken decay init i had already documented

status: historical context only. frozen as of 2026-04-15. do not edit.

## what happened

2026-04-15. fourth paid h200 run (run2_slot_memory) produced 0% passkey at every
tested length. the slot-memory substrate was supposed to fix retrieval after the
first three runs all failed with the matrix memory.

the root cause: the run2_slot_memory preset definition inherited the default
Config value `alpha_log_mean = -0.5`. this gives `alpha_eff = sigmoid(-0.5) = 0.377`
per step. over a 256-token context this compounds to `0.377^256 ≈ 10^-109`, well
below float32 machine epsilon. the state evaporates long before the passkey query
arrives.

## why this is inexcusable

i wrote the exact diagnosis of this bug three days before the run. `wiki/synthesis/linear_attention_retrieval_wall.md`
evidence line 4, committed 2026-04-14, contains:

> "the `god_machine.py` initialization sets `alpha_log_mean = -0.5`. the effective
> retention factor is: alpha_eff = sigmoid(-0.5) = 0.3775. compounding over 256
> tokens: 0.377^256 ≈ 10^-109. this is below float32 machine epsilon. the state
> of a matrix memory layer with standard retention init is literally rounded to
> zero by the time a 256-token passkey needs to be read."

i then built the slot memory substrate, wrote the preset, and let it default to
`alpha_log_mean = -0.5` without overriding. the replacement substrate carried
forward the exact bug i had just proven would prevent retrieval. the run cost pod
time and produced a 0% result that was architecturally predetermined by the init
choice.

## why the prosecutor missed it

the prosecutor passes reviewed the SlotMemory class math, the write policy, the
smoke-test coverage, and the preset field set. the prosecutor did not review the
INIT VALUE for the decay gate because it was not surfaced as a decision — it was
inherited silently from Config defaults. the preset's listed overrides did not
mention `alpha_log_mean` at all, so the reviewer had nothing to flag.

this is a prosecutor protocol gap. "inherited defaults that materially affect the
experiment" need to be part of the review surface. the correct prosecutor question
would have been: "does this preset set `alpha_log_mean` to a value that avoids
the state-evaporation bug you documented in linear_attention_retrieval_wall.md?"

## what i should have done

when writing the preset, the exact sequence should have been:
1. consult linear_attention_retrieval_wall.md before setting any config value
   that affects retention, since that article catalogues the known failure modes
2. explicitly set `alpha_log_mean` in the preset overrides to a value known to
   survive the 256-token retrieval horizon (the decay sweep at
   wiki/tests/decay_sweep_results.md already showed that 32-pattern recall only
   reopens around `decay=0.90`, implying `alpha_log_mean >= 2.2` or similar for
   matrix memory, and higher for slot memory under the soft-allocation scheme)
3. flag `alpha_log_mean` and any other load-bearing init in the preset definition
   so the prosecutor review surface includes it

the fix committed as `7abb781` sets `alpha_log_mean=5.0` in the slot preset so
`alpha_eff = sigmoid(5.0) = 0.9933`, giving `0.9933^256 ≈ 0.18`. passkey content
retains ~18% of its original magnitude at distance 256 — far above float epsilon.

## what the run 2 eval artifact actually says about state structure

the `delta_state_structure_probe` entry in
`neuroloc/output/run2_slot_memory/run2_slot_memory_eval_suite.json` is:

```json
"delta_state_structure_probe": {
  "error": "no DELTA layers had populated state",
  "per_layer": {}
}
```

`run_delta_state_structure_probe` in `god_machine.py` (at the `for layer_idx,
(block, state) in enumerate(zip(model.blocks, states)): if block.layer_type
!= "DELTA" or state is None: continue` loop around line 2573) filters every
per-block iteration by layer type and populated state; if no iteration survives
the filter, the function returns `{"error": "no DELTA layers had populated
state", "per_layer": per_layer}`. the `run2_slot_memory` preset replaces every
DELTA block with SLOT, so every block fails the first branch of the filter and
`per_layer` stays empty. the `mean_structure_ratio` key is never added to the
result dict. the `0.000` that appears in the step log is the
`.get('mean_structure_ratio', 0)` default used by the logger when the key is
absent, not a measurement.

the probe was inapplicable to a SLOT-only model. run 2 provides no structural
state evidence either supporting or contradicting the evaporation diagnosis.
the evaporation diagnosis rests on the init math alone, not on probe data.

## cost of this mistake

one paid h200 run wasted. the pod time was ~72 minutes of training + ~20 minutes
of eval, plus setup. financial cost was small in absolute terms but meaningless
because the result was architecturally predetermined. the deeper cost was:
- burned user trust
- delayed the real investigation of whether slot memory architecture itself can
  retrieve (we still do not know: the failed run conflated substrate change with
  the retention bug, so the substrate is still untested)
- added another "all zero" data point to the pattern without advancing the diagnosis

## rule (self-imposed)

before any preset is smoked, benchmarked, or launched:

1. read linear_attention_retrieval_wall.md and check each of the five evidence
   lines against the preset's init. every evidence-line failure mode that can
   be controlled by init must be explicitly addressed in the preset definition.

2. prosecutor prompts for new presets must include: "what init values does this
   preset rely on by default? are any of them documented as broken in the
   mistakes/ or synthesis/ wiki articles?"

3. pod launches must halt if the preset does not set `alpha_log_mean` explicitly
   and the substrate needs retention. a missing override is a bug, not a default.

## audit: the bug class spans every paid run, not just run 2

after committing `7abb781`, a grep of every preset override dict in
`_resolve_preset` showed the inheritance bug was wider than the single run-2
site. the full picture:

- `god` (default Config, empty overrides): `alpha_log_mean=-0.5` inherited.
  used by god_run and god_run_v2. state evaporation active.
- `run1_baseline_noerasure`: `alpha_log_mean=-0.5` inherited. used by the
  first of the four paid failures. state evaporation active.
- `run4_erasure_ablation`: `alpha_log_mean=-0.5` inherited. never launched,
  but was queued. state evaporation would have been active on launch.
- `run2_slot_memory` (before `7abb781`): `alpha_log_mean=-0.5` inherited.
  used by run 2. state evaporation active.
- `run1a_retention_ablation`: `alpha_log_mean=2.2` explicitly set. the only
  preset that did not inherit the bug. never launched.

this means the "four stacked failure modes" theory from the eight-cause
external review — k-wta on keys, delta erasure ghosting, imagination gradient
bypass, BCM slow EMA — is suspect. every one of the four paid runs the audit covered (god_run, god_run_v2, run1_baseline_noerasure, run2_slot_memory) had state
evaporation active underneath whatever other feature was being studied. any
one of the eight flagged causes could have been misdiagnosed as causing zero
retrieval when the real cause (or a co-cause) was the retention init.

## structural guard added

commit following `7abb781` adds `_assert_preset_retention_safe` in
`god_machine.py`. the guard runs at every `_resolve_preset` call and raises
if a preset uses DELTA or SLOT layers and does not explicitly override
`alpha_log_mean`. this makes the inheritance-of-broken-default bug class
impossible to reintroduce by silence. process rules alone could not prevent
the next instance; a code-level assertion can.

the audit prompted adding `alpha_log_mean=2.2` to `run1_baseline_noerasure`
and `run4_erasure_ablation` as well. the `god` preset remains broken by
design — it is legacy and should not be used; the guard will fire on any
future attempt to invoke it, which is the correct behavior.

## current state

- fix `7abb781` on origin/master: slot preset now forces `alpha_log_mean=5.0`.
  the dead `alpha_log_std=0.1` override was removed after prosecutor C1.
- follow-up fix on origin/master: `run1_baseline_noerasure` and
  `run4_erasure_ablation` each explicitly set `alpha_log_mean=2.2`.
  `_assert_preset_retention_safe` added as structural guard. see prosecutor
  findings C1, C2, I1, I2, I3.
- substrate question (does slot memory retrieve with proper retention?) was
  TESTED on 2026-04-15 by `run2_slot_memory_retention_fixed` and the answer
  is "not under SGD on natural text": the run trained cleanly to val_bpb
  1.4777 with `alpha_log_mean=5.0` (alpha_eff=0.9933, retention 0.18 at 256)
  and FLA active, but partial eval reported passkey 0/100 at 256 and 1024
  before the user halted the pod. there are now FIVE "all zero" paid runs
  in the record. four of those five had state evaporation active
  (alpha_eff=0.377); the fifth (`run2_slot_memory_retention_fixed`) had
  retention fixed and still produced 0/100. the consistent 0% under all
  combinations of architectural variables tested moves the diagnosis off
  the architecture and onto the training objective. analysis at
  `wiki/synthesis/training_objective_vs_architectural_goal.md`. the next
  paid run must train on a corpus that exercises memory routing; another
  LM run on fineweb-edu is predicted to also produce 0/100 and is not
  authorised. run card at
  `wiki/tests/run2_slot_memory_retention_fixed_results.md`.
- additional mistake from the second-launch attempt of
  `run2_slot_memory_retention_fixed`: `flash-linear-attention` was not
  pinned in `requirements.txt` and the pod did not have it, so SlotMemory
  silently fell through to the python recurrent loop at 655 tok/s instead
  of the fused triton kernel's 33,000 tok/s. fix in commit `edcfe5d`
  pinned `flash-linear-attention>=0.4.0` and `datasets>=2.19.0`. mistake
  doc at `wiki/mistakes/run2_slot_memory_fla_silent_fall_through.md`.
  outstanding structural improvement: `god_machine.py` startup should
  warn or fail when any layer's `use_fla=False` after a preset asks for
  FLA; this guard is not yet committed.

## see also

- `wiki/tests/run2_slot_memory_first_launch_results.md` — the run card for the paid run this mistake happened on
- `wiki/tests/run2_slot_memory_retention_fixed_results.md` — the relaunch with the fix applied
- `wiki/tests/god_run_results.md` — the audit found this bug was also present in this earlier paid run
- `wiki/tests/god_run_v2_results.md` — same
- `wiki/tests/run1_baseline_noerasure_results.md` — same
- `wiki/mistakes/run2_slot_memory_fla_silent_fall_through.md` — the other mistake doc on the same run
- `wiki/synthesis/linear_attention_retrieval_wall.md` — where the bug was documented four days before it happened
- `wiki/synthesis/training_objective_vs_architectural_goal.md` — the post-five-runs realisation that the retention fix alone does not recover retrieval
- `wiki/synthesis/slot_memory_design.md` — the substrate the buggy preset was supposed to exercise
