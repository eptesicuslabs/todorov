# mistake: run2_slot_memory inherited the broken decay init i had already documented

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
`alpha_eff = 0.993`, giving `0.993^256 ≈ 0.16`. passkey content retains 16% of
its original magnitude at distance 256 — far above float epsilon.

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

## current state

- fix `7abb781` on origin/master: slot preset now forces `alpha_log_mean=5.0` and
  `alpha_log_std=0.1`
- pod halted, no paid compute running
- substrate question (does slot memory retrieve with proper retention?) remains
  open
- four "all zero" runs in the record, but three of them had either (a) k-wta on
  keys destroying address space or (b) state evaporation from the retention init.
  the architecture has not yet been tested under both dense keys AND non-evaporating
  retention
