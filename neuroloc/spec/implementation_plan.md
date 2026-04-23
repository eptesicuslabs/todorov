# neural machine implementation plan

status: historical pre-pivot implementation sketch. do not read this file as
the active execution plan.

## what this file is now

this document used to describe a live implementation path for a fineweb-backed,
next-token-oriented neural-machine run. that is no longer the active project
method.

the current program state is:

- curriculum is the only active lane
- paid compute is paused indefinitely
- phase 1 for the neural model is now state-and-action-first
- the backlog restart point is the implemented `biology_phase1` battery, then
  latent-world deliberation, then model-side neural-model evaluation

## use instead

for current guidance, read:

1. `neuroloc/wiki/PROJECT_PLAN.md`
2. `neuroloc/spec/blueprint.md`
3. `neuroloc/HANDOFF.md`
4. `neuroloc/SIMULATION_AGENT_PROMPT.md`

## historical note

the older content of this file described:

- fineweb-edu byte-level training
- next-token prediction as the outer objective
- immediate h200 implementation sequencing

those assumptions are now historical, not binding. if a future implementation
plan is needed, it should be rebuilt around synthetic episode worlds, state
probes, action success, and trainability controls instead of reviving the old
language-model-first surface.
