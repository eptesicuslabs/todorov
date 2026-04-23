# benchmark skill taxonomies

status: current (as of 2026-04-23).

working shelf for benchmark taxonomies that decompose evaluation into skill families instead of one scalar score. this page exists to support future phase-1 and phase-2 evaluation design.

## useful benchmark lessons

- perception test is useful because it separates different reasoning demands over the same audiovisual material
- worldsense is useful because it avoids answer-prior shortcuts and keeps hidden state explicit
- popgym-style work is useful because it makes memory and partial observability visible in small tasks

## current project-facing taxonomy

the working taxonomy for neuroloc now uses:

- memory
- abstraction
- physics or dynamics
- semantics or task structure

crossed against:

- descriptive recognition
- explanatory linkage
- predictive use
- counterfactual or planning use

the canonical first visual for this shelf is `e1_skill_by_reasoning_matrix`.

![e1 skill by reasoning matrix](../assets/figures/e1_skill_by_reasoning_matrix.svg)

## why this shelf exists

it keeps evaluation design tied to explicit skill decomposition rather than to one benchmark brand name.

## see also

- [[phase1_evaluation_surface_for_neural_models]]
- [[synthetic_shared_world_bridge]]
- [[visuals_to_phase1_nm_tests]]
- [[visual_sources_cognitive_architecture]]
