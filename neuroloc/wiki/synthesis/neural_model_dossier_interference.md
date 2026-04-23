# neural model dossier: interference

status: current (as of 2026-04-23).

## claim

interference is the central memory failure mode. the neural model must measure target signal against non-target contamination, not just final recall.

## mathematical operation

for a linear state:

```text
q_t^T S_t = target + sum non_target_terms
interference_ratio = ||target|| / (||sum non_target_terms|| + eps)
```

for a slot state, the analogous quantity is target read weight versus total non-target read weight.

## evidence basis

the six paid runs reached zero long-distance retrieval despite strong local distribution fitting. the matrix-memory math predicts mixture under key correlation, while slot memory predicts collapse if address usage is not separated.

## failure mode targeted

memory can appear active while reads are dominated by old content, distractors, correlated keys, or overwritten slots.

## required test material

generate episodes with controllable key correlation, shared-slot pressure, repeated overwrites, and continuing writes after the target event.

## success metrics

- target-to-nontarget ratio
- recall versus correlation slope
- overwrite slope
- continual-write drift
- exact recall after distractors

## controls

- no-memory
- recency-only
- shuffled-address
- oracle target address
- matched write-count budget
- distractor-only episodes

## telemetry

- target read weight
- non-target read mass
- state norm by age
- slot collision count
- retained target contribution over delay

## kill condition

kill the mechanism if final recall degrades at the same slope as the baseline under correlation or if telemetry shows the target contribution is not isolated.

## see also

- [[neural_model_research_test_material_plan]]
