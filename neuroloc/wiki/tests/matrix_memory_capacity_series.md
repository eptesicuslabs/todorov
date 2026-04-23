# matrix-memory capacity series

status: historical context only. frozen as of 2026-04-23. do not edit.

this page groups the early historical matrix-memory evidence that established
the capacity and retention wall before the slot-memory pivot. the underlying
records and this grouping page are both frozen as historical context.

## what this series established

the ordered 2026-04-12 series did not produce a single "best encoding"
answer. it converged on a stronger conclusion: the outer-product decayed
matrix state was bottlenecked by retention and fixed-state capacity before
later naming, loss-shaping, or substrate changes were attempted at scale.

the sequence matters:
- round a tested input encodings on a symmetric baseline
- round b switched to the actual asymmetric matrix-memory operation
- the head-dimension sweep showed widening alone did not reopen capacity
- the decay sweep showed retention, not width, was the first real lever
- the overwrite sweep showed naive erasure hurt at the first useful
  retention knee

## ordered record

- [[tests/encoding_simulation_round_a]] -- symmetric baseline: sign-only
  vs three-level magnitude quantization vs bounded continuous encoding
- [[tests/encoding_simulation_round_b]] -- asymmetric matrix-memory
  baseline matching the project's write rule more closely
- [[tests/head_dim_sweep_results]] -- width sweep showing sub-linear
  reopening of the useful pattern count
- [[tests/decay_sweep_results]] -- retention sweep showing the first
  exact-query reopening at high decay
- [[tests/overwrite_sweep_results]] -- overwrite comparison showing that
  erasure hurts at the first useful retention knee

## downstream relevance

this series feeds directly into the later historical analyses that diagnosed
the matrix-memory ceiling and justified the substrate pivot:
- [[tests/correction_field_trained_prediction_results]]
- [[synthesis/linear_attention_retrieval_wall]]
- [[synthesis/training_objective_vs_architectural_goal]]
- [[synthesis/substrate_requires_architectural_change]]

## see also

- [[tests/index|tests]]
- [[PROJECT_PLAN]]
- [[synthesis/phase1_evaluation_surface_for_neural_models]]
