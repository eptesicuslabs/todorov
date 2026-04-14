# decay sweep results

last updated: 2026-04-12

## what this is

this note records the fine-grained retention sweep for the asymmetric matrix-memory simulation in `neuroloc/simulations/memory/asymmetric_outer_product_recall.py`. the head-dimension sweep had already shown that width alone is not the dominant bottleneck at `d_head=64`; this follow-up isolates the decay axis and asks a narrower question: at what retention setting does the matrix memory first reopen useful recall at 32 and 64 stored patterns?

## run configuration

the sweep used the same asymmetric simulation as round B and the head-dimension note, with head dimension fixed at 64 and a finer decay ladder:

- pattern counts `{8, 16, 32, 48, 64}`
- query noise `{0.0, 0.1}`
- decay `{0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98}`
- erasure both off and on
- 32 trials per cell

the generated artifacts live at:

- `neuroloc/output/decay_sweep/decay_sweep_summary.json`
- `neuroloc/output/decay_sweep/exact_query_summary.json`
- `neuroloc/output/decay_sweep/p_star_vs_decay.png`

## the headline curves

define $p^*(a)$ as the largest tested pattern count whose best cell reaches mean raw cosine above 0.5 at decay $a$.

the sweep gives two closely related curves:

| decay | best-cell $p^*(a)$ | exact-query-only $p^*(a)$ |
|---|---:|---:|
| 0.40 | none | none |
| 0.50 | 8 | 8 |
| 0.60 | 8 | 8 |
| 0.70 | 16 | 8 |
| 0.80 | 16 | 16 |
| 0.85 | 16 | 16 |
| 0.90 | 32 | 32 |
| 0.95 | 64 | 64 |
| 0.98 | 64 | 64 |

the exact-query curve is the cleaner planning signal because it removes the noisy-query cells. on that curve, the first useful reopening happens at `decay=0.90`, where 32-pattern exact recall finally clears threshold, and the first 64-pattern reopening happens at `decay=0.95`.

## the knee cells

three cells summarize the shape of the curve:

1. at `decay=0.40`, even the best exact-query 8-pattern cell stays below threshold: `identity_all`, erasure on, mean raw cosine `0.4791`.
2. at `decay=0.90`, the first exact-query 32-pattern crossing appears: `bounded_all`, erasure off, mean raw cosine `0.5384`.
3. at `decay=0.95`, exact-query 64-pattern recall first crosses threshold: `bounded_all`, erasure off, mean raw cosine `0.5180`.

the transition is not gradual. the 32-pattern exact-query best cell is still only `0.3883` at `decay=0.85`, then jumps above threshold at `0.90`. the 64-pattern exact-query best cell is still only `0.3748` at `0.90`, then crosses at `0.95`.

## what this says about the mechanism

three conclusions are justified by the sweep:

1. retention matters sharply and nonlinearly. the relevant operating-point change is not "slightly slower forgetting" but a threshold-like reopening between `0.85` and `0.90` for 32 patterns, then between `0.90` and `0.95` for 64 patterns.
2. the earlier head-dimension result was not just a width story. at `d_head=64`, the memory can support materially higher load only once retention is high enough to preserve the write history.
3. above 32 patterns, the winning exact-query cells come from dense or softly bounded continuous encodings with erasure off. the finer decay sweep reinforces the earlier conclusion that aggressive discretization is not rescuing the asymmetric memory at the tested loads.

## planning implication

this note does not justify silently changing two baseline variables at once. the overwrite result is strong enough to remove erasure from the scheduled dense baseline, but the retention result should remain a separate follow-up ablation.

that distinction matters because the current dense baseline code initializes `alpha_log_mean = -0.5`, which implies an initial recurrent retention coefficient of approximately $\exp(\log \sigma(-0.5)) = \sigma(-0.5) \approx 0.38$. that is well below the first useful exact-query reopening point at `0.90`. if the no-erasure baseline still fails the passkey gate, the next cheapest discriminant is therefore a slower static-retention initialization, not an immediate claim that the base mechanism is conclusively broken.

## limitations

1. this is still a synthetic-gaussian simulation, not a replay of trained activations.
2. the threshold is intentionally blunt: mean raw cosine above 0.5.
3. the sweep uses 32 trials per cell, which is enough for a stable knee estimate but not enough to settle fine-grained encoding ties.
4. the sweep fixes head dimension at 64. it answers the retention question only at the current architecture width.

## sources

- `neuroloc/simulations/memory/asymmetric_outer_product_recall.py`
- `neuroloc/output/decay_sweep/asymmetric_outer_product_recall/asymmetric_outer_product_recall_metrics.json`
- `neuroloc/output/decay_sweep/decay_sweep_summary.json`
- `neuroloc/output/decay_sweep/exact_query_summary.json`
- `neuroloc/output/decay_sweep/p_star_vs_decay.png`
- `neuroloc/wiki/PROJECT_PLAN.md`

## related docs

- `neuroloc/wiki/tests/encoding_simulation_round_b.md`
- `neuroloc/wiki/tests/head_dim_sweep_results.md`

## update history

- **2026-04-12** — deyan todorov — file created. records the nine-point decay sweep at `d_head=64`, shows the exact-query reopening points at `decay=0.90` for 32 patterns and `decay=0.95` for 64 patterns, and reframes retention as the first post-baseline ablation rather than a variable to bundle into the no-erasure baseline.