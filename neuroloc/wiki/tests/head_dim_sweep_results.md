# head_dim_sweep_results

last updated: 2026-04-12

## what this is

this note records the head-dimension sweep for the asymmetric matrix-memory simulation in `neuroloc/simulations/memory/asymmetric_outer_product_recall.py`. the question was narrow: if the matrix state is made wider, does the largest pattern count with mean raw cosine above 0.5 grow linearly with head dimension, or does it flatten early?

the sweep used the following pattern ladders:

- head dim 32 with pattern counts {4, 8, 16, 24, 32}
- head dim 64 with pattern counts {8, 16, 32, 48, 64}
- head dim 128 with pattern counts {16, 32, 64, 96, 128}
- head dim 192 with pattern counts {24, 48, 96, 144, 192}
- head dim 256 with pattern counts {32, 64, 128, 192, 256}

every run used 32 trials per cell, query noise in {0.0, 0.1}, decay in {0.4, 0.8}, and both erasure settings. all five runs completed locally on windows.

## the headline curve

define $p^*(d)$ as the largest tested pattern count whose best cell reaches mean raw cosine above 0.5.

- $p^*(32)=16$
- $p^*(64)=16$
- $p^*(128)=16$
- $p^*(192)=24$
- $p^*(256)=32$

the curve is sub-linear and nearly flat through 128 dimensions. if capacity scaled linearly from the 32-dimensional point, the 256-dimensional run would have reached $p^*(256)=128$. it reached 32 instead. the normalized ratio $p^*(d) / d$ falls from 0.50 at 32 dimensions to 0.125 at 128, 192, and 256 dimensions.

the generated summary and plot live at:

- `neuroloc/output/head_dim_sweep/head_dim_sweep_summary.json`
- `neuroloc/output/head_dim_sweep/p_star_curve.png`

## threshold-crossing cells

the table below lists the threshold-crossing cell for each head dimension.

| head dim | $p^*(d)$ | mean raw cosine at $p^*(d)$ | winning cell |
|---|---:|---:|---|
| 32 | 16 | 0.5631 | `identity_all`, noise 0.0, decay 0.8, erasure on |
| 64 | 16 | 0.6230 | `identity_all`, noise 0.0, decay 0.8, erasure off |
| 128 | 16 | 0.7611 | `bounded_all`, noise 0.0, decay 0.8, erasure off |
| 192 | 24 | 0.5824 | `bounded_all`, noise 0.0, decay 0.8, erasure off |
| 256 | 32 | 0.5279 | `identity_all`, noise 0.0, decay 0.8, erasure off |

the best cell at the next tested rung above $p^*(d)$ is already below threshold in every case:

| head dim | next tested rung | best mean raw cosine at next rung |
|---|---:|---:|
| 32 | 24 | 0.3930 |
| 64 | 32 | 0.4509 |
| 128 | 32 | 0.4340 |
| 192 | 48 | 0.2858 |
| 256 | 64 | 0.2314 |

that is not the profile of a clean rank-limited regime. it is a narrow threshold band where small gains in width buy only one extra rung before the curve falls back under 0.5.

## the retention split is the real story

every threshold-crossing cell above comes from the slower-retention setting `decay=0.8`. under the harsher `decay=0.4` setting, the sweep nearly collapses:

| head dim | $p^*(d)$ at decay 0.4 | best cell at the lowest threshold-near rung |
|---|---:|---|
| 32 | 8 | `bounded_all`, noise 0.0, erasure on, mean 0.5434 at 8 patterns |
| 64 | none | `bounded_all`, noise 0.1, erasure off, mean 0.4889 at 8 patterns |
| 128 | none | `bounded_all`, noise 0.0, erasure off, mean 0.2918 at 16 patterns |
| 192 | none | `identity_all`, noise 0.1, erasure off, mean 0.1992 at 24 patterns |
| 256 | none | `identity_all`, noise 0.1, erasure on, mean 0.1474 at 32 patterns |

this means the head-dimension story is confounded by state retention. widening the state helps only when the write history is already preserved long enough to be retrievable. at the faster-forgetting setting, larger head dimensions do not rescue recall.

## what this says about the memory mechanism

three conclusions are justified by the sweep:

1. enlarging head dimension helps, but with sharply diminishing returns. the improvement from 32 to 64 is zero on the chosen threshold, from 64 to 128 is also zero, from 128 to 192 buys one rung, and from 192 to 256 buys one more rung.
2. the operative bottleneck is not pure matrix width. if width were the dominant limiter, the threshold curve would move roughly in proportion to head dimension. it does not. the decay setting dominates whether any threshold crossing occurs at all.
3. the best cells still come from dense or softly bounded continuous encodings, not from hard three-level or hard sign-like encodings. widening the state does not reverse the round-b conclusion that the asymmetric memory prefers continuous structure over aggressive discretization at the tested loads.

## did the local crash happen at higher loads?

no. the earlier local windows failure did not reproduce in this sweep. the 256-dimensional run completed with 32 trials at pattern count 256, and all five requested dimensions finished normally.

that matters because it removes the earlier uncertainty around whether the capacity curve stopped early only because the local environment could not execute the high-load cells. in this sweep, the flattening is a property of the metrics, not an artifact of execution stopping.

## planning implication

this note does not override the canonical run ordering in `neuroloc/wiki/PROJECT_PLAN.md`. the current paid-compute branch is baseline isolation, not a width-plus-retention launch.

the local planning implication of this sweep is narrower: do not spend a paid run on a head-dimension-only change at the current faster-forgetting operating point. the sweep supports "wider helps a bit" and rejects "wider alone solves recall." if width is revisited later, it should be tested only as part of a joint retention experiment rather than treated as proof that width alone fixes retrieval.

## limitations

1. this is still a synthetic-gaussian simulation, not a trained-activation replay.
2. the threshold definition is deliberately blunt: mean raw cosine above 0.5. another threshold would move the exact numbers but not the flattening pattern.
3. the winning cell can change across dimensions between `identity_all` and `bounded_all`, so the write-up should be read as a capacity result first and an encoding result second.
4. the sweep used only two decay settings. the next round should sample the retention axis more finely if this mechanism remains in scope for paid training.

## sources

- `neuroloc/simulations/memory/asymmetric_outer_product_recall.py`
- `neuroloc/output/head_dim_sweep/d32/asymmetric_outer_product_recall/asymmetric_outer_product_recall_metrics.json`
- `neuroloc/output/head_dim_sweep/d64/asymmetric_outer_product_recall/asymmetric_outer_product_recall_metrics.json`
- `neuroloc/output/head_dim_sweep/d128/asymmetric_outer_product_recall/asymmetric_outer_product_recall_metrics.json`
- `neuroloc/output/head_dim_sweep/d192/asymmetric_outer_product_recall/asymmetric_outer_product_recall_metrics.json`
- `neuroloc/output/head_dim_sweep/d256/asymmetric_outer_product_recall/asymmetric_outer_product_recall_metrics.json`
- `neuroloc/output/head_dim_sweep/head_dim_sweep_summary.json`
- `neuroloc/output/head_dim_sweep/p_star_curve.png`

## related docs

- `neuroloc/wiki/PROJECT_PLAN.md`

## update history

- **2026-04-12** — deyan todorov — file created. records the five-point head-dimension sweep for the asymmetric matrix-memory simulation, shows the sub-linear $p^*(d)$ curve `{16, 16, 16, 24, 32}`, isolates the retention interaction as the real bottleneck, and recommends against a head-dimension-only paid run at the faster-forgetting operating point.