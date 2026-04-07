# 2026-04-07 k-wta vs threshold pilot

date run: 2026-04-07

status: completed

test type: bridge-validation pilot

script:
- `neuroloc/simulations/lateral_inhibition/wta_dynamics.py`

artifacts:
- `neuroloc/simulations/lateral_inhibition/wta_dynamics.png`
- `neuroloc/simulations/lateral_inhibition/wta_dynamics_anchor.png`
- `neuroloc/simulations/lateral_inhibition/wta_dynamics_metrics.json`

evolution:
- this is the first dated test record for the lateral-inhibition bridge question
- it replaces the earlier plot-only wta script and formalizes the proposal argued in [[lateral_inhibition_to_adaptive_threshold]]
- it also follows the shared artifact pattern established in [[tests/2026-04-07_pattern_completion_baseline|2026-04-07 pattern completion baseline]]

## what was done

- replaced the old inhibition-strength visualization with a matched-sparsity comparison between adaptive threshold and explicit k-wta selection
- calibrated threshold alpha separately for each target support fraction so the comparison was not confounded by different nominal sparsity budgets
- ran support-recovery sweeps across noise and target support fraction
- ran a scaling sweep across network sizes 50, 100, 200, 400
- kept a Brian2 anchor run to preserve the biological competition picture while moving the main verdict to quantitative support-recovery metrics

## configuration

- neurons: 100 for the main sweep
- support fractions: 0.05, 0.10, 0.20, 0.41
- noise levels: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
- selection trials per condition: 40
- support boost: 1.0
- leader bonus: 0.35
- calibration noise level: 0.4
- calibration trials: 256
- scaling network sizes: 50, 100, 200, 400
- scaling trials per point: 20
- numpy seed: 42

## key results

- at 10% true support and noise sigma 0.2, k-wta exact support recovery was 0.925 with 95% ci [0.840, 1.010]
- at the same point, adaptive threshold exact support recovery was 0.500 with 95% ci [0.338, 0.662]
- paired exact-support delta at that point was 0.425 with permutation p = 0.001
- at 10% true support and noise sigma 0.6, the support-f1 difference was small: k-wta 0.443 vs threshold 0.438
- at 10% true support and noise sigma 0.6, k-wta held the active fraction exactly at 0.100 while threshold drifted down to 0.088; paired delta 0.012 with permutation p = 0.002

## verdict

k-wta provides the clearest benefit on two axes this pilot can actually test: exact support recovery in the moderate-noise regime and exact sparsity control in the harder regime.

the hard-noise support-f1 gap is small, so this run does not justify claiming that k-wta dominates thresholding on every retrieval-quality metric. the defensible claim is narrower: explicit top-k competition improves exact-set recovery when the task is still recoverable and removes threshold drift in realized firing rate.

## limitations

- the main comparison uses synthetic activation vectors, not trained language-model activations
- the Brian2 anchor remains qualitative and does not itself produce the statistical verdict
- alpha is recalibrated per support fraction and per scaling condition, which is appropriate for a fair matched-sparsity comparison but not identical to a single fixed learned alpha in a model checkpoint
- this test says nothing yet about training stability or downstream bpb