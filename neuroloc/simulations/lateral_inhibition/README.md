# lateral inhibition simulations

## wta_dynamics.py

a matched-sparsity pilot comparing todorov-style adaptive thresholding against explicit k-wta selection, with one Brian2 anchor run to preserve the biological competition picture.

### what the pilot tests

for each trial, a subset of neurons is designated as the true support. those neurons receive a positive boost, one of them receives an additional leader bonus, and gaussian noise is added to all neurons.

two selection rules then operate on the exact same activation vector:

- adaptive threshold: `|x_i| > alpha * mean(|x|)` with alpha calibrated to match a target active fraction at a reference noise level
- k-wta: keep the top-k activations by absolute value, where `k = round(target_fraction * d)`

the comparison is therefore not "which method is sparser" but "which method recovers the correct support set better under the same sparsity budget."

### outputs

- `wta_dynamics.png` -- main quantitative figure
- `wta_dynamics_anchor.png` -- Brian2 anchor dynamics and late firing rates on one example input
- `wta_dynamics_metrics.json` -- machine-readable trial records, summary statistics, calibration values, and artifact metadata

### quantitative experiments

1. exact support recovery vs noise at 10% true support
2. support f1 vs true support fraction at noise sigma 0.6
3. realized active fraction vs target support fraction at noise sigma 0.6
4. scaling sweep across network sizes 50, 100, 200, 400 at fixed noise sigma 0.6

the Brian2 anchor is qualitative. it shows what recurrent inhibition looks like dynamically, but the main bridge verdict comes from the matched-sparsity threshold vs k-wta measurements.

### how to run

```bash
pip install brian2 numpy scipy matplotlib
cd <project_root>
python neuroloc/simulations/lateral_inhibition/wta_dynamics.py
```

### key metrics

- leader selected
- support precision
- support recall
- support f1
- support jaccard
- exact support recovery
- realized active fraction

### interpretation

if k-wta outperforms adaptive threshold at the same target active fraction, that is direct evidence that relative ranking and explicit competition matter, not just a population-relative threshold.

if adaptive threshold drifts away from the target active fraction as noise rises, that is evidence that mean-based gating is not as stable a sparsity controller as explicit top-k selection.

### limitations

- the main quantitative pilot uses synthetic activation vectors rather than full end-to-end language representations
- the Brian2 anchor remains a minimal one-pool inhibitory circuit, not a realistic interneuron microcircuit
- the threshold alpha is calibrated per support fraction and per scaling condition, which is fair for comparison but not identical to a single fixed learned alpha in a trained model
- this pilot isolates selection quality and sparsity control; it does not answer training-time differentiability or downstream language-model utility by itself

### connection to todorov

this script directly tests the proposal in [[lateral_inhibition_to_adaptive_threshold]]: replace a global mean-relative threshold with explicit k-wta competition and check whether support recovery improves under matched sparsity. it is a bridge-validation pilot, not a claim that todorov already implements biological divisive normalization or full cortical lateral inhibition.
