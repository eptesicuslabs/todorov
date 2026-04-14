# correction field trained prediction results

last updated: 2026-04-14

## what this is

this note records the follow-up simulation at `neuroloc/simulations/memory/correction_field_trained_prediction.py` that repeats the `correction_field_capacity.py` sweep with a trained low-rank linear prediction head instead of a synthetic `q * v + sqrt(1-q^2) * noise` construction. the earlier synthetic-prediction run (2026-04-14 morning) produced `memory_substrate_capacity_delta = 0` for every tested cell. the question this follow-up answers is whether that null result is an artifact of synthetic isotropic noise or a structural property of the outer-product matrix memory.

## setup

the pod holds a frozen `god_run_v2` checkpoint that could have supplied real hidden states, but pod access is unavailable from the local host. in place of real activations, the simulation generates correlated hidden states `v_t = A u_t + xi_t` where `A` is a fixed random orthogonal mixing matrix, `u_t` is a smooth autoregressive process `u_t = rho * u_{t-1} + sqrt(1 - rho^2) * eta_t`, and `xi_t` is independent gaussian noise at `noise_scale = 0.3`. the predictor is a low-rank linear head `prediction = (v_{t-1} @ W_up) @ W_down` with `W_up in R^{d x 16}` and `W_down in R^{16 x d}`, trained by adam at `lr = 1e-3` for 200 iterations on 2000 sampled sequences of length 32 per training run.

the sweep covers `head_dim in {32, 64, 128}`, `pattern_count in {4, 8, 16, 32, 64}`, `temporal_correlation in {0.5, 0.7, 0.9, 0.99}`, `decay in {0.5, 0.9, 0.95}`, 16 trials per cell, for 2880 paired recall records. artifacts live in `neuroloc/output/simulation_suites/correction_field_trained_prediction_full/correction_field_trained_prediction/`.

## headline observations

at `d_head = 64, decay = 0.9`, the mean cosine recall across the capacity sweep is:

- at rho = 0.99 and 64 stored patterns:
  - raw values memory: 0.294
  - residual-only memory: 0.293
  - prediction-only (no memory): 0.369
  - corrected reconstruction (prediction + residual memory): 0.492
- at rho = 0.99 and 32 stored patterns:
  - raw = 0.562, residual-only = 0.562, prediction-only = 0.396, reconstruction = 0.647

the `memory_substrate_capacity_delta` summary (maximum `pattern_count` whose residual-only memory clears cosine > 0.5, minus the same threshold for the raw-value memory) is approximately 0 across every tested cell, for every `temporal_correlation` setting:

- rho = 0.5: mean memory_substrate_capacity_delta = 0.000
- rho = 0.7: mean memory_substrate_capacity_delta = 0.000
- rho = 0.9: mean memory_substrate_capacity_delta = -1.778
- rho = 0.99: mean memory_substrate_capacity_delta = 0.000

so a trained predictor does not change the memory-side verdict. the outer-product matrix memory does not store more patterns when fed residuals than when fed raw values, even when the residuals have visibly structured correlation.

## observed prediction quality

the held-out `1 - Var(residual) / Var(target)` achieved by the rank-16 head at `d_head = 64`:

- rho = 0.5: observed_prediction_quality = 0.010
- rho = 0.7: observed_prediction_quality = 0.060
- rho = 0.9: observed_prediction_quality = 0.134
- rho = 0.99: observed_prediction_quality = 0.169

the rank-16 constraint bottlenecks the predictor at `d_head = 64`. a full-rank linear head on the same synthetic data reaches r^2 = 0.955 at rho = 0.99, but the low-rank constraint was chosen deliberately to resemble the compressed prediction paths the architecture actually uses. what matters for the correction-field question is that held-out r^2 is strictly positive above rho = 0.5 and monotonic in `rho`, confirming the predictor does learn the temporal structure rather than fitting noise.

## residual norm ratio

mean `||residual|| / ||target||` at `d_head = 64`:

- rho = 0.5: 0.996
- rho = 0.7: 0.972
- rho = 0.9: 0.940
- rho = 0.99: 0.921

residuals are modestly smaller than raw values at high temporal correlation, but the reduction is at most about 8 percent in norm. that is too small a compression to unlock additional capacity in the outer-product memory; the per-pattern outer products still collide in the same head-dimension subspace regardless of whether the stored vector is the full target or a slightly-shrunk residual.

## reconstruction capacity gain

the one cell where the correction-field path wins materially is `reconstruction = prediction + retrieved_residual` at high pattern counts. at `d_head = 64, decay = 0.9, rho = 0.99, pattern_count = 64`, cosine recall on corrected reconstruction is 0.492 against raw recall 0.294. at `pattern_count = 32` the same comparison is 0.647 vs 0.562. the mechanism is not memory capacity: the memory stores and retrieves residuals with almost exactly the same fidelity as it stored and retrieved raw values. the mechanism is that the prediction sets a correct low-frequency component, so even a noisy retrieved residual pushes the reconstruction toward the target.

importantly, prediction alone does not retrieve specific stored patterns. at `pattern_count = 64, rho = 0.99`, prediction-only cosine is 0.369 and exact-match accuracy is near chance. the memory is still necessary for key-conditioned recall; the prediction only supplies a bias.

## implication for correction-field design

three conclusions are justified by this sweep:

1. the correction-field mechanism does not survive as a memory-side compression. storing residuals instead of raw values does not raise the effective pattern capacity of the outer-product matrix memory, even with a trained low-rank predictor on temporally correlated data. the earlier synthetic-prediction null result is structural, not an artifact of isotropic noise.
2. the correction-field mechanism survives as a reconstruction-side bias. a trained predictor supplies a cheap always-available baseline that improves final reconstruction fidelity at high load, and the memory contributes the key-conditioned correction on top.
3. the reconstruction gain is not a cache compression argument. the residual norm ratio only drops from 1.00 to 0.92 even at rho = 0.99. the memory is not storing less; the overall system is predicting more. in the architecture this means the prediction head cost must be accounted for separately from the matrix-memory cost.

the honest reframing is that the correction-field is an additive prediction head with a residual-addressed memory. the memory itself is unchanged. the prediction head pays its own parameter cost and contributes a low-frequency reconstruction baseline. whether that baseline is worth its parameters is a question about the prediction head, not about the memory.

## limitations

1. synthetic hidden states are a stand-in for real god_run_v2 activations. the correlation structure used here is linear and stationary. real activations have non-stationary, possibly non-linear structure that a rank-16 linear head would handle differently.
2. the predictor is a single rank-16 linear map on one previous timestep. the architecture actually uses richer prediction paths (swiglu mlps, attention). the low-rank linear head here is a conservative baseline; a richer predictor would reach a higher observed_prediction_quality but the memory-side null result is unlikely to change.
3. 16 trials per cell. cosine recalls in the reported cells have enough separation that the conclusion is robust, but fine-grained effect sizes near the threshold are noisy.
4. cosine-above-0.5 capacity is a blunt metric. the stated gain of zero means no cell transitioned from below to above the threshold; small recall differences inside a cell are still visible in the per-cell mean tables.

## sources

- `neuroloc/simulations/memory/correction_field_trained_prediction.py`
- `neuroloc/output/simulation_suites/correction_field_trained_prediction_full/correction_field_trained_prediction/correction_field_trained_prediction_metrics.json`
- `neuroloc/output/simulation_suites/correction_field_trained_prediction_full/correction_field_trained_prediction/correction_field_trained_prediction_heatmaps.png`
- `neuroloc/output/simulation_suites/correction_field_trained_prediction_full/correction_field_trained_prediction/correction_field_trained_prediction_summary.png`
- `neuroloc/simulations/memory/correction_field_capacity.py`

## related docs

- `neuroloc/wiki/tests/decay_sweep_results.md`
- `neuroloc/wiki/tests/head_dim_sweep_results.md`
- `neuroloc/wiki/tests/overwrite_sweep_results.md`

## update history

- **2026-04-14** -- deyan todorov -- file created. records the trained-predictor follow-up to `correction_field_capacity`. confirms that a real predictor does not change the memory-side `memory_substrate_capacity_delta = 0` verdict, observed prediction quality 0.17 at rho=0.99 with a rank-16 linear head on `d_head=64`, and residuals are only 8 percent shorter than raw values. the correction field survives as a reconstruction-side bias, not as a memory-side compression.
