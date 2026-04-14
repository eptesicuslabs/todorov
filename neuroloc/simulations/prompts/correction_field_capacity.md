# prompt: correction-field capacity simulation

## context

this prompt creates a cpu simulation that tests the core claim of the correction-field memory design described in `neuroloc/wiki/synthesis/correction_field_memory.md`.

the correction-field design changes what gets stored in the project's matrix memory. instead of storing raw key-value outer products, the memory stores prediction residuals: the difference between what a prediction function expects and what actually arrives. the hypothesis is that this increases effective capacity because residuals are smaller than raw values, and reduces write frequency because predictable tokens produce near-zero residuals.

the project's matrix memory operation is:

```
S_t = decay * S_{t-1} + write_gate * k_t @ v_t^T
readout = q^T @ S
```

at d_head=64, simulation round b (2026-04-12) showed that no encoding produces meaningful recall above 8 stored patterns. the capacity ceiling is structural, not encoding-dependent. the correction-field design addresses this by reducing the effective information per stored value, which should increase the number of distinct associations before interference saturates.

## what this simulation tests

the simulation compares two conditions on the SAME matrix memory with the SAME keys:

1. **raw-value storage** (baseline): `v_t = raw_value_t`. this is the current design.
2. **correction-field storage** (test): `v_t = raw_value_t - prediction_t`, where `prediction_t` is a learned or synthetic prediction of raw_value_t.

the key question: at matched d_head, decay, and pattern count, does correction-field storage achieve higher recall fidelity than raw-value storage?

## simulation design

### synthetic prediction model

since we cannot train a real language model in a cpu simulation, we synthesize a prediction function with controllable quality. the prediction quality parameter `q` (0 to 1) controls what fraction of the value's variance the prediction explains.

for each stored value v_i, the synthetic prediction is:

```
prediction_i = q * v_i + sqrt(1 - q^2) * noise_i
```

where noise_i is an independent gaussian vector. this gives:
- q=0: prediction is random noise, residual r_i = v_i - prediction_i has the same norm as v_i (no compression)
- q=0.5: prediction explains 25% of variance, residual norm is ~87% of value norm
- q=0.9: prediction explains 81% of variance, residual norm is ~44% of value norm
- q=0.99: prediction explains 98% of variance, residual norm is ~14% of value norm
- q=1.0: prediction is perfect, residual is zero (infinite compression, nothing stored)

### sweep parameters

- `d_head` in {32, 64, 128}: head dimensions matching the existing simulation suite
- `pattern_count` in {4, 8, 16, 32, 64}: from below capacity to at/above capacity
- `prediction_quality` in {0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99}: the full range from no prediction to near-perfect
- `decay` in {0.5, 0.8, 0.90, 0.95, 1.0}: includes the critical retention knee at 0.90 from the decay sweep
- `trials` = 32 per cell (paired comparison: same random keys/values for both conditions per trial)

### write procedure

for each trial:

1. generate N random key vectors k_1, ..., k_N ~ normal(0, 1/sqrt(d)) and N random value vectors v_1, ..., v_N ~ normal(0, 1/sqrt(d))
2. generate N synthetic predictions: pred_i = q * v_i + sqrt(1 - q^2) * noise_i
3. compute N residuals: r_i = v_i - pred_i
4. **raw-value state**: accumulate S_raw = sum(decay^(N-i) * k_i @ v_i^T)
5. **correction-field state**: accumulate S_corr = sum(decay^(N-i) * k_i @ r_i^T)
6. both states use the SAME keys k_i

### read procedure

for each stored pattern j in {1, ..., N}:

1. query both states with q_j = k_j (exact-query recall, no noise)
2. **raw retrieval**: o_raw = q_j^T @ S_raw. compare to v_j via cosine similarity.
3. **correction retrieval**: o_corr = q_j^T @ S_corr. reconstruction = pred_j + o_corr. compare reconstruction to v_j via cosine similarity.

this is the critical difference: the correction-field retrieval adds the stored correction to the prediction. the prediction already explains q^2 fraction of the variance, so the correction only needs to supply the remaining (1-q^2) fraction.

### metrics

per trial, per condition:
- `mean_cosine_recall`: mean cosine similarity between retrieved/reconstructed value and true value, averaged over all N patterns
- `exact_match_accuracy`: fraction of patterns where the retrieved value's nearest neighbor among the N stored values is the correct one
- `effective_capacity`: largest pattern_count where mean_cosine_recall > 0.5

per cell (across trials):
- mean and 95% ci of each metric
- paired difference between correction-field and raw-value (same keys/values per trial)
- permutation test p-value for the paired difference

### output

- metrics json following the `shared.py` schema (`build_run_record`)
- summary keys: `capacity_raw_vs_corrfield_by_quality`, `cosine_gain_by_quality_and_patterns`, `threshold_shift_by_decay`
- plot: 2d heatmap of (prediction_quality x pattern_count) showing mean cosine recall for both conditions side by side, at each d_head and decay value

## implementation notes

- use `neuroloc/simulations/shared.py` for all infrastructure (rng, json output, validation, plotting style, paired_difference_stats)
- place the file at `neuroloc/simulations/memory/correction_field_capacity.py`
- register in `neuroloc/simulations/suite_registry.py` under a new "compression" group
- use env vars for overridable parameters: `CF_SEED`, `CF_HEAD_DIMS`, `CF_PATTERN_COUNTS`, `CF_PRED_QUALITIES`, `CF_DECAYS`, `CF_TRIALS`
- the paired comparison methodology from `asymmetric_outer_product_recall.py` is mandatory: both conditions see the SAME random keys, values, and noise per trial
- the reconstruction step (pred + correction) is the critical difference from the baseline. do NOT skip it. comparing the raw correction to the raw value would be measuring a different (wrong) thing.

## additional tests in the same file

### surprise-gated write frequency

add a second sweep that measures what fraction of tokens actually write to the state under surprise gating.

generate a sequence of N tokens where fraction `p` are "predictable" (prediction quality q=0.99) and fraction `1-p` are "surprising" (prediction quality q=0.0). sweep `p` in {0.5, 0.7, 0.8, 0.9, 0.95, 0.99}. for each p:

1. compute surprise ratio s_i for each token
2. count how many tokens have s_i > tau for tau in {0.01, 0.05, 0.1, 0.2}
3. measure recall on the surprising tokens only (the ones that actually wrote)

expected result: at p=0.95 (95% predictable), only ~5% of tokens write, and recall on those 5% should be HIGHER than in the all-write baseline because the state is less crowded.

### decay tolerance shift

add a third sweep comparing the decay value at which 50% of 32 patterns are recallable, for raw-value vs correction-field at prediction_quality=0.9.

expected result: the correction-field design should tolerate lower decay values (faster forgetting) because the per-pattern load is smaller.

## what we learn from this simulation

- if correction-field storage consistently increases capacity at matched parameters, the design is worth implementing in `god_machine.py`
- if the capacity increase scales with prediction quality, the self-compression property is confirmed empirically
- if surprise gating reduces write frequency without hurting recall on important tokens, the predictive-filtering mechanism works
- if the decay tolerance shifts downward, the interaction between correction-field storage and decay is favorable
- if NONE of these hold, the correction-field design does not help and the matrix memory's capacity ceiling is not addressable by changing what gets stored (it is a structural limit of the outer-product mechanism itself)

## files to read before implementing

- `neuroloc/wiki/synthesis/correction_field_memory.md` (the design being tested)
- `neuroloc/simulations/memory/asymmetric_outer_product_recall.py` (the existing baseline simulation — reuse its infrastructure and encoding functions)
- `neuroloc/simulations/shared.py` (metrics infrastructure)
- `neuroloc/simulations/suite_registry.py` (registration)
- `neuroloc/wiki/tests/encoding_simulation_round_b.md` (the capacity ceiling finding this simulation tries to overcome)
- `neuroloc/wiki/tests/decay_sweep_results.md` (the decay=0.90 threshold this simulation tries to shift)

## prosecutor requirement

run `feature-dev:code-reviewer` on the simulation file before running. the simulation must be mathematically correct before its results inform any architectural decision. specifically verify:
- the synthetic prediction formula `pred = q * v + sqrt(1-q^2) * noise` correctly controls explained variance
- the reconstruction step `pred + correction` is applied in the right order
- the paired comparison shares the same random draw between conditions
- the surprise ratio formula is numerically stable (check for zero-norm edge case)
- the metrics correctly distinguish "retrieval of raw correction" from "reconstruction quality after adding prediction"
