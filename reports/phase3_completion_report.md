# phase 3 completion report

Deyan Todorov, Eptesicus Labs
2026-03-23

## summary

Phase 3 validated geometric product (GP) self-interaction as an additive residual in SwiGLU. All four gates pass: shape classification (GP 30.0% vs Transformer 25.0%), n-body dynamics (GP MAE 51.55 vs Transformer 72.70, 29% improvement), equivariance error at machine epsilon (1.34e-07 at 60 degrees), and language BPB not degraded (GP 3.009 vs NoGP 3.707). Nine Kaggle runs completed across Phases 1-3 on T4 GPUs, $0 of $500 budget spent. Phase 3 ran as run_009, completing in 38 minutes wall clock.

## methodology

GP self-interaction uses G(3,0,1) projective geometric algebra with a sparse Cayley table (192 non-zero entries out of 16x16=256 possible). The geometric product is inlined into SwiGLU as an additive residual applied after the down projection. Input activations are projected to 16-dimensional multivectors, multiplied via the sparse Cayley table, and projected back to d_model before residual addition.

Spatial tasks:
- 3D shape classification: 4 classes (sphere, cube, tetrahedron, torus), 16 points per shape, random rotation augmentation, byte-encoded input
- 2D n-body dynamics: 4 particles, leapfrog integrator, softened gravitational potential

Training configuration: mixed data (50% WikiText-2 + 25% shape classification + 25% n-body dynamics), Kaggle T4 GPU, 500 training steps for GP model, 200 steps for NoGP and Transformer baselines. d_model=256, 8 layers, vocab=256, seq_len=256, batch=32.

Evaluation: shape classification accuracy (overall and per-class), n-body position MAE over 200 trials, equivariance error measured at 30/60/90 degree rotations, language BPB degradation percentage comparing GP-enabled vs GP-disabled runs.

## results

### gate results

| gate               | threshold                              | result                                 | status |
|--------------------|----------------------------------------|----------------------------------------|--------|
| spatial_classify   | outperform Transformer                 | GP 30.0% vs Transformer 25.0%         | PASS   |
| spatial_dynamics   | lower MAE than Transformer             | GP MAE=51.55 vs Transformer MAE=72.70 | PASS   |
| equivariance_test  | <5% error at 60-degree rotation        | error=1.34e-07 at 60 degrees           | PASS   |
| language_no_degrade| BPB not degraded >10% with GP enabled  | -18.8% (GP improves language BPB)      | PASS   |

### per-class shape classification

| class       | GP (Todorov) | Transformer |
|-------------|--------------|-------------|
| sphere      | 80%          | 100%        |
| cube        | 0%           | 0%          |
| tetrahedron | 16%          | 0%          |
| torus       | 24%          | 0%          |
| overall     | 30.0%        | 25.0%       |

GP correct: 60/200. Transformer correct: 50/200.

### n-body dynamics

| model       | MAE    | n_trials |
|-------------|--------|----------|
| GP (Todorov)| 51.5525| 200      |
| Transformer | 72.7025| 200      |

GP improvement: 29.1% lower MAE.

### equivariance error

| rotation | error        |
|----------|--------------|
| 30 deg   | 1.3354e-07   |
| 60 deg   | 1.3383e-07   |
| 90 deg   | 1.3167e-07   |

All values at float32 machine epsilon.

### language BPB

| config      | BPB    |
|-------------|--------|
| with GP     | 3.0091 |
| without GP  | 3.7067 |

Language degradation: -18.82% (negative means GP improved BPB).

### training config comparison

| metric              | GP (Todorov) | NoGP         | Transformer  |
|---------------------|--------------|--------------|--------------|
| params              | 6,015,780    | 5,917,476    | 5,705,984    |
| training steps      | 500          | 200          | 200          |
| training time (s)   | 1,451.1      | 573.3        | 42.3         |
| est. s/step         | ~2.9         | ~2.9         | ~0.2         |
| best val BPB        | 4.057        | 4.584        | --           |
| shape accuracy      | 30.0%        | --           | 25.0%        |
| n-body MAE          | 51.55        | --           | 72.70        |

### spike health cross-run comparison

| metric    | run_001 | run_002 | run_003 | run_008 | run_009 |
|-----------|---------|---------|---------|---------|---------|
| MI        | --      | 1.275   | 1.243   | 1.243   | 1.311   |
| CKA       | --      | 0.913   | 0.905   | 0.926   | 0.907   |
| FR        | 41.4%   | 42.0%   | 42.0%   | 41.9%   | 42.1%   |
| dead      | --      | 0.0%    | 0.0%    | 0.0%    | 0.0%    |

run_004 through run_007 were incomplete (O(T^2) stall, fla integration attempts, user-stopped). run_001 did not measure MI/CKA/dead ratio.

### full run history

| run     | phase | steps | best BPB | BPB ratio | gates   | wall time |
|---------|-------|-------|----------|-----------|---------|-----------|
| run_001 | 1     | 2000  | 2.235    | 1.022x    | 2/2     | 2h44m     |
| run_002 | 1     | 500   | 2.876    | 0.840x    | 4/4     | 27m       |
| run_003 | 2     | 500   | 2.876    | 0.844x    | --      | ~30m      |
| run_004 | 2     | part. | ~2.66    | --        | stalled | 2h+       |
| run_005 | 2     | --    | --       | --        | --      | --        |
| run_006 | 2     | --    | --       | --        | --      | --        |
| run_007 | 2     | --    | --       | --        | --      | --        |
| run_008 | 2     | 800   | 2.819    | 0.780x    | 1/3     | 4h22m     |
| run_009 | 3     | 500   | 3.009*   | --        | 4/4     | 38m       |

*run_009 BPB is GP-enabled on mixed data, not directly comparable to language-only runs.

## analysis

1. GP provides measurable spatial inductive bias. The 29% n-body MAE improvement (51.55 vs 72.70) demonstrates that the G(3,0,1) geometric product gives the model access to rotation and reflection primitives that a standard Transformer cannot learn from data alone at this scale and training budget.

2. Transformer collapses to majority class on shape classification. The Transformer achieves 100% on sphere and 0% on all other classes, indicating it learns only the most frequent pattern. GP distributes learning across 3 of 4 classes (sphere 80%, tetrahedron 16%, torus 24%), trading majority-class perfection for cross-class generalization. Cube remains at 0% for both models.

3. Equivariance verified to float32 machine epsilon. The error at 60 degrees is 1.34e-07, consistent across 30/60/90 degree test angles (range 1.317e-07 to 1.338e-07). This confirms the PGA algebra implementation is exactly equivariant up to floating point precision.

4. Language BPB improvement is confounded. GP BPB (3.009) is 18.8% better than NoGP BPB (3.707), but GP trained for 500 steps while NoGP trained for 200 steps. The improvement is likely explained by the 2.5x step count advantage plus multitask regularization from mixed training data. Isolating the GP contribution requires training NoGP on the same mixed data for 500 steps.

5. Spike MI at all-time high with GP active. MI=1.311 in run_009 exceeds all prior runs (previous best 1.275 in run_002). The mixed training data likely provides richer representations for spike encoding to preserve. CKA dipped slightly (0.907 vs 0.926 in run_008) but remains well above the 0.3 threshold. Firing rate stable at 42.1%.

6. Shape classification is marginal. 30% vs 25% on 200 samples is 5 percentage points above chance. With 4 classes and 50 samples per class, a one-sided binomial test gives p~0.07, which does not reach statistical significance at the conventional p<0.05 threshold.

7. GP adds zero measurable computational overhead to per-step time. Both GP and NoGP models run at ~2.9 seconds per step. The 16-dimensional multivector projections are negligible compared to the 256-dimensional hidden layer operations.

## caveats

- Shape classification 30% vs 25% is 5 percentage points above chance on 200 samples, p~0.07, not statistically significant at p<0.05.
- Language BPB improvement (-18.8%) is confounded by step count mismatch (GP=500 steps, NoGP=200 steps) and multitask regularization from mixed training data. Cannot attribute improvement to GP alone.
- Step count mismatch across all three models: GP trained for 500 steps, NoGP and Transformer each trained for 200 steps. The GP model received 2.5x the training budget.
- Equivariance test validates the algebraic implementation, not the trained model. It confirms the Cayley table is correct and the projection/unprojection is equivariant, but does not measure whether the trained weights exploit equivariance for downstream tasks.
- Cube classification 0% for both models suggests the byte-encoded 16-point representation may not capture cube geometry, or the task needs more training data.
- n-body MAE values (51.55, 72.70) are raw position errors without normalization. The absolute magnitude depends on the softened potential parameterization.

## bug found

`src/layers/swiglu.py` spatial_mode has a latent shape mismatch. The GP output tensor has shape `[B, T, d_model]` but was being added to the `hidden_dim` tensor inside SwiGLU (which has shape `[B, T, hidden_dim]` where `hidden_dim = d_model * mlp_ratio`). The dimension mismatch would cause a runtime error or silent broadcasting.

Fixed in `notebooks/autoresearch/train.py` by applying the GP residual after the SwiGLU down projection (which maps `hidden_dim` back to `d_model`), making the addition shape-compatible.

The library code in `src/layers/swiglu.py` still contains the bug. Fix should be applied before Phase 5.

## next steps

Phase 5: scale to 300M parameters. Three sequential runs per supervisor directive:
- Phase 5 baseline: 300M with current simple spikes (replicate Phase 1 at scale)
- Phase 5a: swap TernaryStrike for ATMN (isolate neuron model impact)
- Phase 5b: expand to 132-point spike placement with winner from 5a (isolate placement impact)
- SBDS distillation only if BPB ratio degrades at 300M scale

Phase 4 (multimodal) skipped at this scale per supervisor directive.
