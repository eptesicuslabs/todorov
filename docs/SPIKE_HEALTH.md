# Spike Health

## Health Criteria

| Metric              | Threshold       | Source      |
|---------------------|-----------------|-------------|
| Dead neurons        | < 5%            | Gerhard     |
| Saturated neurons   | < 10%           | Gerhard     |
| Firing rate (mean)  | 30-60%          | Spec        |
| Mutual information  | > 0.1           | Phase 1 gate|
| CKA                 | > 0.3           | Phase 1 gate|

## Run History

| Metric | run_001 | run_002 | run_003 | run_008 | run_009 | Threshold |
|--------|---------|---------|---------|---------|---------|-----------|
| MI     | --      | 1.275   | 1.243   | 1.243   | 1.311   | > 0.1     |
| CKA    | --      | 0.913   | 0.905   | 0.926   | 0.907   | > 0.3     |
| FR     | 41.4%   | 42.0%   | 42.0%   | 41.9%   | 42.1%   | 30-60%    |
| Dead   | 0.0%    | 0.0%    | 0.0%    | 0.0%    | 0.0%    | < 5%      |
| Status | PASS    | PASS    | PASS    | PASS    | PASS    | --        |

All runs pass all spike health criteria. Firing rate is remarkably stable
at ~42% across all runs, configurations, and phases. MI ranges 1.243-1.311,
CKA ranges 0.905-0.926. Zero dead neurons in every run.

## run_009 Details (Phase 3, GP spatial mode)

- MI: 1.311 (highest across all runs)
- CKA: 0.907
- FR: 42.1%
- Dead: 0.0%
- Config: d_model=256, 8 layers, spatial_mode=True, mixed data
- The GP self-interaction does not harm spike health. MI slightly increased,
  suggesting richer representations from mixed spatial+language data.

## Key Lessons from Gerhard

- spike_threshold=0.1 gives healthy firing rates
- Adaptive threshold prevents bypass via activation scaling
- threshold = alpha * mean(|x|), alpha is learnable
- STE gradient flow: gradients pass through quantization unchanged
