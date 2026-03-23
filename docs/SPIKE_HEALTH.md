# Spike Health

No trained model spike data yet. Will be populated after Phase 1.

## Health Criteria

| Metric              | Threshold       | Source      |
|---------------------|-----------------|-------------|
| Dead neurons        | < 5%            | Gerhard     |
| Saturated neurons   | < 10%           | Gerhard     |
| Firing rate (mean)  | 30-60%          | Spec        |
| Mutual information  | > 0.1           | Phase 1 gate|
| CKA                 | > 0.3           | Phase 1 gate|

## Key Lessons from Gerhard

- spike_threshold=0.1 gives healthy firing rates
- Adaptive threshold prevents bypass via activation scaling
- threshold = alpha * mean(|x|), alpha is learnable
- STE gradient flow: gradients pass through quantization unchanged
