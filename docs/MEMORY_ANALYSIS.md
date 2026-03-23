# Memory Analysis

## Parameter Count

Total parameters: 350,039,268 (~350M)
Target: ~300M

The 17% overshoot comes primarily from MLA per-head RoPE projections
and SwiGLU hidden dim rounding to 64-byte alignment.

## Memory Budget (INT8, Batch=1)

| Component           | 4K Context | 128K Context | Spec Target (128K) |
|---------------------|------------|--------------|---------------------|
| Weights (INT8)      | 333.66 MB  | 333.66 MB    | ~300 MB             |
| KDA recurrent state | 9.00 MB    | 9.00 MB      | ~5 MB               |
| Mamba-3 SSM state   | 0.75 MB    | 0.75 MB      | ~0.5 MB             |
| MLA compressed cache| 18.00 MB   | 576.00 MB    | ~123 MB             |
| Activations         | 2.00 MB    | 2.00 MB      | ~2 MB               |
| TOTAL               | 363.41 MB  | 921.41 MB    | ~431 MB             |

## Key Observations

1. KDA states are fixed regardless of context length (constant O(1) memory)
2. Mamba-3 states are fixed regardless of context length
3. MLA cache scales linearly with context length
4. Weights dominate at short context; MLA cache dominates at long context

## Discrepancies from Spec

MLA cache is ~4.7x larger than spec target. The spec assumes d_c + d_R = 160
floats per token per layer. Current implementation stores d_c + d_R * num_heads
= 128 + 32 * 8 = 384 floats per token per layer because k_rope is computed
per-head rather than shared across heads.

Fix path: share the RoPE component across heads (single d_R projection instead
of d_R * num_heads). This would bring cache to ~123 MB at 128K, matching spec.

KDA state is 9 MB vs 5 MB spec because the state matrix is head_dim x head_dim
= 128x128 per head. The spec may assume a smaller head_dim or asymmetric K/V dims.

## Layer Counts

| Type   | Count | Percentage |
|--------|-------|------------|
| KDA    | 18    | 75.0%      |
| Mamba3 | 3     | 12.5%      |
| MLA    | 3     | 12.5%      |
| Total  | 24    | 100.0%     |

Pattern: [KDA, KDA, KDA, Mamba3, KDA, KDA, KDA, MLA] x 3
