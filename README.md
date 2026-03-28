# Todorov

A neural architecture built on one mathematical object: the **Compressed
Rotational Bilinear Recurrence (CRBR)**. Every layer instantiates
`z_t = Q(R(B(C(x_t), C(h_{t-1}))))` -- compression, bilinear interaction,
rotational structure, and quantization -- with different parameterizations
drawn from four composable families. The biological constraints (ternary
spikes, adaptive thresholds, recurrent state accumulation) are not
approximations of standard transformers. They are the architecture.

## Latest Results

280M parameters, byte-level, FineWeb-Edu.

BPB ratio vs matched transformer at 280M: **0.72x**
Spike mutual information: **1.246** (threshold 0.1)
Spike CKA similarity: **0.802** (threshold 0.3)
Spike firing rate: **41%** (target 30-60%)

The architecture advantage grows with scale: 16% better at 6M, 28-34%
better at 280M. Ternary spikes preserve information across a 45x scale
increase with zero dead neurons. The geometric product spatial module
improves n-body dynamics prediction by 29% at zero compute overhead.

## The CRBR Framework

Four families, one equation.

**Compression (C).** Every path through Todorov encounters an information
bottleneck. Ternary spike quantization maps activations to {-1, 0, +1}
at 1.58 bits per dimension. Low-rank latent projection compresses key-value
representations to a learned sufficient statistic. Gated activation creates
continuous-valued sparsity. These are all instances of one family: projection
to a lower-dimensional representation, with the dimensionality controlling
the fidelity tradeoff.

**Bilinear interaction (B).** Every layer's core computation is a bilinear
map. The outer product writes key-value associations into recurrent state.
The inner product retrieves by similarity. Element-wise gating selectively
amplifies features. The G(3,0,1) geometric product simultaneously computes
scalar, vector, area, volume, and pseudoscalar products between multivectors
-- the most general bilinear map in the framework, subsuming dot product
and gating as grade restrictions.

**Rotation (R).** Rotational geometry at every scale. RoPE encodes position
via SO(2) rotations. Complex-valued state dynamics evolve via data-dependent
SO(2) rotations. The geometric product produces Pin(3,0,1) rotors encoding
spatial transformations. These form an algebraically nested hierarchy:
SO(2) inside Spin(3) inside Pin(3,0,1). Each layer uses the smallest
rotation group that serves its function.

**Quantization (Q).** Output discretization from ternary spikes (training
time) through INT8 weight compression (deployment time). Rotation-based
quantization methods pre-multiply activations by learned rotations that
align the data distribution with the quantization grid -- the rotational
family R applied inside the compression family C.

## Architecture

Three CRBR parameterizations, one schedule:

**Delta-rule layers** (18 of 24). Matrix-valued recurrent state S updated
via error-correcting writes: S_t = decay * S_{t-1} + gate * k_t v_t^T.
O(1) inference memory. Ternary spikes on key/value paths.

**Continuous-dynamics layers** (3 of 24). Vector-valued state evolved via
discretized differential equations with data-dependent selection. Complex
rotation encodes temporal phase.

**Latent-attention layers** (3 of 24). Full context compressed to low-rank
per-token representations. Exact softmax retrieval over compressed cache.
O(t) memory, O(t^2) compute -- the precise complement to recurrent layers.

**Spatial module.** G(3,0,1) projective geometric algebra as additive
self-interaction in the gated feed-forward network. Zero measurable compute
overhead. 29% improvement on spatial reasoning tasks.

Schedule: (delta, delta, delta, dynamics, delta, delta, delta, attention) x 3.

## Status

11 training runs completed across 5 phases. The architecture scales and
the biological constraints hold. Current work: parallel scan kernels for
the continuous-dynamics layers, membrane-potential neuron validation at
scale, and convergence at competitive token budgets.

## Funding

This research is compute-limited. The architecture is validated but training
at competitive token budgets (100B+ tokens) and scales (1B+ parameters)
requires GPU access beyond what free tiers provide. If you are interested
in funding compute for this work, please reach out to Eptesicus Laboratories.

Eptesicus Laboratories.
