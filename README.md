# Todorov

A neural architecture built on biological computation principles -- ternary
spikes, adaptive thresholds, and recurrent state accumulation -- that
outperforms standard transformers at matched scale, not despite the
information bottleneck but because of it.

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

## Architecture

Three mathematical principles under one framework:

**Delta-rule state.** Layers accumulate context through error-correcting
writes to a fixed-size state matrix. Channel-wise decay gates control
per-feature forgetting. O(1) inference memory per layer regardless of
context length.

**Continuous dynamics.** Layers evolve state through discretized
differential equations with trapezoidal integration. Data-dependent
selection controls what enters state.

**Latent attention.** Layers compress the full context into low-rank
representations, providing exact recall where recurrent state cannot.

All paths pass through ternary spike quantization ({-1, 0, +1}) with an
adaptive threshold (theta = alpha * mean(|x|)). The spikes force the
architecture to represent information sparsely.

**Spatial reasoning.** G(3,0,1) projective geometric algebra as additive
self-interaction inside the gated feed-forward network. Zero measurable
compute overhead.

Layer pattern: 6 delta-rule + 1 continuous-dynamics + 1 latent-attention,
repeated. 3:1 ratio of recurrent to full-attention layers. The adaptive
threshold produces ~41% firing rate, proven scale-invariant from 6M to
280M parameters across 11 runs.

## Status

11 training runs completed across 5 phases. The architecture scales and
the biological constraints hold. Current work focuses on parallel scan
kernels for the continuous-dynamics layers and membrane-potential neuron
validation at scale.

## Funding

This research is compute-limited. The architecture is validated but
training at competitive token budgets (100B+ tokens) and scales (1B+
parameters) requires GPU access beyond what free tiers provide. If you
are interested in funding compute for this work, please reach out to
Eptesicus Laboratories.

Eptesicus Laboratories.
