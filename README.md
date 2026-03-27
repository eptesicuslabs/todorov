# todorov

a neural architecture built on biological computation principles -- ternary
spikes, adaptive thresholds, and recurrent state accumulation -- that
outperforms standard transformers at matched scale, not despite the
information bottleneck but because of it.

## latest results

280m parameters, byte-level, fineweb-edu.

bpb ratio vs matched transformer at 280m: **0.72x**
spike mutual information: **1.246** (threshold 0.1)
spike cka similarity: **0.802** (threshold 0.3)
spike firing rate: **41%** (target 30-60%)

the architecture advantage grows with scale: 16% better at 6m, 28-34%
better at 280m. ternary spikes preserve information across a 45x scale
increase with zero dead neurons. the geometric product spatial module
improves n-body dynamics prediction by 29% at zero compute overhead.

## architecture

three mathematical principles under one framework:

**delta-rule state.** layers accumulate context through error-correcting
writes to a fixed-size state matrix. channel-wise decay gates control
per-feature forgetting. o(1) inference memory per layer regardless of
context length.

**continuous dynamics.** layers evolve state through discretized
differential equations with trapezoidal integration. data-dependent
selection controls what enters state.

**latent attention.** layers compress the full context into low-rank
representations, providing exact recall where recurrent state cannot.

all paths pass through ternary spike quantization ({-1, 0, +1}) with an
adaptive threshold (theta = alpha * mean(|x|)). the spikes force the
architecture to represent information sparsely.

**spatial reasoning.** g(3,0,1) projective geometric algebra as additive
self-interaction inside the gated feed-forward network. zero measurable
compute overhead.

layer pattern: 6 delta-rule + 1 continuous-dynamics + 1 latent-attention,
repeated. 3:1 ratio of recurrent to full-attention layers. the adaptive
threshold produces ~41% firing rate, proven scale-invariant from 6m to
280m params across 11 runs.

## status

11 training runs completed across 5 phases. the architecture scales and
the biological constraints hold. current work focuses on parallel scan
kernels for the continuous-dynamics layers and membrane-potential neuron
validation at scale.

## sponsorship

this research is compute-limited. the architecture is validated but
training at competitive token budgets (100b+ tokens) and scales (1b+
params) requires gpu access beyond what free tiers provide. if you are
interested in sponsoring compute for this work, please reach out to
eptesicus laboratories.

eptesicus laboratories.
