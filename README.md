# todorov

a neural architecture built on biological computation principles: ternary
spikes, adaptive thresholds, and recurrent state accumulation. the thesis
is that these constraints improve language modeling, not despite the
information bottleneck but because of it.

three mathematical principles, one framework:

- delta-rule state layers accumulate context through error-correcting writes
  to a fixed-size state matrix. channel-wise decay gates control per-feature
  forgetting. o(1) inference memory per layer regardless of context length.

- continuous-dynamics layers evolve state through discretized differential
  equations with trapezoidal integration. data-dependent selection controls
  what enters state.

- latent-attention layers compress the full context into low-rank
  representations, providing exact recall where recurrent state cannot.

all paths pass through ternary spike quantization ({-1, 0, +1}) with an
adaptive threshold (theta = alpha * mean(|x|)). the spikes force the
architecture to represent information sparsely. they are not a compression
trick. they are the point.

the spatial module uses g(3,0,1) projective geometric algebra as additive
self-interaction inside the gated feed-forward network. it provides spatial
inductive bias at zero measurable compute overhead.

## results

phase 5 (scale validation, 280m params, byte-level fineweb-edu):
bpb ratio vs transformer: 0.72x (28% better at matched scale). spike mi:
1.246, cka: 0.802, firing rate: 41%. all health metrics well above
thresholds. the architecture advantage GROWS with scale: 16% at 6m,
28-34% at 267-280m.

phase 3 (spatial validation, g(3,0,1) geometric product):
n-body dynamics: 29% mae improvement over transformer baseline.
shape classification: generalized across 3/4 classes (transformer
collapsed to constant predictor). language bpb not degraded. spike mi
reached all-time high (1.311) with geometric product active.

phase 2 (context extension, progressive training 256-2048):
perplexity stability: +4% across 16x extrapolation. linear cache
scaling confirmed. fla o(t) scaling confirmed (1.94-1.97x per 2x length).

phase 1 (proof of concept, 6m params, byte-level wikitext-2):
bpb ratio 0.84x (16% better). ternary spikes preserve 29x more mutual
information than the predecessor architecture (mi 1.275 vs 0.044).

## architecture

layer pattern: 6 delta-rule + 1 continuous-dynamics + 1 latent-attention,
repeated. 3:1 ratio of recurrent to full-attention layers.

the adaptive threshold (alpha * mean(|x|)) with alpha=1.0 produces ~41%
firing rate, proven scale-invariant from 6m to 280m params across 11 runs.
the straight-through estimator provides gradient flow through the ternary
quantization.

132-point expanded spike placement and membrane-potential neurons (adaptive
threshold with per-neuron learned v_th) are implemented but not yet
validated at scale.

## autonomous research

this project uses eara (eptesicus autonomous research agent) for autonomous
experiment loops. 11 runs completed across phases 1-5 (9 kaggle t4 + 2
runpod h200).

## lineage

three prior eptesicus projects: gerhard (ternary spikes on rwkv, proved
spikes work but architecture bottlenecked them), echoloc (geometric
algebra, proved self-interaction works but neighbor-interaction fails),
chimera-edge (validated the 3:1 layer ratio).

eptesicus laboratories.
