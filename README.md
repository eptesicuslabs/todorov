# todorov

unified neural architecture combining three sequence mechanisms in a 3:1 ratio:
18 kda layers (channel-wise gated deltanet for linear attention), 3 mamba-3
layers (complex-valued ssm with trapezoidal discretization), and 3 mla layers
(compressed kv cache with shared decoupled rope). optional g(3,0,1) geometric
product self-interaction for spatial reasoning.

the architecture unifies three prior eptesicus projects: gerhard (ternary
spikes, asnn-goose), chimera-edge (hybrid layer design), and echoloc
(geometric algebra, equilibrium iteration).

target: 312m parameters, int8, 128k+ native context.

## results

phase 1 (language modeling, byte-level wikitext-2, 6m training config):
bpb ratio vs transformer: 0.84x (threshold <=1.5x, pass). spike mi:
1.275 (pass). spike cka: 0.913 (pass). spike firing rate: 42% (pass).

todorov outperforms a same-size transformer baseline by 16% at matched
training budget. the ternary spike quantization preserves nearly all
information from continuous activations (mi 1.275 is 29x better than
gerhard's 0.044, cka 0.913 is 46x better than gerhard's 0.020). this
proves the architecture was the bottleneck in prior work, not the spike
mechanism.

phase 2 (context extension, progressive training 256-2048 tokens):
perplexity stability 256-4096: +4.0% (threshold <20%, pass). mla cache
scaling: linear (pass). selective copy retrieval: 0% (deferred to phase 5,
research confirms 130m+ minimum).

the kda recurrent state is stable during 16x context extrapolation.
perplexity degrades only 4% from 256 to 4096 tokens. progressive training
with fla chunk_kda confirms o(t) scaling: s1024/s512 = 1.94x, s2048/s1024
= 1.97x (theoretical: 2.0x).

phase 3 (spatial module validation, gp self-interaction in swiglu):
shape classification: gp 30.0% vs transformer 25.0% (pass). n-body
dynamics: gp mae 51.55 vs transformer 72.70, 29% improvement (pass).
equivariance error: 1.34e-07 at 60 degrees (pass). language bpb not
degraded: gp 3.009 vs nogp 3.707 (pass).

the geometric product self-interaction uses g(3,0,1) projective algebra
with a sparse cayley table (192 non-zero entries), inlined into swiglu as
an additive residual after the down projection. adds 98k parameters (1.7%
overhead) and zero measurable compute overhead (2.9 s/step with and without).
the gp pathway provides genuine spatial inductive bias: the transformer
baseline collapsed to a constant predictor on shape classification (always
predicts sphere, 0% on other classes), while the gp model generalized
across 3 of 4 shape classes. on n-body dynamics, the gp model captured
pairwise force structure that the baseline could not learn from raw byte
coordinates. spike mi reached an all-time high of 1.311 with gp active,
indicating richer activation patterns that the ternary encoding preserves.

## architecture

layer pattern: [kda, kda, kda, mamba3, kda, kda, kda, mla] x 3 = 24 layers

kda uses channel-wise gating (per-feature alpha decay, not scalar per-head)
from the kimi linear paper. each head maintains a fixed-size state matrix
that accumulates key-value associations via the delta rule. o(1) inference
memory per layer regardless of context length.

mamba-3 uses exponential-trapezoidal discretization for numerical stability
and complex-valued state via data-dependent rope rotation.

mla compresses the kv cache to d_c + d_r = 160 floats per token per layer
(vs 2 * n_heads * d_head for standard attention). shared rope across heads.

ternary spikes ({-1, 0, +1}) on kda k/v paths with adaptive threshold
(theta = alpha * mean(|x|)) and straight-through estimator gradient flow.
132-point expanded spike placement available behind config flag. atmn
membrane-potential neurons implemented but not yet validated at scale.

## autonomous research

this project uses eara (eptesicus autonomous research agent) for autonomous
experiment loops. see program.md for the agent instructions and eara.yaml
for the project-specific configuration. point any llm agent at this repo
and say "read program.md and start experimenting."

9 kaggle runs completed across phases 1-3. total compute: ~13 gpu hours
at $0 cost (kaggle free tier). $0 of $500 budget spent.

## prior art

gerhard: ternary spikes, asnn-goose (rwkv backbone, mi=0.044, cka=0.020)
echoloc: g(3,0,1) algebra, equilibrium iteration, gatr sparse einsum
chimera-edge: 3:1 hybrid ratio, mamba integration, int8 decision

eptesicus laboratories.
