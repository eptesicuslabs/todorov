# Todorov Architecture

## Overview

Todorov is a 24-layer hybrid architecture combining three attention/sequence
mechanisms in a 3:1 ratio pattern, with optional geometric algebra spatial
reasoning. 312M parameters (INT8), 128K+ native context.

## Layer Pattern

    [KDA, KDA, KDA, Mamba3, KDA, KDA, KDA, MLA] x 3 = 24 layers

- 18 KDA layers (75%): channel-wise gated delta-rule linear attention
- 3 Mamba-3 SISO layers (12.5%): complex-valued SSM with trapezoidal discretization
- 3 MLA layers (12.5%): compressed KV cache with shared decoupled RoPE

## KDA (Channel-wise Gated DeltaNet Attention)

Each head maintains a fixed-size state matrix S of shape (d_k, d_v).

Recurrent update (inference):
    S_t = diag(alpha) * S_{t-1} + beta_t * v_t * k_t^T
    o_t = S_t * q_t

Parallel path (training):
    weights[t,s] = (Q @ K^T)[t,s] * alpha_mean^(t-s) * beta[s]
    O = causal(weights) @ V

The parallel path replaces the sequential Python loop with a single
matmul + exponential decay mask. Uses mean alpha across feature dims
as approximation for channel-wise gating. Max approximation error: ~0.1.

- alpha: per-feature decay (channel-wise, learned via sigmoid(alpha_log))
- beta_t: per-token write gate from input projection
- Complexity: O(T) inference, O(T^2) training (but parallel)
- Spike points: Q, K, V, and output projections (4 per layer)

## Mamba-3 SISO

Selective state space model with:
- Exponential-trapezoidal discretization for numerical stability
- Complex-valued state via data-dependent RoPE rotation
- Input gating via SiLU activation
- d_state=32, expand=2 (d_inner = 2*d_model)
- Spike points: input and output projections (2 per layer)

Sequential scan only (data-dependent A prevents simple parallel decomposition).
For Phase 2+: integrate official Mamba selective scan CUDA kernel.

## MLA (Multi-Head Latent Attention)

Standard softmax attention with compressed KV cache:
- Down-project input to d_c=128 latent dimension
- Up-project latent to K and V for attention computation
- Decoupled RoPE: shared d_R=32 dimension (not per-head)
- KV cache stores d_c + d_R = 160 floats per token per layer
- Compression ratio: ~8x vs standard MHA at d_model=1024
- Spike points: Q projection and KV compression (2 per layer)

## SwiGLU MLP

Every layer is followed by a SwiGLU MLP:
    hidden = SiLU(W_gate * x) * (W_up * x)
    output = W_down * hidden

Hidden dimension: d_model * 2.25, rounded to 64-byte alignment.

When spatial_mode=True, adds GP self-interaction:
    gp_out = GP(W_left(x), W_right(x))
    hidden = hidden + gp_out

Spike points: gate and up projection inputs (2 per layer).

## Spike System

Two spike neuron types:

TernarySpike (default):
    threshold = alpha * mean(|x|)
    output = sign(x) * I(|x| >= threshold)
    Memoryless, global adaptive threshold, STE gradient flow.

ATMN (Adaptive Threshold Membrane Neuron):
    h_t = x_t + (1/tau) * u_{t-1}
    s_t = sign(h) * I(|h| >= V_th)
    u_t = h_t - s_t * V_th
    V_th = exp(a), per-neuron learnable threshold.
    Temporal state (membrane potential), residual charge accumulation.

Total spike points when spike_all_projections=True:
    KDA:    4 per layer x 18 layers = 72
    Mamba3: 2 per layer x 3 layers  = 6
    MLA:    2 per layer x 3 layers  = 6
    SwiGLU: 2 per layer x 24 layers = 48
    TOTAL:  132

Default: only KDA K/V (36 total).

## Block Structure

Each layer uses pre-norm (RMSNorm) + residual connection:
    x = x + Attn(RMSNorm(x))
    x = x + MLP(RMSNorm(x))

## Embedding and Output

- Token embedding: vocab_size x d_model
- RoPE: applied within KDA and MLA layers (not globally)
- Output: RMSNorm + linear projection with tied embedding weights

## Configuration

    d_model:    1024
    n_layers:   24
    vocab_size: 32000
    max_seq_len: 131072
    mlp_ratio:  2.25
    total_params: ~312M

## Memory Budget (INT8, Batch=1)

    4K context:   ~352 MB total
    128K context: ~585 MB total (FP32 cache), ~365 MB (FP16 cache)
    1M context:   ~1.9 GB (FP32), ~1.1 GB (FP16)

KDA and Mamba-3 states are fixed regardless of context length.
Only MLA cache scales with context.

## Distillation

Bidirectional SBDS-style:
    L = 0.2 * KL(student || teacher) + 0.7 * KL(teacher || student)
        + 0.1 * MSE(student_features, teacher_features)

Pre-norm feature alignment (MSE on features before activation, not after).
Reverse KL (0.7 weight) is mode-seeking, matching sparse spiking patterns.

## Optional Spatial Module

G(3,0,1) projective geometric algebra with 16-component multivectors.
Grades: scalar (1), vector (4), bivector (6), trivector (4), pseudoscalar (1).

GATr-style sparse einsum replaces nested-loop geometric product.
Pin-equivariant linear layers preserve grade structure.

Spatial reasoning is per-token (self-interaction), not per-neighbor.
