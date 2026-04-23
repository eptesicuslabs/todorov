# Todorov Phase 0 Status Report

historical note (2026-04-22): this is an early project snapshot preserved for archive. it is not a current status surface.

Eptesicus Laboratories
2026-03-22
Author: Deyan Todorov

## Executive Summary

Phase 0 (Foundation) of the Todorov unified architecture is functionally
complete. All core modules are implemented, all 63 unit tests pass, and
the Phase 1 training notebook is ready for Kaggle deployment. Two issues
require attention before scaling: training throughput is 6.3x slower than
the Transformer baseline due to naive sequential recurrence, and the model
is 17% over the 300M parameter target. Both have known fixes with clear
timelines.


## What Was Built

69 files, 3,995 lines of Python, 1,735 lines of research notes.

### Architecture Implementation

24-layer hybrid model with [KDA, KDA, KDA, Mamba3, KDA, KDA, KDA, MLA] x 3:

| Component                  | Status   | Source           |
|----------------------------|----------|------------------|
| KDA (channel-wise gating)  | Complete | New, from papers |
| Mamba-3 SISO (trapezoidal) | Complete | New, from papers |
| MLA (compressed KV cache)  | Complete | New, from papers |
| SwiGLU + optional GP       | Complete | New + Echoloc    |
| TernarySpike (STE)         | Complete | Ported: Gerhard  |
| SpikingBrain validator     | Complete | Ported: Gerhard  |
| G(3,0,1) sparse GP         | Complete | Ported: Echoloc  |
| Equivariant linear         | Complete | Ported: Echoloc  |

### Infrastructure

| Component                      | Status   |
|--------------------------------|----------|
| Training loss (CE + residual)  | Complete |
| AdamW + cosine scheduler       | Complete |
| Sliding window evaluator       | Complete |
| Memory budget calculator       | Complete |
| Kaggle autoresearch pipeline   | Complete |
| Phase gate evaluator           | Complete |
| 11 knowledge files (30+ papers)| Complete |
| All docs and state files       | Complete |


## Test Results

63/63 tests passing across 8 test files:

| Test File          | Tests | Status |
|--------------------|-------|--------|
| test_kda.py        | 10    | PASS   |
| test_mamba3.py     | 7     | PASS   |
| test_mla.py        | 6     | PASS   |
| test_gp.py         | 10    | PASS   |
| test_spikes.py     | 11    | PASS   |
| test_model.py      | 7     | PASS   |
| test_memory.py     | 5     | PASS   |
| test_equivariance.py| 6    | PASS   |
| TOTAL              | 63    | PASS   |

Key validations:
- KDA recurrence: parallel and sequential modes produce consistent output
- Geometric product: Cayley table correctness verified (e1*e1=1, e0*e0=0)
- GP equivariance: sandwich product preserves norms under 60-degree rotation
- Sparse GP: vectorized implementation matches dense loop results
- Ternary spikes: STE gradient flow verified (gradients pass unchanged)
- Full model: forward/backward pass, autoregressive generation


## Memory Budget

Measured on BASE_CONFIG (d_model=1024, 24 layers):

| Component            | 4K Context | 128K Context | Spec Target |
|----------------------|------------|--------------|-------------|
| Weights (INT8)       | 334 MB     | 334 MB       | 300 MB      |
| KDA recurrent states | 9 MB       | 9 MB         | 5 MB        |
| Mamba-3 SSM states   | 0.75 MB    | 0.75 MB      | 0.5 MB      |
| MLA compressed cache | 18 MB      | 576 MB       | 123 MB      |
| Activations          | 2 MB       | 2 MB         | 2 MB        |
| TOTAL                | 364 MB     | 922 MB       | 431 MB      |

Parameter count: 350M (target 300M, 17% over).

Root cause of MLA cache inflation: per-head RoPE projection stores
d_R * num_heads = 256 floats per token instead of shared d_R = 32.
Fix: share RoPE across heads (single d_R projection). This is a
one-line architectural change deferred to Phase 1 tuning.


## Performance Analysis: Training Throughput

### The Problem

Todorov trains 6.3x slower than a same-size Transformer.

End-to-end benchmark (CPU, d=256, 8 layers, seq=256, batch=4):

| Model       | Forward+Backward | Steps/sec | Est. 2000 steps |
|-------------|------------------|-----------|-----------------|
| Todorov     | 5,465 ms         | 0.2       | 182 min         |
| Transformer | 873 ms           | 1.1       | 29 min          |
| RATIO       | 6.3x slower      |           |                 |

### Root Cause: Sequential Python Loops

Both KDA and Mamba-3 implement their recurrence as Python-level for-loops
over the sequence dimension:

    for t in range(seq_len):         # 256 iterations, each a tiny GPU op
        state = alpha * state + beta * outer(k_t, v_t)
        output_t = state @ q_t

A Transformer processes the same sequence in ONE parallel matmul:

    attn = softmax(Q @ K.T) @ V      # one fused CUDA kernel

Each Python loop iteration launches a separate tiny GPU kernel. The GPU
spends most of its time waiting for Python to dispatch the next operation
rather than doing useful compute. This is a kernel launch overhead problem,
not a FLOP problem.

### Measured Per-Component Bottleneck

KDA recurrence (per layer, CPU, batch=4, heads=4, head_dim=64):

| seq_len | Time      | Per token |
|---------|-----------|-----------|
| 64      | 12.8 ms   | 0.20 ms   |
| 128     | 29.0 ms   | 0.23 ms   |
| 256     | 59.8 ms   | 0.23 ms   |
| 512     | 114.5 ms  | 0.22 ms   |

Mamba-3 SSM scan (per layer, CPU, batch=4, d_inner=512, d_state=16):

| seq_len | Time      | Per token |
|---------|-----------|-----------|
| 64      | 22.2 ms   | 0.35 ms   |
| 128     | 49.2 ms   | 0.38 ms   |
| 256     | 76.9 ms   | 0.30 ms   |
| 512     | 178.6 ms  | 0.35 ms   |

Time scales linearly with seq_len because each token is one Python
loop iteration. This is the fundamental constraint.

### Scaling Wall

At full scale (d_model=1024, 24 layers, seq_len=128K), the naive loops
are catastrophic:

    18 KDA layers x 131,072 tokens x 0.23 ms/token = 543 seconds/layer-pass
    3 Mamba-3 layers x 131,072 tokens x 0.35 ms/token = 138 seconds/layer-pass
    Total forward pass: ~11 minutes (before backward!)

This makes full-scale training impossible with the current implementation.

### Fix: Chunkwise Parallel Scan (Phase 2)

The fla (flash-linear-attention) library implements Triton kernels that
process chunks of C=64 tokens in parallel while maintaining the recurrent
state. This replaces the sequential loop with a parallel scan:

    Naive:  seq_len sequential Python iterations
    fla:    seq_len / 64 chunks, each processed by a fused GPU kernel

Projected improvement:

| seq_len  | Naive iterations | fla chunks | Speedup   |
|----------|------------------|------------|-----------|
| 256      | 256              | 4          | ~64x      |
| 4,096    | 4,096            | 64         | ~64x      |
| 131,072  | 131,072          | 2,048      | ~64x      |

The fla library (github.com/fla-org/flash-linear-attention) directly
supports DeltaNet (the KDA recurrence) and GLA (gated linear attention).
Integration requires:

1. Replace the Python loop with fla.ops.delta_rule.chunk_delta_rule()
2. Reshape inputs to match fla's (batch, heads, seq, dim) convention
3. No changes to the mathematical operation -- same recurrence, same
   output, fused into a single Triton kernel

For Mamba-3, the official mamba_ssm package provides a CUDA selective
scan kernel that replaces the Python loop with a fused scan.

### Fix: MLA is Already Fast

MLA uses standard softmax attention (torch.matmul + softmax), which
is already parallelized. No fix needed. MLA layers (3 of 24) are not
a bottleneck.

### Integration Timeline

| Phase   | Action                                    | Expected Speedup |
|---------|-------------------------------------------|------------------|
| Phase 1 | Naive loops, small config, Kaggle T4      | None (acceptable)|
| Phase 2 | Integrate fla chunk_delta_rule for KDA    | ~40-60x for KDA  |
| Phase 2 | Integrate mamba_ssm selective scan kernel  | ~40-60x for SSM  |
| Phase 5 | Full Triton kernel optimization, INT8     | ~1.5-2x vs Tfmr  |

After fla integration, Todorov's training throughput should be within
1.5-2x of a Transformer, which is the accepted cost for O(1) inference
memory. Published benchmarks from the fla library show DeltaNet chunk
kernels achieving 80-95% of FlashAttention-2 throughput on A100.

### Impact on Phase 1

Phase 1 uses a deliberately small configuration to validate architecture
correctness, not training speed:

    d_model=256, n_layers=8, seq_len=256, batch=32, steps=2000

Estimated Phase 1 wall time on Kaggle T4 GPU:

    Todorov:     ~30-60 minutes
    Transformer: ~5-10 minutes
    Total:       ~40-70 minutes (well within 9-hour Kaggle limit)

The slow training does NOT affect Phase 1 gate evaluation, because both
Todorov and the Transformer baseline run the same number of steps on the
same data. The BPB comparison is compute-matched.


## Phase 1 Readiness

The Phase 1 notebook (notebooks/autoresearch/train.py) is complete and
verified:

- Fully autonomous: downloads WikiText-2, trains both models, evaluates,
  generates all artifacts, creates evidence bundle
- Self-contained: all model code inlined (no external imports from repo)
- Deterministic: seeds set for reproducibility
- Generates: training curves, spike stats, state norms, config dump,
  results.json, evidence bundle ZIP

Phase 1 gates:
- Gate A: BPB within 1.5x of Transformer baseline
- Gate B: Spike MI > 0.1, CKA > 0.3, firing rate 30-60%
- Kill condition: BPB > 3x baseline after hyperparameter search


## Key Architectural Decisions

1. GP optimization: replaced 256-iteration nested Python loop with
   GATr-style sparse einsum (precomputed index tensors + index_add_).
   Single vectorized GPU operation.

2. Channel-wise gating: alpha is a per-feature vector (not scalar),
   giving each dimension its own decay rate. From Kimi Linear paper.

3. Trapezoidal discretization: second-order accurate, more stable than
   Euler/ZOH for Mamba-3 SSM dynamics.

4. MLA decoupled RoPE: position-dependent component stored separately
   from content-dependent latent, enabling cache compression.

5. Ternary spikes on KDA K/V paths: validation against pre-spike
   activations (no teacher model required, unlike Gerhard).


## Risk Assessment

| Risk                         | Severity | Status     | Mitigation                          |
|------------------------------|----------|------------|-------------------------------------|
| Training throughput 6.3x slow| HIGH     | Known fix  | fla Triton kernels (Phase 2)        |
| BPB ratio exceeds 1.5x      | HIGH     | Untested   | HP search: LR, residual penalty, spike threshold |
| Spike MI drops below 0.1    | MEDIUM   | Untested   | Threshold calibration to KDA magnitudes |
| MLA cache 4.7x over spec    | MEDIUM   | Known fix  | Share RoPE across heads (Phase 1)   |
| KDA state 80% over spec     | LOW      | Tracking   | 9 MB vs 5 MB; negligible at 128K but 2x at 1M |
| Parameter count 17% over     | LOW      | Known fix  | Tune dimensions (Phase 1)           |
| Spikes cause gradient issues | LOW      | Mitigated  | STE verified, alpha learnable       |
| Kaggle T4 time limit         | LOW      | Mitigated  | Small config fits in ~60 min        |

Note on spike Gate B: at initialization with alpha=1.0, KDA K-projection
has mean |K| = 0.46, giving threshold = 0.46 and firing rate = 42.4%
(within 30-60% target). During training, magnitudes will shift. If firing
rate stays in range but MI drops below 0.1, the threshold needs recalibration
to KDA's learned magnitude distribution -- not Echoloc's GP magnitudes.


## Next Steps

1. Push Phase 1 notebook to Kaggle via autoresearch pipeline
2. Train Todorov + Transformer baseline on byte-level WikiText-2
3. Evaluate gates, iterate on hyperparameters if needed
4. Fix MLA cache (share RoPE) and parameter count (tune dimensions)
5. Integrate fla chunkwise scan kernels for KDA
6. Begin Phase 2 (context extension) with fast kernels
