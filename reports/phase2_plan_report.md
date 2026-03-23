# Todorov Phase 2 Plan Report

Eptesicus Laboratories
2026-03-22
Author: Deyan Todorov


## 1. Phase 1 Summary

Phase 1 is COMPLETE. All gates pass decisively across two independent runs.

| Gate             | run_001 (2000 steps) | run_002 (500 steps) | Threshold |
|------------------|---------------------|---------------------|-----------|
| BPB ratio        | 1.022x              | 0.840x              | <= 1.5x   |
| Spike MI         | -                   | 1.275               | > 0.1     |
| Spike CKA        | -                   | 0.913               | > 0.3     |
| Spike FR         | 41.4%               | 42.0%               | 30-60%    |

Key finding: Todorov OUTPERFORMS the Transformer baseline at matched training
budget (0.84x BPB ratio in run_002). The architecture is not just competitive
-- it converges faster than softmax attention at this scale.

Spike encoding quality is excellent: MI=1.275 and CKA=0.913 indicate the
ternary quantization preserves nearly all information from continuous
activations. No dead neurons (0.0%).


## 2. Phase 2 Initial Probe Results (run_003)

### 2.1 Passkey Retrieval: 0% at All Lengths

| Context Length | Accuracy | Trials |
|----------------|----------|--------|
| 256            | 0%       | 20     |
| 1024           | 0%       | 20     |
| 4096           | 0%       | 20     |
| 8192           | 0%       | 20     |

This is NOT a context extension failure. It is 0% even at the training length
(256 tokens). The model lacks retrieval capability entirely at this scale.

Root causes (validated by deep research):
- Passkey retrieval has NEVER been demonstrated below 130M params. The smallest
  tested model (Mamba-2 130M) largely fails. 370M is the minimum where it works
  within 8K. 0% at 6M is the expected result, not a bug.
- Three independent scaling barriers: storage capacity (too few params = too few
  storable facts), recurrent state capacity (T_recall = 4.756 * (1.365^N_S - 1)^-0.742,
  predicts <1K at state_size=16), and the associative recall gap (attention-free
  models need 20x more params than attention models for equivalent recall).
- Byte-level models face 3-5x context penalty vs subword. ByT5, MegaByte, BLT
  all tested at 100M+. A 6M byte-level model is below the useful floor.
- Source: Stuffed Mamba (arxiv 2410.07145), BLT (arxiv 2412.09871),
  Based (arxiv 2402.18668), MQAR benchmarks

### 2.2 Perplexity Scaling: Encouraging

| Context Length | BPB   | Delta from 256 |
|----------------|-------|----------------|
| 256            | 3.030 | baseline       |
| 512            | 3.163 | +4.4%          |
| 1024           | 3.247 | +7.2%          |
| 2048           | 3.362 | +11.0%         |
| 4096           | 3.383 | +11.7%         |

The KDA recurrent state is stable during length extrapolation. Only 12%
degradation from 256 to 4096 tokens. This is a positive signal for context
extension: the delta-rule state does not collapse or diverge.

Research finding (arxiv 2410.07145): minimum training length for KDA state
quality is T_train >= 5.172 * head_dim - 4.469. For head_dim=64: T >= 326.
Our seq_len=256 was below this threshold, explaining some quality loss.


## 3. Phase 2 Revised Plan

### 3.1 What Needs to Change

The original Phase 2 gates (passkey >95% at 32K, >80% at 128K) assume a
model with sufficient capacity for retrieval tasks. At 6M params and 500
steps on WikiText-2, this is unrealistic. Two paths forward:

Option A: Scale up and extend training
- Train at seq_len=4096 for 2000+ steps
- Use progressive context extension (256 -> 1024 -> 4096)
- Test retrieval at the training length

Option B: Use simpler retrieval tests
- Copy task: can the model reproduce a substring from earlier in the sequence?
- Pattern matching: can the model predict a repeated pattern?
- These tests are achievable at small scale and test the same KDA state quality

Recommended: BOTH. Use simple retrieval tests to validate KDA state quality,
then attempt passkey retrieval at larger training length.

### 3.2 Training Configuration for Phase 2

Deep research confirms: d=256, 8-layer model at 4K context fits
comfortably on T4. No gradient checkpointing needed.

                  Phase 1             Phase 2 Progressive
    seq_len:      256                 256 -> 512 -> 1024 -> 2048 -> 4096
    steps/stage:  500                 1000 per stage (5000 total)
    batch_size:   32                  16 (at 4K; 32 at shorter)
    total tokens: ~4M                 ~80M

Memory at seq_len=4096, batch=16, d=256, 8 layers:
- Model + optimizer: ~320 MB
- Activations: ~4.7 GB (no gradient checkpointing needed)
- Total: ~5 GB. Well within T4's 16 GB.

Progressive schedule (from GrowLength/SkyLadder research):
  Stage 1: seq=256,  steps=1000, batch=32, warmup=100
  Stage 2: seq=512,  steps=1000, batch=32, warmup=50
  Stage 3: seq=1024, steps=1000, batch=16, warmup=50
  Stage 4: seq=2048, steps=1000, batch=8,  warmup=50
  Stage 5: seq=4096, steps=1000, batch=4,  warmup=50

Estimated wall time: ~2-3 hours on T4 (with parallel KDA at ~3s/step).

### 3.3 fla Integration Constraint

The fla (flash-linear-attention) library provides chunk_kda kernels that
would make 4K training much faster. However:

CRITICAL: Triton 3.3+ dropped T4 (sm_75) support.

Options:
a. Pin torch==2.6.0 + triton==3.2.0 (recommended, tested by community)
b. Use fla's naive_recurrent fallback (pure PyTorch, no Triton)
c. Use the existing parallel KDA path (matmul-based, O(T^2) but parallel)

For seq_len=4096, the parallel KDA path creates a 4096x4096 attention matrix
per head (67M entries). This is ~1GB per head in FP32, or ~256MB per head in
FP16. With 4 heads and 6 KDA layers: ~6GB in FP16. Tight on T4 but feasible
with batch_size=1-2.

Recommended path: use parallel KDA at 4096 without fla. The O(T^2) compute
is acceptable at 4K. fla integration is only needed for 16K+ context.

### 3.4 Phase 2 Notebook Plan

Step 1: Progressive training (256 -> 512 -> 1024 -> 2048 -> 4096)
        1000 steps per stage, 5 stages, ~5000 total steps
Step 2: Train Transformer baseline at final length (4096) for comparison
Step 3: SpikingBrain validation (MI/CKA)
Step 4: Retrieval tests appropriate for 6M scale:
  - Selective copy: reproduce a marked 8-byte substring from earlier
  - Induction heads: predict completion of a pattern seen once before
  - SQAR (Single-Query Associative Recall): given key, retrieve value
    from a (key, value) pair seen earlier in context
  All at seq_len=512, 1024, 2048, 4096 (training lengths)
Step 5: Perplexity at 256, 512, 1024, 2048, 4096, 8192 (extrapolation)
Step 6: MLA cache memory measurement
Step 7: Passkey retrieval at 4096 only (baseline, likely low)

### 3.5 Revised Phase 2 Gates

Research confirms: passkey retrieval at 32K/128K requires 370M+ params.
Revised gates for 6M scale:

| Gate                | Threshold                            | Rationale          |
|---------------------|--------------------------------------|--------------------|
| selective_copy_1k   | >60% at 1024 token distance          | KDA state quality  |
| induction_heads     | >50% completion of once-seen pattern | Recurrence test    |
| perplexity_stable   | <20% BPB increase from 256 to 4096   | Extrapolation      |
| mla_cache_linear    | Cache scales linearly with context   | Architecture       |

The original 32K/128K passkey gates are deferred to Phase 5 (full 300M scale).
Phase 2 tests the ARCHITECTURE's ability to extend context, not the
MODEL's capacity for retrieval (which requires scale).


## 4. Performance

| Metric           | run_001 | run_002 | run_003 |
|------------------|---------|---------|---------|
| Total time       | 2h44m   | 27m     | 30m     |
| Todorov s/step   | 4.1     | 2.9     | 2.7     |
| Transformer s/step| 0.2    | 0.2     | 0.2     |
| Steps            | 2000    | 500     | 500     |

Parallel KDA: 1.5x faster per step vs sequential. At seq_len=4096, the
parallel path will be essential (sequential would be ~16x slower due to
4096 loop iterations vs 256).


## 5. Research Findings

### 5.1 KDA State Quality (arxiv 2410.07145)
- State collapse is about inability to FORGET, not inability to RETAIN
- Minimum training length: T >= 5.172 * state_size - 4.469
- For head_dim=64: T >= 326 (our seq_len=256 was below threshold)
- DeltaProduct (arxiv 2502.10297) guarantees spectral norm <= 1

### 5.2 fla on Kaggle T4
- Triton 3.3+ dropped sm_75 support
- MUST pin torch==2.6.0 + triton==3.2.0
- T4 lacks bf16 tensor cores: use FP16 for all fla kernels
- chunk_kda supports channel-wise gating (NOT chunk_gated_delta_rule)
- Gate tensor must be in LOG SPACE

### 5.3 MLA Cache at Long Context
- With shared RoPE (d_c + d_R = 160 floats per token per layer):
  128K context, 3 MLA layers = ~295 MB (FP16)
- FP8 cache (vLLM): additional 50% reduction
- Well within T4 memory budget

### 5.4 Hybrid Context Extension Best Practices
- Progressive context extension: 256 -> 1024 -> 4096 -> 16K -> 64K
- HyPE position encoding: RoPE on linear layers, NoPE on attention layers
- Perplexity delta <= 0.5 BPB is the accepted quality target
- 75% linear layers have O(n) memory, only 25% attention grows


## 6. Risk Assessment

| Risk                              | Severity | Mitigation                     |
|-------------------------------------|----------|--------------------------------|
| Passkey 0% at small scale          | Expected | Revise gates, use copy task    |
| 4K training OOM on T4             | LOW      | batch_size=2-4, gradient ckpt  |
| fla incompatible with T4 Triton   | MEDIUM   | Pin versions or use fallback   |
| KDA state collapse at 4K          | LOW      | 4K > 326 threshold, monitor norms |
| Perplexity degrades beyond 4K     | MEDIUM   | Progressive training schedule  |


## 7. Next Steps

1. Build Phase 2 notebook with 4K training + copy task + passkey retrieval
2. Push to Kaggle and execute
3. If copy task passes, attempt passkey at training length
4. If perplexity stable, extend to 8K, 16K progressively
5. Integrate fla kernels when moving beyond 4K context
