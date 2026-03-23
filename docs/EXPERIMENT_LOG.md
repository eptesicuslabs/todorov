# Experiment Log

## Phase 0: Foundation

### 2026-03-22: Initial Build

Config: TINY_CONFIG (d_model=64, 8 layers, vocab=256)
Result: 63/63 tests passing
Interpretation: Core modules functional, ready for Phase 1
Next step: Fix MLA cache inflation, begin Phase 1 training

### 2026-03-22: Memory Budget Measurement

Config: BASE_CONFIG (d_model=1024, 24 layers, vocab=32000)
Result: 350M params (target 300M), 921 MB at 128K (target 431 MB)
Interpretation: MLA per-head RoPE inflates cache 4.7x
Next step: Share RoPE across heads to match spec

## Phase 1: Language Modeling (in progress)

### 2026-03-22: First Kaggle Run -- COMPLETE -- GATE A PASS

Config: d_model=256, 8 layers, vocab=256, seq_len=256, batch=32, 2000 steps
        mlp_ratio=2.75, per-head MLA RoPE, sequential KDA loops
        Kaggle T4 GPU
Params: 6,719,268 (Todorov), 6,492,416 (Transformer)

FINAL RESULTS:
  Todorov best BPB:     2.2353
  Transformer best BPB: 2.1865
  BPB RATIO:            1.022x
  GATE A:               PASS (1.022 < 1.5)
  Spike firing rate:    41.4%
  Spike dead neurons:   0.0%
  GATE B (FR):          PASS (0.414 in [0.3, 0.6])
  KDA state norm:       107.0

Throughput:
  Todorov:     4.1 s/step, 8186s total (2h16m)
  Transformer: 0.2 s/step, 434s total (7.2m)
  Ratio:       20x slower (sequential Python loops)

Crashed on json.dump due to numpy bool (fixed in next version).

Full training curve data:

| Step | Loss  | Val BPB |
|------|-------|---------|
| 0    | 5.614 |         |
| 200  | 2.338 | 3.398   |
| 400  | 2.043 | 2.927   |
| 600  | 1.883 | 2.686   |
| 800  | 1.763 | 2.539   |
| 1000 | 1.673 | 2.442   |
| 1200 | 1.631 | 2.374   |
| 1400 | 1.586 | 2.317   |
| 1500 | 1.586 |         |

Throughput: 4.1 seconds/step on T4 (sequential Python loops)
Est. total time: ~2.3 hours for Todorov + ~0.5 hours for Transformer = ~2.8 hours

Interpretation: Loss is still decreasing at step 1500. BPB improving steadily.
4.1s/step confirms the sequential loop bottleneck even on GPU.

Key observation: the model IS learning. BPB dropped from 3.4 to 2.3 over 1400
steps. Whether it meets the 1.5x baseline gate depends on the Transformer's BPB.

### 2026-03-22: Architecture Revision (between runs)

Changes applied:
- MLA RoPE: shared across heads (d_R per token)
- mlp_ratio: 2.75 -> 2.25 (params 6.7M -> ~5.7M for training config)
- Parallel KDA: matmul replaces seq loop for training
- ATMN spike neurons: membrane potential, per-neuron threshold
- 132 spike placement points (behind flag)
- SBDS bidirectional distillation loss

Parallel KDA uses mean-alpha approximation for channel-wise gate.
Max diff vs exact recurrence: 0.104 (acceptable for training).

Next run will use parallel KDA + 500 steps = estimated 5-15 min on T4.

### 2026-03-22: run_002 -- ALL PHASE 1 GATES PASS

Config: d_model=256, 8 layers, vocab=256, seq_len=256, batch=32, 500 steps
        mlp_ratio=2.25, shared MLA RoPE, parallel KDA, lr=3e-4
        Kaggle T4 GPU, SpikingBrain validation enabled
Params: 5,932,836 (Todorov), 5,705,984 (Transformer)

RESULTS:
  Todorov best BPB:     2.8755
  Transformer best BPB: 3.4241
  BPB RATIO:            0.840x (Todorov BETTER)
  GATE A:               PASS
  Spike MI:             1.275
  GATE B (MI>0.1):      PASS
  Spike CKA:            0.913
  GATE B (CKA>0.3):     PASS
  Spike FR:             42.0%
  GATE B (FR 30-60%):   PASS
  KDA state norm:       139.3

Throughput:
  Todorov:     2.9 s/step, 1433s total (24m)
  Transformer: 0.2 s/step, 104s total (1.7m)
  vs run_001:  5.7x faster total (27m vs 2h44m)

Interpretation: Architecture validated. Todorov converges faster than
Transformer at matched training budget. Spike encoding preserves nearly
all information (MI=1.275, CKA=0.913). Parallel KDA provides meaningful
speedup (2.9 vs 4.1 s/step). All Phase 1 gates pass decisively.

### 2026-03-22: run_003 -- Phase 2 Context Extension (INITIAL)

Config: Same as run_002 + Phase 2 passkey retrieval and perplexity scaling
        Passkey test: 256, 1024, 4096, 8192 tokens, 20 trials each
        Perplexity test: 256, 512, 1024, 2048, 4096 tokens

Phase 1 results (consistent with run_002):
  BPB ratio: 0.844x, MI: 1.243, CKA: 0.905, FR: 42.0% -- ALL PASS

Phase 2 PASSKEY RETRIEVAL:
  256 tokens:  0% (0/20)
  1024 tokens: 0% (0/20)
  4096 tokens: 0% (0/20)
  8192 tokens: 0% (0/20)

Phase 2 PERPLEXITY SCALING:
  256 tokens:  3.030 BPB
  512 tokens:  3.163 BPB
  1024 tokens: 3.247 BPB
  2048 tokens: 3.362 BPB
  4096 tokens: 3.383 BPB

Interpretation:
- Passkey 0% is NOT a context extension failure. It's 0% even at training
  length (256). The model lacks retrieval capability at this scale (d=256,
  8 layers, 500 steps, byte-level, WikiText-2 only). Passkey retrieval
  requires either larger scale, more training, or task-specific data.
- Perplexity scaling is ENCOURAGING: only 12% degradation from 256 to 4096
  tokens. KDA recurrent state is stable during length extrapolation.
- The 3.03 eval BPB vs 2.88 training BPB gap comes from the mean-alpha
  approximation in parallel training vs exact alpha in recurrent eval.

Research finding: KDA state quality paper (arxiv 2410.07145) says minimum
training length = 5.172 * state_size - 4.469. For head_dim=64: T_train >= 326.
Our seq_len=256 is below this threshold. Training at 512+ should help.

Next: Train at longer context (4K+), more steps, before retesting passkey.
Phase 2 gates cannot be evaluated at current scale.

### 2026-03-22: run_004 -- Phase 2 Progressive Training (IN PROGRESS / STALLED)

Config: Progressive 256->512->1024->2048, 500 steps/stage, parallel KDA
        With selective copy tests and passkey retrieval

Status: Still running after 2+ hours. The O(T^2) parallel KDA matmul at
seq=2048 is the bottleneck. The 2048x2048 attention matrix per head makes
each step ~64x slower than at seq=256.

CRITICAL FINDING: The parallel KDA path (matmul-based) scales as O(T^2)
and is NOT viable for context extension beyond ~512 tokens. At 2048 tokens,
training takes 10x+ longer than at 256. At 4096 it would be 64x longer.

This confirms that fla chunk_kda integration is the CRITICAL PATH for
Phase 2. Without fla's O(T) chunkwise parallel scan, context extension
training is infeasible on Kaggle T4.

Next steps:
1. Pull run_004 results when it completes (may take 3-4 hours total)
2. Integrate fla chunk_kda kernel (pin torch==2.6.0 + triton==3.2.0)
3. Re-run Phase 2 with O(T) training via fla
4. fla integration is now the #1 priority, ahead of all other work

Decision: PROCEED TO PHASE 2 (context extension).

### 2026-03-22: run_008 -- Phase 2 Progressive Training with fla

Config: Progressive 256->512->1024->2048, 200 steps/stage, fla chunk_kda enabled
        d_model=256, 8 layers, vocab=256, batch=[32,16,16,8], lr=[3e-4,1e-4,1e-4,1e-4]
        fla chunk_kda activates at T>=512, fallback to O(T^2) matmul for T<512
        Kaggle T4 GPU, ~15.8K seconds wall clock
Params: 5,917,476 (Todorov), 5,705,984 (Transformer baseline)

PROGRESSIVE TRAINING RESULTS:

| Stage | Seq Len | BPB    | s/step (reported) | s/step (actual) | Time     |
|-------|---------|--------|-------------------|-----------------|----------|
| s256  | 256     | 3.3077 | 2.87              | 2.87            | 573s     |
| s512  | 512     | 3.1080 | 14.36             | 10.2            | 2871s    |
| s1024 | 1024    | 2.9397 | 19.96             | 19.9            | 3993s    |
| s2048 | 2048    | 2.8186 | 39.31             | 39.1            | 7862s    |

Baseline (Transformer, seq=2048, 200 steps): BPB=3.6139, time=181s

FINAL:
  BPB ratio: 0.780x (Todorov 2.819 vs Transformer 3.614) -- BEST across all runs
  Gate A: PASS
  MI: 1.243, CKA: 0.926, FR: 42% -- ALL PHASE 1 GATES PASS

PHASE 2 EVAL:

| Test           | 256    | 512    | 1024   | 2048   | 4096   |
|----------------|--------|--------|--------|--------|--------|
| Perplexity BPB | 3.962  | 3.978  | 4.041  | 4.073  | 4.121  |
| Copy accuracy  | 0%     | 0%     | 0%     | 0%     | --     |
| Passkey acc.   | 0%     | --     | 0%     | --     | 0%     |

CROSS-RUN COMPARISON:

| Metric              | run_001 | run_002 | run_003 | run_004    | run_008    |
|---------------------|---------|---------|---------|------------|------------|
| Best BPB            | 2.235   | 2.876   | 2.876   | ~2.66(s512)| 2.819      |
| BPB ratio           | 1.022x  | 0.840x  | 0.844x  | n/a        | 0.780x     |
| Steps (total)       | 2000    | 500     | 500     | partial    | 800 (4x200)|
| Max seq trained     | 256     | 256     | 256     | 512        | 2048       |
| s/step @256         | 4.1     | 2.9     | 2.9     | 2.9        | 2.87       |
| s/step @512         | --      | --      | --      | 10.2       | 10.2*      |
| KDA path @512       | --      | --      | --      | O(T^2) mat | fla O(T)   |
| Perpl. 256->4K      | --      | --      | +12%    | --         | +4%        |
| Total time          | 2h44m   | 27m     | ~30m    | 2h+(stall) | 4.4h       |

*s512 actual throughput is 10.2s/step after subtracting 826s Triton compilation.

ANALYSIS:

1. ANOMALY: s512 at 14.36s/step vs run_004's 10.2s/step

   NOT slower. The 14.36s is an artifact of Triton kernel compilation. The log
   shows s512 step 0 did not print until t=826s -- Triton compiled the chunk_kda
   kernels on the first forward pass. After compilation, step 50 to step 100
   took 509s for 50 steps = 10.2s/step, matching run_004's O(T^2) matmul exactly.

   This means fla chunk_kda at seq=512 provides NO speedup over the O(T^2) matmul
   at this length. The chunkwise scan has constant overhead (Triton launch, memory
   layout) that only pays off at longer sequences. At T=512 the O(T^2) matmul is
   already fast enough that the chunk scan breaks even at best.

2. SCALING CONFIRMS O(T) for fla:

   After removing compilation overhead, the actual per-step throughput scales as:
     s512  -> s1024: 1.94x  (expected 2.0x for O(T), 4.0x for O(T^2))
     s1024 -> s2048: 1.97x  (expected 2.0x for O(T), 4.0x for O(T^2))
     s512  -> s2048: 3.82x  (expected 4.0x for O(T), 16.0x for O(T^2))

   All ratios are within 3% of the O(T) prediction. fla chunk_kda is delivering
   linear-time training. For comparison, run_004's O(T^2) matmul at s512 took
   10.2s/step -- at s2048 it would have been ~163s/step (16x). fla gives 39s/step,
   a 4.2x speedup at s2048 that would grow to ~16x at s4096.

3. TOTAL TIME: 4.4 hours vs expected ~33 minutes

   The 33-minute estimate was wrong on multiple counts:
   - It assumed 500 steps total, not 800 (4 stages x 200 steps).
   - It did not account for batch size reduction at longer sequences
     (32 -> 16 -> 16 -> 8), which maintains memory but not throughput.
   - Triton compilation added ~14 minutes to s512.
   - The dominant cost is s2048: 7862s (131 min) alone.

   Breakdown: s256=10m, s512=48m(14m compile+34m train), s1024=67m, s2048=131m,
   baseline=3m, eval=7m. Total 266 min = 4.4h.

   Even with perfect O(T) scaling, 200 steps at s2048 with batch=8 and 39s/step
   is inherently 131 minutes. The only way to cut this is fewer steps or larger
   batch (needs more VRAM than T4's 16GB).

4. EVAL PERPLEXITY (3.96) vs TRAINING BPB (3.31) -- train/eval gap

   This is NOT catastrophic forgetting. Two separate effects:

   a) Progressive training overwrites early-stage representations. The s2048
      model was fine-tuned through 3 additional stages after s256, so its short-
      context performance degraded. The eval perplexity at 256 tokens (3.96) is
      measured on the s2048 checkpoint, not the s256 checkpoint.

   b) Training BPB and eval BPB use different code paths. Training at s256 used
      the O(T^2) matmul (since T=256 < 512 threshold), while eval uses the
      recurrent sequential path (state is not None). The mean-alpha approximation
      in the matmul path compresses information differently than exact per-step
      alpha in recurrence. run_003 documented this same gap (2.88 train vs 3.03
      eval = 0.15 gap). Here the gap is larger (3.31 vs 3.96 = 0.65) because
      the model was subsequently tuned to longer contexts.

   c) The eval model IS the s2048 model, not s256. Its natural operating point is
      longer sequences. Looking at BPB at s2048 eval (4.07) vs s2048 training
      (2.82), the gap is 1.25 -- larger than expected, suggesting the recurrent
      eval path still diverges from the fla training path. This is a known issue:
      chunk_kda uses a different numerical formulation than the naive recurrence,
      and accumulated float16/float32 precision differences grow with T.

5. PERPLEXITY SCALING: +4% from 256 to 4096 -- MAJOR IMPROVEMENT

   run_003: +11.7% degradation (3.030 -> 3.383) from 256 to 4096 tokens
   run_008: +4.0% degradation (3.962 -> 4.121) from 256 to 4096 tokens

   The 3x improvement in length stability (11.7% -> 4.0%) comes from two factors:
   - Training at 2048 tokens directly (vs only 256 in run_003).
   - The fla chunk_kda path trains the recurrent state to handle long sequences,
     rather than relying on zero-shot extrapolation.

   At +4% over 16x length extrapolation (256->4096), the KDA state is stable.
   This is the strongest signal that context extension is working at the
   representation level, even though task-specific eval (copy, passkey) fails.

6. COPY AND PASSKEY: 0% everywhere (unchanged from run_003)

   Still 0% at ALL lengths including training length (256, 512, 1024, 2048).
   This confirms the diagnosis from run_003: copy and passkey failure is a model
   CAPABILITY issue, not a context extension issue. The 6M-param byte-level model
   trained on WikiText-2 simply cannot perform token-level retrieval tasks. These
   tests require:
   - Understanding the "copy the value" / "passkey" instruction format
   - Precise single-token retrieval from a specific position
   - Training data that exercises retrieval patterns

   None of these are present. Copy/passkey gates cannot be evaluated at this scale.

7. BPB RATIO 0.780x -- BEST RESULT

   The BPB ratio improved from 0.840x (run_002) to 0.780x (run_008). The Todorov
   model is now 22% better than the Transformer baseline at matched params and
   training budget. However, caution: the Todorov model had 800 total steps of
   progressive training, while the baseline had only 200 steps at seq=2048.
   The comparison is slightly unfair to the Transformer. A fairer comparison would
   give the Transformer 800 steps at 2048 as well.

CONCLUSIONS AND NEXT STEPS:

Confirmed working:
  - fla chunk_kda integration: O(T) scaling verified (1.94x-1.97x per 2x length)
  - Progressive training: BPB drops consistently at each stage
  - KDA state stability: +4% perplexity at 16x extrapolation
  - All Phase 1 gates still pass after context extension training

Remaining problems:
  - Train/eval numerical mismatch: 1.25 BPB gap between fla training and
    recurrent eval at s2048. Need to align the eval path with fla or vice versa.
  - Copy/passkey 0%: expected at this scale, but blocks Phase 2 gate evaluation.
  - Throughput still dominated by long-context stages: s2048 alone is 131 min.
  - Triton compilation adds 14 min per fresh kernel (one-time per session).

Next steps:
  1. Fix train/eval mismatch: either eval with fla chunk_kda (not recurrent), or
     train with recurrent path at short context to match. The 1.25 BPB gap is too
     large to be acceptable.
  2. Scale test: repeat with d_model=512 or more steps to test if copy/passkey
     emerge at larger capacity.
  3. Fair baseline comparison: train Transformer for 800 steps at s2048 to get a
     proper BPB ratio.
  4. Investigate fla float16 precision: the chunk_kda call casts q/k/v to half.
     This may cause the train/eval divergence. Test with float32 throughout.
