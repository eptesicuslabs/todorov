# Changelog

historical note (2026-04-22): this changelog currently reflects the pre-curriculum architecture era. current state and backlog status live in `neuroloc/wiki/PROJECT_PLAN.md` and `docs/STATUS_BOARD.md`.

## 2026-03-23: run_009 COMPLETE -- Phase 3 ALL GATES PASS

- run_009 (GP spatial validation, mixed data, Kaggle T4): COMPLETE
  - GP self-interaction enabled in SwiGLU (spatial_mode=True)
  - Mixed training data: 50% WikiText-2 + 25% 3D shape + 25% n-body
  - Shape classification: GP 30.0% vs Transformer 25.0% -- PASS
    - Per-class GP: sphere=80%, cube=0%, tetrahedron=16%, torus=24%
    - Per-class Transformer: sphere=100%, cube/tetrahedron/torus=0%
    - GP generalizes across classes; Transformer only learns majority class
  - N-body dynamics: GP MAE=51.55 vs Transformer MAE=72.70 (29% better) -- PASS
  - Equivariance: error=1.34e-07 at 60-degree rotation -- PASS
  - Language: BPB with GP=3.009, without GP=3.707, degradation=-18.8% -- PASS
    - GP actually IMPROVES language modeling (negative degradation)
  - Spike health: MI=1.311, CKA=0.907, FR=42.1%, dead=0%
  - Training time: GP 1451s, NoGP 573s, Transformer 42s
  - Params: GP=6,015,780, NoGP=5,917,476, Transformer=5,705,984
  - Total wall clock: ~38 min on Kaggle T4
  - BUG FIX: swiglu.py spatial_mode latent shape mismatch, fixed in train.py
- Phase 3 gate status: 4/4 PASS
- Next: Phase 5 (scale and optimize)

## 2026-03-23: run_008 COMPLETE -- Phase 2 first full progressive run

- run_008 (fla chunk_kda, progressive 256->2048, 200 steps/stage): COMPLETE
  - CRITICAL FIX: matmul path state accumulation (was only last timestep)
  - fla available: True, hybrid KDA (matmul <512, fla >=512)
  - Progressive BPB: 3.31 (256) -> 3.11 (512) -> 2.94 (1024) -> 2.82 (2048)
  - Final BPB ratio: 0.780x (Todorov 2.819 vs Transformer 3.614)
  - Spike health: MI=1.243, CKA=0.926, FR=41.9%, dead=0.0%
  - Phase 2 evals:
    - Perplexity stability: +4.0% from 256 to 4096 -- PASS (<20%)
    - Selective copy: 0% at all lengths (256/512/1024/2048) -- FAIL (>60%)
    - Passkey retrieval: 0% at all lengths (256/1024/4096)
  - Total training time: 15,299s (~4h 15m), wall clock ~4h 22m
  - Per-stage: s256 2.9s/step, s512 14.4s/step, s1024 20.0s/step, s2048 39.3s/step
- Phase 2 gate status: 1/3 PASS (perplexity_stable), 1/3 FAIL (selective_copy), 1/3 PENDING (mla_cache)
- Next: investigate copy/passkey 0% -- model may need task-specific training or longer runs

## 2026-03-22: Phase 1 ALL GATES PASS

- run_002 (parallel KDA, 500 steps): ALL GATES PASS
  - BPB ratio: 0.840x (Todorov 2.876 BETTER than Transformer 3.424)
  - Spike MI: 1.275 (threshold > 0.1)
  - Spike CKA: 0.913 (threshold > 0.3)
  - Spike FR: 42.0% (range 30-60%)
  - Total time: 27 min (vs 2h44m for run_001)
- Fixed C-01: eliminated diagnostic sequential loop in KDA parallel path
- Fixed C-02: firing rate averaged across all spike points
- Fixed C-04: MLA RoPE shared in training script (was already fixed in src/)
- Phase 1 COMPLETE. Proceeding to Phase 2 (context extension)

## 2026-03-22: Phase 1 Gate A PASS (run_001)

- First training run completed: Todorov BPB 2.235, Transformer BPB 2.187
- BPB ratio: 1.022x -- PASSES Gate A threshold of 1.5x
- Spike firing rate: 41.4% -- PASSES Gate B range [30-60%]
- 0.0% dead neurons
- KDA state norm: 107.0
- Throughput: 4.1s/step (Todorov) vs 0.2s/step (Transformer) -- 20x gap
- Fixed json.dump numpy bool serialization crash
- Built scripts/kaggle_exec.py for single-call Kaggle GPU execution
- Remaining: MI/CKA validation (Gate B), parallel KDA GPU validation

## 2026-03-22: Post-Phase-0 Architecture Revision

- Fixed MLA RoPE: shared across heads (d_R per token, not d_R*num_heads)
  Cache at 128K: 576 MB -> 240 MB (FP32), ~120 MB at FP16 (matches spec)
- Reduced mlp_ratio from 2.75 to 2.25: params 349M -> 312M (closer to 300M target)
- Implemented ATMN spike neurons (src/spikes/atmn_spike.py):
  membrane potential dynamics, per-neuron V_th = exp(a), temporal state
- Expanded spike placement from 36 to 132 points (spike_all_projections=True):
  KDA Q/K/V/O + Mamba3 in/out + MLA Q/KV + SwiGLU gate/up
- Upgraded distillation to bidirectional SBDS-style:
  forward KL (0.2) + reverse KL (0.7) + pre-norm feature alignment (0.1)
- Created Colab notebook for fast iteration (real-time output vs blind Kaggle)
- Fixed Kaggle pipeline: T4 accelerator, CUDA fallback, JSON log parser
- Added 13 ATMN tests, total 76/76 passing
- Research: MAR/ATMN, SBDS, fla updates, hybrid architectures, context extension

## 2026-03-22: Initial Foundation (Phase 0)

- Scaffolded full repository structure (60+ files)
- Ported G(3,0,1) algebra from Echoloc with GATr sparse einsum optimization
- Ported TernarySpike and SpikingBrain from Gerhard
- Ported Kaggle autoresearch pipeline from Echoloc
- Implemented KDA layer with channel-wise gating and delta-rule recurrence
- Implemented Mamba-3 SISO with trapezoidal discretization and complex state
- Implemented MLA with compressed KV cache and decoupled RoPE
- Implemented SwiGLU MLP with optional GP self-interaction
- Assembled full 24-layer Todorov model (KDA/Mamba3/MLA pattern)
- Implemented training infrastructure: loss, optimizer, evaluator
- Implemented utility modules: memory budget, convergence, ERF
- Wrote and passed 63 tests across 8 test files
- Measured memory budget: 350M params, 363 MB at 4K, 921 MB at 128K
- Identified MLA cache inflation issue (per-head RoPE vs shared)
