# Run Tracker

historical note (2026-04-22): this file is a pre-curriculum run tracker snapshot. later run history and the curriculum pivot are captured in `neuroloc/wiki/PROJECT_PLAN.md`, `docs/STATUS_BOARD.md`, and `state/program_status.yaml`.

raw numbers from early tracked runs. for later h200 and neural-machine runs, use `neuroloc/wiki/PROJECT_PLAN.md`, `docs/STATUS_BOARD.md`, and `state/program_status.yaml`.

## Phase 1 Runs

### run_001: Kaggle T4, sequential KDA, 2000 steps -- COMPLETE

Config: d_model=256, n_layers=8, vocab=256, seq_len=256, batch=32
        mlp_ratio=2.75, lr=3e-4, warmup=200, KDA K/V spikes, alpha=1.0
Params: 6,719,268 (Todorov), 6,492,416 (Transformer)

TODOROV:

| Step | Train Loss | Val BPB | Elapsed (s) |
|------|-----------|---------|-------------|
| 0    | 5.614     |         | 5           |
| 200  | 2.338     | 3.398   | 810         |
| 400  | 2.043     | 2.927   | 1629        |
| 600  | 1.883     | 2.686   | 2447        |
| 800  | 1.763     | 2.539   | 3265        |
| 1000 | 1.673     | 2.442   | 4083        |
| 1200 | 1.631     | 2.374   | 4901        |
| 1400 | 1.586     | 2.317   | 5719        |
| 1600 | 1.505     | 2.279   | 6537        |
| 1800 | 1.569     | 2.253   | 7356        |
| 2000 | -         | 2.235   | 8186        |

Final: BPB=2.2353, time=8186s, 4.1s/step

TRANSFORMER:

| Step | Train Loss | Val BPB | Elapsed (s) |
|------|-----------|---------|-------------|
| 0    | 5.704     |         | 0           |
| 200  | 2.541     | 3.631   | 40          |
| 400  | 2.400     | 3.414   | 83          |
| 600  | 2.107     | 3.072   | 126         |
| 800  | 1.888     | 2.738   | 170         |
| 1000 | 1.711     | 2.526   | 213         |
| 1200 | 1.586     | 2.390   | 257         |
| 1400 | 1.554     | 2.298   | 300         |
| 1600 | 1.548     | 2.243   | 343         |
| 1800 | 1.524     | 2.206   | 387         |
| 2000 | -         | 2.187   | 434         |

Final: BPB=2.1865, time=434s, 0.2s/step

COMPARISON:
  BPB ratio:      1.022x (Todorov / Transformer)
  Throughput ratio: 18.9x slower
  Gate A:         PASS (1.022 < 1.5)

SPIKE STATS:
  Firing rate:    41.4%
  Dead neurons:   0.0%
  KDA state norm: 107.0
  Gate B (FR):    PASS (0.414 in [0.3, 0.6])

### run_002: Kaggle T4, parallel KDA, 500 steps, SpikingBrain -- COMPLETE

Config: d_model=256, n_layers=8, vocab=256, seq_len=256, batch=32
        mlp_ratio=2.25, shared MLA RoPE, parallel KDA training
        lr=3e-4, warmup=100, KDA K/V spikes, alpha=1.0
Params: 5,932,836 (Todorov), 5,705,984 (Transformer)

TODOROV: Final BPB=2.8755, time=1433s, 2.9s/step
TRANSFORMER: Final BPB=3.4241, time=104s, 0.2s/step

COMPARISON:
  BPB ratio:       0.840x (Todorov BETTER)
  Throughput ratio: 13.8x slower
  Gate A:          PASS

SPIKE VALIDATION:
  MI:              1.275 (threshold > 0.1) PASS
  CKA:             0.913 (threshold > 0.3) PASS
  Firing rate:     42.0% PASS
  Dead neurons:    0.0%
  KDA state norm:  139.3
  All Gate B:      PASS

## Phase 2 Runs

### run_003-007: fla integration attempts (see CHANGELOG for details)

- run_003: Phase 2 probe, passkey 0%, perplexity +12% at 4K (state bug found)
- run_004: Progressive 256->1024, O(T^2) matmul too slow, cancelled at s1024
- run_005: fla first attempt, FP32 slow, 10min Triton JIT compile
- run_006: FP16 fix applied, stopped by user for further fixes
- run_007: hybrid threshold added, stopped by user for state fix

### run_008: Kaggle T4, fla chunk_kda, progressive 256->2048, 200 steps/stage -- COMPLETE

Config: d_model=256, n_layers=8, vocab=256, batch=8, lr=1e-4
        progressive stages: s256->s512->s1024->s2048 (200 steps each)
        hybrid KDA: matmul for seq<512, fla chunk_kda for seq>=512
        FP16 q/k/v in fla path, FP32 gates, F.logsigmoid + 1/sqrt(d) scale
        CRITICAL FIX: full state accumulation in matmul path
Params: 5,917,476 (Todorov), 5,705,984 (Transformer)

TODOROV PROGRESSIVE TRAINING:

| Stage | Seq Len | Steps | Best BPB | Final BPB | Time (s) | s/step |
|-------|---------|-------|----------|-----------|----------|--------|
| s256  | 256     | 200   | 3.3077   | 3.3077    | 573      | 2.87   |
| s512  | 512     | 200   | 3.1080   | 3.1080    | 2,871    | 14.36  |
| s1024 | 1024    | 200   | 2.9397   | 2.9397    | 3,993    | 19.96  |
| s2048 | 2048    | 200   | 2.8186   | 2.8186    | 7,862    | 39.31  |

Total Todorov training: 15,299s (~4h 15m)

TRANSFORMER BASELINE (seq=2048, 200 steps):
  Final BPB:  3.6139
  Time:       181s (0.9s/step)

COMPARISON:
  BPB ratio:       0.780x (Todorov BETTER)
  Gate A:          PASS

SPIKE VALIDATION:
  MI:              1.243 (threshold > 0.1) PASS
  CKA:             0.926 (threshold > 0.3) PASS
  Firing rate:     41.9% PASS
  Dead neurons:    0.0%
  KDA state norm:  86.3
  All Gate B:      PASS

PHASE 2 EVALUATION:

Selective Copy:
| Context | Accuracy | Correct/Total |
|---------|----------|---------------|
| 256     | 0.0%     | 0/20          |
| 512     | 0.0%     | 0/20          |
| 1024    | 0.0%     | 0/20          |
| 2048    | 0.0%     | 0/20          |

Passkey Retrieval:
| Context | Accuracy | Correct/Total |
|---------|----------|---------------|
| 256     | 0.0%     | 0/20          |
| 1024    | 0.0%     | 0/20          |
| 4096    | 0.0%     | 0/20          |

Perplexity Scaling:
| Context | BPB    | Delta from 256 |
|---------|--------|----------------|
| 256     | 3.9617 | --             |
| 512     | 3.9777 | +0.4%          |
| 1024    | 4.0414 | +2.0%          |
| 2048    | 4.0725 | +2.8%          |
| 4096    | 4.1210 | +4.0%          |

PHASE 2 GATES:
  selective_copy_1k (>60%):    0.0% -- FAIL
  perplexity_stable (<20%):    +4.0% -- PASS
  mla_cache_linear:            NOT MEASURED -- PENDING

## Phase 3 Runs

### run_009: Kaggle T4, GP spatial validation, mixed data -- COMPLETE

Config: d_model=256, n_layers=8, vocab=256, seq_len=256, batch=32
        spatial_mode=True (GP self-interaction in SwiGLU)
        Mixed data: 50% WikiText-2, 25% 3D shape, 25% n-body
        Kaggle T4 GPU, ~38 min wall clock
Params: 6,015,780 (GP), 5,917,476 (NoGP), 5,705,984 (Transformer)

SHAPE CLASSIFICATION (4-class):

| Model       | Overall | Sphere | Cube | Tetrahedron | Torus |
|-------------|---------|--------|------|-------------|-------|
| GP (Todorov)| 30.0%   | 80%    | 0%   | 16%         | 24%   |
| Transformer | 25.0%   | 100%   | 0%   | 0%          | 0%    |

N-BODY DYNAMICS:

| Model       | MAE    |
|-------------|--------|
| GP (Todorov)| 51.55  |
| Transformer | 72.70  |

EQUIVARIANCE:
  Rotation: 60 degrees
  Error: 1.34e-07

LANGUAGE BPB:
  With GP:    3.009
  Without GP: 3.707
  Degradation: -18.8% (GP improves language)

SPIKE VALIDATION:
  MI:              1.311 (threshold > 0.1) PASS
  CKA:             0.907 (threshold > 0.3) PASS
  Firing rate:     42.1% PASS
  Dead neurons:    0.0%

TIMING:

| Model       | Time (s) |
|-------------|----------|
| GP (Todorov)| 1,451    |
| NoGP        | 573      |
| Transformer | 42       |

PHASE 3 GATES:
  spatial_classify (GP > Transformer):    30.0% vs 25.0% -- PASS
  spatial_dynamics (GP < Transformer MAE): 51.55 vs 72.70 -- PASS
  equivariance_test (<5% at 60 deg):      1.34e-07 -- PASS
  language_no_degrade (<10% degradation):  -18.8% (improved) -- PASS
