# status board

## current phase: neural machine research (neuroloc)

## status: todorov architecture paused. neural machine spec at neuroloc/spec/blueprint.md

## phase 5: runs 010-011 complete

### run_011: full architecture (KDA+Mamba3+MLA, 6:1:1)

h200, 280m params, full architecture (18 KDA + 3 Mamba3 + 3 MLA).
bpb 2.592, ratio 0.722x (27.8% better than transformer baseline).
spike mi 1.246, cka 0.802. fr measurement bug from gradient
checkpointing (reported 0%, actual ~41%). 3/4 gates pass.
mamba3 sequential scan dominates training time (~10s/step overhead).
atmn too slow at 280m with gradient checkpointing (30+ min/step).

### run_010: KDA+MLA only (no Mamba3)

h200, 267m params, kda+mla only (no mamba3). fla disabled (chunk_kda NaN at
d_model=1024). mamba3 dropped (sequential scan ~15s/step at T=2048, not viable
without parallel scan kernel).

bpb 2.375, ratio 0.663x (33.7% better than transformer baseline at matched
scale). spike health: mi=1.168, cka=0.732, fr=40.8%.

| gate             | threshold                    | run_010 result                  | status |
|------------------|------------------------------|---------------------------------|--------|
| bpb_ratio        | < 1.0 (beat transformer)     | 0.663x (2.375 vs 3.583)        | pass   |
| spike_mi         | > 0.1                        | 1.168                           | pass   |
| spike_cka        | > 0.3                        | 0.732                           | pass   |
| spike_fr         | 30-60%                       | 40.8%                           | pass   |

next: fix fla chunk_kda NaN at d=1024, add mamba3 parallel scan, re-run phase
5 baseline with full 3:1 architecture.

## phase 3: complete -- all gates pass

## Phase 2: COMPLETE -- 2/3 gates PASS (selective copy deferred to Phase 5)

## Phase 1: COMPLETE -- ALL GATES PASS

| Gate             | Result   | Status |
|------------------|----------|--------|
| bpb_threshold    | 0.840x   | PASS   |
| spike_mi         | 1.275    | PASS   |
| spike_cka        | 0.913    | PASS   |
| spike_firing_rate| 42.0%    | PASS   |

## Phase 2 Summary (run_008)

- Progressive BPB: 3.31 (256) -> 3.11 (512) -> 2.94 (1024) -> 2.82 (2048)
- Perplexity stability: +4.0% from 256 to 4096 (PASSES <20% gate)
- Selective copy: 0% at all lengths (FAILS >60% gate -- deferred to Phase 5)
- BPB ratio vs transformer: 0.780x (Todorov better)

## Phase 2 Revised Gates (6M scale)

| Gate                | Threshold                            | run_008 Result                  | Status  |
|---------------------|--------------------------------------|---------------------------------|---------|
| selective_copy_1k   | >60% at 1024 token distance          | 0% at all lengths               | FAIL    |
| perplexity_stable   | <20% BPB increase from 256 to 4096   | +4.0% (3.962 -> 4.121)          | PASS    |
| mla_cache_linear    | Cache scales linearly with context   | Not yet measured                | PENDING |

## Phase 3: COMPLETE (run_009)

GP self-interaction enabled in SwiGLU (spatial_mode=True).
Training on mixed data: 50% WikiText-2 + 25% 3D shape + 25% n-body.

| Gate               | Threshold                              | run_009 Result                         | Status |
|--------------------|----------------------------------------|----------------------------------------|--------|
| spatial_classify   | outperform Transformer                 | GP 30.0% vs Transformer 25.0%         | PASS   |
| spatial_dynamics   | lower MAE than Transformer             | GP MAE=51.55 vs Transformer MAE=72.70  | PASS   |
| equivariance_test  | <5% error at 60-degree rotation        | error=1.34e-07 at 60 degrees           | PASS   |
| language_no_degrade| BPB not degraded >10% with GP enabled  | -18.8% (GP improves language)          | PASS   |

## run_009 Results Summary

Training (Kaggle T4, mixed data, GP enabled):

| Metric             | GP (Todorov)  | No-GP         | Transformer   |
|--------------------|---------------|---------------|---------------|
| Shape classify     | 30.0%         | --            | 25.0%         |
| N-body MAE         | 51.55         | --            | 72.70         |
| Equivariance err   | 1.34e-07      | --            | --            |
| Language BPB       | 3.009         | 3.707         | --            |
| Training time (s)  | 1,451         | 573           | 42            |
| Params             | 6,015,780     | 5,917,476     | 5,705,984     |

Per-class shape classification:

| Class        | GP     | Transformer |
|--------------|--------|-------------|
| sphere       | 80%    | 100%        |
| cube         | 0%     | 0%          |
| tetrahedron  | 16%    | 0%          |
| torus        | 24%    | 0%          |

Spike health: MI=1.311, CKA=0.907, FR=42.1%, dead=0%
Total wall clock: ~38 min on Kaggle T4

Bug found during implementation: src/layers/swiglu.py spatial_mode had latent
shape mismatch (GP output d_model added to hidden_dim tensor). Fixed in train.py:
GP residual applied after down projection.
