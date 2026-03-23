# Status Board

## Current Phase: 2 (Context Extension)

## Status: run_008 COMPLETE -- 1 of 3 Phase 2 gates PASS

## Phase 1: COMPLETE -- ALL GATES PASS

| Gate             | Result   | Status |
|------------------|----------|--------|
| bpb_threshold    | 0.840x   | PASS   |
| spike_mi         | 1.275    | PASS   |
| spike_cka        | 0.913    | PASS   |
| spike_firing_rate| 42.0%    | PASS   |

## Phase 2: IN PROGRESS (run_008 complete)

Prior Phase 2 runs (005-007) focused on fla integration:
- run_005: fla installed successfully, but FP32 and 10-min Triton JIT
- run_006: FP16 fix applied, stopped by user
- run_007: hybrid threshold added, stopped by user

run_008 COMPLETE: first full progressive run with fla + state fix.
- Progressive BPB: 3.31 (256) -> 3.11 (512) -> 2.94 (1024) -> 2.82 (2048)
- Perplexity stability: +4.0% from 256 to 4096 (PASSES <20% gate)
- Selective copy: 0% at all lengths (FAILS >60% gate)
- Passkey retrieval: 0% at all lengths
- Spike health: MI=1.243, CKA=0.926, FR=41.9%
- BPB ratio vs transformer: 0.780x (Todorov better)

CRITICAL BUG FOUND (pre-008): matmul path state_approx only captured last
timestep, not accumulated state. FIXED in run_008 with full accumulation loop.

## Phase 2 Revised Gates (6M scale)

| Gate                | Threshold                            | run_008 Result                  | Status  |
|---------------------|--------------------------------------|---------------------------------|---------|
| selective_copy_1k   | >60% at 1024 token distance          | 0% at all lengths               | FAIL    |
| perplexity_stable   | <20% BPB increase from 256 to 4096   | +4.0% (3.962 -> 4.121)          | PASS    |
| mla_cache_linear    | Cache scales linearly with context   | Not yet measured                | PENDING |

## run_008 Results Summary

Training (Kaggle T4, fla chunk_kda, 200 steps/stage):

| Stage | Seq Len | BPB    | s/step | Time (s) |
|-------|---------|--------|--------|----------|
| s256  | 256     | 3.3077 | 2.87   | 573      |
| s512  | 512     | 3.1080 | 14.36  | 2,871    |
| s1024 | 1024    | 2.9397 | 19.96  | 3,993    |
| s2048 | 2048    | 2.8186 | 39.31  | 7,862    |

Baseline transformer: BPB=3.6139, 200 steps, 181s (0.9s/step)
Total Todorov training: 15,299s (~4h 15m)
Total wall clock (incl. eval): ~4h 22m
