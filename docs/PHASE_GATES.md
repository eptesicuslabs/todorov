# Phase Gates

historical note (2026-04-22): this gating plan predates the curriculum pivot. it should be read as archival phase logic, not as the active program surface.

## Phase 0: Foundation

| Gate             | Description                              | Status  |
|------------------|------------------------------------------|---------|
| tests_pass       | All unit tests pass                      | PASS    |
| memory_budget    | Memory within 5% of theoretical          | PARTIAL |
| parameter_count  | ~300M parameters                         | PARTIAL |

## Phase 1: Language Modeling Baseline -- COMPLETE

| Gate             | Description                                      | Result                  | Status |
|------------------|--------------------------------------------------|-------------------------|--------|
| bpb_threshold    | BPB within 1.5x of Transformer baseline          | 0.840x (run_002)        | PASS   |
| spike_mi         | MI > 0.1                                         | 1.275 (run_002)         | PASS   |
| spike_cka        | CKA > 0.3                                        | 0.913 (run_002)         | PASS   |
| spike_firing_rate| Firing rate 30-60%                               | 42.0% (run_002)         | PASS   |

Kill condition: BPB > 3x Transformer baseline after hyperparameter search.

## Phase 2: Context Extension -- COMPLETE (2/3 PASS, selective copy deferred)

| Gate                | Description                                      | Result                      | Status |
|---------------------|--------------------------------------------------|-----------------------------|--------|
| selective_copy_1k   | >60% at 1024 token distance                      | 0% (run_008)                | FAIL   |
| perplexity_stable   | <20% BPB increase from 256 to 4096               | +4.0% (run_008)             | PASS   |
| mla_cache_linear    | MLA cache grows linearly with context             | not measured                | PENDING|

Original gates (passkey_32k, passkey_128k) replaced with 6M-scale gates after run_003 analysis.

## Phase 3: Spatial Module -- historical snapshot (later partially qualified)

| Gate               | Description                                    | Result                                  | Status |
|--------------------|------------------------------------------------|-----------------------------------------|--------|
| spatial_classify   | 3D shape classification outperforms Transformer| GP 30.0% vs Transformer 25.0% (run_009)| PASS   |
| spatial_dynamics   | n-body dynamics outperforms Transformer        | GP MAE=51.55 vs BL MAE=72.70 (run_009) | PASS   |
| equivariance_test  | Equivariance error <5% at 60-degree rotation   | 1.34e-07 (run_009), later judged vacuous| QUALIFIED |
| language_no_degrade| Language BPB not degraded >10% with GP         | -18.8% at run_009, later judged confounded | QUALIFIED |

## Phase 4: Multimodal Fusion (historical planned phase)

| Gate               | Description                                    | Status  |
|--------------------|------------------------------------------------|---------|
| cross_modal        | Cross-modal transfer demonstrated              | PENDING |
| joint_embedding    | Joint embedding quality verified               | PENDING |

## Phase 5: Scale and Optimize (historical planned phase)

| Gate               | Description                                    | Status  |
|--------------------|------------------------------------------------|---------|
| benchmark          | Competitive with Phi-3-mini class              | PENDING |
| int8_deployed      | INT8 quantization verified                     | PENDING |
