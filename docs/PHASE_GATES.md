# Phase Gates

## Phase 0: Foundation

| Gate             | Description                              | Status  |
|------------------|------------------------------------------|---------|
| tests_pass       | All unit tests pass                      | PASS    |
| memory_budget    | Memory within 5% of theoretical          | PARTIAL |
| parameter_count  | ~300M parameters                         | PARTIAL |

## Phase 1: Language Modeling Baseline

| Gate             | Description                                      | Status  |
|------------------|--------------------------------------------------|---------|
| bpb_threshold    | BPB within 1.5x of Transformer baseline          | PENDING |
| spike_mi         | MI > 0.1                                         | PENDING |
| spike_cka        | CKA > 0.3                                        | PENDING |
| spike_firing_rate| Firing rate 30-60%                               | PENDING |

Kill condition: BPB > 3x Transformer baseline after hyperparameter search.

## Phase 2: Context Extension

| Gate             | Description                                      | Status  |
|------------------|--------------------------------------------------|---------|
| passkey_32k      | Passkey retrieval >95% at 32K                    | PENDING |
| passkey_128k     | Passkey retrieval >80% at 128K                   | PENDING |
| mla_cache_linear | MLA cache grows linearly with context             | PENDING |

## Phase 3: Spatial Module

| Gate               | Description                                    | Status  |
|--------------------|------------------------------------------------|---------|
| spatial_classify   | 3D shape classification outperforms Transformer| PENDING |
| spatial_dynamics   | n-body dynamics outperforms Transformer        | PENDING |
| equivariance_test  | Equivariance error <5% at 60-degree rotation   | PENDING |
| language_no_degrade| Language BPB not degraded >10% with GP         | PENDING |

## Phase 4: Multimodal Fusion

| Gate               | Description                                    | Status  |
|--------------------|------------------------------------------------|---------|
| cross_modal        | Cross-modal transfer demonstrated              | PENDING |
| joint_embedding    | Joint embedding quality verified               | PENDING |

## Phase 5: Scale and Optimize

| Gate               | Description                                    | Status  |
|--------------------|------------------------------------------------|---------|
| benchmark          | Competitive with Phi-3-mini class              | PENDING |
| int8_deployed      | INT8 quantization verified                     | PENDING |
