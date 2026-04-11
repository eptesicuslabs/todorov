# status board

## current phase: neural machine research (neuroloc)

## status: god_run_v2 complete (2026-04-12). F1 math fix did NOT recover retrieval. passkey 0/100 at 256/1024/4096 (CI 3.7%), copy 0/100 at 256/512/1024/2048. bundle-is-broken branch of decision rule. blueprint.md revised with corrected run ordering (dense keys mandatory). run 1 baseline (dense k/v) launching next.

## god_run_v2: god_machine.py re-run with all 17 F + 14 G prosecutor fixes (2026-04-12)

h200, 283m params, 4000 steps on fineweb-edu byte-level, 131m tokens, 59 min runtime.
all 17 F1-F17 + 14 G1-G14 prosecutor findings fixed before launch. critical fix: F1
replaced `torch.sigmoid(log_alpha_eff)` with `torch.exp(log_alpha_eff)` so recurrent path
matches FLA's `exp(g)` convention for log-space gates (prior broken math gave ~28% wrong
alpha_eff at default init). F2 gated `running_state_norm` EMA on `self.training`. F3/F5
merged spike_stats and val_result into jsonl via loop (fixed the class-wide regression of
the step-logger bug). F10 low-rank factorized imag_filter/pc_head (d→64→d). F11 uint8
ByteDataset. 12 more findings covering smoke tests, resume correctness, load errors,
dead code, misleading metrics. cpu oracle verifies alpha_eff math passes before launch.

**training: final val_bpb 1.4453 (+0.050 vs v1's 1.3950), 0.404x vs transformer**

**eval: retrieval 0% at every tested length**

| task | result (100 trials) |
|---|---|
| passkey @256 | 0/100 (CI 3.7%) |
| passkey @1024 | 0/100 (CI 3.7%) |
| passkey @4096 | 0/100 (CI 3.7%) |
| copy @256 | 0/100 (CI 3.7%) |
| copy @512 | 0/100 (CI 3.7%) |
| copy @1024 | 0/100 (CI 3.7%) |
| copy @2048 | 0/100 (CI 3.7%) |

perplexity at length: 1.9254 → 1.8941 → 1.5373 → 1.4776 → 1.4192 (monotonic, attention
uses context). delta state structure probe: mean_structure_ratio 0.977, pairwise_cos
0.003 (statistically noise, not trained memory).

**verdict: F1 math fix was not the root cause. the 5-feature bundle intrinsically destroys
verbatim memory.** external review (2026-04-12) identified 8 root causes. the two critical
ones: (1) k-WTA 20% on keys destroys the address space (only ~13/64 key dims survive,
Hopfield capacity drops from ~9 to ~2 patterns per head); (2) delta erasure with sparse
keys leaves ghost content in the zeroed dimensions. 6 supporting causes: imagination probe
creates a gradient bypass that competes with the delta memory, BCM EMA half-life 69 steps
is too slow, multi-compartment SwiGLU halves effective width, PC head creates loss drag,
FLA/recurrent numerical drift compounds during chunked eval, q/k normalization timing
differs between paths. full detail in `memory/project_v2_diagnosis.md`.

**revised path forward** (blueprint.md updated):
- **run 1 baseline dense**: dense k/v, delta erasure on, no BCM, no multi-compartment,
  no imagination, no PC head. HARD GATE: nonzero passkey at 256. launching next on h200.
- run 2: add k-WTA 50% on values only (keys stay dense)
- run 3: add BCM with momentum 0.95, init running_state_norm to 0.01
- not added until validated: imagination, multi-compartment, PC head, any k-WTA on keys

## god_run: god_machine.py first run (2026-04-11)

h200, 283m params (282,953,496), 4000 steps on fineweb-edu byte-level, 131,072,000 tokens, seed 42.
all 5 blueprint features active: k-wta 20% rate-coded compression, delta-rule erasure,
bcm-adaptive alpha (gamma=0.3), multi-compartment swiglu (k=4), compressed attention via sdpa.
plus always-on imagination probe (learned query into delta state with gated residual, now
low-rank factorized to ~131k params per layer) and per-layer predictive coding diagnostic head.

val_bpb 1.3950 (final). bpb_ratio 0.390x vs 3.58 transformer baseline at matched pipeline.
smooth monotonic decrease: 2.381 → 2.07 → 1.94 → 1.87 → 1.81 → 1.75 → 1.71 → 1.67 → 1.62 →
1.57 → 1.52 → 1.50 → 1.48 → 1.46 → 1.44 → 1.43 → 1.42 → 1.41 → 1.40 → 1.3950.
training loss final 0.9535 at step 3950. throughput ~45,500 tok/s steady. total 3166s (~53 min).
firing rate 0.200 exactly throughout (k-wta target met). no dead neurons.

**retrieval failed at every tested length (n=20 per cell, 95% wilson upper ~14%):**

passkey  @256: 0/20  passkey  @1024: 0/20  passkey  @4096: 0/20
copy     @256: 0/20  copy     @512:  0/20  copy     @1024: 0/20  copy @2048: 0/20

perplexity-at-length (monotonic decrease, attention path uses context):
bpb@256=1.9354  bpb@512=1.8437  bpb@1024=1.4909  bpb@2048=1.4110  bpb@4096=1.3751

delta state structure probe (closed-gate readout with novel keys, NOT image generation):
mean_structure_ratio=0.981, mean_pairwise_cos=-0.003, random_pairwise_cos=0.000.
state is near-orthogonal across 24 delta layers. this is NOT the structured-interpolation
signature (cosine ~0.93) that exp_008 reported. the delta memory accumulated high frobenius
norm but pairwise-orthogonal state that functions as noise, not content-addressable storage.

**diagnosis**: the compressed-attention+mlp path learned to fit the next-byte distribution
(bpb 1.395). the delta-rule memory state is noise. k-wta 20% + delta erasure + bcm alpha +
imagination probe combined so that verbatim retrieval was destroyed while statistical
distribution-fitting worked. this is exactly the lossy-mechanism failure mode that
`wiki/synthesis/compression_beyond_quantization.md` predicts: preserved statistical fit,
destroyed verbatim memory.

**17 prosecutor findings F1-F17 applied before re-run:**

- f1 (p0): bcm train/eval path divergence. recurrent path computed live-state alpha_eff
  per timestep; fla path used running_state_norm buffer. fixed by aligning recurrent path
  to use `_effective_log_alpha` which reads running_state_norm.
- f2 (p0): running_state_norm buffer mutated during eval. fixed with self.training gate.
- f3 (p1): history dict was a second cherry-pick site (class-level regression of the
  step-logger bug). fixed via setdefault loop that merges all spike_stats keys.
- f4 (p1): collect_god_metrics produced length-inconsistent per_layer arrays. fixed by
  pre-allocating [None]*n_layers lists.
- f5 (p1): other metrics_logger sites still cherry-picked val_result. fixed via loop merge.
- f6 (p1): passkey/copy 20 trials was too few (95% wilson upper 14% for 0/20). fixed to 100.
- f7 (p2): load_state_dict strict=False swallowed errors. fixed to raise on missing keys.
- f8 (p2): compartment guard was dead. fixed to check correct aux key name.
- f9 (p2): smoke-test required_god_keys was a hardcoded 9-key list. fixed to derive from aux.
- f10 (p2): imag_filter 1m params * 24 layers = 24m always-active dead weight. fixed to
  low-rank factorization (~131k params/layer). same for pc_head. added post-val warning
  if imag_ratio_mean < 0.02.
- f11 (p2): bytedataset materialized int64 (8x ram). fixed to keep uint8, cast per-slice.
- f12 (p2): no resume correctness test. added _test_resume_correctness() to smoke_test.
- f13 (p3): dead _parallel_no_erasure code. removed; all non-fla paths go through recurrent.
- f14 (p3): non-causal attn_entropy probe with T<=512 gate. removed.
- f15 (p3): hardcoded zero dead_pct/saturated_pct/per_layer_dead_count. removed.
- f16 (p3): dead imports (field), SEED constant, TernaryQuantizer, AdaptiveSpike. removed.
- f17 (p3): per-timestep state.norm O(H*D*D). fixed via f1 (alpha_eff hoisted).

**renamed**: run_imagination_test → run_delta_state_structure_probe. "imagination" was a
misleading metaphor for what is actually a state-structure probe; the byte-level text model
has no image generation capability.

**class-level telemetry gate**: smoke_test now writes collect_god_metrics output through a
real MetricsLogger to a tempfile, reads it back via json.loads, and asserts every enabled
feature's aux keys round-trip through disk. any future regression that drops metrics fails
smoke test before launch. required_god_keys is derived programmatically from aux.

artifacts: `neuroloc/output/god_run/` (eval_suite.json, metadata.json, stdout.log,
metrics.jsonl, results.json). run card: `neuroloc/output/god_run/run_card.md`.

**what comes next**: re-run god_machine.py with all F1-F17 fixes applied and full
telemetry. key question for re-run analysis: did F1 (bcm path divergence) cause retrieval
failure, or is the 5-feature bundle itself the cause? if retrieval still 0% with F1 fixed,
fall back to blueprint sequential isolation starting with run 1 (ternary spike baseline).

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
