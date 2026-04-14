# god_machine.py — run card

## what this is

preset-aware run card for `neuroloc/model/god_machine.py`.

the current default and next planned paid run is `NM_PRESET=run1_baseline_noerasure`: dense keys, dense values, no overwrite subtraction, non-fla path, no bcm, no multi-compartment, no imagination probe, and no pc diagnostic head. the hard gate for that branch is any nonzero `passkey@256`, persisted as `retrieval_gate` in `{name}_results.json`.

the archived all-features bundle remains available as `NM_PRESET=god`. it keeps k-wta rate-coded compression, delta-rule erasure, bcm-like adaptive alpha, multi-compartment swiglu, compressed retrieval, the imagination probe, predictive-coding diagnostics, and the god-tier telemetry stack.

the current default branch is run 1 of the sequential isolation protocol in `neuroloc/spec/blueprint.md`. only the archived `god` preset deviates from that protocol by design as a budget-constrained bundle run with no causal attribution per feature.

## environment requirements

- gpu: h200 (sm_90) strongly preferred. sm_80+ works with bf16. sm_75 and below: disable amp (`cfg.amp = False`), no bf16 tensor cores.
- python: 3.10+
- pytorch: 2.6.0+ with cuda 12.x
- datasets: `pip install --break-system-packages datasets`
- disk: ~10 gb free (5 gb fineweb cache + checkpoints + metrics)

for the current no-erasure baseline, the recurrent path is the intended execution path and does not require triton or flash-linear-attention.

for the archived `god` preset or for parity-debug work, add:

- triton: 3.2.0+ (for the fla kernel)
- flash-linear-attention: install via `pip install --break-system-packages flash-linear-attention`

## quick install on fresh pod

```
pip install --break-system-packages -q datasets
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

if you are intentionally running `NM_PRESET=god` or debugging fla parity, add:

```
pip install --break-system-packages -q flash-linear-attention
python -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule; print('fla ok')"
```

## launch command

```
cd /workspace
SMOKE_TEST=1 python3 neuroloc/model/god_machine.py 2>&1 | tee smoke.log
```

if smoke passes, the current baseline launch is:

use a fresh output directory for each benchmark or full launch unless you are intentionally continuing a prior run with `NM_RESUME=1`.

keep `stdout.log` beside `NM_OUTPUT_DIR`, not inside it, so the freshness gate still protects the artifact directory.

```
mkdir -p /workspace/neuroloc/output/run1_baseline_noerasure_bench
NM_PRESET=run1_baseline_noerasure \
NM_OUTPUT_DIR=/workspace/neuroloc/output/run1_baseline_noerasure_bench \
NM_RUN_NAME=run1_baseline_noerasure_bench \
NM_DATASET=fineweb \
NM_MAX_STEPS=20 \
NM_SKIP_VALIDATION=1 \
NM_SKIP_EVAL=1 \
python3 -u neuroloc/model/god_machine.py 2>&1 | tee /workspace/neuroloc/output/run1_baseline_noerasure_bench.stdout.log
```

that benchmark writes `/workspace/neuroloc/output/run1_baseline_noerasure_benchmark_manifest.json`. keep the full launch in the same output root; the executable now refuses the official full run if that manifest is missing, mismatched, not from a fineweb-backed sm_90+ benchmark, or if the full launch is attempted on a different benchmarked hardware profile. this is a local provenance guard against accidental mismatch and stale replay, not cryptographic attestation against an operator who can rewrite local files.

only after that benchmark finishes cleanly on h200 should the full baseline launch run:

```
mkdir -p /workspace/neuroloc/output/run1_baseline_noerasure
NM_PRESET=run1_baseline_noerasure \
NM_OUTPUT_DIR=/workspace/neuroloc/output/run1_baseline_noerasure \
NM_AUTHORIZE_FULL_RUN=1 \
python3 -u neuroloc/model/god_machine.py 2>&1 | tee /workspace/neuroloc/output/run1_baseline_noerasure.stdout.log
```

to reproduce the archived full-bundle branch intentionally, launch:

```
mkdir -p /workspace/neuroloc/output/god_run
NM_PRESET=god \
NM_OUTPUT_DIR=/workspace/neuroloc/output/god_run \
NM_RUN_NAME=god_run \
python3 -u neuroloc/model/god_machine.py 2>&1 | tee /workspace/neuroloc/output/god_run.stdout.log
```

run the current baseline inside tmux for pod-disconnect resilience:

```
tmux new-session -d -s nm-run1 \
  "bash -l -c 'mkdir -p /workspace/neuroloc/output/run1_baseline_noerasure && NM_PRESET=run1_baseline_noerasure NM_OUTPUT_DIR=/workspace/neuroloc/output/run1_baseline_noerasure NM_AUTHORIZE_FULL_RUN=1 python3 -u /workspace/neuroloc/model/god_machine.py 2>&1 | tee /workspace/neuroloc/output/run1_baseline_noerasure.stdout.log'"
```

monitor:
```
tmux capture-pane -p -t nm-run1 | tail -50
tail -f /workspace/neuroloc/output/run1_baseline_noerasure.stdout.log
tail -f /workspace/neuroloc/output/run1_baseline_noerasure/run1_baseline_noerasure_metrics.jsonl
```

## archived god preset targets

when `NM_PRESET=god`:

- d_model=1024, n_layers=28 (24 delta + 4 attn), head_dim=64, 16 heads
- seq_len=2048, batch_size=16, max_steps=4000, val_interval=200
- lr=3e-4, warmup=200, cosine decay, adamw(0.9, 0.95), weight_decay=0.1
- bf16 amp on h200; fla path for delta-rule; sdpa for attention
- grad_checkpointing on; resume_capable via `{name}_last.pt`
- params: **283m** (not 350m — multi-compartment block-diagonal is ~50% efficient; imagination_filter+pc_head add ~48m; net 283m)
- observed wall clock on h200 for the archived `god` bundle: **~53 min** for `god_run` and **~59 min** for `god_run_v2` at 4000 steps + validation + eval suite
- tokens trained: `4000 * 16 * 2048 = ~131m`

## env variable overrides

- `NM_PRESET` — preset selector. default `run1_baseline_noerasure`. use `god` only when intentionally reproducing the archived all-features bundle; use `run1a_retention_ablation` and `run4_erasure_ablation` for the later named ablations.
- `NM_OUTPUT_DIR` — output directory. set this explicitly for every paid run to avoid mixing branches. if unset, the code now defaults to a preset-specific directory under `neuroloc/output/` (or `/kaggle/working/` on kaggle).
- `NM_RUN_NAME` — run name prefix for artifact files. default comes from the selected preset: `run1_baseline_noerasure`, `run1a_retention_ablation`, `run4_erasure_ablation`, or `god_run`.
- `NM_DATASET` — dataset selector. `auto` chooses wikitext-2 for very short debug runs and fineweb for full runs. the official `run1_baseline_noerasure` surface now requires the fineweb path explicitly; use `NM_DATASET=fineweb` for the benchmark and do not switch that preset to wikitext-2.
- `NM_BATCH_SIZE`, `NM_SEQ_LEN`, `NM_MAX_STEPS`, `NM_VAL_INTERVAL`, `NM_WARMUP_STEPS`, `NM_MAX_SEQ_LEN` — runtime overrides for short timing/oom benchmarks or controlled launch variants. do not use extra overrides on the official run-1 benchmark beyond the documented 20-step contract, and do not use them on the official 4000-step full run at all.
- `NM_AUTHORIZE_FULL_RUN=1` — required for the full `run1_baseline_noerasure` launch after the benchmark gate has been reviewed. without it, the executable refuses the full run, and it now also requires the benchmark manifest in the same output root plus a current device that matches the benchmarked device class.
- `NM_RESUME=1` — explicitly allow resume from `{name}_last.pt`. without this flag, an existing checkpoint makes the run fail fast so a stale benchmark directory cannot masquerade as a fresh gate. benchmark mode now rejects `NM_RESUME=1`; the official 20-step benchmark must be rerun from step 0 before it can mint a fresh manifest.
- `NM_SKIP_VALIDATION=1` — skip training-loop and final validation. use this on the short benchmark so it measures only the training-path recurrent workload.
- `NM_SKIP_EVAL=1` — skip the eval suite after training
- `ALLOW_WIKITEXT_FALLBACK=1` — allow FineWeb failure to fall back to wikitext-2. this is forbidden on the official `run1_baseline_noerasure` benchmark/full-launch surface; reserve it for separately named fallback or debug runs.
- `ALLOW_SYNTHETIC=1` — allow fallback to synthetic random bytes if fineweb/wikitext2 fails (do not use for real runs)
- `SMOKE_TEST=1` — run the cpu smoke test only, no training

## artifacts written (in `NM_OUTPUT_DIR`)

- `{name}_metadata.json` — config + env + git sha + config_hash (written at start)
- `{name}_metrics.jsonl` — streaming per-step events (run_start, step, validation, final_validation, run_end, emergency_flush)
- `{name}_results.json` — atomic snapshot at each val_interval, includes full `history` dict; baseline-style presets also persist `retrieval_gate` here after eval
- `{name}_best.pt` — weights-only, saved when val_bpb improves
- `{name}_last.pt` — resume-capable checkpoint (model + optimizer + scheduler + rng + history + cfg + train_gen_state)
- `{name}_crash.pt` — emergency flush on nan / exception / sigterm
- `{name}_eval_suite.json` — passkey, selective_copy, perplexity_at_length, delta_state_structure_probe results
- `run1_baseline_noerasure_benchmark_manifest.json` — written beside the benchmark/full-run directories after a successful official 20-step FineWeb benchmark; the official full run refuses to start without a matching manifest, matching benchmark metadata/results/metrics artifacts, and a matching git working-tree fingerprint.

## transformer baseline reference

target to beat: **3.57 ± 0.02 bpb at 350m on fineweb-edu after ~131m tokens**

derived from run_010 (267m → 3.583) + run_011 (280m → 3.590) via flat byte-level scaling law (α ≈ 0.25-0.30). no transformer companion run in this script.

## what to watch in metrics.jsonl

each per-step event contains many fields; the key ones:

| field | what to watch for |
|---|---|
| `loss` | monotonic decrease after warmup. spikes > 2x prior are early warning. |
| `grad_norm_pre_clip` | should stabilize in 0.5-5.0 range after warmup. sustained >10 = training instability. |
| `mean_fr` | k-wta realized rate ≈ 0.20 (kwta_fraction). exactly 0.2 = healthy. drift = bug. |
| `per_layer_mean_fr` | per-layer realized firing rate summary. kwta-enabled runs should reflect the target fraction; no-kwta baseline presets should keep overall `mean_fr` at 0.0. |
| `val_bpb` | decrease across validations. target <3.57 at step 4000. |
| `delta_path_per_layer` | `NM_PRESET=god` should stay 0.0 (fla) during gpu training. `run1_baseline_noerasure` and `run1a_retention_ablation` intentionally use the recurrent path and should report 1.0. |
| `delta_erasure_flag_mean` | mean erasure-on flag across delta layers. `run1_baseline_noerasure` and `run1a_retention_ablation` should stay at 0.0. `run4_erasure_ablation` and `NM_PRESET=god` should stay at 1.0. use `delta_erasure_flag_per_layer` for layer detail. |
| `alpha_base_mean_per_layer` | sigmoid(-0.5) ≈ 0.38 at init. should drift under adamw. |
| `imag_gate_mean_per_layer` | starts at sigmoid(-2) ≈ 0.12. growing over training = model trusts imagination more. |
| `imag_ratio_per_layer` | relative contribution of imagination to forward. 0.05-0.30 is healthy; >0.5 = runaway. |
| `state_frobenius_per_layer` | grows early, stabilizes after warmup. monotonic growth = no erasure. sudden drops = instability. |
| `pc_error_l2_per_layer` | decreases as pc_head learns. flat = pc_head not training. |
| `mlp_compartment_l2_per_layer` | vector of 4 values. balanced = healthy compartment use. imbalanced = one compartment dominating. |
| `throughput_tok_per_sec` | for `run1_baseline_noerasure`, establish this from the 20-step h200 benchmark before full launch; do not compare it directly to the archived god-bundle fla numbers. for `NM_PRESET=god`, the observed reference was ~45,500 tok/s. |
| `gpu_mem_max_alloc_gb` | target <50 gb. >120 gb = oom risk. |

## eval suite interpretation

written to `{name}_eval_suite.json` after training completes.

- **passkey @256/1024/4096**: exact-match accuracy for 5-digit passkey hidden in noise. untrained: 0%. trained target: >80% at ctx 256, >50% at ctx 1024, >20% at ctx 4096 (retrieval horizon).
- **selective_copy @256/512/1024/2048**: token-by-token match on 8-byte target. untrained: 0%. trained target: >60% at ctx 256.
- **perplexity_at_length @256..4096**: bpb curve. should be flat or slowly increasing with length. sharp jump = context horizon hit.
- **delta_state_structure_probe.mean_structure_ratio**: trained state's probe output norm vs random state. >1.5 = memory is structured (non-random). untrained: ≈1.0.
- **delta_state_structure_probe.mean_pairwise_cos**: pairwise cosine similarity of probe outputs. trained target: >0.2. means the state acts as a low-dim manifold (hopfield-style structured interpolation above critical load — kalaj 2024).
- **delta_state_structure_probe.random_mean_pairwise_cos**: control, expected ≈0.
- interpretation: if `mean_pairwise_cos > 5 * random_mean_pairwise_cos` AND `mean_structure_ratio > 1.2`, the delta-state structure probe is detecting structured associative memory rather than noise.

## troubleshooting

### symptom: `delta_path_per_layer` is 1.0 during `run1_baseline_noerasure` or `run1a_retention_ablation`
cause: none. those presets intentionally run on the recurrent path.
fix: no action required unless the preset was supposed to be `god`.

### symptom: `delta_path_per_layer` is 1.0 during `NM_PRESET=god`
cause: fla not available or not being used.
fix: `python -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule; print('ok')"`. if import fails, reinstall fla. check `FLA_AVAILABLE` at module import time in stdout.

### symptom: loss goes to nan in the first 50 steps
likely cause: delta-rule state runaway (alpha_eff too close to 1 + no erasure). check `state_frobenius_per_layer` trace — if it grows exponentially, bcm sign might be flipped or gamma_bcm too small.
immediate action: emergency flush fires, crash.pt and last.pt written, results.json snapshot has partial history. diagnose from `metrics.jsonl`.

### symptom: smoke test fails with "fla and recurrent paths diverge"
cause: the cpu recurrence form does not match fla's gated-delta-rule semantics. this is the exact bug caught in prosecution round 2.
fix: the recurrent path should apply erasure to the decayed state (α·S - α·β·k·k^T·S + β·k·v^T), not the raw state.

### symptom: the `run1_baseline_noerasure` benchmark is much slower than expected or risks oom
likely causes (check in order):
1. you are not actually on `NM_PRESET=run1_baseline_noerasure`
2. runtime overrides changed the intended launch shape (`NM_BATCH_SIZE`, `NM_SEQ_LEN`, `NM_MAX_STEPS`)
3. bf16 not active (`amp = False` or sm_75 fallback)
4. sdpa falling back to math backend (head_dim=96 not aligned)
5. the recurrent path is simply too slow at the chosen shape; if so, do not authorize the 4000-step paid run until the benchmark result has been reviewed

for the archived `god` preset only, low throughput plus `delta_path_per_layer = 1.0` means fla is not running and should be debugged as a preset mismatch or missing dependency.

### symptom: imag_ratio is > 0.5 or growing unbounded
likely cause: imagination gate is over-trusting memory. the learned gate has saturated.
action: reduce `imag_gate_init` from -2 to -4 in config (sigmoid(-4)≈0.02 at init). train from scratch.

### symptom: val_bpb plateaus above 4.0 after 2000 steps
likely causes:
1. all 5 features are interacting badly (this is the risk the blueprint's isolation protocol was designed to catch)
2. k-wta too aggressive (kwta_fraction=0.20 is low); try 0.30 or 0.40
3. bcm running_state_norm diverged
4. imagination probe contaminating gradient updates

action: inspect `metrics.jsonl`, check correlations across features, post-hoc disable one feature at a time on a fresh run.

## known limitations

1. **fla and cpu recurrent paths may disagree numerically under erasure** — the parity test at `_test_fla_vs_recurrent_parity` runs on cuda at smoke time, target rel < 1e-2. if it fails, either the fla kernel semantics changed, or the cpu recurrence form needs realignment.
2. **bcm alpha is driven by the same `running_state_norm` ema in both paths** — recurrent and fla now both go through `_effective_log_alpha`. remaining divergence between the paths should be treated as a numerical parity issue, not as an intentional alpha-rule mismatch.
3. **pc_diagnostic is technically a small training signal** — `pc_lambda=1e-4` is small but nonzero. pc_head receives gradient, upstream does not (detached). the bridge document warned about redundancy with backprop; we accept this as the cost of having pc_error as a genuine diagnostic.
4. **imagination probe reads from post-attention out**, not directly from state matrix. causally safe but less ambitious than a learned query into the state. adequate for the archived `god` bundle.
5. **archived `god` bundle only**: no causal attribution per feature. all 5 features interact there. if that bundle beats or loses to the 3.57 bpb baseline, we cannot say WHICH feature was responsible. use the god-metrics correlations post-hoc.
6. **no transformer companion**: comparison target is the empirical 3.57 bpb from run_010/011, not a fresh matched-pipeline run.

## current baseline verification contract

for the current default branch (`run1_baseline_noerasure`), the hard launch gate is:

- [ ] a 20-step h200 benchmark at the full launch shape completes cleanly on the recurrent path, from a fresh directory with no resume, and writes `run1_baseline_noerasure_benchmark_manifest.json` in the shared output root
- [ ] completes 4000 steps without nan / crash
- [ ] eval suite completes and writes `retrieval_gate` into `{name}_results.json`
- [ ] `retrieval_gate.status == "pass"`, meaning `passkey_256_accuracy > 0.0`

supporting diagnostics for that branch:

- `selective_copy` and `delta_state_structure_probe` remain informative but are not the hard gate for run 1.
- language quality still matters for interpretation, but the blocking gate is nonzero passkey at 256, not a fixed bpb threshold.
- `run1_baseline_noerasure` and `run1a_retention_ablation` should stay on the recurrent path; `run4_erasure_ablation` differs only by turning overwrite subtraction back on.

the named ablations still persist `retrieval_gate`, but the benchmark-manifest/device/provenance hardening described here is currently specific to the official `run1_baseline_noerasure` paid-launch surface.

if `run1_baseline_noerasure` finishes with `retrieval_gate.status == "fail"`, the next planned run is `run1a_retention_ablation`, not an immediate architectural rewrite. if eval is skipped, the best checkpoint is missing, or eval errors out, the run is inconclusive and now exits nonzero instead of silently satisfying the launch surface.

for the archived `god` preset, the older >10% passkey and imagination-structure thresholds should be treated as stretch diagnostics, not as the live project gate.

## verification before launch

```
SMOKE_TEST=1 python3 neuroloc/model/god_machine.py
```

expected smoke output (all checks green, ~15 seconds on h200):

```
[...] running cpu smoke test
[...] smoke model params: 155,535
[...] smoke test forward+backward ok. main_loss=5.57xx pc_loss=0.99xx
[...] smoke test per-feature probes ok
[...] smoke test single-token recurrent ok
[...] smoke test greedy_generate ok (len=8)
[...] smoke test passkey_retrieval ok (untrained acc=0.00)
[...] smoke test selective_copy ok (untrained acc=0.00)
[...] smoke test measure_perplexity_at_length ok (bpb=~8.0)
[...] smoke test delta_state_structure_probe ok (structure_ratio=0.9xx pairwise_cos=0.0xx random_pairwise_cos=0.0xx)
[...] delta parallel/recurrent equivalence test (no spikes, channel_wise_gate=False)
[...]   parallel vs recurrent: max_abs=~1e-6, rel=~3e-7 ok
[...] fla-vs-recurrent parity test (with erasure, bcm disabled)    # on h200+fla
[...]   fla vs recurrent: max_abs=?, rel=? (target rel<1e-2)
[...] smoke test run4_erasure_ablation preset ok (...)
[...] smoke test run1_baseline_noerasure preset ok (...)
[...] smoke test run1a_retention_ablation preset ok (...)
[...] all smoke checks passed
```

if any check fails, DO NOT launch training. diagnose via the prosecutor agent or read the failure trace.
