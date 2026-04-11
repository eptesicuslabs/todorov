# god_machine.py — run card

## what this is

one-shot training script for the unified neural machine with all 5 blueprint features active simultaneously (k-wta rate-coded compression, delta rule erasure, bcm-like adaptive alpha, multi-compartment swiglu, compressed retrieval) plus always-on imagination probe, predictive coding diagnostic, and god-tier telemetry. 283m parameters at default config on fineweb-edu bytes.

deviates from the 5-run isolation protocol in `neuroloc/spec/blueprint.md` by design — user override for budget-constrained single-run evaluation. no causal attribution per feature; correlations only.

## environment requirements

- gpu: h200 (sm_90) strongly preferred. sm_80+ works with bf16. sm_75 and below: disable amp (`cfg.amp = False`), no bf16 tensor cores.
- python: 3.10+
- pytorch: 2.6.0+ with cuda 12.x
- triton: 3.2.0+ (for fla kernel)
- flash-linear-attention: install via `pip install --break-system-packages flash-linear-attention`
- datasets: `pip install --break-system-packages datasets`
- disk: ~10 gb free (5 gb fineweb cache + checkpoints + metrics)

## quick install on fresh pod

```
pip install --break-system-packages -q datasets flash-linear-attention
python -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule; print('fla ok')"
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

## launch command

```
cd /workspace
SMOKE_TEST=1 python3 neuroloc/model/god_machine.py 2>&1 | tee smoke.log
```

if smoke passes, then:

```
mkdir -p /workspace/neuroloc/output/god_run
NM_OUTPUT_DIR=/workspace/neuroloc/output/god_run \
NM_RUN_NAME=god_run \
python3 -u neuroloc/model/god_machine.py 2>&1 | tee /workspace/neuroloc/output/god_run/stdout.log
```

run inside tmux for pod-disconnect resilience:

```
tmux new-session -d -s god \
  "bash -l -c 'NM_OUTPUT_DIR=/workspace/neuroloc/output/god_run NM_RUN_NAME=god_run python3 -u /workspace/neuroloc/model/god_machine.py 2>&1 | tee /workspace/neuroloc/output/god_run/stdout.log'"
```

monitor:
```
tmux capture-pane -p -t god | tail -50
tail -f /workspace/neuroloc/output/god_run/god_run_metrics.jsonl
```

## default config targets

- d_model=1024, n_layers=28 (24 delta + 4 attn), head_dim=64, 16 heads
- seq_len=2048, batch_size=16, max_steps=4000, val_interval=200
- lr=3e-4, warmup=200, cosine decay, adamw(0.9, 0.95), weight_decay=0.1
- bf16 amp on h200; fla path for delta-rule; sdpa for attention
- grad_checkpointing on; resume_capable via `{name}_last.pt`
- params: **283m** (not 350m — multi-compartment block-diagonal is ~50% efficient; imagination_filter+pc_head add ~48m; net 283m)
- expected wall clock at target speedup: **100-150 ms/step**, **~8-12 min for 4000 steps + val + eval suite**
- tokens trained: `4000 * 16 * 2048 = ~131m`

## env variable overrides

- `NM_OUTPUT_DIR` — output directory (default `neuroloc/output/god_run`)
- `NM_RUN_NAME` — run name prefix for artifact files (default `god_run`)
- `NM_SKIP_EVAL=1` — skip the eval suite after training
- `ALLOW_SYNTHETIC=1` — allow fallback to synthetic random bytes if fineweb/wikitext2 fails (do not use for real runs)
- `SMOKE_TEST=1` — run the cpu smoke test only, no training

## artifacts written (in `NM_OUTPUT_DIR`)

- `{name}_metadata.json` — config + env + git sha + config_hash (written at start)
- `{name}_metrics.jsonl` — streaming per-step events (run_start, step, validation, final_validation, run_end, emergency_flush)
- `{name}_results.json` — atomic snapshot at each val_interval, includes full `history` dict
- `{name}_best.pt` — weights-only, saved when val_bpb improves
- `{name}_last.pt` — resume-capable checkpoint (model + optimizer + scheduler + rng + history + cfg + train_gen_state)
- `{name}_crash.pt` — emergency flush on nan / exception / sigterm
- `{name}_eval_suite.json` — passkey, selective_copy, perplexity_at_length, imagination results

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
| `dead_pct` | always 0 by construction (k-wta never kills neurons). not a diagnostic. |
| `val_bpb` | decrease across validations. target <3.57 at step 4000. |
| `delta_path_per_layer` | should be 0.0 (fla) at every layer during training on gpu. 1.0 = recurrent fallback (slow). 2.0 = parallel_no_erasure (feature drop!). |
| `delta_erasure_applied_mean` | should be 1.0 with default config. |
| `alpha_base_mean_per_layer` | sigmoid(-0.5) ≈ 0.38 at init. should drift under adamw. |
| `imag_gate_mean_per_layer` | starts at sigmoid(-2) ≈ 0.12. growing over training = model trusts imagination more. |
| `imag_ratio_per_layer` | relative contribution of imagination to forward. 0.05-0.30 is healthy; >0.5 = runaway. |
| `state_frobenius_per_layer` | grows early, stabilizes after warmup. monotonic growth = no erasure. sudden drops = instability. |
| `pc_error_l2_per_layer` | decreases as pc_head learns. flat = pc_head not training. |
| `mlp_compartment_l2_per_layer` | vector of 4 values. balanced = healthy compartment use. imbalanced = one compartment dominating. |
| `throughput_tok_per_sec` | target 20000-30000+ on h200 for 350m. below 10000 = performance regression. |
| `gpu_mem_max_alloc_gb` | target <50 gb. >120 gb = oom risk. |

## eval suite interpretation

written to `{name}_eval_suite.json` after training completes.

- **passkey @256/1024/4096**: exact-match accuracy for 5-digit passkey hidden in noise. untrained: 0%. trained target: >80% at ctx 256, >50% at ctx 1024, >20% at ctx 4096 (retrieval horizon).
- **selective_copy @256/512/1024/2048**: token-by-token match on 8-byte target. untrained: 0%. trained target: >60% at ctx 256.
- **perplexity_at_length @256..4096**: bpb curve. should be flat or slowly increasing with length. sharp jump = context horizon hit.
- **imagination.mean_structure_ratio**: trained state's probe output norm vs random state. >1.5 = memory is structured (non-random). untrained: ≈1.0.
- **imagination.mean_pairwise_cos**: pairwise cosine similarity of probe outputs. trained target: >0.2. means the state acts as a low-dim manifold (hopfield-style structured interpolation above critical load — kalaj 2024).
- **imagination.random_mean_pairwise_cos**: control, expected ≈0.
- interpretation: if `mean_pairwise_cos > 5 * random_mean_pairwise_cos` AND `mean_structure_ratio > 1.2`, imagination is working as designed (delta state is a structured associative memory).

## troubleshooting

### symptom: `delta_path_per_layer` is 1.0 or 2.0 during training (not 0.0)
cause: fla not available or not being used.
fix: `python -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule; print('ok')"`. if import fails, reinstall fla. check `FLA_AVAILABLE` at module import time in stdout.

### symptom: loss goes to nan in the first 50 steps
likely cause: delta-rule state runaway (alpha_eff too close to 1 + no erasure). check `state_frobenius_per_layer` trace — if it grows exponentially, bcm sign might be flipped or gamma_bcm too small.
immediate action: emergency flush fires, crash.pt and last.pt written, results.json snapshot has partial history. diagnose from `metrics.jsonl`.

### symptom: smoke test fails with "fla and recurrent paths diverge"
cause: the cpu recurrence form does not match fla's gated-delta-rule semantics. this is the exact bug caught in prosecution round 2.
fix: the recurrent path should apply erasure to the decayed state (α·S - α·β·k·k^T·S + β·k·v^T), not the raw state.

### symptom: throughput << 20000 tok/s
likely causes (check in order):
1. `delta_path != 0.0` → fla is not running (see above)
2. `grad_checkpointing = True` but `n_params < 100m` → gc overhead without memory benefit; disable for smaller configs
3. bf16 not active (`amp = False` or sm_75 fallback)
4. sdpa falling back to math backend (head_dim=96 not aligned)
5. recurrent_with_erasure being used due to `force_recurrent = True` (defaults ok, but check config)

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
2. **bcm alpha is static under fla** — uses `running_state_norm` ema buffer, not live per-token state norm. recurrent path uses live. the two paths are not mathematically identical. documented.
3. **pc_diagnostic is technically a small training signal** — `pc_lambda=1e-4` is small but nonzero. pc_head receives gradient, upstream does not (detached). the bridge document warned about redundancy with backprop; we accept this as the cost of having pc_error as a genuine diagnostic.
4. **imagination probe reads from post-attention out**, not directly from state matrix. causally safe but less ambitious than a learned query into the state. adequate for run 1.
5. **single run**: no causal attribution per feature. all 5 features interact. if the run beats or loses to the 3.57 bpb baseline, we cannot say WHICH feature was responsible. use the god-metrics correlations post-hoc.
6. **no transformer companion**: comparison target is the empirical 3.57 bpb from run_010/011, not a fresh matched-pipeline run.

## minimum success criterion

the run is successful if ALL of:

- [ ] completes 4000 steps without nan / crash
- [ ] `best_val_bpb < 3.57` on fineweb-edu val
- [ ] `delta_path_per_layer` was 0.0 (fla) throughout training
- [ ] all per-feature probes (kwta, delta erasure, bcm, imag gate, pc, compartment balance) pass at final step
- [ ] eval suite passkey@256 > 10% accuracy (untrained is 0%)
- [ ] imagination `mean_structure_ratio > 1.2` and `pairwise_cos > 3 * random_pairwise_cos`

the run is INCONCLUSIVE (not failed, not succeeded) if:

- completes but val_bpb > 3.57 (features hurt)
- completes but one or more features show degenerate telemetry (e.g., dead imagination gate, runaway state norm)
- completes but retrieval accuracy < 10% (memory not functional)

in inconclusive cases, the next run is a focused ablation using `neuroloc/model/neural_machine.py` (the clean baseline without features) or a single-feature-off god_machine run.

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
[...] smoke test imagination ok (structure_ratio=0.9xx pairwise_cos=0.0xx random_pairwise_cos=0.0xx)
[...] delta parallel/recurrent equivalence test (no spikes, channel_wise_gate=False)
[...]   parallel vs recurrent: max_abs=~1e-6, rel=~3e-7 ok
[...] fla-vs-recurrent parity test (with erasure, bcm disabled)    # on h200+fla
[...]   fla vs recurrent: max_abs=?, rel=? (target rel<1e-2)
[...] all smoke checks passed
```

if any check fails, DO NOT launch training. diagnose via the prosecutor agent or read the failure trace.
