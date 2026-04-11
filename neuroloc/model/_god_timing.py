from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/neuroloc/model")
from god_machine import Config, train_model, download_wikitext2, FLA_AVAILABLE


def main() -> None:
    cfg = Config(
        max_steps=15,
        warmup_steps=3,
        val_interval=10,
    )

    print(f"timing test: full 283m god_machine config, {cfg.max_steps} steps, wikitext-2")
    print(f"  FLA available: {FLA_AVAILABLE}")
    print(f"  d_model={cfg.d_model} n_layers={cfg.n_layers}")
    print(f"  delta_heads={cfg.delta_num_heads} delta_hd={cfg.delta_head_dim}")
    print(f"  batch_size={cfg.batch_size} seq_len={cfg.seq_len}")
    print(f"  kwta={cfg.kwta_enabled} erasure={cfg.delta_erasure_enabled} bcm={cfg.bcm_alpha_enabled}")
    print(f"  imagination={cfg.imagination_enabled} pc={cfg.pc_diagnostic_enabled} multicomp={cfg.multi_compartment_enabled}")
    print(f"  grad_checkpointing={cfg.grad_checkpointing} amp={cfg.amp}")

    train_data, val_data, src = download_wikitext2()
    out = Path("/workspace/neuroloc/output/god_timing")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    result = train_model(cfg, train_data, val_data, "god_timing", out, dataset_source=src)

    print()
    print("=== timing result ===")
    print(f"params: {result['param_count']:,}")
    print(f"total_steps: {result['total_steps']}")
    print(f"total_time_sec: {result['total_time']:.2f}")
    sec_per_step = result["total_time"] / max(1, result["total_steps"])
    print(f"avg sec_per_step: {sec_per_step:.3f}")

    with open(out / "god_timing_results.json") as f:
        r = json.load(f)
    h = r["history"]
    non_val_durations = [d for d in h["step_duration_sec"] if d < 2.0]
    if non_val_durations:
        steady_avg = sum(non_val_durations[-5:]) / max(1, len(non_val_durations[-5:]))
        est_4000 = 4000 * steady_avg
        print(f"steady-state sec/step: {steady_avg:.3f}")
        print(f"estimated 4000-step wall-clock: {est_4000:.0f}s = {est_4000 / 60:.1f}min = {est_4000 / 3600:.2f}hr")

    tps = [round(x, 0) for x in h["throughput_tok_per_sec"]]
    print(f"throughput history: {tps}")
    print(f"gpu_mem_alloc_gb final: {h['gpu_mem_alloc_gb'][-1]:.2f}")
    print(f"gpu_mem_max_alloc_gb final: {h['gpu_mem_max_alloc_gb'][-1]:.2f}")
    print(f"val_bpb: {h['val_bpb']}")
    print(f"per-layer delta_path[0]: {h.get('delta_path_per_layer', [[0]])[0]}")


if __name__ == "__main__":
    main()
