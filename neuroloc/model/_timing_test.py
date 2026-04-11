from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/neuroloc/model")
from neural_machine import Config, train_model, download_wikitext2


def main() -> None:
    cfg = Config(
        max_steps=15,
        warmup_steps=3,
        val_interval=10,
    )

    print(f"timing test: full 350m config, {cfg.max_steps} steps, wikitext-2")
    print(f"  d_model={cfg.d_model} n_layers={cfg.n_layers}")
    print(f"  delta_heads={cfg.delta_num_heads} delta_hd={cfg.delta_head_dim}")
    print(f"  batch_size={cfg.batch_size} seq_len={cfg.seq_len}")
    print(f"  mlp_ratio={cfg.mlp_ratio}")
    print(f"  grad_checkpointing={cfg.grad_checkpointing} amp={cfg.amp}")

    train_data, val_data, src = download_wikitext2()
    out = Path("/workspace/neuroloc/output/timing_test")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    result = train_model(cfg, train_data, val_data, "timing", out, dataset_source=src)

    print()
    print("=== timing result ===")
    print(f"params: {result['param_count']:,}")
    print(f"total_steps: {result['total_steps']}")
    print(f"total_time_sec: {result['total_time']:.2f}")
    sec_per_step = result["total_time"] / max(1, result["total_steps"])
    print(f"sec_per_step: {sec_per_step:.2f}")

    with open(out / "timing_results.json") as f:
        r = json.load(f)
    h = r["history"]
    print(f"step_durations: {[round(x, 2) for x in h['step_duration_sec'][:5]]}")
    print(f"step_durations (steady): {[round(x, 2) for x in h['step_duration_sec'][-5:]]}")
    tps = [round(x, 0) for x in h["throughput_tok_per_sec"]]
    print(f"throughput: {tps}")
    print(f"gpu_mem_alloc_gb final: {h['gpu_mem_alloc_gb'][-1]:.2f}")
    print(f"gpu_mem_max_alloc_gb final: {h['gpu_mem_max_alloc_gb'][-1]:.2f}")
    print(f"val_bpb: {h['val_bpb']}")

    sec_per_step_steady = sum(h["step_duration_sec"][-5:]) / 5.0
    est_4000 = 4000 * sec_per_step_steady
    print()
    print(f"ESTIMATED 4000-step wall-clock: {est_4000:.0f}s = {est_4000 / 60:.1f}min = {est_4000 / 3600:.2f}hr")


if __name__ == "__main__":
    main()
