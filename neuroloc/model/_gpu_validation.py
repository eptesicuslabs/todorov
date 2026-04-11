from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/neuroloc/model")
from neural_machine import Config, train_model, download_wikitext2


def main() -> None:
    cfg = Config(
        d_model=256,
        n_layers=4,
        delta_num_heads=4,
        delta_head_dim=32,
        attn_d_c=32,
        attn_d_R=16,
        attn_num_heads=4,
        mlp_ratio=2.75,
        layer_pattern=("DELTA", "DELTA", "DELTA", "ATTN"),
        batch_size=4,
        seq_len=256,
        max_steps=20,
        warmup_steps=5,
        val_interval=10,
        grad_checkpointing=False,
        amp=True,
    )

    print("gpu validation run: small (~10M), 20 steps, bf16 amp")
    train_data, val_data, src = download_wikitext2()
    out = Path("/workspace/neuroloc/output/gpu_validation")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    result = train_model(cfg, train_data, val_data, "gpu_val", out, dataset_source=src)

    print()
    print("=== result ===")
    param_count = result["param_count"]
    total_steps = result["total_steps"]
    best_val_bpb = result["best_val_bpb"]
    tokens_seen = result["tokens_seen_total"]
    total_time = result["total_time"]
    print(f"params: {param_count:,}")
    print(f"total_steps: {total_steps}")
    print(f"best_val_bpb: {best_val_bpb:.4f}")
    print(f"tokens: {tokens_seen:,}")
    print(f"time: {total_time:.1f}s")

    with open(out / "gpu_val_results.json") as f:
        r = json.load(f)
    h = r["history"]
    print(f"history train_loss len: {len(h['train_loss'])}")
    print(f"history grad_norm_pre_clip[:3]: {h['grad_norm_pre_clip'][:3]}")
    throughput_head = [round(x, 0) for x in h["throughput_tok_per_sec"][:3]]
    print(f"history throughput_tok_per_sec[:3]: {throughput_head}")
    print(f"history gpu_mem_alloc_gb[-1]: {h['gpu_mem_alloc_gb'][-1]:.2f}")
    print(f"history gpu_mem_max_alloc_gb[-1]: {h['gpu_mem_max_alloc_gb'][-1]:.2f}")
    print(f"val_bpb entries: {h['val_bpb']}")
    print(f"val_steps: {h['val_steps']}")

    metrics_lines = (out / "gpu_val_metrics.jsonl").read_text().strip().split("\n")
    print(f"metrics.jsonl lines: {len(metrics_lines)}")
    print(f"last event: {metrics_lines[-1][:120]}")


if __name__ == "__main__":
    main()
