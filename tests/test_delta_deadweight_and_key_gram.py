from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neuroloc.model import god_machine
from neuroloc.model.god_machine import (
    ByteDataset,
    Config,
    DeltaRuleMemory,
    GodMachine,
    collate_fn,
    run_validation,
    train_model,
)


OUTPUT_PATH = PROJECT_ROOT / "neuroloc" / "output" / "test_runs" / "delta_deadweight_experiment.json"
SEED = 42
TIME_BUDGET_SEC = 2 * 60 * 60


def _set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random as _py_random
    _py_random.seed(seed)


def _build_cfg() -> Config:
    return Config(
        d_model=128,
        n_layers=7,
        vocab_size=256,
        max_seq_len=256,
        delta_num_heads=4,
        delta_head_dim=32,
        attn_d_c=32,
        attn_d_R=8,
        attn_num_heads=4,
        mlp_ratio=2.0,
        layer_pattern=("DELTA", "DELTA", "DELTA", "DELTA", "DELTA", "DELTA", "ATTN"),
        batch_size=8,
        seq_len=256,
        max_steps=500,
        warmup_steps=50,
        val_interval=100,
        grad_checkpointing=False,
        amp=False,
        use_fla_if_available=False,
        kwta_enabled=False,
        delta_erasure_enabled=False,
        bcm_alpha_enabled=False,
        multi_compartment_enabled=False,
        imagination_enabled=False,
        pc_diagnostic_enabled=False,
        seed=SEED,
    )


def _make_synthetic_bytes(total_bytes: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    parts: list[np.ndarray] = []
    remaining = total_bytes
    mode_cycle = 0

    lang_p = np.ones(256, dtype=np.float64)
    for code in range(ord("a"), ord("z") + 1):
        lang_p[code] = 25.0
    for code in range(ord("A"), ord("Z") + 1):
        lang_p[code] = 6.0
    lang_p[ord(" ")] = 40.0
    lang_p[ord(".")] = 3.0
    lang_p[ord(",")] = 3.0
    lang_p[ord("\n")] = 2.0
    for code in range(ord("0"), ord("9") + 1):
        lang_p[code] = 2.0
    lang_p = lang_p / lang_p.sum()

    while remaining > 0:
        chunk_len = min(remaining, int(rng.randint(1024, 8192)))
        mode = mode_cycle % 4
        if mode == 0:
            block = rng.choice(256, size=chunk_len, p=lang_p).astype(np.uint8)
        elif mode == 1:
            pattern_len = int(rng.randint(4, 32))
            pattern = rng.randint(32, 127, size=pattern_len).astype(np.uint8)
            reps = (chunk_len + pattern_len - 1) // pattern_len
            block = np.tile(pattern, reps)[:chunk_len]
        elif mode == 2:
            block = rng.randint(0, 256, size=chunk_len, dtype=np.uint8)
        else:
            alphabet_len = int(rng.randint(6, 16))
            alphabet = rng.choice(
                np.arange(32, 127), size=alphabet_len, replace=False
            ).astype(np.uint8)
            indices = rng.randint(0, alphabet_len, size=chunk_len)
            block = alphabet[indices]
        parts.append(block)
        remaining -= chunk_len
        mode_cycle += 1

    data = np.concatenate(parts).astype(np.uint8)
    return data[:total_bytes]


def _set_inference_mode(model: GodMachine) -> None:
    model.train(False)


def _evaluate_bpb(
    model: GodMachine,
    val_data: np.ndarray,
    cfg: Config,
    max_batches: int,
) -> dict:
    val_ds = ByteDataset(val_data, cfg.seq_len)
    if len(val_ds) == 0:
        return {"mean_loss": float("inf"), "bpb": float("inf"), "tokens": 0, "precision": "float32"}
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
    )
    return run_validation(model, val_dl, max_batches=max_batches, use_amp=False, amp_dtype=torch.float32)


def _make_zero_forward(mixer: DeltaRuleMemory):
    def _zero_forward(x: Tensor, state: Any = None, offset: int = 0):
        device = x.device
        out = torch.zeros_like(x)
        if state is not None:
            new_state = state
        else:
            new_state = None
        aux: dict[str, Tensor] = {
            "alpha_base_mean": torch.tensor(0.0, device=device),
            "alpha_eff_mean": torch.tensor(0.0, device=device),
            "beta_mean": torch.tensor(0.0, device=device),
            "state_frobenius": torch.tensor(0.0, device=device),
            "delta_path": torch.tensor(1.0, device=device),
            "delta_erasure_flag": torch.tensor(0.0, device=device),
        }
        return out, new_state, aux
    return _zero_forward


def run_experiment_a(cfg: Config, output_dir: Path, train_data: np.ndarray, val_data: np.ndarray) -> dict:
    _set_all_seeds(SEED)
    t0 = time.time()
    result = train_model(
        cfg=cfg,
        train_data=train_data,
        val_data=val_data,
        name="delta_deadweight_baseline",
        output_dir=output_dir,
        dataset_source="synthetic_mixed_patterns",
        stop_after_steps=None,
        launch_contract=None,
    )
    train_time = time.time() - t0
    baseline_bpb = float(result.get("best_val_bpb", float("nan")))
    if not math.isfinite(baseline_bpb):
        baseline_bpb = float(result.get("final_val_bpb", float("nan")))

    best_ckpt = output_dir / "delta_deadweight_baseline_best.pt"
    if not best_ckpt.exists():
        return {
            "status": "failed",
            "reason": "baseline_ckpt_missing",
            "baseline_bpb": baseline_bpb,
            "train_time_sec": train_time,
        }

    model = GodMachine(cfg).to(god_machine.DEVICE)
    state_dict = torch.load(best_ckpt, map_location=god_machine.DEVICE)
    model.load_state_dict(state_dict, strict=True)
    _set_inference_mode(model)

    baseline_rerun = _evaluate_bpb(model, val_data, cfg, max_batches=50)
    baseline_rerun_bpb = float(baseline_rerun["bpb"])

    originals: list[tuple[DeltaRuleMemory, Any]] = []
    try:
        for block in model.blocks:
            mixer = block.mixer
            if isinstance(mixer, DeltaRuleMemory):
                originals.append((mixer, mixer.forward))
                mixer.forward = _make_zero_forward(mixer)
        zeroed = _evaluate_bpb(model, val_data, cfg, max_batches=50)
    finally:
        for mixer, orig_fn in originals:
            mixer.forward = orig_fn

    zeroed_bpb = float(zeroed["bpb"])
    if math.isfinite(baseline_rerun_bpb) and baseline_rerun_bpb > 0:
        relative_change = (zeroed_bpb - baseline_rerun_bpb) / baseline_rerun_bpb
    else:
        relative_change = float("nan")
    hypothesis_confirmed = (
        math.isfinite(relative_change) and abs(relative_change) < 0.05
    )
    return {
        "status": "ok",
        "baseline_best_val_bpb_training": baseline_bpb,
        "baseline_val_bpb_reevaluated": baseline_rerun_bpb,
        "baseline_val_tokens": int(baseline_rerun.get("tokens", 0)),
        "zeroed_val_bpb": zeroed_bpb,
        "zeroed_val_tokens": int(zeroed.get("tokens", 0)),
        "relative_change": relative_change,
        "pass_criterion_abs_rel_change_lt_0p05": hypothesis_confirmed,
        "hypothesis_dead_weight_confirmed": hypothesis_confirmed,
        "num_delta_layers_zeroed": len(originals),
        "train_time_sec": train_time,
        "checkpoint_path": str(best_ckpt),
    }


def run_experiment_b(cfg: Config, output_dir: Path, val_data: np.ndarray) -> dict:
    best_ckpt = output_dir / "delta_deadweight_baseline_best.pt"
    if not best_ckpt.exists():
        return {"status": "failed", "reason": "baseline_ckpt_missing"}

    model = GodMachine(cfg).to(god_machine.DEVICE)
    state_dict = torch.load(best_ckpt, map_location=god_machine.DEVICE)
    model.load_state_dict(state_dict, strict=True)
    _set_inference_mode(model)

    target_layer_index = 3
    if target_layer_index >= len(model.blocks):
        return {"status": "failed", "reason": f"target_layer_index {target_layer_index} out of range"}
    target_block = model.blocks[target_layer_index]
    if not isinstance(target_block.mixer, DeltaRuleMemory):
        return {
            "status": "failed",
            "reason": f"layer {target_layer_index} is not DELTA: {type(target_block.mixer).__name__}",
        }
    target_mixer: DeltaRuleMemory = target_block.mixer
    nh = target_mixer.nh
    hd = target_mixer.hd

    captured_k: dict[str, Tensor] = {}

    def _hook(module: torch.nn.Module, inputs: tuple, output: Tensor) -> None:
        captured_k["value"] = output.detach().clone()

    handle = target_mixer.k.register_forward_hook(_hook)

    _set_all_seeds(SEED + 1)
    val_ds = ByteDataset(val_data, cfg.seq_len)
    if len(val_ds) < 16:
        handle.remove()
        return {"status": "failed", "reason": f"val dataset too small: {len(val_ds)} < 16"}
    indices = np.random.RandomState(SEED + 1).choice(len(val_ds), size=16, replace=False)
    batch_tensors = [val_ds[int(i)] for i in indices]
    batch = torch.stack(batch_tensors, dim=0).to(god_machine.DEVICE)
    input_ids = batch[:, :-1]

    try:
        with torch.no_grad():
            _ = model(input_ids)
    finally:
        handle.remove()

    if "value" not in captured_k:
        return {"status": "failed", "reason": "key projection not captured"}

    k_raw = captured_k["value"]
    B, T, inner = k_raw.shape
    if inner != nh * hd:
        return {
            "status": "failed",
            "reason": f"captured k inner dim {inner} != nh*hd {nh*hd}",
        }
    k_tokens = k_raw.reshape(B * T, nh, hd)
    total_tokens = k_tokens.shape[0]
    sample_size = min(512, total_tokens)
    rng = np.random.RandomState(SEED + 2)
    sample_idx = rng.choice(total_tokens, size=sample_size, replace=False)
    k_sample = k_tokens[torch.from_numpy(sample_idx).long()]

    per_head_stats: list[dict[str, float]] = []
    diag_means: list[float] = []
    off_diag_means: list[float] = []
    off_diag_abs_means: list[float] = []
    ratios: list[float] = []
    ratios_abs: list[float] = []

    for h in range(nh):
        k_h = k_sample[:, h, :].float()
        gram = k_h @ k_h.t()
        n = gram.shape[0]
        diag = torch.diagonal(gram)
        diag_mean = float(diag.mean().item())
        mask = ~torch.eye(n, dtype=torch.bool, device=gram.device)
        off_diag_vals = gram[mask]
        off_diag_mean = float(off_diag_vals.mean().item())
        off_diag_abs_mean = float(off_diag_vals.abs().mean().item())
        ratio = off_diag_mean / diag_mean if abs(diag_mean) > 1e-12 else float("nan")
        ratio_abs = off_diag_abs_mean / diag_mean if abs(diag_mean) > 1e-12 else float("nan")
        per_head_stats.append(
            {
                "head": h,
                "diagonal_mean": diag_mean,
                "off_diagonal_mean": off_diag_mean,
                "off_diagonal_abs_mean": off_diag_abs_mean,
                "ratio_off_over_diag": ratio,
                "ratio_abs_off_over_diag": ratio_abs,
            }
        )
        diag_means.append(diag_mean)
        off_diag_means.append(off_diag_mean)
        off_diag_abs_means.append(off_diag_abs_mean)
        ratios.append(ratio)
        ratios_abs.append(ratio_abs)

    avg_diag = float(np.mean(diag_means))
    avg_off_diag = float(np.mean(off_diag_means))
    avg_off_diag_abs = float(np.mean(off_diag_abs_means))
    avg_ratio = avg_off_diag / avg_diag if abs(avg_diag) > 1e-12 else float("nan")
    avg_ratio_abs = avg_off_diag_abs / avg_diag if abs(avg_diag) > 1e-12 else float("nan")
    max_abs_ratio = float(
        max((abs(r) for r in ratios if math.isfinite(r)), default=float("nan"))
    )

    hypothesis_confirmed = math.isfinite(avg_ratio) and abs(avg_ratio) > 0.2

    return {
        "status": "ok",
        "target_layer_index": target_layer_index,
        "num_heads": nh,
        "head_dim": hd,
        "batch_size_used": int(B),
        "seq_len_used": int(T),
        "sample_tokens": int(sample_size),
        "per_head": per_head_stats,
        "average_diagonal_mean": avg_diag,
        "average_off_diagonal_mean": avg_off_diag,
        "average_off_diagonal_abs_mean": avg_off_diag_abs,
        "average_ratio_off_over_diag": avg_ratio,
        "average_ratio_abs_off_over_diag": avg_ratio_abs,
        "max_abs_ratio_off_over_diag": max_abs_ratio,
        "pass_criterion_abs_ratio_gt_0p2": hypothesis_confirmed,
        "hypothesis_keys_correlated_confirmed": hypothesis_confirmed,
    }


def main() -> int:
    overall_start = time.time()
    _set_all_seeds(SEED)

    report: dict[str, Any] = {
        "script": "tests/test_delta_deadweight_and_key_gram.py",
        "seed": SEED,
        "device": str(god_machine.DEVICE),
        "time_budget_sec": TIME_BUDGET_SEC,
    }

    tmp_root = Path(tempfile.mkdtemp(prefix="delta_deadweight_"))
    output_run_dir = tmp_root / "run"
    output_run_dir.mkdir(parents=True, exist_ok=False)

    train_size = 3_800_000
    val_size = 200_000
    total_size = train_size + val_size

    try:
        report["synthetic_data_bytes_total"] = total_size
        synthetic = _make_synthetic_bytes(total_size, seed=SEED)
        train_np = synthetic[:train_size].copy()
        val_np = synthetic[train_size:].copy()

        train_bin = tmp_root / "train.bin"
        val_bin = tmp_root / "val.bin"
        train_bin.write_bytes(train_np.tobytes())
        val_bin.write_bytes(val_np.tobytes())
        report["train_bin"] = str(train_bin)
        report["val_bin"] = str(val_bin)
        report["train_bytes"] = int(train_np.size)
        report["val_bytes"] = int(val_np.size)

        cfg = _build_cfg()
        report["config"] = {
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "layer_pattern": list(cfg.layer_pattern),
            "delta_num_heads": cfg.delta_num_heads,
            "delta_head_dim": cfg.delta_head_dim,
            "attn_d_c": cfg.attn_d_c,
            "attn_d_R": cfg.attn_d_R,
            "attn_num_heads": cfg.attn_num_heads,
            "mlp_ratio": cfg.mlp_ratio,
            "batch_size": cfg.batch_size,
            "seq_len": cfg.seq_len,
            "max_steps": cfg.max_steps,
            "warmup_steps": cfg.warmup_steps,
            "val_interval": cfg.val_interval,
            "use_fla_if_available": cfg.use_fla_if_available,
            "kwta_enabled": cfg.kwta_enabled,
            "delta_erasure_enabled": cfg.delta_erasure_enabled,
            "bcm_alpha_enabled": cfg.bcm_alpha_enabled,
            "multi_compartment_enabled": cfg.multi_compartment_enabled,
            "imagination_enabled": cfg.imagination_enabled,
            "pc_diagnostic_enabled": cfg.pc_diagnostic_enabled,
        }

        print(f"[experiment] device={god_machine.DEVICE} seed={SEED}", flush=True)
        print(f"[experiment] train_bytes={train_np.size} val_bytes={val_np.size}", flush=True)
        print(f"[experiment] cfg={report['config']}", flush=True)

        print("[experiment] starting experiment A (dead-weight ablation)", flush=True)
        try:
            exp_a = run_experiment_a(cfg, output_run_dir, train_np, val_np)
        except Exception as exc:
            exp_a = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        report["experiment_a"] = exp_a
        print(f"[experiment] experiment A result: {exp_a}", flush=True)

        if (time.time() - overall_start) > TIME_BUDGET_SEC:
            raise TimeoutError("time budget exceeded before experiment B")

        print("[experiment] starting experiment B (key gram diagnostic)", flush=True)
        try:
            exp_b = run_experiment_b(cfg, output_run_dir, val_np)
        except Exception as exc:
            exp_b = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        report["experiment_b"] = exp_b
        print(f"[experiment] experiment B summary: status={exp_b.get('status')}", flush=True)

        report["elapsed_sec"] = time.time() - overall_start
        report["status"] = "complete"

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str, allow_nan=True)
        print(f"[experiment] wrote json report to {OUTPUT_PATH}", flush=True)

        print("\n=== final summary ===", flush=True)
        a = report["experiment_a"]
        b = report["experiment_b"]
        if a.get("status") == "ok":
            print(
                f"A baseline_val_bpb={a['baseline_val_bpb_reevaluated']:.6f} "
                f"zeroed_val_bpb={a['zeroed_val_bpb']:.6f} "
                f"rel_change={a['relative_change']:.6f} "
                f"dead_weight_confirmed={a['hypothesis_dead_weight_confirmed']}",
                flush=True,
            )
        else:
            print(f"A failed: {a.get('reason') or a.get('error')}", flush=True)
        if b.get("status") == "ok":
            print(
                f"B avg_diag={b['average_diagonal_mean']:.6f} "
                f"avg_off_diag={b['average_off_diagonal_mean']:.6f} "
                f"avg_ratio={b['average_ratio_off_over_diag']:.6f} "
                f"max_abs_ratio={b['max_abs_ratio_off_over_diag']:.6f} "
                f"keys_correlated_confirmed={b['hypothesis_keys_correlated_confirmed']}",
                flush=True,
            )
            for h in b["per_head"]:
                print(
                    f"  head {h['head']}: diag={h['diagonal_mean']:.6f} "
                    f"off_diag={h['off_diagonal_mean']:.6f} "
                    f"ratio={h['ratio_off_over_diag']:.6f}",
                    flush=True,
                )
        else:
            print(f"B failed: {b.get('reason') or b.get('error')}", flush=True)

        return 0

    except Exception as exc:
        report["status"] = "error"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        report["elapsed_sec"] = time.time() - overall_start
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str, allow_nan=True)
        print(f"[experiment] fatal: {exc}", flush=True)
        traceback.print_exc()
        return 1

    finally:
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception:
            pass


def test_delta_deadweight_and_key_gram() -> None:
    rc = main()
    assert rc == 0, f"experiment returned non-zero exit code {rc}"


if __name__ == "__main__":
    sys.exit(main())
