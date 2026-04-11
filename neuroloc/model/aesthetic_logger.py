from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _require_matplotlib() -> Any:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


class AestheticLogger:

    def __init__(self, run_dir: Path, run_name: str) -> None:
        self.run_dir = Path(run_dir)
        self.run_name = run_name
        self.aesthetic_dir = self.run_dir / "aesthetic"
        self.aesthetic_dir.mkdir(parents=True, exist_ok=True)

    def save_val_snapshot(self, step: int, history: dict[str, list], cfg_summary: dict | None = None) -> list[Path]:
        plt = _require_matplotlib()
        step_dir = self.aesthetic_dir / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        out_paths: list[Path] = []

        train_steps = history.get("steps", [])
        train_loss = history.get("train_loss", [])
        val_steps = history.get("val_steps", [])
        val_bpb = history.get("val_bpb", [])

        if train_steps and train_loss:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(train_steps, train_loss, color="#1f77b4", linewidth=0.8, label="train loss")
            if val_steps and val_bpb:
                ax_val = ax.twinx()
                ax_val.plot(val_steps, val_bpb, color="#d62728", marker="o", linewidth=1.5, label="val bpb")
                ax_val.set_ylabel("val bpb", color="#d62728")
                ax_val.tick_params(axis="y", labelcolor="#d62728")
            ax.set_xlabel("step")
            ax.set_ylabel("train loss", color="#1f77b4")
            ax.tick_params(axis="y", labelcolor="#1f77b4")
            ax.set_title(f"{self.run_name} loss and val bpb at step {step}")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            p = step_dir / "01_loss_bpb.png"
            fig.savefig(p, dpi=110)
            plt.close(fig)
            out_paths.append(p)

        probe_keys_timeseries: list[tuple[str, str, str]] = [
            ("alpha_eff_mean_mean", "#2ca02c", "alpha_eff mean (BCM effective decay)"),
            ("imag_ratio_mean", "#9467bd", "imag_ratio mean (imagination contribution)"),
            ("pc_error_l2_mean", "#ff7f0e", "pc_error_l2 mean (diagnostic)"),
            ("state_frobenius_mean", "#8c564b", "state frobenius mean"),
            ("imag_gate_mean_mean", "#e377c2", "imag gate mean"),
        ]
        active_series: list[tuple[str, str, str, list]] = []
        for key, color, label in probe_keys_timeseries:
            series = history.get(key)
            if isinstance(series, list) and series and any(isinstance(v, (int, float)) and math.isfinite(v) for v in series if v is not None):
                clean = [v if isinstance(v, (int, float)) and math.isfinite(v) else float("nan") for v in series]
                active_series.append((key, color, label, clean))

        if active_series and train_steps:
            n = len(active_series)
            ncols = 2 if n > 1 else 1
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows), squeeze=False)
            for i, (key, color, label, series) in enumerate(active_series):
                r, c = divmod(i, ncols)
                ax = axes[r][c]
                xs = train_steps[: len(series)]
                ax.plot(xs, series, color=color, linewidth=0.9)
                ax.set_xlabel("step")
                ax.set_title(label, fontsize=10)
                ax.grid(alpha=0.2)
            for i in range(n, nrows * ncols):
                r, c = divmod(i, ncols)
                axes[r][c].axis("off")
            fig.suptitle(f"{self.run_name} probe metrics at step {step}", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            p = step_dir / "02_probes.png"
            fig.savefig(p, dpi=110)
            plt.close(fig)
            out_paths.append(p)

        per_layer_keys = [
            ("state_frobenius_per_layer", "delta state frobenius per layer"),
            ("alpha_eff_mean_per_layer", "alpha_eff per layer"),
            ("imag_ratio_per_layer", "imag_ratio per layer"),
            ("pc_error_l2_per_layer", "pc_error_l2 per layer"),
            ("kwta_k_rate_per_layer", "kwta k-rate per layer"),
            ("mlp_compartment_l2_per_layer", "mlp compartment l2 per layer"),
        ]
        heatmap_series: list[tuple[str, str, list]] = []
        for key, label in per_layer_keys:
            hist_per_layer = history.get(key)
            if not isinstance(hist_per_layer, list) or not hist_per_layer:
                continue
            last = hist_per_layer[-1]
            if isinstance(last, list) and last:
                heatmap_series.append((key, label, hist_per_layer))

        if heatmap_series:
            import numpy as np
            n = len(heatmap_series)
            fig, axes = plt.subplots(n, 1, figsize=(11, 2.0 * n), squeeze=False)
            for i, (key, label, series) in enumerate(heatmap_series):
                ax = axes[i][0]
                cleaned: list[list[float]] = []
                for row in series:
                    if isinstance(row, list):
                        cleaned.append([
                            v if isinstance(v, (int, float)) and math.isfinite(v) else float("nan")
                            for v in row
                        ])
                if not cleaned:
                    ax.axis("off")
                    continue
                max_layers = max(len(r) for r in cleaned)
                grid = [r + [float("nan")] * (max_layers - len(r)) for r in cleaned]
                arr = np.asarray(grid, dtype=float).T
                im = ax.imshow(arr, aspect="auto", origin="lower", cmap="viridis", interpolation="nearest")
                ax.set_title(label, fontsize=10)
                ax.set_ylabel("layer")
                ax.set_xlabel("val step index" if i == n - 1 else "")
                fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            fig.suptitle(f"{self.run_name} per-layer evolution at step {step}", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            p = step_dir / "03_per_layer_heatmaps.png"
            fig.savefig(p, dpi=110)
            plt.close(fig)
            out_paths.append(p)

        latest_per_layer_keys = [
            ("state_frobenius_per_layer", "delta state frobenius", "#8c564b"),
            ("alpha_eff_mean_per_layer", "alpha_eff", "#2ca02c"),
            ("imag_ratio_per_layer", "imag_ratio", "#9467bd"),
            ("pc_error_l2_per_layer", "pc_error_l2", "#ff7f0e"),
        ]
        latest_bars: list[tuple[str, str, list]] = []
        for key, label, color in latest_per_layer_keys:
            series = history.get(key)
            if isinstance(series, list) and series:
                last = series[-1]
                if isinstance(last, list) and last:
                    latest_bars.append((label, color, [
                        v if isinstance(v, (int, float)) and math.isfinite(v) else 0.0
                        for v in last
                    ]))

        if latest_bars:
            n = len(latest_bars)
            fig, axes = plt.subplots(n, 1, figsize=(11, 1.8 * n), squeeze=False)
            for i, (label, color, values) in enumerate(latest_bars):
                ax = axes[i][0]
                xs = list(range(len(values)))
                ax.bar(xs, values, color=color, width=0.7)
                ax.set_title(label, fontsize=10)
                ax.set_ylabel(label)
                if i == n - 1:
                    ax.set_xlabel("layer")
                ax.grid(axis="y", alpha=0.2)
            fig.suptitle(f"{self.run_name} latest per-layer snapshot at step {step}", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            p = step_dir / "04_latest_bars.png"
            fig.savefig(p, dpi=110)
            plt.close(fig)
            out_paths.append(p)

        summary = {
            "step": step,
            "num_images": len(out_paths),
            "images": [p.name for p in out_paths],
            "config": cfg_summary or {},
        }
        with open(step_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        return out_paths
