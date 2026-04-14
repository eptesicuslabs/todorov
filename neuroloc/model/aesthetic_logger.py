from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg", force=True)


try:
    from neuroloc.simulations.shared import apply_plot_style as _shared_apply_plot_style

    def apply_plot_style() -> None:
        _shared_apply_plot_style()
except Exception:
    def apply_plot_style() -> None:
        import matplotlib.pyplot as plt

        plt.rcParams.update(
            {
                "figure.dpi": 150,
                "savefig.dpi": 150,
                "font.size": 10,
                "axes.titlesize": 12,
                "axes.labelsize": 11,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": True,
                "grid.alpha": 0.18,
                "legend.frameon": False,
                "lines.linewidth": 2.0,
            }
        )


PANEL_FILENAMES: dict[str, str] = {
    "loss_bpb": "loss_bpb.png",
    "state_frobenius_heatmap": "state_frobenius_heatmap.png",
    "alpha_eff_timeseries": "alpha_eff_timeseries.png",
    "layer_snapshot": "layer_snapshot.png",
}


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def _coerce_numeric_list(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    out: list[float] = []
    for item in value:
        if _is_finite_number(item):
            out.append(float(item))
        elif item is None:
            out.append(float("nan"))
        else:
            return None
    return out


def _in_step_range(step: int, step_range: tuple[int, int] | None) -> bool:
    if step_range is None:
        return True
    lo, hi = step_range
    return lo <= step <= hi


def read_metrics_jsonl(path: Path, step_range: tuple[int, int] | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            step = parsed.get("step")
            if step_range is not None:
                step_int: int | None
                if isinstance(step, bool):
                    step_int = None
                elif isinstance(step, int):
                    step_int = step
                elif isinstance(step, float) and math.isfinite(step) and float(step).is_integer():
                    step_int = int(step)
                else:
                    step_int = None
                if step_int is None or not _in_step_range(step_int, step_range):
                    continue
            records.append(parsed)
    return records


class AestheticLogger:

    def __init__(self, jsonl_path: Path | str, output_dir: Path | str) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_records(self, step_range: tuple[int, int] | None = None) -> list[dict[str, Any]]:
        if not self.jsonl_path.exists():
            return []
        return read_metrics_jsonl(self.jsonl_path, step_range=step_range)

    def render_loss_bpb(self, records: list[dict[str, Any]]) -> Path | None:
        import matplotlib.pyplot as plt

        train_steps: list[int] = []
        train_losses: list[float] = []
        val_steps: list[int] = []
        val_bpbs: list[float] = []

        for rec in records:
            step = rec.get("step")
            event = rec.get("event")
            if not isinstance(step, int):
                continue
            if event == "step":
                loss = rec.get("loss")
                if _is_finite_number(loss):
                    train_steps.append(step)
                    train_losses.append(float(loss))
            elif event == "validation":
                bpb = rec.get("val_bpb")
                if bpb is None:
                    bpb = rec.get("bpb")
                if _is_finite_number(bpb):
                    val_steps.append(step)
                    val_bpbs.append(float(bpb))

        if not train_steps and not val_steps:
            return None

        apply_plot_style()
        fig, (ax_loss, ax_bpb) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        if train_steps:
            ax_loss.plot(train_steps, train_losses, color="#1f77b4", linewidth=0.8)
            ax_loss.set_ylabel("train loss")
            ax_loss.set_title("training loss")
        else:
            ax_loss.text(0.5, 0.5, "no training loss recorded", ha="center", va="center", transform=ax_loss.transAxes)
            ax_loss.set_title("training loss")

        if val_steps:
            ax_bpb.plot(val_steps, val_bpbs, color="#d62728", marker="o", linewidth=1.4)
            ax_bpb.set_ylabel("val bpb")
            ax_bpb.set_title("validation bits per byte")
        else:
            ax_bpb.text(0.5, 0.5, "no validation bpb recorded", ha="center", va="center", transform=ax_bpb.transAxes)
            ax_bpb.set_title("validation bits per byte")
        ax_bpb.set_xlabel("step")

        fig.tight_layout()
        out_path = self.output_dir / PANEL_FILENAMES["loss_bpb"]
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def render_state_frobenius_heatmap(self, records: list[dict[str, Any]]) -> Path | None:
        import numpy as np
        import matplotlib.pyplot as plt

        steps: list[int] = []
        per_step: list[list[float]] = []
        for rec in records:
            step = rec.get("step")
            if not isinstance(step, int):
                continue
            series = _coerce_numeric_list(rec.get("state_frobenius_per_layer"))
            if series is None or not series:
                continue
            steps.append(step)
            per_step.append(series)

        if not steps:
            return None

        max_layers = max(len(row) for row in per_step)
        padded: list[list[float]] = [row + [float("nan")] * (max_layers - len(row)) for row in per_step]
        arr = np.asarray(padded, dtype=float).T

        apply_plot_style()
        fig, ax = plt.subplots(figsize=(11, 4.5))
        if len(steps) > 1:
            step_width = (steps[-1] - steps[0]) / max(len(steps) - 1, 1)
        else:
            step_width = 1.0
        x_left = steps[0] - step_width / 2.0
        x_right = steps[-1] + step_width / 2.0
        im = ax.imshow(
            arr,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
            extent=(x_left, x_right, 0, max_layers),
        )
        ax.set_xlabel("step")
        ax.set_ylabel("layer index")
        ax.set_title("state_frobenius_per_layer over training")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="state frobenius")
        fig.tight_layout()
        out_path = self.output_dir / PANEL_FILENAMES["state_frobenius_heatmap"]
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def render_alpha_eff_timeseries(self, records: list[dict[str, Any]]) -> Path | None:
        import matplotlib.pyplot as plt

        steps: list[int] = []
        per_step: list[list[float]] = []
        for rec in records:
            step = rec.get("step")
            if not isinstance(step, int):
                continue
            series = _coerce_numeric_list(rec.get("alpha_eff_mean_per_layer"))
            if series is None or not series:
                continue
            steps.append(step)
            per_step.append(series)

        if not steps:
            return None

        max_layers = max(len(row) for row in per_step)
        per_layer_x: list[list[int]] = [[] for _ in range(max_layers)]
        per_layer_y: list[list[float]] = [[] for _ in range(max_layers)]
        for step, row in zip(steps, per_step):
            for layer_idx in range(max_layers):
                if layer_idx >= len(row):
                    continue
                value = row[layer_idx]
                if not math.isfinite(value):
                    continue
                per_layer_x[layer_idx].append(step)
                per_layer_y[layer_idx].append(value)

        apply_plot_style()
        fig, ax = plt.subplots(figsize=(11, 5))
        cmap = plt.get_cmap("viridis")
        denom = max(max_layers - 1, 1)
        drew_any = False
        for layer_idx in range(max_layers):
            xs = per_layer_x[layer_idx]
            ys = per_layer_y[layer_idx]
            if not xs:
                continue
            color = cmap(layer_idx / denom)
            ax.plot(xs, ys, color=color, linewidth=1.0, label=f"layer {layer_idx}")
            drew_any = True
        if not drew_any:
            plt.close(fig)
            return None

        ax.set_xlabel("step")
        ax.set_ylabel("alpha_eff mean")
        ax.set_title("alpha_eff_mean_per_layer over training (color by layer depth)")
        if max_layers <= 12:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
        else:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_layers - 1))
            sm.set_array([])
            fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="layer index")
        fig.tight_layout()
        out_path = self.output_dir / PANEL_FILENAMES["alpha_eff_timeseries"]
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def render_layer_snapshot(self, records: list[dict[str, Any]]) -> Path | None:
        import matplotlib.pyplot as plt

        snapshot_keys = (
            "delta_erasure_flag_per_layer",
            "delta_path_per_layer",
            "beta_mean_per_layer",
        )
        colors = ("#1f77b4", "#2ca02c", "#d62728")

        latest_step: int | None = None
        latest_values: dict[str, list[float]] = {}
        for rec in records:
            step = rec.get("step")
            if not isinstance(step, int):
                continue
            candidate: dict[str, list[float]] = {}
            for key in snapshot_keys:
                series = _coerce_numeric_list(rec.get(key))
                if series is not None and series:
                    candidate[key] = series
            if not candidate:
                continue
            if latest_step is None or step >= latest_step:
                latest_step = step
                latest_values = candidate

        if latest_step is None or not latest_values:
            return None

        present_keys = [key for key in snapshot_keys if key in latest_values]
        if not present_keys:
            return None

        apply_plot_style()
        fig, axes = plt.subplots(len(present_keys), 1, figsize=(11, 2.2 * len(present_keys)), squeeze=False)
        for idx, key in enumerate(present_keys):
            values = latest_values[key]
            clean_values = [v if math.isfinite(v) else 0.0 for v in values]
            ax = axes[idx][0]
            color = colors[snapshot_keys.index(key) % len(colors)]
            xs = list(range(len(clean_values)))
            ax.bar(xs, clean_values, color=color, width=0.72)
            ax.set_ylabel(key.replace("_per_layer", ""))
            if idx == len(present_keys) - 1:
                ax.set_xlabel("layer index")
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle(f"per-layer snapshot at step {latest_step}")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out_path = self.output_dir / PANEL_FILENAMES["layer_snapshot"]
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def render_all(self, step_range: tuple[int, int] | None = None) -> dict[str, Path]:
        records = self.load_records(step_range=step_range)
        produced: dict[str, Path] = {}
        for panel_name, renderer in (
            ("loss_bpb", self.render_loss_bpb),
            ("state_frobenius_heatmap", self.render_state_frobenius_heatmap),
            ("alpha_eff_timeseries", self.render_alpha_eff_timeseries),
            ("layer_snapshot", self.render_layer_snapshot),
        ):
            path = renderer(records)
            if path is not None:
                produced[panel_name] = path
        return produced


def render_from_jsonl(
    jsonl_path: Path | str,
    output_dir: Path | str,
    step_range: tuple[int, int] | None = None,
) -> dict[str, Path]:
    logger = AestheticLogger(jsonl_path=jsonl_path, output_dir=output_dir)
    return logger.render_all(step_range=step_range)


def _parse_step_range(raw: str | None) -> tuple[int, int] | None:
    if raw is None:
        return None
    parts = raw.split(":")
    if len(parts) != 2:
        raise ValueError(f"step range must be 'lo:hi', got {raw!r}")
    lo = int(parts[0])
    hi = int(parts[1])
    if hi < lo:
        raise ValueError(f"step range requires lo <= hi, got {lo}..{hi}")
    return lo, hi


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aesthetic_logger",
        description="render aesthetic panels from a god_machine metrics.jsonl",
    )
    parser.add_argument("--jsonl", required=True, help="path to metrics jsonl file")
    parser.add_argument("--out-dir", required=True, help="directory to write png panels")
    parser.add_argument(
        "--step-range",
        default=None,
        help="optional step range in the form 'lo:hi' (inclusive); filters step records",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    step_range = _parse_step_range(args.step_range)
    jsonl_path = Path(args.jsonl)
    out_dir = Path(args.out_dir)
    if not jsonl_path.exists():
        print(f"error: jsonl path does not exist: {jsonl_path}", file=sys.stderr)
        return 2
    produced = render_from_jsonl(jsonl_path, out_dir, step_range=step_range)
    if not produced:
        print(f"warning: no panels produced from {jsonl_path}")
        return 1
    for panel, path in produced.items():
        print(f"{panel}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
