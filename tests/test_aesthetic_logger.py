from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neuroloc.model.aesthetic_logger import (
    PANEL_FILENAMES,
    AestheticLogger,
    main,
    read_metrics_jsonl,
    render_from_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")


def _build_synthetic_records(n_steps: int, n_layers: int) -> list[dict]:
    records: list[dict] = []
    records.append({"event": "run_start", "name": "synthetic", "step": 0})
    for step in range(n_steps):
        frob = [1.0 + 0.1 * step + 0.05 * layer for layer in range(n_layers)]
        alpha = [0.6 + 0.01 * step - 0.02 * layer for layer in range(n_layers)]
        beta = [0.4 + 0.005 * step + 0.01 * layer for layer in range(n_layers)]
        erasure = [1.0 if layer % 2 == 0 else 0.0 for layer in range(n_layers)]
        path_flags = [1.0 if step % 3 == 0 else 0.0 for _ in range(n_layers)]
        rec = {
            "event": "step",
            "step": step,
            "loss": 5.0 - 0.2 * step,
            "state_frobenius_per_layer": frob,
            "alpha_eff_mean_per_layer": alpha,
            "beta_mean_per_layer": beta,
            "delta_erasure_flag_per_layer": erasure,
            "delta_path_per_layer": path_flags,
        }
        records.append(rec)
        if step % 4 == 3:
            records.append(
                {
                    "event": "validation",
                    "step": step,
                    "val_bpb": 2.0 - 0.05 * step,
                    "val_loss": 1.3 - 0.03 * step,
                }
            )
    return records


def test_render_from_jsonl_produces_four_pngs(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "metrics.jsonl"
    out_dir = tmp_path / "aesthetic"
    _write_jsonl(jsonl_path, _build_synthetic_records(n_steps=10, n_layers=4))

    produced = render_from_jsonl(jsonl_path, out_dir)

    assert set(produced.keys()) == set(PANEL_FILENAMES.keys())
    for panel_name, filename in PANEL_FILENAMES.items():
        path = out_dir / filename
        assert path.exists(), f"missing {path}"
        assert path.stat().st_size > 0, f"empty {path}"
        assert produced[panel_name] == path


def test_missing_keys_do_not_crash(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "metrics.jsonl"
    out_dir = tmp_path / "aesthetic_missing"
    records: list[dict] = []
    records.append({"event": "run_start", "step": 0})
    for step in range(6):
        rec: dict = {"event": "step", "step": step, "loss": 3.0 - 0.1 * step}
        if step % 2 == 0:
            rec["state_frobenius_per_layer"] = [1.0 + 0.1 * step, 1.1 + 0.1 * step]
        if step % 3 == 0:
            rec["alpha_eff_mean_per_layer"] = [0.5, 0.55]
        if step == 5:
            rec["delta_erasure_flag_per_layer"] = [1.0, 0.0]
            rec["beta_mean_per_layer"] = [0.3, 0.4]
        records.append(rec)
    records.append({"event": "validation", "step": 5, "val_bpb": 1.7})
    _write_jsonl(jsonl_path, records)

    produced = render_from_jsonl(jsonl_path, out_dir)

    assert "loss_bpb" in produced
    for panel, path in produced.items():
        assert path.exists()
        assert path.stat().st_size > 0


def test_partial_keys_across_steps(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "metrics.jsonl"
    out_dir = tmp_path / "aesthetic_partial"
    records = [
        {"event": "step", "step": 0, "loss": 1.0, "state_frobenius_per_layer": [1.0, 2.0]},
        {"event": "step", "step": 1, "loss": 0.9, "alpha_eff_mean_per_layer": [0.5, 0.6]},
        {"event": "step", "step": 2, "loss": 0.8, "state_frobenius_per_layer": [1.1, 2.1]},
        {"event": "validation", "step": 2, "val_bpb": 1.4},
    ]
    _write_jsonl(jsonl_path, records)

    produced = render_from_jsonl(jsonl_path, out_dir)
    assert (out_dir / PANEL_FILENAMES["loss_bpb"]).exists()
    assert (out_dir / PANEL_FILENAMES["state_frobenius_heatmap"]).exists()
    assert (out_dir / PANEL_FILENAMES["alpha_eff_timeseries"]).exists()
    for path in produced.values():
        assert path.stat().st_size > 0


def test_non_finite_values_handled(tmp_path: Path) -> None:
    import json as _json
    jsonl_path = tmp_path / "metrics.jsonl"
    out_dir = tmp_path / "aesthetic_nan"
    records = [
        {"event": "step", "step": 0, "loss": 1.0, "state_frobenius_per_layer": [1.0, None, 2.0]},
        {"event": "step", "step": 1, "loss": 0.9, "state_frobenius_per_layer": [1.1, 2.1, None], "alpha_eff_mean_per_layer": [0.5, None, 0.6]},
        {"event": "step", "step": 2, "loss": 0.8, "state_frobenius_per_layer": [None, 2.2, 3.0], "alpha_eff_mean_per_layer": [0.55, 0.55, None]},
        {"event": "validation", "step": 2, "val_bpb": 1.4},
    ]
    lines = []
    for record in records:
        lines.append(_json.dumps(record))
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    produced = render_from_jsonl(jsonl_path, out_dir)
    assert (out_dir / PANEL_FILENAMES["loss_bpb"]).exists()
    assert (out_dir / PANEL_FILENAMES["state_frobenius_heatmap"]).exists()
    assert (out_dir / PANEL_FILENAMES["alpha_eff_timeseries"]).exists()
    for path in produced.values():
        assert path.stat().st_size > 0


def test_step_range_filter_rejects_float_step_out_of_range(tmp_path: Path) -> None:
    import json as _json
    jsonl_path = tmp_path / "metrics.jsonl"
    out_dir = tmp_path / "aesthetic_float_step"
    records = [
        {"event": "step", "step": 0, "loss": 1.0, "state_frobenius_per_layer": [1.0]},
        {"event": "step", "step": 5.0, "loss": 0.9, "state_frobenius_per_layer": [1.1]},
        {"event": "step", "step": 10.0, "loss": 0.8, "state_frobenius_per_layer": [1.2]},
    ]
    lines = []
    for record in records:
        lines.append(_json.dumps(record))
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger = AestheticLogger(jsonl_path, out_dir)
    logger.records = __import__("neuroloc.model.aesthetic_logger", fromlist=["read_metrics_jsonl"]).read_metrics_jsonl(jsonl_path, step_range=(1, 7))
    assert [int(r["step"]) for r in logger.records] == [5]


def test_empty_jsonl_produces_no_panels(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "metrics.jsonl"
    out_dir = tmp_path / "aesthetic_empty"
    jsonl_path.write_text("", encoding="utf-8")

    produced = render_from_jsonl(jsonl_path, out_dir)

    assert produced == {}
    assert out_dir.exists()


def test_step_range_filters_records(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "metrics.jsonl"
    _write_jsonl(jsonl_path, _build_synthetic_records(n_steps=20, n_layers=3))

    filtered = read_metrics_jsonl(jsonl_path, step_range=(5, 9))
    step_values = [rec.get("step") for rec in filtered if rec.get("event") == "step"]
    assert step_values == [5, 6, 7, 8, 9]


def test_malformed_lines_are_skipped(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "metrics.jsonl"
    records = _build_synthetic_records(n_steps=5, n_layers=3)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")
        handle.write("not json\n")
        handle.write("{\n")
        handle.write("\n")
    out_dir = tmp_path / "aesthetic_malformed"

    produced = render_from_jsonl(jsonl_path, out_dir)
    assert "loss_bpb" in produced


def test_aesthetic_logger_class_exposes_methods(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "metrics.jsonl"
    out_dir = tmp_path / "aesthetic_class"
    _write_jsonl(jsonl_path, _build_synthetic_records(n_steps=8, n_layers=4))

    logger = AestheticLogger(jsonl_path=jsonl_path, output_dir=out_dir)
    records = logger.load_records()
    assert records, "expected records to be loaded"
    produced = logger.render_all()
    assert set(produced) <= set(PANEL_FILENAMES)
    assert "loss_bpb" in produced


def test_cli_entry_point_returns_zero_on_success(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "metrics.jsonl"
    out_dir = tmp_path / "aesthetic_cli"
    _write_jsonl(jsonl_path, _build_synthetic_records(n_steps=6, n_layers=3))

    rc = main(["--jsonl", str(jsonl_path), "--out-dir", str(out_dir)])
    assert rc == 0
    assert (out_dir / PANEL_FILENAMES["loss_bpb"]).exists()


def test_cli_entry_point_missing_jsonl_returns_error(tmp_path: Path) -> None:
    rc = main(["--jsonl", str(tmp_path / "nope.jsonl"), "--out-dir", str(tmp_path / "out")])
    assert rc == 2


def test_integration_arbitrary_jsonl_does_not_crash(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "neuroloc" / "output" / "simulation_suites" / "correction_field_capacity" / "correction_field_capacity_metrics.json"
    if not source.exists():
        pytest.skip(f"benchmark artifact not available: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))

    trials = payload.get("trials")
    if not isinstance(trials, list) or not trials:
        pytest.skip("benchmark artifact has no trials to adapt")

    jsonl_path = tmp_path / "adapted.jsonl"
    lines: list[dict] = []
    for idx, trial in enumerate(trials[:40]):
        if not isinstance(trial, dict):
            continue
        rec: dict = {"event": "step", "step": idx}
        for key, value in trial.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)) and not isinstance(value, bool):
                rec[str(key)] = float(value)
        lines.append(rec)
    _write_jsonl(jsonl_path, lines)

    out_dir = tmp_path / "integration"
    produced = render_from_jsonl(jsonl_path, out_dir)
    assert isinstance(produced, dict)
    for path in produced.values():
        assert path.exists()
        assert path.stat().st_size > 0
