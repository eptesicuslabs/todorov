import numpy as np
import pytest

from neuroloc.simulations.memory import correction_field_capacity
from neuroloc.simulations.shared import validate_metrics_file


def test_build_value_decomposition_matches_requested_energy_fraction() -> None:
    rng = np.random.default_rng(0)
    qualities = np.array([0.0, 0.3, 0.5, 0.9, 0.99], dtype=np.float64)
    prediction, residual, value, surprise_ratio = correction_field_capacity.build_value_decomposition(rng, qualities, 64)
    prediction_fraction = np.sum(prediction ** 2, axis=-1) / np.sum(value ** 2, axis=-1)
    residual_fraction = np.sum(residual ** 2, axis=-1) / np.sum(value ** 2, axis=-1)
    assert np.allclose(prediction_fraction, qualities ** 2, atol=1e-6)
    assert np.allclose(residual_fraction, 1.0 - qualities ** 2, atol=1e-6)
    assert np.allclose(surprise_ratio, 1.0 - qualities ** 2, atol=1e-6)


def test_recall_metrics_separate_prediction_memory_and_reconstruction() -> None:
    rng = np.random.default_rng(1)
    qualities = np.full(8, 0.9, dtype=np.float64)
    keys = correction_field_capacity.normalize_rows(rng.standard_normal((8, 32)))
    prediction, residual, value, _ = correction_field_capacity.build_value_decomposition(rng, qualities, 32)
    residual_state = correction_field_capacity.build_state(keys, residual, decay=1.0)
    prediction_only = correction_field_capacity.evaluate_recall(
        keys,
        value,
        np.zeros_like(residual_state),
        reconstruction_base=prediction,
    )
    memory_only = correction_field_capacity.evaluate_recall(keys, residual, residual_state)
    reconstruction = correction_field_capacity.evaluate_recall(
        keys,
        value,
        residual_state,
        reconstruction_base=prediction,
    )
    assert prediction_only["mean_cosine_recall"] < reconstruction["mean_cosine_recall"]
    assert memory_only["mean_cosine_recall"] != reconstruction["mean_cosine_recall"]
    assert reconstruction["mean_cosine_recall"] <= 1.0


def test_surprise_ratio_separates_predictable_and_novel_tokens() -> None:
    rng = np.random.default_rng(2)
    qualities = np.array([0.99, 0.99, 0.0, 0.0], dtype=np.float64)
    _, _, _, surprise_ratio = correction_field_capacity.build_value_decomposition(rng, qualities, 32)
    assert float(np.max(surprise_ratio[:2])) < 0.05
    assert float(np.min(surprise_ratio[2:])) > 0.95


def test_summary_uses_memory_metrics_for_capacity_and_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(correction_field_capacity, "HEAD_DIMS", [32])
    monkeypatch.setattr(correction_field_capacity, "DECAYS", [0.5, 0.9])
    monkeypatch.setattr(correction_field_capacity, "PREDICTION_QUALITIES", [0.9])
    monkeypatch.setattr(correction_field_capacity, "PATTERN_COUNTS", [4, 8])
    records = [
        {
            "sweep": "capacity",
            "head_dim": 32,
            "decay": 0.5,
            "prediction_quality": 0.9,
            "pattern_count": 4,
            "raw_mean_cosine_recall": 0.4,
            "correction_residual_mean_cosine_recall": 0.6,
            "prediction_only_mean_cosine_recall": 0.9,
            "correction_reconstruction_mean_cosine_recall": 0.95,
            "raw_exact_match_accuracy": 0.0,
            "correction_residual_exact_match_accuracy": 1.0,
            "prediction_only_exact_match_accuracy": 1.0,
            "correction_reconstruction_exact_match_accuracy": 1.0,
        },
        {
            "sweep": "capacity",
            "head_dim": 32,
            "decay": 0.5,
            "prediction_quality": 0.9,
            "pattern_count": 8,
            "raw_mean_cosine_recall": 0.4,
            "correction_residual_mean_cosine_recall": 0.4,
            "prediction_only_mean_cosine_recall": 0.9,
            "correction_reconstruction_mean_cosine_recall": 0.95,
            "raw_exact_match_accuracy": 0.0,
            "correction_residual_exact_match_accuracy": 0.0,
            "prediction_only_exact_match_accuracy": 1.0,
            "correction_reconstruction_exact_match_accuracy": 1.0,
        },
        {
            "sweep": "capacity",
            "head_dim": 32,
            "decay": 0.9,
            "prediction_quality": 0.9,
            "pattern_count": 4,
            "raw_mean_cosine_recall": 0.6,
            "correction_residual_mean_cosine_recall": 0.6,
            "prediction_only_mean_cosine_recall": 0.9,
            "correction_reconstruction_mean_cosine_recall": 0.95,
            "raw_exact_match_accuracy": 1.0,
            "correction_residual_exact_match_accuracy": 1.0,
            "prediction_only_exact_match_accuracy": 1.0,
            "correction_reconstruction_exact_match_accuracy": 1.0,
        },
        {
            "sweep": "capacity",
            "head_dim": 32,
            "decay": 0.9,
            "prediction_quality": 0.9,
            "pattern_count": 8,
            "raw_mean_cosine_recall": 0.6,
            "correction_residual_mean_cosine_recall": 0.6,
            "prediction_only_mean_cosine_recall": 0.9,
            "correction_reconstruction_mean_cosine_recall": 0.95,
            "raw_exact_match_accuracy": 1.0,
            "correction_residual_exact_match_accuracy": 1.0,
            "prediction_only_exact_match_accuracy": 1.0,
            "correction_reconstruction_exact_match_accuracy": 1.0,
        },
    ]

    capacity_summary, _, _, _ = correction_field_capacity.summarize_capacity_cells(records)
    threshold_summary = correction_field_capacity.summarize_threshold_shift(records)

    capacity_entry = capacity_summary["32"]["0.5"]["0.9"]
    assert capacity_entry["raw_effective_capacity"] == 0
    assert capacity_entry["correction_field_memory_effective_capacity"] == 4
    assert capacity_entry["prediction_only_effective_capacity"] == 8
    assert capacity_entry["correction_field_reconstruction_effective_capacity"] == 8

    threshold_entry = threshold_summary["32"]
    assert threshold_entry["raw_threshold_decay"] == 0.9
    assert threshold_entry["correction_field_memory_threshold_decay"] == 0.9
    assert threshold_entry["prediction_only_threshold_decay"] == 0.5
    assert threshold_entry["correction_field_reconstruction_threshold_decay"] == 0.5


def test_main_emits_declared_artifacts_and_summary_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(correction_field_capacity, "HEAD_DIMS", [32])
    monkeypatch.setattr(correction_field_capacity, "DECAYS", [0.5, 0.9])
    monkeypatch.setattr(correction_field_capacity, "PREDICTION_QUALITIES", [0.0, 0.9])
    monkeypatch.setattr(correction_field_capacity, "PATTERN_COUNTS", [4, 8])
    monkeypatch.setattr(correction_field_capacity, "TRIALS", 1)
    monkeypatch.setenv("SIM_OUTPUT_DIR", str(tmp_path))

    correction_field_capacity.main()

    metrics_path = tmp_path / "correction_field_capacity_metrics.json"
    heatmap_path = tmp_path / "correction_field_capacity_heatmaps.png"
    summary_path = tmp_path / "correction_field_capacity_summary.png"
    payload = validate_metrics_file(metrics_path)
    assert heatmap_path.exists()
    assert summary_path.exists()
    assert "capacity_raw_vs_corrfield_by_quality" in payload["summary"]
    assert "cosine_gain_by_quality_and_patterns" in payload["summary"]
    assert "threshold_shift_by_decay" in payload["summary"]