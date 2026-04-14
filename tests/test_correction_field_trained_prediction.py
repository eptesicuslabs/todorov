from pathlib import Path

import numpy as np
import pytest

from neuroloc.simulations.memory import correction_field_trained_prediction
from neuroloc.simulations.shared import validate_metrics_file


def test_prediction_head_training_decreases_mse() -> None:
    rng = np.random.default_rng(11)
    head_dim = 32
    mixing = correction_field_trained_prediction.random_orthogonal(rng, head_dim)
    _, _, quality, final_loss, loss_trajectory = correction_field_trained_prediction.train_prediction_head(
        rng=rng,
        mixing_matrix=mixing,
        temporal_correlation=0.9,
        head_dim=head_dim,
        rank=16,
        n_sequences=512,
        sequence_length=24,
        noise_scale=0.1,
        n_iterations=200,
        learning_rate=1e-3,
        batch_size=128,
    )
    assert len(loss_trajectory) == 200
    assert final_loss < loss_trajectory[0]
    assert quality > 0.0


def test_prediction_quality_near_zero_at_zero_temporal_correlation() -> None:
    rng = np.random.default_rng(23)
    head_dim = 32
    qualities_zero: list[float] = []
    qualities_high: list[float] = []
    for _ in range(3):
        mixing = correction_field_trained_prediction.random_orthogonal(rng, head_dim)
        _, _, quality_zero, _, _ = correction_field_trained_prediction.train_prediction_head(
            rng=rng,
            mixing_matrix=mixing,
            temporal_correlation=0.0,
            head_dim=head_dim,
            rank=16,
            n_sequences=2000,
            sequence_length=24,
            noise_scale=0.1,
            n_iterations=200,
            learning_rate=1e-3,
            batch_size=128,
        )
        _, _, quality_high, _, _ = correction_field_trained_prediction.train_prediction_head(
            rng=rng,
            mixing_matrix=mixing,
            temporal_correlation=0.9,
            head_dim=head_dim,
            rank=16,
            n_sequences=2000,
            sequence_length=24,
            noise_scale=0.1,
            n_iterations=200,
            learning_rate=1e-3,
            batch_size=128,
        )
        qualities_zero.append(float(quality_zero))
        qualities_high.append(float(quality_high))
    mean_zero = float(np.mean(qualities_zero))
    mean_high = float(np.mean(qualities_high))
    assert mean_zero < 0.05, f"held-out quality at rho=0 above training bias: {qualities_zero}"
    assert mean_high > mean_zero + 0.05, (
        f"temporally correlated data not more predictable than zero-structure data: "
        f"mean_zero={mean_zero}, mean_high={mean_high}"
    )


def test_memory_build_matches_existing_sim_math_without_structure() -> None:
    rng = np.random.default_rng(31)
    head_dim = 32
    pattern_count = 6
    decay = 0.9
    keys = correction_field_trained_prediction.normalize_rows(
        rng.standard_normal((pattern_count, head_dim))
    )
    values = rng.standard_normal((pattern_count, head_dim))
    state = correction_field_trained_prediction.build_state(keys, values, decay)
    expected = np.zeros((head_dim, head_dim), dtype=np.float64)
    for key_vec, value_vec in zip(keys, values):
        expected = decay * expected + np.outer(key_vec, value_vec)
    assert np.allclose(state, expected, atol=1e-9)
    raw_only = correction_field_trained_prediction.evaluate_recall(keys, values, state)
    zero_base = np.zeros_like(values)
    with_zero_base = correction_field_trained_prediction.evaluate_recall(
        keys, values, state, reconstruction_base=zero_base
    )
    assert np.isclose(raw_only["mean_cosine_recall"], with_zero_base["mean_cosine_recall"], atol=1e-9)
    assert np.isclose(raw_only["exact_match_accuracy"], with_zero_base["exact_match_accuracy"], atol=1e-9)


def test_generate_correlated_sequences_has_expected_autocorrelation() -> None:
    rng = np.random.default_rng(97)
    head_dim = 32
    mixing = correction_field_trained_prediction.random_orthogonal(rng, head_dim)
    samples = correction_field_trained_prediction.generate_correlated_sequences(
        rng=rng,
        mixing_matrix=mixing,
        temporal_correlation=0.0,
        n_sequences=256,
        sequence_length=32,
        noise_scale=0.0,
    )
    previous = samples[:, :-1, :].reshape(-1, head_dim)
    current = samples[:, 1:, :].reshape(-1, head_dim)
    previous_centered = previous - previous.mean(axis=0, keepdims=True)
    current_centered = current - current.mean(axis=0, keepdims=True)
    per_dim_correlation = np.sum(previous_centered * current_centered, axis=0) / (
        np.sqrt(np.sum(previous_centered * previous_centered, axis=0))
        * np.sqrt(np.sum(current_centered * current_centered, axis=0))
    )
    assert float(np.mean(np.abs(per_dim_correlation))) < 0.1


def test_main_emits_declared_artifacts_and_required_summary_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(correction_field_trained_prediction, "HEAD_DIMS", [32])
    monkeypatch.setattr(correction_field_trained_prediction, "DECAYS", [0.5, 0.9])
    monkeypatch.setattr(correction_field_trained_prediction, "TEMPORAL_CORRELATIONS", [0.5, 0.9])
    monkeypatch.setattr(correction_field_trained_prediction, "PATTERN_COUNTS", [4, 8])
    monkeypatch.setattr(correction_field_trained_prediction, "TRIALS", 2)
    monkeypatch.setattr(correction_field_trained_prediction, "TRAIN_SEQUENCES", 64)
    monkeypatch.setattr(correction_field_trained_prediction, "TRAIN_SEQUENCE_LENGTH", 16)
    monkeypatch.setattr(correction_field_trained_prediction, "TRAIN_ITERS", 25)
    monkeypatch.setattr(correction_field_trained_prediction, "BATCH_SIZE", 32)
    monkeypatch.setenv("SIM_OUTPUT_DIR", str(tmp_path))

    correction_field_trained_prediction.main()

    metrics_path = tmp_path / "correction_field_trained_prediction_metrics.json"
    heatmap_path = tmp_path / "correction_field_trained_prediction_heatmaps.png"
    summary_path = tmp_path / "correction_field_trained_prediction_summary.png"
    payload = validate_metrics_file(metrics_path)
    assert heatmap_path.exists()
    assert summary_path.exists()
    assert "memory_substrate_capacity_delta_by_temporal_correlation" in payload["summary"]
    assert "reconstruction_capacity_gain" in payload["summary"]
    assert "observed_prediction_quality_by_correlation" in payload["summary"]
