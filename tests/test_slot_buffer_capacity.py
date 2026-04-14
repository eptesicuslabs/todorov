import json
from pathlib import Path

import numpy as np
import pytest

from neuroloc.simulations.memory import slot_buffer_capacity


def test_slot_read_exact_match_one_slot_perfect_query() -> None:
    rng = np.random.default_rng(0)
    key = slot_buffer_capacity.sample_unit_vectors(rng, 1, 32)[0]
    value = slot_buffer_capacity.sample_unit_vectors(rng, 1, 32)[0]
    output = slot_buffer_capacity.slot_read_softmax(key, key[None, :], value[None, :], 1.0)
    cosine = float(output @ value / (np.linalg.norm(output) * np.linalg.norm(value) + 1e-12))
    assert cosine > 0.9999


def test_slot_read_softmax_isolates_target_with_orthogonal_slots() -> None:
    rng = np.random.default_rng(1)
    slots = 8
    dim = 64
    keys = slot_buffer_capacity.sample_unit_vectors(rng, slots, dim)
    keys, _ = np.linalg.qr(keys.T)
    keys = keys[:, :slots].T
    values = slot_buffer_capacity.sample_unit_vectors(rng, slots, dim)
    for target in range(slots):
        output = slot_buffer_capacity.slot_read_softmax(keys[target], keys, values, 0.1)
        cosine = float(output @ values[target] / (np.linalg.norm(output) * np.linalg.norm(values[target]) + 1e-12))
        assert cosine > 0.95


def test_matrix_memory_has_interference_at_expected_scale() -> None:
    rng = np.random.default_rng(2)
    slots = 64
    dim = 64
    keys = slot_buffer_capacity.sample_unit_vectors(rng, slots, dim)
    values = slot_buffer_capacity.sample_unit_vectors(rng, slots, dim)
    state = np.zeros((dim, dim), dtype=np.float64)
    for idx in range(slots):
        state = slot_buffer_capacity.matrix_memory_write(state, keys[idx], values[idx])
    cosines = []
    for target in range(slots):
        output = slot_buffer_capacity.matrix_memory_read(keys[target], state)
        cosines.append(output @ values[target] / (np.linalg.norm(output) * np.linalg.norm(values[target]) + 1e-12))
    mean_cosine = float(np.mean(cosines))
    assert mean_cosine < 0.9


def test_noisy_query_preserves_unit_norm_and_direction() -> None:
    rng = np.random.default_rng(3)
    key = slot_buffer_capacity.sample_unit_vectors(rng, 1, 32)[0]
    for noise_std in [0.0, 0.1, 0.5]:
        for _ in range(5):
            perturbed = slot_buffer_capacity.noisy_query(rng, key, noise_std)
            assert abs(float(np.linalg.norm(perturbed)) - 1.0) < 1e-6
            if noise_std == 0.0:
                assert float(perturbed @ key) > 0.9999


def test_main_produces_required_summary_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIM_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(slot_buffer_capacity, "HEAD_DIMS", [32, 64])
    monkeypatch.setattr(slot_buffer_capacity, "SLOT_COUNTS", [16, 64])
    monkeypatch.setattr(slot_buffer_capacity, "TEMPERATURES", [1.0])
    monkeypatch.setattr(slot_buffer_capacity, "QUERY_NOISE", [0.0])
    monkeypatch.setattr(slot_buffer_capacity, "TRIALS", 2)
    summary = slot_buffer_capacity.main()
    assert "pass_criterion_met" in summary
    assert "pass_criterion_slot_cosine_mean" in summary
    assert "pass_criterion_slot_exact_mean" in summary
    assert "capacity_threshold_by_dim" in summary
    assert "by_cell" in summary
    metrics_path = tmp_path / "slot_buffer_capacity_metrics.json"
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["run"]["status"] == "completed"
    assert len(payload["trials"]) == 2 * 2 * 2
    assert payload["summary"]["pass_criterion_met"] is True or payload["summary"]["pass_criterion_met"] is False
