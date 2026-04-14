from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuroloc.simulations.memory import multi_resolution_head_split
from neuroloc.simulations.shared import validate_metrics_file


def test_build_stream_is_deterministic_under_seed() -> None:
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    stream_a = multi_resolution_head_split.build_stream(rng_a, 32, 16)
    stream_b = multi_resolution_head_split.build_stream(rng_b, 32, 16)
    assert np.array_equal(stream_a["keys"], stream_b["keys"])
    assert np.array_equal(stream_a["values"], stream_b["values"])
    assert np.array_equal(stream_a["classes"], stream_b["classes"])
    assert np.allclose(stream_a["surprise"], stream_b["surprise"])


def test_run_single_trial_is_deterministic_under_seed() -> None:
    rng_a = np.random.default_rng(77)
    rng_b = np.random.default_rng(77)
    result_a = multi_resolution_head_split.run_single_trial(rng_a, 32, 16)
    result_b = multi_resolution_head_split.run_single_trial(rng_b, 32, 16)
    for config in ("uniform", "split"):
        for class_name in multi_resolution_head_split.CLASS_NAMES:
            entry_a = result_a[config]["per_class"][class_name]
            entry_b = result_b[config]["per_class"][class_name]
            assert entry_a["mean_cosine"] == entry_b["mean_cosine"]
            assert entry_a["top1_accuracy"] == entry_b["top1_accuracy"]
            assert entry_a["count"] == entry_b["count"]


def test_uniform_configuration_matches_reference_matrix_memory_math() -> None:
    rng = np.random.default_rng(11)
    head_dim = 16
    stream_length = 20
    keys = multi_resolution_head_split.normalize_rows(rng.standard_normal((stream_length, head_dim)))
    values = multi_resolution_head_split.normalize_rows(rng.standard_normal((stream_length, head_dim)))
    surprise = np.zeros(stream_length, dtype=np.float64)
    decay = 0.9
    head_config = {
        "group": "uniform",
        "decay": decay,
        "surprise_tau": 0.0,
        "head_dim": head_dim,
    }
    head_state, write_flags = multi_resolution_head_split.run_head_state(head_config, keys, values, surprise)
    expected = np.zeros((head_dim, head_dim), dtype=np.float64)
    for idx in range(stream_length):
        expected = decay * expected + np.outer(keys[idx], values[idx])
    assert np.allclose(head_state, expected)
    assert bool(np.all(write_flags))


def test_split_with_only_fast_heads_matches_uniform_with_fast_decay_and_zero_surprise_tau() -> None:
    rng_a = np.random.default_rng(91)
    rng_b = np.random.default_rng(91)
    head_dim = 16
    stream_length = 16
    stream_a = multi_resolution_head_split.build_stream(rng_a, stream_length, head_dim)
    stream_b = multi_resolution_head_split.build_stream(rng_b, stream_length, head_dim)
    assert np.array_equal(stream_a["keys"], stream_b["keys"])
    split_only_fast = multi_resolution_head_split.build_head_configuration(
        "split",
        head_dim,
        num_fast=multi_resolution_head_split.TOTAL_HEADS,
        num_medium=0,
        num_slow=0,
        fast_decay=0.5,
    )
    uniform_with_matched_decay = multi_resolution_head_split.build_head_configuration(
        "uniform",
        head_dim,
        uniform_decay=0.5,
    )
    assert len(split_only_fast) == len(uniform_with_matched_decay)
    for split_cfg, uniform_cfg in zip(split_only_fast, uniform_with_matched_decay):
        split_state, _ = multi_resolution_head_split.run_head_state(
            split_cfg,
            stream_a["keys"],
            stream_a["values"],
            stream_a["surprise"],
        )
        uniform_state, _ = multi_resolution_head_split.run_head_state(
            uniform_cfg,
            stream_b["keys"],
            stream_b["values"],
            stream_b["surprise"],
        )
        assert np.allclose(split_state, uniform_state)


def test_main_emits_declared_artifacts_and_summary_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(multi_resolution_head_split, "STREAM_LENGTHS", [16, 32])
    monkeypatch.setattr(multi_resolution_head_split, "HEAD_DIMS", [16])
    monkeypatch.setattr(multi_resolution_head_split, "TRIALS", 2)
    monkeypatch.setattr(multi_resolution_head_split, "QUERIES_PER_CLASS", 3)
    monkeypatch.setenv("SIM_OUTPUT_DIR", str(tmp_path))

    multi_resolution_head_split.main()

    metrics_path = tmp_path / "multi_resolution_head_split_metrics.json"
    figure_path = tmp_path / "multi_resolution_head_split.png"
    payload = validate_metrics_file(metrics_path)
    assert figure_path.exists()
    assert "recall_by_class_and_configuration" in payload["summary"]
    assert "capacity_gain_split_over_uniform_by_class" in payload["summary"]
    assert "crossover_pattern_count_by_class" in payload["summary"]
    for class_name in multi_resolution_head_split.CLASS_NAMES:
        assert class_name in payload["summary"]["capacity_gain_split_over_uniform_by_class"]
        assert class_name in payload["summary"]["crossover_pattern_count_by_class"]
