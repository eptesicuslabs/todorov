import json
from pathlib import Path

import numpy as np
import pytest

from neuroloc.simulations.memory import slot_surprise_writes


def test_surprise_score_anchor_self_is_near_zero() -> None:
    rng = np.random.default_rng(0)
    anchor = slot_surprise_writes.sample_unit_vectors(rng, 1, 64)[0]
    assert slot_surprise_writes.surprise_score(anchor, anchor) == pytest.approx(0.0, abs=1e-10)


def test_surprise_score_orthogonal_is_half() -> None:
    rng = np.random.default_rng(1)
    anchor = slot_surprise_writes.sample_unit_vectors(rng, 1, 64)[0]
    orthogonal = np.zeros(64)
    orthogonal[0] = 1.0
    orthogonal = orthogonal - (orthogonal @ anchor) * anchor
    orthogonal = orthogonal / np.linalg.norm(orthogonal)
    score = slot_surprise_writes.surprise_score(orthogonal, anchor)
    assert score == pytest.approx(0.5, abs=1e-10)


def test_surprise_score_antiparallel_is_one() -> None:
    rng = np.random.default_rng(11)
    anchor = slot_surprise_writes.sample_unit_vectors(rng, 1, 64)[0]
    score = slot_surprise_writes.surprise_score(-anchor, anchor)
    assert score == pytest.approx(1.0, abs=1e-10)


def test_synthesize_stream_respects_predictable_fraction() -> None:
    rng = np.random.default_rng(2)
    anchor = slot_surprise_writes.sample_unit_vectors(rng, 1, 64)[0]
    length = 400
    keys, values, flags = slot_surprise_writes.synthesize_stream(rng, anchor, length, 0.9)
    assert keys.shape == (length, 64)
    assert values.shape == (length, 64)
    assert flags.shape == (length,)
    actual_surprise_ratio = float(np.mean(flags))
    assert 0.05 <= actual_surprise_ratio <= 0.20
    predictable_overlaps = [
        float(keys[t] @ anchor) for t in range(length) if not flags[t]
    ]
    surprising_overlaps = [
        float(keys[t] @ anchor) for t in range(length) if flags[t]
    ]
    assert float(np.mean(predictable_overlaps)) > 0.9
    assert abs(float(np.mean(surprising_overlaps))) < 0.3


def test_slot_write_and_retrieve_handles_target_not_written() -> None:
    rng = np.random.default_rng(3)
    anchor = slot_surprise_writes.sample_unit_vectors(rng, 1, 32)[0]
    length = 40
    predictable = anchor[None, :] + 0.01 * rng.standard_normal((length, 32))
    predictable = slot_surprise_writes.normalize_rows(predictable)
    flags = np.zeros(length, dtype=bool)
    result = slot_surprise_writes.slot_write_and_retrieve(
        predictable,
        predictable,
        flags,
        anchor,
        slot_count=8,
        tau=0.5,
        target_position=10,
        intervening_distance=20,
        temperature=0.1,
    )
    assert result["retrieved"] is False


def test_main_emits_required_summary_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIM_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(slot_surprise_writes, "HEAD_DIMS", [64])
    monkeypatch.setattr(slot_surprise_writes, "SLOT_COUNTS", [32])
    monkeypatch.setattr(slot_surprise_writes, "INTERVENING_DISTANCES", [64])
    monkeypatch.setattr(slot_surprise_writes, "PREDICTABLE_FRACTIONS", [0.9])
    monkeypatch.setattr(slot_surprise_writes, "SURPRISE_TAUS", [0.1])
    monkeypatch.setattr(slot_surprise_writes, "TEMPERATURE", 0.1)
    monkeypatch.setattr(slot_surprise_writes, "TRIALS", 3)
    summary = slot_surprise_writes.main()
    assert "pass_criterion_threshold" in summary
    assert "pass_criterion_best_cosine" in summary
    assert "pass_criterion_met" in summary
    assert "cells" in summary
    metrics_path = tmp_path / "slot_surprise_writes_metrics.json"
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["run"]["status"] == "completed"
