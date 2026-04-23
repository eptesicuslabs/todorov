from pathlib import Path

import pytest

from neuroloc.simulations.memory.episodic_replay_reuse import (
    build_base_strengths,
    build_rng,
    build_summary,
    evaluate_cell,
)
from neuroloc.simulations.memory import episodic_replay_reuse
from neuroloc.simulations.shared import validate_metrics_file


def test_main_emits_required_summary_keys(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SIM_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(episodic_replay_reuse, "TRIALS", 8)
    monkeypatch.setattr(episodic_replay_reuse, "FEATURE_DIM", 16)
    monkeypatch.setattr(episodic_replay_reuse, "ACTION_COUNT", 4)
    monkeypatch.setattr(episodic_replay_reuse, "CUE_DROPS", [0.25, 0.5])
    monkeypatch.setattr(episodic_replay_reuse, "DISTRACTOR_COUNTS", [0, 4, 8])
    monkeypatch.setattr(episodic_replay_reuse, "REPLAY_STEPS", 3)
    monkeypatch.setattr(episodic_replay_reuse, "DECAY", 0.9)
    monkeypatch.setattr(episodic_replay_reuse, "DISTRACTOR_OVERLAP", 0.75)

    summary = episodic_replay_reuse.main()

    assert "state_probe_accuracy" in summary
    assert "action_success" in summary
    assert "joint_success" in summary
    assert "replay_reuse_gain" in summary
    assert "targeted_vs_random_replay_gap" in summary
    assert "distractor_decay_curve" in summary
    assert summary["replay_reuse_gain"] >= 0.0

    metrics_path = tmp_path / "episodic_replay_reuse_metrics.json"
    figure_path = tmp_path / "episodic_replay_reuse.png"
    payload = validate_metrics_file(metrics_path)
    assert figure_path.exists()
    assert payload["run"]["status"] == "completed"


def test_evaluate_cell_is_deterministic_under_seed(monkeypatch) -> None:
    monkeypatch.setattr(episodic_replay_reuse, "FEATURE_DIM", 12)
    monkeypatch.setattr(episodic_replay_reuse, "ACTION_COUNT", 4)
    monkeypatch.setattr(episodic_replay_reuse, "REPLAY_STEPS", 3)
    monkeypatch.setattr(episodic_replay_reuse, "DECAY", 0.9)
    monkeypatch.setattr(episodic_replay_reuse, "DISTRACTOR_OVERLAP", 0.75)

    rows_a = evaluate_cell(build_rng(5), cue_drop=0.5, distractor_count=8, trial_count=8)
    rows_b = evaluate_cell(build_rng(5), cue_drop=0.5, distractor_count=8, trial_count=8)

    assert rows_a == rows_b


def test_base_strengths_only_grow_after_observation(monkeypatch) -> None:
    monkeypatch.setattr(episodic_replay_reuse, "DECAY", 0.9)

    strengths = build_base_strengths(3)

    assert strengths.tolist() == pytest.approx([0.729, 0.81, 0.9, 1.0])


def test_targeted_replay_beats_no_replay_and_recency(monkeypatch) -> None:
    monkeypatch.setattr(episodic_replay_reuse, "FEATURE_DIM", 16)
    monkeypatch.setattr(episodic_replay_reuse, "ACTION_COUNT", 4)
    monkeypatch.setattr(episodic_replay_reuse, "REPLAY_STEPS", 4)
    monkeypatch.setattr(episodic_replay_reuse, "DECAY", 0.85)
    monkeypatch.setattr(episodic_replay_reuse, "DISTRACTOR_OVERLAP", 0.8)

    rows = evaluate_cell(build_rng(13), cue_drop=0.5, distractor_count=8, trial_count=48)
    summary = build_summary(rows)

    targeted_joint = summary["policy_metrics"]["targeted_replay"]["joint_success"]["mean"]
    no_replay_joint = summary["policy_metrics"]["no_replay"]["joint_success"]["mean"]
    recency_joint = summary["policy_metrics"]["recency_only"]["joint_success"]["mean"]

    assert targeted_joint >= no_replay_joint
    assert targeted_joint >= recency_joint
    assert summary["replay_reuse_gain"] >= 0.0
