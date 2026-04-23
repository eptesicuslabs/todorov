from pathlib import Path

import pytest

from neuroloc.simulations.memory.episodic_separation_completion import (
    build_rng,
    build_summary,
    evaluate_cell,
)
from neuroloc.simulations.memory import episodic_separation_completion
from neuroloc.simulations.shared import validate_metrics_file


def test_main_emits_required_summary_keys(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SIM_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(episodic_separation_completion, "TRIALS", 8)
    monkeypatch.setattr(episodic_separation_completion, "FEATURE_DIM", 16)
    monkeypatch.setattr(episodic_separation_completion, "ACTION_COUNT", 4)
    monkeypatch.setattr(episodic_separation_completion, "OVERLAPS", [0.5, 0.9])
    monkeypatch.setattr(episodic_separation_completion, "CUE_DROPS", [0.0, 0.5])
    monkeypatch.setattr(episodic_separation_completion, "DISTRACTOR_COUNTS", [0, 4])

    summary = episodic_separation_completion.main()

    assert "state_probe_accuracy" in summary
    assert "action_success" in summary
    assert "joint_success" in summary
    assert "separation_margin" in summary
    assert "completion_accuracy_by_cue_drop" in summary
    assert "novelty_detection_accuracy" in summary
    assert "delayed_recall_after_distractors" in summary
    assert summary["state_probe_accuracy"] >= summary["policy_metrics"]["no_memory"]["state_probe_accuracy"]["mean"]

    metrics_path = tmp_path / "episodic_separation_completion_metrics.json"
    figure_path = tmp_path / "episodic_separation_completion.png"
    payload = validate_metrics_file(metrics_path)
    assert figure_path.exists()
    assert payload["run"]["status"] == "completed"
    assert payload["summary"]["policy_metrics"]["oracle_read"]["action_success"]["mean"] == 1.0


def test_evaluate_cell_is_deterministic_under_seed(monkeypatch) -> None:
    monkeypatch.setattr(episodic_separation_completion, "FEATURE_DIM", 12)
    monkeypatch.setattr(episodic_separation_completion, "ACTION_COUNT", 4)

    rows_a = evaluate_cell(build_rng(7), overlap=0.8, cue_drop=0.5, distractor_count=4, trial_count=6)
    rows_b = evaluate_cell(build_rng(7), overlap=0.8, cue_drop=0.5, distractor_count=4, trial_count=6)

    assert rows_a == rows_b


def test_feature_dim_one_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(episodic_separation_completion, "FEATURE_DIM", 1)

    with pytest.raises(ValueError, match="ESC_FEATURE_DIM must be >= 2"):
        evaluate_cell(build_rng(2), overlap=0.0, cue_drop=0.0, distractor_count=0, trial_count=1)


def test_primary_policy_beats_controls_on_completion(monkeypatch) -> None:
    monkeypatch.setattr(episodic_separation_completion, "FEATURE_DIM", 16)
    monkeypatch.setattr(episodic_separation_completion, "ACTION_COUNT", 4)

    rows = evaluate_cell(build_rng(11), overlap=0.9, cue_drop=0.5, distractor_count=4, trial_count=32)
    summary = build_summary(rows)

    primary_state = summary["policy_metrics"]["episodic_index"]["state_probe_accuracy"]["mean"]
    shuffled_state = summary["policy_metrics"]["shuffled_mapping"]["state_probe_accuracy"]["mean"]
    no_memory_state = summary["policy_metrics"]["no_memory"]["state_probe_accuracy"]["mean"]
    oracle_state = summary["policy_metrics"]["oracle_read"]["state_probe_accuracy"]["mean"]
    primary_joint = summary["policy_metrics"]["episodic_index"]["joint_success"]["mean"]
    no_memory_joint = summary["policy_metrics"]["no_memory"]["joint_success"]["mean"]

    assert oracle_state == 1.0
    assert primary_state >= shuffled_state
    assert primary_state >= no_memory_state
    assert primary_joint >= no_memory_joint
    assert summary["novelty_detection_accuracy"] > 0.0
