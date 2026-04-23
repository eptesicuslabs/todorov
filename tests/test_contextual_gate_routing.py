from pathlib import Path

import pytest

from neuroloc.simulations.memory.contextual_gate_routing import (
    build_rng,
    build_summary,
    evaluate_trials,
    sample_derangement,
)
from neuroloc.simulations.memory import contextual_gate_routing
from neuroloc.simulations.shared import validate_metrics_file


def test_main_emits_required_summary_keys(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SIM_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(contextual_gate_routing, "TRIALS", 24)
    monkeypatch.setattr(contextual_gate_routing, "N_CUES", 6)
    monkeypatch.setattr(contextual_gate_routing, "N_CONTEXTS", 3)
    monkeypatch.setattr(contextual_gate_routing, "ACTION_COUNT", 4)
    monkeypatch.setattr(contextual_gate_routing, "CONTEXTUAL_FRACTION", 0.5)

    summary = contextual_gate_routing.main()

    assert "state_probe_accuracy" in summary
    assert "action_success" in summary
    assert "joint_success" in summary
    assert "context_gate_dependency" in summary
    assert "false_bind_rate_under_wrong_context" in summary
    assert "action_success_by_trial_type" in summary
    assert "joint_success_by_trial_type" in summary
    assert summary["context_gate_dependency"] > 0.0

    metrics_path = tmp_path / "contextual_gate_routing_metrics.json"
    figure_path = tmp_path / "contextual_gate_routing.png"
    payload = validate_metrics_file(metrics_path)
    assert figure_path.exists()
    assert payload["run"]["status"] == "completed"


def test_evaluate_trials_is_deterministic_under_seed(monkeypatch) -> None:
    monkeypatch.setattr(contextual_gate_routing, "N_CUES", 6)
    monkeypatch.setattr(contextual_gate_routing, "N_CONTEXTS", 3)
    monkeypatch.setattr(contextual_gate_routing, "ACTION_COUNT", 4)
    monkeypatch.setattr(contextual_gate_routing, "CONTEXTUAL_FRACTION", 0.5)

    rows_a = evaluate_trials(build_rng(9), trial_count=24)
    rows_b = evaluate_trials(build_rng(9), trial_count=24)

    assert rows_a == rows_b


def test_sample_derangement_has_no_fixed_points() -> None:
    permutation = sample_derangement(build_rng(3), 5)

    assert all(idx != value for idx, value in enumerate(permutation.tolist()))


def test_single_context_configuration_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(contextual_gate_routing, "N_CONTEXTS", 1)

    with pytest.raises(ValueError, match="CGR_CONTEXTS must be >= 2"):
        evaluate_trials(build_rng(1), trial_count=8)


def test_gate_closed_stays_correct_on_feedforward_trials(monkeypatch) -> None:
    monkeypatch.setattr(contextual_gate_routing, "N_CUES", 6)
    monkeypatch.setattr(contextual_gate_routing, "N_CONTEXTS", 3)
    monkeypatch.setattr(contextual_gate_routing, "ACTION_COUNT", 4)
    monkeypatch.setattr(contextual_gate_routing, "CONTEXTUAL_FRACTION", 0.0)

    rows = evaluate_trials(build_rng(15), trial_count=48)
    summary = build_summary(rows)

    assert summary["joint_success_by_trial_type"]["correct_context"]["feedforward"] == 1.0
    assert summary["joint_success_by_trial_type"]["bottom_up_only"]["feedforward"] == 1.0
    assert summary["joint_success_by_trial_type"]["forced_gate_closed"]["feedforward"] == 1.0


def test_correct_context_beats_gate_closed_on_contextual_trials(monkeypatch) -> None:
    monkeypatch.setattr(contextual_gate_routing, "N_CUES", 8)
    monkeypatch.setattr(contextual_gate_routing, "N_CONTEXTS", 3)
    monkeypatch.setattr(contextual_gate_routing, "ACTION_COUNT", 4)
    monkeypatch.setattr(contextual_gate_routing, "CONTEXTUAL_FRACTION", 1.0)

    rows = evaluate_trials(build_rng(17), trial_count=64)
    summary = build_summary(rows)

    correct_context = summary["joint_success_by_trial_type"]["correct_context"]["contextual"]
    forced_closed = summary["joint_success_by_trial_type"]["forced_gate_closed"]["contextual"]
    shuffled_context = summary["joint_success_by_trial_type"]["shuffled_context"]["contextual"]

    assert correct_context >= forced_closed
    assert correct_context >= shuffled_context
    assert summary["false_bind_rate_under_wrong_context"] > 0.0
    assert summary["false_bind_rate_under_wrong_context"] < 1.0
