from pathlib import Path

from neuroloc.simulations.memory import contextual_recall_world
from neuroloc.simulations.shared import validate_metrics_file


def test_main_emits_required_summary_keys(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SIM_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(contextual_recall_world, "N_EPISODES", 12)
    monkeypatch.setattr(contextual_recall_world, "SEQ_LEN", 10)
    monkeypatch.setattr(contextual_recall_world, "N_IDENTITIES", 8)
    monkeypatch.setattr(contextual_recall_world, "N_ACTIVE", 3)
    monkeypatch.setattr(contextual_recall_world, "TRACK_LENGTH", 17)
    monkeypatch.setattr(contextual_recall_world, "OCCLUSION_PROB", 0.2)
    monkeypatch.setattr(contextual_recall_world, "FEATURE_DROPOUT", 0.3)

    summary = contextual_recall_world.main()

    assert "recognition_mean_candidate_count" in summary
    assert "recollection_mean_lag" in summary
    assert "prediction_mean_step_distance" in summary
    assert "compression_mean_ratio" in summary
    assert "imagination_novelty_rate" in summary
    assert "reasoning_mean_margin" in summary
    assert summary["compression_mean_ratio"] > 1.0

    metrics_path = tmp_path / "contextual_recall_world_metrics.json"
    figure_path = tmp_path / "contextual_recall_world.png"
    payload = validate_metrics_file(metrics_path)
    assert figure_path.exists()
    assert payload["run"]["status"] == "completed"
    assert len(payload["trials"]) == 12
