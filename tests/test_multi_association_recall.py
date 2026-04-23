from pathlib import Path

from neuroloc.simulations.memory import multi_association_recall
from neuroloc.simulations.shared import validate_metrics_file


def test_main_emits_required_summary_keys(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SIM_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(multi_association_recall, "HEAD_DIMS", [32, 64])
    monkeypatch.setattr(multi_association_recall, "CUE_COUNTS", [8, 12])
    monkeypatch.setattr(multi_association_recall, "VALUES_PER_CUE", [2, 3])
    monkeypatch.setattr(multi_association_recall, "QUERY_NOISE", [0.0, 0.1])
    monkeypatch.setattr(multi_association_recall, "TRIALS", 3)

    summary = multi_association_recall.main()

    assert "slot_bundle_exact_rate" in summary
    assert "slot_value_hit_rate" in summary
    assert "matrix_bundle_exact_rate" in summary
    assert "matrix_value_hit_rate" in summary
    assert "slot_minus_matrix_exact_gain" in summary
    assert summary["by_cell"]

    metrics_path = tmp_path / "multi_association_recall_metrics.json"
    figure_path = tmp_path / "multi_association_recall.png"
    payload = validate_metrics_file(metrics_path)
    assert figure_path.exists()
    assert payload["run"]["status"] == "completed"
