from pathlib import Path

import numpy as np

from neuroloc.simulations.memory import slot_key_interference_sweep
from neuroloc.simulations.shared import validate_metrics_file


def test_sample_correlated_keys_tracks_requested_pairwise_cosine() -> None:
    rng = np.random.default_rng(0)
    for requested in [0.0, 0.5, 0.8, 0.95]:
        keys = slot_key_interference_sweep.sample_correlated_keys(
            rng=rng,
            count=64,
            dim=64,
            correlation=requested,
        )
        realized = slot_key_interference_sweep.mean_off_diagonal_cosine(keys)
        assert abs(realized - requested) < 0.12


def test_main_emits_required_summary_keys(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SIM_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(slot_key_interference_sweep, "HEAD_DIMS", [32, 64])
    monkeypatch.setattr(slot_key_interference_sweep, "SLOT_COUNTS", [16, 32])
    monkeypatch.setattr(slot_key_interference_sweep, "CORRELATIONS", [0.0, 0.8, 0.95])
    monkeypatch.setattr(slot_key_interference_sweep, "TEMPERATURES", [0.1])
    monkeypatch.setattr(slot_key_interference_sweep, "QUERY_NOISE", [0.0])
    monkeypatch.setattr(slot_key_interference_sweep, "TRIALS", 3)

    summary = slot_key_interference_sweep.main()

    assert "slot_interference_slope_by_dim" in summary
    assert "matrix_interference_slope_by_dim" in summary
    assert "slot_minus_matrix_gain_at_high_correlation" in summary
    assert summary["by_cell"]
    assert "realized_key_correlation" in next(iter(summary["by_cell"].values()))

    metrics_path = tmp_path / "slot_key_interference_sweep_metrics.json"
    figure_path = tmp_path / "slot_key_interference_sweep.png"
    payload = validate_metrics_file(metrics_path)
    assert figure_path.exists()
    assert payload["run"]["status"] == "completed"
    assert payload["summary"]["by_cell"]
