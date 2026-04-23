from pathlib import Path

import numpy as np

from neuroloc.simulations.memory import delayed_cue_world
from neuroloc.simulations.shared import validate_metrics_file


def test_single_goal_noise_path_stays_defined() -> None:
    rng = np.random.default_rng(0)
    observations = delayed_cue_world.sample_cue_observations(
        rng=rng,
        goal=0,
        cue_count=4,
        noise_level=0.4,
        n_goals=1,
    )

    assert observations == [0, 0, 0, 0]


def test_delay_steps_change_recency_baseline() -> None:
    no_delay = delayed_cue_world.evaluate_cell(
        rng=np.random.default_rng(1),
        cue_count=1,
        noise_level=0.0,
        episodes=256,
        delay_steps=0,
    )
    long_delay = delayed_cue_world.evaluate_cell(
        rng=np.random.default_rng(1),
        cue_count=1,
        noise_level=0.0,
        episodes=256,
        delay_steps=6,
    )

    assert np.mean(no_delay["last_accuracy"]) == 1.0
    assert np.mean(long_delay["last_accuracy"]) < 0.5
    assert np.mean(long_delay["bayes_accuracy"]) == 1.0


def test_main_emits_required_summary_keys(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SIM_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(delayed_cue_world, "N_GOALS", 4)
    monkeypatch.setattr(delayed_cue_world, "CUE_COUNTS", [1, 3, 5])
    monkeypatch.setattr(delayed_cue_world, "NOISE_LEVELS", [0.0, 0.2, 0.4])
    monkeypatch.setattr(delayed_cue_world, "DELAY_STEPS", 6)
    monkeypatch.setattr(delayed_cue_world, "EPISODES", 64)

    summary = delayed_cue_world.main()

    assert "bayes_accuracy_by_noise" in summary
    assert "last_cue_accuracy_by_noise" in summary
    assert "memory_advantage_by_noise" in summary
    assert "oracle_accuracy" in summary
    assert "no_memory_accuracy" in summary
    assert summary["oracle_accuracy"] == 1.0
    assert summary["by_cell"]

    metrics_path = tmp_path / "delayed_cue_world_metrics.json"
    figure_path = tmp_path / "delayed_cue_world.png"
    payload = validate_metrics_file(metrics_path)
    assert figure_path.exists()
    assert payload["run"]["status"] == "completed"
