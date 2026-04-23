from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from shared import (
    apply_plot_style,
    build_rng,
    build_run_record,
    child_rng,
    env_int,
    env_list,
    mean_confidence_interval,
    output_dir_for,
    require_positive,
    require_positive_list,
    require_unit_interval_list,
    utc_now_iso,
    write_json,
)

SCRIPT_PATH = Path(__file__).resolve()
SEED = env_int("DCW_SEED", 42)
N_GOALS = env_int("DCW_GOALS", 4)
CUE_COUNTS = env_list("DCW_CUE_COUNTS", int, [1, 3, 5])
NOISE_LEVELS = env_list("DCW_NOISE_LEVELS", float, [0.0, 0.2, 0.4])
DELAY_STEPS = env_int("DCW_DELAY_STEPS", 8)
EPISODES = env_int("DCW_EPISODES", 256)

require_positive("DCW_GOALS", N_GOALS)
require_positive_list("DCW_CUE_COUNTS", CUE_COUNTS)
require_unit_interval_list("DCW_NOISE_LEVELS", NOISE_LEVELS, allow_zero=True)
require_positive("DCW_DELAY_STEPS", DELAY_STEPS)
require_positive("DCW_EPISODES", EPISODES)

EPS = 1e-9


def sample_cue_observations(
    rng: np.random.Generator,
    goal: int,
    cue_count: int,
    noise_level: float,
    n_goals: int,
    delay_steps: int = 0,
) -> list[int]:
    if n_goals <= 1:
        return [int(goal)] * int(cue_count + delay_steps)
    observations: list[int] = []
    for _ in range(cue_count):
        if noise_level <= 0.0 or rng.random() >= noise_level:
            observations.append(int(goal))
            continue
        distractors = [idx for idx in range(n_goals) if idx != goal]
        observations.append(int(rng.choice(np.array(distractors, dtype=np.int64))))
    for _ in range(delay_steps):
        observations.append(int(rng.integers(0, n_goals)))
    return observations


def argmax_with_last_tiebreak(values: np.ndarray) -> int:
    best_value = float(np.max(values))
    candidates = np.nonzero(np.isclose(values, best_value))[0]
    return int(candidates[-1])


def majority_vote_prediction(observations: list[int], n_goals: int) -> int:
    counts = np.zeros(n_goals, dtype=np.int64)
    for obs in observations:
        counts[int(obs)] += 1
    return argmax_with_last_tiebreak(counts.astype(np.float64))


def bayes_prediction(observations: list[int], noise_level: float, n_goals: int) -> tuple[int, float]:
    posterior = np.full(n_goals, 1.0 / float(n_goals), dtype=np.float64)
    if n_goals == 1:
        return 0, 0.0
    off_target = noise_level / float(max(n_goals - 1, 1))
    for obs in observations:
        likelihood = np.full(n_goals, off_target, dtype=np.float64)
        likelihood[int(obs)] = 1.0 - noise_level
        posterior = posterior * likelihood
        posterior = posterior / max(float(np.sum(posterior)), EPS)
    entropy = float(-(posterior * np.log2(np.clip(posterior, EPS, None))).sum())
    return int(np.argmax(posterior)), entropy


def evaluate_cell(
    rng: np.random.Generator,
    cue_count: int,
    noise_level: float,
    episodes: int,
    delay_steps: int | None = None,
) -> dict[str, list[float]]:
    bayes_accuracy: list[float] = []
    last_accuracy: list[float] = []
    majority_accuracy: list[float] = []
    posterior_entropy: list[float] = []
    goals: list[int] = []
    no_memory_accuracy = 1.0 / float(N_GOALS)
    effective_delay = DELAY_STEPS if delay_steps is None else int(delay_steps)
    for _ in range(episodes):
        trial_rng = child_rng(rng)
        goal = int(trial_rng.integers(0, N_GOALS))
        observations = sample_cue_observations(
            trial_rng,
            goal,
            cue_count,
            noise_level,
            N_GOALS,
            delay_steps=effective_delay,
        )
        bayes_pred, entropy = bayes_prediction(observations[:cue_count], noise_level, N_GOALS)
        last_pred = int(observations[-1])
        majority_pred = majority_vote_prediction(observations, N_GOALS)
        bayes_accuracy.append(float(bayes_pred == goal))
        last_accuracy.append(float(last_pred == goal))
        majority_accuracy.append(float(majority_pred == goal))
        posterior_entropy.append(float(entropy))
        goals.append(goal)
    return {
        "bayes_accuracy": bayes_accuracy,
        "last_accuracy": last_accuracy,
        "majority_accuracy": majority_accuracy,
        "posterior_entropy": posterior_entropy,
        "goals": [float(v) for v in goals],
        "no_memory_accuracy": [no_memory_accuracy] * episodes,
    }


def summarize_cell(cell: dict[str, list[float]]) -> dict[str, Any]:
    return {
        "bayes_accuracy": mean_confidence_interval(cell["bayes_accuracy"], bounds=(0.0, 1.0)),
        "last_accuracy": mean_confidence_interval(cell["last_accuracy"], bounds=(0.0, 1.0)),
        "majority_accuracy": mean_confidence_interval(cell["majority_accuracy"], bounds=(0.0, 1.0)),
        "posterior_entropy": mean_confidence_interval(cell["posterior_entropy"]),
        "no_memory_accuracy": float(np.mean(cell["no_memory_accuracy"])),
    }


def build_summary(cells: dict[tuple[int, float], dict[str, list[float]]]) -> dict[str, Any]:
    by_cell: dict[str, Any] = {}
    for (cue_count, noise_level), cell in cells.items():
        key = f"c{cue_count}_n{format(float(noise_level), 'g').replace('.', 'p')}"
        by_cell[key] = summarize_cell(cell)

    reference_cue_count = max(int(v) for v in CUE_COUNTS)
    bayes_accuracy_by_noise: dict[str, float | None] = {}
    last_cue_accuracy_by_noise: dict[str, float | None] = {}
    majority_accuracy_by_noise: dict[str, float | None] = {}
    memory_advantage_by_noise: dict[str, float | None] = {}
    posterior_entropy_by_noise: dict[str, float | None] = {}
    no_memory_accuracy = 1.0 / float(N_GOALS)
    for noise_level in NOISE_LEVELS:
        key = (reference_cue_count, float(noise_level))
        if key not in cells:
            suffix = f"noise{format(float(noise_level), 'g').replace('.', 'p')}"
            bayes_accuracy_by_noise[suffix] = None
            last_cue_accuracy_by_noise[suffix] = None
            majority_accuracy_by_noise[suffix] = None
            memory_advantage_by_noise[suffix] = None
            posterior_entropy_by_noise[suffix] = None
            continue
        cell_summary = summarize_cell(cells[key])
        suffix = f"noise{format(float(noise_level), 'g').replace('.', 'p')}"
        bayes_mean = cell_summary["bayes_accuracy"]["mean"]
        last_mean = cell_summary["last_accuracy"]["mean"]
        bayes_accuracy_by_noise[suffix] = bayes_mean
        last_cue_accuracy_by_noise[suffix] = last_mean
        majority_accuracy_by_noise[suffix] = cell_summary["majority_accuracy"]["mean"]
        posterior_entropy_by_noise[suffix] = cell_summary["posterior_entropy"]["mean"]
        if bayes_mean is None or last_mean is None:
            memory_advantage_by_noise[suffix] = None
        else:
            memory_advantage_by_noise[suffix] = float(bayes_mean) - float(last_mean)
    return {
        "bayes_accuracy_by_noise": bayes_accuracy_by_noise,
        "last_cue_accuracy_by_noise": last_cue_accuracy_by_noise,
        "majority_accuracy_by_noise": majority_accuracy_by_noise,
        "memory_advantage_by_noise": memory_advantage_by_noise,
        "posterior_entropy_by_noise": posterior_entropy_by_noise,
        "oracle_accuracy": 1.0,
        "no_memory_accuracy": no_memory_accuracy,
        "delay_steps": DELAY_STEPS,
        "by_cell": by_cell,
    }


def build_figure(cells: dict[tuple[int, float], dict[str, list[float]]], output_path: Path) -> None:
    apply_plot_style()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference_cue_count = max(int(v) for v in CUE_COUNTS)
    noise_x = [float(v) for v in NOISE_LEVELS]
    bayes = []
    last = []
    majority = []
    entropy = []
    for noise_level in NOISE_LEVELS:
        key = (reference_cue_count, float(noise_level))
        bayes.append(float(np.mean(cells[key]["bayes_accuracy"])))
        last.append(float(np.mean(cells[key]["last_accuracy"])))
        majority.append(float(np.mean(cells[key]["majority_accuracy"])))
        entropy.append(float(np.mean(cells[key]["posterior_entropy"])))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(noise_x, bayes, marker="o", label="bayes belief")
    axes[0].plot(noise_x, majority, marker="s", label="majority vote")
    axes[0].plot(noise_x, last, marker="x", linestyle="--", label="last cue")
    axes[0].axhline(1.0 / float(N_GOALS), color="gray", linestyle=":", label="no memory")
    axes[0].set_title(f"delayed-cue accuracy (c={reference_cue_count})")
    axes[0].set_xlabel("cue noise")
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)

    for cue_count in CUE_COUNTS:
        cue_entropy = []
        for noise_level in NOISE_LEVELS:
            cue_entropy.append(float(np.mean(cells[(int(cue_count), float(noise_level))]["posterior_entropy"])))
        axes[1].plot(noise_x, cue_entropy, marker="o", label=f"cues={cue_count}")
    axes[1].set_title("posterior entropy")
    axes[1].set_xlabel("cue noise")
    axes[1].set_ylabel("bits")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> dict[str, object]:
    started = utc_now_iso()
    t0 = time.time()
    rng = build_rng(SEED)
    output_dir = output_dir_for(SCRIPT_PATH)
    cells: dict[tuple[int, float], dict[str, list[float]]] = {}
    for cue_count in CUE_COUNTS:
        for noise_level in NOISE_LEVELS:
            key = (int(cue_count), float(noise_level))
            cells[key] = evaluate_cell(
                rng=rng,
                cue_count=int(cue_count),
                noise_level=float(noise_level),
                episodes=EPISODES,
                delay_steps=DELAY_STEPS,
            )

    summary = build_summary(cells)
    figure_path = output_dir / "delayed_cue_world.png"
    build_figure(cells, figure_path)

    finished = utc_now_iso()
    duration = time.time() - t0
    metrics_path = output_dir / "delayed_cue_world_metrics.json"
    trials: list[dict[str, Any]] = []
    for (cue_count, noise_level), cell in cells.items():
        for trial_idx, (
            bayes_accuracy,
            last_accuracy,
            majority_accuracy,
            posterior_entropy,
            goal,
        ) in enumerate(
            zip(
                cell["bayes_accuracy"],
                cell["last_accuracy"],
                cell["majority_accuracy"],
                cell["posterior_entropy"],
                cell["goals"],
                strict=True,
            )
        ):
            trials.append(
                {
                    "cue_count": int(cue_count),
                    "noise_level": float(noise_level),
                    "delay_steps": DELAY_STEPS,
                    "trial_id": int(trial_idx),
                    "goal": int(goal),
                    "bayes_accuracy": float(bayes_accuracy),
                    "last_accuracy": float(last_accuracy),
                    "majority_accuracy": float(majority_accuracy),
                    "posterior_entropy": float(posterior_entropy),
                    "no_memory_accuracy": 1.0 / float(N_GOALS),
                }
            )
    record = build_run_record(
        simulation_name="delayed_cue_world",
        script_path=SCRIPT_PATH,
        started_at_utc=started,
        finished_at_utc=finished,
        duration_sec=duration,
        parameters={
            "n_goals": N_GOALS,
            "cue_counts": [int(v) for v in CUE_COUNTS],
            "noise_levels": [float(v) for v in NOISE_LEVELS],
            "delay_steps": DELAY_STEPS,
        },
        seed_numpy=SEED,
        n_trials=len(trials),
        summary=summary,
        statistics={},
        trials=trials,
        artifacts=[
            {"name": "delayed_cue_world.png", "path": figure_path, "type": "figure"},
            {"name": "delayed_cue_world_metrics.json", "path": metrics_path, "type": "metrics"},
        ],
        warnings=[],
    )
    write_json(metrics_path, record)
    return summary


if __name__ == "__main__":
    main()
