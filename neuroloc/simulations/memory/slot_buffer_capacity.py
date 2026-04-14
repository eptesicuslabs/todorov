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
    env_float,
    env_int,
    env_list,
    mean_confidence_interval,
    output_dir_for,
    paired_difference_stats,
    require_non_negative_list,
    require_positive,
    require_positive_list,
    require_unit_interval_list,
    utc_now_iso,
    write_json,
)

SCRIPT_PATH = Path(__file__).resolve()
SEED = env_int("SLOT_CAP_SEED", 42)
HEAD_DIMS = env_list("SLOT_CAP_HEAD_DIMS", int, [32, 64, 128])
SLOT_COUNTS = env_list("SLOT_CAP_SLOTS", int, [16, 32, 64, 128])
TEMPERATURES = env_list("SLOT_CAP_TEMPERATURES", float, [0.1, 1.0, 10.0])
QUERY_NOISE = env_list("SLOT_CAP_QUERY_NOISE", float, [0.0, 0.1, 0.2])
TRIALS = env_int("SLOT_CAP_TRIALS", 32)

require_positive_list("SLOT_CAP_HEAD_DIMS", HEAD_DIMS)
require_positive_list("SLOT_CAP_SLOTS", SLOT_COUNTS)
require_positive_list("SLOT_CAP_TEMPERATURES", TEMPERATURES)
require_non_negative_list("SLOT_CAP_QUERY_NOISE", QUERY_NOISE)
require_positive("SLOT_CAP_TRIALS", TRIALS)

EPS = 1e-9
PASS_COSINE_THRESHOLD = 0.95
PASS_EXACT_THRESHOLD = 0.90
PASS_N = 64
PASS_D = 64
PASS_QUERY_NOISE = 0.0


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, EPS, None)


def sample_unit_vectors(rng: np.random.Generator, count: int, dim: int) -> np.ndarray:
    raw = rng.standard_normal((count, dim))
    return normalize_rows(raw)


def noisy_query(rng: np.random.Generator, key: np.ndarray, noise_std: float) -> np.ndarray:
    if noise_std <= 0.0:
        return key
    perturbation = rng.standard_normal(key.shape)
    perturbation = perturbation - (perturbation @ key) * key / max(float(key @ key), EPS)
    perturbation = perturbation / max(float(np.linalg.norm(perturbation)), EPS)
    combined = key + noise_std * perturbation
    return combined / max(float(np.linalg.norm(combined)), EPS)


def slot_read_softmax(
    query: np.ndarray, slot_keys: np.ndarray, slot_values: np.ndarray, temperature: float
) -> np.ndarray:
    scores = slot_keys @ query
    scaled = scores / max(temperature, EPS)
    shifted = scaled - np.max(scaled)
    weights = np.exp(shifted)
    total = np.sum(weights)
    if total <= 0.0:
        return np.zeros_like(slot_values[0])
    weights = weights / total
    return weights @ slot_values


def matrix_memory_read(query: np.ndarray, state: np.ndarray) -> np.ndarray:
    return query @ state


def matrix_memory_write(state: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    return state + np.outer(key, value)


def evaluate_cell(
    rng: np.random.Generator,
    slot_count: int,
    head_dim: int,
    temperature: float,
    noise_std: float,
    trial_count: int,
) -> dict[str, Any]:
    slot_cosines: list[float] = []
    slot_exact: list[float] = []
    matrix_cosines: list[float] = []
    matrix_exact: list[float] = []
    for _ in range(trial_count):
        trial_rng = child_rng(rng)
        keys = sample_unit_vectors(trial_rng, slot_count, head_dim)
        values = sample_unit_vectors(trial_rng, slot_count, head_dim)
        slot_keys = keys.copy()
        slot_values = values.copy()
        matrix_state = np.zeros((head_dim, head_dim), dtype=np.float64)
        for slot in range(slot_count):
            matrix_state = matrix_memory_write(matrix_state, keys[slot], values[slot])
        trial_slot_cos: list[float] = []
        trial_slot_correct: list[int] = []
        trial_matrix_cos: list[float] = []
        trial_matrix_correct: list[int] = []
        for target in range(slot_count):
            query = noisy_query(trial_rng, keys[target], noise_std)
            slot_output = slot_read_softmax(query, slot_keys, slot_values, temperature)
            matrix_output = matrix_memory_read(query, matrix_state)
            slot_norm = max(float(np.linalg.norm(slot_output)), EPS)
            matrix_norm = max(float(np.linalg.norm(matrix_output)), EPS)
            target_norm = max(float(np.linalg.norm(values[target])), EPS)
            trial_slot_cos.append(float((slot_output @ values[target]) / (slot_norm * target_norm)))
            trial_matrix_cos.append(float((matrix_output @ values[target]) / (matrix_norm * target_norm)))
            slot_scores = values @ slot_output
            matrix_scores = values @ matrix_output
            trial_slot_correct.append(int(np.argmax(slot_scores) == target))
            trial_matrix_correct.append(int(np.argmax(matrix_scores) == target))
        slot_cosines.append(float(np.mean(trial_slot_cos)))
        slot_exact.append(float(np.mean(trial_slot_correct)))
        matrix_cosines.append(float(np.mean(trial_matrix_cos)))
        matrix_exact.append(float(np.mean(trial_matrix_correct)))
    return {
        "slot_cosine": slot_cosines,
        "slot_exact": slot_exact,
        "matrix_cosine": matrix_cosines,
        "matrix_exact": matrix_exact,
    }


def format_value(value: float) -> str:
    return format(float(value), "g").replace(".", "p").replace("-", "m")


def summarize_cell(cell: dict[str, list[float]]) -> dict[str, Any]:
    return {
        "slot_cosine": mean_confidence_interval(cell["slot_cosine"], bounds=(-1.0, 1.0)),
        "slot_exact": mean_confidence_interval(cell["slot_exact"], bounds=(0.0, 1.0)),
        "matrix_cosine": mean_confidence_interval(cell["matrix_cosine"], bounds=(-1.0, 1.0)),
        "matrix_exact": mean_confidence_interval(cell["matrix_exact"], bounds=(0.0, 1.0)),
        "paired_cosine_gain": paired_difference_stats(
            cell["matrix_cosine"], cell["slot_cosine"], SEED
        ),
        "paired_exact_gain": paired_difference_stats(
            cell["matrix_exact"], cell["slot_exact"], SEED + 1
        ),
    }


def build_summary(
    cells: dict[tuple[int, int, float, float], dict[str, list[float]]],
) -> dict[str, Any]:
    by_cell: dict[str, Any] = {}
    for (head_dim, slots, temperature, noise), cell in cells.items():
        key = f"d{head_dim}_N{slots}_t{format_value(temperature)}_q{format_value(noise)}"
        by_cell[key] = summarize_cell(cell)
    best_cell: dict[str, Any] | None = None
    best_temperature: float | None = None
    best_cosine: float = -1.0
    for temperature in TEMPERATURES:
        key = (PASS_D, PASS_N, float(temperature), float(PASS_QUERY_NOISE))
        if key not in cells:
            continue
        cell_values = cells[key]["slot_cosine"]
        if not cell_values:
            continue
        mean_value = float(np.mean(cell_values))
        if mean_value > best_cosine:
            best_cosine = mean_value
            best_temperature = float(temperature)
            best_cell = summarize_cell(cells[key])
    if best_cell is None:
        best_cell = {
            "slot_cosine": {"n": 0, "mean": None, "std": None, "ci95": {"low": None, "high": None}},
            "slot_exact": {"n": 0, "mean": None, "std": None, "ci95": {"low": None, "high": None}},
            "matrix_cosine": {"n": 0, "mean": None, "std": None, "ci95": {"low": None, "high": None}},
            "matrix_exact": {"n": 0, "mean": None, "std": None, "ci95": {"low": None, "high": None}},
        }
    slot_cosine_mean = best_cell["slot_cosine"]["mean"]
    slot_exact_mean = best_cell["slot_exact"]["mean"]
    slot_cosine_ok = (
        slot_cosine_mean is not None and float(slot_cosine_mean) >= PASS_COSINE_THRESHOLD
    )
    slot_exact_ok = (
        slot_exact_mean is not None and float(slot_exact_mean) >= PASS_EXACT_THRESHOLD
    )
    pass_criterion = bool(slot_cosine_ok and slot_exact_ok)
    capacity_threshold_by_dim: dict[str, Any] = {}
    for head_dim in HEAD_DIMS:
        per_dim: dict[str, Any] = {}
        for temperature in TEMPERATURES:
            largest_pass = 0
            for slots in SLOT_COUNTS:
                key = (int(head_dim), int(slots), float(temperature), float(PASS_QUERY_NOISE))
                if key not in cells:
                    continue
                cos_values = cells[key]["slot_cosine"]
                if not cos_values:
                    continue
                if float(np.mean(cos_values)) >= PASS_COSINE_THRESHOLD:
                    largest_pass = max(largest_pass, int(slots))
            per_dim[f"temperature{format_value(float(temperature))}"] = largest_pass
        capacity_threshold_by_dim[f"d{head_dim}"] = per_dim
    return {
        "pass_criterion_slot_cosine_mean": slot_cosine_mean,
        "pass_criterion_slot_exact_mean": slot_exact_mean,
        "pass_criterion_threshold_cosine": PASS_COSINE_THRESHOLD,
        "pass_criterion_threshold_exact": PASS_EXACT_THRESHOLD,
        "pass_criterion_met": pass_criterion,
        "pass_criterion_best_temperature": best_temperature,
        "pass_reference_cell": (
            f"d{PASS_D}_N{PASS_N}_t{format_value(float(best_temperature))}_q{format_value(float(PASS_QUERY_NOISE))}"
            if best_temperature is not None
            else None
        ),
        "capacity_threshold_by_dim": capacity_threshold_by_dim,
        "by_cell": by_cell,
    }


def build_figure(cells: dict[tuple[int, int, float, float], dict[str, list[float]]], output_path: Path) -> None:
    apply_plot_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(HEAD_DIMS), figsize=(5 * len(HEAD_DIMS), 4), sharey=True)
    if len(HEAD_DIMS) == 1:
        axes = [axes]
    for ax, head_dim in zip(axes, HEAD_DIMS):
        for temperature in TEMPERATURES:
            slot_means = []
            matrix_means = []
            x_values = []
            for slots in SLOT_COUNTS:
                key = (int(head_dim), int(slots), float(temperature), float(PASS_QUERY_NOISE))
                if key not in cells:
                    continue
                slot_means.append(float(np.mean(cells[key]["slot_cosine"])))
                matrix_means.append(float(np.mean(cells[key]["matrix_cosine"])))
                x_values.append(int(slots))
            if not x_values:
                continue
            ax.plot(x_values, slot_means, marker="o", label=f"slot t={temperature}")
            ax.plot(x_values, matrix_means, marker="x", linestyle="--", label=f"matrix baseline t={temperature}")
        ax.axhline(PASS_COSINE_THRESHOLD, color="gray", linestyle=":", alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("slots (N)")
        ax.set_title(f"head_dim={head_dim}")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("mean cosine retrieval")
    fig.suptitle("slot buffer vs matrix memory standalone capacity (query noise=0)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> dict[str, Any]:
    started_at = utc_now_iso()
    start_wall = time.time()
    rng = build_rng(SEED)
    cells: dict[tuple[int, int, float, float], dict[str, list[float]]] = {}
    trial_records: list[dict[str, Any]] = []
    for head_dim in HEAD_DIMS:
        for slots in SLOT_COUNTS:
            for temperature in TEMPERATURES:
                for noise in QUERY_NOISE:
                    cell_rng = child_rng(rng)
                    cell = evaluate_cell(
                        cell_rng,
                        int(slots),
                        int(head_dim),
                        float(temperature),
                        float(noise),
                        int(TRIALS),
                    )
                    cells[(int(head_dim), int(slots), float(temperature), float(noise))] = cell
                    for trial_idx in range(len(cell["slot_cosine"])):
                        trial_records.append(
                            {
                                "head_dim": int(head_dim),
                                "slot_count": int(slots),
                                "temperature": float(temperature),
                                "query_noise": float(noise),
                                "trial_id": int(trial_idx),
                                "slot_cosine": cell["slot_cosine"][trial_idx],
                                "slot_exact": cell["slot_exact"][trial_idx],
                                "matrix_cosine": cell["matrix_cosine"][trial_idx],
                                "matrix_exact": cell["matrix_exact"][trial_idx],
                            }
                        )
    summary = build_summary(cells)
    finished_at = utc_now_iso()
    duration = time.time() - start_wall
    output_dir = output_dir_for(SCRIPT_PATH)
    metrics_path = output_dir / "slot_buffer_capacity_metrics.json"
    figure_path = output_dir / "slot_buffer_capacity.png"
    build_figure(cells, figure_path)
    artifact_entries = [
        {
            "name": "slot_buffer_capacity.png",
            "path": str(figure_path),
            "type": "figure",
            "description": "slot buffer vs matrix memory capacity by head dim and temperature",
        },
        {
            "name": "slot_buffer_capacity_metrics.json",
            "path": str(metrics_path),
            "type": "metrics",
            "description": "capacity and exact retrieval summary per cell",
        },
    ]
    best_temperature_used = summary.get("pass_criterion_best_temperature")
    parameters = {
        "head_dims": list(HEAD_DIMS),
        "slot_counts": list(SLOT_COUNTS),
        "temperatures": list(TEMPERATURES),
        "query_noise": list(QUERY_NOISE),
        "trials": int(TRIALS),
        "pass_cosine_threshold": PASS_COSINE_THRESHOLD,
        "pass_exact_threshold": PASS_EXACT_THRESHOLD,
        "pass_d": PASS_D,
        "pass_n": PASS_N,
        "pass_temperature_used": (
            float(best_temperature_used) if best_temperature_used is not None else None
        ),
        "pass_query_noise": float(PASS_QUERY_NOISE),
    }
    statistics = {"elapsed_seconds": duration, "cell_count": len(cells)}
    warnings: list[str] = [
        "matrix memory baseline uses no decay; the actual god_machine.py delta-rule layer includes exponential retention, so the reported matrix baseline cosine is a lower bound on a more realistic comparison",
    ]
    record = build_run_record(
        simulation_name="slot_buffer_capacity",
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_sec=duration,
        parameters=parameters,
        seed_numpy=SEED,
        n_trials=int(TRIALS) * len(cells),
        summary=summary,
        statistics=statistics,
        trials=trial_records,
        artifacts=artifact_entries,
        warnings=warnings,
    )
    write_json(metrics_path, record)
    return summary


if __name__ == "__main__":
    summary = main()
    print("pass_criterion_met:", summary["pass_criterion_met"])
    print("slot cosine at pass cell:", summary["pass_criterion_slot_cosine_mean"])
    print("slot exact at pass cell:", summary["pass_criterion_slot_exact_mean"])
