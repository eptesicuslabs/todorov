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
SEED = env_int("SLOT_INT_SEED", 42)
HEAD_DIMS = env_list("SLOT_INT_HEAD_DIMS", int, [32, 64])
SLOT_COUNTS = env_list("SLOT_INT_SLOTS", int, [16, 64])
CORRELATIONS = env_list("SLOT_INT_CORRELATIONS", float, [0.0, 0.5, 0.8, 0.95])
TEMPERATURES = env_list("SLOT_INT_TEMPERATURES", float, [0.1, 1.0])
QUERY_NOISE = env_list("SLOT_INT_QUERY_NOISE", float, [0.0, 0.1])
TRIALS = env_int("SLOT_INT_TRIALS", 32)

require_positive_list("SLOT_INT_HEAD_DIMS", HEAD_DIMS)
require_positive_list("SLOT_INT_SLOTS", SLOT_COUNTS)
require_unit_interval_list("SLOT_INT_CORRELATIONS", CORRELATIONS, allow_zero=True)
require_positive_list("SLOT_INT_TEMPERATURES", TEMPERATURES)
require_non_negative_list("SLOT_INT_QUERY_NOISE", QUERY_NOISE)
require_positive("SLOT_INT_TRIALS", TRIALS)

EPS = 1e-9
REFERENCE_NOISE = 0.0


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, EPS, None)


def sample_unit_vectors(rng: np.random.Generator, count: int, dim: int) -> np.ndarray:
    return normalize_rows(rng.standard_normal((count, dim)))


def sample_correlated_keys(
    rng: np.random.Generator,
    count: int,
    dim: int,
    correlation: float,
) -> np.ndarray:
    anchor = sample_unit_vectors(rng, 1, dim)[0]
    if correlation >= 1.0 - EPS:
        return np.repeat(anchor[None, :], count, axis=0)
    noise = rng.standard_normal((count, dim))
    projection = (noise @ anchor)[:, None]
    orthogonal_noise = noise - projection * anchor[None, :]
    orthogonal_noise = normalize_rows(orthogonal_noise)
    return np.sqrt(max(correlation, 0.0)) * anchor[None, :] + np.sqrt(max(1.0 - correlation, 0.0)) * orthogonal_noise


def mean_off_diagonal_cosine(keys: np.ndarray) -> float:
    if int(keys.shape[0]) <= 1:
        return 1.0
    gram = keys @ keys.T
    count = int(keys.shape[0])
    mask = ~np.eye(count, dtype=bool)
    return float(np.mean(gram[mask]))


def noisy_query(rng: np.random.Generator, key: np.ndarray, noise_std: float) -> np.ndarray:
    if noise_std <= 0.0:
        return key
    perturbation = rng.standard_normal(key.shape)
    projection = float(np.dot(perturbation, key))
    perturbation = perturbation - projection * key / max(float(np.dot(key, key)), EPS)
    perturbation = perturbation / max(float(np.linalg.norm(perturbation)), EPS)
    combined = key + noise_std * perturbation
    return combined / max(float(np.linalg.norm(combined)), EPS)


def slot_read_softmax(
    query: np.ndarray,
    slot_keys: np.ndarray,
    slot_values: np.ndarray,
    temperature: float,
) -> np.ndarray:
    scores = slot_keys @ query
    scaled = scores / max(float(temperature), EPS)
    shifted = scaled - float(np.max(scaled))
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.zeros_like(slot_values[0])
    weights = weights / total
    return weights @ slot_values


def matrix_read(query: np.ndarray, state: np.ndarray) -> np.ndarray:
    return query @ state


def matrix_write(state: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    return state + np.outer(key, value)


def evaluate_cell(
    rng: np.random.Generator,
    head_dim: int,
    slot_count: int,
    correlation: float,
    temperature: float,
    noise_std: float,
    trial_count: int,
) -> dict[str, list[float]]:
    slot_cosines: list[float] = []
    matrix_cosines: list[float] = []
    slot_exact: list[float] = []
    matrix_exact: list[float] = []
    realized_key_correlation: list[float] = []
    for _ in range(trial_count):
        trial_rng = child_rng(rng)
        keys = sample_correlated_keys(trial_rng, slot_count, head_dim, correlation)
        values = sample_unit_vectors(trial_rng, slot_count, head_dim)
        realized_key_correlation.append(mean_off_diagonal_cosine(keys))
        state = np.zeros((head_dim, head_dim), dtype=np.float64)
        for key, value in zip(keys, values, strict=True):
            state = matrix_write(state, key, value)
        trial_slot_cos: list[float] = []
        trial_matrix_cos: list[float] = []
        trial_slot_exact: list[int] = []
        trial_matrix_exact: list[int] = []
        for target in range(slot_count):
            query = noisy_query(trial_rng, keys[target], noise_std)
            slot_output = slot_read_softmax(query, keys, values, temperature)
            matrix_output = matrix_read(query, state)
            slot_norm = max(float(np.linalg.norm(slot_output)), EPS)
            matrix_norm = max(float(np.linalg.norm(matrix_output)), EPS)
            value_norm = max(float(np.linalg.norm(values[target])), EPS)
            trial_slot_cos.append(float(np.dot(slot_output, values[target]) / (slot_norm * value_norm)))
            trial_matrix_cos.append(float(np.dot(matrix_output, values[target]) / (matrix_norm * value_norm)))
            trial_slot_exact.append(int(np.argmax(values @ slot_output) == target))
            trial_matrix_exact.append(int(np.argmax(values @ matrix_output) == target))
        slot_cosines.append(float(np.mean(trial_slot_cos)))
        matrix_cosines.append(float(np.mean(trial_matrix_cos)))
        slot_exact.append(float(np.mean(trial_slot_exact)))
        matrix_exact.append(float(np.mean(trial_matrix_exact)))
    return {
        "slot_cosine": slot_cosines,
        "matrix_cosine": matrix_cosines,
        "slot_exact": slot_exact,
        "matrix_exact": matrix_exact,
        "realized_key_correlation": realized_key_correlation,
    }


def format_value(value: float) -> str:
    return format(float(value), "g").replace(".", "p").replace("-", "m")


def summarize_cell(cell: dict[str, list[float]]) -> dict[str, Any]:
    return {
        "slot_cosine": mean_confidence_interval(cell["slot_cosine"], bounds=(-1.0, 1.0)),
        "matrix_cosine": mean_confidence_interval(cell["matrix_cosine"], bounds=(-1.0, 1.0)),
        "slot_exact": mean_confidence_interval(cell["slot_exact"], bounds=(0.0, 1.0)),
        "matrix_exact": mean_confidence_interval(cell["matrix_exact"], bounds=(0.0, 1.0)),
        "realized_key_correlation": mean_confidence_interval(cell["realized_key_correlation"], bounds=(-1.0, 1.0)),
        "paired_cosine_gain": paired_difference_stats(cell["matrix_cosine"], cell["slot_cosine"], SEED),
        "paired_exact_gain": paired_difference_stats(cell["matrix_exact"], cell["slot_exact"], SEED + 1),
    }


def build_summary(
    cells: dict[tuple[int, int, float, float, float], dict[str, list[float]]],
) -> dict[str, Any]:
    by_cell: dict[str, Any] = {}
    for (head_dim, slot_count, correlation, temperature, noise_std), cell in cells.items():
        key = (
            f"d{head_dim}_N{slot_count}_rho{format_value(correlation)}"
            f"_t{format_value(temperature)}_q{format_value(noise_std)}"
        )
        by_cell[key] = summarize_cell(cell)

    reference_slots = max(int(value) for value in SLOT_COUNTS)
    reference_temperature = min(float(value) for value in TEMPERATURES)
    reference_noise = REFERENCE_NOISE if REFERENCE_NOISE in {float(value) for value in QUERY_NOISE} else min(float(value) for value in QUERY_NOISE)
    low_corr = min(float(value) for value in CORRELATIONS)
    high_corr = max(float(value) for value in CORRELATIONS)

    slot_interference_slope_by_dim: dict[str, float | None] = {}
    matrix_interference_slope_by_dim: dict[str, float | None] = {}
    slot_minus_matrix_gain_at_high_correlation: dict[str, float | None] = {}
    for head_dim in HEAD_DIMS:
        low_key = (int(head_dim), reference_slots, low_corr, reference_temperature, reference_noise)
        high_key = (int(head_dim), reference_slots, high_corr, reference_temperature, reference_noise)
        if low_key not in cells or high_key not in cells:
            slot_interference_slope_by_dim[f"d{head_dim}"] = None
            matrix_interference_slope_by_dim[f"d{head_dim}"] = None
            slot_minus_matrix_gain_at_high_correlation[f"d{head_dim}"] = None
            continue
        slot_low = float(np.mean(cells[low_key]["slot_cosine"]))
        slot_high = float(np.mean(cells[high_key]["slot_cosine"]))
        matrix_low = float(np.mean(cells[low_key]["matrix_cosine"]))
        matrix_high = float(np.mean(cells[high_key]["matrix_cosine"]))
        delta_corr = max(high_corr - low_corr, EPS)
        slot_interference_slope_by_dim[f"d{head_dim}"] = (slot_high - slot_low) / delta_corr
        matrix_interference_slope_by_dim[f"d{head_dim}"] = (matrix_high - matrix_low) / delta_corr
        slot_minus_matrix_gain_at_high_correlation[f"d{head_dim}"] = slot_high - matrix_high
    return {
        "slot_interference_slope_by_dim": slot_interference_slope_by_dim,
        "matrix_interference_slope_by_dim": matrix_interference_slope_by_dim,
        "slot_minus_matrix_gain_at_high_correlation": slot_minus_matrix_gain_at_high_correlation,
        "reference_slots": reference_slots,
        "reference_temperature": reference_temperature,
        "reference_query_noise": reference_noise,
        "by_cell": by_cell,
    }


def build_figure(
    cells: dict[tuple[int, int, float, float, float], dict[str, list[float]]],
    output_path: Path,
) -> None:
    apply_plot_style()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference_slots = max(int(value) for value in SLOT_COUNTS)
    reference_temperature = min(float(value) for value in TEMPERATURES)
    reference_noise = REFERENCE_NOISE if REFERENCE_NOISE in {float(value) for value in QUERY_NOISE} else min(float(value) for value in QUERY_NOISE)

    fig, axes = plt.subplots(1, len(HEAD_DIMS), figsize=(5 * len(HEAD_DIMS), 4), sharey=True)
    if len(HEAD_DIMS) == 1:
        axes = [axes]
    for ax, head_dim in zip(axes, HEAD_DIMS):
        slot_means: list[float] = []
        matrix_means: list[float] = []
        x_values: list[float] = []
        for correlation in CORRELATIONS:
            key = (int(head_dim), reference_slots, float(correlation), reference_temperature, reference_noise)
            if key not in cells:
                continue
            slot_means.append(float(np.mean(cells[key]["slot_cosine"])))
            matrix_means.append(float(np.mean(cells[key]["matrix_cosine"])))
            x_values.append(float(correlation))
        ax.plot(x_values, slot_means, marker="o", label="slot")
        ax.plot(x_values, matrix_means, marker="x", linestyle="--", label="matrix")
        ax.set_title(f"interference sweep d={head_dim}")
        ax.set_xlabel("key correlation")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("mean recall cosine")
    axes[0].legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> dict[str, object]:
    started = utc_now_iso()
    t0 = time.time()
    rng = build_rng(SEED)
    output_dir = output_dir_for(SCRIPT_PATH)
    cells: dict[tuple[int, int, float, float, float], dict[str, list[float]]] = {}
    for head_dim in HEAD_DIMS:
        for slot_count in SLOT_COUNTS:
            for correlation in CORRELATIONS:
                for temperature in TEMPERATURES:
                    for noise_std in QUERY_NOISE:
                        key = (
                            int(head_dim),
                            int(slot_count),
                            float(correlation),
                            float(temperature),
                            float(noise_std),
                        )
                        cells[key] = evaluate_cell(
                            rng=rng,
                            head_dim=int(head_dim),
                            slot_count=int(slot_count),
                            correlation=float(correlation),
                            temperature=float(temperature),
                            noise_std=float(noise_std),
                            trial_count=TRIALS,
                        )

    summary = build_summary(cells)
    figure_path = output_dir / "slot_key_interference_sweep.png"
    build_figure(cells, figure_path)

    finished = utc_now_iso()
    duration = time.time() - t0
    metrics_path = output_dir / "slot_key_interference_sweep_metrics.json"
    trials: list[dict[str, Any]] = []
    for (head_dim, slot_count, correlation, temperature, noise_std), cell in cells.items():
        for trial_idx, (slot_cos, matrix_cos, slot_exact, matrix_exact) in enumerate(
            zip(
                cell["slot_cosine"],
                cell["matrix_cosine"],
                cell["slot_exact"],
                cell["matrix_exact"],
                strict=True,
            )
        ):
            trials.append(
                {
                    "head_dim": int(head_dim),
                    "slot_count": int(slot_count),
                    "correlation": float(correlation),
                    "temperature": float(temperature),
                    "query_noise": float(noise_std),
                    "trial_id": int(trial_idx),
                    "slot_cosine": float(slot_cos),
                    "matrix_cosine": float(matrix_cos),
                    "slot_exact": float(slot_exact),
                    "matrix_exact": float(matrix_exact),
                    "realized_key_correlation": float(cell["realized_key_correlation"][trial_idx]),
                }
            )
    record = build_run_record(
        simulation_name="slot_key_interference_sweep",
        script_path=SCRIPT_PATH,
        started_at_utc=started,
        finished_at_utc=finished,
        duration_sec=duration,
        parameters={
            "head_dims": [int(v) for v in HEAD_DIMS],
            "slot_counts": [int(v) for v in SLOT_COUNTS],
            "correlations": [float(v) for v in CORRELATIONS],
            "temperatures": [float(v) for v in TEMPERATURES],
            "query_noise": [float(v) for v in QUERY_NOISE],
        },
        seed_numpy=SEED,
        n_trials=len(trials),
        summary=summary,
        statistics={},
        trials=trials,
        artifacts=[
            {"name": "slot_key_interference_sweep.png", "path": figure_path, "type": "figure"},
            {"name": "slot_key_interference_sweep_metrics.json", "path": metrics_path, "type": "metrics"},
        ],
        warnings=[],
    )
    write_json(metrics_path, record)
    return summary


if __name__ == "__main__":
    main()
