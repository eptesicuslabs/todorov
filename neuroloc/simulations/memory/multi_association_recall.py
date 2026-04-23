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
    utc_now_iso,
    write_json,
)

SCRIPT_PATH = Path(__file__).resolve()
SEED = env_int("MAR_SEED", 42)
HEAD_DIMS = env_list("MAR_HEAD_DIMS", int, [32, 64])
CUE_COUNTS = env_list("MAR_CUE_COUNTS", int, [8, 16])
VALUES_PER_CUE = env_list("MAR_VALUES_PER_CUE", int, [2, 3, 4])
QUERY_NOISE = env_list("MAR_QUERY_NOISE", float, [0.0, 0.1])
TEMPERATURE = env_float("MAR_TEMPERATURE", 0.1)
TRIALS = env_int("MAR_TRIALS", 32)

require_positive_list("MAR_HEAD_DIMS", HEAD_DIMS)
require_positive_list("MAR_CUE_COUNTS", CUE_COUNTS)
require_positive_list("MAR_VALUES_PER_CUE", VALUES_PER_CUE)
require_non_negative_list("MAR_QUERY_NOISE", QUERY_NOISE)
require_positive("MAR_TEMPERATURE", TEMPERATURE)
require_positive("MAR_TRIALS", TRIALS)

EPS = 1e-9


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, EPS, None)


def sample_unit_vectors(rng: np.random.Generator, count: int, dim: int) -> np.ndarray:
    return normalize_rows(rng.standard_normal((count, dim)))


def noisy_query(rng: np.random.Generator, key: np.ndarray, noise_std: float) -> np.ndarray:
    if noise_std <= 0.0:
        return key
    perturbation = rng.standard_normal(key.shape)
    projection = float(np.dot(perturbation, key))
    perturbation = perturbation - projection * key / max(float(np.dot(key, key)), EPS)
    perturbation = perturbation / max(float(np.linalg.norm(perturbation)), EPS)
    combined = key + noise_std * perturbation
    return combined / max(float(np.linalg.norm(combined)), EPS)


def slot_weights(query: np.ndarray, slot_keys: np.ndarray, temperature: float) -> np.ndarray:
    scores = slot_keys @ query
    scaled = scores / max(float(temperature), EPS)
    shifted = scaled - float(np.max(scaled))
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.zeros_like(scores)
    return weights / total


def top_k_indices(values: np.ndarray, k: int) -> np.ndarray:
    ranked = sorted(range(int(values.shape[0])), key=lambda idx: float(values[idx]))
    top = ranked[-int(k):]
    return np.asarray(top, dtype=np.int64)


def decode_top_k(output: np.ndarray, value_bank: np.ndarray, k: int) -> np.ndarray:
    scores = value_bank @ output
    return top_k_indices(scores, k)


def evaluate_cell(
    rng: np.random.Generator,
    head_dim: int,
    cue_count: int,
    values_per_cue: int,
    noise_std: float,
    trial_count: int,
) -> dict[str, list[float]]:
    slot_bundle_exact: list[float] = []
    matrix_bundle_exact: list[float] = []
    slot_value_hit: list[float] = []
    matrix_value_hit: list[float] = []
    slot_bundle_cosine: list[float] = []
    matrix_bundle_cosine: list[float] = []
    for _ in range(trial_count):
        trial_rng = child_rng(rng)
        cues = sample_unit_vectors(trial_rng, cue_count, head_dim)
        value_count = cue_count * values_per_cue
        value_bank = sample_unit_vectors(trial_rng, value_count, head_dim)
        slot_keys = np.repeat(cues, values_per_cue, axis=0)
        slot_values = value_bank.copy()
        cue_to_value_ids = np.arange(value_count, dtype=np.int64).reshape(cue_count, values_per_cue)
        matrix_state = np.zeros((head_dim, head_dim), dtype=np.float64)
        for key, value in zip(slot_keys, slot_values, strict=True):
            matrix_state = matrix_state + np.outer(key, value)

        trial_slot_exact: list[int] = []
        trial_matrix_exact: list[int] = []
        trial_slot_hit: list[float] = []
        trial_matrix_hit: list[float] = []
        trial_slot_cos: list[float] = []
        trial_matrix_cos: list[float] = []
        for cue_idx in range(cue_count):
            query = noisy_query(trial_rng, cues[cue_idx], noise_std)
            target_ids = cue_to_value_ids[cue_idx]
            target_set = set(int(v) for v in target_ids.tolist())

            weights = slot_weights(query, slot_keys, TEMPERATURE)
            slot_output = weights @ slot_values
            slot_top = decode_top_k(slot_output, value_bank, values_per_cue)
            slot_retrieved = set(int(v) for v in slot_top.tolist())

            matrix_output = query @ matrix_state
            matrix_top = decode_top_k(matrix_output, value_bank, values_per_cue)
            matrix_retrieved = set(int(v) for v in matrix_top.tolist())

            target_sum = slot_values[target_ids].sum(axis=0)
            slot_norm = max(float(np.linalg.norm(slot_output)), EPS)
            matrix_norm = max(float(np.linalg.norm(matrix_output)), EPS)
            target_norm = max(float(np.linalg.norm(target_sum)), EPS)
            trial_slot_cos.append(float(np.dot(slot_output, target_sum) / (slot_norm * target_norm)))
            trial_matrix_cos.append(float(np.dot(matrix_output, target_sum) / (matrix_norm * target_norm)))
            trial_slot_exact.append(int(slot_retrieved == target_set))
            trial_matrix_exact.append(int(matrix_retrieved == target_set))
            trial_slot_hit.append(float(len(slot_retrieved & target_set)) / float(values_per_cue))
            trial_matrix_hit.append(float(len(matrix_retrieved & target_set)) / float(values_per_cue))
        slot_bundle_exact.append(float(np.mean(trial_slot_exact)))
        matrix_bundle_exact.append(float(np.mean(trial_matrix_exact)))
        slot_value_hit.append(float(np.mean(trial_slot_hit)))
        matrix_value_hit.append(float(np.mean(trial_matrix_hit)))
        slot_bundle_cosine.append(float(np.mean(trial_slot_cos)))
        matrix_bundle_cosine.append(float(np.mean(trial_matrix_cos)))
    return {
        "slot_bundle_exact": slot_bundle_exact,
        "matrix_bundle_exact": matrix_bundle_exact,
        "slot_value_hit": slot_value_hit,
        "matrix_value_hit": matrix_value_hit,
        "slot_bundle_cosine": slot_bundle_cosine,
        "matrix_bundle_cosine": matrix_bundle_cosine,
    }


def summarize_cell(cell: dict[str, list[float]]) -> dict[str, Any]:
    return {
        "slot_bundle_exact": mean_confidence_interval(cell["slot_bundle_exact"], bounds=(0.0, 1.0)),
        "matrix_bundle_exact": mean_confidence_interval(cell["matrix_bundle_exact"], bounds=(0.0, 1.0)),
        "slot_value_hit": mean_confidence_interval(cell["slot_value_hit"], bounds=(0.0, 1.0)),
        "matrix_value_hit": mean_confidence_interval(cell["matrix_value_hit"], bounds=(0.0, 1.0)),
        "slot_bundle_cosine": mean_confidence_interval(cell["slot_bundle_cosine"], bounds=(-1.0, 1.0)),
        "matrix_bundle_cosine": mean_confidence_interval(cell["matrix_bundle_cosine"], bounds=(-1.0, 1.0)),
        "paired_exact_gain": paired_difference_stats(
            cell["matrix_bundle_exact"],
            cell["slot_bundle_exact"],
            SEED,
        ),
        "paired_hit_gain": paired_difference_stats(
            cell["matrix_value_hit"],
            cell["slot_value_hit"],
            SEED + 1,
        ),
    }


def build_summary(
    cells: dict[tuple[int, int, int, float], dict[str, list[float]]],
) -> dict[str, Any]:
    by_cell: dict[str, Any] = {}
    for (head_dim, cue_count, values_per_cue, noise_std), cell in cells.items():
        key = f"d{head_dim}_c{cue_count}_k{values_per_cue}_q{format(float(noise_std), 'g').replace('.', 'p')}"
        by_cell[key] = summarize_cell(cell)

    reference_key = (
        max(int(v) for v in HEAD_DIMS),
        max(int(v) for v in CUE_COUNTS),
        max(int(v) for v in VALUES_PER_CUE),
        min(float(v) for v in QUERY_NOISE),
    )
    reference_cell = summarize_cell(cells[reference_key])
    slot_bundle_exact_rate = reference_cell["slot_bundle_exact"]["mean"]
    matrix_bundle_exact_rate = reference_cell["matrix_bundle_exact"]["mean"]
    slot_value_hit_rate = reference_cell["slot_value_hit"]["mean"]
    matrix_value_hit_rate = reference_cell["matrix_value_hit"]["mean"]
    slot_minus_matrix_exact_gain = None
    if slot_bundle_exact_rate is not None and matrix_bundle_exact_rate is not None:
        slot_minus_matrix_exact_gain = float(slot_bundle_exact_rate) - float(matrix_bundle_exact_rate)
    return {
        "slot_bundle_exact_rate": slot_bundle_exact_rate,
        "slot_value_hit_rate": slot_value_hit_rate,
        "matrix_bundle_exact_rate": matrix_bundle_exact_rate,
        "matrix_value_hit_rate": matrix_value_hit_rate,
        "slot_minus_matrix_exact_gain": slot_minus_matrix_exact_gain,
        "by_cell": by_cell,
    }


def build_figure(
    cells: dict[tuple[int, int, int, float], dict[str, list[float]]],
    output_path: Path,
) -> None:
    apply_plot_style()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference_dim = max(int(v) for v in HEAD_DIMS)
    reference_cues = max(int(v) for v in CUE_COUNTS)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for noise_std in QUERY_NOISE:
        slot_series: list[float] = []
        matrix_series: list[float] = []
        x_values: list[int] = []
        for values_per_cue in VALUES_PER_CUE:
            key = (reference_dim, reference_cues, int(values_per_cue), float(noise_std))
            if key not in cells:
                continue
            slot_series.append(float(np.mean(cells[key]["slot_bundle_exact"])))
            matrix_series.append(float(np.mean(cells[key]["matrix_bundle_exact"])))
            x_values.append(int(values_per_cue))
        axes[0].plot(x_values, slot_series, marker="o", label=f"slot q={noise_std}")
        axes[0].plot(x_values, matrix_series, marker="x", linestyle="--", label=f"matrix q={noise_std}")
        slot_hits = []
        matrix_hits = []
        for values_per_cue in VALUES_PER_CUE:
            key = (reference_dim, reference_cues, int(values_per_cue), float(noise_std))
            if key not in cells:
                continue
            slot_hits.append(float(np.mean(cells[key]["slot_value_hit"])))
            matrix_hits.append(float(np.mean(cells[key]["matrix_value_hit"])))
        axes[1].plot(x_values, slot_hits, marker="o", label=f"slot q={noise_std}")
        axes[1].plot(x_values, matrix_hits, marker="x", linestyle="--", label=f"matrix q={noise_std}")
    axes[0].set_title("bundle exact rate")
    axes[1].set_title("value hit rate")
    for ax in axes:
        ax.set_xlabel("values per cue")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("rate")
    axes[0].legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> dict[str, object]:
    started = utc_now_iso()
    t0 = time.time()
    rng = build_rng(SEED)
    output_dir = output_dir_for(SCRIPT_PATH)
    cells: dict[tuple[int, int, int, float], dict[str, list[float]]] = {}
    for head_dim in HEAD_DIMS:
        for cue_count in CUE_COUNTS:
            for values_per_cue in VALUES_PER_CUE:
                for noise_std in QUERY_NOISE:
                    key = (int(head_dim), int(cue_count), int(values_per_cue), float(noise_std))
                    cells[key] = evaluate_cell(
                        rng=rng,
                        head_dim=int(head_dim),
                        cue_count=int(cue_count),
                        values_per_cue=int(values_per_cue),
                        noise_std=float(noise_std),
                        trial_count=TRIALS,
                    )

    summary = build_summary(cells)
    figure_path = output_dir / "multi_association_recall.png"
    build_figure(cells, figure_path)

    finished = utc_now_iso()
    duration = time.time() - t0
    metrics_path = output_dir / "multi_association_recall_metrics.json"
    trials: list[dict[str, Any]] = []
    for (head_dim, cue_count, values_per_cue, noise_std), cell in cells.items():
        for trial_idx, (
            slot_exact,
            matrix_exact,
            slot_hit,
            matrix_hit,
            slot_cos,
            matrix_cos,
        ) in enumerate(
            zip(
                cell["slot_bundle_exact"],
                cell["matrix_bundle_exact"],
                cell["slot_value_hit"],
                cell["matrix_value_hit"],
                cell["slot_bundle_cosine"],
                cell["matrix_bundle_cosine"],
                strict=True,
            )
        ):
            trials.append(
                {
                    "head_dim": int(head_dim),
                    "cue_count": int(cue_count),
                    "values_per_cue": int(values_per_cue),
                    "query_noise": float(noise_std),
                    "trial_id": int(trial_idx),
                    "slot_bundle_exact": float(slot_exact),
                    "matrix_bundle_exact": float(matrix_exact),
                    "slot_value_hit": float(slot_hit),
                    "matrix_value_hit": float(matrix_hit),
                    "slot_bundle_cosine": float(slot_cos),
                    "matrix_bundle_cosine": float(matrix_cos),
                }
            )
    record = build_run_record(
        simulation_name="multi_association_recall",
        script_path=SCRIPT_PATH,
        started_at_utc=started,
        finished_at_utc=finished,
        duration_sec=duration,
        parameters={
            "head_dims": [int(v) for v in HEAD_DIMS],
            "cue_counts": [int(v) for v in CUE_COUNTS],
            "values_per_cue": [int(v) for v in VALUES_PER_CUE],
            "query_noise": [float(v) for v in QUERY_NOISE],
            "temperature": float(TEMPERATURE),
        },
        seed_numpy=SEED,
        n_trials=len(trials),
        summary=summary,
        statistics={},
        trials=trials,
        artifacts=[
            {"name": "multi_association_recall.png", "path": figure_path, "type": "figure"},
            {"name": "multi_association_recall_metrics.json", "path": metrics_path, "type": "metrics"},
        ],
        warnings=[],
    )
    write_json(metrics_path, record)
    return summary


if __name__ == "__main__":
    main()
