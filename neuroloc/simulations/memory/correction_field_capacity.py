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
    paired_difference_stats,
    require_positive,
    require_positive_list,
    require_unit_interval_list,
    utc_now_iso,
    write_json,
)

SCRIPT_PATH = Path(__file__).resolve()
SEED = env_int("CF_SEED", 42)
HEAD_DIMS = env_list("CF_HEAD_DIMS", int, [32, 64, 128])
PATTERN_COUNTS = env_list("CF_PATTERN_COUNTS", int, [4, 8, 16, 32, 64])
PREDICTION_QUALITIES = env_list("CF_PRED_QUALITIES", float, [0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99])
DECAYS = env_list("CF_DECAYS", float, [0.5, 0.8, 0.9, 0.95, 1.0])
TRIALS = env_int("CF_TRIALS", 32)

require_positive_list("CF_HEAD_DIMS", HEAD_DIMS)
require_positive_list("CF_PATTERN_COUNTS", PATTERN_COUNTS)
require_unit_interval_list("CF_PRED_QUALITIES", PREDICTION_QUALITIES, allow_zero=True)
require_unit_interval_list("CF_DECAYS", DECAYS, allow_zero=True)
require_positive("CF_TRIALS", TRIALS)
if any(int(head_dim) < 2 for head_dim in HEAD_DIMS):
    raise ValueError("CF_HEAD_DIMS must be >= 2 for the orthogonal residual construction")

COSINE_THRESHOLD = 0.5
EPS = 1e-8
SURPRISE_PREDICTABLE_FRACTIONS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
SURPRISE_THRESHOLDS = [0.01, 0.05, 0.1, 0.2]
SURPRISE_PREDICTABLE_QUALITY = 0.99
SURPRISE_NOVEL_QUALITY = 0.0


def as_matrix(x: np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(x, dtype=np.float64)
    if array.ndim == 1:
        return array[None, :], True
    return array, False


def normalize_rows(x: np.ndarray) -> np.ndarray:
    matrix, squeezed = as_matrix(x)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    normalized = matrix / np.clip(norms, EPS, None)
    if squeezed:
        return normalized[0]
    return normalized


def row_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_matrix, _ = as_matrix(a)
    b_matrix, _ = as_matrix(b)
    numerators = np.sum(a_matrix * b_matrix, axis=-1)
    denominators = np.linalg.norm(a_matrix, axis=-1) * np.linalg.norm(b_matrix, axis=-1)
    return numerators / np.clip(denominators, EPS, None)


def cosine_score_matrix(queries: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    query_norms = np.linalg.norm(queries, axis=-1, keepdims=True)
    candidate_norms = np.linalg.norm(candidates, axis=-1, keepdims=True)
    normalized_queries = queries / np.clip(query_norms, EPS, None)
    normalized_candidates = candidates / np.clip(candidate_norms, EPS, None)
    return normalized_queries @ normalized_candidates.T


def orthogonal_unit_noise(rng: np.random.Generator, basis: np.ndarray) -> np.ndarray:
    basis_matrix = normalize_rows(basis)
    noise = rng.standard_normal(basis_matrix.shape)
    noise = noise - np.sum(noise * basis_matrix, axis=-1, keepdims=True) * basis_matrix
    retry_mask = np.linalg.norm(noise, axis=-1) <= EPS
    while np.any(retry_mask):
        replacement = rng.standard_normal((int(np.sum(retry_mask)), basis_matrix.shape[1]))
        replacement_basis = basis_matrix[retry_mask]
        replacement = replacement - np.sum(replacement * replacement_basis, axis=-1, keepdims=True) * replacement_basis
        noise[retry_mask] = replacement
        retry_mask = np.linalg.norm(noise, axis=-1) <= EPS
    return normalize_rows(noise)


def build_value_decomposition(
    rng: np.random.Generator,
    qualities: np.ndarray,
    head_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    qualities_array = np.asarray(qualities, dtype=np.float64)
    predictable_component = normalize_rows(rng.standard_normal((qualities_array.size, head_dim)))
    innovation_component = orthogonal_unit_noise(rng, predictable_component)
    prediction = predictable_component * qualities_array[:, None]
    residual = innovation_component * np.sqrt(np.clip(1.0 - qualities_array ** 2, 0.0, 1.0))[:, None]
    value = prediction + residual
    surprise_ratio = np.sum(residual ** 2, axis=-1) / np.clip(np.sum(value ** 2, axis=-1), EPS, None)
    return prediction, residual, value, surprise_ratio


def build_state(keys: np.ndarray, values: np.ndarray, decay: float, write_gates: np.ndarray | None = None) -> np.ndarray:
    state = np.zeros((keys.shape[1], values.shape[1]), dtype=np.float64)
    if write_gates is None:
        gates = np.ones(keys.shape[0], dtype=np.float64)
    else:
        gates = np.asarray(write_gates, dtype=np.float64)
    for key_vec, value_vec, gate in zip(keys, values, gates):
        state = decay * state + float(gate) * np.outer(key_vec, value_vec)
    return state


def evaluate_recall(
    keys: np.ndarray,
    target_values: np.ndarray,
    state: np.ndarray,
    reconstruction_base: np.ndarray | None = None,
    eval_indices: np.ndarray | None = None,
) -> dict[str, float]:
    if eval_indices is None:
        indices = np.arange(keys.shape[0], dtype=int)
    else:
        indices = np.asarray(eval_indices, dtype=int)
    if indices.size == 0:
        return {
            "mean_cosine_recall": 0.0,
            "exact_match_accuracy": 0.0,
            "mean_output_norm": 0.0,
        }
    retrieved = keys[indices] @ state
    reconstructed = retrieved if reconstruction_base is None else reconstruction_base[indices] + retrieved
    cosines = row_cosines(reconstructed, target_values[indices])
    nearest = np.argmax(cosine_score_matrix(reconstructed, target_values), axis=-1)
    return {
        "mean_cosine_recall": float(np.mean(cosines)),
        "exact_match_accuracy": float(np.mean(nearest == indices)),
        "mean_output_norm": float(np.mean(np.linalg.norm(reconstructed, axis=-1))),
    }


def value_quality_key(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return format(float(value), "g")


def filter_records(records: list[dict[str, Any]], sweep: str, metric: str, **filters: object) -> list[float]:
    values: list[float] = []
    for record in records:
        if record.get("sweep") != sweep:
            continue
        matches = True
        for key, expected in filters.items():
            value = record.get(key)
            if isinstance(expected, float):
                if value is None:
                    matches = False
                    break
                if not np.isclose(float(value), expected):
                    matches = False
                    break
            else:
                if value != expected:
                    matches = False
                    break
        if not matches:
            continue
        metric_value = record.get(metric)
        if metric_value is None:
            continue
        values.append(float(metric_value))
    return values


def closest_member(values: list[float], target: float) -> float:
    return min(values, key=lambda value: abs(float(value) - float(target)))


def summarize_capacity_cells(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    capacity_summary: dict[str, Any] = {}
    cosine_gain_summary: dict[str, Any] = {}
    paired_cosine_stats: dict[str, Any] = {}
    paired_accuracy_stats: dict[str, Any] = {}
    for head_dim in HEAD_DIMS:
        head_key = value_quality_key(head_dim)
        capacity_summary[head_key] = {}
        cosine_gain_summary[head_key] = {}
        paired_cosine_stats[head_key] = {}
        paired_accuracy_stats[head_key] = {}
        for decay in DECAYS:
            decay_key = value_quality_key(decay)
            capacity_summary[head_key][decay_key] = {}
            cosine_gain_summary[head_key][decay_key] = {}
            paired_cosine_stats[head_key][decay_key] = {}
            paired_accuracy_stats[head_key][decay_key] = {}
            for quality in PREDICTION_QUALITIES:
                quality_key = value_quality_key(quality)
                raw_capacity_candidates: list[int] = []
                correction_memory_capacity_candidates: list[int] = []
                prediction_only_capacity_candidates: list[int] = []
                correction_reconstruction_capacity_candidates: list[int] = []
                cosine_gain_summary[head_key][decay_key][quality_key] = {}
                paired_cosine_stats[head_key][decay_key][quality_key] = {}
                paired_accuracy_stats[head_key][decay_key][quality_key] = {}
                for pattern_count in PATTERN_COUNTS:
                    pattern_key = value_quality_key(pattern_count)
                    raw_values = filter_records(
                        records,
                        sweep="capacity",
                        metric="raw_mean_cosine_recall",
                        head_dim=head_dim,
                        decay=decay,
                        prediction_quality=quality,
                        pattern_count=pattern_count,
                    )
                    correction_memory_values = filter_records(
                        records,
                        sweep="capacity",
                        metric="correction_residual_mean_cosine_recall",
                        head_dim=head_dim,
                        decay=decay,
                        prediction_quality=quality,
                        pattern_count=pattern_count,
                    )
                    prediction_only_values = filter_records(
                        records,
                        sweep="capacity",
                        metric="prediction_only_mean_cosine_recall",
                        head_dim=head_dim,
                        decay=decay,
                        prediction_quality=quality,
                        pattern_count=pattern_count,
                    )
                    correction_reconstruction_values = filter_records(
                        records,
                        sweep="capacity",
                        metric="correction_reconstruction_mean_cosine_recall",
                        head_dim=head_dim,
                        decay=decay,
                        prediction_quality=quality,
                        pattern_count=pattern_count,
                    )
                    raw_accuracy = filter_records(
                        records,
                        sweep="capacity",
                        metric="raw_exact_match_accuracy",
                        head_dim=head_dim,
                        decay=decay,
                        prediction_quality=quality,
                        pattern_count=pattern_count,
                    )
                    correction_memory_accuracy = filter_records(
                        records,
                        sweep="capacity",
                        metric="correction_residual_exact_match_accuracy",
                        head_dim=head_dim,
                        decay=decay,
                        prediction_quality=quality,
                        pattern_count=pattern_count,
                    )
                    prediction_only_accuracy = filter_records(
                        records,
                        sweep="capacity",
                        metric="prediction_only_exact_match_accuracy",
                        head_dim=head_dim,
                        decay=decay,
                        prediction_quality=quality,
                        pattern_count=pattern_count,
                    )
                    correction_reconstruction_accuracy = filter_records(
                        records,
                        sweep="capacity",
                        metric="correction_reconstruction_exact_match_accuracy",
                        head_dim=head_dim,
                        decay=decay,
                        prediction_quality=quality,
                        pattern_count=pattern_count,
                    )
                    if raw_values and float(np.mean(raw_values)) > COSINE_THRESHOLD:
                        raw_capacity_candidates.append(int(pattern_count))
                    if correction_memory_values and float(np.mean(correction_memory_values)) > COSINE_THRESHOLD:
                        correction_memory_capacity_candidates.append(int(pattern_count))
                    if prediction_only_values and float(np.mean(prediction_only_values)) > COSINE_THRESHOLD:
                        prediction_only_capacity_candidates.append(int(pattern_count))
                    if correction_reconstruction_values and float(np.mean(correction_reconstruction_values)) > COSINE_THRESHOLD:
                        correction_reconstruction_capacity_candidates.append(int(pattern_count))
                    paired_cosine_stats[head_key][decay_key][quality_key][pattern_key] = {
                        "raw": mean_confidence_interval(raw_values, bounds=(0.0, 1.0)),
                        "correction_field_memory": mean_confidence_interval(correction_memory_values, bounds=(0.0, 1.0)),
                        "prediction_only": mean_confidence_interval(prediction_only_values, bounds=(0.0, 1.0)),
                        "correction_field_reconstruction": mean_confidence_interval(correction_reconstruction_values, bounds=(0.0, 1.0)),
                        "memory_gain_vs_raw": paired_difference_stats(raw_values, correction_memory_values, SEED + head_dim + pattern_count),
                        "reconstruction_gain_vs_prediction": paired_difference_stats(
                            prediction_only_values,
                            correction_reconstruction_values,
                            SEED + head_dim + pattern_count + 11,
                        ),
                    }
                    paired_accuracy_stats[head_key][decay_key][quality_key][pattern_key] = {
                        "raw": mean_confidence_interval(raw_accuracy, bounds=(0.0, 1.0)),
                        "correction_field_memory": mean_confidence_interval(correction_memory_accuracy, bounds=(0.0, 1.0)),
                        "prediction_only": mean_confidence_interval(prediction_only_accuracy, bounds=(0.0, 1.0)),
                        "correction_field_reconstruction": mean_confidence_interval(correction_reconstruction_accuracy, bounds=(0.0, 1.0)),
                        "memory_gain_vs_raw": paired_difference_stats(
                            raw_accuracy,
                            correction_memory_accuracy,
                            SEED + head_dim + pattern_count + 1,
                        ),
                        "reconstruction_gain_vs_prediction": paired_difference_stats(
                            prediction_only_accuracy,
                            correction_reconstruction_accuracy,
                            SEED + head_dim + pattern_count + 12,
                        ),
                    }
                    cosine_gain_summary[head_key][decay_key][quality_key][pattern_key] = {
                        "memory_gain_vs_raw": float(np.mean(correction_memory_values) - np.mean(raw_values)),
                        "reconstruction_gain_vs_prediction": float(
                            np.mean(correction_reconstruction_values) - np.mean(prediction_only_values)
                        ),
                    }
                raw_capacity = max(raw_capacity_candidates, default=0)
                correction_memory_capacity = max(correction_memory_capacity_candidates, default=0)
                prediction_only_capacity = max(prediction_only_capacity_candidates, default=0)
                correction_reconstruction_capacity = max(correction_reconstruction_capacity_candidates, default=0)
                capacity_summary[head_key][decay_key][quality_key] = {
                    "raw_effective_capacity": raw_capacity,
                    "correction_field_memory_effective_capacity": correction_memory_capacity,
                    "memory_capacity_gain": int(correction_memory_capacity - raw_capacity),
                    "prediction_only_effective_capacity": prediction_only_capacity,
                    "correction_field_reconstruction_effective_capacity": correction_reconstruction_capacity,
                    "reconstruction_capacity_gain_over_prediction": int(
                        correction_reconstruction_capacity - prediction_only_capacity
                    ),
                }
    return capacity_summary, cosine_gain_summary, paired_cosine_stats, paired_accuracy_stats


def summarize_threshold_shift(records: list[dict[str, Any]]) -> dict[str, Any]:
    threshold_quality = closest_member(PREDICTION_QUALITIES, 0.9)
    threshold_pattern_count = 32 if 32 in PATTERN_COUNTS else int(closest_member([float(value) for value in PATTERN_COUNTS], 32.0))
    summary: dict[str, Any] = {}
    for head_dim in HEAD_DIMS:
        raw_threshold = None
        correction_memory_threshold = None
        prediction_only_threshold = None
        correction_reconstruction_threshold = None
        for decay in sorted(DECAYS):
            raw_values = filter_records(
                records,
                sweep="capacity",
                metric="raw_mean_cosine_recall",
                head_dim=head_dim,
                decay=decay,
                prediction_quality=threshold_quality,
                pattern_count=threshold_pattern_count,
            )
            correction_memory_values = filter_records(
                records,
                sweep="capacity",
                metric="correction_residual_mean_cosine_recall",
                head_dim=head_dim,
                decay=decay,
                prediction_quality=threshold_quality,
                pattern_count=threshold_pattern_count,
            )
            prediction_only_values = filter_records(
                records,
                sweep="capacity",
                metric="prediction_only_mean_cosine_recall",
                head_dim=head_dim,
                decay=decay,
                prediction_quality=threshold_quality,
                pattern_count=threshold_pattern_count,
            )
            correction_reconstruction_values = filter_records(
                records,
                sweep="capacity",
                metric="correction_reconstruction_mean_cosine_recall",
                head_dim=head_dim,
                decay=decay,
                prediction_quality=threshold_quality,
                pattern_count=threshold_pattern_count,
            )
            if raw_threshold is None and raw_values and float(np.mean(raw_values)) > COSINE_THRESHOLD:
                raw_threshold = float(decay)
            if correction_memory_threshold is None and correction_memory_values and float(np.mean(correction_memory_values)) > COSINE_THRESHOLD:
                correction_memory_threshold = float(decay)
            if prediction_only_threshold is None and prediction_only_values and float(np.mean(prediction_only_values)) > COSINE_THRESHOLD:
                prediction_only_threshold = float(decay)
            if correction_reconstruction_threshold is None and correction_reconstruction_values and float(np.mean(correction_reconstruction_values)) > COSINE_THRESHOLD:
                correction_reconstruction_threshold = float(decay)
        memory_shift = None
        reconstruction_shift = None
        if raw_threshold is not None and correction_memory_threshold is not None:
            memory_shift = float(raw_threshold - correction_memory_threshold)
        if prediction_only_threshold is not None and correction_reconstruction_threshold is not None:
            reconstruction_shift = float(prediction_only_threshold - correction_reconstruction_threshold)
        summary[value_quality_key(head_dim)] = {
            "prediction_quality": float(threshold_quality),
            "pattern_count": int(threshold_pattern_count),
            "raw_threshold_decay": raw_threshold,
            "correction_field_memory_threshold_decay": correction_memory_threshold,
            "memory_threshold_shift": memory_shift,
            "prediction_only_threshold_decay": prediction_only_threshold,
            "correction_field_reconstruction_threshold_decay": correction_reconstruction_threshold,
            "reconstruction_threshold_shift_over_prediction": reconstruction_shift,
        }
    return summary


def summarize_observed_prediction_fraction(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for quality in PREDICTION_QUALITIES:
        observed = filter_records(
            records,
            sweep="capacity",
            metric="observed_prediction_fraction",
            prediction_quality=quality,
        )
        summary[value_quality_key(quality)] = mean_confidence_interval(observed, bounds=(0.0, 1.0))
    return summary


def summarize_surprise_sweep(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    write_fraction_summary: dict[str, Any] = {}
    recall_gain_summary: dict[str, Any] = {}
    for tau in SURPRISE_THRESHOLDS:
        tau_key = value_quality_key(tau)
        write_fraction_summary[tau_key] = {}
        recall_gain_summary[tau_key] = {}
        for predictable_fraction in SURPRISE_PREDICTABLE_FRACTIONS:
            predictable_key = value_quality_key(predictable_fraction)
            write_fractions = filter_records(
                records,
                sweep="surprise_gating",
                metric="write_fraction",
                surprise_threshold=tau,
                predictable_fraction=predictable_fraction,
            )
            ungated = filter_records(
                records,
                sweep="surprise_gating",
                metric="ungated_mean_cosine_recall",
                surprise_threshold=tau,
                predictable_fraction=predictable_fraction,
            )
            gated = filter_records(
                records,
                sweep="surprise_gating",
                metric="gated_mean_cosine_recall",
                surprise_threshold=tau,
                predictable_fraction=predictable_fraction,
            )
            write_fraction_summary[tau_key][predictable_key] = mean_confidence_interval(write_fractions, bounds=(0.0, 1.0))
            recall_gain_summary[tau_key][predictable_key] = {
                "ungated": mean_confidence_interval(ungated, bounds=(0.0, 1.0)),
                "gated": mean_confidence_interval(gated, bounds=(0.0, 1.0)),
                "gain": paired_difference_stats(ungated, gated, SEED + int(round(tau * 1000)) + int(round(predictable_fraction * 1000))),
            }
    return write_fraction_summary, recall_gain_summary


def heatmap_matrix(
    records: list[dict[str, Any]],
    head_dim: int,
    decay: float,
    metric: str,
) -> np.ndarray:
    matrix = np.zeros((len(PREDICTION_QUALITIES), len(PATTERN_COUNTS)), dtype=np.float64)
    for q_idx, quality in enumerate(PREDICTION_QUALITIES):
        for p_idx, pattern_count in enumerate(PATTERN_COUNTS):
            values = filter_records(
                records,
                sweep="capacity",
                metric=metric,
                head_dim=head_dim,
                decay=decay,
                prediction_quality=quality,
                pattern_count=pattern_count,
            )
            matrix[q_idx, p_idx] = float(np.mean(values)) if values else np.nan
    return matrix


def selected_heatmap_decay() -> float:
    if any(np.isclose(float(decay), 0.9) for decay in DECAYS):
        return float(closest_member([float(value) for value in DECAYS], 0.9))
    return float(DECAYS[min(len(DECAYS) - 1, 0)])


def build_heatmaps_figure(records: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = len(HEAD_DIMS) * len(DECAYS)
    fig, axes = plt.subplots(rows, 2, figsize=(12, max(8, rows * 2.1)), constrained_layout=True)
    if rows == 1:
        axes = np.asarray([axes])
    image = None
    row_idx = 0
    for head_dim in HEAD_DIMS:
        for decay in DECAYS:
            raw_matrix = heatmap_matrix(records, head_dim, decay, "raw_mean_cosine_recall")
            correction_matrix = heatmap_matrix(records, head_dim, decay, "correction_reconstruction_mean_cosine_recall")
            for col_idx, (title, matrix) in enumerate((("raw values", raw_matrix), ("correction-field reconstruction", correction_matrix))):
                ax = axes[row_idx, col_idx]
                image = ax.imshow(matrix, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
                ax.set_xticks(range(len(PATTERN_COUNTS)))
                ax.set_xticklabels([str(value) for value in PATTERN_COUNTS])
                ax.set_yticks(range(len(PREDICTION_QUALITIES)))
                ax.set_yticklabels([format(float(value), "g") for value in PREDICTION_QUALITIES])
                ax.set_title(f"{title} | d={head_dim} decay={format(float(decay), 'g')}")
                ax.set_xlabel("pattern count")
                if col_idx == 0:
                    ax.set_ylabel("prediction quality")
            row_idx += 1
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85, label="mean cosine recall")
    plt.savefig(output_path)
    plt.close(fig)


def build_summary_figure(
    capacity_summary: dict[str, Any],
    threshold_shift_summary: dict[str, Any],
    surprise_write_summary: dict[str, Any],
    surprise_recall_summary: dict[str, Any],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    selected_decay = selected_heatmap_decay()
    selected_decay_key = value_quality_key(selected_decay)
    selected_tau = value_quality_key(closest_member([float(value) for value in SURPRISE_THRESHOLDS], 0.1))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    ax = axes[0, 0]
    for head_dim in HEAD_DIMS:
        gains = []
        for quality in PREDICTION_QUALITIES:
            entry = capacity_summary[value_quality_key(head_dim)][selected_decay_key][value_quality_key(quality)]
            gains.append(float(entry["memory_capacity_gain"]))
        ax.plot(PREDICTION_QUALITIES, gains, marker="o", label=f"d={head_dim}")
    ax.axhline(0.0, color="#111827", linewidth=1)
    ax.set_xlabel("prediction quality")
    ax.set_ylabel("memory capacity gain")
    ax.set_title(f"memory gain at decay={format(selected_decay, 'g')}")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    x_positions = np.arange(len(HEAD_DIMS))
    raw_thresholds = []
    correction_thresholds = []
    for head_dim in HEAD_DIMS:
        entry = threshold_shift_summary[value_quality_key(head_dim)]
        raw_thresholds.append(entry["raw_threshold_decay"] if entry["raw_threshold_decay"] is not None else np.nan)
        correction_thresholds.append(
            entry["correction_field_memory_threshold_decay"] if entry["correction_field_memory_threshold_decay"] is not None else np.nan
        )
    ax.bar(x_positions - 0.18, raw_thresholds, width=0.36, label="raw")
    ax.bar(x_positions + 0.18, correction_thresholds, width=0.36, label="correction field memory")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"d={value}" for value in HEAD_DIMS])
    ax.set_ylabel("first decay above 0.5 cosine")
    ax.set_title("threshold decay at q=0.9, n=32")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for tau in SURPRISE_THRESHOLDS:
        tau_key = value_quality_key(tau)
        ys = [
            surprise_write_summary[tau_key][value_quality_key(predictable_fraction)]["mean"]
            for predictable_fraction in SURPRISE_PREDICTABLE_FRACTIONS
        ]
        ax.plot(SURPRISE_PREDICTABLE_FRACTIONS, ys, marker="o", label=f"tau={format(float(tau), 'g')}")
    ax.set_xlabel("predictable fraction")
    ax.set_ylabel("write fraction")
    ax.set_title("surprise-gated write frequency")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ys = [
        surprise_recall_summary[selected_tau][value_quality_key(predictable_fraction)]["gain"]["mean_difference"]
        for predictable_fraction in SURPRISE_PREDICTABLE_FRACTIONS
    ]
    ax.plot(SURPRISE_PREDICTABLE_FRACTIONS, ys, marker="o", color="#0f766e")
    ax.axhline(0.0, color="#111827", linewidth=1)
    ax.set_xlabel("predictable fraction")
    ax.set_ylabel("gated - ungated cosine")
    ax.set_title(f"surprising-token recall gain at tau={selected_tau}")

    plt.savefig(output_path)
    plt.close(fig)


def main() -> None:
    apply_plot_style()

    started = utc_now_iso()
    t0 = time.time()
    rng = build_rng(SEED)

    capacity_records: list[dict[str, Any]] = []
    for head_dim in HEAD_DIMS:
        for decay in DECAYS:
            for prediction_quality in PREDICTION_QUALITIES:
                qualities = np.full(max(PATTERN_COUNTS), float(prediction_quality), dtype=np.float64)
                for pattern_count in PATTERN_COUNTS:
                    for trial_id in range(TRIALS):
                        trial_rng = child_rng(rng)
                        keys = normalize_rows(trial_rng.standard_normal((pattern_count, head_dim)))
                        prediction, residual, value, surprise_ratio = build_value_decomposition(
                            trial_rng,
                            qualities[:pattern_count],
                            head_dim,
                        )
                        raw_state = build_state(keys, value, decay)
                        correction_state = build_state(keys, residual, decay)
                        prediction_only_metrics = evaluate_recall(
                            keys,
                            value,
                            np.zeros_like(correction_state),
                            reconstruction_base=prediction,
                        )
                        raw_metrics = evaluate_recall(keys, value, raw_state)
                        correction_memory_metrics = evaluate_recall(keys, residual, correction_state)
                        correction_reconstruction_metrics = evaluate_recall(
                            keys,
                            value,
                            correction_state,
                            reconstruction_base=prediction,
                        )
                        prediction_energy = np.sum(prediction ** 2, axis=-1)
                        residual_energy = np.sum(residual ** 2, axis=-1)
                        value_energy = np.sum(value ** 2, axis=-1)
                        capacity_records.append(
                            {
                                "sweep": "capacity",
                                "scenario_id": f"d{head_dim}_decay{format(float(decay), 'g')}_q{format(float(prediction_quality), 'g')}_n{pattern_count}_t{trial_id}",
                                "trial_id": int(trial_id),
                                "head_dim": int(head_dim),
                                "pattern_count": int(pattern_count),
                                "prediction_quality": float(prediction_quality),
                                "decay": float(decay),
                                "observed_prediction_fraction": float(np.mean(prediction_energy / np.clip(value_energy, EPS, None))),
                                "observed_residual_fraction": float(np.mean(residual_energy / np.clip(value_energy, EPS, None))),
                                "observed_surprise_ratio": float(np.mean(surprise_ratio)),
                                "prediction_only_mean_cosine_recall": prediction_only_metrics["mean_cosine_recall"],
                                "prediction_only_exact_match_accuracy": prediction_only_metrics["exact_match_accuracy"],
                                "raw_mean_cosine_recall": raw_metrics["mean_cosine_recall"],
                                "correction_residual_mean_cosine_recall": correction_memory_metrics["mean_cosine_recall"],
                                "correction_reconstruction_mean_cosine_recall": correction_reconstruction_metrics["mean_cosine_recall"],
                                "raw_exact_match_accuracy": raw_metrics["exact_match_accuracy"],
                                "correction_residual_exact_match_accuracy": correction_memory_metrics["exact_match_accuracy"],
                                "correction_reconstruction_exact_match_accuracy": correction_reconstruction_metrics["exact_match_accuracy"],
                                "prediction_only_mean_output_norm": prediction_only_metrics["mean_output_norm"],
                                "raw_mean_output_norm": raw_metrics["mean_output_norm"],
                                "correction_residual_mean_output_norm": correction_memory_metrics["mean_output_norm"],
                                "correction_reconstruction_mean_output_norm": correction_reconstruction_metrics["mean_output_norm"],
                                "raw_state_fro_norm": float(np.linalg.norm(raw_state)),
                                "correction_state_fro_norm": float(np.linalg.norm(correction_state)),
                            }
                        )

    surprise_records: list[dict[str, Any]] = []
    surprise_head_dim = 64 if 64 in HEAD_DIMS else int(HEAD_DIMS[min(len(HEAD_DIMS) - 1, 0)])
    surprise_pattern_count = max(PATTERN_COUNTS)
    surprise_decay = closest_member([float(value) for value in DECAYS], 0.9)
    for predictable_fraction in SURPRISE_PREDICTABLE_FRACTIONS:
        predictable_count = int(round(surprise_pattern_count * predictable_fraction))
        predictable_count = min(max(predictable_count, 1), surprise_pattern_count - 1)
        novel_count = surprise_pattern_count - predictable_count
        for trial_id in range(TRIALS):
            trial_rng = child_rng(rng)
            qualities = np.array(
                [SURPRISE_PREDICTABLE_QUALITY] * predictable_count + [SURPRISE_NOVEL_QUALITY] * novel_count,
                dtype=np.float64,
            )
            qualities = qualities[trial_rng.permutation(surprise_pattern_count)]
            keys = normalize_rows(trial_rng.standard_normal((surprise_pattern_count, surprise_head_dim)))
            prediction, residual, value, surprise_ratio = build_value_decomposition(trial_rng, qualities, surprise_head_dim)
            ungated_state = build_state(keys, residual, surprise_decay)
            surprising_indices = np.flatnonzero(np.isclose(qualities, SURPRISE_NOVEL_QUALITY))
            predictable_indices = np.flatnonzero(np.isclose(qualities, SURPRISE_PREDICTABLE_QUALITY))
            for surprise_threshold in SURPRISE_THRESHOLDS:
                write_mask = (surprise_ratio > surprise_threshold).astype(np.float64)
                gated_state = build_state(keys, residual, surprise_decay, write_gates=write_mask)
                ungated_metrics = evaluate_recall(
                    keys,
                    value,
                    ungated_state,
                    reconstruction_base=prediction,
                    eval_indices=surprising_indices,
                )
                gated_metrics = evaluate_recall(
                    keys,
                    value,
                    gated_state,
                    reconstruction_base=prediction,
                    eval_indices=surprising_indices,
                )
                surprise_records.append(
                    {
                        "sweep": "surprise_gating",
                        "scenario_id": f"p{format(float(predictable_fraction), 'g')}_tau{format(float(surprise_threshold), 'g')}_t{trial_id}",
                        "trial_id": int(trial_id),
                        "head_dim": int(surprise_head_dim),
                        "pattern_count": int(surprise_pattern_count),
                        "decay": float(surprise_decay),
                        "predictable_fraction": float(predictable_fraction),
                        "predictable_count": int(predictable_count),
                        "novel_count": int(novel_count),
                        "surprise_threshold": float(surprise_threshold),
                        "write_fraction": float(np.mean(write_mask)),
                        "predictable_write_fraction": float(np.mean(write_mask[predictable_indices])) if predictable_indices.size else 0.0,
                        "novel_write_fraction": float(np.mean(write_mask[surprising_indices])) if surprising_indices.size else 0.0,
                        "predictable_surprise_ratio": float(np.mean(surprise_ratio[predictable_indices])) if predictable_indices.size else 0.0,
                        "novel_surprise_ratio": float(np.mean(surprise_ratio[surprising_indices])) if surprising_indices.size else 0.0,
                        "ungated_mean_cosine_recall": ungated_metrics["mean_cosine_recall"],
                        "gated_mean_cosine_recall": gated_metrics["mean_cosine_recall"],
                        "ungated_exact_match_accuracy": ungated_metrics["exact_match_accuracy"],
                        "gated_exact_match_accuracy": gated_metrics["exact_match_accuracy"],
                    }
                )

    all_records = [*capacity_records, *surprise_records]
    capacity_summary, cosine_gain_summary, paired_cosine_stats, paired_accuracy_stats = summarize_capacity_cells(capacity_records)
    threshold_shift_summary = summarize_threshold_shift(capacity_records)
    observed_prediction_summary = summarize_observed_prediction_fraction(capacity_records)
    surprise_write_summary, surprise_recall_summary = summarize_surprise_sweep(surprise_records)

    output_dir = output_dir_for(SCRIPT_PATH)
    heatmap_path = output_dir / "correction_field_capacity_heatmaps.png"
    summary_path = output_dir / "correction_field_capacity_summary.png"
    metrics_path = output_dir / "correction_field_capacity_metrics.json"

    build_heatmaps_figure(capacity_records, heatmap_path)
    build_summary_figure(
        capacity_summary,
        threshold_shift_summary,
        surprise_write_summary,
        surprise_recall_summary,
        summary_path,
    )

    warnings: list[str] = []
    for head_dim in HEAD_DIMS:
        threshold_entry = threshold_shift_summary[value_quality_key(head_dim)]
        if threshold_entry["raw_threshold_decay"] is None:
            warnings.append(f"raw baseline never crossed {COSINE_THRESHOLD} cosine for head_dim={head_dim} at q=0.9, n=32")
        if threshold_entry["correction_field_memory_threshold_decay"] is None:
            warnings.append(f"correction-field memory never crossed {COSINE_THRESHOLD} cosine for head_dim={head_dim} at q=0.9, n=32")
    warnings.append(
        "surprise_gating uses a two-class toy mixture with q in {0.99, 0.0}; write fractions there measure separability of that mixture, not a graded learned surprise estimator."
    )

    finished = utc_now_iso()
    duration = time.time() - t0

    record = build_run_record(
        simulation_name="correction_field_capacity",
        script_path=SCRIPT_PATH,
        started_at_utc=started,
        finished_at_utc=finished,
        duration_sec=duration,
        parameters={
            "head_dims": HEAD_DIMS,
            "pattern_counts": PATTERN_COUNTS,
            "prediction_qualities": PREDICTION_QUALITIES,
            "decays": DECAYS,
            "trials": TRIALS,
            "cosine_threshold": COSINE_THRESHOLD,
            "surprise_predictable_fractions": SURPRISE_PREDICTABLE_FRACTIONS,
            "surprise_thresholds": SURPRISE_THRESHOLDS,
            "surprise_predictable_quality": SURPRISE_PREDICTABLE_QUALITY,
            "surprise_novel_quality": SURPRISE_NOVEL_QUALITY,
        },
        seed_numpy=SEED,
        n_trials=len(all_records),
        summary={
            "capacity_raw_vs_corrfield_by_quality": capacity_summary,
            "cosine_gain_by_quality_and_patterns": cosine_gain_summary,
            "threshold_shift_by_decay": threshold_shift_summary,
            "observed_prediction_variance_by_quality": observed_prediction_summary,
            "surprise_write_fraction_by_predictable_fraction": surprise_write_summary,
            "surprise_recall_gain_by_threshold": surprise_recall_summary,
        },
        statistics={
            "paired_cosine_gain_by_cell": paired_cosine_stats,
            "paired_exact_match_gain_by_cell": paired_accuracy_stats,
        },
        trials=all_records,
        artifacts=[
            {"name": heatmap_path.name, "path": heatmap_path, "type": "plot"},
            {"name": summary_path.name, "path": summary_path, "type": "plot"},
            {"name": metrics_path.name, "path": metrics_path, "type": "metrics"},
        ],
        warnings=warnings,
    )
    write_json(metrics_path, record)

    selected_decay = selected_heatmap_decay()
    selected_decay_key = value_quality_key(selected_decay)
    selected_quality_key = value_quality_key(closest_member([float(value) for value in PREDICTION_QUALITIES], 0.9))
    selected_capacity = capacity_summary[value_quality_key(surprise_head_dim)][selected_decay_key][selected_quality_key]
    print(
        "correction_field_capacity "
        f"d={surprise_head_dim} decay={format(selected_decay, 'g')} q={selected_quality_key} "
        f"raw_capacity={selected_capacity['raw_effective_capacity']} "
        f"prediction_only_capacity={selected_capacity['prediction_only_effective_capacity']} "
        f"corr_memory_capacity={selected_capacity['correction_field_memory_effective_capacity']} "
        f"recon_capacity={selected_capacity['correction_field_reconstruction_effective_capacity']}"
    )
    print(f"wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()