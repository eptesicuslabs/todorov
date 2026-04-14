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
SEED = env_int("CFT_SEED", 42)
HEAD_DIMS = env_list("CFT_HEAD_DIMS", int, [32, 64, 128])
PATTERN_COUNTS = env_list("CFT_PATTERN_COUNTS", int, [4, 8, 16, 32, 64])
TEMPORAL_CORRELATIONS = env_list("CFT_TEMPORAL_CORRELATIONS", float, [0.5, 0.7, 0.9, 0.99])
DECAYS = env_list("CFT_DECAYS", float, [0.5, 0.9, 0.95])
TRIALS = env_int("CFT_TRIALS", 16)
TRAIN_SEQUENCES = env_int("CFT_TRAIN_SEQUENCES", 2000)
TRAIN_SEQUENCE_LENGTH = env_int("CFT_TRAIN_SEQUENCE_LENGTH", 32)
TRAIN_ITERS = env_int("CFT_TRAIN_ITERS", 200)
TRAIN_LR = float(__import__("os").environ.get("CFT_TRAIN_LR", "1e-3"))
TRAIN_RANK = env_int("CFT_TRAIN_RANK", 16)
NOISE_SCALE = float(__import__("os").environ.get("CFT_NOISE_SCALE", "0.3"))
BATCH_SIZE = env_int("CFT_BATCH_SIZE", 256)

require_positive_list("CFT_HEAD_DIMS", HEAD_DIMS)
require_positive_list("CFT_PATTERN_COUNTS", PATTERN_COUNTS)
require_unit_interval_list("CFT_TEMPORAL_CORRELATIONS", TEMPORAL_CORRELATIONS, allow_zero=True)
require_unit_interval_list("CFT_DECAYS", DECAYS, allow_zero=True)
require_positive("CFT_TRIALS", TRIALS)
require_positive("CFT_TRAIN_SEQUENCES", TRAIN_SEQUENCES)
require_positive("CFT_TRAIN_SEQUENCE_LENGTH", TRAIN_SEQUENCE_LENGTH)
require_positive("CFT_TRAIN_ITERS", TRAIN_ITERS)
require_positive("CFT_TRAIN_RANK", TRAIN_RANK)
require_positive("CFT_BATCH_SIZE", BATCH_SIZE)
if TRAIN_LR <= 0:
    raise ValueError("CFT_TRAIN_LR must be > 0")
if NOISE_SCALE < 0:
    raise ValueError("CFT_NOISE_SCALE must be >= 0")

COSINE_THRESHOLD = 0.5
EPS = 1e-8
HELD_OUT_SEQUENCES = 512


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


def random_orthogonal(rng: np.random.Generator, dim: int) -> np.ndarray:
    matrix = rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(matrix)
    sign_correction = np.sign(np.diag(r))
    sign_correction[sign_correction == 0] = 1.0
    return q * sign_correction


def generate_correlated_sequences(
    rng: np.random.Generator,
    mixing_matrix: np.ndarray,
    temporal_correlation: float,
    n_sequences: int,
    sequence_length: int,
    noise_scale: float,
) -> np.ndarray:
    dim = mixing_matrix.shape[0]
    rho = float(np.clip(temporal_correlation, 0.0, 1.0 - 1e-6))
    innovation_scale = float(np.sqrt(max(1.0 - rho * rho, 0.0)))
    u = rng.standard_normal((n_sequences, dim))
    samples = np.empty((n_sequences, sequence_length, dim), dtype=np.float64)
    for t in range(sequence_length):
        eta = rng.standard_normal((n_sequences, dim))
        u = rho * u + innovation_scale * eta
        xi = rng.standard_normal((n_sequences, dim)) * noise_scale
        samples[:, t, :] = u @ mixing_matrix.T + xi
    return samples


def momentum_optimizer_init(shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(shape, dtype=np.float64), np.zeros(shape, dtype=np.float64)


def momentum_optimizer_step(
    param: np.ndarray,
    grad: np.ndarray,
    first_moment: np.ndarray,
    second_moment: np.ndarray,
    step: int,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = beta1 * first_moment + (1.0 - beta1) * grad
    second = beta2 * second_moment + (1.0 - beta2) * (grad * grad)
    first_corrected = first / (1.0 - beta1 ** step)
    second_corrected = second / (1.0 - beta2 ** step)
    updated = param - learning_rate * first_corrected / (np.sqrt(second_corrected) + eps)
    return updated, first, second


def train_prediction_head(
    rng: np.random.Generator,
    mixing_matrix: np.ndarray,
    temporal_correlation: float,
    head_dim: int,
    rank: int,
    n_sequences: int,
    sequence_length: int,
    noise_scale: float,
    n_iterations: int,
    learning_rate: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float, float, list[float]]:
    train_samples = generate_correlated_sequences(
        rng,
        mixing_matrix,
        temporal_correlation,
        n_sequences,
        sequence_length,
        noise_scale,
    )
    inputs = train_samples[:, :-1, :].reshape(-1, head_dim)
    targets = train_samples[:, 1:, :].reshape(-1, head_dim)
    n_samples = inputs.shape[0]
    scale_up = np.sqrt(1.0 / max(head_dim, 1))
    scale_down = np.sqrt(1.0 / max(rank, 1))
    w_up = rng.standard_normal((head_dim, rank)) * scale_up
    w_down = rng.standard_normal((rank, head_dim)) * scale_down
    m_up, v_up = momentum_optimizer_init(w_up.shape)
    m_down, v_down = momentum_optimizer_init(w_down.shape)
    initial_variance = float(np.var(targets))
    loss_trajectory: list[float] = []
    effective_batch = int(min(batch_size, n_samples))
    for iteration in range(1, int(n_iterations) + 1):
        indices = rng.integers(0, n_samples, size=effective_batch)
        x_batch = inputs[indices]
        y_batch = targets[indices]
        hidden = x_batch @ w_up
        prediction = hidden @ w_down
        residual = prediction - y_batch
        loss = float(np.mean(residual * residual))
        loss_trajectory.append(loss)
        grad_w_down = hidden.T @ residual / float(effective_batch) * 2.0
        grad_hidden = residual @ w_down.T
        grad_w_up = x_batch.T @ grad_hidden / float(effective_batch) * 2.0
        w_up, m_up, v_up = momentum_optimizer_step(w_up, grad_w_up, m_up, v_up, iteration, learning_rate)
        w_down, m_down, v_down = momentum_optimizer_step(w_down, grad_w_down, m_down, v_down, iteration, learning_rate)
    held_out = generate_correlated_sequences(
        rng,
        mixing_matrix,
        temporal_correlation,
        HELD_OUT_SEQUENCES,
        sequence_length,
        noise_scale,
    )
    held_inputs = held_out[:, :-1, :].reshape(-1, head_dim)
    held_targets = held_out[:, 1:, :].reshape(-1, head_dim)
    held_predictions = (held_inputs @ w_up) @ w_down
    held_residuals = held_targets - held_predictions
    target_variance = float(np.var(held_targets))
    residual_variance = float(np.var(held_residuals))
    observed_prediction_quality = float(1.0 - residual_variance / max(target_variance, EPS))
    final_loss = float(loss_trajectory[-1]) if loss_trajectory else float("nan")
    return w_up, w_down, observed_prediction_quality, final_loss, loss_trajectory


def build_state(keys: np.ndarray, values: np.ndarray, decay: float) -> np.ndarray:
    state = np.zeros((keys.shape[1], values.shape[1]), dtype=np.float64)
    for key_vec, value_vec in zip(keys, values):
        state = decay * state + np.outer(key_vec, value_vec)
    return state


def evaluate_recall(
    keys: np.ndarray,
    target_values: np.ndarray,
    state: np.ndarray,
    reconstruction_base: np.ndarray | None = None,
) -> dict[str, float]:
    retrieved = keys @ state
    reconstructed = retrieved if reconstruction_base is None else reconstruction_base + retrieved
    cosines = row_cosines(reconstructed, target_values)
    nearest = np.argmax(cosine_score_matrix(reconstructed, target_values), axis=-1)
    return {
        "mean_cosine_recall": float(np.mean(cosines)),
        "exact_match_accuracy": float(np.mean(nearest == np.arange(target_values.shape[0]))),
        "mean_output_norm": float(np.mean(np.linalg.norm(reconstructed, axis=-1))),
    }


def sample_trial_sequence(
    rng: np.random.Generator,
    mixing_matrix: np.ndarray,
    temporal_correlation: float,
    pattern_count: int,
    noise_scale: float,
) -> np.ndarray:
    samples = generate_correlated_sequences(
        rng,
        mixing_matrix,
        temporal_correlation,
        1,
        pattern_count + 1,
        noise_scale,
    )
    return samples[0]


def value_key(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return format(float(value), "g")


def filter_records(records: list[dict[str, Any]], **filters: object) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    for record in records:
        hit = True
        for key, expected in filters.items():
            value = record.get(key)
            if isinstance(expected, float):
                if value is None or not np.isclose(float(value), expected):
                    hit = False
                    break
            else:
                if value != expected:
                    hit = False
                    break
        if hit:
            matched.append(record)
    return matched


def metric_values(records: list[dict[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for record in records:
        if metric in record and record[metric] is not None:
            values.append(float(record[metric]))
    return values


def summarize_capacity_cells(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    capacity_summary: dict[str, Any] = {}
    reconstruction_summary: dict[str, Any] = {}
    paired_stats: dict[str, Any] = {}
    for head_dim in HEAD_DIMS:
        head_key = value_key(head_dim)
        capacity_summary[head_key] = {}
        reconstruction_summary[head_key] = {}
        paired_stats[head_key] = {}
        for decay in DECAYS:
            decay_key = value_key(decay)
            capacity_summary[head_key][decay_key] = {}
            reconstruction_summary[head_key][decay_key] = {}
            paired_stats[head_key][decay_key] = {}
            for temporal_correlation in TEMPORAL_CORRELATIONS:
                correlation_key = value_key(temporal_correlation)
                raw_capacity_candidates: list[int] = []
                correction_memory_capacity_candidates: list[int] = []
                prediction_only_capacity_candidates: list[int] = []
                correction_reconstruction_capacity_candidates: list[int] = []
                reconstruction_summary[head_key][decay_key][correlation_key] = {}
                paired_stats[head_key][decay_key][correlation_key] = {}
                for pattern_count in PATTERN_COUNTS:
                    pattern_key = value_key(pattern_count)
                    subset = filter_records(
                        records,
                        head_dim=head_dim,
                        decay=decay,
                        temporal_correlation=temporal_correlation,
                        pattern_count=pattern_count,
                    )
                    raw_values = metric_values(subset, "raw_mean_cosine_recall")
                    correction_memory_values = metric_values(subset, "correction_residual_mean_cosine_recall")
                    prediction_only_values = metric_values(subset, "prediction_only_mean_cosine_recall")
                    correction_reconstruction_values = metric_values(subset, "correction_reconstruction_mean_cosine_recall")
                    if raw_values and float(np.mean(raw_values)) > COSINE_THRESHOLD:
                        raw_capacity_candidates.append(int(pattern_count))
                    if correction_memory_values and float(np.mean(correction_memory_values)) > COSINE_THRESHOLD:
                        correction_memory_capacity_candidates.append(int(pattern_count))
                    if prediction_only_values and float(np.mean(prediction_only_values)) > COSINE_THRESHOLD:
                        prediction_only_capacity_candidates.append(int(pattern_count))
                    if correction_reconstruction_values and float(np.mean(correction_reconstruction_values)) > COSINE_THRESHOLD:
                        correction_reconstruction_capacity_candidates.append(int(pattern_count))
                    reconstruction_summary[head_key][decay_key][correlation_key][pattern_key] = {
                        "raw": mean_confidence_interval(raw_values, bounds=(-1.0, 1.0)),
                        "correction_field_memory": mean_confidence_interval(correction_memory_values, bounds=(-1.0, 1.0)),
                        "prediction_only": mean_confidence_interval(prediction_only_values, bounds=(-1.0, 1.0)),
                        "correction_field_reconstruction": mean_confidence_interval(correction_reconstruction_values, bounds=(-1.0, 1.0)),
                    }
                    paired_stats[head_key][decay_key][correlation_key][pattern_key] = {
                        "memory_gain_vs_raw": paired_difference_stats(raw_values, correction_memory_values, SEED + head_dim + pattern_count),
                        "reconstruction_gain_vs_prediction": paired_difference_stats(
                            prediction_only_values,
                            correction_reconstruction_values,
                            SEED + head_dim + pattern_count + 11,
                        ),
                    }
                raw_capacity = max(raw_capacity_candidates, default=0)
                correction_memory_capacity = max(correction_memory_capacity_candidates, default=0)
                prediction_only_capacity = max(prediction_only_capacity_candidates, default=0)
                correction_reconstruction_capacity = max(correction_reconstruction_capacity_candidates, default=0)
                capacity_summary[head_key][decay_key][correlation_key] = {
                    "raw_effective_capacity": int(raw_capacity),
                    "correction_field_memory_effective_capacity": int(correction_memory_capacity),
                    "memory_substrate_capacity_delta": int(correction_memory_capacity - raw_capacity),
                    "prediction_only_effective_capacity": int(prediction_only_capacity),
                    "correction_field_reconstruction_effective_capacity": int(correction_reconstruction_capacity),
                    "reconstruction_capacity_gain_over_prediction": int(
                        correction_reconstruction_capacity - prediction_only_capacity
                    ),
                }
    return capacity_summary, reconstruction_summary, paired_stats


def summarize_memory_substrate_capacity_delta_by_correlation(capacity_summary: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for correlation in TEMPORAL_CORRELATIONS:
        correlation_key = value_key(correlation)
        memory_gains: list[float] = []
        reconstruction_gains: list[float] = []
        for head_dim in HEAD_DIMS:
            head_key = value_key(head_dim)
            for decay in DECAYS:
                decay_key = value_key(decay)
                entry = capacity_summary[head_key][decay_key][correlation_key]
                memory_gains.append(float(entry["memory_substrate_capacity_delta"]))
                reconstruction_gains.append(float(entry["reconstruction_capacity_gain_over_prediction"]))
        summary[correlation_key] = {
            "memory_substrate_capacity_delta": mean_confidence_interval(memory_gains),
            "reconstruction_capacity_gain_over_prediction": mean_confidence_interval(reconstruction_gains),
        }
    return summary


def summarize_residual_norm_ratios(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for correlation in TEMPORAL_CORRELATIONS:
        correlation_key = value_key(correlation)
        summary[correlation_key] = {}
        for head_dim in HEAD_DIMS:
            head_key = value_key(head_dim)
            subset = filter_records(records, head_dim=head_dim, temporal_correlation=correlation)
            ratios = metric_values(subset, "residual_norm_ratio")
            summary[correlation_key][head_key] = mean_confidence_interval(ratios, bounds=(0.0, 10.0))
    return summary


def summarize_observed_prediction_quality(training_records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for correlation in TEMPORAL_CORRELATIONS:
        correlation_key = value_key(correlation)
        subset = filter_records(training_records, temporal_correlation=correlation)
        values = metric_values(subset, "observed_prediction_quality")
        summary[correlation_key] = mean_confidence_interval(values, bounds=(-1.0, 1.0))
    return summary


def summarize_training_convergence(training_records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for correlation in TEMPORAL_CORRELATIONS:
        correlation_key = value_key(correlation)
        subset = filter_records(training_records, temporal_correlation=correlation)
        initial_losses = metric_values(subset, "initial_loss")
        final_losses = metric_values(subset, "final_loss")
        summary[correlation_key] = {
            "initial_loss": mean_confidence_interval(initial_losses),
            "final_loss": mean_confidence_interval(final_losses),
            "mean_loss_reduction": (
                float(np.mean(initial_losses) - np.mean(final_losses))
                if initial_losses and final_losses
                else None
            ),
        }
    return summary


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
            raw_matrix = np.zeros((len(TEMPORAL_CORRELATIONS), len(PATTERN_COUNTS)), dtype=np.float64)
            recon_matrix = np.zeros_like(raw_matrix)
            for r_idx, correlation in enumerate(TEMPORAL_CORRELATIONS):
                for p_idx, pattern_count in enumerate(PATTERN_COUNTS):
                    subset = filter_records(
                        records,
                        head_dim=head_dim,
                        decay=decay,
                        temporal_correlation=correlation,
                        pattern_count=pattern_count,
                    )
                    raw_matrix[r_idx, p_idx] = (
                        float(np.mean(metric_values(subset, "raw_mean_cosine_recall")))
                        if subset
                        else np.nan
                    )
                    recon_matrix[r_idx, p_idx] = (
                        float(np.mean(metric_values(subset, "correction_reconstruction_mean_cosine_recall")))
                        if subset
                        else np.nan
                    )
            for col_idx, (title, matrix) in enumerate((("raw values", raw_matrix), ("corrected reconstruction", recon_matrix))):
                ax = axes[row_idx, col_idx]
                image = ax.imshow(matrix, aspect="auto", origin="lower", vmin=-0.2, vmax=1.0, cmap="viridis")
                ax.set_xticks(range(len(PATTERN_COUNTS)))
                ax.set_xticklabels([str(value) for value in PATTERN_COUNTS])
                ax.set_yticks(range(len(TEMPORAL_CORRELATIONS)))
                ax.set_yticklabels([format(float(value), "g") for value in TEMPORAL_CORRELATIONS])
                ax.set_title(f"{title} | d={head_dim} decay={format(float(decay), 'g')}")
                ax.set_xlabel("pattern count")
                if col_idx == 0:
                    ax.set_ylabel("temporal correlation")
            row_idx += 1
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85, label="mean cosine recall")
    plt.savefig(output_path)
    plt.close(fig)


def build_summary_figure(
    capacity_summary: dict[str, Any],
    prediction_quality_summary: dict[str, Any],
    residual_norm_summary: dict[str, Any],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    ax = axes[0]
    for head_dim in HEAD_DIMS:
        head_key = value_key(head_dim)
        decay_key = value_key(DECAYS[min(len(DECAYS) - 1, 0)])
        gains = []
        for correlation in TEMPORAL_CORRELATIONS:
            entry = capacity_summary[head_key][decay_key][value_key(correlation)]
            gains.append(float(entry["memory_substrate_capacity_delta"]))
        ax.plot(TEMPORAL_CORRELATIONS, gains, marker="o", label=f"d={head_dim}")
    ax.axhline(0.0, color="#111827", linewidth=1)
    ax.set_xlabel("temporal correlation")
    ax.set_ylabel("memory capacity gain")
    ax.set_title("memory gain vs raw under trained predictions")
    ax.legend(fontsize=8)

    ax = axes[1]
    xs = TEMPORAL_CORRELATIONS
    ys = [
        prediction_quality_summary[value_key(correlation)]["mean"]
        if prediction_quality_summary[value_key(correlation)]["mean"] is not None
        else np.nan
        for correlation in TEMPORAL_CORRELATIONS
    ]
    ax.plot(xs, ys, marker="o", color="#0f766e")
    ax.set_xlabel("temporal correlation")
    ax.set_ylabel("observed prediction quality")
    ax.set_title("prediction head held-out r^2")

    ax = axes[2]
    for head_dim in HEAD_DIMS:
        ratios = []
        for correlation in TEMPORAL_CORRELATIONS:
            entry = residual_norm_summary[value_key(correlation)][value_key(head_dim)]
            ratios.append(entry["mean"] if entry["mean"] is not None else np.nan)
        ax.plot(TEMPORAL_CORRELATIONS, ratios, marker="o", label=f"d={head_dim}")
    ax.axhline(1.0, color="#111827", linewidth=1)
    ax.set_xlabel("temporal correlation")
    ax.set_ylabel("residual norm / value norm")
    ax.set_title("compression of residual vs raw values")
    ax.legend(fontsize=8)

    plt.savefig(output_path)
    plt.close(fig)


def main() -> None:
    apply_plot_style()

    started = utc_now_iso()
    t0 = time.time()
    rng = build_rng(SEED)

    training_records: list[dict[str, Any]] = []
    trained_heads: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}
    mixing_matrices: dict[int, np.ndarray] = {}
    for head_dim in HEAD_DIMS:
        mixing_rng = child_rng(rng)
        mixing_matrices[int(head_dim)] = random_orthogonal(mixing_rng, int(head_dim))
        for temporal_correlation in TEMPORAL_CORRELATIONS:
            train_rng = child_rng(rng)
            w_up, w_down, observed_quality, final_loss, loss_trajectory = train_prediction_head(
                train_rng,
                mixing_matrices[int(head_dim)],
                float(temporal_correlation),
                int(head_dim),
                int(TRAIN_RANK),
                int(TRAIN_SEQUENCES),
                int(TRAIN_SEQUENCE_LENGTH),
                float(NOISE_SCALE),
                int(TRAIN_ITERS),
                float(TRAIN_LR),
                int(BATCH_SIZE),
            )
            trained_heads[(int(head_dim), float(temporal_correlation))] = (w_up, w_down)
            initial_loss = float(loss_trajectory[0]) if loss_trajectory else float("nan")
            training_records.append(
                {
                    "head_dim": int(head_dim),
                    "temporal_correlation": float(temporal_correlation),
                    "initial_loss": initial_loss,
                    "final_loss": float(final_loss),
                    "loss_reduction": float(initial_loss - final_loss),
                    "observed_prediction_quality": float(observed_quality),
                    "iterations": int(TRAIN_ITERS),
                    "rank": int(TRAIN_RANK),
                }
            )

    capacity_records: list[dict[str, Any]] = []
    for head_dim in HEAD_DIMS:
        mixing_matrix = mixing_matrices[int(head_dim)]
        for decay in DECAYS:
            for temporal_correlation in TEMPORAL_CORRELATIONS:
                w_up, w_down = trained_heads[(int(head_dim), float(temporal_correlation))]
                for pattern_count in PATTERN_COUNTS:
                    for trial_id in range(TRIALS):
                        trial_rng = child_rng(rng)
                        sequence = sample_trial_sequence(
                            trial_rng,
                            mixing_matrix,
                            float(temporal_correlation),
                            int(pattern_count),
                            float(NOISE_SCALE),
                        )
                        previous_values = sequence[:-1]
                        target_values = sequence[1:]
                        keys = normalize_rows(trial_rng.standard_normal((int(pattern_count), int(head_dim))))
                        prediction = (previous_values @ w_up) @ w_down
                        residual = target_values - prediction
                        raw_state = build_state(keys, target_values, float(decay))
                        correction_state = build_state(keys, residual, float(decay))
                        prediction_only_metrics = evaluate_recall(
                            keys,
                            target_values,
                            np.zeros_like(correction_state),
                            reconstruction_base=prediction,
                        )
                        raw_metrics = evaluate_recall(keys, target_values, raw_state)
                        correction_memory_metrics = evaluate_recall(keys, residual, correction_state)
                        correction_reconstruction_metrics = evaluate_recall(
                            keys,
                            target_values,
                            correction_state,
                            reconstruction_base=prediction,
                        )
                        value_norm = float(np.mean(np.linalg.norm(target_values, axis=-1)))
                        residual_norm = float(np.mean(np.linalg.norm(residual, axis=-1)))
                        residual_norm_ratio = float(residual_norm / max(value_norm, EPS))
                        value_energy = float(np.mean(np.sum(target_values ** 2, axis=-1)))
                        residual_energy = float(np.mean(np.sum(residual ** 2, axis=-1)))
                        capacity_records.append(
                            {
                                "scenario_id": (
                                    f"d{head_dim}_decay{format(float(decay), 'g')}_r{format(float(temporal_correlation), 'g')}"
                                    f"_n{pattern_count}_t{trial_id}"
                                ),
                                "trial_id": int(trial_id),
                                "head_dim": int(head_dim),
                                "pattern_count": int(pattern_count),
                                "temporal_correlation": float(temporal_correlation),
                                "decay": float(decay),
                                "prediction_only_mean_cosine_recall": prediction_only_metrics["mean_cosine_recall"],
                                "prediction_only_exact_match_accuracy": prediction_only_metrics["exact_match_accuracy"],
                                "raw_mean_cosine_recall": raw_metrics["mean_cosine_recall"],
                                "raw_exact_match_accuracy": raw_metrics["exact_match_accuracy"],
                                "correction_residual_mean_cosine_recall": correction_memory_metrics["mean_cosine_recall"],
                                "correction_residual_exact_match_accuracy": correction_memory_metrics["exact_match_accuracy"],
                                "correction_reconstruction_mean_cosine_recall": correction_reconstruction_metrics["mean_cosine_recall"],
                                "correction_reconstruction_exact_match_accuracy": correction_reconstruction_metrics["exact_match_accuracy"],
                                "mean_value_norm": value_norm,
                                "mean_residual_norm": residual_norm,
                                "residual_norm_ratio": residual_norm_ratio,
                                "mean_value_energy": value_energy,
                                "mean_residual_energy": residual_energy,
                                "observed_residual_variance_fraction": float(residual_energy / max(value_energy, EPS)),
                                "raw_state_fro_norm": float(np.linalg.norm(raw_state)),
                                "correction_state_fro_norm": float(np.linalg.norm(correction_state)),
                            }
                        )

    capacity_summary, reconstruction_summary, paired_stats = summarize_capacity_cells(capacity_records)
    memory_capacity_by_correlation = summarize_memory_substrate_capacity_delta_by_correlation(capacity_summary)
    prediction_quality_summary = summarize_observed_prediction_quality(training_records)
    training_summary = summarize_training_convergence(training_records)
    residual_norm_summary = summarize_residual_norm_ratios(capacity_records)

    output_dir = output_dir_for(SCRIPT_PATH)
    heatmap_path = output_dir / "correction_field_trained_prediction_heatmaps.png"
    summary_path = output_dir / "correction_field_trained_prediction_summary.png"
    metrics_path = output_dir / "correction_field_trained_prediction_metrics.json"

    build_heatmaps_figure(capacity_records, heatmap_path)
    build_summary_figure(capacity_summary, prediction_quality_summary, residual_norm_summary, summary_path)

    warnings: list[str] = []
    for record in training_records:
        if record["loss_reduction"] <= 0:
            warnings.append(
                f"training did not reduce mse for head_dim={record['head_dim']} "
                f"temporal_correlation={record['temporal_correlation']}"
            )
    for correlation in TEMPORAL_CORRELATIONS:
        entry = prediction_quality_summary[value_key(correlation)]
        if entry["mean"] is not None and float(entry["mean"]) < 0.0:
            warnings.append(f"observed prediction quality below zero at temporal_correlation={correlation}")
    warnings.append(
        "synthetic hidden states are a stand-in for god_run_v2 activations; run replaced with real checkpoint when pod access is available."
    )

    finished = utc_now_iso()
    duration = time.time() - t0

    all_records = [
        {**entry, "sweep": "training"} for entry in training_records
    ] + [
        {**entry, "sweep": "capacity"} for entry in capacity_records
    ]

    record = build_run_record(
        simulation_name="correction_field_trained_prediction",
        script_path=SCRIPT_PATH,
        started_at_utc=started,
        finished_at_utc=finished,
        duration_sec=duration,
        parameters={
            "head_dims": HEAD_DIMS,
            "pattern_counts": PATTERN_COUNTS,
            "temporal_correlations": TEMPORAL_CORRELATIONS,
            "decays": DECAYS,
            "trials": TRIALS,
            "train_sequences": TRAIN_SEQUENCES,
            "train_sequence_length": TRAIN_SEQUENCE_LENGTH,
            "train_iterations": TRAIN_ITERS,
            "train_learning_rate": TRAIN_LR,
            "train_rank": TRAIN_RANK,
            "batch_size": BATCH_SIZE,
            "noise_scale": NOISE_SCALE,
            "cosine_threshold": COSINE_THRESHOLD,
            "held_out_sequences": HELD_OUT_SEQUENCES,
        },
        seed_numpy=SEED,
        n_trials=len(all_records),
        summary={
            "memory_substrate_capacity_delta_by_temporal_correlation": memory_capacity_by_correlation,
            "reconstruction_capacity_gain": reconstruction_summary,
            "observed_prediction_quality_by_correlation": prediction_quality_summary,
            "training_convergence_by_correlation": training_summary,
            "residual_norm_ratio_by_head_dim": residual_norm_summary,
            "capacity_by_correlation_and_decay": capacity_summary,
        },
        statistics={
            "paired_difference_by_cell": paired_stats,
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

    qualities = [prediction_quality_summary[value_key(correlation)]["mean"] for correlation in TEMPORAL_CORRELATIONS]
    print(
        "correction_field_trained_prediction "
        f"prediction_quality_by_correlation={['{:g}'.format(float(value)) if value is not None else 'nan' for value in qualities]} "
        f"duration_sec={duration:.1f}"
    )
    print(f"wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
