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
    require_positive,
    require_positive_list,
    require_unit_interval,
    utc_now_iso,
    write_json,
)

SCRIPT_PATH = Path(__file__).resolve()
SEED = env_int("MRHS_SEED", 42)
STREAM_LENGTHS = env_list("MRHS_STREAM_LENGTHS", int, [32, 64, 128, 256, 512])
HEAD_DIMS = env_list("MRHS_HEAD_DIMS", int, [32, 64, 128])
TRIALS = env_int("MRHS_TRIALS", 16)
TOTAL_HEADS = env_int("MRHS_TOTAL_HEADS", 16)
FAST_HEADS = env_int("MRHS_FAST_HEADS", 8)
MEDIUM_HEADS = env_int("MRHS_MEDIUM_HEADS", 6)
SLOW_HEADS = env_int("MRHS_SLOW_HEADS", 2)
UNIFORM_DECAY = env_float("MRHS_UNIFORM_DECAY", 0.9)
FAST_DECAY = env_float("MRHS_FAST_DECAY", 0.5)
MEDIUM_DECAY = env_float("MRHS_MEDIUM_DECAY", 0.95)
SLOW_DECAY = env_float("MRHS_SLOW_DECAY", 0.999)
MEDIUM_SURPRISE_TAU = env_float("MRHS_MEDIUM_SURPRISE_TAU", 0.1)
SLOW_SURPRISE_TAU = env_float("MRHS_SLOW_SURPRISE_TAU", 0.5)
IMMEDIATE_WINDOW = env_int("MRHS_IMMEDIATE_WINDOW", 4)
RECENT_WINDOW = env_int("MRHS_RECENT_WINDOW", 32)
RARE_WINDOW = env_int("MRHS_RARE_WINDOW", 200)
IMMEDIATE_CLASS_FRACTION = env_float("MRHS_IMMEDIATE_FRACTION", 0.6)
RECENT_CLASS_FRACTION = env_float("MRHS_RECENT_FRACTION", 0.3)
RARE_CLASS_FRACTION = env_float("MRHS_RARE_FRACTION", 0.1)
QUERIES_PER_CLASS = env_int("MRHS_QUERIES_PER_CLASS", 8)
COSINE_THRESHOLD = env_float("MRHS_COSINE_THRESHOLD", 0.25)
EPS = 1e-8

require_positive_list("MRHS_STREAM_LENGTHS", STREAM_LENGTHS)
require_positive_list("MRHS_HEAD_DIMS", HEAD_DIMS)
require_positive("MRHS_TRIALS", TRIALS)
require_positive("MRHS_TOTAL_HEADS", TOTAL_HEADS)
require_positive("MRHS_FAST_HEADS", FAST_HEADS)
require_positive("MRHS_MEDIUM_HEADS", MEDIUM_HEADS)
require_positive("MRHS_SLOW_HEADS", SLOW_HEADS)
require_unit_interval("MRHS_UNIFORM_DECAY", UNIFORM_DECAY)
require_unit_interval("MRHS_FAST_DECAY", FAST_DECAY)
require_unit_interval("MRHS_MEDIUM_DECAY", MEDIUM_DECAY)
require_unit_interval("MRHS_SLOW_DECAY", SLOW_DECAY)
require_unit_interval("MRHS_MEDIUM_SURPRISE_TAU", MEDIUM_SURPRISE_TAU, allow_zero=True)
require_unit_interval("MRHS_SLOW_SURPRISE_TAU", SLOW_SURPRISE_TAU, allow_zero=True)
require_positive("MRHS_IMMEDIATE_WINDOW", IMMEDIATE_WINDOW)
require_positive("MRHS_RECENT_WINDOW", RECENT_WINDOW)
require_positive("MRHS_RARE_WINDOW", RARE_WINDOW)
require_unit_interval("MRHS_IMMEDIATE_FRACTION", IMMEDIATE_CLASS_FRACTION, allow_zero=True)
require_unit_interval("MRHS_RECENT_FRACTION", RECENT_CLASS_FRACTION, allow_zero=True)
require_unit_interval("MRHS_RARE_FRACTION", RARE_CLASS_FRACTION, allow_zero=True)
require_positive("MRHS_QUERIES_PER_CLASS", QUERIES_PER_CLASS)
require_unit_interval("MRHS_COSINE_THRESHOLD", COSINE_THRESHOLD, allow_zero=True)

if abs((IMMEDIATE_CLASS_FRACTION + RECENT_CLASS_FRACTION + RARE_CLASS_FRACTION) - 1.0) > 1e-6:
    raise ValueError("MRHS class fractions must sum to 1.0")

if FAST_HEADS + MEDIUM_HEADS + SLOW_HEADS != TOTAL_HEADS:
    raise ValueError("MRHS_FAST_HEADS + MRHS_MEDIUM_HEADS + MRHS_SLOW_HEADS must equal MRHS_TOTAL_HEADS")

CLASS_NAMES = ("immediate", "recent", "rare")
CONFIG_NAMES = ("uniform", "split")
CLASS_SEED_OFFSETS = {"immediate": 1, "recent": 2, "rare": 3}


def normalize_rows(x: np.ndarray) -> np.ndarray:
    matrix = np.asarray(x, dtype=np.float64)
    if matrix.ndim == 1:
        norm = np.linalg.norm(matrix)
        return matrix / max(norm, EPS)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.clip(norms, EPS, None)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_vec = np.asarray(a, dtype=np.float64)
    b_vec = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a_vec) * np.linalg.norm(b_vec)
    if denom <= EPS:
        return 0.0
    return float(np.dot(a_vec, b_vec) / denom)


def build_stream(rng: np.random.Generator, stream_length: int, head_dim: int) -> dict[str, np.ndarray]:
    rare_count = max(1, int(round(stream_length * RARE_CLASS_FRACTION)))
    recent_count = max(1, int(round(stream_length * RECENT_CLASS_FRACTION)))
    immediate_count = stream_length - rare_count - recent_count
    if immediate_count < 1:
        immediate_count = 1
        recent_count = max(1, (stream_length - rare_count - immediate_count))
        rare_count = stream_length - immediate_count - recent_count
    classes = np.asarray(
        ["immediate"] * immediate_count + ["recent"] * recent_count + ["rare"] * rare_count,
        dtype=object,
    )
    permutation = rng.permutation(stream_length)
    classes = classes[permutation]
    keys = normalize_rows(rng.standard_normal((stream_length, head_dim)))
    values = normalize_rows(rng.standard_normal((stream_length, head_dim)))
    surprise = np.zeros(stream_length, dtype=np.float64)
    for idx in range(stream_length):
        label = classes[idx]
        if label == "immediate":
            surprise[idx] = float(rng.uniform(0.0, 0.09))
        elif label == "recent":
            surprise[idx] = float(rng.uniform(0.15, 0.4))
        else:
            surprise[idx] = float(rng.uniform(0.6, 0.99))
    return {
        "keys": keys,
        "values": values,
        "classes": classes,
        "surprise": surprise,
    }


def build_head_configuration(
    config_name: str,
    head_dim: int,
    num_fast: int | None = None,
    num_medium: int | None = None,
    num_slow: int | None = None,
    fast_decay: float | None = None,
    medium_decay: float | None = None,
    slow_decay: float | None = None,
    medium_surprise_tau: float | None = None,
    slow_surprise_tau: float | None = None,
    uniform_decay: float | None = None,
    total_heads: int | None = None,
) -> list[dict[str, Any]]:
    resolved_fast_decay = float(FAST_DECAY if fast_decay is None else fast_decay)
    resolved_medium_decay = float(MEDIUM_DECAY if medium_decay is None else medium_decay)
    resolved_slow_decay = float(SLOW_DECAY if slow_decay is None else slow_decay)
    resolved_medium_surprise_tau = float(MEDIUM_SURPRISE_TAU if medium_surprise_tau is None else medium_surprise_tau)
    resolved_slow_surprise_tau = float(SLOW_SURPRISE_TAU if slow_surprise_tau is None else slow_surprise_tau)
    resolved_uniform_decay = float(UNIFORM_DECAY if uniform_decay is None else uniform_decay)
    resolved_total_heads = int(TOTAL_HEADS if total_heads is None else total_heads)
    if config_name == "uniform":
        return [
            {
                "group": "uniform",
                "decay": resolved_uniform_decay,
                "surprise_tau": 0.0,
                "head_dim": int(head_dim),
            }
            for _ in range(resolved_total_heads)
        ]
    if config_name == "split":
        resolved_num_fast = int(FAST_HEADS if num_fast is None else num_fast)
        resolved_num_medium = int(MEDIUM_HEADS if num_medium is None else num_medium)
        resolved_num_slow = int(SLOW_HEADS if num_slow is None else num_slow)
        if resolved_num_fast + resolved_num_medium + resolved_num_slow <= 0:
            raise ValueError("split configuration requires at least one head")
        heads: list[dict[str, Any]] = []
        for _ in range(resolved_num_fast):
            heads.append(
                {
                    "group": "fast",
                    "decay": resolved_fast_decay,
                    "surprise_tau": 0.0,
                    "head_dim": int(head_dim),
                }
            )
        for _ in range(resolved_num_medium):
            heads.append(
                {
                    "group": "medium",
                    "decay": resolved_medium_decay,
                    "surprise_tau": resolved_medium_surprise_tau,
                    "head_dim": int(head_dim),
                }
            )
        for _ in range(resolved_num_slow):
            heads.append(
                {
                    "group": "slow",
                    "decay": resolved_slow_decay,
                    "surprise_tau": resolved_slow_surprise_tau,
                    "head_dim": int(head_dim),
                }
            )
        return heads
    raise ValueError(f"unknown configuration: {config_name}")


def should_write(surprise_value: float, surprise_tau: float) -> bool:
    if surprise_tau <= 0.0:
        return True
    return float(surprise_value) > float(surprise_tau)


def run_head_state(
    head_config: dict[str, Any],
    keys: np.ndarray,
    values: np.ndarray,
    surprise: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    decay = float(head_config["decay"])
    surprise_tau = float(head_config["surprise_tau"])
    head_dim = int(head_config["head_dim"])
    state = np.zeros((head_dim, head_dim), dtype=np.float64)
    write_flags = np.zeros(keys.shape[0], dtype=bool)
    for idx in range(keys.shape[0]):
        write = should_write(surprise[idx], surprise_tau)
        write_flags[idx] = bool(write)
        state = decay * state
        if write:
            state = state + np.outer(keys[idx], values[idx])
    return state, write_flags


def query_heads(
    head_configs: list[dict[str, Any]],
    head_states: list[np.ndarray],
    query_key: np.ndarray,
) -> np.ndarray:
    outputs = [query_key @ state for _, state in zip(head_configs, head_states)]
    stacked = np.stack(outputs, axis=0)
    return np.mean(stacked, axis=0)


def select_query_targets(
    classes: np.ndarray,
    rng: np.random.Generator,
    stream_length: int,
) -> dict[str, np.ndarray]:
    targets: dict[str, np.ndarray] = {}
    final_index = stream_length - 1
    for class_name, window in (
        ("immediate", IMMEDIATE_WINDOW),
        ("recent", RECENT_WINDOW),
        ("rare", RARE_WINDOW),
    ):
        target_low = max(0, final_index - window + 1)
        target_high = final_index + 1
        candidate_indices = np.flatnonzero(classes == class_name)
        in_window = candidate_indices[(candidate_indices >= target_low) & (candidate_indices < target_high)]
        if in_window.size == 0:
            targets[class_name] = np.zeros(0, dtype=int)
            continue
        selection_size = min(QUERIES_PER_CLASS, in_window.size)
        chosen = rng.choice(in_window, size=selection_size, replace=False)
        targets[class_name] = np.asarray(chosen, dtype=int)
    return targets


def evaluate_recall_for_class(
    head_configs: list[dict[str, Any]],
    head_states: list[np.ndarray],
    stream: dict[str, np.ndarray],
    target_indices: np.ndarray,
) -> dict[str, float]:
    if target_indices.size == 0:
        return {"mean_cosine": float("nan"), "top1_accuracy": float("nan"), "count": 0}
    keys = stream["keys"]
    values = stream["values"]
    cosines = []
    top1_flags = []
    for idx in target_indices:
        query_key = keys[int(idx)]
        readout = query_heads(head_configs, head_states, query_key)
        cosines_by_value = np.array(
            [cosine_similarity(readout, values[j]) for j in range(values.shape[0])],
            dtype=np.float64,
        )
        cosines.append(cosines_by_value[int(idx)])
        top1_flags.append(int(np.argmax(cosines_by_value) == int(idx)))
    return {
        "mean_cosine": float(np.mean(cosines)),
        "top1_accuracy": float(np.mean(top1_flags)),
        "count": int(target_indices.size),
    }


def run_single_trial(
    rng: np.random.Generator,
    stream_length: int,
    head_dim: int,
) -> dict[str, Any]:
    stream_rng = child_rng(rng)
    stream = build_stream(stream_rng, stream_length, head_dim)
    query_rng = child_rng(rng)
    targets = select_query_targets(stream["classes"], query_rng, stream_length)
    results: dict[str, Any] = {}
    for config_name in CONFIG_NAMES:
        head_configs = build_head_configuration(config_name, head_dim)
        head_states: list[np.ndarray] = []
        write_fractions_by_group: dict[str, list[float]] = {}
        for config in head_configs:
            state, write_flags = run_head_state(
                config,
                stream["keys"],
                stream["values"],
                stream["surprise"],
            )
            head_states.append(state)
            group_name = str(config["group"])
            write_fractions_by_group.setdefault(group_name, []).append(
                float(np.mean(write_flags.astype(np.float64)))
            )
        per_class_results: dict[str, dict[str, float]] = {}
        for class_name in CLASS_NAMES:
            per_class_results[class_name] = evaluate_recall_for_class(
                head_configs,
                head_states,
                stream,
                targets[class_name],
            )
        aggregated_write_fraction = {
            group_name: float(np.mean(values))
            for group_name, values in write_fractions_by_group.items()
        }
        results[config_name] = {
            "per_class": per_class_results,
            "write_fraction_by_group": aggregated_write_fraction,
        }
    return results


def collect_records(
    all_trials: list[dict[str, Any]],
    metric: str,
) -> dict[tuple[int, int, str, str], list[float]]:
    grouped: dict[tuple[int, int, str, str], list[float]] = {}
    for trial in all_trials:
        stream_length = int(trial["stream_length"])
        head_dim = int(trial["head_dim"])
        for config_name in CONFIG_NAMES:
            config_results = trial[config_name]
            for class_name in CLASS_NAMES:
                class_results = config_results["per_class"][class_name]
                metric_value = class_results.get(metric)
                if metric_value is None:
                    continue
                if isinstance(metric_value, float) and not np.isfinite(metric_value):
                    continue
                key = (stream_length, head_dim, config_name, class_name)
                grouped.setdefault(key, []).append(float(metric_value))
    return grouped


def summarize_recall_by_class_and_configuration(all_trials: list[dict[str, Any]]) -> dict[str, Any]:
    grouped_cosine = collect_records(all_trials, "mean_cosine")
    grouped_top1 = collect_records(all_trials, "top1_accuracy")
    summary: dict[str, Any] = {}
    for stream_length in STREAM_LENGTHS:
        stream_key = str(int(stream_length))
        summary[stream_key] = {}
        for head_dim in HEAD_DIMS:
            head_key = str(int(head_dim))
            summary[stream_key][head_key] = {}
            for config_name in CONFIG_NAMES:
                summary[stream_key][head_key][config_name] = {}
                for class_name in CLASS_NAMES:
                    key = (int(stream_length), int(head_dim), config_name, class_name)
                    cosine_values = grouped_cosine.get(key, [])
                    top1_values = grouped_top1.get(key, [])
                    summary[stream_key][head_key][config_name][class_name] = {
                        "cosine": mean_confidence_interval(cosine_values, bounds=(-1.0, 1.0)),
                        "top1": mean_confidence_interval(top1_values, bounds=(0.0, 1.0)),
                    }
    return summary


def summarize_capacity_gain(all_trials: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    grouped_cosine = collect_records(all_trials, "mean_cosine")
    gain_summary: dict[str, Any] = {}
    paired_stats: dict[str, Any] = {}
    for class_name in CLASS_NAMES:
        gain_summary[class_name] = {}
        paired_stats[class_name] = {}
        for stream_length in STREAM_LENGTHS:
            stream_key = str(int(stream_length))
            gain_summary[class_name][stream_key] = {}
            paired_stats[class_name][stream_key] = {}
            for head_dim in HEAD_DIMS:
                head_key = str(int(head_dim))
                uniform_values = grouped_cosine.get(
                    (int(stream_length), int(head_dim), "uniform", class_name),
                    [],
                )
                split_values = grouped_cosine.get(
                    (int(stream_length), int(head_dim), "split", class_name),
                    [],
                )
                if uniform_values and split_values:
                    gain = float(np.mean(split_values) - np.mean(uniform_values))
                    paired = paired_difference_stats(
                        uniform_values,
                        split_values,
                        int(SEED + int(stream_length) + int(head_dim) + CLASS_SEED_OFFSETS[class_name]),
                    )
                else:
                    gain = 0.0
                    paired = {
                        "n": 0,
                        "mean": None,
                        "mean_difference": None,
                        "p_value_permutation": None,
                        "p_value_ttest": None,
                        "effect_size_dz": None,
                    }
                gain_summary[class_name][stream_key][head_key] = {
                    "uniform_mean_cosine": float(np.mean(uniform_values)) if uniform_values else 0.0,
                    "split_mean_cosine": float(np.mean(split_values)) if split_values else 0.0,
                    "split_minus_uniform": float(gain),
                    "uniform_n": int(len(uniform_values)),
                    "split_n": int(len(split_values)),
                }
                paired_stats[class_name][stream_key][head_key] = paired
    return gain_summary, paired_stats


def summarize_crossover_pattern_count_by_class(all_trials: list[dict[str, Any]]) -> dict[str, Any]:
    grouped_cosine = collect_records(all_trials, "mean_cosine")
    summary: dict[str, Any] = {}
    for class_name in CLASS_NAMES:
        summary[class_name] = {}
        for head_dim in HEAD_DIMS:
            head_key = str(int(head_dim))
            crossover_length: int = -1
            beats_above = 0
            total_above = 0
            for stream_length in STREAM_LENGTHS:
                uniform_values = grouped_cosine.get(
                    (int(stream_length), int(head_dim), "uniform", class_name),
                    [],
                )
                split_values = grouped_cosine.get(
                    (int(stream_length), int(head_dim), "split", class_name),
                    [],
                )
                if not uniform_values or not split_values:
                    continue
                split_mean = float(np.mean(split_values))
                uniform_mean = float(np.mean(uniform_values))
                if split_mean > uniform_mean:
                    if crossover_length < 0:
                        crossover_length = int(stream_length)
                    beats_above += 1
                total_above += 1
            summary[class_name][head_key] = {
                "crossover_stream_length": int(crossover_length),
                "crossover_observed": int(1 if crossover_length > 0 else 0),
                "lengths_where_split_beats_uniform": int(beats_above),
                "lengths_compared": int(total_above),
            }
    return summary


def build_figure(all_trials: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    grouped_cosine = collect_records(all_trials, "mean_cosine")
    num_rows = len(CLASS_NAMES)
    num_cols = len(HEAD_DIMS)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4.2 * num_cols, 3.2 * num_rows), constrained_layout=True, squeeze=False)
    axes_grid = np.asarray(axes).reshape(num_rows, num_cols)
    for row_idx, class_name in enumerate(CLASS_NAMES):
        for col_idx, head_dim in enumerate(HEAD_DIMS):
            ax = axes_grid[row_idx, col_idx]
            for config_name in CONFIG_NAMES:
                means = []
                lows = []
                highs = []
                for stream_length in STREAM_LENGTHS:
                    values = grouped_cosine.get(
                        (int(stream_length), int(head_dim), config_name, class_name),
                        [],
                    )
                    ci = mean_confidence_interval(values, bounds=(-1.0, 1.0))
                    if ci["mean"] is None:
                        means.append(float("nan"))
                        lows.append(float("nan"))
                        highs.append(float("nan"))
                    else:
                        means.append(float(ci["mean"]))
                        lows.append(float(ci["ci95"]["low"]))
                        highs.append(float(ci["ci95"]["high"]))
                means_array = np.asarray(means, dtype=np.float64)
                lows_array = np.asarray(lows, dtype=np.float64)
                highs_array = np.asarray(highs, dtype=np.float64)
                ax.plot(STREAM_LENGTHS, means_array, marker="o", label=config_name)
                ax.fill_between(STREAM_LENGTHS, lows_array, highs_array, alpha=0.15)
            ax.axhline(COSINE_THRESHOLD, color="#991b1b", linestyle="--", alpha=0.45)
            ax.set_xlabel("stream length")
            ax.set_ylabel("mean cosine recall")
            ax.set_title(f"{class_name} | d_head={head_dim}")
            ax.set_ylim(-0.2, 1.05)
            ax.legend(fontsize=8)
    fig.suptitle("multi-resolution head split vs uniform, paired per trial")
    plt.savefig(output_path)
    plt.close(fig)


def main() -> None:
    apply_plot_style()
    started = utc_now_iso()
    t0 = time.time()
    rng = build_rng(SEED)

    trial_records: list[dict[str, Any]] = []
    for stream_length in STREAM_LENGTHS:
        for head_dim in HEAD_DIMS:
            for trial_id in range(TRIALS):
                cell_rng = child_rng(rng)
                trial_results = run_single_trial(cell_rng, int(stream_length), int(head_dim))
                record = {
                    "scenario_id": f"L{stream_length}_d{head_dim}_t{trial_id}",
                    "trial_id": int(trial_id),
                    "stream_length": int(stream_length),
                    "head_dim": int(head_dim),
                    "uniform": {
                        "per_class": {
                            class_name: {
                                "mean_cosine": float(trial_results["uniform"]["per_class"][class_name]["mean_cosine"])
                                if np.isfinite(trial_results["uniform"]["per_class"][class_name]["mean_cosine"])
                                else 0.0,
                                "top1_accuracy": float(trial_results["uniform"]["per_class"][class_name]["top1_accuracy"])
                                if np.isfinite(trial_results["uniform"]["per_class"][class_name]["top1_accuracy"])
                                else 0.0,
                                "count": int(trial_results["uniform"]["per_class"][class_name]["count"]),
                            }
                            for class_name in CLASS_NAMES
                        },
                        "write_fraction_by_group": {
                            group_name: float(value)
                            for group_name, value in trial_results["uniform"]["write_fraction_by_group"].items()
                        },
                    },
                    "split": {
                        "per_class": {
                            class_name: {
                                "mean_cosine": float(trial_results["split"]["per_class"][class_name]["mean_cosine"])
                                if np.isfinite(trial_results["split"]["per_class"][class_name]["mean_cosine"])
                                else 0.0,
                                "top1_accuracy": float(trial_results["split"]["per_class"][class_name]["top1_accuracy"])
                                if np.isfinite(trial_results["split"]["per_class"][class_name]["top1_accuracy"])
                                else 0.0,
                                "count": int(trial_results["split"]["per_class"][class_name]["count"]),
                            }
                            for class_name in CLASS_NAMES
                        },
                        "write_fraction_by_group": {
                            group_name: float(value)
                            for group_name, value in trial_results["split"]["write_fraction_by_group"].items()
                        },
                    },
                }
                trial_records.append(record)

    recall_summary = summarize_recall_by_class_and_configuration(trial_records)
    capacity_gain_summary, paired_stats = summarize_capacity_gain(trial_records)
    crossover_summary = summarize_crossover_pattern_count_by_class(trial_records)

    output_dir = output_dir_for(SCRIPT_PATH)
    figure_path = output_dir / "multi_resolution_head_split.png"
    metrics_path = output_dir / "multi_resolution_head_split_metrics.json"

    build_figure(trial_records, figure_path)

    warnings: list[str] = []
    for class_name in CLASS_NAMES:
        for head_dim in HEAD_DIMS:
            head_key = str(int(head_dim))
            entry = crossover_summary[class_name][head_key]
            if entry["lengths_compared"] == 0:
                warnings.append(
                    f"no paired comparisons available for class={class_name} head_dim={head_dim}"
                )
    warnings.append(
        "surprise values are synthetic uniform draws per class label; a trained model would generate its own surprise distribution"
    )
    warnings.append(
        "the stream's class labels drive both the surprise distribution and which tokens are eligible as query targets; the paired draw per (stream_length, head_dim, trial) keeps the comparison deterministic across configurations"
    )

    finished = utc_now_iso()
    duration = time.time() - t0

    record = build_run_record(
        simulation_name="multi_resolution_head_split",
        script_path=SCRIPT_PATH,
        started_at_utc=started,
        finished_at_utc=finished,
        duration_sec=duration,
        parameters={
            "stream_lengths": STREAM_LENGTHS,
            "head_dims": HEAD_DIMS,
            "trials": TRIALS,
            "total_heads": TOTAL_HEADS,
            "fast_heads": FAST_HEADS,
            "medium_heads": MEDIUM_HEADS,
            "slow_heads": SLOW_HEADS,
            "uniform_decay": UNIFORM_DECAY,
            "fast_decay": FAST_DECAY,
            "medium_decay": MEDIUM_DECAY,
            "slow_decay": SLOW_DECAY,
            "medium_surprise_tau": MEDIUM_SURPRISE_TAU,
            "slow_surprise_tau": SLOW_SURPRISE_TAU,
            "immediate_window": IMMEDIATE_WINDOW,
            "recent_window": RECENT_WINDOW,
            "rare_window": RARE_WINDOW,
            "immediate_class_fraction": IMMEDIATE_CLASS_FRACTION,
            "recent_class_fraction": RECENT_CLASS_FRACTION,
            "rare_class_fraction": RARE_CLASS_FRACTION,
            "queries_per_class": QUERIES_PER_CLASS,
            "cosine_threshold": COSINE_THRESHOLD,
        },
        seed_numpy=SEED,
        n_trials=len(trial_records),
        summary={
            "recall_by_class_and_configuration": recall_summary,
            "capacity_gain_split_over_uniform_by_class": capacity_gain_summary,
            "crossover_pattern_count_by_class": crossover_summary,
        },
        statistics={
            "paired_split_vs_uniform_by_class": paired_stats,
        },
        trials=trial_records,
        artifacts=[
            {"name": figure_path.name, "path": figure_path, "type": "plot"},
            {"name": metrics_path.name, "path": metrics_path, "type": "metrics"},
        ],
        warnings=warnings,
    )
    write_json(metrics_path, record)

    print(f"multi_resolution_head_split done in {duration:.1f}s, trials={len(trial_records)}")
    print(f"wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
