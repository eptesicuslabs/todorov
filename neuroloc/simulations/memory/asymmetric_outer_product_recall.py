from __future__ import annotations

import sys
import time
from pathlib import Path

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
    output_dir_for,
    mean_confidence_interval,
    paired_difference_stats,
    require_non_negative,
    require_non_negative_list,
    require_positive,
    require_positive_list,
    require_unit_interval,
    require_unit_interval_list,
    utc_now_iso,
    write_json,
)

SCRIPT_PATH = Path(__file__).resolve()
SEED = env_int("ASYM_SEED", 42)
HEAD_DIM = env_int("ASYM_HEAD_DIM", 64)
PATTERN_COUNTS = env_list("ASYM_PATTERN_COUNTS", int, [8, 16, 32, 48])
QUERY_NOISE_LEVELS = env_list("ASYM_QUERY_NOISE", float, [0.0, 0.1, 0.2])
DECAY_VALUES = env_list("ASYM_DECAYS", float, [0.4, 0.8, 0.95])
TRIALS = env_int("ASYM_TRIALS", 24)
WRITE_GATE = env_float("ASYM_WRITE_GATE", 1.0)
TOPK_FRACTION = env_float("ASYM_TOPK_FRACTION", 0.2)
TERNARY_ALPHA = env_float("ASYM_TERNARY_ALPHA", 1.0)

require_positive("ASYM_HEAD_DIM", HEAD_DIM)
require_positive_list("ASYM_PATTERN_COUNTS", PATTERN_COUNTS)
require_non_negative_list("ASYM_QUERY_NOISE", QUERY_NOISE_LEVELS)
require_unit_interval_list("ASYM_DECAYS", DECAY_VALUES, allow_zero=True)
require_positive("ASYM_TRIALS", TRIALS)
require_non_negative("ASYM_WRITE_GATE", WRITE_GATE)
require_unit_interval("ASYM_TOPK_FRACTION", TOPK_FRACTION)
require_positive("ASYM_TERNARY_ALPHA", TERNARY_ALPHA)

ENCODING_SPECS = [
    {"name": "identity_all", "label": "identity", "q": "identity", "k": "identity", "v": "identity"},
    {"name": "bounded_all", "label": "tanh", "q": "bounded", "k": "bounded", "v": "bounded"},
    {"name": "sign_all", "label": "sign", "q": "sign", "k": "sign", "v": "sign"},
    {"name": "ternary_global_all", "label": "global ternary", "q": "ternary_global", "k": "ternary_global", "v": "ternary_global"},
    {"name": "ternary_per_dim_all", "label": "per-dim ternary", "q": "ternary_per_dim", "k": "ternary_per_dim", "v": "ternary_per_dim"},
    {"name": "topk20_all", "label": "top-k all", "q": "topk", "k": "topk", "v": "topk"},
    {"name": "dense_key_topk20_value", "label": "dense key / top-k value", "q": "identity", "k": "identity", "v": "topk"},
]


def as_matrix(x):
    array = np.asarray(x, dtype=np.float64)
    if array.ndim == 1:
        return array[None, :], True
    return array, False


def normalize_rows(x):
    matrix, squeezed = as_matrix(x)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    normalized = matrix / np.clip(norms, 1e-8, None)
    if squeezed:
        return normalized[0]
    return normalized


def cosine_similarity(a, b):
    a_vec = np.asarray(a, dtype=np.float64)
    b_vec = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a_vec) * np.linalg.norm(b_vec)
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(a_vec, b_vec) / denom)


def fit_encoder(kind, matrix):
    data = np.asarray(matrix, dtype=np.float64)
    if kind in {"identity", "bounded", "sign", "topk"}:
        return {}
    if kind == "ternary_global":
        return {"threshold": float(TERNARY_ALPHA * np.mean(np.abs(data)))}
    if kind == "ternary_per_dim":
        return {"thresholds": TERNARY_ALPHA * np.mean(np.abs(data), axis=0, keepdims=True)}
    raise ValueError(f"unknown encoder kind: {kind}")


def apply_encoder(x, kind, params):
    matrix, squeezed = as_matrix(x)
    if kind == "identity":
        encoded = matrix.copy()
    elif kind == "bounded":
        encoded = np.tanh(matrix)
    elif kind == "sign":
        encoded = np.sign(matrix)
    elif kind == "ternary_global":
        threshold = float(params["threshold"])
        encoded = np.sign(matrix) * (np.abs(matrix) >= threshold)
    elif kind == "ternary_per_dim":
        thresholds = np.asarray(params["thresholds"], dtype=np.float64)
        encoded = np.sign(matrix) * (np.abs(matrix) >= thresholds)
    elif kind == "topk":
        k = max(1, int(round(matrix.shape[-1] * TOPK_FRACTION)))
        ranking = np.abs(matrix)
        topk_idx = np.argpartition(ranking, -k, axis=-1)[:, -k:]
        mask = np.zeros_like(matrix)
        row_idx = np.arange(matrix.shape[0])[:, None]
        mask[row_idx, topk_idx] = 1.0
        encoded = matrix * mask
    else:
        raise ValueError(f"unknown encoder kind: {kind}")
    if squeezed:
        return encoded[0]
    return encoded


def encode_trial(keys_raw, values_raw, spec):
    key_params = fit_encoder(spec["k"], keys_raw)
    value_params = fit_encoder(spec["v"], values_raw)
    query_params = fit_encoder(spec["q"], keys_raw)
    keys = normalize_rows(apply_encoder(keys_raw, spec["k"], key_params))
    values = apply_encoder(values_raw, spec["v"], value_params)
    return keys, values, key_params, value_params, query_params


def update_state(state, key_vec, value_vec, decay, write_gate, erasure):
    decayed = decay * state
    if erasure:
        overwrite = np.outer(key_vec, key_vec @ decayed)
        decayed = decayed - write_gate * overwrite
    return decayed + write_gate * np.outer(key_vec, value_vec)


def build_state(keys, values, decay, write_gate, erasure):
    state = np.zeros((keys.shape[1], values.shape[1]), dtype=np.float64)
    for key_vec, value_vec in zip(keys, values):
        state = update_state(state, key_vec, value_vec, decay, write_gate, erasure)
    return state


def query_state(state, query_vec):
    return query_vec @ state


def trial_record(keys_raw, values_raw, target_idx, noisy_query, pattern_count, query_noise, decay, erasure, spec, scenario_id):
    keys, values, key_params, _, query_params = encode_trial(keys_raw, values_raw, spec)
    state = build_state(keys, values, decay, WRITE_GATE, erasure)
    query = normalize_rows(apply_encoder(noisy_query, spec["q"], query_params))
    output = query_state(state, query)
    raw_cosines = [cosine_similarity(output, candidate) for candidate in values_raw]
    stored_cosines = [cosine_similarity(output, candidate) for candidate in values]
    raw_rank = int(np.argmax(raw_cosines))
    stored_rank = int(np.argmax(stored_cosines))
    sorted_raw = np.sort(np.asarray(raw_cosines))[::-1]
    margin = float(sorted_raw[0] - sorted_raw[1]) if sorted_raw.size > 1 else float(sorted_raw[0])
    return {
        "scenario_id": scenario_id,
        "encoding": spec["name"],
        "encoding_label": spec["label"],
        "pattern_count": int(pattern_count),
        "query_noise": float(query_noise),
        "decay": float(decay),
        "erasure": bool(erasure),
        "target_idx": target_idx,
        "raw_cosine": float(raw_cosines[target_idx]),
        "stored_cosine": float(stored_cosines[target_idx]),
        "raw_top1": float(raw_rank == target_idx),
        "stored_top1": float(stored_rank == target_idx),
        "raw_margin": margin,
        "query_norm": float(np.linalg.norm(query)),
        "output_norm": float(np.linalg.norm(output)),
        "state_fro_norm": float(np.linalg.norm(state)),
        "key_nonzero_rate": float(np.mean(keys != 0.0)),
        "value_nonzero_rate": float(np.mean(values != 0.0)),
        "query_nonzero_rate": float(np.mean(query != 0.0)),
        "key_threshold_mean": float(np.mean(key_params.get("thresholds", [key_params.get("threshold", 0.0)]))),
    }


def filter_metric(records, metric_name, **filters):
    values = []
    for record in records:
        matches = True
        for key, expected in filters.items():
            value = record[key]
            if isinstance(expected, float):
                if not np.isclose(float(value), expected):
                    matches = False
                    break
            else:
                if value != expected:
                    matches = False
                    break
        if matches:
            values.append(float(record[metric_name]))
    return values


def paired_metric(records, metric_name, left_encoding, right_encoding, **filters):
    left = {}
    right = {}
    for record in records:
        matches = True
        for key, expected in filters.items():
            value = record[key]
            if isinstance(expected, float):
                if not np.isclose(float(value), expected):
                    matches = False
                    break
            else:
                if value != expected:
                    matches = False
                    break
        if not matches:
            continue
        if record["encoding"] == left_encoding:
            left[record["scenario_id"]] = float(record[metric_name])
        elif record["encoding"] == right_encoding:
            right[record["scenario_id"]] = float(record[metric_name])
    scenario_ids = sorted(set(left) & set(right))
    return [left[item] for item in scenario_ids], [right[item] for item in scenario_ids]


def summarize_by_encoding(records, metric_name, **filters):
    summary = {}
    for spec in ENCODING_SPECS:
        values = filter_metric(records, metric_name, encoding=spec["name"], **filters)
        summary[spec["name"]] = mean_confidence_interval(values, bounds=(0.0, 1.0) if "top1" in metric_name else None)
    return summary


def series_for(records, metric_name, x_key, x_values, **fixed):
    result = {}
    for spec in ENCODING_SPECS:
        result[spec["name"]] = []
        for x_value in x_values:
            values = filter_metric(records, metric_name, encoding=spec["name"], **fixed, **{x_key: x_value})
            summary = mean_confidence_interval(values, bounds=(0.0, 1.0) if "top1" in metric_name else None)
            summary[x_key] = float(x_value)
            result[spec["name"]].append(summary)
    return result


def main():
    apply_plot_style()
    import matplotlib.pyplot as plt

    started = utc_now_iso()
    t0 = time.time()
    rng = build_rng(SEED)

    all_records = []
    for decay in DECAY_VALUES:
        for erasure in (False, True):
            for pattern_count in PATTERN_COUNTS:
                for query_noise in QUERY_NOISE_LEVELS:
                    for trial_id in range(TRIALS):
                        trial_rng = child_rng(rng)
                        keys_raw = normalize_rows(trial_rng.standard_normal((pattern_count, HEAD_DIM)))
                        values_raw = normalize_rows(trial_rng.standard_normal((pattern_count, HEAD_DIM)))
                        target_idx = int(trial_rng.integers(0, pattern_count))
                        noisy_query = keys_raw[target_idx] + query_noise * trial_rng.standard_normal(HEAD_DIM)
                        scenario_id = (
                            f"p{pattern_count}_q{query_noise:.4f}_d{decay:.4f}_e{int(erasure)}_t{trial_id}"
                        )
                        for spec in ENCODING_SPECS:
                            record = trial_record(
                                keys_raw,
                                values_raw,
                                target_idx,
                                noisy_query,
                                pattern_count,
                                query_noise,
                                decay,
                                erasure,
                                spec,
                                scenario_id,
                            )
                            record["trial_id"] = int(trial_id)
                            all_records.append(record)

    output_dir = output_dir_for(SCRIPT_PATH)

    figure_path = output_dir / "asymmetric_outer_product_recall.png"
    metrics_path = output_dir / "asymmetric_outer_product_recall_metrics.json"

    base_decay = 0.4 if any(np.isclose(DECAY_VALUES, 0.4)) else float(DECAY_VALUES[0])
    base_count = 32 if 32 in PATTERN_COUNTS else int(PATTERN_COUNTS[min(len(PATTERN_COUNTS) - 1, 1)])
    base_noise = 0.1 if any(np.isclose(QUERY_NOISE_LEVELS, 0.1)) else float(QUERY_NOISE_LEVELS[0])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes[0, 0]
    exact_series = series_for(
        all_records,
        metric_name="raw_cosine",
        x_key="pattern_count",
        x_values=PATTERN_COUNTS,
        decay=base_decay,
        query_noise=0.0,
        erasure=False,
    )
    for spec in ENCODING_SPECS:
        xs = [entry["pattern_count"] for entry in exact_series[spec["name"]]]
        ys = [entry["mean"] for entry in exact_series[spec["name"]]]
        ax.plot(xs, ys, marker="o", label=spec["label"])
    ax.axhline(0.5, color="#991b1b", linestyle="--", alpha=0.5)
    ax.set_xlabel("stored patterns")
    ax.set_ylabel("raw recall cosine")
    ax.set_title(f"exact-query recall at decay={base_decay:.2f}, erasure=off")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    noise_series = series_for(
        all_records,
        metric_name="raw_cosine",
        x_key="query_noise",
        x_values=QUERY_NOISE_LEVELS,
        decay=base_decay,
        pattern_count=base_count,
        erasure=False,
    )
    for spec in ENCODING_SPECS:
        xs = [entry["query_noise"] for entry in noise_series[spec["name"]]]
        ys = [entry["mean"] for entry in noise_series[spec["name"]]]
        ax.plot(xs, ys, marker="o", label=spec["label"])
    ax.axhline(0.5, color="#991b1b", linestyle="--", alpha=0.5)
    ax.set_xlabel("query noise")
    ax.set_ylabel("raw recall cosine")
    ax.set_title(f"noise sensitivity at {base_count} patterns, decay={base_decay:.2f}")

    ax = axes[1, 0]
    decay_series = series_for(
        all_records,
        metric_name="raw_cosine",
        x_key="decay",
        x_values=DECAY_VALUES,
        pattern_count=base_count,
        query_noise=0.0,
        erasure=False,
    )
    for spec in ENCODING_SPECS:
        xs = [entry["decay"] for entry in decay_series[spec["name"]]]
        ys = [entry["mean"] for entry in decay_series[spec["name"]]]
        ax.plot(xs, ys, marker="o", label=spec["label"])
    ax.axhline(0.5, color="#991b1b", linestyle="--", alpha=0.5)
    ax.set_xlabel("decay coefficient")
    ax.set_ylabel("raw recall cosine")
    ax.set_title(f"decay sensitivity at {base_count} patterns")

    ax = axes[1, 1]
    erasure_deltas = []
    labels = []
    for spec in ENCODING_SPECS:
        no_erase = filter_metric(
            all_records,
            "raw_cosine",
            encoding=spec["name"],
            pattern_count=base_count,
            query_noise=base_noise,
            decay=base_decay,
            erasure=False,
        )
        with_erase = filter_metric(
            all_records,
            "raw_cosine",
            encoding=spec["name"],
            pattern_count=base_count,
            query_noise=base_noise,
            decay=base_decay,
            erasure=True,
        )
        erasure_deltas.append(float(np.mean(with_erase) - np.mean(no_erase)))
        labels.append(spec["label"])
    ax.bar(range(len(labels)), erasure_deltas)
    ax.axhline(0.0, color="#111827", linewidth=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("mean raw cosine delta")
    ax.set_title(f"erasure effect at {base_count} patterns, noise={base_noise:.2f}")

    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()

    selected_exact = summarize_by_encoding(
        all_records,
        "raw_cosine",
        pattern_count=base_count,
        query_noise=0.0,
        decay=base_decay,
        erasure=False,
    )
    selected_noisy = summarize_by_encoding(
        all_records,
        "raw_cosine",
        pattern_count=base_count,
        query_noise=base_noise,
        decay=base_decay,
        erasure=False,
    )
    selected_top1 = summarize_by_encoding(
        all_records,
        "raw_top1",
        pattern_count=base_count,
        query_noise=0.0,
        decay=base_decay,
        erasure=False,
    )

    best_exact = max(
        (
            {
                "encoding": spec["name"],
                "label": spec["label"],
                "raw_cosine_mean": selected_exact[spec["name"]]["mean"],
            }
            for spec in ENCODING_SPECS
        ),
        key=lambda item: float("-inf") if item["raw_cosine_mean"] is None else item["raw_cosine_mean"],
    )

    threshold_hits = []
    for pattern_count in PATTERN_COUNTS:
        for spec in ENCODING_SPECS:
            values = filter_metric(
                all_records,
                "raw_cosine",
                encoding=spec["name"],
                pattern_count=pattern_count,
                query_noise=0.0,
                decay=base_decay,
                erasure=False,
            )
            if values and float(np.mean(values)) >= 0.5:
                threshold_hits.append(
                    {
                        "encoding": spec["name"],
                        "pattern_count": int(pattern_count),
                        "raw_cosine_mean": float(np.mean(values)),
                    }
                )

    sparse_key_sparse_value, dense_key_sparse_value = paired_metric(
        all_records,
        "raw_cosine",
        "topk20_all",
        "dense_key_topk20_value",
        pattern_count=base_count,
        query_noise=base_noise,
        decay=base_decay,
        erasure=False,
    )

    summary = {
        "selected_condition": {
            "pattern_count": base_count,
            "query_noise": base_noise,
            "decay": base_decay,
            "erasure": False,
        },
        "selected_exact_query_raw_cosine": selected_exact,
        "selected_noisy_query_raw_cosine": selected_noisy,
        "selected_exact_query_top1": selected_top1,
        "best_exact_query_encoding": best_exact,
        "threshold_hits": threshold_hits,
        "exact_query_series": exact_series,
        "noise_series": noise_series,
        "decay_series": decay_series,
    }

    statistics = {}
    if dense_key_sparse_value and sparse_key_sparse_value:
        statistics["dense_key_vs_sparse_key_topk"] = paired_difference_stats(
            sparse_key_sparse_value,
            dense_key_sparse_value,
            SEED,
        )

    finished = utc_now_iso()
    duration = time.time() - t0

    record = build_run_record(
        simulation_name="asymmetric_outer_product_recall",
        script_path=SCRIPT_PATH,
        started_at_utc=started,
        finished_at_utc=finished,
        duration_sec=duration,
        parameters={
            "head_dim": HEAD_DIM,
            "pattern_counts": PATTERN_COUNTS,
            "query_noise_levels": QUERY_NOISE_LEVELS,
            "decay_values": DECAY_VALUES,
            "trials": TRIALS,
            "write_gate": WRITE_GATE,
            "topk_fraction": TOPK_FRACTION,
            "ternary_alpha": TERNARY_ALPHA,
            "encoding_specs": ENCODING_SPECS,
        },
        seed_numpy=SEED,
        n_trials=len(all_records),
        summary=summary,
        statistics=statistics,
        trials=all_records,
        artifacts=[
            {"path": figure_path.as_posix(), "type": "figure"},
            {"path": metrics_path.as_posix(), "type": "metrics"},
        ],
        warnings=[
            "synthetic gaussian keys and values, not trained activations",
            "rope and token-position structure are omitted so this isolates write/read geometry rather than full sequence modeling",
            "decay=0.4 is intentionally harsh and may make uniform recall over long write sequences structurally difficult",
        ],
    )

    write_json(metrics_path, record)

    print(f"done in {duration:.1f}s")
    print(f"best exact-query encoding at {base_count} patterns, decay={base_decay:.2f}: {best_exact['label']} ({best_exact['raw_cosine_mean']:.4f})")
    if threshold_hits:
        first_hit = max(threshold_hits, key=lambda item: (item["pattern_count"], item["raw_cosine_mean"]))
        print(
            f"threshold >= 0.5 reached by {first_hit['encoding']} at pattern_count={first_hit['pattern_count']} "
            f"with mean cosine={first_hit['raw_cosine_mean']:.4f}"
        )
    else:
        print("no encoding crossed the 0.5 cosine threshold under the exact-query baseline")


if __name__ == "__main__":
    main()