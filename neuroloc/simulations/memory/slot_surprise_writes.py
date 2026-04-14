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
    require_non_negative_list,
    require_positive,
    require_positive_list,
    require_unit_interval,
    require_unit_interval_list,
    utc_now_iso,
    write_json,
)

SCRIPT_PATH = Path(__file__).resolve()
SEED = env_int("SLOT_SW_SEED", 42)
HEAD_DIMS = env_list("SLOT_SW_HEAD_DIMS", int, [64, 128])
SLOT_COUNTS = env_list("SLOT_SW_SLOTS", int, [32, 64, 128])
INTERVENING_DISTANCES = env_list("SLOT_SW_DISTANCES", int, [64, 128, 256, 512])
PREDICTABLE_FRACTIONS = env_list("SLOT_SW_PREDICTABLE", float, [0.80, 0.90, 0.95])
SURPRISE_TAUS = env_list("SLOT_SW_TAUS", float, [0.05, 0.1, 0.2])
TEMPERATURE = env_float("SLOT_SW_TEMPERATURE", 0.1)
TRIALS = env_int("SLOT_SW_TRIALS", 32)

require_positive_list("SLOT_SW_HEAD_DIMS", HEAD_DIMS)
require_positive_list("SLOT_SW_SLOTS", SLOT_COUNTS)
require_positive_list("SLOT_SW_DISTANCES", INTERVENING_DISTANCES)
require_unit_interval_list("SLOT_SW_PREDICTABLE", PREDICTABLE_FRACTIONS)
require_unit_interval_list("SLOT_SW_TAUS", SURPRISE_TAUS)
require_unit_interval("SLOT_SW_TEMPERATURE", TEMPERATURE)
require_positive("SLOT_SW_TRIALS", TRIALS)

EPS = 1e-9
PASS_COSINE_THRESHOLD = 0.80
PASS_DISTANCE = 256
PASS_PREDICTABLE = 0.90
PASS_TAU = 0.1


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, EPS, None)


def sample_unit_vectors(rng: np.random.Generator, count: int, dim: int) -> np.ndarray:
    return normalize_rows(rng.standard_normal((count, dim)))


def synthesize_stream(
    rng: np.random.Generator,
    anchor: np.ndarray,
    length: int,
    predictable_fraction: float,
    surprise_level: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dim = int(anchor.shape[0])
    keys = np.zeros((length, dim), dtype=np.float64)
    values = np.zeros((length, dim), dtype=np.float64)
    is_surprising = np.zeros(length, dtype=bool)
    for t in range(length):
        if rng.random() < predictable_fraction:
            residual = rng.standard_normal(dim) * 0.05
            raw = anchor + residual
            keys[t] = raw / max(float(np.linalg.norm(raw)), EPS)
            value_residual = rng.standard_normal(dim) * 0.05
            raw_v = anchor + value_residual
            values[t] = raw_v / max(float(np.linalg.norm(raw_v)), EPS)
        else:
            raw = rng.standard_normal(dim) * surprise_level
            keys[t] = raw / max(float(np.linalg.norm(raw)), EPS)
            raw_v = rng.standard_normal(dim)
            values[t] = raw_v / max(float(np.linalg.norm(raw_v)), EPS)
            is_surprising[t] = True
    return keys, values, is_surprising


def surprise_score(key: np.ndarray, anchor: np.ndarray) -> float:
    overlap = float(key @ anchor)
    clipped = max(-1.0, min(1.0, overlap))
    return (1.0 - clipped) / 2.0


def slot_write_and_retrieve(
    keys: np.ndarray,
    values: np.ndarray,
    is_surprising: np.ndarray,
    anchor: np.ndarray,
    slot_count: int,
    tau: float,
    target_position: int,
    intervening_distance: int,
    temperature: float,
) -> dict[str, Any]:
    dim = keys.shape[1]
    slot_keys = np.zeros((slot_count, dim), dtype=np.float64)
    slot_values = np.zeros((slot_count, dim), dtype=np.float64)
    slot_age = np.zeros(slot_count, dtype=np.int64)
    slot_active = np.zeros(slot_count, dtype=bool)
    slot_holds_target = np.zeros(slot_count, dtype=bool)
    write_count = 0
    target_slot: int | None = None
    total_steps = target_position + intervening_distance + 1
    for t in range(total_steps):
        current_key = keys[t]
        current_value = values[t]
        s_t = surprise_score(current_key, anchor)
        slot_age += 1
        if s_t >= tau:
            if not np.any(slot_active):
                chosen = 0
            elif np.all(slot_active):
                chosen = int(np.argmax(slot_age))
            else:
                inactive_indices = np.where(~slot_active)[0]
                chosen = int(inactive_indices[0])
            slot_keys[chosen] = current_key
            slot_values[chosen] = current_value
            slot_age[chosen] = 0
            slot_active[chosen] = True
            slot_holds_target[chosen] = False
            write_count += 1
            if t == target_position:
                target_slot = chosen
                slot_holds_target[chosen] = True
    if target_slot is None:
        return {
            "retrieved": False,
            "write_count": write_count,
            "target_cosine": float("nan"),
            "target_exact": 0,
            "target_evicted": False,
        }
    query = keys[target_position]
    scores = slot_keys @ query
    scaled = scores / max(temperature, EPS)
    shifted = scaled - np.max(scaled)
    weights = np.exp(shifted)
    weights = weights * slot_active
    total = np.sum(weights)
    if total <= 0.0:
        return {
            "retrieved": True,
            "write_count": write_count,
            "target_cosine": 0.0,
            "target_exact": 0,
            "target_evicted": not bool(slot_holds_target[target_slot]),
        }
    weights = weights / total
    output = weights @ slot_values
    target_value = values[target_position]
    output_norm = max(float(np.linalg.norm(output)), EPS)
    target_norm = max(float(np.linalg.norm(target_value)), EPS)
    cosine = float(output @ target_value / (output_norm * target_norm))
    top_slot = int(np.argmax(weights))
    target_exact = bool(
        bool(slot_holds_target[target_slot])
        and int(top_slot) == int(target_slot)
    )
    return {
        "retrieved": True,
        "write_count": write_count,
        "target_cosine": cosine,
        "target_exact": int(target_exact),
        "target_slot_final_age": int(slot_age[target_slot]) if slot_active[target_slot] else -1,
        "target_evicted": not bool(slot_holds_target[target_slot]),
    }


def evaluate_cell(
    rng: np.random.Generator,
    head_dim: int,
    slot_count: int,
    intervening_distance: int,
    predictable_fraction: float,
    tau: float,
    temperature: float,
    trial_count: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    skipped_no_surprise = 0
    skipped_insufficient_distance = 0
    skipped_unretrieved = 0
    evicted_count = 0
    leading_headroom = max(64, intervening_distance // 2)
    for trial in range(trial_count):
        trial_rng = child_rng(rng)
        anchor = sample_unit_vectors(trial_rng, 1, head_dim)[0]
        total_length = intervening_distance + leading_headroom
        keys, values, is_surprising = synthesize_stream(
            trial_rng, anchor, total_length, predictable_fraction
        )
        surprising_positions = np.where(is_surprising)[0]
        if len(surprising_positions) == 0:
            skipped_no_surprise += 1
            continue
        target_position = int(surprising_positions[0])
        available_distance = total_length - 1 - target_position
        if available_distance < intervening_distance:
            skipped_insufficient_distance += 1
            continue
        result = slot_write_and_retrieve(
            keys,
            values,
            is_surprising,
            anchor,
            slot_count,
            tau,
            target_position,
            intervening_distance,
            temperature,
        )
        if not result["retrieved"]:
            skipped_unretrieved += 1
            continue
        if result.get("target_evicted", False):
            evicted_count += 1
        records.append(
            {
                "trial": trial,
                "head_dim": head_dim,
                "slot_count": slot_count,
                "intervening_distance": intervening_distance,
                "predictable_fraction": predictable_fraction,
                "tau": tau,
                "temperature": temperature,
                "write_count": int(result["write_count"]),
                "target_cosine": float(result["target_cosine"]),
                "target_exact": int(result["target_exact"]),
                "target_evicted": int(bool(result.get("target_evicted", False))),
            }
        )
    skips = {
        "no_surprise": skipped_no_surprise,
        "insufficient_distance": skipped_insufficient_distance,
        "unretrieved": skipped_unretrieved,
        "evicted_counted_in_records": evicted_count,
    }
    return records, skips


def format_value(value: float) -> str:
    return format(float(value), "g").replace(".", "p").replace("-", "m")


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_cell: dict[str, Any] = {}
    cells: dict[tuple[int, int, int, float, float], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            int(record["head_dim"]),
            int(record["slot_count"]),
            int(record["intervening_distance"]),
            float(record["predictable_fraction"]),
            float(record["tau"]),
        )
        cells.setdefault(key, []).append(record)
    for (head_dim, slots, distance, predictable, tau), cell_records in cells.items():
        cosines = [r["target_cosine"] for r in cell_records if np.isfinite(r["target_cosine"])]
        exact = [r["target_exact"] for r in cell_records]
        writes = [r["write_count"] for r in cell_records]
        cell_key = (
            f"d{head_dim}_N{slots}_dist{distance}_p{format_value(predictable)}_tau{format_value(tau)}"
        )
        by_cell[cell_key] = {
            "target_cosine": mean_confidence_interval(cosines, bounds=(-1.0, 1.0)),
            "target_exact": mean_confidence_interval(exact, bounds=(0.0, 1.0)),
            "write_count": mean_confidence_interval(writes),
        }
    pass_cosine_mean: float | None = None
    pass_cell_label: str | None = None
    for head_dim in HEAD_DIMS:
        for slots in SLOT_COUNTS:
            key = (
                int(head_dim),
                int(slots),
                int(PASS_DISTANCE),
                float(PASS_PREDICTABLE),
                float(PASS_TAU),
            )
            if key not in cells:
                continue
            cosines = [r["target_cosine"] for r in cells[key] if np.isfinite(r["target_cosine"])]
            if not cosines:
                continue
            mean_value = float(np.mean(cosines))
            if pass_cosine_mean is None or mean_value > pass_cosine_mean:
                pass_cosine_mean = mean_value
                pass_cell_label = (
                    f"d{head_dim}_N{slots}_dist{int(PASS_DISTANCE)}_p{format_value(float(PASS_PREDICTABLE))}_tau{format_value(float(PASS_TAU))}"
                )
    pass_criterion_met = pass_cosine_mean is not None and pass_cosine_mean >= PASS_COSINE_THRESHOLD
    return {
        "pass_criterion_threshold": PASS_COSINE_THRESHOLD,
        "pass_criterion_best_cosine": pass_cosine_mean,
        "pass_criterion_met": bool(pass_criterion_met),
        "pass_reference_cell": pass_cell_label,
        "cells": by_cell,
    }


def build_figure(records: list[dict[str, Any]], output_path: Path) -> None:
    apply_plot_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(HEAD_DIMS), figsize=(5 * len(HEAD_DIMS), 4), sharey=True)
    if len(HEAD_DIMS) == 1:
        axes = [axes]
    for ax, head_dim in zip(axes, HEAD_DIMS):
        for slots in SLOT_COUNTS:
            for predictable in PREDICTABLE_FRACTIONS:
                xs: list[int] = []
                ys: list[float] = []
                for distance in INTERVENING_DISTANCES:
                    key = (int(head_dim), int(slots), int(distance), float(predictable), float(PASS_TAU))
                    cosines = [
                        r["target_cosine"]
                        for r in records
                        if (
                            int(r["head_dim"]) == int(head_dim)
                            and int(r["slot_count"]) == int(slots)
                            and int(r["intervening_distance"]) == int(distance)
                            and float(r["predictable_fraction"]) == float(predictable)
                            and float(r["tau"]) == float(PASS_TAU)
                            and np.isfinite(r["target_cosine"])
                        )
                    ]
                    if not cosines:
                        continue
                    xs.append(distance)
                    ys.append(float(np.mean(cosines)))
                if not xs:
                    continue
                ax.plot(xs, ys, marker="o", label=f"N={slots}, p={predictable}")
        ax.axhline(PASS_COSINE_THRESHOLD, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("intervening distance (tokens)")
        ax.set_title(f"head_dim={head_dim}")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("mean retrieval cosine (target token)")
    fig.suptitle(f"slot buffer surprise-gated writes (tau={PASS_TAU})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> dict[str, Any]:
    started_at = utc_now_iso()
    start_wall = time.time()
    rng = build_rng(SEED)
    trial_records: list[dict[str, Any]] = []
    skip_totals = {
        "no_surprise": 0,
        "insufficient_distance": 0,
        "unretrieved": 0,
        "evicted_counted_in_records": 0,
    }
    for head_dim in HEAD_DIMS:
        for slots in SLOT_COUNTS:
            for distance in INTERVENING_DISTANCES:
                for predictable in PREDICTABLE_FRACTIONS:
                    for tau in SURPRISE_TAUS:
                        cell_rng = child_rng(rng)
                        records, skips = evaluate_cell(
                            cell_rng,
                            int(head_dim),
                            int(slots),
                            int(distance),
                            float(predictable),
                            float(tau),
                            float(TEMPERATURE),
                            int(TRIALS),
                        )
                        trial_records.extend(records)
                        for key_name in skip_totals:
                            skip_totals[key_name] += int(skips.get(key_name, 0))
    summary = build_summary(trial_records)
    finished_at = utc_now_iso()
    duration = time.time() - start_wall
    output_dir = output_dir_for(SCRIPT_PATH)
    metrics_path = output_dir / "slot_surprise_writes_metrics.json"
    figure_path = output_dir / "slot_surprise_writes.png"
    build_figure(trial_records, figure_path)
    artifact_entries = [
        {
            "name": "slot_surprise_writes.png",
            "path": str(figure_path),
            "type": "figure",
            "description": "slot retrieval cosine vs intervening distance by config",
        },
        {
            "name": "slot_surprise_writes_metrics.json",
            "path": str(metrics_path),
            "type": "metrics",
            "description": "retrieval success under surprise-gated writes",
        },
    ]
    parameters = {
        "head_dims": list(HEAD_DIMS),
        "slot_counts": list(SLOT_COUNTS),
        "intervening_distances": list(INTERVENING_DISTANCES),
        "predictable_fractions": list(PREDICTABLE_FRACTIONS),
        "surprise_taus": list(SURPRISE_TAUS),
        "temperature": float(TEMPERATURE),
        "trials": int(TRIALS),
        "pass_cosine_threshold": PASS_COSINE_THRESHOLD,
        "pass_distance": int(PASS_DISTANCE),
        "pass_predictable": float(PASS_PREDICTABLE),
        "pass_tau": float(PASS_TAU),
    }
    statistics = {
        "elapsed_seconds": duration,
        "trial_record_count": len(trial_records),
        "skip_totals": skip_totals,
    }
    warnings: list[str] = []
    if skip_totals["no_surprise"] > 0:
        warnings.append(
            f"skipped {skip_totals['no_surprise']} trials: no surprising token appeared in the stream"
        )
    if skip_totals["insufficient_distance"] > 0:
        warnings.append(
            f"skipped {skip_totals['insufficient_distance']} trials: target position too late for requested intervening distance"
        )
    if skip_totals["unretrieved"] > 0:
        warnings.append(
            f"skipped {skip_totals['unretrieved']} trials: target not written before retrieval"
        )
    if skip_totals["evicted_counted_in_records"] > 0:
        warnings.append(
            f"target slot was evicted in {skip_totals['evicted_counted_in_records']} recorded trials; target_evicted flag set per record"
        )
    record = build_run_record(
        simulation_name="slot_surprise_writes",
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_sec=duration,
        parameters=parameters,
        seed_numpy=SEED,
        n_trials=len(trial_records),
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
    print("best cosine at distance=256, p=0.90, tau=0.1:", summary["pass_criterion_best_cosine"])
    print("pass reference cell:", summary["pass_reference_cell"])
