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
    env_float,
    env_int,
    env_list,
    mean_confidence_interval,
    output_dir_for,
    require_non_negative_list,
    require_positive,
    require_positive_list,
    require_unit_interval_list,
    utc_now_iso,
    write_json,
)

SCRIPT_PATH = Path(__file__).resolve()
SEED = env_int("ERR_SEED", 42)
TRIALS = env_int("ERR_TRIALS", 64)
FEATURE_DIM = env_int("ERR_FEATURE_DIM", 20)
ACTION_COUNT = env_int("ERR_ACTION_COUNT", 4)
CUE_DROPS = env_list("ERR_CUE_DROPS", float, [0.25, 0.5])
DISTRACTOR_COUNTS = env_list("ERR_DISTRACTORS", int, [0, 4, 8, 16])
REPLAY_STEPS = env_int("ERR_REPLAY_STEPS", 3)
DECAY = env_float("ERR_DECAY", 0.9)
DISTRACTOR_OVERLAP = env_float("ERR_DISTRACTOR_OVERLAP", 0.75)

require_positive("ERR_TRIALS", TRIALS)
require_positive("ERR_FEATURE_DIM", FEATURE_DIM)
require_positive("ERR_ACTION_COUNT", ACTION_COUNT)
require_unit_interval_list("ERR_CUE_DROPS", CUE_DROPS, allow_zero=True)
require_non_negative_list("ERR_DISTRACTORS", DISTRACTOR_COUNTS)
require_positive("ERR_REPLAY_STEPS", REPLAY_STEPS)
if not 0.0 < DECAY <= 1.0:
    raise ValueError("ERR_DECAY must be in (0, 1]")
if not 0.0 <= DISTRACTOR_OVERLAP <= 1.0:
    raise ValueError("ERR_DISTRACTOR_OVERLAP must be in [0, 1]")

POLICIES = ("targeted_replay", "no_replay", "random_replay", "recency_only")
PRIMARY_POLICY = "targeted_replay"


def sample_binary_pattern(rng: np.random.Generator, feature_dim: int) -> np.ndarray:
    return rng.integers(0, 2, size=feature_dim, dtype=np.int64)


def build_related_pattern(base: np.ndarray, overlap: float, rng: np.random.Generator) -> np.ndarray:
    feature_dim = int(base.shape[0])
    keep_count = int(round(float(overlap) * feature_dim))
    keep_count = max(0, min(feature_dim, keep_count))
    keep_idx = rng.choice(feature_dim, size=keep_count, replace=False) if keep_count > 0 else np.array([], dtype=np.int64)
    pattern = 1 - base
    pattern[keep_idx] = base[keep_idx]
    return pattern.astype(np.int64)


def observe_pattern(rng: np.random.Generator, prototype: np.ndarray, cue_drop: float) -> tuple[np.ndarray, np.ndarray]:
    feature_dim = int(prototype.shape[0])
    keep_count = int(round((1.0 - float(cue_drop)) * feature_dim))
    keep_count = max(1, min(feature_dim, keep_count))
    keep_idx = rng.choice(feature_dim, size=keep_count, replace=False)
    mask = np.zeros(feature_dim, dtype=bool)
    mask[keep_idx] = True
    return prototype.copy(), mask


def match_score(query: np.ndarray, mask: np.ndarray, prototype: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return 0.0
    return float(np.mean(prototype[mask] == query[mask]))


def retrieve_weighted(
    query: np.ndarray,
    mask: np.ndarray,
    prototypes: np.ndarray,
    strengths: np.ndarray,
) -> int:
    scores = np.array([match_score(query, mask, prototype) for prototype in prototypes], dtype=np.float64)
    weighted = scores * strengths
    return int(np.argmax(weighted))


def build_base_strengths(distractor_count: int) -> np.ndarray:
    strengths = np.zeros(distractor_count + 1, dtype=np.float64)
    strengths[0] = 1.0
    for distractor_idx in range(distractor_count):
        strengths *= float(DECAY)
        strengths[distractor_idx + 1] = 1.0
    return strengths


def evaluate_cell(
    rng: np.random.Generator,
    cue_drop: float,
    distractor_count: int,
    trial_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial_id in range(trial_count):
        target_prototype = sample_binary_pattern(rng, FEATURE_DIM)
        target_action = int(rng.integers(0, ACTION_COUNT))
        distractor_prototypes = np.vstack(
            [build_related_pattern(target_prototype, DISTRACTOR_OVERLAP, rng) for _ in range(distractor_count)]
        ).astype(np.int64) if distractor_count > 0 else np.zeros((0, FEATURE_DIM), dtype=np.int64)
        distractor_actions = rng.integers(0, ACTION_COUNT, size=distractor_count, dtype=np.int64) if distractor_count > 0 else np.zeros(0, dtype=np.int64)
        query, mask = observe_pattern(rng, target_prototype, cue_drop)

        all_prototypes = np.vstack([target_prototype[None, :], distractor_prototypes]).astype(np.int64)
        all_actions = np.concatenate([[target_action], distractor_actions]).astype(np.int64)

        base_strengths = build_base_strengths(distractor_count)

        policy_strengths = {
            "no_replay": base_strengths.copy(),
            "targeted_replay": base_strengths.copy(),
            "random_replay": base_strengths.copy(),
            "recency_only": base_strengths.copy(),
        }
        policy_strengths["targeted_replay"][0] += float(REPLAY_STEPS)
        for _ in range(REPLAY_STEPS):
            replay_idx = int(rng.integers(0, all_prototypes.shape[0]))
            policy_strengths["random_replay"][replay_idx] += 1.0

        recency_choice = int(all_prototypes.shape[0] - 1) if distractor_count > 0 else 0
        for policy in POLICIES:
            if policy == "recency_only":
                predicted_state = int(recency_choice)
            else:
                predicted_state = retrieve_weighted(query, mask, all_prototypes, policy_strengths[policy])
            predicted_action = int(all_actions[predicted_state])
            rows.append(
                {
                    "trial_id": int(trial_id),
                    "policy": policy,
                    "cue_drop": float(cue_drop),
                    "distractors": int(distractor_count),
                    "true_state": 0,
                    "predicted_state": int(predicted_state),
                    "true_action": int(target_action),
                    "predicted_action": int(predicted_action),
                    "state_correct": float(int(predicted_state == 0)),
                    "action_correct": float(int(predicted_action == int(target_action))),
                    "joint_correct": float(int(predicted_state == 0 and predicted_action == int(target_action))),
                }
            )
    return rows


def summarize_policy(rows: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    policy_rows = [row for row in rows if row["policy"] == policy]
    return {
        "state_probe_accuracy": mean_confidence_interval(
            [row["state_correct"] for row in policy_rows],
            bounds=(0.0, 1.0),
        ),
        "action_success": mean_confidence_interval(
            [row["action_correct"] for row in policy_rows],
            bounds=(0.0, 1.0),
        ),
        "joint_success": mean_confidence_interval(
            [row["joint_correct"] for row in policy_rows],
            bounds=(0.0, 1.0),
        ),
    }


def distractor_decay_curve(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for policy in POLICIES:
        curve: dict[str, float] = {}
        for distractors in sorted(set(int(row["distractors"]) for row in rows)):
            values = [
                row["joint_correct"]
                for row in rows
                if row["policy"] == policy and int(row["distractors"]) == distractors
            ]
            curve[str(distractors)] = float(np.mean(values)) if values else 0.0
        result[policy] = curve
    return result


def cue_drop_curve(rows: list[dict[str, Any]], policy: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for cue_drop in sorted(set(float(row["cue_drop"]) for row in rows)):
        values = [
            row["state_correct"]
            for row in rows
            if row["policy"] == policy and float(row["cue_drop"]) == cue_drop
        ]
        result[format(cue_drop, "g")] = float(np.mean(values)) if values else 0.0
    return result


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    policy_metrics = {policy: summarize_policy(rows, policy) for policy in POLICIES}
    targeted_joint = float(policy_metrics["targeted_replay"]["joint_success"]["mean"])
    no_replay_joint = float(policy_metrics["no_replay"]["joint_success"]["mean"])
    random_joint = float(policy_metrics["random_replay"]["joint_success"]["mean"])
    return {
        "state_probe_accuracy": float(policy_metrics[PRIMARY_POLICY]["state_probe_accuracy"]["mean"]),
        "action_success": float(policy_metrics[PRIMARY_POLICY]["action_success"]["mean"]),
        "joint_success": targeted_joint,
        "replay_reuse_gain": targeted_joint - no_replay_joint,
        "targeted_vs_random_replay_gap": targeted_joint - random_joint,
        "distractor_decay_curve": distractor_decay_curve(rows),
        "state_probe_accuracy_by_cue_drop": cue_drop_curve(rows, PRIMARY_POLICY),
        "policy_metrics": policy_metrics,
    }


def build_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary_rows = [row for row in rows if row["policy"] == PRIMARY_POLICY]
    return {
        "state_probe_accuracy": mean_confidence_interval(
            [row["state_correct"] for row in primary_rows],
            bounds=(0.0, 1.0),
        ),
        "action_success": mean_confidence_interval(
            [row["action_correct"] for row in primary_rows],
            bounds=(0.0, 1.0),
        ),
        "joint_success": mean_confidence_interval(
            [row["joint_correct"] for row in primary_rows],
            bounds=(0.0, 1.0),
        ),
    }


def build_figure(rows: list[dict[str, Any]], output_path: Path) -> None:
    apply_plot_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    distractor_labels = sorted(set(int(row["distractors"]) for row in rows))
    decay = distractor_decay_curve(rows)
    for policy in POLICIES:
        axes[0].plot(distractor_labels, [decay[policy][str(v)] for v in distractor_labels], marker="o", label=policy)
    axes[0].set_title("joint success vs distractors")
    axes[0].set_xlabel("distractor count")
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(frameon=False, fontsize=8)

    cue_labels = sorted(set(float(row["cue_drop"]) for row in rows))
    for policy in ("targeted_replay", "no_replay", "recency_only"):
        curve = cue_drop_curve(rows, policy)
        axes[1].plot(cue_labels, [curve[format(v, "g")] for v in cue_labels], marker="o", label=policy)
    axes[1].set_title("state probe vs cue drop")
    axes[1].set_xlabel("cue drop")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend(frameon=False, fontsize=8)

    summary = build_summary(rows)
    gains = [summary["replay_reuse_gain"], summary["targeted_vs_random_replay_gap"]]
    axes[2].bar(["targeted-no", "targeted-random"], gains, color=["#0f766e", "#1d4ed8"])
    axes[2].axhline(0.0, color="#111827", linewidth=1)
    axes[2].set_title("replay advantages")
    axes[2].set_ylabel("accuracy delta")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> dict[str, Any]:
    started = utc_now_iso()
    t0 = time.time()
    rng = build_rng(SEED)
    output_dir = output_dir_for(SCRIPT_PATH)

    rows: list[dict[str, Any]] = []
    for cue_drop in CUE_DROPS:
        for distractors in DISTRACTOR_COUNTS:
            rows.extend(
                evaluate_cell(
                    rng=rng,
                    cue_drop=float(cue_drop),
                    distractor_count=int(distractors),
                    trial_count=TRIALS,
                )
            )

    figure_path = output_dir / "episodic_replay_reuse.png"
    build_figure(rows, figure_path)

    summary = build_summary(rows)
    statistics = build_statistics(rows)

    finished = utc_now_iso()
    duration = time.time() - t0
    metrics_path = output_dir / "episodic_replay_reuse_metrics.json"
    record = build_run_record(
        simulation_name="episodic_replay_reuse",
        script_path=SCRIPT_PATH,
        started_at_utc=started,
        finished_at_utc=finished,
        duration_sec=duration,
        parameters={
            "trials": TRIALS,
            "feature_dim": FEATURE_DIM,
            "action_count": ACTION_COUNT,
            "cue_drops": [float(v) for v in CUE_DROPS],
            "distractors": [int(v) for v in DISTRACTOR_COUNTS],
            "replay_steps": REPLAY_STEPS,
            "decay": DECAY,
            "distractor_overlap": DISTRACTOR_OVERLAP,
        },
        seed_numpy=SEED,
        n_trials=len(rows),
        summary=summary,
        statistics=statistics,
        trials=rows,
        artifacts=[
            {"name": "episodic_replay_reuse.png", "path": figure_path, "type": "figure"},
            {"name": "episodic_replay_reuse_metrics.json", "path": metrics_path, "type": "metrics"},
        ],
        warnings=[],
    )
    write_json(metrics_path, record)
    return summary


if __name__ == "__main__":
    main()
