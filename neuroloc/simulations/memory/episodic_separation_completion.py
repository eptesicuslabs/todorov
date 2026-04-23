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
SEED = env_int("ESC_SEED", 42)
TRIALS = env_int("ESC_TRIALS", 64)
FEATURE_DIM = env_int("ESC_FEATURE_DIM", 20)
ACTION_COUNT = env_int("ESC_ACTION_COUNT", 4)
OVERLAPS = env_list("ESC_OVERLAPS", float, [0.5, 0.75, 0.9])
CUE_DROPS = env_list("ESC_CUE_DROPS", float, [0.0, 0.25, 0.5])
DISTRACTOR_COUNTS = env_list("ESC_DISTRACTORS", int, [0, 4, 8])

require_positive("ESC_TRIALS", TRIALS)
require_positive("ESC_FEATURE_DIM", FEATURE_DIM)
require_positive("ESC_ACTION_COUNT", ACTION_COUNT)
require_unit_interval_list("ESC_OVERLAPS", OVERLAPS, allow_zero=True)
require_unit_interval_list("ESC_CUE_DROPS", CUE_DROPS, allow_zero=True)
require_non_negative_list("ESC_DISTRACTORS", DISTRACTOR_COUNTS)

NOVEL_STATE = -1
NOVEL_ACTION = ACTION_COUNT
PRIMARY_POLICY = "episodic_index"
POLICIES = ("episodic_index", "oracle_read", "shuffled_mapping", "no_memory")


def validate_configuration() -> None:
    if FEATURE_DIM < 2:
        raise ValueError("ESC_FEATURE_DIM must be >= 2")


def sample_binary_pattern(rng: np.random.Generator, feature_dim: int) -> np.ndarray:
    return rng.integers(0, 2, size=feature_dim, dtype=np.int64)


def build_related_pattern(
    base: np.ndarray,
    overlap: float,
    rng: np.random.Generator,
) -> np.ndarray:
    feature_dim = int(base.shape[0])
    keep_count = int(round(float(overlap) * feature_dim))
    keep_count = max(0, min(feature_dim, keep_count))
    keep_idx = rng.choice(feature_dim, size=keep_count, replace=False) if keep_count > 0 else np.array([], dtype=np.int64)
    pattern = 1 - base
    pattern[keep_idx] = base[keep_idx]
    return pattern.astype(np.int64)


def select_completion_mask(
    rng: np.random.Generator,
    target: np.ndarray,
    competitor: np.ndarray,
    cue_drop: float,
) -> np.ndarray:
    feature_dim = int(target.shape[0])
    keep_count = int(round((1.0 - float(cue_drop)) * feature_dim))
    minimum_keep = 2 if feature_dim > 1 else 1
    keep_count = max(minimum_keep, min(feature_dim, keep_count))
    diff_idx = np.nonzero(target != competitor)[0]
    must_keep: list[int] = []
    if diff_idx.size > 0:
        must_keep.append(int(rng.choice(diff_idx)))
    available = [idx for idx in range(feature_dim) if idx not in must_keep]
    remaining = max(0, keep_count - len(must_keep))
    chosen = list(must_keep)
    if remaining > 0:
        chosen.extend(int(v) for v in rng.choice(np.array(available, dtype=np.int64), size=remaining, replace=False).tolist())
    mask = np.zeros(feature_dim, dtype=bool)
    mask[np.array(chosen, dtype=np.int64)] = True
    return mask


def build_novel_query(
    target: np.ndarray,
    competitor: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    query = target.copy()
    visible_idx = np.nonzero(mask)[0]
    diff_idx = [int(idx) for idx in visible_idx.tolist() if int(target[idx]) != int(competitor[idx])]
    if diff_idx:
        swap_idx = int(rng.choice(np.array(diff_idx, dtype=np.int64)))
        query[swap_idx] = int(competitor[swap_idx])
    shared_idx = [int(idx) for idx in visible_idx.tolist() if int(target[idx]) == int(competitor[idx])]
    if shared_idx:
        novelty_idx = int(rng.choice(np.array(shared_idx, dtype=np.int64)))
        query[novelty_idx] = int(1 - target[novelty_idx])
    return query.astype(np.int64)


def match_scores(
    query: np.ndarray,
    mask: np.ndarray,
    stored_patterns: np.ndarray,
) -> np.ndarray:
    if int(mask.sum()) == 0:
        return np.zeros(stored_patterns.shape[0], dtype=np.float64)
    visible = stored_patterns[:, mask]
    target = query[mask][None, :]
    return np.mean(visible == target, axis=1, dtype=np.float64)


def episodic_index_predict(
    query: np.ndarray,
    mask: np.ndarray,
    stored_patterns: np.ndarray,
    state_ids: np.ndarray,
    actions: np.ndarray,
) -> tuple[int, int, float]:
    scores = match_scores(query, mask, stored_patterns)
    score_list = [float(value) for value in scores.tolist()]
    best_idx = max(range(len(score_list)), key=score_list.__getitem__)
    if len(score_list) > 1:
        second_best = max(score for idx, score in enumerate(score_list) if idx != best_idx)
    else:
        second_best = 0.0
    best_score = float(scores[best_idx])
    margin = best_score - second_best
    if best_score >= 1.0 - 1e-9:
        return int(state_ids[best_idx]), int(actions[best_idx]), float(margin)
    return NOVEL_STATE, NOVEL_ACTION, float(margin)


def build_policy_tables(
    rng: np.random.Generator,
    episode_count: int,
    actions: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    base_states = np.arange(episode_count, dtype=np.int64)
    shuffled = rng.permutation(base_states)
    shuffled_actions = actions[shuffled]
    return {
        "episodic_index": (base_states, actions),
        "oracle_read": (base_states, actions),
        "shuffled_mapping": (shuffled.astype(np.int64), shuffled_actions.astype(np.int64)),
        "no_memory": (base_states, actions),
    }


def evaluate_cell(
    rng: np.random.Generator,
    overlap: float,
    cue_drop: float,
    distractor_count: int,
    trial_count: int,
) -> list[dict[str, Any]]:
    validate_configuration()
    rows: list[dict[str, Any]] = []
    for trial_id in range(trial_count):
        target = sample_binary_pattern(rng, FEATURE_DIM)
        competitor = build_related_pattern(target, overlap, rng)
        distractors = [sample_binary_pattern(rng, FEATURE_DIM) for _ in range(distractor_count)]
        stored_patterns = np.vstack([target, competitor, *distractors]).astype(np.int64)
        actions = rng.integers(0, ACTION_COUNT, size=stored_patterns.shape[0], dtype=np.int64)
        policy_tables = build_policy_tables(rng, stored_patterns.shape[0], actions)
        completion_mask = select_completion_mask(rng, target, competitor, cue_drop)
        completion_query = target.copy()
        novel_query = build_novel_query(target, competitor, completion_mask, rng)

        scenarios = (
            ("completion", completion_query, completion_mask, 0, int(actions[0])),
            ("novelty", novel_query, completion_mask, NOVEL_STATE, NOVEL_ACTION),
        )
        for query_type, query, mask, true_state, true_action in scenarios:
            primary_state, primary_action, margin = episodic_index_predict(
                query=query,
                mask=mask,
                stored_patterns=stored_patterns,
                state_ids=policy_tables[PRIMARY_POLICY][0],
                actions=policy_tables[PRIMARY_POLICY][1],
            )
            for policy in POLICIES:
                if policy == "oracle_read":
                    predicted_state = int(true_state)
                    predicted_action = int(true_action)
                elif policy == "no_memory":
                    predicted_state = NOVEL_STATE
                    predicted_action = NOVEL_ACTION
                else:
                    predicted_state, predicted_action, _ = episodic_index_predict(
                        query=query,
                        mask=mask,
                        stored_patterns=stored_patterns,
                        state_ids=policy_tables[policy][0],
                        actions=policy_tables[policy][1],
                    )
                rows.append(
                    {
                        "trial_id": int(trial_id),
                        "policy": policy,
                        "overlap": float(overlap),
                        "cue_drop": float(cue_drop),
                        "distractors": int(distractor_count),
                        "query_type": query_type,
                        "true_state": int(true_state),
                        "predicted_state": int(predicted_state),
                        "true_action": int(true_action),
                        "predicted_action": int(predicted_action),
                        "state_correct": float(int(predicted_state == int(true_state))),
                        "action_correct": float(int(predicted_action == int(true_action))),
                        "joint_correct": float(int(predicted_state == int(true_state) and predicted_action == int(true_action))),
                        "primary_margin": float(margin),
                    }
                )
    return rows


def summarize_policy(rows: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    completion_rows = [row for row in rows if row["policy"] == policy and row["query_type"] == "completion"]
    novelty_rows = [row for row in rows if row["policy"] == policy and row["query_type"] == "novelty"]
    return {
        "state_probe_accuracy": mean_confidence_interval(
            [row["state_correct"] for row in completion_rows],
            bounds=(0.0, 1.0),
        ),
        "action_success": mean_confidence_interval(
            [row["action_correct"] for row in completion_rows],
            bounds=(0.0, 1.0),
        ),
        "joint_success": mean_confidence_interval(
            [row["joint_correct"] for row in completion_rows],
            bounds=(0.0, 1.0),
        ),
        "novelty_detection_accuracy": mean_confidence_interval(
            [row["state_correct"] for row in novelty_rows],
            bounds=(0.0, 1.0),
        ),
    }


def completion_by_cue_drop(rows: list[dict[str, Any]], policy: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for cue_drop in sorted(set(float(row["cue_drop"]) for row in rows)):
        values = [
            row["state_correct"]
            for row in rows
            if row["policy"] == policy and row["query_type"] == "completion" and float(row["cue_drop"]) == float(cue_drop)
        ]
        result[format(cue_drop, "g")] = float(np.mean(values)) if values else 0.0
    return result


def delayed_recall_by_distractors(rows: list[dict[str, Any]], policy: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for distractors in sorted(set(int(row["distractors"]) for row in rows)):
        values = [
            row["joint_correct"]
            for row in rows
            if row["policy"] == policy and row["query_type"] == "completion" and int(row["distractors"]) == distractors
        ]
        result[str(distractors)] = float(np.mean(values)) if values else 0.0
    return result


def novelty_accuracy(rows: list[dict[str, Any]], policy: str) -> float:
    values = [row["state_correct"] for row in rows if row["policy"] == policy and row["query_type"] == "novelty"]
    return float(np.mean(values)) if values else 0.0


def mean_margin(rows: list[dict[str, Any]], policy: str) -> float:
    values = [row["primary_margin"] for row in rows if row["policy"] == policy and row["query_type"] == "completion"]
    return float(np.mean(values)) if values else 0.0


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = summarize_policy(rows, PRIMARY_POLICY)
    policy_metrics = {policy: summarize_policy(rows, policy) for policy in POLICIES}
    return {
        "state_probe_accuracy": float(primary["state_probe_accuracy"]["mean"]),
        "action_success": float(primary["action_success"]["mean"]),
        "joint_success": float(primary["joint_success"]["mean"]),
        "separation_margin": mean_margin(rows, PRIMARY_POLICY),
        "completion_accuracy_by_cue_drop": completion_by_cue_drop(rows, PRIMARY_POLICY),
        "novelty_detection_accuracy": novelty_accuracy(rows, PRIMARY_POLICY),
        "delayed_recall_after_distractors": delayed_recall_by_distractors(rows, PRIMARY_POLICY),
        "policy_metrics": policy_metrics,
    }


def build_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary_rows = [row for row in rows if row["policy"] == PRIMARY_POLICY]
    return {
        "completion_state_accuracy": mean_confidence_interval(
            [row["state_correct"] for row in primary_rows if row["query_type"] == "completion"],
            bounds=(0.0, 1.0),
        ),
        "novelty_state_accuracy": mean_confidence_interval(
            [row["state_correct"] for row in primary_rows if row["query_type"] == "novelty"],
            bounds=(0.0, 1.0),
        ),
        "action_success": mean_confidence_interval(
            [row["action_correct"] for row in primary_rows if row["query_type"] == "completion"],
            bounds=(0.0, 1.0),
        ),
        "joint_success": mean_confidence_interval(
            [row["joint_correct"] for row in primary_rows if row["query_type"] == "completion"],
            bounds=(0.0, 1.0),
        ),
        "separation_margin": mean_confidence_interval(
            [row["primary_margin"] for row in primary_rows if row["query_type"] == "completion"],
        ),
    }


def build_figure(rows: list[dict[str, Any]], output_path: Path) -> None:
    apply_plot_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    cue_labels = sorted(set(float(row["cue_drop"]) for row in rows))
    for policy in ("episodic_index", "shuffled_mapping", "no_memory"):
        values = [completion_by_cue_drop(rows, policy)[format(label, "g")] for label in cue_labels]
        axes[0, 0].plot(cue_labels, values, marker="o", label=policy)
    axes[0, 0].set_title("completion state accuracy")
    axes[0, 0].set_xlabel("cue drop")
    axes[0, 0].set_ylabel("accuracy")
    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 0].legend(frameon=False)

    overlap_labels = sorted(set(float(row["overlap"]) for row in rows))
    novelty_values = []
    for overlap in overlap_labels:
        overlap_rows = [row for row in rows if row["policy"] == PRIMARY_POLICY and row["query_type"] == "novelty" and float(row["overlap"]) == overlap]
        novelty_values.append(float(np.mean([row["state_correct"] for row in overlap_rows])) if overlap_rows else 0.0)
    axes[0, 1].plot(overlap_labels, novelty_values, marker="o", color="#b91c1c")
    axes[0, 1].set_title("novelty detection")
    axes[0, 1].set_xlabel("feature overlap")
    axes[0, 1].set_ylabel("accuracy")
    axes[0, 1].set_ylim(0.0, 1.05)

    distractor_labels = sorted(set(int(row["distractors"]) for row in rows))
    for policy in ("episodic_index", "oracle_read", "no_memory"):
        values = [delayed_recall_by_distractors(rows, policy)[str(label)] for label in distractor_labels]
        axes[1, 0].plot(distractor_labels, values, marker="o", label=policy)
    axes[1, 0].set_title("joint success after distractors")
    axes[1, 0].set_xlabel("distractor count")
    axes[1, 0].set_ylabel("accuracy")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].legend(frameon=False)

    margin_by_overlap = []
    for overlap in overlap_labels:
        overlap_rows = [
            row["primary_margin"]
            for row in rows
            if row["policy"] == PRIMARY_POLICY and row["query_type"] == "completion" and float(row["overlap"]) == overlap
        ]
        margin_by_overlap.append(float(np.mean(overlap_rows)) if overlap_rows else 0.0)
    axes[1, 1].plot(overlap_labels, margin_by_overlap, marker="o", color="#0f766e")
    axes[1, 1].set_title("separation margin")
    axes[1, 1].set_xlabel("feature overlap")
    axes[1, 1].set_ylabel("margin")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> dict[str, Any]:
    validate_configuration()
    started = utc_now_iso()
    t0 = time.time()
    rng = build_rng(SEED)
    output_dir = output_dir_for(SCRIPT_PATH)

    rows: list[dict[str, Any]] = []
    for overlap in OVERLAPS:
        for cue_drop in CUE_DROPS:
            for distractors in DISTRACTOR_COUNTS:
                rows.extend(
                    evaluate_cell(
                        rng=rng,
                        overlap=float(overlap),
                        cue_drop=float(cue_drop),
                        distractor_count=int(distractors),
                        trial_count=TRIALS,
                    )
                )

    figure_path = output_dir / "episodic_separation_completion.png"
    build_figure(rows, figure_path)

    summary = build_summary(rows)
    statistics = build_statistics(rows)

    finished = utc_now_iso()
    duration = time.time() - t0
    metrics_path = output_dir / "episodic_separation_completion_metrics.json"
    record = build_run_record(
        simulation_name="episodic_separation_completion",
        script_path=SCRIPT_PATH,
        started_at_utc=started,
        finished_at_utc=finished,
        duration_sec=duration,
        parameters={
            "trials": TRIALS,
            "feature_dim": FEATURE_DIM,
            "action_count": ACTION_COUNT,
            "overlaps": [float(v) for v in OVERLAPS],
            "cue_drops": [float(v) for v in CUE_DROPS],
            "distractors": [int(v) for v in DISTRACTOR_COUNTS],
        },
        seed_numpy=SEED,
        n_trials=len(rows),
        summary=summary,
        statistics=statistics,
        trials=rows,
        artifacts=[
            {"name": "episodic_separation_completion.png", "path": figure_path, "type": "figure"},
            {"name": "episodic_separation_completion_metrics.json", "path": metrics_path, "type": "metrics"},
        ],
        warnings=[],
    )
    write_json(metrics_path, record)
    return summary


if __name__ == "__main__":
    main()
