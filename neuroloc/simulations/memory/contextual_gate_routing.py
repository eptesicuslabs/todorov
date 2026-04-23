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
    mean_confidence_interval,
    output_dir_for,
    require_positive,
    require_unit_interval,
    utc_now_iso,
    write_json,
)

SCRIPT_PATH = Path(__file__).resolve()
SEED = env_int("CGR_SEED", 42)
TRIALS = env_int("CGR_TRIALS", 96)
N_CUES = env_int("CGR_CUES", 8)
N_CONTEXTS = env_int("CGR_CONTEXTS", 3)
ACTION_COUNT = env_int("CGR_ACTIONS", 4)
CONTEXTUAL_FRACTION = env_float("CGR_CONTEXTUAL_FRACTION", 0.5)

require_positive("CGR_TRIALS", TRIALS)
require_positive("CGR_CUES", N_CUES)
require_positive("CGR_CONTEXTS", N_CONTEXTS)
require_positive("CGR_ACTIONS", ACTION_COUNT)
require_unit_interval("CGR_CONTEXTUAL_FRACTION", CONTEXTUAL_FRACTION, allow_zero=True)

POLICIES = ("correct_context", "bottom_up_only", "shuffled_context", "forced_gate_open", "forced_gate_closed")
PRIMARY_POLICY = "correct_context"


def build_tables(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    contextual_states = np.arange(N_CUES * N_CONTEXTS, dtype=np.int64).reshape(N_CUES, N_CONTEXTS)
    contextual_actions = rng.integers(0, ACTION_COUNT, size=(N_CUES, N_CONTEXTS), dtype=np.int64)
    feedforward_states = (1000 + np.arange(N_CUES, dtype=np.int64)).astype(np.int64)
    feedforward_actions = rng.integers(0, ACTION_COUNT, size=N_CUES, dtype=np.int64)
    return contextual_states, contextual_actions, feedforward_states, feedforward_actions


def validate_configuration() -> None:
    if N_CONTEXTS < 2:
        raise ValueError("CGR_CONTEXTS must be >= 2")


def sample_derangement(rng: np.random.Generator, size: int) -> np.ndarray:
    base = np.arange(size, dtype=np.int64)
    if size <= 1:
        return base
    while True:
        perm = rng.permutation(base)
        if not np.any(perm == base):
            return perm.astype(np.int64)


def predict_trial(
    cue_id: int,
    context_id: int,
    requires_context: bool,
    policy: str,
    contextual_states: np.ndarray,
    contextual_actions: np.ndarray,
    feedforward_states: np.ndarray,
    feedforward_actions: np.ndarray,
    shuffled_contexts: np.ndarray,
) -> tuple[int, int]:
    if policy == "correct_context":
        if requires_context:
            return int(contextual_states[cue_id, context_id]), int(contextual_actions[cue_id, context_id])
        return int(feedforward_states[cue_id]), int(feedforward_actions[cue_id])
    if policy in {"bottom_up_only", "forced_gate_closed"}:
        if not requires_context:
            return int(feedforward_states[cue_id]), int(feedforward_actions[cue_id])
        return int(contextual_states[cue_id, 0]), int(contextual_actions[cue_id, 0])
    if policy == "shuffled_context":
        if requires_context:
            shuffled = int(shuffled_contexts[context_id])
            return int(contextual_states[cue_id, shuffled]), int(contextual_actions[cue_id, shuffled])
        return int(feedforward_states[cue_id]), int(feedforward_actions[cue_id])
    if policy == "forced_gate_open":
        return int(contextual_states[cue_id, context_id]), int(contextual_actions[cue_id, context_id])
    raise ValueError(f"unknown policy: {policy}")


def evaluate_trials(rng: np.random.Generator, trial_count: int) -> list[dict[str, Any]]:
    validate_configuration()
    contextual_states, contextual_actions, feedforward_states, feedforward_actions = build_tables(rng)
    shuffled_contexts = sample_derangement(rng, N_CONTEXTS)
    rows: list[dict[str, Any]] = []
    for trial_id in range(trial_count):
        cue_id = int(rng.integers(0, N_CUES))
        context_id = int(rng.integers(0, N_CONTEXTS))
        requires_context = bool(rng.random() < CONTEXTUAL_FRACTION)
        if requires_context:
            true_state = int(contextual_states[cue_id, context_id])
            true_action = int(contextual_actions[cue_id, context_id])
        else:
            true_state = int(feedforward_states[cue_id])
            true_action = int(feedforward_actions[cue_id])
        for policy in POLICIES:
            predicted_state, predicted_action = predict_trial(
                cue_id=cue_id,
                context_id=context_id,
                requires_context=requires_context,
                policy=policy,
                contextual_states=contextual_states,
                contextual_actions=contextual_actions,
                feedforward_states=feedforward_states,
                feedforward_actions=feedforward_actions,
                shuffled_contexts=shuffled_contexts,
            )
            rows.append(
                {
                    "trial_id": int(trial_id),
                    "policy": policy,
                    "cue_id": int(cue_id),
                    "context_id": int(context_id),
                    "requires_context": int(requires_context),
                    "true_state": int(true_state),
                    "predicted_state": int(predicted_state),
                    "true_action": int(true_action),
                    "predicted_action": int(predicted_action),
                    "state_correct": float(int(predicted_state == true_state)),
                    "action_correct": float(int(predicted_action == true_action)),
                    "joint_correct": float(int(predicted_state == true_state and predicted_action == true_action)),
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


def by_trial_type(rows: list[dict[str, Any]], policy: str, metric_key: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for requires_context in (0, 1):
        label = "contextual" if requires_context else "feedforward"
        values = [
            row[metric_key]
            for row in rows
            if row["policy"] == policy and int(row["requires_context"]) == requires_context
        ]
        result[label] = float(np.mean(values)) if values else 0.0
    return result


def false_bind_rate(rows: list[dict[str, Any]]) -> float:
    contextual_rows = [
        row
        for row in rows
        if row["policy"] == "shuffled_context" and int(row["requires_context"]) == 1
    ]
    values = [
        float(int(row["predicted_action"] != row["true_action"]))
        for row in contextual_rows
    ]
    return float(np.mean(values)) if values else 0.0


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    policy_metrics = {policy: summarize_policy(rows, policy) for policy in POLICIES}
    action_success_by_trial_type = {
        policy: by_trial_type(rows, policy, "action_correct")
        for policy in POLICIES
    }
    joint_success_by_trial_type = {
        policy: by_trial_type(rows, policy, "joint_correct")
        for policy in POLICIES
    }
    return {
        "state_probe_accuracy": float(policy_metrics[PRIMARY_POLICY]["state_probe_accuracy"]["mean"]),
        "action_success": float(policy_metrics[PRIMARY_POLICY]["action_success"]["mean"]),
        "joint_success": float(policy_metrics[PRIMARY_POLICY]["joint_success"]["mean"]),
        "context_gate_dependency": (
            joint_success_by_trial_type[PRIMARY_POLICY]["contextual"]
            - joint_success_by_trial_type["forced_gate_closed"]["contextual"]
        ),
        "false_bind_rate_under_wrong_context": false_bind_rate(rows),
        "action_success_by_trial_type": action_success_by_trial_type,
        "joint_success_by_trial_type": joint_success_by_trial_type,
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

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    policies = list(POLICIES)
    joint_means = [float(summarize_policy(rows, policy)["joint_success"]["mean"]) for policy in policies]
    axes[0].bar(policies, joint_means, color="#2563eb")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("overall joint success")
    axes[0].tick_params(axis="x", rotation=25)

    summary = build_summary(rows)
    for policy in ("correct_context", "forced_gate_open", "forced_gate_closed", "shuffled_context"):
        curve = summary["joint_success_by_trial_type"][policy]
        axes[1].plot(["feedforward", "contextual"], [curve["feedforward"], curve["contextual"]], marker="o", label=policy)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("joint success by trial type")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].bar(
        ["gate gap", "false bind"],
        [summary["context_gate_dependency"], summary["false_bind_rate_under_wrong_context"]],
        color=["#0f766e", "#b91c1c"],
    )
    axes[2].set_title("gating effects")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> dict[str, Any]:
    validate_configuration()
    started = utc_now_iso()
    t0 = time.time()
    rng = build_rng(SEED)
    output_dir = output_dir_for(SCRIPT_PATH)

    rows = evaluate_trials(rng, TRIALS)

    figure_path = output_dir / "contextual_gate_routing.png"
    build_figure(rows, figure_path)

    summary = build_summary(rows)
    statistics = build_statistics(rows)

    finished = utc_now_iso()
    duration = time.time() - t0
    metrics_path = output_dir / "contextual_gate_routing_metrics.json"
    record = build_run_record(
        simulation_name="contextual_gate_routing",
        script_path=SCRIPT_PATH,
        started_at_utc=started,
        finished_at_utc=finished,
        duration_sec=duration,
        parameters={
            "trials": TRIALS,
            "n_cues": N_CUES,
            "n_contexts": N_CONTEXTS,
            "action_count": ACTION_COUNT,
            "contextual_fraction": CONTEXTUAL_FRACTION,
        },
        seed_numpy=SEED,
        n_trials=len(rows),
        summary=summary,
        statistics=statistics,
        trials=rows,
        artifacts=[
            {"name": "contextual_gate_routing.png", "path": figure_path, "type": "figure"},
            {"name": "contextual_gate_routing_metrics.json", "path": metrics_path, "type": "metrics"},
        ],
        warnings=[],
    )
    write_json(metrics_path, record)
    return summary


if __name__ == "__main__":
    main()
