import sys
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.append(str(SIM_ROOT))

from shared import apply_plot_style, build_rng, build_run_record, child_rng, child_seed, mean_confidence_interval, paired_difference_stats, utc_now_iso, write_json


def generate_random_patterns(n_neurons, n_patterns, rng):
    return rng.choice([-1, 1], size=(n_patterns, n_neurons))


def compute_weights(patterns):
    n_neurons = patterns.shape[1]
    weights = np.zeros((n_neurons, n_neurons), dtype=float)
    for pattern in patterns:
        weights += np.outer(pattern, pattern)
    weights /= float(n_neurons)
    np.fill_diagonal(weights, 0.0)
    return weights


def shuffled_control_weights(weights, rng):
    upper = np.triu_indices_from(weights, k=1)
    values = weights[upper].copy()
    rng.shuffle(values)
    shuffled = np.zeros_like(weights)
    shuffled[upper] = values
    shuffled[(upper[1], upper[0])] = values
    return shuffled


def corrupt_pattern(pattern, noise_level, rng):
    corrupted = pattern.copy()
    n_flip = int(round(noise_level * len(pattern)))
    if n_flip > 0:
        flip_indices = rng.choice(len(pattern), size=n_flip, replace=False)
        corrupted[flip_indices] *= -1
    return corrupted


def hopfield_update(state, weights, rng, max_iters=100):
    state = state.copy()
    states_over_time = [state.copy()]
    for _ in range(max_iters):
        previous = state.copy()
        for index in rng.permutation(len(state)):
            field = float(np.dot(weights[index], state))
            state[index] = 1 if field >= 0 else -1
        states_over_time.append(state.copy())
        if np.array_equal(state, previous):
            break
    return states_over_time


def overlap(state, pattern):
    return float(np.dot(state, pattern) / len(pattern))


def retrieval_metrics(final_state, target_pattern, steps):
    bit_error_rate = float(np.mean(final_state != target_pattern))
    return {
        "overlap": overlap(final_state, target_pattern),
        "bit_error_rate": bit_error_rate,
        "exact_retrieval": float(bit_error_rate == 0.0),
        "convergence_steps": int(steps),
    }


def run_condition(cue, target_pattern, weights, update_seed):
    trajectory = hopfield_update(cue, weights, np.random.default_rng(update_seed))
    final_state = trajectory[-1]
    metrics = retrieval_metrics(final_state, target_pattern, len(trajectory) - 1)
    return trajectory, final_state, metrics


def summarize_metric(records, experiment_name, condition_name, metric_name, x_key, x_values):
    summaries = []
    for x_value in x_values:
        values = [
            record["metrics"][metric_name]
            for record in records
            if record["experiment"] == experiment_name
            and record["condition"] == condition_name
            and np.isclose(record[x_key], x_value)
        ]
        summary = mean_confidence_interval(values)
        summary.update({x_key: float(x_value), "metric": metric_name, "condition": condition_name})
        summaries.append(summary)
    return summaries


def summarize_scaling(records, network_sizes):
    summaries = {}
    for n_size in network_sizes:
        size_records = [record for record in records if record["n_neurons"] == int(n_size)]
        load_values = sorted({float(record["load_fraction"]) for record in size_records})
        size_summaries = []
        for load_value in load_values:
            values = [
                record["metrics"]["exact_retrieval"]
                for record in size_records
                if np.isclose(record["load_fraction"], load_value)
            ]
            summary = mean_confidence_interval(values)
            summary.update(
                {
                    "load_fraction": float(load_value),
                    "metric": "exact_retrieval",
                    "condition": "hebbian",
                    "n_neurons": int(n_size),
                }
            )
            size_summaries.append(summary)
        summaries[int(n_size)] = size_summaries
    return summaries


def run_corruption_sweep(n_neurons, n_patterns, noise_levels, n_trials, seed):
    rng = build_rng(seed)
    records = []
    for trial_id in range(n_trials):
        trial_rng = child_rng(rng)
        patterns = generate_random_patterns(n_neurons, n_patterns, trial_rng)
        hebbian_weights = compute_weights(patterns)
        control_weights = shuffled_control_weights(hebbian_weights, child_rng(trial_rng))
        target_index = int(trial_rng.integers(0, n_patterns))
        target_pattern = patterns[target_index]
        for noise_level in noise_levels:
            cue = corrupt_pattern(target_pattern, noise_level, child_rng(trial_rng))
            update_seed = child_seed(trial_rng)
            for condition_name, weights in (("hebbian", hebbian_weights), ("shuffled_control", control_weights)):
                _, _, metrics = run_condition(cue, target_pattern, weights, update_seed)
                records.append(
                    {
                        "experiment": "corruption_sweep",
                        "trial_id": trial_id,
                        "condition": condition_name,
                        "noise_level": float(noise_level),
                        "n_neurons": int(n_neurons),
                        "n_patterns": int(n_patterns),
                        "metrics": metrics,
                    }
                )
    return records


def run_capacity_sweep(n_neurons, pattern_counts, noise_level, n_trials, seed):
    rng = build_rng(seed)
    records = []
    for trial_id in range(n_trials):
        trial_rng = child_rng(rng)
        for n_patterns in pattern_counts:
            patterns = generate_random_patterns(n_neurons, int(n_patterns), child_rng(trial_rng))
            hebbian_weights = compute_weights(patterns)
            target_index = int(trial_rng.integers(0, int(n_patterns)))
            target_pattern = patterns[target_index]
            cue = corrupt_pattern(target_pattern, noise_level, child_rng(trial_rng))
            _, _, metrics = run_condition(cue, target_pattern, hebbian_weights, child_seed(trial_rng))
            records.append(
                {
                    "experiment": "capacity_sweep",
                    "trial_id": trial_id,
                    "condition": "hebbian",
                    "noise_level": float(noise_level),
                    "n_neurons": int(n_neurons),
                    "n_patterns": int(n_patterns),
                    "load_fraction": float(n_patterns / n_neurons),
                    "metrics": metrics,
                }
            )
    return records


def run_scaling_sweep(network_sizes, load_factors, noise_level, n_trials, seed):
    rng = build_rng(seed)
    records = []
    for trial_id in range(n_trials):
        trial_rng = child_rng(rng)
        for n_neurons in network_sizes:
            for load_factor in load_factors:
                n_patterns = max(1, int(round(load_factor * n_neurons)))
                patterns = generate_random_patterns(int(n_neurons), n_patterns, child_rng(trial_rng))
                hebbian_weights = compute_weights(patterns)
                target_index = int(trial_rng.integers(0, n_patterns))
                target_pattern = patterns[target_index]
                cue = corrupt_pattern(target_pattern, noise_level, child_rng(trial_rng))
                _, _, metrics = run_condition(cue, target_pattern, hebbian_weights, child_seed(trial_rng))
                records.append(
                    {
                        "experiment": "scaling_sweep",
                        "trial_id": trial_id,
                        "condition": "hebbian",
                        "noise_level": float(noise_level),
                        "n_neurons": int(n_neurons),
                        "n_patterns": int(n_patterns),
                        "load_fraction": float(n_patterns / n_neurons),
                        "metrics": metrics,
                    }
                )
    return records


def build_visual_example(n_neurons, n_patterns, noise_level, seed):
    rng = build_rng(seed)
    patterns = generate_random_patterns(n_neurons, n_patterns, rng)
    hebbian_weights = compute_weights(patterns)
    control_weights = shuffled_control_weights(hebbian_weights, child_rng(rng))
    target_pattern = patterns[0]
    cue = corrupt_pattern(target_pattern, noise_level, child_rng(rng))
    update_seed = child_seed(rng)
    hebbian_trajectory, hebbian_final, _ = run_condition(cue, target_pattern, hebbian_weights, update_seed)
    control_trajectory, control_final, _ = run_condition(cue, target_pattern, control_weights, update_seed)
    return {
        "target": target_pattern,
        "cue": cue,
        "hebbian_final": hebbian_final,
        "control_final": control_final,
        "hebbian_overlap": [overlap(state, target_pattern) for state in hebbian_trajectory],
        "control_overlap": [overlap(state, target_pattern) for state in control_trajectory],
    }


def main():
    apply_plot_style()

    started_at_utc = utc_now_iso()
    started_perf = perf_counter()

    simulation_name = "pattern_completion"
    output_dir = Path(__file__).parent
    figure_path = output_dir / "pattern_completion.png"
    metrics_path = output_dir / "pattern_completion_metrics.json"

    parameters = {
        "seed_numpy": 42,
        "n_neurons": 200,
        "n_patterns": 5,
        "noise_levels": [round(float(level), 2) for level in np.arange(0.0, 0.55, 0.05)],
        "corruption_trials": 30,
        "capacity_pattern_counts": [2, 8, 14, 20, 28, 36, 44, 52, 60],
        "capacity_trials": 20,
        "capacity_noise_level": 0.1,
        "scaling_network_sizes": [100, 200, 300],
        "scaling_load_factors": [0.05, 0.1, 0.138, 0.18, 0.22],
        "scaling_trials": 12,
        "scaling_noise_level": 0.1,
        "max_iters": 100,
    }

    n_neurons = parameters["n_neurons"]
    n_patterns = parameters["n_patterns"]
    noise_levels = np.array(parameters["noise_levels"])
    pattern_counts = np.array(parameters["capacity_pattern_counts"])
    network_sizes = np.array(parameters["scaling_network_sizes"])
    load_factors = np.array(parameters["scaling_load_factors"])

    corruption_records = run_corruption_sweep(
        n_neurons=n_neurons,
        n_patterns=n_patterns,
        noise_levels=noise_levels,
        n_trials=parameters["corruption_trials"],
        seed=parameters["seed_numpy"],
    )
    capacity_records = run_capacity_sweep(
        n_neurons=n_neurons,
        pattern_counts=pattern_counts,
        noise_level=parameters["capacity_noise_level"],
        n_trials=parameters["capacity_trials"],
        seed=parameters["seed_numpy"] + 1,
    )
    scaling_records = run_scaling_sweep(
        network_sizes=network_sizes,
        load_factors=load_factors,
        noise_level=parameters["scaling_noise_level"],
        n_trials=parameters["scaling_trials"],
        seed=parameters["seed_numpy"] + 2,
    )
    visual_example = build_visual_example(
        n_neurons=n_neurons,
        n_patterns=n_patterns,
        noise_level=0.3,
        seed=parameters["seed_numpy"] + 3,
    )

    overlap_hebbian = summarize_metric(corruption_records, "corruption_sweep", "hebbian", "overlap", "noise_level", noise_levels)
    overlap_control = summarize_metric(corruption_records, "corruption_sweep", "shuffled_control", "overlap", "noise_level", noise_levels)
    exact_hebbian = summarize_metric(corruption_records, "corruption_sweep", "hebbian", "exact_retrieval", "noise_level", noise_levels)
    exact_control = summarize_metric(corruption_records, "corruption_sweep", "shuffled_control", "exact_retrieval", "noise_level", noise_levels)
    ber_hebbian = summarize_metric(corruption_records, "corruption_sweep", "hebbian", "bit_error_rate", "noise_level", noise_levels)
    ber_control = summarize_metric(corruption_records, "corruption_sweep", "shuffled_control", "bit_error_rate", "noise_level", noise_levels)
    capacity_exact = summarize_metric(capacity_records, "capacity_sweep", "hebbian", "exact_retrieval", "n_patterns", pattern_counts)

    scaling_exact = summarize_scaling(scaling_records, network_sizes)

    target_noise = 0.3
    hebbian_target_overlap = [
        record["metrics"]["overlap"]
        for record in corruption_records
        if record["condition"] == "hebbian" and np.isclose(record["noise_level"], target_noise)
    ]
    control_target_overlap = [
        record["metrics"]["overlap"]
        for record in corruption_records
        if record["condition"] == "shuffled_control" and np.isclose(record["noise_level"], target_noise)
    ]
    target_noise_stats = paired_difference_stats(control_target_overlap, hebbian_target_overlap, seed=parameters["seed_numpy"] + 4)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    axes[0, 0].plot(visual_example["hebbian_overlap"], color="#111827", label="hebbian")
    axes[0, 0].plot(visual_example["control_overlap"], color="#d97706", label="shuffled control")
    axes[0, 0].axhline(1.0, color="#9ca3af", linestyle="--", linewidth=1)
    axes[0, 0].set_xlabel("update step")
    axes[0, 0].set_ylabel("overlap with target")
    axes[0, 0].set_title("convergence at 30% corruption")
    axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].legend(loc="lower right")

    axes[0, 1].plot(noise_levels * 100, [entry["mean"] for entry in overlap_hebbian], color="#111827", marker="o", label="hebbian")
    axes[0, 1].fill_between(
        noise_levels * 100,
        [entry["ci95"]["low"] for entry in overlap_hebbian],
        [entry["ci95"]["high"] for entry in overlap_hebbian],
        color="#111827",
        alpha=0.12,
    )
    axes[0, 1].plot(noise_levels * 100, [entry["mean"] for entry in overlap_control], color="#d97706", marker="s", label="shuffled control")
    axes[0, 1].fill_between(
        noise_levels * 100,
        [entry["ci95"]["low"] for entry in overlap_control],
        [entry["ci95"]["high"] for entry in overlap_control],
        color="#d97706",
        alpha=0.12,
    )
    axes[0, 1].set_xlabel("corruption level (%)")
    axes[0, 1].set_ylabel("mean overlap")
    axes[0, 1].set_title("retrieval overlap with control")
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].legend(loc="lower left")

    axes[0, 2].plot(noise_levels * 100, [entry["mean"] for entry in exact_hebbian], color="#111827", marker="o", label="hebbian")
    axes[0, 2].fill_between(
        noise_levels * 100,
        [entry["ci95"]["low"] for entry in exact_hebbian],
        [entry["ci95"]["high"] for entry in exact_hebbian],
        color="#111827",
        alpha=0.12,
    )
    axes[0, 2].plot(noise_levels * 100, [entry["mean"] for entry in exact_control], color="#d97706", marker="s", label="shuffled control")
    axes[0, 2].fill_between(
        noise_levels * 100,
        [entry["ci95"]["low"] for entry in exact_control],
        [entry["ci95"]["high"] for entry in exact_control],
        color="#d97706",
        alpha=0.12,
    )
    axes[0, 2].set_xlabel("corruption level (%)")
    axes[0, 2].set_ylabel("exact retrieval rate")
    axes[0, 2].set_title("exact recovery probability")
    axes[0, 2].set_ylim(-0.05, 1.05)

    axes[1, 0].plot(pattern_counts, [entry["mean"] for entry in capacity_exact], color="#111827", marker="o")
    axes[1, 0].fill_between(
        pattern_counts,
        [entry["ci95"]["low"] for entry in capacity_exact],
        [entry["ci95"]["high"] for entry in capacity_exact],
        color="#111827",
        alpha=0.12,
    )
    axes[1, 0].axvline(0.138 * n_neurons, color="#b91c1c", linestyle="--", linewidth=1.5, label=f"0.138n = {0.138 * n_neurons:.1f}")
    axes[1, 0].set_xlabel("stored patterns")
    axes[1, 0].set_ylabel("exact retrieval rate")
    axes[1, 0].set_title("n=200 load sweep at 10% corruption")
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].legend(loc="upper right")

    color_map = {100: "#2563eb", 200: "#111827", 300: "#059669"}
    for n_size in network_sizes:
        summaries = scaling_exact[int(n_size)]
        load_positions = [entry["load_fraction"] for entry in summaries]
        axes[1, 1].plot(load_positions, [entry["mean"] for entry in summaries], marker="o", color=color_map[int(n_size)], label=f"n={int(n_size)}")
        axes[1, 1].fill_between(
            load_positions,
            [entry["ci95"]["low"] for entry in summaries],
            [entry["ci95"]["high"] for entry in summaries],
            color=color_map[int(n_size)],
            alpha=0.1,
        )
    axes[1, 1].axvline(0.138, color="#b91c1c", linestyle="--", linewidth=1.5, label="theoretical load")
    axes[1, 1].set_xlabel("load fraction (p / n)")
    axes[1, 1].set_ylabel("exact retrieval rate")
    axes[1, 1].set_title("scaling baseline across network size")
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].legend(loc="upper right")

    display_width = 20
    display_height = 10
    combined = np.concatenate(
        [
            visual_example["target"][: display_width * display_height].reshape(display_height, display_width),
            visual_example["cue"][: display_width * display_height].reshape(display_height, display_width),
            visual_example["hebbian_final"][: display_width * display_height].reshape(display_height, display_width),
            visual_example["control_final"][: display_width * display_height].reshape(display_height, display_width),
        ],
        axis=1,
    )
    axes[1, 2].imshow(combined, cmap="binary", aspect="equal", interpolation="nearest")
    axes[1, 2].set_title("target | cue | hebbian | control")
    axes[1, 2].set_xticks([display_width * 0.5, display_width * 1.5, display_width * 2.5, display_width * 3.5])
    axes[1, 2].set_xticklabels(["target", "cue", "hebbian", "control"])
    axes[1, 2].set_yticks([])
    axes[1, 2].grid(False)

    fig.suptitle("ca3-like attractor baseline: pattern completion with control and scaling", fontsize=15, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    finished_at_utc = utc_now_iso()
    duration_sec = round(perf_counter() - started_perf, 3)

    summary = {
        "hebbian_overlap_at_30pct_corruption": mean_confidence_interval(hebbian_target_overlap),
        "control_overlap_at_30pct_corruption": mean_confidence_interval(control_target_overlap),
        "paired_overlap_difference_at_30pct_corruption": target_noise_stats,
        "hebbian_exact_retrieval_curve": exact_hebbian,
        "control_exact_retrieval_curve": exact_control,
        "hebbian_bit_error_curve": ber_hebbian,
        "control_bit_error_curve": ber_control,
    }

    statistics = {
        "corruption_overlap": {"hebbian": overlap_hebbian, "shuffled_control": overlap_control},
        "corruption_exact_retrieval": {"hebbian": exact_hebbian, "shuffled_control": exact_control},
        "capacity_exact_retrieval": capacity_exact,
        "scaling_exact_retrieval": scaling_exact,
    }

    warnings = [
        "this simulation is a ca3-like attractor baseline, not a hippocampal system model.",
        "pattern separation is not modeled here and kda equivalence should not be inferred from these results.",
        "the theoretical 0.138n line is shown for context; this run estimates exact retrieval under a specific corruption protocol rather than proving the asymptotic limit.",
    ]

    all_trials = corruption_records + capacity_records + scaling_records
    payload = build_run_record(
        simulation_name=simulation_name,
        script_path=Path(__file__).resolve(),
        started_at_utc=started_at_utc,
        finished_at_utc=finished_at_utc,
        duration_sec=duration_sec,
        parameters=parameters,
        seed_numpy=parameters["seed_numpy"],
        n_trials=len(all_trials),
        summary=summary,
        statistics=statistics,
        trials=all_trials,
        artifacts=[
            {"type": "png", "path": figure_path.name},
            {"type": "json", "path": metrics_path.name},
        ],
        warnings=warnings,
        extra_configuration={"control_conditions": ["hebbian", "shuffled_control"]},
    )
    write_json(metrics_path, payload)

    target_summary = summary["hebbian_overlap_at_30pct_corruption"]
    print(f"saved: {figure_path}")
    print(f"saved: {metrics_path}")
    print(f"hebbian overlap at 30% corruption: {target_summary['mean']:.3f} [{target_summary['ci95']['low']:.3f}, {target_summary['ci95']['high']:.3f}]")
    print(f"paired overlap delta vs control at 30% corruption: {target_noise_stats['mean_difference']:.3f}, p={target_noise_stats['p_value_permutation']:.4f}")


if __name__ == "__main__":
    main()
