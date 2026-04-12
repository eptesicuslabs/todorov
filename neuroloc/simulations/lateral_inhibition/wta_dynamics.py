import sys
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from brian2 import Network, NetworkOperation, NeuronGroup, SpikeMonitor, StateMonitor, defaultclock, ms, prefs, start_scope, volt

SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from shared import apply_plot_style, build_rng, build_run_record, child_rng, env_float, env_int, env_list, mean_confidence_interval, output_dir_for, paired_difference_stats, require_non_negative, require_non_negative_list, require_positive, require_positive_list, require_unit_interval, require_unit_interval_list, utc_now_iso, write_json

prefs.codegen.target = "numpy"
defaultclock.dt = 0.1 * ms


def generate_trial_activations(n_neurons, target_fraction, support_boost, leader_bonus, noise_sigma, rng):
    support_count = max(1, int(round(target_fraction * n_neurons)))
    leader_idx = int(rng.integers(0, n_neurons))
    support_mask = np.zeros(n_neurons, dtype=bool)
    support_mask[leader_idx] = True
    if support_count > 1:
        pool = np.delete(np.arange(n_neurons), leader_idx)
        extra = rng.choice(pool, size=support_count - 1, replace=False)
        support_mask[extra] = True
    activations = noise_sigma * rng.standard_normal(n_neurons)
    activations[support_mask] += support_boost
    activations[leader_idx] += leader_bonus
    return activations, support_mask, leader_idx


def threshold_mask(activations, alpha):
    cutoff = float(alpha * np.mean(np.abs(activations)))
    return np.abs(activations) > cutoff, cutoff


def kwta_mask(activations, target_fraction, rng):
    k = max(1, int(round(target_fraction * len(activations))))
    ranking = np.abs(activations) + 1e-9 * rng.standard_normal(len(activations))
    selected = np.zeros(len(activations), dtype=bool)
    selected[np.argpartition(ranking, -k)[-k:]] = True
    return selected, k


def selection_metrics(selected_mask, support_mask, leader_idx):
    hits = int(np.sum(selected_mask & support_mask))
    selected_count = int(np.sum(selected_mask))
    support_count = int(np.sum(support_mask))
    precision = float(hits / selected_count) if selected_count else 0.0
    recall = float(hits / support_count) if support_count else 0.0
    f1 = float(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    union = int(np.sum(selected_mask | support_mask))
    jaccard = float(hits / union) if union else 1.0
    return {
        "leader_selected": float(selected_mask[leader_idx]),
        "support_precision": precision,
        "support_recall": recall,
        "support_f1": f1,
        "support_jaccard": jaccard,
        "exact_support": float(np.array_equal(selected_mask, support_mask)),
        "active_fraction": float(selected_count / len(selected_mask)),
        "active_count": selected_count,
        "support_count": support_count,
    }


def calibrate_alpha(n_neurons, target_fraction, support_boost, leader_bonus, calibration_noise_level, calibration_trials, seed):
    rng = build_rng(seed)
    calibration_inputs = [
        generate_trial_activations(
            n_neurons=n_neurons,
            target_fraction=target_fraction,
            support_boost=support_boost,
            leader_bonus=leader_bonus,
            noise_sigma=calibration_noise_level,
            rng=child_rng(rng),
        )[0]
        for _ in range(calibration_trials)
    ]
    low = 0.0
    high = 8.0
    for _ in range(48):
        mid = 0.5 * (low + high)
        mean_active = float(np.mean([np.mean(threshold_mask(activations, mid)[0]) for activations in calibration_inputs]))
        if mean_active > target_fraction:
            low = mid
        else:
            high = mid
    alpha = 0.5 * (low + high)
    matched_fraction = float(np.mean([np.mean(threshold_mask(activations, alpha)[0]) for activations in calibration_inputs]))
    return alpha, matched_fraction


def filter_records(records, condition, metric_name, **filters):
    values = []
    for record in records:
        if record["condition"] != condition:
            continue
        matches = True
        for key, expected in filters.items():
            value = record[key]
            if isinstance(expected, float):
                if not np.isclose(value, expected):
                    matches = False
                    break
            else:
                if value != expected:
                    matches = False
                    break
        if matches:
            values.append(record["metrics"][metric_name])
    return values


def summarize_metric(records, metric_name, x_key, x_values, condition, **filters):
    summaries = []
    for x_value in x_values:
        query = dict(filters)
        query[x_key] = float(x_value) if isinstance(x_value, (float, np.floating)) else x_value
        summary = mean_confidence_interval(filter_records(records, condition, metric_name, **query))
        summary.update({x_key: float(x_value), "metric": metric_name, "condition": condition})
        summaries.append(summary)
    return summaries


def resolve_fraction_key(calibration: dict[float, dict[str, float]], target_fraction: float) -> float:
    for key in calibration:
        if np.isclose(float(key), float(target_fraction)):
            return float(key)
    raise KeyError(f"no calibrated fraction found for {target_fraction}")


def run_selection_sweep(n_neurons, target_fractions, noise_levels, trials_per_condition, support_boost, leader_bonus, calibration_noise_level, calibration_trials, seed):
    rng = build_rng(seed)
    records = []
    threshold_calibration = {}
    for target_fraction in target_fractions:
        alpha, matched_fraction = calibrate_alpha(
            n_neurons=n_neurons,
            target_fraction=target_fraction,
            support_boost=support_boost,
            leader_bonus=leader_bonus,
            calibration_noise_level=calibration_noise_level,
            calibration_trials=calibration_trials,
            seed=seed + int(round(target_fraction * 1000)),
        )
        threshold_calibration[float(target_fraction)] = {"alpha": float(alpha), "matched_fraction": float(matched_fraction)}
    for target_fraction in target_fractions:
        alpha = threshold_calibration[float(target_fraction)]["alpha"]
        for noise_level in noise_levels:
            for trial_id in range(trials_per_condition):
                trial_rng = child_rng(rng)
                activations, support_mask, leader_idx = generate_trial_activations(
                    n_neurons=n_neurons,
                    target_fraction=target_fraction,
                    support_boost=support_boost,
                    leader_bonus=leader_bonus,
                    noise_sigma=noise_level,
                    rng=trial_rng,
                )
                threshold_selected, cutoff = threshold_mask(activations, alpha)
                kwta_selected, k_value = kwta_mask(activations, target_fraction, child_rng(trial_rng))
                for condition_name, selected_mask in (("adaptive_threshold", threshold_selected), ("kwta", kwta_selected)):
                    metrics = selection_metrics(selected_mask, support_mask, leader_idx)
                    records.append(
                        {
                            "experiment": "selection_sweep",
                            "trial_id": int(trial_id),
                            "condition": condition_name,
                            "n_neurons": int(n_neurons),
                            "target_fraction": float(target_fraction),
                            "noise_level": float(noise_level),
                            "leader_idx": int(leader_idx),
                            "threshold_alpha": float(alpha),
                            "threshold_cutoff": float(cutoff),
                            "kwta_k": int(k_value),
                            "metrics": metrics,
                        }
                    )
    return records, threshold_calibration


def run_scaling_sweep(network_sizes, target_fractions, noise_level, trials_per_condition, support_boost, leader_bonus, calibration_noise_level, calibration_trials, seed):
    rng = build_rng(seed)
    records = []
    threshold_calibration = {}
    for n_neurons in network_sizes:
        threshold_calibration[int(n_neurons)] = {}
        for target_fraction in target_fractions:
            alpha, matched_fraction = calibrate_alpha(
                n_neurons=int(n_neurons),
                target_fraction=target_fraction,
                support_boost=support_boost,
                leader_bonus=leader_bonus,
                calibration_noise_level=calibration_noise_level,
                calibration_trials=calibration_trials,
                seed=seed + int(n_neurons) + int(round(target_fraction * 1000)),
            )
            threshold_calibration[int(n_neurons)][float(target_fraction)] = {"alpha": float(alpha), "matched_fraction": float(matched_fraction)}
    for n_neurons in network_sizes:
        for target_fraction in target_fractions:
            alpha = threshold_calibration[int(n_neurons)][float(target_fraction)]["alpha"]
            for trial_id in range(trials_per_condition):
                trial_rng = child_rng(rng)
                activations, support_mask, leader_idx = generate_trial_activations(
                    n_neurons=int(n_neurons),
                    target_fraction=target_fraction,
                    support_boost=support_boost,
                    leader_bonus=leader_bonus,
                    noise_sigma=noise_level,
                    rng=trial_rng,
                )
                threshold_selected, cutoff = threshold_mask(activations, alpha)
                kwta_selected, k_value = kwta_mask(activations, target_fraction, child_rng(trial_rng))
                for condition_name, selected_mask in (("adaptive_threshold", threshold_selected), ("kwta", kwta_selected)):
                    metrics = selection_metrics(selected_mask, support_mask, leader_idx)
                    records.append(
                        {
                            "experiment": "scaling_sweep",
                            "trial_id": int(trial_id),
                            "condition": condition_name,
                            "n_neurons": int(n_neurons),
                            "target_fraction": float(target_fraction),
                            "noise_level": float(noise_level),
                            "leader_idx": int(leader_idx),
                            "threshold_alpha": float(alpha),
                            "threshold_cutoff": float(cutoff),
                            "kwta_k": int(k_value),
                            "metrics": metrics,
                        }
                    )
    return records, threshold_calibration


def build_example(n_neurons, target_fraction, support_boost, leader_bonus, noise_sigma, alpha, seed):
    rng = build_rng(seed)
    activations, support_mask, leader_idx = generate_trial_activations(
        n_neurons=n_neurons,
        target_fraction=target_fraction,
        support_boost=support_boost,
        leader_bonus=leader_bonus,
        noise_sigma=noise_sigma,
        rng=rng,
    )
    threshold_selected, cutoff = threshold_mask(activations, alpha)
    kwta_selected, k_value = kwta_mask(activations, target_fraction, child_rng(rng))
    sorted_idx = np.argsort(np.abs(activations))[::-1]
    return {
        "activations": activations,
        "support_mask": support_mask,
        "leader_idx": int(leader_idx),
        "threshold_selected": threshold_selected,
        "kwta_selected": kwta_selected,
        "threshold_cutoff": float(cutoff),
        "kwta_k": int(k_value),
        "sorted_idx": sorted_idx,
    }


def run_brian2_anchor(activations, support_mask, leader_idx, inhibition_strength, sim_duration_ms):
    start_scope()
    n_neurons = len(activations)
    shifted = activations - float(np.min(activations))
    input_currents = np.clip(0.3 + 1.1 * shifted, 0.1, None)

    eqs_excit = '''
    dv/dt = (-v + I_ext - g_I * v_inh) / (10*ms) : volt
    I_ext : volt
    v_inh : volt
    g_I : 1
    '''

    excit = NeuronGroup(n_neurons, eqs_excit, threshold='v > 1*volt', reset='v = 0*volt', refractory=cast(Any, '2*ms'), method='euler')
    excit.v = 0 * volt
    excit.I_ext = input_currents * volt
    excit.g_I = inhibition_strength
    excit.v_inh = 0 * volt

    eqs_inhib = '''
    dv_I/dt = (-v_I + I_total) / (5*ms) : volt
    I_total : volt
    '''

    inhib = NeuronGroup(1, eqs_inhib, method='euler')
    inhib.v_I = 0 * volt

    def update_inhibition_step():
        mean_v = np.mean(excit.v[:] / volt)
        inhib.I_total = mean_v * volt
        excit.v_inh = inhib.v_I[0]

    update_inhibition = NetworkOperation(update_inhibition_step, dt=defaultclock.dt)

    support_indices = np.flatnonzero(support_mask)
    top_support = support_indices[np.argsort(activations[support_indices])[::-1][: min(3, len(support_indices))]]
    competitor_indices = np.flatnonzero(~support_mask)
    top_competitors = competitor_indices[np.argsort(activations[competitor_indices])[::-1][: min(3, len(competitor_indices))]]
    record_indices = list(dict.fromkeys([int(leader_idx), *top_support.tolist(), *top_competitors.tolist()]))

    spike_monitor = SpikeMonitor(excit)
    state_monitor = StateMonitor(excit, 'v', record=record_indices, dt=0.5 * ms)
    net = Network(excit, inhib, spike_monitor, state_monitor, update_inhibition)
    net.run(sim_duration_ms * ms)

    late_window_ms = min(float(sim_duration_ms), 50.0)
    late_window_start = max(0.0, float(sim_duration_ms) - late_window_ms)
    late_window_duration_sec = max(late_window_ms / 1000.0, 1e-6)
    late_rates = np.zeros(n_neurons, dtype=float)
    for idx in range(n_neurons):
        spike_times = spike_monitor.spike_trains()[idx] / ms
        late_rates[idx] = float(np.sum(spike_times > late_window_start) / late_window_duration_sec)

    traces = state_monitor.v / volt
    return {
        "input_currents": input_currents,
        "record_indices": record_indices,
        "time_ms": state_monitor.t / ms,
        "traces": traces,
        "late_rates": late_rates,
    }


def main():
    apply_plot_style()

    started_at_utc = utc_now_iso()
    started_perf = perf_counter()

    simulation_name = "wta_dynamics"
    output_dir = output_dir_for(Path(__file__))
    figure_path = output_dir / "wta_dynamics.png"
    anchor_path = output_dir / "wta_dynamics_anchor.png"
    metrics_path = output_dir / "wta_dynamics_metrics.json"

    parameters = {
        "seed_numpy": env_int("WTA_SEED", 42),
        "n_neurons": env_int("WTA_N_NEURONS", 100),
        "support_boost": env_float("WTA_SUPPORT_BOOST", 1.0),
        "leader_bonus": env_float("WTA_LEADER_BONUS", 0.35),
        "target_fractions": env_list("WTA_TARGET_FRACTIONS", float, [0.05, 0.1, 0.2, 0.41]),
        "noise_levels": env_list("WTA_NOISE_LEVELS", float, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        "focus_fraction": env_float("WTA_FOCUS_FRACTION", 0.1),
        "recovery_noise_level": env_float("WTA_RECOVERY_NOISE_LEVEL", 0.2),
        "focus_noise_level": env_float("WTA_FOCUS_NOISE_LEVEL", 0.6),
        "selection_trials": env_int("WTA_SELECTION_TRIALS", 40),
        "calibration_noise_level": env_float("WTA_CALIBRATION_NOISE_LEVEL", 0.4),
        "calibration_trials": env_int("WTA_CALIBRATION_TRIALS", 256),
        "scaling_network_sizes": env_list("WTA_SCALING_NETWORK_SIZES", int, [50, 100, 200, 400]),
        "scaling_target_fractions": env_list("WTA_SCALING_TARGET_FRACTIONS", float, [0.1, 0.41]),
        "scaling_noise_level": env_float("WTA_SCALING_NOISE_LEVEL", 0.6),
        "scaling_trials": env_int("WTA_SCALING_TRIALS", 20),
        "anchor_target_fraction": env_float("WTA_ANCHOR_TARGET_FRACTION", 0.1),
        "anchor_noise_level": env_float("WTA_ANCHOR_NOISE_LEVEL", 0.6),
        "anchor_inhibition_strength": env_float("WTA_ANCHOR_INHIBITION_STRENGTH", 5.0),
        "anchor_duration_ms": env_float("WTA_ANCHOR_DURATION_MS", 150.0),
    }

    for required_fraction in (parameters["focus_fraction"], parameters["anchor_target_fraction"]):
        if not any(np.isclose(required_fraction, value) for value in parameters["target_fractions"]):
            parameters["target_fractions"] = sorted([*parameters["target_fractions"], float(required_fraction)])
    for required_noise in (parameters["recovery_noise_level"], parameters["focus_noise_level"], parameters["anchor_noise_level"]):
        if not any(np.isclose(required_noise, value) for value in parameters["noise_levels"]):
            parameters["noise_levels"] = sorted([*parameters["noise_levels"], float(required_noise)])

    require_positive("WTA_N_NEURONS", parameters["n_neurons"])
    parameters["target_fractions"] = require_unit_interval_list("WTA_TARGET_FRACTIONS", parameters["target_fractions"])
    parameters["noise_levels"] = require_non_negative_list("WTA_NOISE_LEVELS", parameters["noise_levels"])
    require_unit_interval("WTA_FOCUS_FRACTION", parameters["focus_fraction"])
    require_non_negative("WTA_RECOVERY_NOISE_LEVEL", parameters["recovery_noise_level"])
    require_non_negative("WTA_FOCUS_NOISE_LEVEL", parameters["focus_noise_level"])
    require_positive("WTA_SELECTION_TRIALS", parameters["selection_trials"])
    require_non_negative("WTA_CALIBRATION_NOISE_LEVEL", parameters["calibration_noise_level"])
    require_positive("WTA_CALIBRATION_TRIALS", parameters["calibration_trials"])
    parameters["scaling_network_sizes"] = require_positive_list("WTA_SCALING_NETWORK_SIZES", parameters["scaling_network_sizes"])
    parameters["scaling_target_fractions"] = require_unit_interval_list("WTA_SCALING_TARGET_FRACTIONS", parameters["scaling_target_fractions"])
    require_non_negative("WTA_SCALING_NOISE_LEVEL", parameters["scaling_noise_level"])
    require_positive("WTA_SCALING_TRIALS", parameters["scaling_trials"])
    require_unit_interval("WTA_ANCHOR_TARGET_FRACTION", parameters["anchor_target_fraction"])
    require_non_negative("WTA_ANCHOR_NOISE_LEVEL", parameters["anchor_noise_level"])
    require_positive("WTA_ANCHOR_INHIBITION_STRENGTH", parameters["anchor_inhibition_strength"])
    require_positive("WTA_ANCHOR_DURATION_MS", parameters["anchor_duration_ms"])

    target_fractions = np.array(parameters["target_fractions"], dtype=float)
    noise_levels = np.array(parameters["noise_levels"], dtype=float)
    network_sizes = np.array(parameters["scaling_network_sizes"], dtype=int)
    scaling_target_fractions = np.array(parameters["scaling_target_fractions"], dtype=float)

    selection_records, selection_calibration = run_selection_sweep(
        n_neurons=parameters["n_neurons"],
        target_fractions=target_fractions,
        noise_levels=noise_levels,
        trials_per_condition=parameters["selection_trials"],
        support_boost=parameters["support_boost"],
        leader_bonus=parameters["leader_bonus"],
        calibration_noise_level=parameters["calibration_noise_level"],
        calibration_trials=parameters["calibration_trials"],
        seed=parameters["seed_numpy"],
    )

    scaling_records, scaling_calibration = run_scaling_sweep(
        network_sizes=network_sizes,
        target_fractions=scaling_target_fractions,
        noise_level=parameters["scaling_noise_level"],
        trials_per_condition=parameters["scaling_trials"],
        support_boost=parameters["support_boost"],
        leader_bonus=parameters["leader_bonus"],
        calibration_noise_level=parameters["calibration_noise_level"],
        calibration_trials=parameters["calibration_trials"],
        seed=parameters["seed_numpy"] + 1,
    )

    focus_fraction = resolve_fraction_key(selection_calibration, float(parameters["focus_fraction"]))
    recovery_noise = float(parameters["recovery_noise_level"])
    focus_noise = float(parameters["focus_noise_level"])
    focus_alpha = selection_calibration[focus_fraction]["alpha"]
    anchor_fraction = resolve_fraction_key(selection_calibration, float(parameters["anchor_target_fraction"]))
    anchor_alpha = selection_calibration[anchor_fraction]["alpha"]
    example = build_example(
        n_neurons=parameters["n_neurons"],
        target_fraction=anchor_fraction,
        support_boost=parameters["support_boost"],
        leader_bonus=parameters["leader_bonus"],
        noise_sigma=float(parameters["anchor_noise_level"]),
        alpha=anchor_alpha,
        seed=parameters["seed_numpy"] + 2,
    )
    brian2_anchor = run_brian2_anchor(
        activations=example["activations"],
        support_mask=example["support_mask"],
        leader_idx=example["leader_idx"],
        inhibition_strength=parameters["anchor_inhibition_strength"],
        sim_duration_ms=parameters["anchor_duration_ms"],
    )

    exact_by_noise_threshold = summarize_metric(
        selection_records,
        metric_name="exact_support",
        x_key="noise_level",
        x_values=noise_levels,
        condition="adaptive_threshold",
        target_fraction=focus_fraction,
    )
    exact_by_noise_kwta = summarize_metric(
        selection_records,
        metric_name="exact_support",
        x_key="noise_level",
        x_values=noise_levels,
        condition="kwta",
        target_fraction=focus_fraction,
    )
    f1_by_fraction_threshold = summarize_metric(
        selection_records,
        metric_name="support_f1",
        x_key="target_fraction",
        x_values=target_fractions,
        condition="adaptive_threshold",
        noise_level=focus_noise,
    )
    f1_by_fraction_kwta = summarize_metric(
        selection_records,
        metric_name="support_f1",
        x_key="target_fraction",
        x_values=target_fractions,
        condition="kwta",
        noise_level=focus_noise,
    )
    active_by_fraction_threshold = summarize_metric(
        selection_records,
        metric_name="active_fraction",
        x_key="target_fraction",
        x_values=target_fractions,
        condition="adaptive_threshold",
        noise_level=focus_noise,
    )
    active_by_fraction_kwta = summarize_metric(
        selection_records,
        metric_name="active_fraction",
        x_key="target_fraction",
        x_values=target_fractions,
        condition="kwta",
        noise_level=focus_noise,
    )

    scaling_summaries = {}
    for target_fraction in scaling_target_fractions:
        scaling_summaries[float(target_fraction)] = {
            "adaptive_threshold": summarize_metric(
                scaling_records,
                metric_name="support_f1",
                x_key="n_neurons",
                x_values=network_sizes,
                condition="adaptive_threshold",
                target_fraction=float(target_fraction),
                noise_level=parameters["scaling_noise_level"],
            ),
            "kwta": summarize_metric(
                scaling_records,
                metric_name="support_f1",
                x_key="n_neurons",
                x_values=network_sizes,
                condition="kwta",
                target_fraction=float(target_fraction),
                noise_level=parameters["scaling_noise_level"],
            ),
        }

    recovery_exact_threshold = filter_records(selection_records, "adaptive_threshold", "exact_support", target_fraction=focus_fraction, noise_level=recovery_noise)
    recovery_exact_kwta = filter_records(selection_records, "kwta", "exact_support", target_fraction=focus_fraction, noise_level=recovery_noise)
    focus_f1_threshold = filter_records(selection_records, "adaptive_threshold", "support_f1", target_fraction=focus_fraction, noise_level=focus_noise)
    focus_f1_kwta = filter_records(selection_records, "kwta", "support_f1", target_fraction=focus_fraction, noise_level=focus_noise)
    focus_active_threshold = filter_records(selection_records, "adaptive_threshold", "active_fraction", target_fraction=focus_fraction, noise_level=focus_noise)
    focus_active_kwta = filter_records(selection_records, "kwta", "active_fraction", target_fraction=focus_fraction, noise_level=focus_noise)

    exact_delta = paired_difference_stats(recovery_exact_threshold, recovery_exact_kwta, seed=parameters["seed_numpy"] + 3)
    f1_delta = paired_difference_stats(focus_f1_threshold, focus_f1_kwta, seed=parameters["seed_numpy"] + 4)
    active_delta = paired_difference_stats(focus_active_threshold, focus_active_kwta, seed=parameters["seed_numpy"] + 5)

    sorted_idx = example["sorted_idx"]
    sorted_abs = np.abs(example["activations"])[sorted_idx]
    support_sorted = example["support_mask"][sorted_idx]
    threshold_sorted = example["threshold_selected"][sorted_idx]
    kwta_sorted = example["kwta_selected"][sorted_idx]

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    axes[0, 0].plot(np.arange(len(sorted_abs)), sorted_abs, color="#111827")
    axes[0, 0].scatter(np.flatnonzero(support_sorted), sorted_abs[support_sorted], color="#b91c1c", s=28, label="true support")
    axes[0, 0].scatter(np.flatnonzero(threshold_sorted), sorted_abs[threshold_sorted], facecolors="none", edgecolors="#d97706", s=42, linewidths=1.2, label="threshold")
    axes[0, 0].scatter(np.flatnonzero(kwta_sorted), sorted_abs[kwta_sorted], marker="x", color="#2563eb", s=28, label="k-wta")
    axes[0, 0].axhline(example["threshold_cutoff"], color="#d97706", linestyle="--", linewidth=1.2)
    axes[0, 0].axvline(example["kwta_k"] - 0.5, color="#2563eb", linestyle=":", linewidth=1.2)
    axes[0, 0].set_xlabel("neuron rank by |activation|")
    axes[0, 0].set_ylabel("|activation|")
    axes[0, 0].set_title("example input and selected neurons")
    axes[0, 0].legend(loc="upper right")

    mask_matrix = np.vstack([support_sorted.astype(float), threshold_sorted.astype(float), kwta_sorted.astype(float)])
    axes[0, 1].imshow(mask_matrix, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    axes[0, 1].set_yticks([0, 1, 2])
    axes[0, 1].set_yticklabels(["ground truth", "threshold", "k-wta"])
    axes[0, 1].set_xlabel("neuron rank by |activation|")
    axes[0, 1].set_title("selection masks on the same trial")
    axes[0, 1].grid(False)

    axes[0, 2].plot(noise_levels, [entry["mean"] for entry in exact_by_noise_threshold], color="#d97706", marker="o", label="threshold")
    axes[0, 2].fill_between(
        noise_levels,
        [entry["ci95"]["low"] for entry in exact_by_noise_threshold],
        [entry["ci95"]["high"] for entry in exact_by_noise_threshold],
        color="#d97706",
        alpha=0.14,
    )
    axes[0, 2].plot(noise_levels, [entry["mean"] for entry in exact_by_noise_kwta], color="#2563eb", marker="s", label="k-wta")
    axes[0, 2].fill_between(
        noise_levels,
        [entry["ci95"]["low"] for entry in exact_by_noise_kwta],
        [entry["ci95"]["high"] for entry in exact_by_noise_kwta],
        color="#2563eb",
        alpha=0.14,
    )
    axes[0, 2].set_xlabel("noise sigma")
    axes[0, 2].set_ylabel("exact support recovery")
    axes[0, 2].set_title(f"matched sparsity at {int(round(focus_fraction * 100))}% support")
    axes[0, 2].set_ylim(-0.05, 1.05)
    axes[0, 2].legend(loc="upper right")

    axes[1, 0].plot(target_fractions * 100, [entry["mean"] for entry in f1_by_fraction_threshold], color="#d97706", marker="o", label="threshold")
    axes[1, 0].fill_between(
        target_fractions * 100,
        [entry["ci95"]["low"] for entry in f1_by_fraction_threshold],
        [entry["ci95"]["high"] for entry in f1_by_fraction_threshold],
        color="#d97706",
        alpha=0.14,
    )
    axes[1, 0].plot(target_fractions * 100, [entry["mean"] for entry in f1_by_fraction_kwta], color="#2563eb", marker="s", label="k-wta")
    axes[1, 0].fill_between(
        target_fractions * 100,
        [entry["ci95"]["low"] for entry in f1_by_fraction_kwta],
        [entry["ci95"]["high"] for entry in f1_by_fraction_kwta],
        color="#2563eb",
        alpha=0.14,
    )
    axes[1, 0].set_xlabel("true support fraction (%)")
    axes[1, 0].set_ylabel("support f1")
    axes[1, 0].set_title(f"support recovery at noise sigma = {focus_noise:.2f}")
    axes[1, 0].set_ylim(-0.05, 1.05)

    axes[1, 1].plot(target_fractions * 100, [entry["mean"] for entry in active_by_fraction_threshold], color="#d97706", marker="o", label="threshold")
    axes[1, 1].fill_between(
        target_fractions * 100,
        [entry["ci95"]["low"] for entry in active_by_fraction_threshold],
        [entry["ci95"]["high"] for entry in active_by_fraction_threshold],
        color="#d97706",
        alpha=0.14,
    )
    axes[1, 1].plot(target_fractions * 100, [entry["mean"] for entry in active_by_fraction_kwta], color="#2563eb", marker="s", label="k-wta")
    axes[1, 1].fill_between(
        target_fractions * 100,
        [entry["ci95"]["low"] for entry in active_by_fraction_kwta],
        [entry["ci95"]["high"] for entry in active_by_fraction_kwta],
        color="#2563eb",
        alpha=0.14,
    )
    axes[1, 1].plot(target_fractions * 100, target_fractions, color="#111827", linestyle="--", linewidth=1.2, label="target")
    axes[1, 1].set_xlabel("target support fraction (%)")
    axes[1, 1].set_ylabel("realized active fraction")
    axes[1, 1].set_title(f"sparsity control at noise sigma = {focus_noise:.2f}")
    axes[1, 1].set_ylim(-0.05, 0.55)
    axes[1, 1].legend(loc="upper left")

    scaling_palette = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(scaling_target_fractions)))
    scaling_colors = {
        float(target_fraction): scaling_palette[idx]
        for idx, target_fraction in enumerate(scaling_target_fractions)
    }
    for target_fraction in scaling_target_fractions:
        threshold_entries = scaling_summaries[float(target_fraction)]["adaptive_threshold"]
        kwta_entries = scaling_summaries[float(target_fraction)]["kwta"]
        color = scaling_colors[float(target_fraction)]
        axes[1, 2].plot(network_sizes, [entry["mean"] for entry in threshold_entries], color=color, linestyle="--", marker="o", label=f"threshold {int(round(target_fraction * 100))}%")
        axes[1, 2].plot(network_sizes, [entry["mean"] for entry in kwta_entries], color=color, linestyle="-", marker="s", label=f"k-wta {int(round(target_fraction * 100))}%")
    axes[1, 2].set_xlabel("network size")
    axes[1, 2].set_ylabel("support f1")
    axes[1, 2].set_title(f"scaling at noise sigma = {parameters['scaling_noise_level']:.2f}")
    axes[1, 2].set_ylim(-0.05, 1.05)
    axes[1, 2].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    support_set = set(np.flatnonzero(example["support_mask"]).tolist())
    for slot, neuron_idx in enumerate(brian2_anchor["record_indices"]):
        if neuron_idx == example["leader_idx"]:
            color = "#b91c1c"
            label = f"leader {neuron_idx}"
            linewidth = 1.8
        elif neuron_idx in support_set:
            color = "#2563eb"
            label = f"support {neuron_idx}"
            linewidth = 1.2
        else:
            color = "#6b7280"
            label = f"competitor {neuron_idx}"
            linewidth = 1.0
        axes2[0].plot(brian2_anchor["time_ms"], brian2_anchor["traces"][slot], color=color, linewidth=linewidth, alpha=0.9, label=label)
    axes2[0].axhline(1.0, color="#111827", linestyle="--", linewidth=1.0, alpha=0.4)
    axes2[0].set_xlabel("time (ms)")
    axes2[0].set_ylabel("membrane potential")
    axes2[0].set_title("Brian2 anchor dynamics")
    axes2[0].legend(loc="upper right", fontsize=8)

    anchor_sorted_idx = np.argsort(brian2_anchor["input_currents"])
    anchor_rates = brian2_anchor["late_rates"][anchor_sorted_idx]
    anchor_support_sorted = example["support_mask"][anchor_sorted_idx]
    bar_colors = np.where(anchor_support_sorted, "#2563eb", "#d1d5db")
    axes2[1].bar(np.arange(len(anchor_rates)), anchor_rates, color=bar_colors, edgecolor="none")
    leader_position = int(np.where(anchor_sorted_idx == example["leader_idx"])[0][0])
    axes2[1].bar(leader_position, anchor_rates[leader_position], color="#b91c1c", edgecolor="none")
    axes2[1].set_xlabel("neuron rank by input current")
    axes2[1].set_ylabel("late firing rate (Hz)")
    axes2[1].set_title("Brian2 anchor late-phase rates")

    plt.tight_layout()
    plt.savefig(anchor_path, bbox_inches="tight")
    plt.close(fig2)

    finished_at_utc = utc_now_iso()
    duration_sec = perf_counter() - started_perf

    summary = {
        "threshold_calibration": selection_calibration,
        "selected_conditions": {
            "focus_fraction": focus_fraction,
            "recovery_noise_level": recovery_noise,
            "focus_noise_level": focus_noise,
            "anchor_target_fraction": anchor_fraction,
            "anchor_noise_level": float(parameters["anchor_noise_level"]),
        },
        "threshold_exact_support_recovery": mean_confidence_interval(recovery_exact_threshold),
        "kwta_exact_support_recovery": mean_confidence_interval(recovery_exact_kwta),
        "paired_exact_support_delta_kwta_minus_threshold": exact_delta,
        "threshold_f1_focus": mean_confidence_interval(focus_f1_threshold),
        "kwta_f1_focus": mean_confidence_interval(focus_f1_kwta),
        "paired_f1_delta_kwta_minus_threshold": f1_delta,
        "threshold_active_fraction_focus": mean_confidence_interval(focus_active_threshold),
        "kwta_active_fraction_focus": mean_confidence_interval(focus_active_kwta),
        "paired_active_fraction_delta_kwta_minus_threshold": active_delta,
    }

    statistics = {
        "exact_support_by_noise": {
            "adaptive_threshold": exact_by_noise_threshold,
            "kwta": exact_by_noise_kwta,
        },
        "support_f1_by_fraction": {
            "adaptive_threshold": f1_by_fraction_threshold,
            "kwta": f1_by_fraction_kwta,
        },
        "active_fraction_by_fraction": {
            "adaptive_threshold": active_by_fraction_threshold,
            "kwta": active_by_fraction_kwta,
        },
        "scaling_support_f1": scaling_summaries,
        "scaling_threshold_calibration": scaling_calibration,
    }

    run_record = build_run_record(
        simulation_name=simulation_name,
        script_path=Path(__file__),
        started_at_utc=started_at_utc,
        finished_at_utc=finished_at_utc,
        duration_sec=duration_sec,
        parameters=parameters,
        seed_numpy=parameters["seed_numpy"],
        n_trials=len(selection_records) + len(scaling_records),
        summary=summary,
        statistics=statistics,
        trials=selection_records + scaling_records,
        artifacts=[
            {"path": figure_path.as_posix(), "type": "figure", "description": "matched-sparsity threshold vs k-wta pilot"},
            {"path": anchor_path.as_posix(), "type": "figure", "description": "Brian2 anchor dynamics"},
            {"path": metrics_path.as_posix(), "type": "metrics_json", "description": "machine-readable trial records and summaries"},
        ],
        warnings=[],
        extra_configuration={"control_conditions": ["adaptive_threshold", "kwta"]},
    )
    write_json(metrics_path, run_record)

    print(f"saved {figure_path.name}, {anchor_path.name}, and {metrics_path.name} to {output_dir}")
    print(
        f"kwta exact support at focus fraction {focus_fraction:.3f}, recovery noise {recovery_noise:.3f}: "
        f"{summary['kwta_exact_support_recovery']['mean']:.3f} "
        f"[{summary['kwta_exact_support_recovery']['ci95']['low']:.3f}, {summary['kwta_exact_support_recovery']['ci95']['high']:.3f}]"
    )
    print(
        f"threshold exact support at focus fraction {focus_fraction:.3f}, recovery noise {recovery_noise:.3f}: "
        f"{summary['threshold_exact_support_recovery']['mean']:.3f} "
        f"[{summary['threshold_exact_support_recovery']['ci95']['low']:.3f}, {summary['threshold_exact_support_recovery']['ci95']['high']:.3f}]"
    )
    print(
        f"paired exact-support delta kwta-threshold at focus fraction {focus_fraction:.3f}, recovery noise {recovery_noise:.3f}: "
        f"{summary['paired_exact_support_delta_kwta_minus_threshold']['mean_difference']:.3f}, "
        f"p={summary['paired_exact_support_delta_kwta_minus_threshold']['p_value_permutation']:.4f}"
    )
    print(
        f"paired active-fraction delta kwta-threshold at focus fraction {focus_fraction:.3f}, focus noise {focus_noise:.3f}: "
        f"{summary['paired_active_fraction_delta_kwta_minus_threshold']['mean_difference']:.3f}, "
        f"p={summary['paired_active_fraction_delta_kwta_minus_threshold']['p_value_permutation']:.4f}"
    )


if __name__ == "__main__":
    main()
