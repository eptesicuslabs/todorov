import sys
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from brian2 import Mohm, NeuronGroup, StateMonitor, TimedArray, defaultclock, mV, ms, nA, prefs, run, start_scope

SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from shared import apply_plot_style, build_rng, build_run_record, child_rng, ensure_close_member, env_float, env_int, env_list, leak_tau_condition_name, mean_confidence_interval, output_dir_for, paired_difference_stats, require_non_negative, require_non_negative_list, require_positive, require_positive_list, utc_now_iso, write_json

prefs.codegen.target = "numpy"


def make_gap_sequence(total_steps, first_start, pulse_width, gap_steps, amplitude_mv):
    sequence = np.zeros(total_steps, dtype=float)
    first_end = first_start + pulse_width
    second_start = first_end + gap_steps
    second_end = second_start + pulse_width
    sequence[first_start:first_end] = amplitude_mv
    sequence[second_start:second_end] = amplitude_mv
    return sequence, first_end, second_start, second_end


def simulate_discrete(inputs_mv, retention, threshold_mv):
    n_trials, total_steps = inputs_mv.shape
    state = np.zeros(n_trials, dtype=float)
    traces = np.zeros((n_trials, total_steps), dtype=float)
    spikes = np.zeros((n_trials, total_steps), dtype=float)
    for step in range(total_steps):
        h = retention * state + inputs_mv[:, step]
        spike_step = np.sign(h) * (np.abs(h) > threshold_mv)
        state = h - spike_step * threshold_mv
        traces[:, step] = state
        spikes[:, step] = spike_step
    return traces, spikes


def simulate_brian2_anchor(inputs_mv, dt_ms, tau_m_ms, resistance_mohm):
    start_scope()
    defaultclock.dt = dt_ms * ms
    stimulus = TimedArray((inputs_mv / resistance_mohm) * nA, dt=dt_ms * ms)
    eqs = "dv/dt = (-v + R * stimulus(t)) / tau_m : volt"
    neuron = NeuronGroup(1, eqs, method="euler", namespace={"stimulus": stimulus, "R": resistance_mohm * Mohm, "tau_m": tau_m_ms * ms})
    neuron.v = 0 * mV
    monitor = StateMonitor(neuron, "v", record=True)
    run(len(inputs_mv) * dt_ms * ms)
    return monitor.t / ms, monitor.v[0] / mV


def fit_decay_tau(trace_mv, start_index, dt_ms):
    tail = np.asarray(trace_mv[start_index:], dtype=float)
    positive = tail > 1e-6
    if int(np.sum(positive)) < 4:
        return None
    times = np.arange(tail.size, dtype=float)[positive] * dt_ms
    values = tail[positive]
    slope, intercept = np.polyfit(times, np.log(values), 1)
    if slope >= 0:
        return None
    return float(-1.0 / slope)


def summarize_tau_sweep(records, sweep_names):
    summaries = []
    for condition_name in sweep_names:
        values = [record["peak_error_mv"] for record in records if record["condition"] == condition_name]
        summary = mean_confidence_interval(values)
        summary.update({"condition": condition_name})
        summaries.append(summary)
    return summaries


def main():
    apply_plot_style()

    started_at_utc = utc_now_iso()
    started_perf = perf_counter()

    simulation_name = "lif_fi_curve"
    output_dir = output_dir_for(Path(__file__))
    figure_path = output_dir / "lif_leak_validation.png"
    metrics_path = output_dir / "lif_leak_validation_metrics.json"

    parameters = {
        "seed_numpy": env_int("LIF_SEED", 17),
        "dt_ms": env_float("LIF_DT_MS", 1.0),
        "threshold_mv": env_float("LIF_THRESHOLD_MV", 15.0),
        "atmn_tau": env_float("LIF_ATMN_TAU", 2.0),
        "leak_tau_ms": env_list("LIF_LEAK_TAU_MS", float, [5.0, 10.0, 20.0, 30.0]),
        "baseline_leak_tau_ms": env_float("LIF_BASELINE_TAU_MS", 20.0),
        "gap_values_ms": env_list("LIF_GAP_VALUES_MS", float, [1.0, 4.0, 8.0, 16.0, 32.0, 64.0]),
        "gap_trials": env_int("LIF_GAP_TRIALS", 32),
        "gap_total_steps": env_int("LIF_GAP_TOTAL_STEPS", 120),
        "pulse_start_ms": env_float("LIF_PULSE_START_MS", 10.0),
        "pulse_width_ms": env_float("LIF_PULSE_WIDTH_MS", 4.0),
        "pulse_amplitude_mv": env_float("LIF_PULSE_AMPLITUDE_MV", 0.1),
        "spike_probe_gap_ms": env_float("LIF_SPIKE_PROBE_GAP_MS", 16.0),
        "spike_probe_pulse_amplitude_mv": env_float("LIF_SPIKE_PROBE_PULSE_AMPLITUDE_MV", 4.0),
        "gap_noise_sigma_mv": env_float("LIF_GAP_NOISE_SIGMA_MV", 0.004),
        "drift_lengths": env_list("LIF_DRIFT_LENGTHS", int, [128, 256, 512, 1024]),
        "drift_trials": env_int("LIF_DRIFT_TRIALS", 48),
        "drift_noise_sigma_mv": env_float("LIF_DRIFT_NOISE_SIGMA_MV", 0.5),
        "anchor_tau_m_ms": env_float("LIF_ANCHOR_TAU_M_MS", 20.0),
        "anchor_resistance_mohm": env_float("LIF_ANCHOR_RESISTANCE_MOHM", 100.0),
        "example_gap_ms": env_float("LIF_EXAMPLE_GAP_MS", 16.0),
    }

    parameters["gap_values_ms"] = ensure_close_member(parameters["gap_values_ms"], float(parameters["example_gap_ms"]))
    parameters["leak_tau_ms"] = ensure_close_member(parameters["leak_tau_ms"], float(parameters["baseline_leak_tau_ms"]))

    require_positive("LIF_DT_MS", parameters["dt_ms"])
    require_positive("LIF_THRESHOLD_MV", parameters["threshold_mv"])
    require_positive("LIF_ATMN_TAU", parameters["atmn_tau"])
    parameters["leak_tau_ms"] = require_positive_list("LIF_LEAK_TAU_MS", parameters["leak_tau_ms"])
    require_positive("LIF_BASELINE_TAU_MS", parameters["baseline_leak_tau_ms"])
    parameters["gap_values_ms"] = require_positive_list("LIF_GAP_VALUES_MS", parameters["gap_values_ms"])
    require_positive("LIF_GAP_TRIALS", parameters["gap_trials"])
    require_positive("LIF_GAP_TOTAL_STEPS", parameters["gap_total_steps"])
    require_non_negative("LIF_PULSE_START_MS", parameters["pulse_start_ms"])
    require_positive("LIF_PULSE_WIDTH_MS", parameters["pulse_width_ms"])
    require_positive("LIF_PULSE_AMPLITUDE_MV", parameters["pulse_amplitude_mv"])
    require_positive("LIF_SPIKE_PROBE_GAP_MS", parameters["spike_probe_gap_ms"])
    require_positive("LIF_SPIKE_PROBE_PULSE_AMPLITUDE_MV", parameters["spike_probe_pulse_amplitude_mv"])
    require_non_negative("LIF_GAP_NOISE_SIGMA_MV", parameters["gap_noise_sigma_mv"])
    parameters["drift_lengths"] = require_positive_list("LIF_DRIFT_LENGTHS", parameters["drift_lengths"])
    require_positive("LIF_DRIFT_TRIALS", parameters["drift_trials"])
    require_non_negative("LIF_DRIFT_NOISE_SIGMA_MV", parameters["drift_noise_sigma_mv"])
    require_positive("LIF_ANCHOR_TAU_M_MS", parameters["anchor_tau_m_ms"])
    require_positive("LIF_ANCHOR_RESISTANCE_MOHM", parameters["anchor_resistance_mohm"])
    require_positive("LIF_EXAMPLE_GAP_MS", parameters["example_gap_ms"])
    parameters["drift_lengths"] = sorted(parameters["drift_lengths"])

    rng = build_rng(parameters["seed_numpy"])
    threshold_mv = parameters["threshold_mv"]
    dt_ms = parameters["dt_ms"]
    pulse_width_steps = int(round(parameters["pulse_width_ms"] / dt_ms))
    pulse_start_steps = int(round(parameters["pulse_start_ms"] / dt_ms))
    if pulse_width_steps < 1:
        raise ValueError("LIF_PULSE_WIDTH_MS must span at least one timestep")
    for gap_ms in parameters["gap_values_ms"]:
        gap_steps = int(round(gap_ms / dt_ms))
        if gap_steps < 1:
            raise ValueError("LIF_GAP_VALUES_MS entries must span at least one timestep")
        final_step = pulse_start_steps + pulse_width_steps + gap_steps + pulse_width_steps
        if final_step > parameters["gap_total_steps"]:
            raise ValueError("Configured pulse sequence does not fit inside LIF_GAP_TOTAL_STEPS")
    spike_probe_gap_steps = int(round(parameters["spike_probe_gap_ms"] / dt_ms))
    if spike_probe_gap_steps < 1:
        raise ValueError("LIF_SPIKE_PROBE_GAP_MS must span at least one timestep")
    spike_probe_final_step = pulse_start_steps + pulse_width_steps + spike_probe_gap_steps + pulse_width_steps
    if spike_probe_final_step > parameters["gap_total_steps"]:
        raise ValueError("Configured LIF spike probe does not fit inside LIF_GAP_TOTAL_STEPS")
    leak_conditions = {
        leak_tau_condition_name(tau_ms): float(np.exp(-dt_ms / tau_ms)) for tau_ms in parameters["leak_tau_ms"]
    }
    explicit_condition_name = leak_tau_condition_name(parameters["baseline_leak_tau_ms"])
    summary_condition_aliases = {
        "explicit_leak": explicit_condition_name,
        "atmn_carry": "atmn_carry_tau2",
        "integrator_control": "integrator_control",
    }
    condition_retention = {
        **leak_conditions,
        "atmn_carry_tau2": float(1.0 / parameters["atmn_tau"]),
        "integrator_control": 1.0,
    }
    plot_conditions = [explicit_condition_name, "atmn_carry_tau2", "integrator_control"]
    tau_plot_conditions = [*leak_conditions.keys(), "atmn_carry_tau2", "integrator_control"]
    condition_colors = {
        explicit_condition_name: "#1f77b4",
        "atmn_carry_tau2": "#d62728",
        "integrator_control": "#2ca02c",
    }

    gap_records = []
    anchor_by_gap = {}
    for gap_ms in parameters["gap_values_ms"]:
        gap_steps = int(round(gap_ms / dt_ms))
        base_sequence, first_end, second_start, second_end = make_gap_sequence(
            total_steps=parameters["gap_total_steps"],
            first_start=pulse_start_steps,
            pulse_width=pulse_width_steps,
            gap_steps=gap_steps,
            amplitude_mv=parameters["pulse_amplitude_mv"],
        )
        anchor_time_ms, anchor_trace_mv = simulate_brian2_anchor(
            inputs_mv=base_sequence,
            dt_ms=dt_ms,
            tau_m_ms=parameters["anchor_tau_m_ms"],
            resistance_mohm=parameters["anchor_resistance_mohm"],
        )
        anchor_first_peak = float(np.max(anchor_trace_mv[pulse_start_steps:first_end]))
        anchor_second_peak = float(np.max(anchor_trace_mv[second_start:second_end]))
        anchor_gap_value = float(anchor_trace_mv[second_start - 1]) if second_start > 0 else 0.0
        anchor_tail_residual = float(np.abs(anchor_trace_mv[-1]))
        anchor_tau_fit = fit_decay_tau(anchor_trace_mv, first_end, dt_ms)
        anchor_by_gap[float(gap_ms)] = {
            "time_ms": anchor_time_ms,
            "trace_mv": anchor_trace_mv,
            "first_peak_mv": anchor_first_peak,
            "second_peak_mv": anchor_second_peak,
            "gap_value_mv": anchor_gap_value,
            "tail_residual_mv": anchor_tail_residual,
            "tau_fit_ms": anchor_tau_fit,
            "first_end_step": int(first_end),
            "second_start_step": int(second_start),
            "second_end_step": int(second_end),
        }
        for trial_id in range(parameters["gap_trials"]):
            trial_rng = child_rng(rng)
            noisy_inputs = np.repeat(base_sequence[None, :], 1, axis=0)
            noisy_inputs += parameters["gap_noise_sigma_mv"] * trial_rng.standard_normal(noisy_inputs.shape)
            for condition_name, retention in condition_retention.items():
                traces_mv, spikes = simulate_discrete(noisy_inputs, retention, threshold_mv)
                trace_mv = traces_mv[0]
                spike_trace = spikes[0]
                first_peak = float(np.max(trace_mv[pulse_start_steps:first_end]))
                second_peak = float(np.max(trace_mv[second_start:second_end]))
                gap_value = float(trace_mv[second_start - 1]) if second_start > 0 else 0.0
                tau_fit = fit_decay_tau(trace_mv, first_end, dt_ms)
                gap_records.append(
                    {
                        "experiment": "gap_sweep",
                        "trial_id": int(trial_id),
                        "condition": condition_name,
                        "gap_ms": float(gap_ms),
                        "retention": float(retention),
                        "first_peak_mv": first_peak,
                        "second_peak_mv": second_peak,
                        "gap_value_mv": gap_value,
                        "peak_error_mv": float(abs(second_peak - anchor_second_peak)),
                        "retained_fraction": float(0.0 if first_peak <= 1e-9 else gap_value / first_peak),
                        "second_spike": float(np.any(np.abs(spike_trace[second_start:second_end]) > 0.0)),
                        "tail_residual_mv": float(np.abs(trace_mv[-1])),
                        "tau_fit_ms": tau_fit,
                    }
                )

    drift_records = []
    for length in parameters["drift_lengths"]:
        for trial_id in range(parameters["drift_trials"]):
            trial_rng = child_rng(rng)
            inputs_mv = parameters["drift_noise_sigma_mv"] * trial_rng.standard_normal((1, int(length)))
            for condition_name, retention in condition_retention.items():
                traces_mv, spikes = simulate_discrete(inputs_mv, retention, threshold_mv)
                trace_mv = traces_mv[0]
                spike_trace = spikes[0]
                times = np.arange(int(length), dtype=float) * dt_ms
                drift_slope = float(np.polyfit(times, trace_mv, 1)[0]) if int(length) > 1 else 0.0
                drift_records.append(
                    {
                        "experiment": "drift_scaling",
                        "trial_id": int(trial_id),
                        "condition": condition_name,
                        "length_steps": int(length),
                        "retention": float(retention),
                        "state_std_mv": float(np.std(trace_mv, ddof=1)) if int(length) > 1 else 0.0,
                        "final_abs_mv": float(np.abs(trace_mv[-1])),
                        "mean_abs_mv": float(np.mean(np.abs(trace_mv))),
                        "drift_slope_mv_per_ms": drift_slope,
                        "spike_rate": float(np.mean(np.abs(spike_trace) > 0.0)),
                    }
                )

    spike_probe_records = []
    spike_probe_sequence, spike_probe_first_end, spike_probe_second_start, spike_probe_second_end = make_gap_sequence(
        total_steps=parameters["gap_total_steps"],
        first_start=pulse_start_steps,
        pulse_width=pulse_width_steps,
        gap_steps=spike_probe_gap_steps,
        amplitude_mv=parameters["spike_probe_pulse_amplitude_mv"],
    )
    for trial_id in range(parameters["gap_trials"]):
        trial_rng = child_rng(rng)
        noisy_inputs = np.repeat(spike_probe_sequence[None, :], 1, axis=0)
        noisy_inputs += parameters["gap_noise_sigma_mv"] * trial_rng.standard_normal(noisy_inputs.shape)
        for condition_name, retention in condition_retention.items():
            traces_mv, spikes = simulate_discrete(noisy_inputs, retention, threshold_mv)
            trace_mv = traces_mv[0]
            spike_trace = spikes[0]
            spike_probe_records.append(
                {
                    "experiment": "spike_probe",
                    "trial_id": int(trial_id),
                    "condition": condition_name,
                    "gap_ms": float(parameters["spike_probe_gap_ms"]),
                    "pulse_amplitude_mv": float(parameters["spike_probe_pulse_amplitude_mv"]),
                    "first_peak_mv": float(np.max(trace_mv[pulse_start_steps:spike_probe_first_end])),
                    "second_peak_mv": float(np.max(trace_mv[spike_probe_second_start:spike_probe_second_end])),
                    "first_spike": float(np.any(np.abs(spike_trace[pulse_start_steps:spike_probe_first_end]) > 0.0)),
                    "second_spike": float(np.any(np.abs(spike_trace[spike_probe_second_start:spike_probe_second_end]) > 0.0)),
                }
            )

    main_gap_curves = {}
    for condition_name in plot_conditions:
        peak_error_curve = []
        retained_curve = []
        spike_curve = []
        for gap_ms in parameters["gap_values_ms"]:
            peak_error_values = [record["peak_error_mv"] for record in gap_records if record["condition"] == condition_name and np.isclose(record["gap_ms"], gap_ms)]
            retained_values = [record["retained_fraction"] for record in gap_records if record["condition"] == condition_name and np.isclose(record["gap_ms"], gap_ms)]
            spike_values = [record["second_spike"] for record in gap_records if record["condition"] == condition_name and np.isclose(record["gap_ms"], gap_ms)]
            peak_error_summary = mean_confidence_interval(peak_error_values)
            retained_summary = mean_confidence_interval(retained_values)
            spike_summary = mean_confidence_interval(spike_values)
            peak_error_summary.update({"gap_ms": float(gap_ms)})
            retained_summary.update({"gap_ms": float(gap_ms)})
            spike_summary.update({"gap_ms": float(gap_ms)})
            peak_error_curve.append(peak_error_summary)
            retained_curve.append(retained_summary)
            spike_curve.append(spike_summary)
        main_gap_curves[condition_name] = {
            "peak_error_mv": peak_error_curve,
            "retained_fraction": retained_curve,
            "second_spike": spike_curve,
        }

    drift_curves = {}
    for condition_name in plot_conditions:
        std_curve = []
        final_curve = []
        spike_curve = []
        for length in parameters["drift_lengths"]:
            std_values = [record["state_std_mv"] for record in drift_records if record["condition"] == condition_name and record["length_steps"] == int(length)]
            final_values = [record["final_abs_mv"] for record in drift_records if record["condition"] == condition_name and record["length_steps"] == int(length)]
            spike_values = [record["spike_rate"] for record in drift_records if record["condition"] == condition_name and record["length_steps"] == int(length)]
            std_summary = mean_confidence_interval(std_values)
            final_summary = mean_confidence_interval(final_values)
            spike_summary = mean_confidence_interval(spike_values)
            std_summary.update({"length_steps": int(length)})
            final_summary.update({"length_steps": int(length)})
            spike_summary.update({"length_steps": int(length)})
            std_curve.append(std_summary)
            final_curve.append(final_summary)
            spike_curve.append(spike_summary)
        drift_curves[condition_name] = {
            "state_std_mv": std_curve,
            "final_abs_mv": final_curve,
            "spike_rate": spike_curve,
        }

    tau_sweep_summary = summarize_tau_sweep(
        gap_records,
        tau_plot_conditions,
    )

    paired_gap_trial_means = {}
    for condition_name in condition_retention:
        values = []
        for trial_id in range(parameters["gap_trials"]):
            trial_values = [record["peak_error_mv"] for record in gap_records if record["condition"] == condition_name and record["trial_id"] == int(trial_id)]
            values.append(float(np.mean(trial_values)))
        paired_gap_trial_means[condition_name] = values

    focal_gap_ms = float(parameters["example_gap_ms"])
    paired_retained_fraction = {}
    for condition_name in condition_retention:
        paired_retained_fraction[condition_name] = [
            record["retained_fraction"]
            for record in gap_records
            if record["condition"] == condition_name and np.isclose(record["gap_ms"], focal_gap_ms)
        ]

    paired_drift_std = {}
    paired_drift_final = {}
    max_length = int(max(parameters["drift_lengths"]))
    for condition_name in condition_retention:
        paired_drift_std[condition_name] = [record["state_std_mv"] for record in drift_records if record["condition"] == condition_name and record["length_steps"] == max_length]
        paired_drift_final[condition_name] = [record["final_abs_mv"] for record in drift_records if record["condition"] == condition_name and record["length_steps"] == max_length]

    paired_spike_probe_first = {}
    paired_spike_probe_second = {}
    for condition_name in condition_retention:
        paired_spike_probe_first[condition_name] = [
            record["first_spike"]
            for record in spike_probe_records
            if record["condition"] == condition_name
        ]
        paired_spike_probe_second[condition_name] = [
            record["second_spike"]
            for record in spike_probe_records
            if record["condition"] == condition_name
        ]

    statistics = {
        "gap_mean_peak_error_explicit_vs_atmn": paired_difference_stats(paired_gap_trial_means["atmn_carry_tau2"], paired_gap_trial_means[explicit_condition_name], seed=301),
        "gap_mean_peak_error_explicit_vs_integrator": paired_difference_stats(paired_gap_trial_means["integrator_control"], paired_gap_trial_means[explicit_condition_name], seed=302),
        "selected_gap_retained_fraction_explicit_vs_atmn": paired_difference_stats(paired_retained_fraction["atmn_carry_tau2"], paired_retained_fraction[explicit_condition_name], seed=306),
        "selected_gap_retained_fraction_explicit_vs_integrator": paired_difference_stats(paired_retained_fraction["integrator_control"], paired_retained_fraction[explicit_condition_name], seed=307),
        "selected_max_length_state_std_explicit_vs_atmn": paired_difference_stats(paired_drift_std["atmn_carry_tau2"], paired_drift_std[explicit_condition_name], seed=303),
        "selected_max_length_state_std_explicit_vs_integrator": paired_difference_stats(paired_drift_std["integrator_control"], paired_drift_std[explicit_condition_name], seed=304),
        "selected_max_length_final_abs_explicit_vs_atmn": paired_difference_stats(paired_drift_final["atmn_carry_tau2"], paired_drift_final[explicit_condition_name], seed=305),
        "selected_max_length_final_abs_explicit_vs_integrator": paired_difference_stats(paired_drift_final["integrator_control"], paired_drift_final[explicit_condition_name], seed=308),
        "spike_probe_first_spike_explicit_vs_atmn": paired_difference_stats(paired_spike_probe_first["atmn_carry_tau2"], paired_spike_probe_first[explicit_condition_name], seed=309),
        "spike_probe_first_spike_explicit_vs_integrator": paired_difference_stats(paired_spike_probe_first["integrator_control"], paired_spike_probe_first[explicit_condition_name], seed=310),
        "spike_probe_second_spike_explicit_vs_atmn": paired_difference_stats(paired_spike_probe_second["atmn_carry_tau2"], paired_spike_probe_second[explicit_condition_name], seed=311),
        "spike_probe_second_spike_explicit_vs_integrator": paired_difference_stats(paired_spike_probe_second["integrator_control"], paired_spike_probe_second[explicit_condition_name], seed=312),
    }

    example_gap = float(parameters["example_gap_ms"])
    example_gap_steps = int(round(example_gap / dt_ms))
    example_sequence, _, example_second_start, example_second_end = make_gap_sequence(
        total_steps=parameters["gap_total_steps"],
        first_start=pulse_start_steps,
        pulse_width=pulse_width_steps,
        gap_steps=example_gap_steps,
        amplitude_mv=parameters["pulse_amplitude_mv"],
    )
    example_inputs = example_sequence[None, :]
    example_traces = {}
    for condition_name in plot_conditions:
        example_traces[condition_name] = simulate_discrete(example_inputs, condition_retention[condition_name], threshold_mv)[0][0]
    example_anchor = anchor_by_gap[example_gap]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(np.arange(example_inputs.shape[1]) * dt_ms, example_sequence, color="#7f7f7f", linestyle="--", linewidth=1.5, label="input")
    axes[0, 0].plot(example_anchor["time_ms"], example_anchor["trace_mv"], color="#000000", linewidth=2.2, label="brian2 lif anchor")
    for condition_name in plot_conditions:
        axes[0, 0].plot(np.arange(example_inputs.shape[1]) * dt_ms, example_traces[condition_name], color=condition_colors[condition_name], label=condition_name.replace("_", " "))
    axes[0, 0].axhline(y=threshold_mv, color="#999999", linestyle=":", linewidth=1.2, label="threshold")
    axes[0, 0].set_title(f"paired-pulse integration at {example_gap:.0f} ms gap")
    axes[0, 0].set_xlabel("time (ms)")
    axes[0, 0].set_ylabel("state (mV rel. rest)")
    axes[0, 0].legend(fontsize=8)

    for condition_name in plot_conditions:
        x_values = [item["gap_ms"] for item in main_gap_curves[condition_name]["peak_error_mv"]]
        y_values = [item["mean"] for item in main_gap_curves[condition_name]["peak_error_mv"]]
        y_low = [item["ci95"]["low"] for item in main_gap_curves[condition_name]["peak_error_mv"]]
        y_high = [item["ci95"]["high"] for item in main_gap_curves[condition_name]["peak_error_mv"]]
        axes[0, 1].plot(x_values, y_values, marker="o", color=condition_colors[condition_name], label=condition_name.replace("_", " "))
        axes[0, 1].fill_between(x_values, y_low, y_high, color=condition_colors[condition_name], alpha=0.15)
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].set_title("second-pulse peak error vs. gap")
    axes[0, 1].set_xlabel("gap (ms)")
    axes[0, 1].set_ylabel("peak error to lif anchor (mV)")
    axes[0, 1].legend(fontsize=8)

    for condition_name in plot_conditions:
        x_values = [item["length_steps"] for item in drift_curves[condition_name]["state_std_mv"]]
        y_values = [item["mean"] for item in drift_curves[condition_name]["state_std_mv"]]
        y_low = [item["ci95"]["low"] for item in drift_curves[condition_name]["state_std_mv"]]
        y_high = [item["ci95"]["high"] for item in drift_curves[condition_name]["state_std_mv"]]
        axes[1, 0].plot(x_values, y_values, marker="o", color=condition_colors[condition_name], label=condition_name.replace("_", " "))
        axes[1, 0].fill_between(x_values, y_low, y_high, color=condition_colors[condition_name], alpha=0.15)
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].set_title("state variance under zero-mean noise")
    axes[1, 0].set_xlabel("sequence length (steps)")
    axes[1, 0].set_ylabel("state std (mV)")
    axes[1, 0].legend(fontsize=8)

    tau_labels = [item["condition"].replace("_", " ") for item in tau_sweep_summary]
    tau_means = [item["mean"] for item in tau_sweep_summary]
    tau_lows = [item["ci95"]["low"] for item in tau_sweep_summary]
    tau_highs = [item["ci95"]["high"] for item in tau_sweep_summary]
    x_positions = np.arange(len(tau_labels))
    axes[1, 1].bar(x_positions, tau_means, color=[condition_colors.get(label.replace(" ", "_"), "#7f7f7f") for label in tau_labels], alpha=0.85)
    axes[1, 1].errorbar(x_positions, tau_means, yerr=[np.array(tau_means) - np.array(tau_lows), np.array(tau_highs) - np.array(tau_means)], fmt="none", ecolor="#222222", capsize=4)
    axes[1, 1].set_xticks(x_positions)
    axes[1, 1].set_xticklabels(tau_labels, rotation=20, ha="right")
    axes[1, 1].set_title("mean gap-sweep peak error")
    axes[1, 1].set_ylabel("peak error to lif anchor (mV)")

    plt.tight_layout()
    plt.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "main_comparison": {
            "explicit_leak_condition": explicit_condition_name,
            "baseline_condition": "atmn_carry_tau2",
            "control_condition": "integrator_control",
            "condition_aliases": summary_condition_aliases,
            "selected_gap_ms": float(focal_gap_ms),
            "max_drift_length_steps": int(max_length),
            "mean_gap_peak_error_mv": {
                "explicit_leak": float(np.mean(paired_gap_trial_means[explicit_condition_name])),
                "atmn_carry": float(np.mean(paired_gap_trial_means["atmn_carry_tau2"])),
                "integrator_control": float(np.mean(paired_gap_trial_means["integrator_control"])),
            },
            "selected_max_length_state_std_mv": {
                "explicit_leak": float(np.mean(paired_drift_std[explicit_condition_name])),
                "atmn_carry": float(np.mean(paired_drift_std["atmn_carry_tau2"])),
                "integrator_control": float(np.mean(paired_drift_std["integrator_control"])),
            },
            "selected_max_length_final_abs_mv": {
                "explicit_leak": float(np.mean(paired_drift_final[explicit_condition_name])),
                "atmn_carry": float(np.mean(paired_drift_final["atmn_carry_tau2"])),
                "integrator_control": float(np.mean(paired_drift_final["integrator_control"])),
            },
            "selected_gap_retained_fraction": {
                "explicit_leak": float(np.mean(paired_retained_fraction[explicit_condition_name])),
                "atmn_carry": float(np.mean(paired_retained_fraction["atmn_carry_tau2"])),
                "integrator_control": float(np.mean(paired_retained_fraction["integrator_control"])),
            },
        },
        "spike_probe": {
            "selected_gap_ms": float(parameters["spike_probe_gap_ms"]),
            "pulse_amplitude_mv": float(parameters["spike_probe_pulse_amplitude_mv"]),
            "selected_gap_first_spike": {
                "explicit_leak": float(np.mean(paired_spike_probe_first[explicit_condition_name])),
                "atmn_carry": float(np.mean(paired_spike_probe_first["atmn_carry_tau2"])),
                "integrator_control": float(np.mean(paired_spike_probe_first["integrator_control"])),
            },
            "selected_gap_second_spike": {
                "explicit_leak": float(np.mean(paired_spike_probe_second[explicit_condition_name])),
                "atmn_carry": float(np.mean(paired_spike_probe_second["atmn_carry_tau2"])),
                "integrator_control": float(np.mean(paired_spike_probe_second["integrator_control"])),
            },
        },
        "gap_curves": main_gap_curves,
        "drift_curves": drift_curves,
        "tau_sweep": tau_sweep_summary,
        "anchor_example": {
            "gap_ms": example_gap,
            "second_peak_mv": example_anchor["second_peak_mv"],
            "tau_fit_ms": example_anchor["tau_fit_ms"],
            "tail_residual_mv": example_anchor["tail_residual_mv"],
        },
    }

    warnings = [
        "the statistical verdict is based on matched discrete-time toy recurrences, not on downstream language-model training.",
        "the brian2 lif trace is a qualitative biological anchor and reference target for pulse integration, not a claim about production runtime.",
        "this pilot separates temporal integration fidelity from drift, so a condition can win on one axis and lose on the other.",
    ]

    finished_at_utc = utc_now_iso()
    duration_sec = perf_counter() - started_perf
    run_record = build_run_record(
        simulation_name=simulation_name,
        script_path=Path(__file__),
        started_at_utc=started_at_utc,
        finished_at_utc=finished_at_utc,
        duration_sec=duration_sec,
        parameters=parameters,
        seed_numpy=parameters["seed_numpy"],
        n_trials=len(gap_records) + len(drift_records),
        summary=summary,
        statistics=statistics,
        trials=gap_records + drift_records + spike_probe_records,
        artifacts=[
            {"path": figure_path.as_posix(), "type": "figure"},
            {"path": metrics_path.as_posix(), "type": "metrics_json"},
        ],
        warnings=warnings,
        extra_configuration={"condition_retention": condition_retention},
    )
    write_json(metrics_path, run_record)

    print(f"mean gap peak error explicit leak: {summary['main_comparison']['mean_gap_peak_error_mv']['explicit_leak']:.3f} mV")
    print(f"mean gap peak error atmn carry: {summary['main_comparison']['mean_gap_peak_error_mv']['atmn_carry']:.3f} mV")
    print(f"gap {summary['main_comparison']['selected_gap_ms']:.1f} retained fraction explicit leak: {summary['main_comparison']['selected_gap_retained_fraction']['explicit_leak']:.3f}")
    print(f"gap {summary['main_comparison']['selected_gap_ms']:.1f} retained fraction atmn carry: {summary['main_comparison']['selected_gap_retained_fraction']['atmn_carry']:.3f}")
    print(f"length {summary['main_comparison']['max_drift_length_steps']} state std explicit leak: {summary['main_comparison']['selected_max_length_state_std_mv']['explicit_leak']:.3f} mV")
    print(f"length {summary['main_comparison']['max_drift_length_steps']} state std atmn carry: {summary['main_comparison']['selected_max_length_state_std_mv']['atmn_carry']:.3f} mV")
    print(f"spike probe second-spike rate explicit leak: {summary['spike_probe']['selected_gap_second_spike']['explicit_leak']:.3f}")
    print(f"spike probe second-spike rate integrator: {summary['spike_probe']['selected_gap_second_spike']['integrator_control']:.3f}")
    print(f"saved: {figure_path.name}")
    print(f"saved: {metrics_path.name}")


if __name__ == "__main__":
    main()