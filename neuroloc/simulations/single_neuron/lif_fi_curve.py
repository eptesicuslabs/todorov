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
    sys.path.append(str(SIM_ROOT))

from shared import apply_plot_style, build_rng, build_run_record, child_rng, mean_confidence_interval, paired_difference_stats, utc_now_iso, write_json

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
    output_dir = Path(__file__).parent
    figure_path = output_dir / "lif_leak_validation.png"
    metrics_path = output_dir / "lif_leak_validation_metrics.json"

    parameters = {
        "seed_numpy": 17,
        "dt_ms": 1.0,
        "threshold_mv": 15.0,
        "atmn_tau": 2.0,
        "leak_tau_ms": [5.0, 10.0, 20.0, 30.0],
        "gap_values_ms": [1.0, 4.0, 8.0, 16.0, 32.0, 64.0],
        "gap_trials": 32,
        "gap_total_steps": 120,
        "pulse_start_ms": 10.0,
        "pulse_width_ms": 4.0,
        "pulse_amplitude_mv": 0.1,
        "gap_noise_sigma_mv": 0.004,
        "drift_lengths": [128, 256, 512, 1024],
        "drift_trials": 48,
        "drift_noise_sigma_mv": 0.5,
        "anchor_tau_m_ms": 20.0,
        "anchor_resistance_mohm": 100.0,
        "example_gap_ms": 16.0,
    }

    rng = build_rng(parameters["seed_numpy"])
    threshold_mv = parameters["threshold_mv"]
    dt_ms = parameters["dt_ms"]
    pulse_width_steps = int(round(parameters["pulse_width_ms"] / dt_ms))
    pulse_start_steps = int(round(parameters["pulse_start_ms"] / dt_ms))
    leak_conditions = {
        f"explicit_leak_tau{int(tau_ms)}": float(np.exp(-dt_ms / tau_ms)) for tau_ms in parameters["leak_tau_ms"]
    }
    condition_retention = {
        **leak_conditions,
        "atmn_carry_tau2": float(1.0 / parameters["atmn_tau"]),
        "integrator_control": 1.0,
    }
    condition_colors = {
        "explicit_leak_tau20": "#1f77b4",
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

    main_gap_curves = {}
    for condition_name in ["explicit_leak_tau20", "atmn_carry_tau2", "integrator_control"]:
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
    for condition_name in ["explicit_leak_tau20", "atmn_carry_tau2", "integrator_control"]:
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
        ["explicit_leak_tau5", "explicit_leak_tau10", "explicit_leak_tau20", "explicit_leak_tau30", "atmn_carry_tau2", "integrator_control"],
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
    max_length = int(parameters["drift_lengths"][-1])
    for condition_name in condition_retention:
        paired_drift_std[condition_name] = [record["state_std_mv"] for record in drift_records if record["condition"] == condition_name and record["length_steps"] == max_length]
        paired_drift_final[condition_name] = [record["final_abs_mv"] for record in drift_records if record["condition"] == condition_name and record["length_steps"] == max_length]

    statistics = {
        "gap_mean_peak_error_explicit_vs_atmn": paired_difference_stats(paired_gap_trial_means["atmn_carry_tau2"], paired_gap_trial_means["explicit_leak_tau20"], seed=301),
        "gap_mean_peak_error_explicit_vs_integrator": paired_difference_stats(paired_gap_trial_means["integrator_control"], paired_gap_trial_means["explicit_leak_tau20"], seed=302),
        "retained_fraction_gap16_explicit_vs_atmn": paired_difference_stats(paired_retained_fraction["atmn_carry_tau2"], paired_retained_fraction["explicit_leak_tau20"], seed=306),
        "retained_fraction_gap16_explicit_vs_integrator": paired_difference_stats(paired_retained_fraction["integrator_control"], paired_retained_fraction["explicit_leak_tau20"], seed=307),
        "drift_state_std_explicit_vs_atmn_length1024": paired_difference_stats(paired_drift_std["atmn_carry_tau2"], paired_drift_std["explicit_leak_tau20"], seed=303),
        "drift_state_std_explicit_vs_integrator_length1024": paired_difference_stats(paired_drift_std["integrator_control"], paired_drift_std["explicit_leak_tau20"], seed=304),
        "drift_final_abs_explicit_vs_atmn_length1024": paired_difference_stats(paired_drift_final["atmn_carry_tau2"], paired_drift_final["explicit_leak_tau20"], seed=305),
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
    for condition_name in ["explicit_leak_tau20", "atmn_carry_tau2", "integrator_control"]:
        example_traces[condition_name] = simulate_discrete(example_inputs, condition_retention[condition_name], threshold_mv)[0][0]
    example_anchor = anchor_by_gap[example_gap]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(np.arange(example_inputs.shape[1]) * dt_ms, example_sequence, color="#7f7f7f", linestyle="--", linewidth=1.5, label="input")
    axes[0, 0].plot(example_anchor["time_ms"], example_anchor["trace_mv"], color="#000000", linewidth=2.2, label="brian2 lif anchor")
    for condition_name in ["explicit_leak_tau20", "atmn_carry_tau2", "integrator_control"]:
        axes[0, 0].plot(np.arange(example_inputs.shape[1]) * dt_ms, example_traces[condition_name], color=condition_colors[condition_name], label=condition_name.replace("_", " "))
    axes[0, 0].axhline(y=threshold_mv, color="#999999", linestyle=":", linewidth=1.2, label="threshold")
    axes[0, 0].set_title("paired-pulse integration at 16 ms gap")
    axes[0, 0].set_xlabel("time (ms)")
    axes[0, 0].set_ylabel("state (mV rel. rest)")
    axes[0, 0].legend(fontsize=8)

    for condition_name in ["explicit_leak_tau20", "atmn_carry_tau2", "integrator_control"]:
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

    for condition_name in ["explicit_leak_tau20", "atmn_carry_tau2", "integrator_control"]:
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
            "explicit_leak_condition": "explicit_leak_tau20",
            "baseline_condition": "atmn_carry_tau2",
            "control_condition": "integrator_control",
            "mean_gap_peak_error_mv": {
                "explicit_leak_tau20": float(np.mean(paired_gap_trial_means["explicit_leak_tau20"])),
                "atmn_carry_tau2": float(np.mean(paired_gap_trial_means["atmn_carry_tau2"])),
                "integrator_control": float(np.mean(paired_gap_trial_means["integrator_control"])),
            },
            "length1024_state_std_mv": {
                "explicit_leak_tau20": float(np.mean(paired_drift_std["explicit_leak_tau20"])),
                "atmn_carry_tau2": float(np.mean(paired_drift_std["atmn_carry_tau2"])),
                "integrator_control": float(np.mean(paired_drift_std["integrator_control"])),
            },
            "length1024_final_abs_mv": {
                "explicit_leak_tau20": float(np.mean(paired_drift_final["explicit_leak_tau20"])),
                "atmn_carry_tau2": float(np.mean(paired_drift_final["atmn_carry_tau2"])),
            },
            "gap16_retained_fraction": {
                "explicit_leak_tau20": float(np.mean(paired_retained_fraction["explicit_leak_tau20"])),
                "atmn_carry_tau2": float(np.mean(paired_retained_fraction["atmn_carry_tau2"])),
                "integrator_control": float(np.mean(paired_retained_fraction["integrator_control"])),
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
        n_trials=parameters["gap_trials"] + parameters["drift_trials"],
        summary=summary,
        statistics=statistics,
        trials=gap_records + drift_records,
        artifacts=[
            {"path": figure_path.as_posix(), "type": "figure"},
            {"path": metrics_path.as_posix(), "type": "metrics_json"},
        ],
        warnings=warnings,
        extra_configuration={"condition_retention": condition_retention},
    )
    write_json(metrics_path, run_record)

    print(f"mean gap peak error explicit leak: {summary['main_comparison']['mean_gap_peak_error_mv']['explicit_leak_tau20']:.3f} mV")
    print(f"mean gap peak error atmn carry: {summary['main_comparison']['mean_gap_peak_error_mv']['atmn_carry_tau2']:.3f} mV")
    print(f"gap 16 retained fraction explicit leak: {summary['main_comparison']['gap16_retained_fraction']['explicit_leak_tau20']:.3f}")
    print(f"gap 16 retained fraction atmn carry: {summary['main_comparison']['gap16_retained_fraction']['atmn_carry_tau2']:.3f}")
    print(f"length 1024 state std explicit leak: {summary['main_comparison']['length1024_state_std_mv']['explicit_leak_tau20']:.3f} mV")
    print(f"length 1024 state std atmn carry: {summary['main_comparison']['length1024_state_std_mv']['atmn_carry_tau2']:.3f} mV")
    print(f"saved: {figure_path.name}")
    print(f"saved: {metrics_path.name}")


if __name__ == "__main__":
    main()