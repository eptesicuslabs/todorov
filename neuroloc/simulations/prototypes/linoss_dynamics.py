from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared import (
    apply_plot_style,
    build_rng,
    build_run_record,
    child_rng,
    mean_confidence_interval,
    paired_difference_stats,
    utc_now_iso,
    write_json,
)

SCRIPT_PATH = Path(__file__).resolve()
SEED = 42
D_STATE = 64
D_INPUT = 32
SEQ_LENGTHS = [64, 128, 256, 512, 1024]
TRIALS = 20


def make_oscillatory_A(d_state, rng):
    freqs = rng.uniform(0.01, 0.5, size=d_state // 2)
    A = np.zeros((d_state, d_state))
    for i in range(d_state // 2):
        idx = 2 * i
        A[idx, idx + 1] = -freqs[i]
        A[idx + 1, idx] = freqs[i]
    return A


def make_decay_A(d_state, rng):
    decay = -rng.uniform(0.01, 0.1, size=d_state)
    return np.diag(decay)


def discretize_exact(A, B, dt=1.0):
    A_bar = expm(dt * A)
    d = A.shape[0]
    try:
        A_inv = np.linalg.solve(A, np.eye(d))
        B_bar = A_inv @ (A_bar - np.eye(d)) @ B
    except np.linalg.LinAlgError:
        B_bar = dt * B
    return A_bar, B_bar


def run_sequential(A_bar, B_bar, inputs):
    T, d_in = inputs.shape
    d_state = A_bar.shape[0]
    states = np.zeros((T, d_state))
    h = np.zeros(d_state)
    for t in range(T):
        h = A_bar @ h + B_bar @ inputs[t]
        states[t] = h
    return states


def measure_retention_decoder(states, inputs, target_window=None):
    T = states.shape[0]
    d_state = states.shape[1]
    d_input = inputs.shape[1]
    if target_window is None:
        target_window = T // 4

    early_input = inputs[:target_window]
    late_state = states[3 * T // 4:]

    if len(late_state) < 2 or len(early_input) < 2:
        return 0.0

    n_samples = min(len(late_state), len(early_input))
    n_train = max(1, int(n_samples * 0.7))
    n_test = n_samples - n_train

    if n_test < 2:
        return 0.0

    X = late_state[:n_samples]
    y = early_input[:n_samples, 0]

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    try:
        ridge_lambda = 1.0
        W_decoder = np.linalg.solve(
            X_train.T @ X_train + ridge_lambda * np.eye(d_state),
            X_train.T @ y_train
        )
        y_pred = X_test @ W_decoder
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        if ss_tot == 0:
            return 0.0
        r2 = 1.0 - ss_res / ss_tot
        return float(max(r2, 0.0))
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def measure_autonomous_stability(A_bar, d_state, n_steps=200):
    h = np.ones(d_state) / np.sqrt(d_state)
    norms = [np.linalg.norm(h)]
    for _ in range(n_steps):
        h = A_bar @ h
        norms.append(np.linalg.norm(h))
    norms = np.array(norms)
    return float(norms[-1] / max(norms[0], 1e-10))


def main():
    apply_plot_style()
    import matplotlib.pyplot as plt

    started = utc_now_iso()
    t0 = time.time()
    rng = build_rng(SEED)

    all_results = []

    for seq_len in SEQ_LENGTHS:
        for trial in range(TRIALS):
            trial_rng = child_rng(rng)

            B = trial_rng.standard_normal((D_STATE, D_INPUT)) * 0.1
            inputs = trial_rng.standard_normal((seq_len, D_INPUT)) * 0.1

            A_osc = make_oscillatory_A(D_STATE, trial_rng)
            A_bar_osc, B_bar_osc = discretize_exact(A_osc, B)
            states_osc = run_sequential(A_bar_osc, B_bar_osc, inputs)

            A_decay = make_decay_A(D_STATE, trial_rng)
            A_bar_decay, B_bar_decay = discretize_exact(A_decay, B)
            states_decay = run_sequential(A_bar_decay, B_bar_decay, inputs)

            osc_retention = measure_retention_decoder(states_osc, inputs)
            decay_retention = measure_retention_decoder(states_decay, inputs)
            osc_auto = measure_autonomous_stability(A_bar_osc, D_STATE)
            decay_auto = measure_autonomous_stability(A_bar_decay, D_STATE)

            all_results.append({
                "seq_len": seq_len,
                "trial": trial,
                "dynamics": "oscillatory",
                "retention_r2": osc_retention,
                "autonomous_norm_ratio": osc_auto,
            })
            all_results.append({
                "seq_len": seq_len,
                "trial": trial,
                "dynamics": "exponential_decay",
                "retention_r2": decay_retention,
                "autonomous_norm_ratio": decay_auto,
            })

        osc_ret = np.mean([r["retention_r2"] for r in all_results if r["seq_len"] == seq_len and r["dynamics"] == "oscillatory"])
        dec_ret = np.mean([r["retention_r2"] for r in all_results if r["seq_len"] == seq_len and r["dynamics"] == "exponential_decay"])
        print(f"seq_len={seq_len}: oscillatory R2={osc_ret:.4f}, decay R2={dec_ret:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    for dyn in ["oscillatory", "exponential_decay"]:
        rets = [np.mean([r["retention_r2"] for r in all_results if r["seq_len"] == sl and r["dynamics"] == dyn]) for sl in SEQ_LENGTHS]
        ax1.plot(SEQ_LENGTHS, rets, marker="o", label=dyn)
    ax1.set_xlabel("sequence length")
    ax1.set_ylabel("decoder R2 (early input from late state)")
    ax1.set_title("information retention: oscillatory vs decay")
    ax1.set_xscale("log", base=2)
    ax1.legend()

    ax2 = axes[1]
    for dyn in ["oscillatory", "exponential_decay"]:
        autos = [np.mean([r["autonomous_norm_ratio"] for r in all_results if r["seq_len"] == sl and r["dynamics"] == dyn]) for sl in SEQ_LENGTHS]
        ax2.plot(SEQ_LENGTHS, autos, marker="o", label=dyn)
    ax2.set_xlabel("sequence length (used for grouping)")
    ax2.set_ylabel("autonomous norm ratio (200 steps)")
    ax2.set_title("autonomous stability (1.0 = energy preserving)")
    ax2.axhline(y=1.0, color="red", linestyle="--", alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    fig_path = SCRIPT_PATH.parent / "linoss_dynamics.png"
    plt.savefig(fig_path)
    plt.close()

    summary = {}
    for sl in [256, 1024]:
        for dyn in ["oscillatory", "exponential_decay"]:
            rets = [r["retention_r2"] for r in all_results if r["seq_len"] == sl and r["dynamics"] == dyn]
            summary[f"{dyn}_retention_seq{sl}"] = mean_confidence_interval(rets)

    osc_1024 = [r["retention_r2"] for r in all_results if r["seq_len"] == 1024 and r["dynamics"] == "oscillatory"]
    dec_1024 = [r["retention_r2"] for r in all_results if r["seq_len"] == 1024 and r["dynamics"] == "exponential_decay"]
    summary["osc_vs_decay_retention_1024"] = paired_difference_stats(dec_1024, osc_1024, SEED)

    finished = utc_now_iso()
    duration = time.time() - t0

    record = build_run_record(
        simulation_name="linoss_dynamics_comparison",
        script_path=SCRIPT_PATH,
        started_at_utc=started,
        finished_at_utc=finished,
        duration_sec=duration,
        parameters={
            "d_state": D_STATE,
            "d_input": D_INPUT,
            "seq_lengths": SEQ_LENGTHS,
            "trials": TRIALS,
        },
        seed_numpy=SEED,
        n_trials=len(all_results),
        summary=summary,
        statistics={},
        trials=all_results,
        artifacts=[
            {"name": "linoss_dynamics.png", "type": "figure"},
            {"name": "linoss_dynamics_metrics.json", "type": "metrics"},
        ],
        warnings=[
            "both dynamics use exact matrix exponential discretization (expm) for fair comparison",
            "retention measured via linear decoder R2 from late states to early inputs",
            "autonomous stability measured without input injection (pure dynamics)",
        ],
    )

    metrics_path = SCRIPT_PATH.parent / "linoss_dynamics_metrics.json"
    write_json(metrics_path, record)

    print(f"\ndone in {duration:.1f}s")


if __name__ == "__main__":
    main()
