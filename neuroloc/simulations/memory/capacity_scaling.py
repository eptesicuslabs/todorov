from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared import (
    apply_plot_style,
    build_rng,
    build_run_record,
    child_rng,
    env_float,
    env_int,
    env_list,
    independent_difference_stats,
    output_dir_for,
    require_positive,
    require_positive_list,
    require_unit_interval,
    utc_now_iso,
    write_json,
)

SCRIPT_PATH = Path(__file__).resolve()
SEED = env_int("CAPACITY_SEED", 42)
NETWORK_SIZES = env_list("CAPACITY_NETWORK_SIZES", int, [200, 500, 1000, 2000])
LOAD_FACTORS = env_list("CAPACITY_LOAD_FACTORS", float, [0.02, 0.05, 0.08, 0.10, 0.138, 0.20, 0.30])
CORRUPTION = env_float("CAPACITY_CORRUPTION", 0.05)
TRIALS = env_int("CAPACITY_TRIALS", 30)
MAX_ITERS = env_int("CAPACITY_MAX_ITERS", 50)
TERNARY_FIRING_RATE = env_float("CAPACITY_TERNARY_FIRING_RATE", 0.41)
TARGET_ALPHA = env_float("CAPACITY_TARGET_ALPHA", 0.138)

if not any(np.isclose(alpha, TARGET_ALPHA) for alpha in LOAD_FACTORS):
    LOAD_FACTORS = sorted([*LOAD_FACTORS, TARGET_ALPHA])

require_positive_list("CAPACITY_NETWORK_SIZES", NETWORK_SIZES)
require_positive_list("CAPACITY_LOAD_FACTORS", LOAD_FACTORS)
require_unit_interval("CAPACITY_CORRUPTION", CORRUPTION, allow_zero=True)
require_positive("CAPACITY_TRIALS", TRIALS)
require_positive("CAPACITY_MAX_ITERS", MAX_ITERS)
require_unit_interval("CAPACITY_TERNARY_FIRING_RATE", TERNARY_FIRING_RATE, allow_zero=True)
require_positive("CAPACITY_TARGET_ALPHA", TARGET_ALPHA)


def generate_binary_patterns(rng, n, p):
    return rng.choice([-1, 1], size=(p, n)).astype(np.float64)


def generate_ternary_patterns(rng, n, p, firing_rate=0.41):
    patterns = np.zeros((p, n), dtype=np.float64)
    for i in range(p):
        mask = rng.random(n) < firing_rate
        signs = rng.choice([-1, 1], size=n)
        patterns[i] = mask * signs
    return patterns


def store_patterns(patterns):
    p, n = patterns.shape
    W = (patterns.T @ patterns) / n
    np.fill_diagonal(W, 0.0)
    return W


def retrieve_binary(W, probe, max_iters=50):
    state = probe.copy()
    for _ in range(max_iters):
        prev = state.copy()
        field = W @ state
        state = np.sign(field)
        state[state == 0] = 1.0
        if np.array_equal(state, prev):
            break
    return state


def retrieve_ternary(W, probe, theta, max_iters=50):
    state = probe.copy()
    for _ in range(max_iters):
        prev = state.copy()
        field = W @ state
        state = np.where(field > theta, 1.0, np.where(field < -theta, -1.0, 0.0))
        if np.array_equal(state, prev):
            break
    return state


def calibrate_theta(W, patterns, rng, target_fr=0.41, n_samples=20):
    n = W.shape[0]
    fields = []
    indices = rng.choice(len(patterns), size=min(n_samples, len(patterns)), replace=False)
    for i in indices:
        f = W @ patterns[i]
        fields.append(np.abs(f))
    all_fields = np.concatenate(fields)
    theta = np.percentile(all_fields, (1.0 - target_fr) * 100)
    return max(theta, 1e-8)


def corrupt_binary(rng, pattern, fraction):
    corrupted = pattern.copy()
    n = len(pattern)
    n_flip = int(n * fraction)
    indices = rng.choice(n, size=n_flip, replace=False)
    corrupted[indices] *= -1
    return corrupted


def corrupt_ternary(rng, pattern, fraction):
    corrupted = pattern.copy()
    n = len(pattern)
    n_corrupt = int(n * fraction)
    indices = rng.choice(n, size=n_corrupt, replace=False)
    for idx in indices:
        current = corrupted[idx]
        options = [v for v in [-1, 0, 1] if v != current]
        corrupted[idx] = rng.choice(options)
    return corrupted


def overlap(retrieved, original):
    n = len(original)
    return float(np.dot(retrieved, original) / n)


def exact_match(retrieved, original):
    return float(np.array_equal(retrieved, original))


def run_capacity_sweep(rng, n, load_factors, pattern_type, corruption, trials, max_iters):
    results = []
    trial_records = []
    for alpha in load_factors:
        p = max(1, int(alpha * n))
        exact_matches = []
        overlaps = []

        for t in range(trials):
            trial_rng = child_rng(rng)

            if pattern_type == "binary":
                patterns = generate_binary_patterns(trial_rng, n, p)
            else:
                patterns = generate_ternary_patterns(trial_rng, n, p, TERNARY_FIRING_RATE)

            W = store_patterns(patterns)

            probe_idx = trial_rng.integers(0, p)
            original = patterns[probe_idx]

            if pattern_type == "binary":
                corrupted = corrupt_binary(trial_rng, original, corruption)
                retrieved = retrieve_binary(W, corrupted, max_iters)
            else:
                corrupted = corrupt_ternary(trial_rng, original, corruption)
                theta = calibrate_theta(W, patterns, trial_rng, TERNARY_FIRING_RATE)
                retrieved = retrieve_ternary(W, corrupted, theta, max_iters)

            exact_matches.append(exact_match(retrieved, original))
            overlaps.append(overlap(retrieved, original))
            trial_records.append({
                "n": n,
                "alpha": float(alpha),
                "p": p,
                "pattern_type": pattern_type,
                "corruption": float(corruption),
                "trial_id": int(t),
                "exact_retrieval": float(exact_matches[-1]),
                "overlap": float(overlaps[-1]),
            })

        results.append({
            "n": n,
            "alpha": alpha,
            "p": p,
            "pattern_type": pattern_type,
            "corruption": corruption,
            "exact_retrieval_mean": float(np.mean(exact_matches)),
            "exact_retrieval_std": float(np.std(exact_matches)),
            "overlap_mean": float(np.mean(overlaps)),
            "overlap_std": float(np.std(overlaps)),
            "trials": trials,
        })

        print(f"  n={n} alpha={alpha:.3f} p={p} {pattern_type}: exact={np.mean(exact_matches):.3f} overlap={np.mean(overlaps):.3f}")

    return results, trial_records


def main():
    apply_plot_style()
    import matplotlib.pyplot as plt

    started = utc_now_iso()
    t0 = time.time()
    rng = build_rng(SEED)

    all_results = []
    trial_records = []

    for n in NETWORK_SIZES:
        print(f"network size n={n}")
        binary_results, binary_trials = run_capacity_sweep(
            child_rng(rng), n, LOAD_FACTORS, "binary", CORRUPTION, TRIALS, MAX_ITERS
        )
        all_results.extend(binary_results)
        trial_records.extend(binary_trials)

        ternary_results, ternary_trials = run_capacity_sweep(
            child_rng(rng), n, LOAD_FACTORS, "ternary", CORRUPTION, TRIALS, MAX_ITERS
        )
        all_results.extend(ternary_results)
        trial_records.extend(ternary_trials)

    output_dir = output_dir_for(SCRIPT_PATH)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    for n in NETWORK_SIZES:
        alphas = [r["alpha"] for r in all_results if r["n"] == n and r["pattern_type"] == "binary"]
        exacts = [r["exact_retrieval_mean"] for r in all_results if r["n"] == n and r["pattern_type"] == "binary"]
        ax1.plot(alphas, exacts, marker="o", label=f"n={n}")
    ax1.axvline(x=TARGET_ALPHA, color="red", linestyle="--", alpha=0.5, label=f"target alpha={TARGET_ALPHA:.3f}")
    ax1.set_xlabel("load factor (p/n)")
    ax1.set_ylabel("exact retrieval rate")
    ax1.set_title("binary patterns: capacity curve")
    ax1.legend(fontsize=8)

    ax2 = axes[0, 1]
    for n in NETWORK_SIZES:
        alphas = [r["alpha"] for r in all_results if r["n"] == n and r["pattern_type"] == "ternary"]
        exacts = [r["exact_retrieval_mean"] for r in all_results if r["n"] == n and r["pattern_type"] == "ternary"]
        ax2.plot(alphas, exacts, marker="o", label=f"n={n}")
    ax2.axvline(x=TARGET_ALPHA, color="red", linestyle="--", alpha=0.5, label=f"target alpha={TARGET_ALPHA:.3f}")
    ax2.set_xlabel("load factor (p/n)")
    ax2.set_ylabel("exact retrieval rate")
    ax2.set_title("ternary patterns (41% fr): capacity curve")
    ax2.legend(fontsize=8)

    ax3 = axes[1, 0]
    for n in NETWORK_SIZES:
        alphas = [r["alpha"] for r in all_results if r["n"] == n and r["pattern_type"] == "binary"]
        ovlps = [r["overlap_mean"] for r in all_results if r["n"] == n and r["pattern_type"] == "binary"]
        ax3.plot(alphas, ovlps, marker="o", label=f"n={n}")
    ax3.axvline(x=TARGET_ALPHA, color="red", linestyle="--", alpha=0.5)
    ax3.set_xlabel("load factor (p/n)")
    ax3.set_ylabel("mean overlap")
    ax3.set_title("binary patterns: overlap vs load")
    ax3.legend(fontsize=8)

    ax4 = axes[1, 1]
    for n in NETWORK_SIZES:
        alphas = [r["alpha"] for r in all_results if r["n"] == n and r["pattern_type"] == "ternary"]
        ovlps = [r["overlap_mean"] for r in all_results if r["n"] == n and r["pattern_type"] == "ternary"]
        ax4.plot(alphas, ovlps, marker="o", label=f"n={n}")
    ax4.axvline(x=TARGET_ALPHA, color="red", linestyle="--", alpha=0.5)
    ax4.set_xlabel("load factor (p/n)")
    ax4.set_ylabel("mean overlap")
    ax4.set_title("ternary patterns (41% fr): overlap vs load")
    ax4.legend(fontsize=8)

    plt.tight_layout()
    fig_path = output_dir / "capacity_scaling.png"
    plt.savefig(fig_path)
    plt.close()

    binary_at_target = [r for r in all_results if r["pattern_type"] == "binary" and abs(r["alpha"] - TARGET_ALPHA) < 1e-6]
    ternary_at_target = [r for r in all_results if r["pattern_type"] == "ternary" and abs(r["alpha"] - TARGET_ALPHA) < 1e-6]

    summary = {
        "target_alpha": float(TARGET_ALPHA),
        "binary_at_target_alpha": {n: next((r["exact_retrieval_mean"] for r in binary_at_target if r["n"] == n), None) for n in NETWORK_SIZES},
        "ternary_at_target_alpha": {n: next((r["exact_retrieval_mean"] for r in ternary_at_target if r["n"] == n), None) for n in NETWORK_SIZES},
    }

    statistics = {}
    binary_exact_trials = [
        record["exact_retrieval"]
        for record in trial_records
        if record["pattern_type"] == "binary" and abs(record["alpha"] - TARGET_ALPHA) < 1e-6
    ]
    ternary_exact_trials = [
        record["exact_retrieval"]
        for record in trial_records
        if record["pattern_type"] == "ternary" and abs(record["alpha"] - TARGET_ALPHA) < 1e-6
    ]
    if binary_exact_trials and ternary_exact_trials:
        statistics["binary_vs_ternary_at_target_alpha"] = independent_difference_stats(
            binary_exact_trials,
            ternary_exact_trials,
            SEED,
        )

    finished = utc_now_iso()
    duration = time.time() - t0

    record = build_run_record(
        simulation_name="capacity_scaling",
        script_path=SCRIPT_PATH,
        started_at_utc=started,
        finished_at_utc=finished,
        duration_sec=duration,
        parameters={
            "network_sizes": NETWORK_SIZES,
            "load_factors": LOAD_FACTORS,
            "corruption": CORRUPTION,
            "trials": TRIALS,
            "max_iters": MAX_ITERS,
            "ternary_firing_rate": TERNARY_FIRING_RATE,
            "target_alpha": TARGET_ALPHA,
        },
        seed_numpy=SEED,
        n_trials=len(trial_records),
        summary=summary,
        statistics=statistics,
        trials=trial_records,
        artifacts=[
            {"path": fig_path.as_posix(), "type": "figure"},
            {"path": (output_dir / "capacity_scaling_metrics.json").as_posix(), "type": "metrics"},
        ],
        warnings=[
            "ternary retrieval uses calibrated theta; results depend on calibration quality",
            "5% corruption is low but still conflates capacity with basin size to a small degree",
        ],
    )

    metrics_path = output_dir / "capacity_scaling_metrics.json"
    write_json(metrics_path, record)

    print(f"\ndone in {duration:.1f}s")
    print(f"binary at alpha={TARGET_ALPHA:.3f}: {summary['binary_at_target_alpha']}")
    print(f"ternary at alpha={TARGET_ALPHA:.3f}: {summary['ternary_at_target_alpha']}")


if __name__ == "__main__":
    main()
