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
    mean_confidence_interval,
    paired_difference_stats,
    utc_now_iso,
    write_json,
)

SCRIPT_PATH = Path(__file__).resolve()
SEED = 42
NETWORK_SIZES = [200, 500, 1000, 2000]
LOAD_FACTORS = [0.02, 0.05, 0.08, 0.10, 0.138, 0.20, 0.30]
CORRUPTION = 0.05
TRIALS = 30
MAX_ITERS = 50
TERNARY_FIRING_RATE = 0.41


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

    return results


def main():
    apply_plot_style()
    import matplotlib.pyplot as plt

    started = utc_now_iso()
    t0 = time.time()
    rng = build_rng(SEED)

    all_results = []

    for n in NETWORK_SIZES:
        print(f"network size n={n}")
        binary_results = run_capacity_sweep(
            child_rng(rng), n, LOAD_FACTORS, "binary", CORRUPTION, TRIALS, MAX_ITERS
        )
        all_results.extend(binary_results)

        ternary_results = run_capacity_sweep(
            child_rng(rng), n, LOAD_FACTORS, "ternary", CORRUPTION, TRIALS, MAX_ITERS
        )
        all_results.extend(ternary_results)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    for n in NETWORK_SIZES:
        alphas = [r["alpha"] for r in all_results if r["n"] == n and r["pattern_type"] == "binary"]
        exacts = [r["exact_retrieval_mean"] for r in all_results if r["n"] == n and r["pattern_type"] == "binary"]
        ax1.plot(alphas, exacts, marker="o", label=f"n={n}")
    ax1.axvline(x=0.138, color="red", linestyle="--", alpha=0.5, label="0.138N theoretical")
    ax1.set_xlabel("load factor (p/n)")
    ax1.set_ylabel("exact retrieval rate")
    ax1.set_title("binary patterns: capacity curve")
    ax1.legend(fontsize=8)

    ax2 = axes[0, 1]
    for n in NETWORK_SIZES:
        alphas = [r["alpha"] for r in all_results if r["n"] == n and r["pattern_type"] == "ternary"]
        exacts = [r["exact_retrieval_mean"] for r in all_results if r["n"] == n and r["pattern_type"] == "ternary"]
        ax2.plot(alphas, exacts, marker="o", label=f"n={n}")
    ax2.axvline(x=0.138, color="red", linestyle="--", alpha=0.5, label="0.138N theoretical")
    ax2.set_xlabel("load factor (p/n)")
    ax2.set_ylabel("exact retrieval rate")
    ax2.set_title("ternary patterns (41% fr): capacity curve")
    ax2.legend(fontsize=8)

    ax3 = axes[1, 0]
    for n in NETWORK_SIZES:
        alphas = [r["alpha"] for r in all_results if r["n"] == n and r["pattern_type"] == "binary"]
        ovlps = [r["overlap_mean"] for r in all_results if r["n"] == n and r["pattern_type"] == "binary"]
        ax3.plot(alphas, ovlps, marker="o", label=f"n={n}")
    ax3.axvline(x=0.138, color="red", linestyle="--", alpha=0.5)
    ax3.set_xlabel("load factor (p/n)")
    ax3.set_ylabel("mean overlap")
    ax3.set_title("binary patterns: overlap vs load")
    ax3.legend(fontsize=8)

    ax4 = axes[1, 1]
    for n in NETWORK_SIZES:
        alphas = [r["alpha"] for r in all_results if r["n"] == n and r["pattern_type"] == "ternary"]
        ovlps = [r["overlap_mean"] for r in all_results if r["n"] == n and r["pattern_type"] == "ternary"]
        ax4.plot(alphas, ovlps, marker="o", label=f"n={n}")
    ax4.axvline(x=0.138, color="red", linestyle="--", alpha=0.5)
    ax4.set_xlabel("load factor (p/n)")
    ax4.set_ylabel("mean overlap")
    ax4.set_title("ternary patterns (41% fr): overlap vs load")
    ax4.legend(fontsize=8)

    plt.tight_layout()
    fig_path = SCRIPT_PATH.parent / "capacity_scaling.png"
    plt.savefig(fig_path)
    plt.close()

    binary_at_014 = [r for r in all_results if r["pattern_type"] == "binary" and abs(r["alpha"] - 0.138) < 0.001]
    ternary_at_014 = [r for r in all_results if r["pattern_type"] == "ternary" and abs(r["alpha"] - 0.138) < 0.001]

    summary = {
        "binary_at_0138": {n: next((r["exact_retrieval_mean"] for r in binary_at_014 if r["n"] == n), None) for n in NETWORK_SIZES},
        "ternary_at_0138": {n: next((r["exact_retrieval_mean"] for r in ternary_at_014 if r["n"] == n), None) for n in NETWORK_SIZES},
    }

    if binary_at_014 and ternary_at_014:
        b_vals = [r["exact_retrieval_mean"] for r in binary_at_014]
        t_vals = [r["exact_retrieval_mean"] for r in ternary_at_014]
        if len(b_vals) == len(t_vals):
            summary["binary_vs_ternary_at_0138"] = paired_difference_stats(t_vals, b_vals, SEED)

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
        },
        seed_numpy=SEED,
        n_trials=len(all_results),
        summary=summary,
        statistics={},
        trials=all_results,
        artifacts=[
            {"name": "capacity_scaling.png", "type": "figure"},
            {"name": "capacity_scaling_metrics.json", "type": "metrics"},
        ],
        warnings=[
            "ternary retrieval uses calibrated theta; results depend on calibration quality",
            "5% corruption is low but still conflates capacity with basin size to a small degree",
        ],
    )

    metrics_path = SCRIPT_PATH.parent / "capacity_scaling_metrics.json"
    write_json(metrics_path, record)

    print(f"\ndone in {duration:.1f}s")
    print(f"binary at alpha=0.138: {summary['binary_at_0138']}")
    print(f"ternary at alpha=0.138: {summary['ternary_at_0138']}")


if __name__ == "__main__":
    main()
