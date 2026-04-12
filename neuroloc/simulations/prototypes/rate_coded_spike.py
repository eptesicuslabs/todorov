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
SEED = env_int("RATE_SPIKE_SEED", 42)
N_NEURONS = env_int("RATE_SPIKE_N_NEURONS", 500)
N_PATTERNS_LIST = env_list("RATE_SPIKE_N_PATTERNS", int, [5, 10, 20, 30, 50])
CORRUPTION = env_float("RATE_SPIKE_CORRUPTION", 0.10)
TRIALS = env_int("RATE_SPIKE_TRIALS", 30)
MAX_ITERS = env_int("RATE_SPIKE_MAX_ITERS", 50)

require_positive("RATE_SPIKE_N_NEURONS", N_NEURONS)
require_positive_list("RATE_SPIKE_N_PATTERNS", N_PATTERNS_LIST)
require_unit_interval("RATE_SPIKE_CORRUPTION", CORRUPTION, allow_zero=True)
require_positive("RATE_SPIKE_TRIALS", TRIALS)
require_positive("RATE_SPIKE_MAX_ITERS", MAX_ITERS)


def generate_binary_patterns(rng, n, p):
    return rng.choice([-1, 1], size=(p, n)).astype(np.float64)


def generate_ternary_patterns(rng, n, p, firing_rate=0.41):
    patterns = np.zeros((p, n), dtype=np.float64)
    for i in range(p):
        mask = rng.random(n) < firing_rate
        signs = rng.choice([-1, 1], size=n)
        patterns[i] = mask * signs
    return patterns


def generate_rate_coded_patterns(rng, n, p):
    patterns = rng.random((p, n)).astype(np.float64)
    patterns -= patterns.mean(axis=0, keepdims=True)
    return patterns


def store_hebbian(patterns):
    p, n = patterns.shape
    W = (patterns.T @ patterns) / n
    np.fill_diagonal(W, 0.0)
    return W


def retrieve_binary(W, probe, max_iters=50):
    state = probe.copy()
    for _ in range(max_iters):
        prev = state.copy()
        state = np.sign(W @ state)
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


def retrieve_rate_coded(W, probe, max_iters=50):
    state = probe.copy()
    for _ in range(max_iters):
        prev = state.copy()
        field = W @ state
        state = np.tanh(field)
        if np.max(np.abs(state - prev)) < 1e-6:
            break
    return state


def corrupt_binary(rng, pattern, fraction):
    corrupted = pattern.copy()
    n = len(pattern)
    indices = rng.choice(n, size=int(n * fraction), replace=False)
    corrupted[indices] *= -1
    return corrupted


def corrupt_ternary(rng, pattern, fraction):
    corrupted = pattern.copy()
    n = len(pattern)
    indices = rng.choice(n, size=int(n * fraction), replace=False)
    for idx in indices:
        current = corrupted[idx]
        options = [v for v in [-1, 0, 1] if v != current]
        corrupted[idx] = rng.choice(options)
    return corrupted


def corrupt_rate_coded(rng, pattern, fraction):
    corrupted = pattern.copy()
    n = len(pattern)
    indices = rng.choice(n, size=int(n * fraction), replace=False)
    corrupted[indices] = rng.standard_normal(len(indices)) * np.std(pattern)
    return corrupted


def cosine_similarity(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def run_experiment(rng, n, p, coding, corruption, trials, max_iters):
    cosines = []
    exacts = []
    trial_records = []

    for t in range(trials):
        trial_rng = child_rng(rng)

        if coding == "binary":
            patterns = generate_binary_patterns(trial_rng, n, p)
        elif coding == "ternary":
            patterns = generate_ternary_patterns(trial_rng, n, p)
        elif coding == "rate":
            patterns = generate_rate_coded_patterns(trial_rng, n, p)

        W = store_hebbian(patterns)
        probe_idx = trial_rng.integers(0, p)
        original = patterns[probe_idx]

        if coding == "binary":
            corrupted = corrupt_binary(trial_rng, original, corruption)
            retrieved = retrieve_binary(W, corrupted, max_iters)
        elif coding == "ternary":
            corrupted = corrupt_ternary(trial_rng, original, corruption)
            abs_fields = np.abs(W @ corrupted)
            theta = np.percentile(abs_fields, 59)
            retrieved = retrieve_ternary(W, corrupted, max(theta, 1e-8), max_iters)
        elif coding == "rate":
            corrupted = corrupt_rate_coded(trial_rng, original, corruption)
            retrieved = retrieve_rate_coded(W, corrupted, max_iters)

        cos = cosine_similarity(retrieved, original)
        exact = float(np.allclose(retrieved, original, atol=0.05))
        cosines.append(cos)
        exacts.append(exact)
        trial_records.append({
            "coding": coding,
            "n": n,
            "p": p,
            "alpha": p / n,
            "corruption": float(corruption),
            "trial_id": int(t),
            "cosine": float(cos),
            "exact": float(exact),
        })

    return {
        "coding": coding,
        "n": n,
        "p": p,
        "alpha": p / n,
        "corruption": corruption,
        "cosine_mean": float(np.mean(cosines)),
        "cosine_std": float(np.std(cosines)),
        "exact_mean": float(np.mean(exacts)),
        "trials": trials,
    }, trial_records


def main():
    apply_plot_style()
    import matplotlib.pyplot as plt

    started = utc_now_iso()
    t0 = time.time()
    rng = build_rng(SEED)

    codings = ["binary", "ternary", "rate"]
    all_results = []
    trial_records = []

    for p in N_PATTERNS_LIST:
        for coding in codings:
            result, result_trials = run_experiment(
                child_rng(rng), N_NEURONS, p, coding, CORRUPTION, TRIALS, MAX_ITERS
            )
            all_results.append(result)
            trial_records.extend(result_trials)
            print(f"  {coding} p={p}: cosine={result['cosine_mean']:.4f} exact={result['exact_mean']:.3f}")

    output_dir = output_dir_for(SCRIPT_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    for coding in codings:
        alphas = [r["alpha"] for r in all_results if r["coding"] == coding]
        cosines = [r["cosine_mean"] for r in all_results if r["coding"] == coding]
        ax1.plot(alphas, cosines, marker="o", label=coding)
    ax1.set_xlabel("load factor (p/n)")
    ax1.set_ylabel("cosine similarity (retrieved vs original)")
    ax1.set_title("retrieval quality: binary vs ternary vs rate-coded")
    ax1.legend()

    ax2 = axes[1]
    for coding in codings:
        alphas = [r["alpha"] for r in all_results if r["coding"] == coding]
        exacts = [r["exact_mean"] for r in all_results if r["coding"] == coding]
        ax2.plot(alphas, exacts, marker="o", label=coding)
    ax2.set_xlabel("load factor (p/n)")
    ax2.set_ylabel("approximate retrieval rate (atol=0.05)")
    ax2.set_title("exact retrieval: binary vs ternary vs rate-coded")
    ax2.legend()

    plt.tight_layout()
    fig_path = output_dir / "rate_coded_spike.png"
    plt.savefig(fig_path)
    plt.close()

    reference_pattern_count = 10 if 10 in N_PATTERNS_LIST else N_PATTERNS_LIST[min(len(N_PATTERNS_LIST) - 1, 0)]
    summary = {}
    summary["reference_pattern_count"] = int(reference_pattern_count)
    summary["reference_cosine"] = {
        coding: next((r["cosine_mean"] for r in all_results if r["coding"] == coding and r["p"] == reference_pattern_count), None)
        for coding in codings
    }
    summary["reference_exact"] = {
        coding: next((r["exact_mean"] for r in all_results if r["coding"] == coding and r["p"] == reference_pattern_count), None)
        for coding in codings
    }

    statistics = {}
    for left, right in (("binary", "ternary"), ("binary", "rate"), ("ternary", "rate")):
        left_values = [record["cosine"] for record in trial_records if record["coding"] == left and record["p"] == reference_pattern_count]
        right_values = [record["cosine"] for record in trial_records if record["coding"] == right and record["p"] == reference_pattern_count]
        if left_values and right_values:
            statistics[f"{left}_vs_{right}_reference_cosine"] = independent_difference_stats(
                left_values,
                right_values,
                SEED,
            )

    finished = utc_now_iso()
    duration = time.time() - t0

    record = build_run_record(
        simulation_name="rate_coded_spike",
        script_path=SCRIPT_PATH,
        started_at_utc=started,
        finished_at_utc=finished,
        duration_sec=duration,
        parameters={
            "n_neurons": N_NEURONS,
            "n_patterns_list": N_PATTERNS_LIST,
            "corruption": CORRUPTION,
            "trials": TRIALS,
            "max_iters": MAX_ITERS,
            "codings": codings,
            "reference_pattern_count": reference_pattern_count,
        },
        seed_numpy=SEED,
        n_trials=len(trial_records),
        summary=summary,
        statistics=statistics,
        trials=trial_records,
        artifacts=[
            {"path": fig_path.as_posix(), "type": "figure"},
            {"path": (output_dir / "rate_coded_spike_metrics.json").as_posix(), "type": "metrics"},
        ],
        warnings=[
            "rate-coded retrieval uses tanh activation (continuous state update), not sign",
            "exact match for rate-coded uses atol=0.05 (approximate, not exact)",
        ],
    )

    metrics_path = output_dir / "rate_coded_spike_metrics.json"
    write_json(metrics_path, record)

    print(f"\ndone in {duration:.1f}s")
    for coding in codings:
        cosine = summary["reference_cosine"][coding]
        if cosine is None:
            print(f"  {coding}: no data at p={reference_pattern_count}")
        else:
            print(f"  {coding} at p={reference_pattern_count}: cosine={cosine:.4f}")


if __name__ == "__main__":
    main()
