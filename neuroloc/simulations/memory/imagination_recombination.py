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
N_NEURONS = 500
N_PATTERNS = 20
N_RECOMBINATIONS = 100
TRIALS = 24
CORRUPTION_FOR_RETRIEVAL = 0.05
MAX_ITERS = 30


def generate_patterns(rng, n, p):
    return rng.choice([-1, 1], size=(p, n)).astype(np.float64)


def store_patterns(patterns):
    p, n = patterns.shape
    W = (patterns.T @ patterns) / n
    np.fill_diagonal(W, 0.0)
    return W


def retrieve(W, probe, max_iters=30):
    state = probe.copy()
    for _ in range(max_iters):
        prev = state.copy()
        state = np.sign(W @ state)
        state[state == 0] = 1.0
        if np.array_equal(state, prev):
            break
    return state


def recombine(rng, pattern_a, pattern_b, mix_fraction=0.5):
    n = len(pattern_a)
    mask = rng.random(n) < mix_fraction
    combined = np.where(mask, pattern_a, pattern_b)
    return combined


def nearest_stored_overlap(retrieved, patterns):
    overlaps = patterns @ retrieved / len(retrieved)
    best_idx = np.argmax(np.abs(overlaps))
    return float(overlaps[best_idx]), int(best_idx)


def is_novel(retrieved, patterns, threshold=0.95):
    overlaps = np.abs(patterns @ retrieved / len(retrieved))
    return bool(np.max(overlaps) < threshold)


def main():
    apply_plot_style()
    import matplotlib.pyplot as plt

    started = utc_now_iso()
    t0 = time.time()
    rng = build_rng(SEED)

    all_trials = []
    retrieval_overlaps = []
    recombination_overlaps = []
    random_overlaps = []
    novelty_rates = []

    for trial in range(TRIALS):
        trial_rng = child_rng(rng)
        patterns = generate_patterns(trial_rng, N_NEURONS, N_PATTERNS)
        W = store_patterns(patterns)

        for i in range(min(5, N_PATTERNS)):
            corrupted = patterns[i].copy()
            n_flip = int(N_NEURONS * CORRUPTION_FOR_RETRIEVAL)
            flip_idx = trial_rng.choice(N_NEURONS, size=n_flip, replace=False)
            corrupted[flip_idx] *= -1
            retrieved = retrieve(W, corrupted, MAX_ITERS)
            overlap = float(np.dot(retrieved, patterns[i]) / N_NEURONS)
            retrieval_overlaps.append(overlap)

        novel_count = 0
        for r in range(N_RECOMBINATIONS):
            idx_a, idx_b = trial_rng.choice(N_PATTERNS, size=2, replace=False)
            mix = trial_rng.uniform(0.3, 0.7)
            combined = recombine(trial_rng, patterns[idx_a], patterns[idx_b], mix)
            imagined = retrieve(W, combined, MAX_ITERS)

            best_overlap, best_idx = nearest_stored_overlap(imagined, patterns)
            novel = is_novel(imagined, patterns, threshold=0.95)
            if novel:
                novel_count += 1

            parent_a_overlap = float(np.dot(imagined, patterns[idx_a]) / N_NEURONS)
            parent_b_overlap = float(np.dot(imagined, patterns[idx_b]) / N_NEURONS)

            recombination_overlaps.append(best_overlap)

            all_trials.append({
                "trial": trial,
                "recombination": r,
                "parent_a": int(idx_a),
                "parent_b": int(idx_b),
                "mix_fraction": float(mix),
                "best_overlap": best_overlap,
                "best_match_idx": best_idx,
                "parent_a_overlap": parent_a_overlap,
                "parent_b_overlap": parent_b_overlap,
                "is_novel": novel,
            })

        for _ in range(N_RECOMBINATIONS):
            random_probe = trial_rng.choice([-1, 1], size=N_NEURONS).astype(np.float64)
            random_retrieved = retrieve(W, random_probe, MAX_ITERS)
            best_overlap, _ = nearest_stored_overlap(random_retrieved, patterns)
            random_overlaps.append(best_overlap)

        novelty_rates.append(novel_count / N_RECOMBINATIONS)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax1 = axes[0]
    ax1.hist(retrieval_overlaps, bins=30, alpha=0.7, label="retrieval (5% corrupt)", color="blue")
    ax1.hist(recombination_overlaps, bins=30, alpha=0.7, label="recombination", color="orange")
    ax1.hist(random_overlaps, bins=30, alpha=0.7, label="random probe", color="gray")
    ax1.set_xlabel("overlap with nearest stored pattern")
    ax1.set_ylabel("count")
    ax1.set_title("retrieval vs recombination vs random")
    ax1.legend()

    ax2 = axes[1]
    parent_a_ovlps = [t["parent_a_overlap"] for t in all_trials]
    parent_b_ovlps = [t["parent_b_overlap"] for t in all_trials]
    ax2.scatter(parent_a_ovlps, parent_b_ovlps, alpha=0.1, s=5)
    ax2.set_xlabel("overlap with parent a")
    ax2.set_ylabel("overlap with parent b")
    ax2.set_title("recombined patterns: parent similarity")
    ax2.axhline(y=0, color="gray", alpha=0.3)
    ax2.axvline(x=0, color="gray", alpha=0.3)

    ax3 = axes[2]
    ax3.bar(["retrieval", "recombination", "random"],
            [np.mean(retrieval_overlaps), np.mean(recombination_overlaps), np.mean(random_overlaps)],
            yerr=[np.std(retrieval_overlaps), np.std(recombination_overlaps), np.std(random_overlaps)],
            capsize=5)
    ax3.set_ylabel("mean overlap with nearest stored")
    ax3.set_title(f"novelty rate: {np.mean(novelty_rates):.1%}")

    plt.tight_layout()
    fig_path = SCRIPT_PATH.parent / "imagination_recombination.png"
    plt.savefig(fig_path)
    plt.close()

    summary = {
        "retrieval_overlap": mean_confidence_interval(retrieval_overlaps),
        "recombination_overlap": mean_confidence_interval(recombination_overlaps),
        "random_overlap": mean_confidence_interval(random_overlaps),
        "novelty_rate": mean_confidence_interval(novelty_rates),
        "recomb_vs_random": paired_difference_stats(
            random_overlaps[:len(recombination_overlaps)],
            recombination_overlaps,
            SEED
        ),
    }

    finished = utc_now_iso()
    duration = time.time() - t0

    record = build_run_record(
        simulation_name="imagination_recombination",
        script_path=SCRIPT_PATH,
        started_at_utc=started,
        finished_at_utc=finished,
        duration_sec=duration,
        parameters={
            "n_neurons": N_NEURONS,
            "n_patterns": N_PATTERNS,
            "n_recombinations": N_RECOMBINATIONS,
            "trials": TRIALS,
            "corruption": CORRUPTION_FOR_RETRIEVAL,
            "max_iters": MAX_ITERS,
        },
        seed_numpy=SEED,
        n_trials=len(all_trials),
        summary=summary,
        statistics={},
        trials=all_trials,
        artifacts=[
            {"name": "imagination_recombination.png", "type": "figure"},
            {"name": "imagination_recombination_metrics.json", "type": "metrics"},
        ],
        warnings=["binary patterns only; ternary recombination not tested"],
    )

    metrics_path = SCRIPT_PATH.parent / "imagination_recombination_metrics.json"
    write_json(metrics_path, record)

    print(f"done in {duration:.1f}s")
    print(f"retrieval overlap: {np.mean(retrieval_overlaps):.4f}")
    print(f"recombination overlap: {np.mean(recombination_overlaps):.4f}")
    print(f"random overlap: {np.mean(random_overlaps):.4f}")
    print(f"novelty rate: {np.mean(novelty_rates):.1%}")


if __name__ == "__main__":
    main()
