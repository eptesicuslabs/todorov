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
N = 500
P = 30
SPARSITY_LEVELS = [0.01, 0.05, 0.10, 0.20, 0.50, 1.0]
TRIALS = 20
MAX_ITERS = 30
CORRUPTION = 0.10


def generate_patterns(rng, n, p):
    return rng.choice([-1, 1], size=(p, n)).astype(np.float64)


def make_dense_weights(patterns):
    p, n = patterns.shape
    W = (patterns.T @ patterns) / n
    np.fill_diagonal(W, 0.0)
    return W


def make_random_sparse(W, rng, sparsity):
    n = W.shape[0]
    n_connections = int(n * n * sparsity / 2)
    mask = np.zeros((n, n), dtype=bool)
    upper_indices = np.triu_indices(n, k=1)
    total_possible = len(upper_indices[0])
    chosen = rng.choice(total_possible, size=min(n_connections, total_possible), replace=False)
    mask[upper_indices[0][chosen], upper_indices[1][chosen]] = True
    mask = mask | mask.T
    np.fill_diagonal(mask, False)
    return W * mask


def make_smallworld(W, rng, sparsity, rewire_prob=0.1):
    n = W.shape[0]
    k = max(2, int(n * sparsity))
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(1, k // 2 + 1):
            mask[i, (i + j) % n] = True
            mask[i, (i - j) % n] = True
    for i in range(n):
        for j in range(1, k // 2 + 1):
            if rng.random() < rewire_prob:
                mask[i, (i + j) % n] = False
                new_target = rng.integers(0, n)
                while new_target == i or mask[i, new_target]:
                    new_target = rng.integers(0, n)
                mask[i, new_target] = True
    mask = mask | mask.T
    np.fill_diagonal(mask, False)
    return W * mask


def retrieve(W, probe, max_iters=30):
    state = probe.copy()
    for _ in range(max_iters):
        prev = state.copy()
        state = np.sign(W @ state)
        state[state == 0] = 1.0
        if np.array_equal(state, prev):
            break
    return state


def path_length_estimate(mask, n_samples=50, rng=None):
    n = mask.shape[0]
    if rng is None:
        indices = list(range(min(n_samples, n)))
    else:
        indices = rng.choice(n, size=min(n_samples, n), replace=False)
    lengths = []
    for src in indices:
        visited = {src}
        frontier = {src}
        depth = 0
        while frontier and depth < 20:
            depth += 1
            next_frontier = set()
            for node in frontier:
                neighbors = set(np.where(mask[node])[0])
                new_nodes = neighbors - visited
                next_frontier.update(new_nodes)
                visited.update(new_nodes)
            frontier = next_frontier
            if len(visited) > n * 0.9:
                break
        if len(visited) > 1:
            lengths.append(depth)
    return float(np.mean(lengths)) if lengths else float("inf")


def clustering_coefficient(mask, n_samples=50, rng=None):
    n = mask.shape[0]
    if rng is None:
        indices = list(range(min(n_samples, n)))
    else:
        indices = rng.choice(n, size=min(n_samples, n), replace=False)
    coeffs = []
    for i in indices:
        neighbors = np.where(mask[i])[0]
        k = len(neighbors)
        if k < 2:
            continue
        links = 0
        for a in range(k):
            for b in range(a + 1, k):
                if mask[neighbors[a], neighbors[b]]:
                    links += 1
        coeffs.append(2.0 * links / (k * (k - 1)))
    return float(np.mean(coeffs)) if coeffs else 0.0


def main():
    apply_plot_style()
    import matplotlib.pyplot as plt

    started = utc_now_iso()
    t0 = time.time()
    rng = build_rng(SEED)

    all_trials = []

    for sparsity in SPARSITY_LEVELS:
        for topology in ["dense", "random_sparse", "smallworld"]:
            if topology == "dense" and sparsity < 1.0:
                continue
            if topology != "dense" and sparsity >= 1.0:
                continue

            exact_rates = []
            overlaps = []

            for trial in range(TRIALS):
                trial_rng = child_rng(rng)
                patterns = generate_patterns(trial_rng, N, P)
                W_full = make_dense_weights(patterns)

                if topology == "dense":
                    W = W_full
                elif topology == "random_sparse":
                    W = make_random_sparse(W_full, trial_rng, sparsity)
                elif topology == "smallworld":
                    W = make_smallworld(W_full, trial_rng, sparsity)

                trial_exact = 0
                trial_overlap = 0
                n_test = min(5, P)
                for i in range(n_test):
                    corrupted = patterns[i].copy()
                    n_flip = int(N * CORRUPTION)
                    flip_idx = trial_rng.choice(N, size=n_flip, replace=False)
                    corrupted[flip_idx] *= -1
                    retrieved = retrieve(W, corrupted, MAX_ITERS)
                    trial_exact += float(np.array_equal(retrieved, patterns[i]))
                    trial_overlap += float(np.dot(retrieved, patterns[i]) / N)

                exact_rates.append(trial_exact / n_test)
                overlaps.append(trial_overlap / n_test)

            mask = (W != 0).astype(bool) if topology != "dense" else np.ones((N, N), dtype=bool)
            np.fill_diagonal(mask, False)
            actual_sparsity = float(np.mean(mask))
            cc = clustering_coefficient(mask, rng=child_rng(rng))
            pl = path_length_estimate(mask, rng=child_rng(rng))

            result = {
                "topology": topology,
                "target_sparsity": sparsity,
                "actual_sparsity": actual_sparsity,
                "exact_retrieval": mean_confidence_interval(exact_rates),
                "overlap": mean_confidence_interval(overlaps),
                "clustering_coefficient": cc,
                "path_length": pl,
            }
            all_trials.append(result)

            print(f"{topology} sp={sparsity:.2f}: exact={np.mean(exact_rates):.3f} overlap={np.mean(overlaps):.3f} CC={cc:.3f} PL={pl:.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    random_sp = [r["target_sparsity"] for r in all_trials if r["topology"] == "random_sparse"]
    random_ex = [r["exact_retrieval"]["mean"] for r in all_trials if r["topology"] == "random_sparse"]
    sw_sp = [r["target_sparsity"] for r in all_trials if r["topology"] == "smallworld"]
    sw_ex = [r["exact_retrieval"]["mean"] for r in all_trials if r["topology"] == "smallworld"]
    dense_ex = [r["exact_retrieval"]["mean"] for r in all_trials if r["topology"] == "dense"]

    ax1 = axes[0]
    ax1.plot(random_sp, random_ex, marker="o", label="random sparse")
    ax1.plot(sw_sp, sw_ex, marker="s", label="small-world")
    if dense_ex:
        ax1.axhline(y=dense_ex[0], color="red", linestyle="--", label="dense")
    ax1.set_xlabel("connection fraction")
    ax1.set_ylabel("exact retrieval rate")
    ax1.set_title("retrieval quality vs connectivity")
    ax1.legend()

    ax2 = axes[1]
    random_cc = [r["clustering_coefficient"] for r in all_trials if r["topology"] == "random_sparse"]
    sw_cc = [r["clustering_coefficient"] for r in all_trials if r["topology"] == "smallworld"]
    ax2.plot(random_sp, random_cc, marker="o", label="random sparse")
    ax2.plot(sw_sp, sw_cc, marker="s", label="small-world")
    ax2.set_xlabel("connection fraction")
    ax2.set_ylabel("clustering coefficient")
    ax2.set_title("topology: clustering")
    ax2.legend()

    ax3 = axes[2]
    random_pl = [r["path_length"] for r in all_trials if r["topology"] == "random_sparse"]
    sw_pl = [r["path_length"] for r in all_trials if r["topology"] == "smallworld"]
    ax3.plot(random_sp, random_pl, marker="o", label="random sparse")
    ax3.plot(sw_sp, sw_pl, marker="s", label="small-world")
    ax3.set_xlabel("connection fraction")
    ax3.set_ylabel("mean path length")
    ax3.set_title("topology: path length")
    ax3.legend()

    plt.tight_layout()
    fig_path = SCRIPT_PATH.parent / "sparse_topology.png"
    plt.savefig(fig_path)
    plt.close()

    summary = {}
    for r in all_trials:
        key = f"{r['topology']}_sp{r['target_sparsity']}"
        summary[key] = {
            "exact_retrieval": r["exact_retrieval"]["mean"],
            "overlap": r["overlap"]["mean"],
            "clustering": r["clustering_coefficient"],
            "path_length": r["path_length"],
        }

    finished = utc_now_iso()
    duration = time.time() - t0

    record = build_run_record(
        simulation_name="sparse_topology",
        script_path=SCRIPT_PATH,
        started_at_utc=started,
        finished_at_utc=finished,
        duration_sec=duration,
        parameters={
            "n": N, "p": P, "sparsity_levels": SPARSITY_LEVELS,
            "trials": TRIALS, "corruption": CORRUPTION,
        },
        seed_numpy=SEED,
        n_trials=len(all_trials),
        summary=summary,
        statistics={},
        trials=all_trials,
        artifacts=[
            {"name": "sparse_topology.png", "type": "figure"},
            {"name": "sparse_topology_metrics.json", "type": "metrics"},
        ],
        warnings=[],
    )

    write_json(SCRIPT_PATH.parent / "sparse_topology_metrics.json", record)
    print(f"\ndone in {duration:.1f}s")


if __name__ == "__main__":
    main()
