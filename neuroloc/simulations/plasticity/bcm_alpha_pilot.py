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
N_HEADS = 8
HEAD_DIM = 64
SEQ_LENGTHS = [64, 128, 256, 512, 1024]
GAMMA_VALUES = [0.0, 0.1, 0.3, 0.5, 1.0]
ALPHA_LOG_INIT = -2.0
BETA_MEAN = 0.5
TRIALS_PER_CONDITION = 24


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def run_kda_sequence(rng, seq_len, alpha_log, gamma, head_dim, n_heads):
    alpha_base = sigmoid(alpha_log)

    state = np.zeros((n_heads, head_dim, head_dim))
    state_norms = []
    effective_alphas = []

    for t in range(seq_len):
        k = rng.standard_normal((n_heads, head_dim))
        v = rng.standard_normal((n_heads, head_dim))
        beta = sigmoid(rng.standard_normal(n_heads) * 0.5 + BETA_MEAN)

        if gamma > 0:
            state_norm = np.linalg.norm(state.reshape(n_heads, -1), axis=-1, keepdims=True)
            log_norm = np.log(np.maximum(state_norm, 1e-6))
            alpha_eff = sigmoid(alpha_log + gamma * log_norm.squeeze())
        else:
            alpha_eff = np.full(n_heads, alpha_base)

        effective_alphas.append(alpha_eff.copy())

        for h in range(n_heads):
            state[h] = alpha_eff[h] * state[h] + beta[h] * np.outer(k[h], v[h])

        norms = np.linalg.norm(state.reshape(n_heads, -1), axis=-1)
        state_norms.append(norms.copy())

    return {
        "final_state_norm": float(np.mean(norms)),
        "max_state_norm": float(np.max(np.array(state_norms))),
        "mean_state_norm": float(np.mean(np.array(state_norms))),
        "state_norm_std": float(np.std(np.array(state_norms)[-seq_len // 4 :])),
        "norm_growth_ratio": float(np.mean(np.array(state_norms)[-1]) / max(np.mean(np.array(state_norms)[0]), 1e-10)),
        "mean_alpha_eff": float(np.mean(np.array(effective_alphas))),
        "alpha_range": float(np.max(np.array(effective_alphas)) - np.min(np.array(effective_alphas))),
    }


def run_retrieval_test(rng, seq_len, alpha_log, gamma, head_dim, n_heads, n_queries=16):
    alpha_base = sigmoid(alpha_log)

    state = np.zeros((n_heads, head_dim, head_dim))
    stored_keys = []
    stored_values = []

    for t in range(seq_len):
        k = rng.standard_normal((n_heads, head_dim))
        k = k / np.linalg.norm(k, axis=-1, keepdims=True)
        v = rng.standard_normal((n_heads, head_dim))
        beta = sigmoid(rng.standard_normal(n_heads) * 0.5 + BETA_MEAN)

        if gamma > 0:
            state_norm = np.linalg.norm(state.reshape(n_heads, -1), axis=-1, keepdims=True)
            log_norm = np.log(np.maximum(state_norm, 1e-6))
            alpha_eff = sigmoid(alpha_log + gamma * log_norm.squeeze())
        else:
            alpha_eff = np.full(n_heads, alpha_base)

        for h in range(n_heads):
            state[h] = alpha_eff[h] * state[h] + beta[h] * np.outer(k[h], v[h])

        if t >= seq_len - n_queries:
            stored_keys.append(k.copy())
            stored_values.append(v.copy())

    cosine_sims = []
    for i, (k_stored, v_stored) in enumerate(zip(stored_keys, stored_values)):
        for h in range(n_heads):
            retrieved = state[h] @ k_stored[h]
            norm_r = np.linalg.norm(retrieved)
            norm_v = np.linalg.norm(v_stored[h])
            if norm_r > 0 and norm_v > 0:
                cosine_sims.append(float(np.dot(retrieved, v_stored[h]) / (norm_r * norm_v)))

    return {
        "mean_retrieval_cosine": float(np.mean(cosine_sims)) if cosine_sims else 0.0,
        "retrieval_cosine_std": float(np.std(cosine_sims)) if cosine_sims else 0.0,
    }


def main():
    apply_plot_style()
    import matplotlib.pyplot as plt

    started = utc_now_iso()
    t0 = time.time()
    rng = build_rng(SEED)

    all_trials = []
    norm_results = {}
    retrieval_results = {}

    for seq_len in SEQ_LENGTHS:
        for gamma in GAMMA_VALUES:
            label = f"gamma_{gamma}_seq_{seq_len}"
            norms_trials = []
            retrieval_trials = []

            for trial in range(TRIALS_PER_CONDITION):
                trial_rng = child_rng(rng)

                norm_result = run_kda_sequence(
                    trial_rng, seq_len, ALPHA_LOG_INIT, gamma, HEAD_DIM, N_HEADS
                )

                ret_rng = child_rng(rng)
                ret_result = run_retrieval_test(
                    ret_rng, seq_len, ALPHA_LOG_INIT, gamma, HEAD_DIM, N_HEADS
                )

                trial_record = {
                    "experiment": "bcm_alpha",
                    "seq_len": seq_len,
                    "gamma": gamma,
                    "trial": trial,
                    **norm_result,
                    **ret_result,
                }
                all_trials.append(trial_record)
                norms_trials.append(norm_result["state_norm_std"])
                retrieval_trials.append(ret_result["mean_retrieval_cosine"])

            norm_results[label] = norms_trials
            retrieval_results[label] = retrieval_trials

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    for gamma in GAMMA_VALUES:
        stds = [np.mean(norm_results[f"gamma_{gamma}_seq_{sl}"]) for sl in SEQ_LENGTHS]
        ax1.plot(SEQ_LENGTHS, stds, marker="o", label=f"gamma={gamma}")
    ax1.set_xlabel("sequence length")
    ax1.set_ylabel("state norm std (last quarter)")
    ax1.set_title("state stability vs sequence length")
    ax1.legend()
    ax1.set_xscale("log", base=2)

    ax2 = axes[0, 1]
    for gamma in GAMMA_VALUES:
        cosines = [np.mean(retrieval_results[f"gamma_{gamma}_seq_{sl}"]) for sl in SEQ_LENGTHS]
        ax2.plot(SEQ_LENGTHS, cosines, marker="o", label=f"gamma={gamma}")
    ax2.set_xlabel("sequence length")
    ax2.set_ylabel("mean retrieval cosine similarity")
    ax2.set_title("retrieval quality vs sequence length")
    ax2.legend()
    ax2.set_xscale("log", base=2)

    ax3 = axes[1, 0]
    longest = SEQ_LENGTHS[-1]
    gamma_stds = [np.mean(norm_results[f"gamma_{g}_seq_{longest}"]) for g in GAMMA_VALUES]
    gamma_rets = [np.mean(retrieval_results[f"gamma_{g}_seq_{longest}"]) for g in GAMMA_VALUES]
    ax3.bar(range(len(GAMMA_VALUES)), gamma_stds, tick_label=[str(g) for g in GAMMA_VALUES])
    ax3.set_xlabel("gamma")
    ax3.set_ylabel("state norm std")
    ax3.set_title(f"stability at seq_len={longest}")

    ax4 = axes[1, 1]
    ax4.bar(range(len(GAMMA_VALUES)), gamma_rets, tick_label=[str(g) for g in GAMMA_VALUES])
    ax4.set_xlabel("gamma")
    ax4.set_ylabel("retrieval cosine")
    ax4.set_title(f"retrieval at seq_len={longest}")

    plt.tight_layout()
    fig_path = SCRIPT_PATH.parent / "bcm_alpha_pilot.png"
    plt.savefig(fig_path)
    plt.close()

    baseline_stds = [t["state_norm_std"] for t in all_trials if t["gamma"] == 0.0 and t["seq_len"] == SEQ_LENGTHS[-1]]
    best_gamma = None
    best_improvement = 0.0
    paired_stats = {}

    for gamma in GAMMA_VALUES:
        if gamma == 0.0:
            continue
        test_stds = [t["state_norm_std"] for t in all_trials if t["gamma"] == gamma and t["seq_len"] == SEQ_LENGTHS[-1]]
        ps = paired_difference_stats(baseline_stds, test_stds, SEED)
        paired_stats[f"stability_gamma_{gamma}"] = ps
        if ps["mean_difference"] is not None and ps["mean_difference"] < best_improvement:
            best_improvement = ps["mean_difference"]
            best_gamma = gamma

    baseline_rets = [t["mean_retrieval_cosine"] for t in all_trials if t["gamma"] == 0.0 and t["seq_len"] == SEQ_LENGTHS[-1]]
    for gamma in GAMMA_VALUES:
        if gamma == 0.0:
            continue
        test_rets = [t["mean_retrieval_cosine"] for t in all_trials if t["gamma"] == gamma and t["seq_len"] == SEQ_LENGTHS[-1]]
        ps = paired_difference_stats(baseline_rets, test_rets, SEED + 1)
        paired_stats[f"retrieval_gamma_{gamma}"] = ps

    summary = {
        "best_gamma_for_stability": best_gamma,
        "best_stability_improvement": best_improvement,
        "baseline_stability": mean_confidence_interval(baseline_stds),
        "baseline_retrieval": mean_confidence_interval(baseline_rets),
    }

    for gamma in GAMMA_VALUES:
        if gamma == 0.0:
            continue
        key_stab = f"stability_gamma_{gamma}"
        key_ret = f"retrieval_gamma_{gamma}"
        summary[f"gamma_{gamma}_stability_delta"] = paired_stats[key_stab].get("mean_difference")
        summary[f"gamma_{gamma}_stability_p"] = paired_stats[key_stab].get("p_value_permutation")
        summary[f"gamma_{gamma}_retrieval_delta"] = paired_stats[key_ret].get("mean_difference")
        summary[f"gamma_{gamma}_retrieval_p"] = paired_stats[key_ret].get("p_value_permutation")

    finished = utc_now_iso()
    duration = time.time() - t0

    record = build_run_record(
        simulation_name="bcm_alpha_pilot",
        script_path=SCRIPT_PATH,
        started_at_utc=started,
        finished_at_utc=finished,
        duration_sec=duration,
        parameters={
            "n_heads": N_HEADS,
            "head_dim": HEAD_DIM,
            "seq_lengths": SEQ_LENGTHS,
            "gamma_values": GAMMA_VALUES,
            "alpha_log_init": ALPHA_LOG_INIT,
            "beta_mean": BETA_MEAN,
            "trials_per_condition": TRIALS_PER_CONDITION,
        },
        seed_numpy=SEED,
        n_trials=len(all_trials),
        summary=summary,
        statistics=paired_stats,
        trials=all_trials,
        artifacts=[
            {"name": "bcm_alpha_pilot.png", "type": "figure"},
            {"name": "bcm_alpha_pilot_metrics.json", "type": "metrics"},
        ],
        warnings=[],
    )

    metrics_path = SCRIPT_PATH.parent / "bcm_alpha_pilot_metrics.json"
    write_json(metrics_path, record)

    print(f"done in {duration:.1f}s, {len(all_trials)} trials")
    print(f"best gamma for stability: {best_gamma} (delta: {best_improvement:.4f})")
    print(f"baseline stability (norm std): {summary['baseline_stability']['mean']:.4f}")
    print(f"baseline retrieval (cosine): {summary['baseline_retrieval']['mean']:.4f}")
    for gamma in GAMMA_VALUES:
        if gamma == 0.0:
            continue
        sd = summary.get(f"gamma_{gamma}_stability_delta")
        sp = summary.get(f"gamma_{gamma}_stability_p")
        rd = summary.get(f"gamma_{gamma}_retrieval_delta")
        rp = summary.get(f"gamma_{gamma}_retrieval_p")
        print(f"gamma={gamma}: stability delta={sd:.4f} p={sp:.4f}, retrieval delta={rd:.4f} p={rp:.4f}")


if __name__ == "__main__":
    main()
