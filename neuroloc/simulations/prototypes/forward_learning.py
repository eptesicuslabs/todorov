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
D = 128
N_PAIRS_LIST = [10, 30, 50, 80, 100, 128]
EPOCHS = 100
TRIALS = 10
LR = 0.01


def generate_structured_task(rng, d, n_pairs):
    M_true = rng.standard_normal((d, d)) * 0.1
    keys = rng.standard_normal((n_pairs, d))
    keys = keys / np.linalg.norm(keys, axis=1, keepdims=True)
    values = (M_true @ keys.T).T
    values = values / np.linalg.norm(values, axis=1, keepdims=True)
    return keys, values, M_true


def train_gradient_descent(keys, values, d, lr, epochs):
    W = np.zeros((d, d))
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(len(keys)):
            error = W @ keys[i] - values[i]
            W -= lr * np.outer(error, keys[i])
            epoch_loss += float(np.sum(error ** 2))
        losses.append(epoch_loss / len(keys))
    return W, losses


def train_one_shot_hebbian(keys, values, d):
    n = len(keys)
    W = (values.T @ keys) / n
    return W, [0.0]


def train_one_shot_pseudoinverse(keys, values, d):
    W = values.T @ np.linalg.pinv(keys.T)
    return W, [0.0]


def train_forward_hebbian_with_decay(keys, values, d, lr, epochs):
    W = np.zeros((d, d))
    losses = []
    decay = 0.01
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(len(keys)):
            predicted = W @ keys[i]
            W = (1.0 - decay) * W + lr * np.outer(values[i], keys[i])
            error = predicted - values[i]
            epoch_loss += float(np.sum(error ** 2))
        losses.append(epoch_loss / len(keys))
    return W, losses


def train_delta_rule_local(keys, values, d, lr, epochs):
    W = np.zeros((d, d))
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(len(keys)):
            predicted = W @ keys[i]
            error = values[i] - predicted
            W += lr * np.outer(error, keys[i])
            epoch_loss += float(np.sum(error ** 2))
        losses.append(epoch_loss / len(keys))
    return W, losses


def evaluate(W, keys, values):
    cosines = []
    mses = []
    for i in range(len(keys)):
        predicted = W @ keys[i]
        norm_p = np.linalg.norm(predicted)
        norm_v = np.linalg.norm(values[i])
        if norm_p > 0 and norm_v > 0:
            cosines.append(float(np.dot(predicted, values[i]) / (norm_p * norm_v)))
        mses.append(float(np.sum((predicted - values[i]) ** 2)))
    return float(np.mean(cosines)), float(np.mean(mses))


def main():
    apply_plot_style()
    import matplotlib.pyplot as plt

    started = utc_now_iso()
    t0 = time.time()
    rng = build_rng(SEED)

    methods = ["gradient_descent", "one_shot_hebbian", "pseudoinverse", "forward_hebbian", "delta_rule_local"]
    all_results = []

    for n_pairs in N_PAIRS_LIST:
        for trial in range(TRIALS):
            trial_rng = child_rng(rng)
            keys, values, M_true = generate_structured_task(trial_rng, D, n_pairs)

            W_gd, losses_gd = train_gradient_descent(keys, values, D, LR, EPOCHS)
            cos_gd, mse_gd = evaluate(W_gd, keys, values)

            W_hebb, _ = train_one_shot_hebbian(keys, values, D)
            cos_hebb, mse_hebb = evaluate(W_hebb, keys, values)

            W_pinv, _ = train_one_shot_pseudoinverse(keys, values, D)
            cos_pinv, mse_pinv = evaluate(W_pinv, keys, values)

            W_fh, losses_fh = train_forward_hebbian_with_decay(keys, values, D, LR, EPOCHS)
            cos_fh, mse_fh = evaluate(W_fh, keys, values)

            W_dr, losses_dr = train_delta_rule_local(keys, values, D, LR, EPOCHS)
            cos_dr, mse_dr = evaluate(W_dr, keys, values)

            for method, cos, mse in [
                ("gradient_descent", cos_gd, mse_gd),
                ("one_shot_hebbian", cos_hebb, mse_hebb),
                ("pseudoinverse", cos_pinv, mse_pinv),
                ("forward_hebbian", cos_fh, mse_fh),
                ("delta_rule_local", cos_dr, mse_dr),
            ]:
                all_results.append({
                    "method": method,
                    "n_pairs": n_pairs,
                    "trial": trial,
                    "train_cosine": cos,
                    "train_mse": mse,
                })

        for method in methods:
            cos = np.mean([r["train_cosine"] for r in all_results if r["method"] == method and r["n_pairs"] == n_pairs])
            print(f"  {method} n={n_pairs}: train_cosine={cos:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    for method in methods:
        cosines = [np.mean([r["train_cosine"] for r in all_results if r["method"] == method and r["n_pairs"] == n]) for n in N_PAIRS_LIST]
        ax1.plot(N_PAIRS_LIST, cosines, marker="o", label=method)
    ax1.set_xlabel("number of stored pairs")
    ax1.set_ylabel("train cosine (recall quality)")
    ax1.set_title(f"memorization: gradient vs hebbian vs pseudoinverse (d={D})")
    ax1.legend(fontsize=8)

    ax2 = axes[1]
    for method in methods:
        mses = [np.mean([r["train_mse"] for r in all_results if r["method"] == method and r["n_pairs"] == n]) for n in N_PAIRS_LIST]
        ax2.plot(N_PAIRS_LIST, mses, marker="o", label=method)
    ax2.set_xlabel("number of stored pairs")
    ax2.set_ylabel("train MSE")
    ax2.set_title("memorization error vs capacity")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig_path = SCRIPT_PATH.parent / "forward_learning.png"
    plt.savefig(fig_path)
    plt.close()

    summary = {}
    for method in methods:
        for n in [50, 128]:
            cosines = [r["train_cosine"] for r in all_results if r["method"] == method and r["n_pairs"] == n]
            summary[f"{method}_n{n}_cosine"] = mean_confidence_interval(cosines)

    gd_128 = [r["train_cosine"] for r in all_results if r["method"] == "gradient_descent" and r["n_pairs"] == 128]
    hebb_128 = [r["train_cosine"] for r in all_results if r["method"] == "one_shot_hebbian" and r["n_pairs"] == 128]
    fh_128 = [r["train_cosine"] for r in all_results if r["method"] == "forward_hebbian" and r["n_pairs"] == 128]
    if len(gd_128) == len(hebb_128):
        summary["gd_vs_hebbian_n128"] = paired_difference_stats(hebb_128, gd_128, SEED)
    if len(gd_128) == len(fh_128):
        summary["gd_vs_forward_hebbian_n128"] = paired_difference_stats(fh_128, gd_128, SEED)

    finished = utc_now_iso()
    duration = time.time() - t0

    record = build_run_record(
        simulation_name="forward_learning_comparison",
        script_path=SCRIPT_PATH,
        started_at_utc=started,
        finished_at_utc=finished,
        duration_sec=duration,
        parameters={
            "d": D,
            "n_pairs_list": N_PAIRS_LIST,
            "epochs": EPOCHS,
            "trials": TRIALS,
            "lr": LR,
        },
        seed_numpy=SEED,
        n_trials=len(all_results),
        summary=summary,
        statistics={},
        trials=all_results,
        artifacts=[
            {"name": "forward_learning.png", "type": "figure"},
            {"name": "forward_learning_metrics.json", "type": "metrics"},
        ],
        warnings=[
            "task is structured: values = M_true @ keys (learnable linear function)",
            "evaluates on TRAINING keys (memorization recall), not test keys",
            "pseudoinverse is the theoretical optimum for linear association",
            "forward_hebbian uses outer product with mild decay -- genuinely local, no error signal",
        ],
    )

    metrics_path = SCRIPT_PATH.parent / "forward_learning_metrics.json"
    write_json(metrics_path, record)

    print(f"\ndone in {duration:.1f}s")
    for method in methods:
        cos = np.mean([r["train_cosine"] for r in all_results if r["method"] == method and r["n_pairs"] == 128])
        print(f"  {method} at n=128: cosine={cos:.4f}")


if __name__ == "__main__":
    main()
