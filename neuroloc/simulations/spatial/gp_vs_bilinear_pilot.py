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
    discrete_mutual_information,
    linear_cka,
    mean_confidence_interval,
    paired_difference_stats,
    utc_now_iso,
    write_json,
)

SCRIPT_PATH = Path(__file__).resolve()
SEED = 42
D_MODEL = 32
D_MV = 16
N_SAMPLES = 256
TRIALS = 16
NOISE_LEVELS = [0.0, 0.2, 0.5, 1.0]


GP_TABLE = None


def build_pga_cayley_table():
    table = np.zeros((16, 16, 16))

    signs = np.zeros((4, 4), dtype=int)
    signs[0] = [1, 1, 1, 1]
    signs[1] = [1, -1, 1, -1]
    signs[2] = [1, -1, -1, 1]
    signs[3] = [1, 1, -1, -1]

    for i in range(16):
        for j in range(16):
            grade_i = bin(i).count("1")
            grade_j = bin(j).count("1")
            grade_out = (grade_i + grade_j) % 5
            k = (i ^ j) % 16

            sign = 1
            if grade_i > 0 and grade_j > 0:
                si = min(grade_i, 3)
                sj = min(grade_j, 3)
                sign = signs[si][sj]
                if (i & j) != 0:
                    sign *= (-1) ** bin(i & j).count("1")

            table[i, j, k] += sign

    return table


def geometric_product(a, b, table):
    return np.einsum("ij,ik,jkl->il", a, b, table)


def quaternion_product(a, b):
    d = a.shape[1]
    out = np.zeros_like(a)
    for s in range(0, d, 4):
        e = min(s + 4, d)
        if e - s < 4:
            out[:, s:e] = a[:, s:e] * b[:, s:e]
            continue
        w1, x1, y1, z1 = a[:, s], a[:, s+1], a[:, s+2], a[:, s+3]
        w2, x2, y2, z2 = b[:, s], b[:, s+1], b[:, s+2], b[:, s+3]
        out[:, s] = w1*w2 - x1*x2 - y1*y2 - z1*z2
        out[:, s+1] = w1*x2 + x1*w2 + y1*z2 - z1*y2
        out[:, s+2] = w1*y2 - x1*z2 + y1*w2 + z1*x2
        out[:, s+3] = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return out


def run_trial(rng, method, d_model, d_mv, n_samples, noise_sigma):
    x = rng.standard_normal((n_samples, d_model))

    W_left = rng.standard_normal((d_model, d_mv)) * 0.02
    W_right = rng.standard_normal((d_model, d_mv)) * 0.02
    W_out = rng.standard_normal((d_mv, d_model)) * 0.02

    a = x @ W_left
    b = x @ W_right

    if noise_sigma > 0:
        a = a + rng.standard_normal(a.shape) * noise_sigma
        b = b + rng.standard_normal(b.shape) * noise_sigma

    if method == "pga":
        global GP_TABLE
        if GP_TABLE is None:
            GP_TABLE = build_pga_cayley_table()
        interaction = geometric_product(a, b, GP_TABLE)
    elif method == "quaternion":
        interaction = quaternion_product(a, b)
    elif method == "random_bilinear":
        B_random = rng.standard_normal((d_mv, d_mv, d_mv)) * 0.02
        interaction = np.einsum("ni,nj,ijk->nk", a, b, B_random)
    elif method == "elementwise":
        interaction = a * b
    else:
        raise ValueError(f"unknown method: {method}")

    output = x + interaction @ W_out

    x_binned = np.digitize(x[:, :8], bins=np.linspace(-3, 3, 20))
    out_binned = np.digitize(output[:, :8], bins=np.linspace(-3, 3, 20))

    mi_values = []
    for dim in range(min(8, d_model)):
        mi = discrete_mutual_information(x_binned[:, dim], out_binned[:, dim])
        if mi is not None:
            mi_values.append(mi)

    cka = linear_cka(x, output)

    output_norm = float(np.mean(np.linalg.norm(output, axis=-1)))
    interaction_norm = float(np.mean(np.linalg.norm(interaction, axis=-1)))
    output_var = float(np.var(output))

    return {
        "method": method,
        "noise_sigma": noise_sigma,
        "mean_mi": float(np.mean(mi_values)) if mi_values else 0.0,
        "cka": cka,
        "output_norm": output_norm,
        "interaction_norm": interaction_norm,
        "output_variance": output_var,
    }


def main():
    apply_plot_style()
    import matplotlib.pyplot as plt

    started = utc_now_iso()
    t0 = time.time()
    rng = build_rng(SEED)

    methods = ["pga", "quaternion", "random_bilinear", "elementwise"]
    all_trials = []

    for noise in NOISE_LEVELS:
        for method in methods:
            for trial in range(TRIALS):
                trial_rng = child_rng(rng)
                result = run_trial(trial_rng, method, D_MODEL, D_MV, N_SAMPLES, noise)
                result["trial"] = trial
                all_trials.append(result)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax1 = axes[0]
    for method in methods:
        mis = [np.mean([t["mean_mi"] for t in all_trials if t["method"] == method and t["noise_sigma"] == n]) for n in NOISE_LEVELS]
        ax1.plot(NOISE_LEVELS, mis, marker="o", label=method)
    ax1.set_xlabel("noise sigma")
    ax1.set_ylabel("mean mutual information (bits)")
    ax1.set_title("mi: input to output")
    ax1.legend()

    ax2 = axes[1]
    for method in methods:
        ckas = [np.mean([t["cka"] for t in all_trials if t["method"] == method and t["noise_sigma"] == n and t["cka"] is not None]) for n in NOISE_LEVELS]
        ax2.plot(NOISE_LEVELS, ckas, marker="o", label=method)
    ax2.set_xlabel("noise sigma")
    ax2.set_ylabel("linear CKA")
    ax2.set_title("representational similarity")
    ax2.legend()

    ax3 = axes[2]
    for method in methods:
        norms = [np.mean([t["interaction_norm"] for t in all_trials if t["method"] == method and t["noise_sigma"] == n]) for n in NOISE_LEVELS]
        ax3.plot(NOISE_LEVELS, norms, marker="o", label=method)
    ax3.set_xlabel("noise sigma")
    ax3.set_ylabel("interaction norm")
    ax3.set_title("interaction magnitude")
    ax3.legend()

    plt.tight_layout()
    fig_path = SCRIPT_PATH.parent / "gp_vs_bilinear_pilot.png"
    plt.savefig(fig_path)
    plt.close()

    summary = {}
    paired_stats = {}

    for noise in [0.0, 0.5]:
        pga_mi = [t["mean_mi"] for t in all_trials if t["method"] == "pga" and t["noise_sigma"] == noise]
        for other in ["quaternion", "random_bilinear", "elementwise"]:
            other_mi = [t["mean_mi"] for t in all_trials if t["method"] == other and t["noise_sigma"] == noise]
            key = f"mi_pga_vs_{other}_noise_{noise}"
            paired_stats[key] = paired_difference_stats(other_mi, pga_mi, SEED)

        pga_cka = [t["cka"] for t in all_trials if t["method"] == "pga" and t["noise_sigma"] == noise and t["cka"] is not None]
        for other in ["quaternion", "random_bilinear", "elementwise"]:
            other_cka = [t["cka"] for t in all_trials if t["method"] == other and t["noise_sigma"] == noise and t["cka"] is not None]
            key = f"cka_pga_vs_{other}_noise_{noise}"
            if pga_cka and other_cka and len(pga_cka) == len(other_cka):
                paired_stats[key] = paired_difference_stats(other_cka, pga_cka, SEED + 1)

    for method in methods:
        mi_vals = [t["mean_mi"] for t in all_trials if t["method"] == method and t["noise_sigma"] == 0.0]
        cka_vals = [t["cka"] for t in all_trials if t["method"] == method and t["noise_sigma"] == 0.0 and t["cka"] is not None]
        summary[f"{method}_mi_noise0"] = mean_confidence_interval(mi_vals)
        summary[f"{method}_cka_noise0"] = mean_confidence_interval(cka_vals)

    finished = utc_now_iso()
    duration = time.time() - t0

    record = build_run_record(
        simulation_name="gp_vs_bilinear_pilot",
        script_path=SCRIPT_PATH,
        started_at_utc=started,
        finished_at_utc=finished,
        duration_sec=duration,
        parameters={
            "d_model": D_MODEL,
            "d_mv": D_MV,
            "n_samples": N_SAMPLES,
            "trials": TRIALS,
            "noise_levels": NOISE_LEVELS,
            "methods": methods,
        },
        seed_numpy=SEED,
        n_trials=len(all_trials),
        summary=summary,
        statistics=paired_stats,
        trials=all_trials,
        artifacts=[
            {"name": "gp_vs_bilinear_pilot.png", "type": "figure"},
            {"name": "gp_vs_bilinear_pilot_metrics.json", "type": "metrics"},
        ],
        warnings=["this tests random projections, not trained weights -- results indicate structure of the bilinear map, not trained performance"],
    )

    metrics_path = SCRIPT_PATH.parent / "gp_vs_bilinear_pilot_metrics.json"
    write_json(metrics_path, record)

    print(f"done in {duration:.1f}s, {len(all_trials)} trials")
    for method in methods:
        mi = summary[f"{method}_mi_noise0"]["mean"]
        cka = summary[f"{method}_cka_noise0"]["mean"]
        print(f"{method}: MI={mi:.4f}, CKA={cka:.4f}")

    print("\npga vs others at noise=0:")
    for other in ["quaternion", "random_bilinear", "elementwise"]:
        key = f"mi_pga_vs_{other}_noise_0.0"
        if key in paired_stats:
            d = paired_stats[key].get("mean_difference", 0)
            p = paired_stats[key].get("p_value_permutation", 1)
            print(f"  pga vs {other}: MI delta={d:.4f}, p={p:.4f}")


if __name__ == "__main__":
    main()
