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
    env_int,
    env_list,
    discrete_mutual_information,
    linear_cka,
    mean_confidence_interval,
    output_dir_for,
    paired_difference_stats,
    require_positive,
    require_positive_list,
    require_unit_interval_list,
    utc_now_iso,
    write_json,
)

SCRIPT_PATH = Path(__file__).resolve()
SEED = env_int("HTERN_SEED", 42)
D_VALUES = env_list("HTERN_D_VALUES", int, [64, 128, 256])
K_FRACTIONS = env_list("HTERN_K_FRACTIONS", float, [0.05, 0.10, 0.20, 0.41])
N_SAMPLES = env_int("HTERN_N_SAMPLES", 256)
TRIALS = env_int("HTERN_TRIALS", 16)

require_positive_list("HTERN_D_VALUES", D_VALUES)
require_unit_interval_list("HTERN_K_FRACTIONS", K_FRACTIONS)
require_positive("HTERN_N_SAMPLES", N_SAMPLES)
require_positive("HTERN_TRIALS", TRIALS)
if all(np.isclose(k_fraction, 0.41) for k_fraction in K_FRACTIONS):
    raise ValueError("HTERN_K_FRACTIONS must include at least one non-0.41 fraction")


def standard_ternary(x, alpha=1.0):
    threshold = alpha * np.mean(np.abs(x), axis=-1, keepdims=True)
    signs = np.sign(x)
    mask = np.abs(x) > threshold
    return signs * mask


def hierarchical_ternary(x, k_fraction):
    n, d = x.shape
    k = max(1, int(d * k_fraction))
    abs_x = np.abs(x)
    thresholds = np.partition(abs_x, -k, axis=-1)[:, -k]
    mask = abs_x >= thresholds[:, None]
    signs = np.sign(x)
    return signs * mask


def compute_metrics(x_original, x_quantized):
    n, d = x_original.shape

    firing_rate = float(np.mean(x_quantized != 0))

    mi_values = []
    for dim in range(min(8, d)):
        x_binned = np.digitize(x_original[:, dim], bins=np.linspace(-3, 3, 20))
        q_vals = x_quantized[:, dim].astype(int)
        mi = discrete_mutual_information(x_binned, q_vals)
        if mi is not None:
            mi_values.append(mi)

    mean_mi = float(np.mean(mi_values)) if mi_values else 0.0
    cka = linear_cka(x_original, x_quantized)

    bits_per_dim = 0.0
    if firing_rate > 0:
        p_nonzero = firing_rate
        p_zero = 1.0 - p_nonzero
        if p_zero > 0 and p_nonzero > 0:
            bits_per_dim = -(p_zero * np.log2(p_zero) + p_nonzero * np.log2(p_nonzero))
            if p_nonzero > 0:
                bits_per_dim += p_nonzero * 1.0

    return {
        "firing_rate": firing_rate,
        "mean_mi": mean_mi,
        "cka": cka if cka is not None else 0.0,
        "bits_per_dim": bits_per_dim,
    }


def main():
    apply_plot_style()
    import matplotlib.pyplot as plt

    started = utc_now_iso()
    t0 = time.time()
    rng = build_rng(SEED)

    all_trials = []

    for d in D_VALUES:
        for trial in range(TRIALS):
            trial_rng = child_rng(rng)
            x = trial_rng.standard_normal((N_SAMPLES, d))
            q_standard = standard_ternary(x)
            m_std = compute_metrics(x, q_standard)
            for k_frac in K_FRACTIONS:
                q_hierarchical = hierarchical_ternary(x, k_frac)
                m_hier = compute_metrics(x, q_hierarchical)
                all_trials.append({
                    "d": d,
                    "k_fraction": k_frac,
                    "trial": trial,
                    "standard_mi": m_std["mean_mi"],
                    "standard_cka": m_std["cka"],
                    "standard_fr": m_std["firing_rate"],
                    "standard_bpd": m_std["bits_per_dim"],
                    "hierarchical_mi": m_hier["mean_mi"],
                    "hierarchical_cka": m_hier["cka"],
                    "hierarchical_fr": m_hier["firing_rate"],
                    "hierarchical_bpd": m_hier["bits_per_dim"],
                })

    output_dir = output_dir_for(SCRIPT_PATH)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax1 = axes[0]
    for d in D_VALUES:
        mi_ratios = []
        for k in K_FRACTIONS:
            std_mi = np.mean([t["standard_mi"] for t in all_trials if t["d"] == d and t["k_fraction"] == k])
            hier_mi = np.mean([t["hierarchical_mi"] for t in all_trials if t["d"] == d and t["k_fraction"] == k])
            mi_ratios.append(hier_mi / max(std_mi, 1e-10))
        ax1.plot(K_FRACTIONS, mi_ratios, marker="o", label=f"d={d}")
    ax1.set_xlabel("k-wta fraction")
    ax1.set_ylabel("mi ratio (hierarchical / standard)")
    ax1.set_title("information retention vs sparsity")
    ax1.axhline(y=1.0, color="red", linestyle="--", alpha=0.3)
    ax1.legend()

    ax2 = axes[1]
    for d in D_VALUES:
        ckas = [np.mean([t["hierarchical_cka"] for t in all_trials if t["d"] == d and t["k_fraction"] == k]) for k in K_FRACTIONS]
        ax2.plot(K_FRACTIONS, ckas, marker="o", label=f"d={d}")
    ax2.set_xlabel("k-wta fraction")
    ax2.set_ylabel("cka (hierarchical vs original)")
    ax2.set_title("structural similarity preserved")
    ax2.legend()

    ax3 = axes[2]
    for d in D_VALUES:
        bpds = [np.mean([t["hierarchical_bpd"] for t in all_trials if t["d"] == d and t["k_fraction"] == k]) for k in K_FRACTIONS]
        ax3.plot(K_FRACTIONS, bpds, marker="o", label=f"d={d}")
    ax3.axhline(y=1.39, color="red", linestyle="--", alpha=0.3, label="standard ternary at 41% (~1.39)")
    ax3.set_xlabel("k-wta fraction")
    ax3.set_ylabel("bits per dimension")
    ax3.set_title("compression ratio")
    ax3.legend()

    plt.tight_layout()
    fig_path = output_dir / "hierarchical_ternary.png"
    plt.savefig(fig_path)
    plt.close()

    summary = {}
    paired_stats = {}
    reference_dimension = 256 if 256 in D_VALUES else max(D_VALUES)
    standard_anchor_fraction = float(K_FRACTIONS[0])
    selected_k_values = [k for k in K_FRACTIONS if not np.isclose(k, 0.41)]
    for k in selected_k_values:
        reference_records = [t for t in all_trials if np.isclose(t["k_fraction"], k) and t["d"] == reference_dimension]
        std_mi = [t["standard_mi"] for t in reference_records]
        hier_mi = [t["hierarchical_mi"] for t in reference_records]
        std_cka = [t["standard_cka"] for t in reference_records]
        hier_cka = [t["hierarchical_cka"] for t in reference_records]
        if std_mi and hier_mi:
            paired_stats[f"mi_k{k}_vs_standard"] = paired_difference_stats(std_mi, hier_mi, SEED)
        if std_cka and hier_cka:
            paired_stats[f"cka_k{k}_vs_standard"] = paired_difference_stats(std_cka, hier_cka, SEED)
        summary[f"k{k}_reference_mi"] = mean_confidence_interval(hier_mi)
        summary[f"k{k}_reference_cka"] = mean_confidence_interval(hier_cka)
        summary[f"k{k}_reference_bpd"] = mean_confidence_interval([t["hierarchical_bpd"] for t in reference_records])

    if selected_k_values:
        selected_fraction = float(selected_k_values[0])
        selected_records = [t for t in all_trials if np.isclose(t["k_fraction"], selected_fraction) and t["d"] == reference_dimension]
        summary["selected_hierarchical_fraction"] = selected_fraction
        summary["selected_hierarchical_reference_mi"] = mean_confidence_interval([t["hierarchical_mi"] for t in selected_records])
        summary["selected_hierarchical_reference_cka"] = mean_confidence_interval([t["hierarchical_cka"] for t in selected_records])

    summary["reference_dimension"] = int(reference_dimension)
    summary["standard_reference_mi"] = mean_confidence_interval(
        [t["standard_mi"] for t in all_trials if np.isclose(t["k_fraction"], standard_anchor_fraction) and t["d"] == reference_dimension]
    )
    summary["standard_reference_cka"] = mean_confidence_interval(
        [t["standard_cka"] for t in all_trials if np.isclose(t["k_fraction"], standard_anchor_fraction) and t["d"] == reference_dimension]
    )

    finished = utc_now_iso()
    duration = time.time() - t0

    record = build_run_record(
        simulation_name="hierarchical_ternary",
        script_path=SCRIPT_PATH,
        started_at_utc=started,
        finished_at_utc=finished,
        duration_sec=duration,
        parameters={
            "d_values": D_VALUES,
            "k_fractions": K_FRACTIONS,
            "n_samples": N_SAMPLES,
            "trials": TRIALS,
            "reference_dimension": reference_dimension,
        },
        seed_numpy=SEED,
        n_trials=len(all_trials),
        summary=summary,
        statistics=paired_stats,
        trials=all_trials,
        artifacts=[
            {"path": fig_path.as_posix(), "type": "figure"},
            {"path": (output_dir / "hierarchical_ternary_metrics.json").as_posix(), "type": "metrics"},
        ],
        warnings=["synthetic gaussian data, not trained model activations"],
    )

    metrics_path = output_dir / "hierarchical_ternary_metrics.json"
    write_json(metrics_path, record)

    print(f"done in {duration:.1f}s, {len(all_trials)} trials")
    for k in K_FRACTIONS:
        mi = np.mean([t["hierarchical_mi"] for t in all_trials if t["k_fraction"] == k and t["d"] == reference_dimension])
        cka = np.mean([t["hierarchical_cka"] for t in all_trials if t["k_fraction"] == k and t["d"] == reference_dimension])
        bpd = np.mean([t["hierarchical_bpd"] for t in all_trials if t["k_fraction"] == k and t["d"] == reference_dimension])
        print(f"  k={k:.2f}: MI={mi:.4f} CKA={cka:.4f} bits/dim={bpd:.3f}")


if __name__ == "__main__":
    main()
