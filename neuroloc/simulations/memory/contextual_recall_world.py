from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

SIM_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
for p in [str(SIM_ROOT), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from shared import (
    apply_plot_style,
    build_run_record,
    env_float,
    env_int,
    mean_confidence_interval,
    output_dir_for,
    require_positive,
    require_unit_interval,
    utc_now_iso,
    write_json,
)
from neuroloc.data.nm_worlds import generate_nm_world_batch

SCRIPT_PATH = Path(__file__).resolve()
SEED = env_int("CTX_WORLD_SEED", 42)
N_EPISODES = env_int("CTX_WORLD_EPISODES", 64)
SEQ_LEN = env_int("CTX_WORLD_SEQ_LEN", 12)
N_IDENTITIES = env_int("CTX_WORLD_IDENTITIES", 12)
N_ACTIVE = env_int("CTX_WORLD_ACTIVE", 4)
TRACK_LENGTH = env_int("CTX_WORLD_TRACK_LENGTH", 21)
OCCLUSION_PROB = env_float("CTX_WORLD_OCCLUSION", 0.25)
FEATURE_DROPOUT = env_float("CTX_WORLD_FEATURE_DROPOUT", 0.35)

require_positive("CTX_WORLD_EPISODES", N_EPISODES)
require_positive("CTX_WORLD_SEQ_LEN", SEQ_LEN)
require_positive("CTX_WORLD_IDENTITIES", N_IDENTITIES)
require_positive("CTX_WORLD_ACTIVE", N_ACTIVE)
require_positive("CTX_WORLD_TRACK_LENGTH", TRACK_LENGTH)
require_unit_interval("CTX_WORLD_OCCLUSION", OCCLUSION_PROB, allow_zero=True)
require_unit_interval("CTX_WORLD_FEATURE_DROPOUT", FEATURE_DROPOUT, allow_zero=True)


def main() -> dict[str, object]:
    apply_plot_style()
    import matplotlib.pyplot as plt

    started = utc_now_iso()
    t0 = time.time()
    output_dir = output_dir_for(SCRIPT_PATH)

    episodes = generate_nm_world_batch(
        n_episodes=N_EPISODES,
        seed=SEED,
        seq_len=SEQ_LEN,
        n_identities=N_IDENTITIES,
        n_active=N_ACTIVE,
        track_length=TRACK_LENGTH,
        occlusion_prob=OCCLUSION_PROB,
        feature_dropout_prob=FEATURE_DROPOUT,
    )

    candidate_counts = [int(ep["tasks"]["recognition"]["candidate_count"]) for ep in episodes]
    recollection_lags = [int(ep["tasks"]["recollection"]["lag"]) for ep in episodes]
    prediction_steps = [int(ep["tasks"]["prediction"]["step_distance"]) for ep in episodes]
    compression_ratios = [float(ep["tasks"]["compression"]["ratio"]) for ep in episodes]
    imagination_novel = [int(bool(ep["tasks"]["imagination"]["is_novel"])) for ep in episodes]
    imagination_plausible = [int(bool(ep["tasks"]["imagination"]["is_plausible"])) for ep in episodes]
    reasoning_margins = [float(ep["tasks"]["reasoning"]["margin"]) for ep in episodes]
    reasoning_winners = [int(ep["tasks"]["reasoning"]["winner_local_index"]) for ep in episodes]

    recognition_targets = [int(ep["tasks"]["recognition"]["target_identity"]) for ep in episodes]
    recognition_hist = np.bincount(np.array(recognition_targets, dtype=np.int64), minlength=N_IDENTITIES)
    probs = recognition_hist / max(int(recognition_hist.sum()), 1)
    nonzero_probs = probs[probs > 0.0]
    recognition_entropy_bits = float(-(nonzero_probs * np.log2(nonzero_probs)).sum()) if nonzero_probs.size else 0.0

    trials = []
    for idx, ep in enumerate(episodes):
        trials.append(
            {
                "episode": int(idx),
                "recognition_candidate_count": int(ep["tasks"]["recognition"]["candidate_count"]),
                "recollection_lag": int(ep["tasks"]["recollection"]["lag"]),
                "prediction_step_distance": int(ep["tasks"]["prediction"]["step_distance"]),
                "compression_ratio": float(ep["tasks"]["compression"]["ratio"]),
                "imagination_is_novel": int(bool(ep["tasks"]["imagination"]["is_novel"])),
                "imagination_is_plausible": int(bool(ep["tasks"]["imagination"]["is_plausible"])),
                "reasoning_margin": float(ep["tasks"]["reasoning"]["margin"]),
                "reasoning_winner_local_index": int(ep["tasks"]["reasoning"]["winner_local_index"]),
            }
        )

    summary = {
        "recognition_mean_candidate_count": float(np.mean(candidate_counts)),
        "recognition_ambiguity_rate": float(np.mean(np.array(candidate_counts) > 1)),
        "recognition_identity_entropy_bits": recognition_entropy_bits,
        "recollection_mean_lag": float(np.mean(recollection_lags)),
        "prediction_mean_step_distance": float(np.mean(prediction_steps)),
        "compression_mean_ratio": float(np.mean(compression_ratios)),
        "imagination_novelty_rate": float(np.mean(imagination_novel)),
        "imagination_plausibility_rate": float(np.mean(imagination_plausible)),
        "reasoning_mean_margin": float(np.mean(reasoning_margins)),
        "reasoning_winner_balance": mean_confidence_interval(reasoning_winners, bounds=(0.0, float(max(N_ACTIVE - 1, 1)))),
    }
    statistics = {
        "recognition_candidate_count": mean_confidence_interval(candidate_counts, bounds=(1.0, float(N_IDENTITIES))),
        "recollection_lag": mean_confidence_interval(recollection_lags, bounds=(1.0, float(SEQ_LEN))),
        "prediction_step_distance": mean_confidence_interval(prediction_steps, bounds=(0.0, float(TRACK_LENGTH))),
        "compression_ratio": mean_confidence_interval(compression_ratios),
        "imagination_novel": mean_confidence_interval(imagination_novel, bounds=(0.0, 1.0)),
        "reasoning_margin": mean_confidence_interval(reasoning_margins),
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.reshape(-1)
    axes[0].hist(candidate_counts, bins=np.arange(1, max(candidate_counts) + 2) - 0.5, color="#1d4ed8", alpha=0.85)
    axes[0].set_title("recognition candidate count")
    axes[0].set_xlabel("candidate ids")
    axes[0].set_ylabel("episodes")

    axes[1].hist(recollection_lags, bins=np.arange(1, max(recollection_lags) + 2) - 0.5, color="#0f766e", alpha=0.85)
    axes[1].set_title("recollection lag")
    axes[1].set_xlabel("steps")

    axes[2].hist(prediction_steps, bins=np.arange(0, max(prediction_steps) + 2) - 0.5, color="#7c3aed", alpha=0.85)
    axes[2].set_title("prediction step distance")
    axes[2].set_xlabel("position change")

    axes[3].hist(compression_ratios, bins=12, color="#b45309", alpha=0.85)
    axes[3].set_title("raw / latent ratio")
    axes[3].set_xlabel("ratio")

    axes[4].bar(["novel", "plausible"], [float(np.mean(imagination_novel)), float(np.mean(imagination_plausible))], color=["#dc2626", "#059669"])
    axes[4].set_ylim(0.0, 1.0)
    axes[4].set_title("imagination validity")

    axes[5].hist(reasoning_margins, bins=12, color="#111827", alpha=0.85)
    axes[5].set_title("reasoning margin")
    axes[5].set_xlabel("time-to-boundary margin")

    plt.tight_layout()
    figure_path = output_dir / "contextual_recall_world.png"
    plt.savefig(figure_path)
    plt.close()

    finished = utc_now_iso()
    duration = time.time() - t0
    metrics_path = output_dir / "contextual_recall_world_metrics.json"
    record = build_run_record(
        simulation_name="contextual_recall_world",
        script_path=SCRIPT_PATH,
        started_at_utc=started,
        finished_at_utc=finished,
        duration_sec=duration,
        parameters={
            "n_episodes": N_EPISODES,
            "seq_len": SEQ_LEN,
            "n_identities": N_IDENTITIES,
            "n_active": N_ACTIVE,
            "track_length": TRACK_LENGTH,
            "occlusion_prob": OCCLUSION_PROB,
            "feature_dropout_prob": FEATURE_DROPOUT,
        },
        seed_numpy=SEED,
        n_trials=N_EPISODES,
        summary=summary,
        statistics=statistics,
        trials=trials,
        artifacts=[
            {"name": "contextual_recall_world.png", "path": figure_path, "type": "figure"},
            {"name": "contextual_recall_world_metrics.json", "path": metrics_path, "type": "metrics"},
        ],
        warnings=[],
    )
    write_json(metrics_path, record)
    return summary


if __name__ == "__main__":
    main()
