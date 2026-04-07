from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy import stats

SCHEMA_VERSION = "neuroloc.sim.metrics/v1"


def apply_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
        }
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def find_project_root(path: Path) -> Path:
    resolved = path.resolve()
    for candidate in [resolved] + list(resolved.parents):
        if (candidate / "CLAUDE.md").exists() and (candidate / "requirements.txt").exists():
            return candidate
    return resolved.parent


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


def build_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def child_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))


def child_rng(rng: np.random.Generator) -> np.random.Generator:
    return np.random.default_rng(child_seed(rng))


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return [sanitize_json(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return sanitize_json(value.item())
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(sanitize_json(payload), indent=2) + "\n", encoding="utf-8")


def parameter_hash(parameters: dict[str, Any]) -> str:
    serialized = json.dumps(sanitize_json(parameters), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def mean_confidence_interval(values: Any, confidence_level: float = 0.95) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    count = int(array.size)
    if count == 0:
        return {"n": 0, "mean": None, "std": None, "ci95": {"low": None, "high": None}}
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if count > 1 else 0.0
    if count > 1 and std > 0:
        interval = stats.ttest_1samp(array, popmean=0.0).confidence_interval(confidence_level=confidence_level)
        low = float(interval.low)
        high = float(interval.high)
    else:
        low = mean
        high = mean
    return {"n": count, "mean": mean, "std": std, "ci95": {"low": low, "high": high}}


def paired_difference_stats(reference: Any, test: Any, seed: int) -> dict[str, Any]:
    reference_array = np.asarray(reference, dtype=float)
    test_array = np.asarray(test, dtype=float)
    mask = np.isfinite(reference_array) & np.isfinite(test_array)
    reference_array = reference_array[mask]
    test_array = test_array[mask]
    differences = test_array - reference_array
    summary = mean_confidence_interval(differences)
    if differences.size > 1:
        permutation = stats.permutation_test(
            (reference_array, test_array),
            statistic=lambda x, y: np.mean(y - x),
            permutation_type="samples",
            n_resamples=1999,
            alternative="two-sided",
            random_state=np.random.default_rng(seed),
        )
        difference_std = float(np.std(differences, ddof=1))
        t_pvalue = float(stats.ttest_rel(test_array, reference_array).pvalue) if difference_std > 0 else None
        effect = float(np.mean(differences) / difference_std) if difference_std > 0 else None
        summary.update(
            {
                "mean_difference": float(np.mean(differences)),
                "p_value_permutation": float(permutation.pvalue),
                "p_value_ttest": t_pvalue,
                "effect_size_dz": effect,
            }
        )
    else:
        summary.update(
            {
                "mean_difference": float(np.mean(differences)) if differences.size == 1 else None,
                "p_value_permutation": None,
                "p_value_ttest": None,
                "effect_size_dz": None,
            }
        )
    return summary


def discrete_mutual_information(x: Any, y: Any) -> float | None:
    x_array = np.asarray(x)
    y_array = np.asarray(y)
    if x_array.size == 0 or y_array.size == 0 or x_array.shape[0] != y_array.shape[0]:
        return None
    x_values, x_inverse = np.unique(x_array, return_inverse=True)
    y_values, y_inverse = np.unique(y_array, return_inverse=True)
    joint = np.zeros((x_values.size, y_values.size), dtype=float)
    np.add.at(joint, (x_inverse, y_inverse), 1.0)
    joint /= float(joint.sum())
    marginal_x = joint.sum(axis=1, keepdims=True)
    marginal_y = joint.sum(axis=0, keepdims=True)
    product = marginal_x * marginal_y
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log2(joint[mask] / product[mask])))


def linear_cka(x: Any, y: Any) -> float | None:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if x_array.ndim != 2 or y_array.ndim != 2 or x_array.shape[0] != y_array.shape[0]:
        return None
    x_centered = x_array - np.mean(x_array, axis=0, keepdims=True)
    y_centered = y_array - np.mean(y_array, axis=0, keepdims=True)
    cross = x_centered.T @ y_centered
    norm_x = np.linalg.norm(x_centered.T @ x_centered, ord="fro")
    norm_y = np.linalg.norm(y_centered.T @ y_centered, ord="fro")
    if norm_x == 0 or norm_y == 0:
        return None
    return float((np.linalg.norm(cross, ord="fro") ** 2) / (norm_x * norm_y))


def git_metadata(project_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except Exception:
        return {"git_commit": None, "git_dirty": None}


def build_run_record(
    simulation_name: str,
    script_path: Path,
    started_at_utc: str,
    finished_at_utc: str,
    duration_sec: float,
    parameters: dict[str, Any],
    seed_numpy: int,
    n_trials: int,
    summary: dict[str, Any],
    statistics: dict[str, Any],
    trials: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    warnings: list[str],
    extra_configuration: dict[str, Any] | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    project_root = find_project_root(script_path)
    git_info = git_metadata(project_root)
    configuration = {
        "parameters": parameters,
        "parameter_hash": parameter_hash(parameters),
        "seed_numpy": seed_numpy,
        "numpy_bit_generator": np.random.default_rng(seed_numpy).bit_generator.__class__.__name__,
        "n_trials": n_trials,
    }
    if extra_configuration:
        configuration.update(extra_configuration)
    return sanitize_json(
        {
            "schema_version": SCHEMA_VERSION,
            "run": {
                "run_id": f"{simulation_name}_{started_at_utc.replace(':', '').replace('-', '').replace('T', '_').replace('Z', '')}",
                "simulation_name": simulation_name,
                "status": status,
                "started_at_utc": started_at_utc,
                "finished_at_utc": finished_at_utc,
                "duration_sec": duration_sec,
            },
            "environment": {
                "python_version": sys.version.split()[0],
                "numpy_version": np.__version__,
                "scipy_version": scipy.__version__,
                "platform": platform.platform(),
                "cpu": platform.processor() or None,
                "script_path": relative_to_root(script_path, project_root),
                **git_info,
            },
            "configuration": configuration,
            "summary": summary,
            "statistics": statistics,
            "trials": trials,
            "artifacts": artifacts,
            "warnings": warnings,
        }
    )