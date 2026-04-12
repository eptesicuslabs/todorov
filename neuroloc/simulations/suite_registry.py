from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SimulationSpec:
    simulation_id: str
    category: str
    script_relative_path: str
    metrics_filename: str
    artifact_filenames: tuple[str, ...]
    smoke_env: dict[str, str]
    required_summary_keys: tuple[str, ...] = ()
    minimum_summary_values: tuple[tuple[str, float], ...] = ()
    required_modules: tuple[str, ...] = ()

    def script_path(self, project_root: Path) -> Path:
        return project_root / self.script_relative_path


SIMULATION_SPECS: dict[str, SimulationSpec] = {
    "capacity_scaling": SimulationSpec(
        simulation_id="capacity_scaling",
        category="compression",
        script_relative_path="neuroloc/simulations/memory/capacity_scaling.py",
        metrics_filename="capacity_scaling_metrics.json",
        artifact_filenames=("capacity_scaling.png", "capacity_scaling_metrics.json"),
        smoke_env={
            "CAPACITY_NETWORK_SIZES": "64,128",
            "CAPACITY_LOAD_FACTORS": "0.05,0.10,0.138",
            "CAPACITY_TRIALS": "2",
            "CAPACITY_MAX_ITERS": "12",
        },
        required_summary_keys=("binary_at_target_alpha", "ternary_at_target_alpha"),
    ),
    "rate_coded_spike": SimulationSpec(
        simulation_id="rate_coded_spike",
        category="compression",
        script_relative_path="neuroloc/simulations/prototypes/rate_coded_spike.py",
        metrics_filename="rate_coded_spike_metrics.json",
        artifact_filenames=("rate_coded_spike.png", "rate_coded_spike_metrics.json"),
        smoke_env={
            "RATE_SPIKE_N_NEURONS": "128",
            "RATE_SPIKE_N_PATTERNS": "5,10",
            "RATE_SPIKE_TRIALS": "2",
            "RATE_SPIKE_MAX_ITERS": "12",
        },
        required_summary_keys=("reference_cosine.binary", "reference_cosine.ternary", "reference_cosine.rate"),
    ),
    "hierarchical_ternary": SimulationSpec(
        simulation_id="hierarchical_ternary",
        category="compression",
        script_relative_path="neuroloc/simulations/sparse_coding/hierarchical_ternary.py",
        metrics_filename="hierarchical_ternary_metrics.json",
        artifact_filenames=("hierarchical_ternary.png", "hierarchical_ternary_metrics.json"),
        smoke_env={
            "HTERN_D_VALUES": "64,256",
            "HTERN_K_FRACTIONS": "0.05,0.20,0.41",
            "HTERN_N_SAMPLES": "64",
            "HTERN_TRIALS": "2",
        },
        required_summary_keys=(
            "standard_reference_mi",
            "standard_reference_cka",
            "selected_hierarchical_reference_mi",
            "selected_hierarchical_reference_cka",
        ),
    ),
    "asymmetric_outer_product_recall": SimulationSpec(
        simulation_id="asymmetric_outer_product_recall",
        category="compression",
        script_relative_path="neuroloc/simulations/memory/asymmetric_outer_product_recall.py",
        metrics_filename="asymmetric_outer_product_recall_metrics.json",
        artifact_filenames=("asymmetric_outer_product_recall.png", "asymmetric_outer_product_recall_metrics.json"),
        smoke_env={
            "ASYM_TRIALS": "2",
            "ASYM_PATTERN_COUNTS": "8,16",
            "ASYM_QUERY_NOISE": "0.0,0.1",
            "ASYM_DECAYS": "0.4,0.8",
        },
        required_summary_keys=("best_exact_query_encoding", "selected_exact_query_raw_cosine"),
    ),
    "wta_dynamics": SimulationSpec(
        simulation_id="wta_dynamics",
        category="spiking",
        script_relative_path="neuroloc/simulations/lateral_inhibition/wta_dynamics.py",
        metrics_filename="wta_dynamics_metrics.json",
        artifact_filenames=("wta_dynamics.png", "wta_dynamics_anchor.png", "wta_dynamics_metrics.json"),
        smoke_env={
            "WTA_N_NEURONS": "40",
            "WTA_TARGET_FRACTIONS": "0.10,0.41",
            "WTA_NOISE_LEVELS": "0.2,0.6",
            "WTA_SELECTION_TRIALS": "2",
            "WTA_CALIBRATION_TRIALS": "32",
            "WTA_SCALING_NETWORK_SIZES": "40,80",
            "WTA_SCALING_TARGET_FRACTIONS": "0.10,0.41",
            "WTA_SCALING_TRIALS": "2",
            "WTA_ANCHOR_DURATION_MS": "75",
        },
        required_summary_keys=("kwta_exact_support_recovery", "threshold_exact_support_recovery"),
        required_modules=("brian2",),
    ),
    "lif_fi_curve": SimulationSpec(
        simulation_id="lif_fi_curve",
        category="spiking",
        script_relative_path="neuroloc/simulations/single_neuron/lif_fi_curve.py",
        metrics_filename="lif_leak_validation_metrics.json",
        artifact_filenames=("lif_leak_validation.png", "lif_leak_validation_metrics.json"),
        smoke_env={
            "LIF_GAP_VALUES_MS": "4,16",
            "LIF_GAP_TRIALS": "2",
            "LIF_DRIFT_LENGTHS": "64,128",
            "LIF_DRIFT_TRIALS": "2",
        },
        required_summary_keys=(
            "main_comparison.selected_gap_retained_fraction.integrator_control",
            "main_comparison.selected_max_length_final_abs_mv.integrator_control",
            "spike_probe.selected_gap_second_spike.integrator_control",
        ),
        minimum_summary_values=(("spike_probe.selected_gap_second_spike.integrator_control", 0.01),),
        required_modules=("brian2",),
    ),
}


SUITES: dict[str, tuple[str, ...]] = {
    "compression": (
        "capacity_scaling",
        "rate_coded_spike",
        "hierarchical_ternary",
        "asymmetric_outer_product_recall",
    ),
    "spiking": (
        "wta_dynamics",
        "lif_fi_curve",
    ),
    "precompute": (
        "capacity_scaling",
        "rate_coded_spike",
        "hierarchical_ternary",
        "asymmetric_outer_product_recall",
        "wta_dynamics",
        "lif_fi_curve",
    ),
}


def get_suite_specs(suite_name: str) -> list[SimulationSpec]:
    if suite_name not in SUITES:
        raise KeyError(f"unknown suite: {suite_name}")
    return [SIMULATION_SPECS[item] for item in SUITES[suite_name]]


def all_specs() -> list[SimulationSpec]:
    return list(SIMULATION_SPECS.values())


def to_jsonable(spec: SimulationSpec) -> dict[str, Any]:
    return {
        "simulation_id": spec.simulation_id,
        "category": spec.category,
        "script_relative_path": spec.script_relative_path,
        "metrics_filename": spec.metrics_filename,
        "artifact_filenames": list(spec.artifact_filenames),
        "smoke_env": dict(spec.smoke_env),
        "required_summary_keys": list(spec.required_summary_keys),
        "minimum_summary_values": [
            {"key": dotted_key, "minimum": minimum}
            for dotted_key, minimum in spec.minimum_summary_values
        ],
        "required_modules": list(spec.required_modules),
    }
