from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import yaml


STATE_DIR = Path("state")
DOCS_DIR = Path("docs")

PHASE_GATES = {
    0: {
        "tests_pass": "All unit tests pass",
        "memory_budget": "Memory budget within limits",
        "parameter_count": "Parameter count within target",
    },
    1: {
        "bpb_threshold": "BPB within 1.5x of Transformer baseline",
        "spike_mi": "MI > 0.1",
        "spike_cka": "CKA > 0.3",
        "spike_firing_rate": "Firing rate 30-60%",
    },
    2: {
        "passkey_32k": "Passkey retrieval >95% at 32K",
        "passkey_128k": "Passkey retrieval >80% at 128K",
        "mla_cache_linear": "MLA cache grows linearly with context",
    },
    3: {
        "spatial_classification": "3D shape classification outperforms Transformer",
        "spatial_dynamics": "n-body dynamics outperforms Transformer",
        "equivariance_test": "Equivariance error <5% at 60-degree rotation",
        "language_no_degrade": "Language BPB not degraded >10% with GP",
    },
    4: {
        "cross_modal_transfer": "Cross-modal transfer demonstrated",
        "joint_embedding": "Joint embedding quality verified",
    },
    5: {
        "benchmark_competitive": "Competitive with Phi-3-mini class",
        "int8_deployed": "INT8 quantization verified",
    },
}


def load_gate_results() -> dict:
    gate_file = STATE_DIR / "gate_results.yaml"
    if gate_file.exists():
        with open(gate_file) as f:
            return yaml.safe_load(f) or {}
    return {}


def evaluate_gates(phase: int) -> dict[str, str]:
    results = load_gate_results()
    phase_key = f"phase_{phase}"
    phase_results = results.get(phase_key, {})

    evaluation = {}
    gates = PHASE_GATES.get(phase, {})

    for gate_name, description in gates.items():
        gate_data = phase_results.get(gate_name, {})
        status = gate_data.get("status", "PENDING")
        evaluation[gate_name] = status

    return evaluation


def print_evaluation(phase: int) -> None:
    evaluation = evaluate_gates(phase)
    gates = PHASE_GATES.get(phase, {})

    print(f"[{datetime.now().isoformat()}] Phase {phase} Gate Evaluation")
    print(f"{'='*60}")

    all_pass = True
    has_red = False

    for gate_name, status in evaluation.items():
        description = gates.get(gate_name, "")
        indicator = "PASS" if status == "PASS" else ("FAIL" if status == "FAIL" else "PENDING")
        print(f"  [{indicator:7s}] {gate_name}: {description}")

        if status != "PASS":
            all_pass = False
        if status == "FAIL":
            has_red = True

    print(f"{'='*60}")
    if all_pass:
        print(f"  VERDICT: GREEN -- proceed to Phase {phase + 1}")
    elif has_red:
        print(f"  VERDICT: RED -- STOP. Review failures before proceeding.")
    else:
        print(f"  VERDICT: YELLOW -- gates pending. Complete evaluation.")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=0)
    args = parser.parse_args()

    print_evaluation(args.phase)


if __name__ == "__main__":
    main()
