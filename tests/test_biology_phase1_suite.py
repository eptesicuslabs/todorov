import sys
from pathlib import Path

from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES, get_suite_specs
from neuroloc.simulations.suite_runner import run_specs


def test_biology_phase1_smoke_suite(tmp_path: Path) -> None:
    results = run_specs(
        specs=get_suite_specs("biology_phase1"),
        profile="smoke",
        output_root=tmp_path / "biology_phase1",
        python_executable=sys.executable,
        timeout_sec=300,
    )
    failures = [
        (result.simulation_id, result.validation_error, result.stderr_tail)
        for result in results
        if not result.ok
    ]
    assert not failures, failures


def test_registry_exposes_biology_phase1_suite() -> None:
    assert "biology_phase1" in SUITES
    assert "episodic_separation_completion" in SUITES["biology_phase1"]
    assert "episodic_replay_reuse" in SUITES["biology_phase1"]
    assert "contextual_gate_routing" in SUITES["biology_phase1"]
    assert SIMULATION_SPECS["episodic_separation_completion"].category == "biology_phase1"
