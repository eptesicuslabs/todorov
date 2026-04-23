import sys
from pathlib import Path

from neuroloc.simulations.suite_registry import get_suite_specs
from neuroloc.simulations.suite_runner import run_specs


def test_phase1_nm_smoke_suite(tmp_path: Path) -> None:
    results = run_specs(
        specs=get_suite_specs("phase1_nm"),
        profile="smoke",
        output_root=tmp_path / "phase1_nm",
        python_executable=sys.executable,
        timeout_sec=300,
    )
    failures = [
        (result.simulation_id, result.validation_error, result.stderr_tail)
        for result in results
        if not result.ok
    ]
    assert not failures, failures
