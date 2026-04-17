import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from neuroloc.model import god_machine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOD_MACHINE = PROJECT_ROOT / "neuroloc" / "model" / "god_machine.py"
CLEARED_ENV_KEYS = {
    "ALLOW_SYNTHETIC",
    "ALLOW_WIKITEXT_FALLBACK",
    "SMOKE_TEST",
}


def _god_machine_env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("NM_") or key in CLEARED_ENV_KEYS:
            env.pop(key, None)
    env["PYTHONUNBUFFERED"] = "1"
    env["NM_OUTPUT_DIR"] = str(tmp_path / "run")
    env.update(overrides)
    return env


def _run_god_machine(
    stdout_path: Path,
    env: dict[str, str],
    cwd: Path | None = None,
) -> tuple[subprocess.CompletedProcess[str], str]:
    with open(stdout_path, "w", encoding="utf-8") as handle:
        completed = subprocess.run(
            [sys.executable, str(GOD_MACHINE)],
            cwd=cwd or PROJECT_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=60,
        )
    return completed, stdout_path.read_text(encoding="utf-8")


def _make_dir_link(link_path: Path, target_path: Path) -> bool:
    try:
        os.symlink(target_path, link_path, target_is_directory=True)
        return True
    except OSError:
        if os.name != "nt":
            return False
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link_path), str(target_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.returncode == 0


def _write_valid_benchmark_manifest(
    tmp_path: Path,
    benchmark_output_dir: Path | None = None,
    manifest_output_dir: Path | None = None,
) -> Path:
    manifest_anchor = manifest_output_dir or (tmp_path / "run")
    manifest_path = god_machine._run1_benchmark_manifest_path(manifest_anchor)
    benchmark_output_dir = benchmark_output_dir or (tmp_path / "run1_baseline_noerasure_bench")
    benchmark_output_dir.mkdir(parents=True, exist_ok=True)
    run_name = "run1_baseline_noerasure_bench"
    metadata_path = benchmark_output_dir / f"{run_name}_metadata.json"
    results_path = benchmark_output_dir / f"{run_name}_results.json"
    metrics_path = benchmark_output_dir / f"{run_name}_metrics.jsonl"
    contract_hash = god_machine._run1_benchmark_contract_hash()
    metadata_path.write_text(
        json.dumps(
            {
                "run": run_name,
                "config_hash": contract_hash,
                "dataset_source": "fineweb-edu",
            }
        ),
        encoding="utf-8",
    )
    results_path.write_text(
        json.dumps(
            {
                "run": run_name,
                "current_step": 20,
                "final": True,
                "metadata": {"config_hash": contract_hash},
            }
        ),
        encoding="utf-8",
    )
    metrics_path.write_text('{"event":"run_start"}\n{"event":"run_end"}\n', encoding="utf-8")
    git = god_machine._capture_git_metadata()
    git_state = god_machine._capture_git_state_fingerprint(ignore_paths={manifest_path})
    payload = {
        "preset": "run1_baseline_noerasure",
        "status": "completed",
        "config_hash": contract_hash,
        "dataset_mode": "fineweb",
        "dataset_source": "fineweb-edu",
        "total_steps": 20,
        "output_dir": str(benchmark_output_dir),
        "run_name": run_name,
        "device": "cuda",
        "device_capability": [9, 0],
        "device_total_memory_gb": 141.0,
        "device_multi_processor_count": 132,
        "git_sha": git.get("sha"),
        "git_dirty": git.get("dirty"),
        "git_state_fingerprint": git_state.get("fingerprint"),
        "artifact_hashes": {
            "metadata": god_machine._sha256_file(metadata_path),
            "results": god_machine._sha256_file(results_path),
            "metrics": god_machine._sha256_file(metrics_path),
        },
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


class _DummyLoadResult:
    missing_keys: list[str] = []
    unexpected_keys: list[str] = []


class _DummyModel:
    def __init__(self, cfg: god_machine.Config) -> None:
        self.cfg = cfg

    def to(self, device: object) -> "_DummyModel":
        return self

    def load_state_dict(self, state: dict[str, object], strict: bool = False) -> _DummyLoadResult:
        return _DummyLoadResult()


def _stub_god_machine_main(
    monkeypatch: pytest.MonkeyPatch,
    *,
    create_best_checkpoint: bool = True,
    eval_behavior=None,
    default_output_root: Path | None = None,
) -> None:
    monkeypatch.setattr(god_machine, "GodMachine", _DummyModel)
    if default_output_root is not None:
        monkeypatch.setattr(god_machine, "DEFAULT_OUTPUT_ROOT", default_output_root)
    monkeypatch.setattr(
        god_machine,
        "_device_metadata",
        lambda: {
            "device": "cuda",
            "device_capability": [9, 0],
            "device_total_memory_gb": 141.0,
            "device_multi_processor_count": 132,
        },
    )
    monkeypatch.setattr(god_machine, "_require_run1_full_launch_device", lambda benchmark_manifest: None)
    monkeypatch.setattr(
        god_machine,
        "download_fineweb_edu",
        lambda max_bytes=5_000_000_000, allow_wikitext_fallback=False: (b"abcd" * 64, b"abcd" * 16, "fineweb-edu"),
    )
    monkeypatch.setattr(god_machine.torch, "load", lambda *args, **kwargs: {})

    def fake_train_model(
        cfg: god_machine.Config,
        train_data: bytes,
        val_data: bytes,
        name: str,
        output_dir: Path,
        dataset_source: str = "unknown",
        stop_after_steps: int | None = None,
        launch_contract: dict[str, object] | None = None,
    ) -> dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        config_hash = god_machine._config_hash(god_machine._cfg_to_dict(cfg))
        (output_dir / f"{name}_metadata.json").write_text(
            json.dumps({"run": name, "config_hash": config_hash, "dataset_source": dataset_source}),
            encoding="utf-8",
        )
        (output_dir / f"{name}_results.json").write_text(
            json.dumps({"run": name, "current_step": cfg.max_steps, "final": True, "metadata": {"config_hash": config_hash}}),
            encoding="utf-8",
        )
        (output_dir / f"{name}_metrics.jsonl").write_text('{"event":"run_start"}\n{"event":"run_end"}\n', encoding="utf-8")
        if create_best_checkpoint:
            (output_dir / f"{name}_best.pt").write_bytes(b"checkpoint")
        return {
            "best_val_bpb": 1.0,
            "total_steps": cfg.max_steps,
            "tokens_seen_total": 0,
        }

    monkeypatch.setattr(god_machine, "train_model", fake_train_model)
    if eval_behavior is not None:
        monkeypatch.setattr(god_machine, "run_eval_suite", eval_behavior)


def test_cli_benchmark_contract_allows_external_stdout_log(tmp_path: Path) -> None:
    stdout_path = tmp_path / "run1_baseline_noerasure_bench.stdout.log"
    completed, output = _run_god_machine(
        stdout_path,
        _god_machine_env(
            tmp_path,
            NM_PRESET="run1_baseline_noerasure",
            NM_RUN_NAME="run1_baseline_noerasure_bench",
            NM_DATASET="fineweb",
            NM_MAX_STEPS="20",
            NM_SKIP_VALIDATION="1",
            NM_SKIP_EVAL="1",
            NM_BATCH_SIZE="4",
            NM_SEQ_LEN="256",
            NM_MAX_SEQ_LEN="256",
        ),
    )
    assert completed.returncode != 0
    assert "benchmark mode requires the full launch shape" in output
    assert "is not fresh" not in output
    assert stdout_path.exists()
    assert not (tmp_path / "run" / "stdout.log").exists()


def test_canonical_output_path_is_repo_anchored_for_relative_paths() -> None:
    relative_path = Path("neuroloc") / "output" / "relative_surface"
    canonical = god_machine._canonical_output_path(relative_path)
    assert canonical == (god_machine.REPO_ROOT / relative_path).resolve(strict=False)


@pytest.mark.parametrize("run_name", ["nested/run", "nested\\run"])
def test_cli_rejects_run_name_with_path_separators(tmp_path: Path, run_name: str) -> None:
    stdout_path = tmp_path / "invalid_run_name.stdout.log"
    completed, output = _run_god_machine(
        stdout_path,
        _god_machine_env(
            tmp_path,
            NM_PRESET="run1_baseline_noerasure",
            NM_RUN_NAME=run_name,
            NM_DATASET="fineweb",
            NM_MAX_STEPS="20",
            NM_SKIP_VALIDATION="1",
            NM_SKIP_EVAL="1",
        ),
    )
    assert completed.returncode != 0
    assert "invalid run name" in output


def test_cli_rejects_absolute_run_name(tmp_path: Path) -> None:
    stdout_path = tmp_path / "invalid_abs_run_name.stdout.log"
    completed, output = _run_god_machine(
        stdout_path,
        _god_machine_env(
            tmp_path,
            NM_PRESET="run1_baseline_noerasure",
            NM_RUN_NAME=str((tmp_path / "absolute_name").resolve()),
            NM_DATASET="fineweb",
            NM_MAX_STEPS="20",
            NM_SKIP_VALIDATION="1",
            NM_SKIP_EVAL="1",
        ),
    )
    assert completed.returncode != 0
    assert "invalid run name" in output


def test_cli_full_launch_rejects_benchmark_only_skip_flags(tmp_path: Path) -> None:
    stdout_path = tmp_path / "run1_baseline_noerasure.stdout.log"
    completed, output = _run_god_machine(
        stdout_path,
        _god_machine_env(
            tmp_path,
            NM_PRESET="run1_baseline_noerasure",
            NM_AUTHORIZE_FULL_RUN="1",
            NM_MAX_STEPS="21",
            NM_SKIP_VALIDATION="1",
            NM_SKIP_EVAL="1",
        ),
    )
    assert completed.returncode != 0
    assert "benchmark-only flags" in output


def test_cli_full_launch_requires_benchmark_manifest(tmp_path: Path) -> None:
    stdout_path = tmp_path / "run1_baseline_noerasure.stdout.log"
    completed, output = _run_god_machine(
        stdout_path,
        _god_machine_env(
            tmp_path,
            NM_PRESET="run1_baseline_noerasure",
            NM_AUTHORIZE_FULL_RUN="1",
            NM_DATASET="fineweb",
        ),
    )
    assert completed.returncode != 0
    assert "requires benchmark manifest" in output


def test_cli_benchmark_rejects_wikitext_fallback_on_official_surface(tmp_path: Path) -> None:
    stdout_path = tmp_path / "run1_baseline_noerasure_bench.stdout.log"
    completed, output = _run_god_machine(
        stdout_path,
        _god_machine_env(
            tmp_path,
            NM_PRESET="run1_baseline_noerasure",
            NM_RUN_NAME="run1_baseline_noerasure_bench",
            NM_DATASET="fineweb",
            NM_MAX_STEPS="20",
            NM_SKIP_VALIDATION="1",
            NM_SKIP_EVAL="1",
            NM_BATCH_SIZE="16",
            NM_SEQ_LEN="2048",
            NM_MAX_SEQ_LEN="2048",
            ALLOW_WIKITEXT_FALLBACK="1",
        ),
    )
    assert completed.returncode != 0
    assert "ALLOW_WIKITEXT_FALLBACK=1 is forbidden" in output


def test_cli_benchmark_requires_sm90_cuda(tmp_path: Path) -> None:
    stdout_path = tmp_path / "run1_baseline_noerasure_bench.stdout.log"
    completed, output = _run_god_machine(
        stdout_path,
        _god_machine_env(
            tmp_path,
            NM_PRESET="run1_baseline_noerasure",
            NM_RUN_NAME="run1_baseline_noerasure_bench",
            NM_DATASET="fineweb",
            NM_MAX_STEPS="20",
            NM_SKIP_VALIDATION="1",
            NM_SKIP_EVAL="1",
            NM_BATCH_SIZE="16",
            NM_SEQ_LEN="2048",
            NM_MAX_SEQ_LEN="2048",
        ),
    )
    assert completed.returncode != 0
    assert "benchmark mode requires an sm_90-or-newer cuda device" in output


def test_main_benchmark_defaults_emit_manifest_compatible_with_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_god_machine_main(monkeypatch, default_output_root=tmp_path)
    monkeypatch.setenv("NM_PRESET", "run1_baseline_noerasure")
    monkeypatch.setenv("NM_DATASET", "fineweb")
    monkeypatch.setenv("NM_MAX_STEPS", "20")
    monkeypatch.setenv("NM_SKIP_VALIDATION", "1")
    monkeypatch.setenv("NM_SKIP_EVAL", "1")
    monkeypatch.setenv("NM_BATCH_SIZE", "16")
    monkeypatch.setenv("NM_SEQ_LEN", "2048")
    monkeypatch.setenv("NM_MAX_SEQ_LEN", "2048")
    monkeypatch.delenv("NM_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("NM_RUN_NAME", raising=False)
    monkeypatch.delenv("NM_RESUME", raising=False)
    monkeypatch.delenv("SMOKE_TEST", raising=False)

    god_machine.main()

    manifest_path = tmp_path / "run1_baseline_noerasure_benchmark_manifest.json"
    assert manifest_path.exists()
    payload = god_machine._require_run1_benchmark_manifest(tmp_path / "run1_baseline_noerasure")
    assert payload["run_name"] == "run1_baseline_noerasure_bench"
    assert payload["output_dir"] == str(tmp_path / "run1_baseline_noerasure_bench")


def test_main_benchmark_writer_preserves_symlinked_output_alias_for_reader_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external_target = tmp_path / "outside_target" / "run1_baseline_noerasure_bench"
    external_target.mkdir(parents=True, exist_ok=True)
    linked_output_dir = tmp_path / "linked_bench"
    if not _make_dir_link(linked_output_dir, external_target):
        pytest.skip("unable to create directory symlink or junction in this environment")
    _stub_god_machine_main(monkeypatch)
    monkeypatch.setenv("NM_PRESET", "run1_baseline_noerasure")
    monkeypatch.setenv("NM_OUTPUT_DIR", str(linked_output_dir))
    monkeypatch.setenv("NM_RUN_NAME", "run1_baseline_noerasure_bench")
    monkeypatch.setenv("NM_DATASET", "fineweb")
    monkeypatch.setenv("NM_MAX_STEPS", "20")
    monkeypatch.setenv("NM_SKIP_VALIDATION", "1")
    monkeypatch.setenv("NM_SKIP_EVAL", "1")
    monkeypatch.setenv("NM_BATCH_SIZE", "16")
    monkeypatch.setenv("NM_SEQ_LEN", "2048")
    monkeypatch.setenv("NM_MAX_SEQ_LEN", "2048")
    monkeypatch.delenv("NM_RESUME", raising=False)
    monkeypatch.delenv("SMOKE_TEST", raising=False)

    god_machine.main()

    manifest_path = god_machine._run1_benchmark_manifest_path(tmp_path / "run1_baseline_noerasure")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["output_dir"] == str(god_machine._absolute_output_path(linked_output_dir))
    with pytest.raises(RuntimeError, match="forbidden on the official run1 surface"):
        god_machine._require_run1_benchmark_manifest(tmp_path / "run1_baseline_noerasure")


def test_cli_benchmark_rejects_resume_replay(tmp_path: Path) -> None:
    stdout_path = tmp_path / "run1_baseline_noerasure_bench.stdout.log"
    completed, output = _run_god_machine(
        stdout_path,
        _god_machine_env(
            tmp_path,
            NM_PRESET="run1_baseline_noerasure",
            NM_RUN_NAME="run1_baseline_noerasure_bench",
            NM_DATASET="fineweb",
            NM_MAX_STEPS="20",
            NM_SKIP_VALIDATION="1",
            NM_SKIP_EVAL="1",
            NM_BATCH_SIZE="16",
            NM_SEQ_LEN="2048",
            NM_MAX_SEQ_LEN="2048",
            NM_RESUME="1",
        ),
    )
    assert completed.returncode != 0
    assert "benchmark mode requires a fresh directory" in output


def test_cli_full_launch_rejects_device_mismatch_even_with_manifest(tmp_path: Path) -> None:
    _write_valid_benchmark_manifest(tmp_path)
    stdout_path = tmp_path / "run1_baseline_noerasure.stdout.log"
    completed, output = _run_god_machine(
        stdout_path,
        _god_machine_env(
            tmp_path,
            NM_PRESET="run1_baseline_noerasure",
            NM_AUTHORIZE_FULL_RUN="1",
            NM_DATASET="fineweb",
        ),
    )
    assert completed.returncode != 0
    assert "same device class recorded by the benchmark manifest" in output


def test_cli_full_launch_preflight_is_cwd_independent_from_repo_subdir(tmp_path: Path) -> None:
    _write_valid_benchmark_manifest(tmp_path)
    stdout_path = tmp_path / "run1_baseline_noerasure_subdir.stdout.log"
    completed, output = _run_god_machine(
        stdout_path,
        _god_machine_env(
            tmp_path,
            NM_PRESET="run1_baseline_noerasure",
            NM_AUTHORIZE_FULL_RUN="1",
            NM_DATASET="fineweb",
        ),
        cwd=PROJECT_ROOT / "neuroloc",
    )
    assert completed.returncode != 0
    assert "same device class recorded by the benchmark manifest" in output


def test_cli_full_launch_preflight_is_cwd_independent_from_outside_repo(tmp_path: Path) -> None:
    _write_valid_benchmark_manifest(tmp_path)
    outside_cwd = tmp_path / "outside_cwd"
    outside_cwd.mkdir(parents=True, exist_ok=True)
    stdout_path = tmp_path / "run1_baseline_noerasure_outside.stdout.log"
    completed, output = _run_god_machine(
        stdout_path,
        _god_machine_env(
            tmp_path,
            NM_PRESET="run1_baseline_noerasure",
            NM_AUTHORIZE_FULL_RUN="1",
            NM_DATASET="fineweb",
        ),
        cwd=outside_cwd,
    )
    assert completed.returncode != 0
    assert "same device class recorded by the benchmark manifest" in output


def test_cli_full_launch_rejects_runtime_override_drift(tmp_path: Path) -> None:
    _write_valid_benchmark_manifest(tmp_path)
    stdout_path = tmp_path / "run1_baseline_noerasure.stdout.log"
    completed, output = _run_god_machine(
        stdout_path,
        _god_machine_env(
            tmp_path,
            NM_PRESET="run1_baseline_noerasure",
            NM_AUTHORIZE_FULL_RUN="1",
            NM_DATASET="fineweb",
            NM_BATCH_SIZE="8",
        ),
    )
    assert completed.returncode != 0
    assert "canonical 4000-step config" in output


def test_cli_full_launch_rejects_hand_authored_manifest_without_artifacts(tmp_path: Path) -> None:
    manifest_path = god_machine._run1_benchmark_manifest_path(tmp_path / "run")
    manifest_path.write_text(
        json.dumps(
            {
                "preset": "run1_baseline_noerasure",
                "status": "completed",
                "config_hash": god_machine._run1_benchmark_contract_hash(),
                "dataset_mode": "fineweb",
                "dataset_source": "fineweb-edu",
                "total_steps": 20,
                "output_dir": str(tmp_path / "missing_bench"),
                "run_name": "run1_baseline_noerasure_bench",
                "device": "cuda",
                "device_capability": [9, 0],
                "git_sha": god_machine._capture_git_metadata().get("sha"),
                "git_dirty": god_machine._capture_git_metadata().get("dirty"),
            }
        ),
        encoding="utf-8",
    )
    stdout_path = tmp_path / "run1_baseline_noerasure.stdout.log"
    completed, output = _run_god_machine(
        stdout_path,
        _god_machine_env(
            tmp_path,
            NM_PRESET="run1_baseline_noerasure",
            NM_AUTHORIZE_FULL_RUN="1",
            NM_DATASET="fineweb",
        ),
    )
    assert completed.returncode != 0
    assert "missing artifact hashes" in output


def test_cli_full_launch_rejects_missing_metrics_artifact(tmp_path: Path) -> None:
    _write_valid_benchmark_manifest(tmp_path)
    metrics_path = tmp_path / "run1_baseline_noerasure_bench" / "run1_baseline_noerasure_bench_metrics.jsonl"
    metrics_path.unlink()
    stdout_path = tmp_path / "run1_baseline_noerasure.stdout.log"
    completed, output = _run_god_machine(
        stdout_path,
        _god_machine_env(
            tmp_path,
            NM_PRESET="run1_baseline_noerasure",
            NM_AUTHORIZE_FULL_RUN="1",
            NM_DATASET="fineweb",
        ),
    )
    assert completed.returncode != 0
    assert "references missing artifact" in output


def test_cli_full_launch_rejects_manifest_from_external_root(tmp_path: Path) -> None:
    external_output_dir = tmp_path / "elsewhere" / "run1_baseline_noerasure_bench"
    _write_valid_benchmark_manifest(tmp_path, benchmark_output_dir=external_output_dir)
    stdout_path = tmp_path / "run1_baseline_noerasure.stdout.log"
    completed, output = _run_god_machine(
        stdout_path,
        _god_machine_env(
            tmp_path,
            NM_PRESET="run1_baseline_noerasure",
            NM_AUTHORIZE_FULL_RUN="1",
            NM_DATASET="fineweb",
        ),
    )
    assert completed.returncode != 0
    assert "outside the shared output root" in output


def test_cli_full_launch_rejects_symlinked_benchmark_root_escape(tmp_path: Path) -> None:
    external_target = tmp_path / "outside_target" / "run1_baseline_noerasure_bench"
    external_target.mkdir(parents=True, exist_ok=True)
    linked_output_dir = tmp_path / "linked_bench"
    if not _make_dir_link(linked_output_dir, external_target):
        pytest.skip("unable to create directory symlink or junction in this environment")
    _write_valid_benchmark_manifest(tmp_path, benchmark_output_dir=linked_output_dir)
    stdout_path = tmp_path / "run1_baseline_noerasure.stdout.log"
    completed, output = _run_god_machine(
        stdout_path,
        _god_machine_env(
            tmp_path,
            NM_PRESET="run1_baseline_noerasure",
            NM_AUTHORIZE_FULL_RUN="1",
            NM_DATASET="fineweb",
        ),
    )
    assert completed.returncode != 0
    assert "forbidden on the official run1 surface" in output


def test_cli_full_launch_rejects_output_dir_inside_symlinked_parent(tmp_path: Path) -> None:
    external_root = tmp_path / "outside_target"
    external_root.mkdir(parents=True, exist_ok=True)
    linked_parent = tmp_path / "linked_parent"
    if not _make_dir_link(linked_parent, external_root):
        pytest.skip("unable to create directory symlink or junction in this environment")
    linked_output_dir = linked_parent / "run1_baseline_noerasure"
    linked_benchmark_dir = linked_parent / "run1_baseline_noerasure_bench"
    _write_valid_benchmark_manifest(
        tmp_path,
        benchmark_output_dir=linked_benchmark_dir,
        manifest_output_dir=linked_output_dir,
    )
    stdout_path = tmp_path / "run1_baseline_noerasure_parent_escape.stdout.log"
    completed, output = _run_god_machine(
        stdout_path,
        _god_machine_env(
            tmp_path,
            NM_PRESET="run1_baseline_noerasure",
            NM_OUTPUT_DIR=str(linked_output_dir),
            NM_AUTHORIZE_FULL_RUN="1",
            NM_DATASET="fineweb",
        ),
    )
    assert completed.returncode != 0
    assert "forbidden on the official run1 surface" in output


def test_cli_full_launch_rejects_hash_consistent_forged_artifacts(tmp_path: Path) -> None:
    manifest_path = god_machine._run1_benchmark_manifest_path(tmp_path / "run")
    benchmark_output_dir = tmp_path / "run1_baseline_noerasure_bench"
    benchmark_output_dir.mkdir(parents=True, exist_ok=True)
    run_name = "run1_baseline_noerasure_bench"
    metadata_path = benchmark_output_dir / f"{run_name}_metadata.json"
    results_path = benchmark_output_dir / f"{run_name}_results.json"
    metrics_path = benchmark_output_dir / f"{run_name}_metrics.jsonl"
    metadata_path.write_text(json.dumps({"run": run_name}), encoding="utf-8")
    results_path.write_text(json.dumps({"run": run_name, "final": True, "current_step": 20}), encoding="utf-8")
    metrics_path.write_text('{"event":"run_start"}\n{"event":"run_end"}\n', encoding="utf-8")
    git = god_machine._capture_git_metadata()
    git_state = god_machine._capture_git_state_fingerprint(ignore_paths={manifest_path})
    manifest_path.write_text(
        json.dumps(
            {
                "preset": "run1_baseline_noerasure",
                "status": "completed",
                "config_hash": god_machine._run1_benchmark_contract_hash(),
                "dataset_mode": "fineweb",
                "dataset_source": "fineweb-edu",
                "total_steps": 20,
                "output_dir": str(benchmark_output_dir),
                "run_name": run_name,
                "device": "cuda",
                "device_capability": [9, 0],
                "device_total_memory_gb": 141.0,
                "device_multi_processor_count": 132,
                "git_sha": git.get("sha"),
                "git_dirty": git.get("dirty"),
                "git_state_fingerprint": git_state.get("fingerprint"),
                "artifact_hashes": {
                    "metadata": god_machine._sha256_file(metadata_path),
                    "results": god_machine._sha256_file(results_path),
                    "metrics": god_machine._sha256_file(metrics_path),
                },
            }
        ),
        encoding="utf-8",
    )
    stdout_path = tmp_path / "run1_baseline_noerasure.stdout.log"
    completed, output = _run_god_machine(
        stdout_path,
        _god_machine_env(
            tmp_path,
            NM_PRESET="run1_baseline_noerasure",
            NM_AUTHORIZE_FULL_RUN="1",
            NM_DATASET="fineweb",
        ),
    )
    assert completed.returncode != 0
    assert "does not match manifest config hash" in output


@pytest.mark.parametrize("preset", ["run1a_retention_ablation", "run4_erasure_ablation"])
def test_cli_named_ablations_reject_skip_eval_flags(tmp_path: Path, preset: str) -> None:
    stdout_path = tmp_path / f"{preset}.stdout.log"
    completed, output = _run_god_machine(
        stdout_path,
        _god_machine_env(
            tmp_path,
            NM_PRESET=preset,
            NM_SKIP_VALIDATION="1",
            NM_SKIP_EVAL="1",
        ),
    )
    assert completed.returncode != 0
    assert "benchmark-only flags" in output


def test_incomplete_retrieval_gate_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="best_checkpoint_missing"):
        god_machine._require_completed_retrieval_gate(
            {"status": "not_evaluated", "reason": "best_checkpoint_missing"},
            skip_eval=False,
        )


def test_skipped_retrieval_gate_is_allowed_for_benchmark_mode() -> None:
    god_machine._require_completed_retrieval_gate(
        {"status": "not_evaluated", "reason": "NM_SKIP_EVAL"},
        skip_eval=True,
    )


def test_failed_retrieval_gate_requires_pass_when_configured() -> None:
    with pytest.raises(RuntimeError, match="retrieval gate failed"):
        god_machine._require_completed_retrieval_gate(
            {"status": "fail", "metric": "passkey_256", "passkey_256_accuracy": 0.0},
            skip_eval=False,
            require_pass=True,
        )


def test_main_full_launch_persists_failed_retrieval_gate_and_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "run"
    _write_valid_benchmark_manifest(tmp_path, manifest_output_dir=output_dir)
    _stub_god_machine_main(
        monkeypatch,
        eval_behavior=lambda model, val_data, output_dir, run_name: {
            "passkey": {"256": {"accuracy": 0.0}},
            "selective_copy": {"256": {"accuracy": 0.0}},
            "delta_state_structure_probe": {"mean_structure_ratio": 1.0},
        },
    )
    monkeypatch.setenv("NM_PRESET", "run1_baseline_noerasure")
    monkeypatch.setenv("NM_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("NM_AUTHORIZE_FULL_RUN", "1")
    monkeypatch.setenv("NM_DATASET", "fineweb")
    monkeypatch.delenv("SMOKE_TEST", raising=False)
    monkeypatch.delenv("NM_SKIP_EVAL", raising=False)
    monkeypatch.delenv("NM_SKIP_VALIDATION", raising=False)
    monkeypatch.delenv("NM_RESUME", raising=False)
    monkeypatch.delenv("NM_RUN_NAME", raising=False)

    with pytest.raises(RuntimeError, match="retrieval gate failed"):
        god_machine.main()

    results_path = output_dir / "run1_baseline_noerasure_results.json"
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["retrieval_gate"]["status"] == "fail"
    assert payload["retrieval_gate"]["passkey_256_accuracy"] == 0.0


def test_main_full_launch_persists_missing_checkpoint_gate_and_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "run"
    _write_valid_benchmark_manifest(tmp_path, manifest_output_dir=output_dir)
    _stub_god_machine_main(monkeypatch, create_best_checkpoint=False)
    monkeypatch.setenv("NM_PRESET", "run1_baseline_noerasure")
    monkeypatch.setenv("NM_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("NM_AUTHORIZE_FULL_RUN", "1")
    monkeypatch.setenv("NM_DATASET", "fineweb")
    monkeypatch.delenv("SMOKE_TEST", raising=False)
    monkeypatch.delenv("NM_SKIP_EVAL", raising=False)
    monkeypatch.delenv("NM_SKIP_VALIDATION", raising=False)
    monkeypatch.delenv("NM_RESUME", raising=False)
    monkeypatch.delenv("NM_RUN_NAME", raising=False)

    with pytest.raises(RuntimeError, match="retrieval gate did not complete"):
        god_machine.main()

    results_path = output_dir / "run1_baseline_noerasure_results.json"
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["retrieval_gate"]["status"] == "not_evaluated"
    assert payload["retrieval_gate"]["reason"] == "best_checkpoint_missing"


def test_main_full_launch_persists_eval_error_gate_and_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "run"
    _write_valid_benchmark_manifest(tmp_path, manifest_output_dir=output_dir)

    def raise_eval_error(model: object, val_data: bytes, output_dir: Path, run_name: str) -> dict[str, object]:
        raise ValueError("boom")

    _stub_god_machine_main(monkeypatch, eval_behavior=raise_eval_error)
    monkeypatch.setenv("NM_PRESET", "run1_baseline_noerasure")
    monkeypatch.setenv("NM_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("NM_AUTHORIZE_FULL_RUN", "1")
    monkeypatch.setenv("NM_DATASET", "fineweb")
    monkeypatch.delenv("SMOKE_TEST", raising=False)
    monkeypatch.delenv("NM_SKIP_EVAL", raising=False)
    monkeypatch.delenv("NM_SKIP_VALIDATION", raising=False)
    monkeypatch.delenv("NM_RESUME", raising=False)
    monkeypatch.delenv("NM_RUN_NAME", raising=False)

    with pytest.raises(RuntimeError, match="eval suite runtime failure"):
        god_machine.main()

    results_path = output_dir / "run1_baseline_noerasure_results.json"
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["retrieval_gate"]["status"] == "eval_error"
    assert "ValueError: boom" in payload["retrieval_gate"]["error"]


def test_main_full_launch_persists_checkpoint_mismatch_gate_and_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "run"
    _write_valid_benchmark_manifest(tmp_path, manifest_output_dir=output_dir)

    class MismatchResult:
        missing_keys = ["delta.weight"]
        unexpected_keys: list[str] = []

    _stub_god_machine_main(
        monkeypatch,
        eval_behavior=lambda model, val_data, output_dir, run_name: {
            "passkey": {"256": {"accuracy": 1.0}},
            "selective_copy": {"256": {"accuracy": 1.0}},
            "delta_state_structure_probe": {"mean_structure_ratio": 1.0},
        },
    )
    monkeypatch.setattr(_DummyModel, "load_state_dict", lambda self, state, strict=False: MismatchResult())
    monkeypatch.setenv("NM_PRESET", "run1_baseline_noerasure")
    monkeypatch.setenv("NM_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("NM_AUTHORIZE_FULL_RUN", "1")
    monkeypatch.setenv("NM_DATASET", "fineweb")
    monkeypatch.delenv("SMOKE_TEST", raising=False)
    monkeypatch.delenv("NM_SKIP_EVAL", raising=False)
    monkeypatch.delenv("NM_SKIP_VALIDATION", raising=False)
    monkeypatch.delenv("NM_RESUME", raising=False)
    monkeypatch.delenv("NM_RUN_NAME", raising=False)

    with pytest.raises(RuntimeError, match="checkpoint load mismatch"):
        god_machine.main()

    results_path = output_dir / "run1_baseline_noerasure_results.json"
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["retrieval_gate"]["status"] == "checkpoint_mismatch"
    assert "checkpoint load mismatch" in payload["retrieval_gate"]["error"]


def test_main_full_launch_persists_checkpoint_load_error_gate_and_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "run"
    _write_valid_benchmark_manifest(tmp_path, manifest_output_dir=output_dir)
    _stub_god_machine_main(
        monkeypatch,
        eval_behavior=lambda model, val_data, output_dir, run_name: {
            "passkey": {"256": {"accuracy": 1.0}},
            "selective_copy": {"256": {"accuracy": 1.0}},
            "delta_state_structure_probe": {"mean_structure_ratio": 1.0},
        },
    )
    monkeypatch.setattr(god_machine.torch, "load", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad checkpoint")))
    monkeypatch.setenv("NM_PRESET", "run1_baseline_noerasure")
    monkeypatch.setenv("NM_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("NM_AUTHORIZE_FULL_RUN", "1")
    monkeypatch.setenv("NM_DATASET", "fineweb")
    monkeypatch.delenv("SMOKE_TEST", raising=False)
    monkeypatch.delenv("NM_SKIP_EVAL", raising=False)
    monkeypatch.delenv("NM_SKIP_VALIDATION", raising=False)
    monkeypatch.delenv("NM_RESUME", raising=False)
    monkeypatch.delenv("NM_RUN_NAME", raising=False)

    with pytest.raises(RuntimeError, match="eval suite runtime failure"):
        god_machine.main()

    results_path = output_dir / "run1_baseline_noerasure_results.json"
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["retrieval_gate"]["status"] == "checkpoint_load_error"
    assert "ValueError: bad checkpoint" in payload["retrieval_gate"]["error"]


def test_download_fineweb_cache_is_cwd_independent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data_root"
    cache_dir = data_root / "fineweb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "train.bin").write_bytes(bytes(range(32)))
    (cache_dir / "val.bin").write_bytes(bytes(range(16)))
    other_cwd = tmp_path / "outside_cwd"
    other_cwd.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("NM_DATA_ROOT", str(data_root))
    monkeypatch.chdir(other_cwd)
    train_data, val_data, dataset_source = god_machine.download_fineweb_edu()
    assert dataset_source == "fineweb-edu-cached"
    assert len(train_data) == 32
    assert len(val_data) == 16


def test_train_model_rejects_resume_contract_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = god_machine.Config(
        d_model=32,
        n_layers=2,
        vocab_size=256,
        max_seq_len=32,
        delta_num_heads=2,
        delta_head_dim=16,
        attn_d_c=16,
        attn_d_R=8,
        attn_num_heads=2,
        mlp_ratio=2.0,
        layer_pattern=("DELTA", "ATTN"),
        kwta_enabled=False,
        delta_erasure_enabled=False,
        bcm_alpha_enabled=False,
        imagination_enabled=False,
        pc_diagnostic_enabled=False,
        multi_compartment_enabled=False,
        use_fla_if_available=False,
        batch_size=2,
        seq_len=16,
        max_steps=4,
        warmup_steps=1,
        val_interval=2,
        grad_checkpointing=False,
        amp=False,
        seed=123,
    )
    synth = bytes(range(256)) * 32
    train_data = synth[: len(synth) * 3 // 4]
    val_data = synth[len(synth) * 3 // 4 :]
    output_dir = tmp_path / "resume_contract"
    god_machine.train_model(
        cfg,
        train_data,
        val_data,
        name="resume_contract",
        output_dir=output_dir,
        dataset_source="synthetic",
        stop_after_steps=2,
        launch_contract={"manifest": "one"},
    )
    monkeypatch.setenv("NM_RESUME", "1")
    with pytest.raises(RuntimeError, match="launch contract does not match"):
        god_machine.train_model(
            cfg,
            train_data,
            val_data,
            name="resume_contract",
            output_dir=output_dir,
            dataset_source="synthetic",
            launch_contract={"manifest": "two"},
        )


def test_git_state_fingerprint_ignores_in_repo_manifest_path() -> None:
    manifest_path = PROJECT_ROOT / "neuroloc" / "output" / "pytest_manifest_ignore.json"
    if manifest_path.exists():
        manifest_path.unlink()
    before = god_machine._capture_git_state_fingerprint(ignore_paths={manifest_path})
    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("{}", encoding="utf-8")
        after = god_machine._capture_git_state_fingerprint(ignore_paths={manifest_path})
    finally:
        if manifest_path.exists():
            manifest_path.unlink()
    assert before["error"] is None
    assert after["error"] is None
    assert before["fingerprint"] == after["fingerprint"]


def test_git_state_fingerprint_ignores_documented_benchmark_stdout_path() -> None:
    stdout_path = PROJECT_ROOT / "neuroloc" / "output" / "run1_baseline_noerasure_bench.stdout.log"
    if stdout_path.exists():
        stdout_path.unlink()
    before = god_machine._capture_git_state_fingerprint(ignore_paths={stdout_path})
    try:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("benchmark stdout\n", encoding="utf-8")
        after = god_machine._capture_git_state_fingerprint(ignore_paths={stdout_path})
    finally:
        if stdout_path.exists():
            stdout_path.unlink()
    assert before["error"] is None
    assert after["error"] is None
    assert before["fingerprint"] == after["fingerprint"]


def test_git_state_fingerprint_ignores_in_repo_full_run_output_dir_and_stdout_log() -> None:
    output_dir = PROJECT_ROOT / "neuroloc" / "output" / "pytest_resume_surface"
    stdout_path = output_dir.parent / f"{output_dir.name}.stdout.log"
    if stdout_path.exists():
        stdout_path.unlink()
    if output_dir.exists():
        for child in output_dir.iterdir():
            child.unlink()
        output_dir.rmdir()
    before = god_machine._capture_git_state_fingerprint(ignore_paths={output_dir, stdout_path})
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run1_last.pt").write_bytes(b"checkpoint")
        (output_dir / "run1_metrics.jsonl").write_text('{"event":"step"}\n', encoding="utf-8")
        stdout_path.write_text("stdout\n", encoding="utf-8")
        after = god_machine._capture_git_state_fingerprint(ignore_paths={output_dir, stdout_path})
    finally:
        if stdout_path.exists():
            stdout_path.unlink()
        if output_dir.exists():
            for child in output_dir.iterdir():
                child.unlink()
            output_dir.rmdir()
    assert before["error"] is None
    assert after["error"] is None
    assert before["fingerprint"] == after["fingerprint"]


def test_full_launch_device_helper_rejects_same_capability_different_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = {
        "device": "cuda",
        "device_capability": [9, 0],
        "device_total_memory_gb": 141.0,
        "device_multi_processor_count": 132,
    }
    monkeypatch.setattr(
        god_machine,
        "_device_metadata",
        lambda: {
            "device": "cuda",
            "device_capability": [9, 0],
            "device_total_memory_gb": 80.0,
            "device_multi_processor_count": 114,
        },
    )
    with pytest.raises(RuntimeError, match="memory profile"):
        god_machine._require_run1_full_launch_device(manifest)


def test_assert_preset_retention_safe_raises_when_delta_layer_missing_override() -> None:
    overrides = {"kwta_enabled": False, "delta_erasure_enabled": False}
    with pytest.raises(RuntimeError, match="alpha_log_mean"):
        god_machine._assert_preset_retention_safe("unguarded_delta_preset", overrides)


def test_assert_preset_retention_safe_raises_for_slot_layer_without_override() -> None:
    overrides = {
        "kwta_enabled": False,
        "layer_pattern": ("SLOT", "SLOT", "ATTN"),
    }
    with pytest.raises(RuntimeError, match="alpha_log_mean"):
        god_machine._assert_preset_retention_safe("unguarded_slot_preset", overrides)


def test_assert_preset_retention_safe_accepts_explicit_override() -> None:
    overrides = {"kwta_enabled": False, "alpha_log_mean": 2.2}
    god_machine._assert_preset_retention_safe("compliant_preset", overrides)


def test_assert_preset_retention_safe_accepts_non_retentive_layer_pattern() -> None:
    overrides = {
        "kwta_enabled": False,
        "layer_pattern": ("ATTN",),
    }
    god_machine._assert_preset_retention_safe("attn_only_preset", overrides)


def test_resolve_preset_raises_for_god_preset() -> None:
    with pytest.raises(RuntimeError, match="alpha_log_mean"):
        god_machine._resolve_preset("god")


@pytest.mark.parametrize(
    "preset",
    [
        "run4_erasure_ablation",
        "run1_baseline_noerasure",
        "run1a_retention_ablation",
        "run2_slot_memory",
    ],
)
def test_named_presets_all_set_alpha_log_mean_explicitly(preset: str) -> None:
    overrides, _, _ = god_machine._resolve_preset_raw(preset)
    assert "alpha_log_mean" in overrides, (
        f"preset {preset!r} must explicitly set alpha_log_mean so the retention "
        f"guard accepts it and future agents cannot reintroduce the inherited-default "
        f"bug documented in wiki/mistakes/run2_slot_memory_decay_copy_paste.md. "
        f"this test reads the raw preset definition directly to bypass the guard, "
        f"so it catches cases where the guard silently allows a bug through."
    )


def test_resolve_preset_wires_retention_guard_into_resolution_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_preset = "sentinel_missing_alpha_log_mean"

    def fake_raw(preset: str) -> tuple[dict[str, Any], str, str]:
        if preset == sentinel_preset:
            return (
                {
                    "kwta_enabled": False,
                    "delta_erasure_enabled": False,
                },
                f"preset: {sentinel_preset} (diagnostic)",
                sentinel_preset,
            )
        raise ValueError(preset)

    monkeypatch.setattr(god_machine, "_resolve_preset_raw", fake_raw)

    overrides, _, _ = god_machine._resolve_preset_raw(sentinel_preset)
    assert "alpha_log_mean" not in overrides

    with pytest.raises(RuntimeError, match="alpha_log_mean"):
        god_machine._resolve_preset(sentinel_preset)


def test_assert_fla_available_if_requested_permits_when_preset_does_not_want_fla() -> None:
    cfg = god_machine.Config(use_fla_if_available=False)
    god_machine._assert_fla_available_if_requested(cfg)


def test_assert_fla_available_if_requested_raises_when_preset_wants_fla_but_package_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(god_machine, "FLA_AVAILABLE", False)
    monkeypatch.setattr(god_machine, "fused_recurrent_simple_gla", None)
    cfg = god_machine.Config(use_fla_if_available=True)
    with pytest.raises(RuntimeError, match="flash-linear-attention is not"):
        god_machine._assert_fla_available_if_requested(cfg)


def test_assert_fla_available_if_requested_permits_when_package_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_fn = object()
    monkeypatch.setattr(god_machine, "FLA_AVAILABLE", True)
    monkeypatch.setattr(god_machine, "fused_recurrent_simple_gla", sentinel_fn)
    cfg = god_machine.Config(use_fla_if_available=True)
    god_machine._assert_fla_available_if_requested(cfg)


def test_nm_force_no_fla_flips_config_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NM_FORCE_NO_FLA", "1")
    cfg = god_machine.Config(use_fla_if_available=True)
    overrides = god_machine._apply_runtime_env_overrides(cfg)
    assert cfg.use_fla_if_available is False
    assert any("NM_FORCE_NO_FLA" in s for s in overrides)