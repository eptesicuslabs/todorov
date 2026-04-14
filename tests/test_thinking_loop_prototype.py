from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIM_MODULE_PATH = PROJECT_ROOT / "neuroloc" / "simulations" / "reasoning" / "thinking_loop_prototype.py"
SIM_ROOT = PROJECT_ROOT / "neuroloc" / "simulations"

if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))


def _load_sim_module():
    spec = importlib.util.spec_from_file_location("thinking_loop_prototype", SIM_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load thinking_loop_prototype module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_task_generator_is_deterministic_under_seed() -> None:
    module = _load_sim_module()
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    tokens_a, targets_a = module.generate_task_batch(rng_a, 32, 6, module.MODULUS, module.N_OPERATIONS)
    tokens_b, targets_b = module.generate_task_batch(rng_b, 32, 6, module.MODULUS, module.N_OPERATIONS)
    assert np.array_equal(tokens_a, tokens_b)
    assert np.array_equal(targets_a, targets_b)


def test_task_targets_are_in_modulus_range() -> None:
    module = _load_sim_module()
    rng = np.random.default_rng(7)
    _, targets = module.generate_task_batch(rng, 64, 8, module.MODULUS, module.N_OPERATIONS)
    assert targets.min() >= 0
    assert targets.max() < module.MODULUS


def test_compute_result_matches_reference_sequence() -> None:
    module = _load_sim_module()
    operations = np.array([[0, 2, 1]])
    constants = np.array([[3, 2, 1]])
    start_values = np.zeros(1, dtype=np.int64)
    targets = module.compute_result(start_values, operations, constants, module.MODULUS)
    expected = ((0 + 3) * 2 - 1) % module.MODULUS
    assert int(targets[0]) == int(expected)


def test_no_thinking_trains_to_full_accuracy_on_trivial_depth() -> None:
    module = _load_sim_module()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    max_seq_len = 2
    vocab_size = module.build_vocab_size(module.MODULUS, module.N_OPERATIONS)
    model = module.ReasoningModel(
        vocab_size=vocab_size,
        modulus=module.MODULUS,
        d_model=48,
        depth=4,
        max_seq_len=max_seq_len,
    )
    history = module.train_model(
        model=model,
        rng=rng,
        n_ops_train=2,
        n_train=4096,
        n_val=512,
        n_epochs=40,
        batch_size=64,
        learning_rate=3e-3,
        think_steps=1,
        device=torch.device("cpu"),
    )
    assert history["val_acc"][-1] > 0.99


def test_thinking_k1_matches_no_thinking_forward() -> None:
    module = _load_sim_module()
    torch.manual_seed(41)
    rng = np.random.default_rng(41)
    max_seq_len = 4
    vocab_size = module.build_vocab_size(module.MODULUS, module.N_OPERATIONS)
    model = module.ReasoningModel(
        vocab_size=vocab_size,
        modulus=module.MODULUS,
        d_model=32,
        depth=3,
        max_seq_len=max_seq_len,
    )
    tokens, _ = module.generate_task_batch(rng, 16, 4, module.MODULUS, module.N_OPERATIONS)
    tokens_t = torch.from_numpy(tokens)
    with torch.no_grad():
        logits_k1, trace_k1 = model(tokens_t, n_think=1)
        logits_once, summary_once = model.forward_once(tokens_t, None)
    assert torch.allclose(logits_k1, logits_once, atol=1e-6)
    assert len(trace_k1) == 1
    assert torch.allclose(trace_k1[0], summary_once, atol=1e-6)


def test_smoke_run_emits_required_summary_keys(tmp_path: Path) -> None:
    output_dir = tmp_path / "thinking_smoke"
    output_dir.mkdir()
    env = {
        **os.environ,
        "SIM_OUTPUT_DIR": str(output_dir),
        "THINK_N_TRAIN": "256",
        "THINK_N_VAL": "64",
        "THINK_N_TEST": "64",
        "THINK_N_EPOCHS": "2",
        "THINK_BATCH_SIZE": "64",
        "THINK_D_MODEL": "24",
        "THINK_DEPTH": "2",
        "THINK_DEEP_DEPTH": "3",
        "THINK_N_LIST": "2,4",
        "THINK_N_TRAIN_CAP": "2",
        "THINK_K_LIST": "1,2",
        "THINK_N_SEEDS": "2",
        "THINK_TRAIN_K": "1",
        "THINK_CONV_K_MAX": "4",
    }
    completed = subprocess.run(
        [sys.executable, str(SIM_MODULE_PATH)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    metrics_path = output_dir / "thinking_loop_prototype_metrics.json"
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    required_keys = (
        "accuracy_by_K_and_N",
        "per_step_gain_by_task_difficulty",
        "generalization_gap_thinking_vs_feedforward",
        "fixed_point_convergence_fraction",
    )
    for key in required_keys:
        assert key in summary, f"missing summary key: {key}"
    assert isinstance(summary["accuracy_by_K_and_N"], dict)
    assert isinstance(summary["per_step_gain_by_task_difficulty"], dict)
    for k_key, per_n in summary["accuracy_by_K_and_N"].items():
        assert isinstance(per_n, dict)
        for n_key, entry in per_n.items():
            assert entry.get("mean") is not None


def test_registry_exposes_reasoning_suite() -> None:
    from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES

    assert "thinking_loop_prototype" in SIMULATION_SPECS
    assert SIMULATION_SPECS["thinking_loop_prototype"].category == "reasoning"
    assert "reasoning" in SUITES
    assert "thinking_loop_prototype" in SUITES["reasoning"]


@pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None, reason="matplotlib required")
def test_model_parameters_stay_near_target() -> None:
    module = _load_sim_module()
    vocab_size = module.build_vocab_size(module.MODULUS, module.N_OPERATIONS)
    model = module.ReasoningModel(
        vocab_size=vocab_size,
        modulus=module.MODULUS,
        d_model=48,
        depth=4,
        max_seq_len=32,
    )
    n_params = module.count_parameters(model)
    assert 20_000 <= n_params <= 120_000
