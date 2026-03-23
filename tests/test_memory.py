import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from config import TINY_CONFIG, BASE_CONFIG, TodorovConfig
from src.utils.memory import calculate_theoretical_memory, measure_actual_memory, verify_memory_budget
from src.model.todorov import Todorov


class TestMemoryBudget:

    def test_theoretical_calculation(self) -> None:
        budget = calculate_theoretical_memory(BASE_CONFIG, context_length=4096, precision_bytes=1)
        assert budget.weights_mb > 0
        assert budget.kda_states_mb > 0
        assert budget.mamba3_states_mb > 0
        assert budget.mla_cache_mb > 0
        assert budget.total_mb > 0

    def test_mla_cache_scales_linearly(self) -> None:
        budget_4k = calculate_theoretical_memory(BASE_CONFIG, context_length=4096)
        budget_128k = calculate_theoretical_memory(BASE_CONFIG, context_length=131072)
        ratio = budget_128k.mla_cache_mb / max(budget_4k.mla_cache_mb, 1e-6)
        expected_ratio = 131072 / 4096
        assert abs(ratio - expected_ratio) / expected_ratio < 0.01

    def test_kda_state_constant(self) -> None:
        budget_4k = calculate_theoretical_memory(BASE_CONFIG, context_length=4096)
        budget_128k = calculate_theoretical_memory(BASE_CONFIG, context_length=131072)
        assert abs(budget_4k.kda_states_mb - budget_128k.kda_states_mb) < 0.01

    def test_actual_measurement(self) -> None:
        model = Todorov(TINY_CONFIG)
        actual = measure_actual_memory(model)
        assert actual["param_count"] > 0
        assert actual["param_mb"] > 0

    def test_verify_budget_tiny(self) -> None:
        config = TINY_CONFIG
        model = Todorov(config)
        result = verify_memory_budget(config, model, tolerance=0.20)
        assert "theoretical_weights_mb" in result
        assert "actual_weights_mb" in result
        assert result["param_count"] > 0
