from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from config import TodorovConfig


@dataclass
class MemoryBudget:
    weights_mb: float
    kda_states_mb: float
    mamba3_states_mb: float
    mla_cache_mb: float
    activations_mb: float
    total_mb: float
    context_length: int
    precision_bytes: int


def calculate_theoretical_memory(
    config: TodorovConfig,
    context_length: int = 131072,
    precision_bytes: int = 1,
    batch_size: int = 1,
) -> MemoryBudget:
    layer_types = config.layer_types
    n_kda = sum(1 for lt in layer_types if lt == "KDA")
    n_mamba3 = sum(1 for lt in layer_types if lt == "Mamba3")
    n_mla = sum(1 for lt in layer_types if lt == "MLA")

    d = config.d_model
    h_kda = config.kda.num_heads
    hd_kda = config.kda.head_dim

    param_count = 0
    param_count += config.vocab_size * d
    for lt in layer_types:
        param_count += 2 * d
        if lt == "KDA":
            param_count += 4 * d * h_kda * hd_kda
            param_count += d * h_kda
            param_count += h_kda * hd_kda
        elif lt == "Mamba3":
            expand = config.mamba3.expand
            d_inner = d * expand
            d_state = config.mamba3.d_state
            param_count += d * d_inner * 2
            param_count += d * d_state * 2
            param_count += d * d_inner
            param_count += d_inner * d
            param_count += d_inner
        elif lt == "MLA":
            d_c = config.mla.d_c
            d_R = config.mla.d_R
            n_h = config.mla.num_heads
            param_count += d * d_c
            param_count += d_c * d
            param_count += d_c * d
            param_count += d * d
            param_count += d * d_R
            param_count += d_c * d_R
            param_count += d * d
        hidden_dim = int(d * config.mlp.ratio)
        hidden_dim = ((hidden_dim + 63) // 64) * 64
        param_count += 2 * d * hidden_dim + hidden_dim * d
        param_count += 2 * d

    weights_mb = param_count * precision_bytes / (1024 * 1024)

    kda_state_bytes = n_kda * batch_size * h_kda * hd_kda * hd_kda * 4
    kda_states_mb = kda_state_bytes / (1024 * 1024)

    mamba3_state_bytes = n_mamba3 * batch_size * (d * config.mamba3.expand) * config.mamba3.d_state * 4
    mamba3_states_mb = mamba3_state_bytes / (1024 * 1024)

    cache_per_token = config.mla.d_c + config.mla.d_R
    mla_cache_bytes = n_mla * batch_size * context_length * cache_per_token * 4
    mla_cache_mb = mla_cache_bytes / (1024 * 1024)

    activation_bytes = batch_size * d * 4 * 512
    activations_mb = activation_bytes / (1024 * 1024)

    total_mb = weights_mb + kda_states_mb + mamba3_states_mb + mla_cache_mb + activations_mb

    return MemoryBudget(
        weights_mb=round(weights_mb, 2),
        kda_states_mb=round(kda_states_mb, 2),
        mamba3_states_mb=round(mamba3_states_mb, 2),
        mla_cache_mb=round(mla_cache_mb, 2),
        activations_mb=round(activations_mb, 2),
        total_mb=round(total_mb, 2),
        context_length=context_length,
        precision_bytes=precision_bytes,
    )


def measure_actual_memory(model: nn.Module) -> dict:
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    param_count = sum(p.numel() for p in model.parameters())

    return {
        "param_count": param_count,
        "param_mb": round(param_bytes / (1024 * 1024), 2),
        "buffer_mb": round(buffer_bytes / (1024 * 1024), 2),
        "total_mb": round((param_bytes + buffer_bytes) / (1024 * 1024), 2),
    }


def verify_memory_budget(
    config: TodorovConfig,
    model: nn.Module,
    tolerance: float = 0.05,
) -> dict:
    theoretical = calculate_theoretical_memory(config, context_length=4096, precision_bytes=4)
    actual = measure_actual_memory(model)

    weight_diff_pct = abs(actual["param_mb"] - theoretical.weights_mb) / max(theoretical.weights_mb, 1e-6)

    return {
        "theoretical_weights_mb": theoretical.weights_mb,
        "actual_weights_mb": actual["param_mb"],
        "weight_diff_pct": round(weight_diff_pct * 100, 2),
        "within_tolerance": weight_diff_pct <= tolerance,
        "param_count": actual["param_count"],
        "theoretical_total_4k": theoretical.total_mb,
    }
