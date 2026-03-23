from __future__ import annotations

import torch
from torch import Tensor, nn


def measure_erf(
    model: nn.Module,
    input_ids: Tensor,
    target_position: int = -1,
    target_dim: int = 0,
) -> Tensor:
    model.eval()
    input_ids = input_ids.detach().clone()

    embedding = None
    for module in model.modules():
        if hasattr(module, "token_embedding"):
            embedding = module.token_embedding
            break

    if embedding is None:
        return torch.zeros(input_ids.shape[1])

    embeds = embedding(input_ids).detach().requires_grad_(True)

    hook_output = []

    def replace_embedding_hook(module: nn.Module, args: tuple, output: Tensor) -> Tensor:
        return embeds

    for module in model.modules():
        if hasattr(module, "token_embedding"):
            handle = module.register_forward_hook(replace_embedding_hook)
            break

    logits, _, _ = model(input_ids)
    scalar = logits[:, target_position, target_dim].sum()
    scalar.backward()

    handle.remove()

    if embeds.grad is not None:
        erf = embeds.grad.abs().mean(dim=(0, 2))
    else:
        erf = torch.zeros(input_ids.shape[1])

    model.train()
    return erf


def compute_erf_statistics(erf: Tensor) -> dict:
    normalized = erf / (erf.max() + 1e-8)
    cumulative = torch.cumsum(normalized, dim=0) / (normalized.sum() + 1e-8)
    effective_length = (cumulative < 0.9).sum().item() + 1

    return {
        "effective_length": effective_length,
        "total_length": len(erf),
        "coverage_ratio": effective_length / len(erf),
        "peak_position": erf.argmax().item(),
        "max_gradient": erf.max().item(),
    }
