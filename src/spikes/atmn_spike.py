from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ATMNQuantizer(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        membrane_potential: Tensor,
        threshold: Tensor,
    ) -> Tensor:
        ctx.save_for_backward(membrane_potential, threshold)
        output = torch.zeros_like(membrane_potential)
        output[membrane_potential >= threshold] = 1.0
        output[membrane_potential <= -threshold] = -1.0
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        return grad_output.clone(), None


class ATMNSpike(nn.Module):

    def __init__(
        self,
        d_features: int,
        tau: float = 2.0,
        threshold_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_features = d_features
        self.tau = tau
        self.threshold_log = nn.Parameter(torch.full((d_features,), threshold_init))
        self.register_buffer("membrane_potential", torch.zeros(1))
        self.register_buffer("running_spike_density", torch.tensor(0.0))
        self.register_buffer("num_updates", torch.tensor(0))
        self._state_initialized = False

    @property
    def threshold(self) -> Tensor:
        return torch.exp(self.threshold_log)

    def reset_state(self, batch_size: int = 1, device: torch.device | None = None) -> None:
        dev = device or self.threshold_log.device
        self.membrane_potential = torch.zeros(batch_size, self.d_features, device=dev)
        self._state_initialized = True

    def _ensure_state(self, x: Tensor) -> None:
        if not self._state_initialized or self.membrane_potential.shape[0] != x.shape[0]:
            flat = x.reshape(-1, x.shape[-1])
            self.reset_state(flat.shape[0], x.device)

    def forward(self, x: Tensor) -> Tensor:
        original_shape = x.shape
        features_dim = x.shape[-1]

        if features_dim != self.d_features:
            x_flat = x.reshape(-1, features_dim)
        else:
            x_flat = x.reshape(-1, features_dim)

        batch_flat = x_flat.shape[0]

        if not self._state_initialized or self.membrane_potential.shape[0] != batch_flat:
            self.reset_state(batch_flat, x.device)

        v_th = self.threshold

        h = x_flat + (1.0 / self.tau) * self.membrane_potential
        spikes = ATMNQuantizer.apply(h, v_th)
        self.membrane_potential = (h - spikes * v_th).detach()

        if self.training:
            with torch.no_grad():
                density = (spikes != 0).float().mean()
                self.running_spike_density = 0.99 * self.running_spike_density + 0.01 * density
                self.num_updates += 1

        return spikes.view(original_shape)

    def get_spike_density(self) -> float:
        return self.running_spike_density.item()

    def get_threshold_mean(self) -> float:
        return self.threshold.mean().item()

    def extra_repr(self) -> str:
        return f"d_features={self.d_features}, tau={self.tau}"


def create_spike_module(
    spike_type: str,
    d_features: int | None = None,
    **kwargs,
) -> nn.Module:
    if spike_type == "ternary":
        from src.spikes.ternary_spike import AdaptiveTernarySpike
        return AdaptiveTernarySpike(
            alpha_init=kwargs.get("alpha_init", 1.0),
            learnable_alpha=kwargs.get("learnable_alpha", True),
            min_threshold=kwargs.get("min_threshold", 0.01),
            max_threshold=kwargs.get("max_threshold", 10.0),
        )
    elif spike_type == "atmn":
        if d_features is None:
            raise ValueError("d_features required for ATMN spikes")
        return ATMNSpike(
            d_features=d_features,
            tau=kwargs.get("tau", 2.0),
            threshold_init=kwargs.get("threshold_init", 0.0),
        )
    else:
        raise ValueError(f"Unknown spike_type: {spike_type}")
