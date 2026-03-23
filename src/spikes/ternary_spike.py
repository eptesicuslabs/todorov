import torch
import torch.nn as nn
from typing import Optional


class TernaryQuantizer(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        threshold: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, threshold)
        ctx.training = training

        output = torch.zeros_like(x)
        output[x > threshold] = 1.0
        output[x < -threshold] = -1.0
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        grad_x = grad_output.clone()
        return grad_x, None, None


class AdaptiveTernarySpike(nn.Module):

    def __init__(
        self,
        alpha_init: float = 1.0,
        learnable_alpha: bool = True,
        min_threshold: float = 0.01,
        max_threshold: float = 10.0,
    ) -> None:
        super().__init__()

        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha_init))
        else:
            self.register_buffer("alpha", torch.tensor(alpha_init))

        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        self.register_buffer("running_spike_density", torch.tensor(0.0))
        self.register_buffer("running_threshold", torch.tensor(alpha_init))
        self.register_buffer("num_updates", torch.tensor(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_abs = torch.mean(torch.abs(x))
        threshold = self.alpha * mean_abs
        threshold = torch.clamp(threshold, self.min_threshold, self.max_threshold)

        spikes = TernaryQuantizer.apply(x, threshold, self.training)

        if self.training:
            with torch.no_grad():
                density = (spikes != 0).float().mean()
                momentum = 0.99
                self.running_spike_density = (
                    momentum * self.running_spike_density + (1 - momentum) * density
                )
                self.running_threshold = (
                    momentum * self.running_threshold + (1 - momentum) * threshold
                )
                self.num_updates += 1

        return spikes

    def get_spike_density(self) -> float:
        return self.running_spike_density.item()

    def get_threshold(self) -> float:
        return self.running_threshold.item()

    def get_alpha(self) -> float:
        return self.alpha.item()

    def extra_repr(self) -> str:
        return f"alpha={self.alpha.item():.3f}, learnable={isinstance(self.alpha, nn.Parameter)}"


def compute_spike_statistics(
    spikes: torch.Tensor,
    prev_spikes: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    with torch.no_grad():
        stats: dict[str, float] = {
            "density": (spikes != 0).float().mean().item(),
            "positive_fraction": (spikes == 1).float().mean().item(),
            "negative_fraction": (spikes == -1).float().mean().item(),
            "zero_fraction": (spikes == 0).float().mean().item(),
        }

        if prev_spikes is not None and prev_spikes.shape == spikes.shape:
            stats["flicker_rate"] = (spikes != prev_spikes).float().mean().item()

        return stats
