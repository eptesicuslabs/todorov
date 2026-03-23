from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Mamba3Layer(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_state: int = 32,
        expand: int = 2,
        use_rope: bool = True,
        use_trapezoidal: bool = True,
        use_spikes: bool = False,
        spike_all: bool = False,
        spike_type: str = "ternary",
        spike_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = d_model * expand
        self.use_rope = use_rope
        self.use_trapezoidal = use_trapezoidal
        self.use_spikes = use_spikes
        self.spike_all = spike_all

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)

        self.dt_proj = nn.Linear(d_model, self.d_inner, bias=True)
        self.dt_bias = nn.Parameter(torch.zeros(self.d_inner) - 4.0)

        if use_rope:
            self.rope_freq = nn.Parameter(torch.randn(d_state // 2) * 0.01)

        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        _spike_kwargs = spike_kwargs or {}

        if use_spikes and spike_all:
            from src.spikes.atmn_spike import create_spike_module
            self.in_spike = create_spike_module(spike_type, d_features=d_model, **_spike_kwargs)
            self.out_spike = create_spike_module(spike_type, d_features=self.d_inner, **_spike_kwargs)
        else:
            self.in_spike = None
            self.out_spike = None

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.uniform_(self.A_log, -4.0, -1.0)

    def _discretize_trapezoidal(
        self, A: Tensor, dt: Tensor
    ) -> tuple[Tensor, Tensor]:
        dtA = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        A_bar = (1.0 + dtA / 2.0) / (1.0 - dtA / 2.0)
        B_bar = dt.unsqueeze(-1) / (1.0 - dtA / 2.0)
        return A_bar, B_bar

    def _discretize_exponential(
        self, A: Tensor, dt: Tensor
    ) -> tuple[Tensor, Tensor]:
        dtA = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        A_bar = torch.exp(dtA)
        B_bar = dt.unsqueeze(-1) * torch.ones_like(dtA)
        return A_bar, B_bar

    def _apply_data_dependent_rope(
        self, state: Tensor, t: int
    ) -> Tensor:
        if not self.use_rope:
            return state
        half_n = self.d_state // 2
        freq = self.rope_freq.to(state.device)
        angle = freq * t
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        real = state[..., :half_n]
        imag = state[..., half_n:]
        rotated_real = real * cos_a - imag * sin_a
        rotated_imag = real * sin_a + imag * cos_a
        return torch.cat([rotated_real, rotated_imag], dim=-1)

    def forward(
        self,
        x: Tensor,
        state: Tensor | None = None,
        offset: int = 0,
    ) -> tuple[Tensor, Tensor, dict]:
        batch_size, seq_len, _ = x.shape
        aux = {}

        if self.use_spikes and self.in_spike is not None:
            pre_spike_in = x.clone()
            x_spiked = self.in_spike(x)
            aux["pre_spike_in"] = pre_spike_in
            aux["in_spikes"] = x_spiked
            xz = self.in_proj(x_spiked)
        else:
            xz = self.in_proj(x)

        x_inner, z = xz.chunk(2, dim=-1)
        z = F.silu(z)

        A = -torch.exp(self.A_log)
        B = self.B_proj(x)
        C = self.C_proj(x)
        dt = F.softplus(self.dt_proj(x) + self.dt_bias)

        if self.use_trapezoidal:
            A_bar, B_bar = self._discretize_trapezoidal(A, dt)
        else:
            A_bar, B_bar = self._discretize_exponential(A, dt)

        if state is None:
            state = torch.zeros(
                batch_size, self.d_inner, self.d_state,
                device=x.device, dtype=x.dtype,
            )

        outputs = []
        for t in range(seq_len):
            B_t = B[:, t].unsqueeze(1).expand(-1, self.d_inner, -1)
            x_t = x_inner[:, t].unsqueeze(-1)

            state = A_bar[:, t] * state + B_bar[:, t] * B_t * x_t
            state = self._apply_data_dependent_rope(state, t + offset)

            C_t = C[:, t].unsqueeze(1).expand(-1, self.d_inner, -1)
            y_t = (state * C_t).sum(dim=-1)
            outputs.append(y_t)

        output = torch.stack(outputs, dim=1)
        output = output * z
        output = self.norm(output)

        if self.use_spikes and self.out_spike is not None:
            pre_spike_out = output.clone()
            output = self.out_spike(output)
            aux["pre_spike_out"] = pre_spike_out
            aux["out_spikes"] = output

        output = self.out_proj(output)

        aux["state_utilization"] = (state.abs() > 1e-6).float().mean().item()

        return output, state, aux
