from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SwiGLU(nn.Module):

    def __init__(
        self,
        d_model: int,
        ratio: float = 2.75,
        spatial_mode: bool = False,
        use_spikes: bool = False,
        spike_type: str = "ternary",
        spike_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.spatial_mode = spatial_mode
        self.use_spikes = use_spikes
        hidden_dim = int(d_model * ratio)
        hidden_dim = ((hidden_dim + 63) // 64) * 64

        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

        if spatial_mode:
            self.w_left = nn.Linear(d_model, 16, bias=False)
            self.w_right = nn.Linear(d_model, 16, bias=False)
            self.gp_proj = nn.Linear(16, d_model, bias=False)

        _spike_kwargs = spike_kwargs or {}

        if use_spikes:
            from src.spikes.atmn_spike import create_spike_module
            self.gate_spike = create_spike_module(spike_type, d_features=d_model, **_spike_kwargs)
            self.up_spike = create_spike_module(spike_type, d_features=d_model, **_spike_kwargs)
        else:
            self.gate_spike = None
            self.up_spike = None

    def _geometric_product_self_interaction(self, x: Tensor) -> Tensor:
        from src.algebra.geometric_product import geometric_product
        from src.algebra.multivector import Multivector

        left = Multivector(self.w_left(x))
        right = Multivector(self.w_right(x))
        product = geometric_product(left, right)
        return self.gp_proj(product.values)

    def forward(self, x: Tensor) -> tuple[Tensor, dict]:
        aux = {}

        if self.use_spikes and self.gate_spike is not None:
            pre_spike_gate = x.clone()
            gate_input = self.gate_spike(x)
            aux["pre_spike_gate"] = pre_spike_gate
            aux["gate_spikes"] = gate_input
        else:
            gate_input = x

        if self.use_spikes and self.up_spike is not None:
            pre_spike_up = x.clone()
            up_input = self.up_spike(x)
            aux["pre_spike_up"] = pre_spike_up
            aux["up_spikes"] = up_input
        else:
            up_input = x

        gate = F.silu(self.w_gate(gate_input))
        up = self.w_up(up_input)
        out = self.w_down(gate * up)

        if self.spatial_mode:
            out = out + self._geometric_product_self_interaction(x)

        return out, aux
