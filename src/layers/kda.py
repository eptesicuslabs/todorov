from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RotaryPositionEncoding(nn.Module):

    def __init__(self, dim: int, max_seq_len: int = 131072, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def _update_cache(self, seq_len: int, device: torch.device) -> None:
        if seq_len <= self._seq_len_cached and self._cos_cached is not None:
            return
        self._seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos_cached = emb.cos()
        self._sin_cached = emb.sin()

    def forward(self, x: Tensor, offset: int = 0) -> tuple[Tensor, Tensor]:
        seq_len = x.shape[-2] + offset
        self._update_cache(seq_len, x.device)
        cos = self._cos_cached[offset:offset + x.shape[-2]]
        sin = self._sin_cached[offset:offset + x.shape[-2]]
        return cos, sin


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    d = x.shape[-1]
    x1 = x[..., :d // 2]
    x2 = x[..., d // 2:]
    cos = cos[..., :d // 2]
    sin = sin[..., :d // 2]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class KDALayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        head_dim: int = 128,
        channel_wise_gate: bool = True,
        use_spikes: bool = False,
        spike_module: nn.Module | None = None,
        spike_all: bool = False,
        spike_type: str = "ternary",
        spike_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.channel_wise_gate = channel_wise_gate
        self.inner_dim = num_heads * head_dim

        self.q_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.o_proj = nn.Linear(self.inner_dim, d_model, bias=False)

        self.beta_proj = nn.Linear(d_model, num_heads, bias=True)

        if channel_wise_gate:
            self.alpha_log = nn.Parameter(torch.zeros(num_heads, head_dim))
        else:
            self.alpha_log = nn.Parameter(torch.zeros(num_heads, 1))

        self.rope = RotaryPositionEncoding(head_dim)

        self.use_spikes = use_spikes
        self.spike_all = spike_all
        _spike_kwargs = spike_kwargs or {}

        if use_spikes:
            from src.spikes.atmn_spike import create_spike_module
            self.k_spike = create_spike_module(spike_type, d_features=self.inner_dim, **_spike_kwargs)
            self.v_spike = create_spike_module(spike_type, d_features=self.inner_dim, **_spike_kwargs)
        else:
            self.k_spike = None
            self.v_spike = None

        if use_spikes and spike_all:
            from src.spikes.atmn_spike import create_spike_module
            self.q_spike = create_spike_module(spike_type, d_features=self.inner_dim, **_spike_kwargs)
            self.o_spike = create_spike_module(spike_type, d_features=d_model, **_spike_kwargs)
        else:
            self.q_spike = None
            self.o_spike = None

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.alpha_log, mean=-2.0, std=0.5)

    def _forward_parallel(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        alpha: Tensor,
        beta: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size, seq_len, num_heads, head_dim = q.shape

        q_h = q.transpose(1, 2)
        k_h = k.transpose(1, 2)
        v_h = v.transpose(1, 2)

        qk = torch.matmul(q_h, k_h.transpose(-2, -1))

        if self.channel_wise_gate:
            alpha_mean = alpha.mean(dim=-1)
        else:
            alpha_mean = alpha.squeeze(-1)
        log_alpha = torch.log(alpha_mean.clamp(min=1e-6))

        positions = torch.arange(seq_len, device=q.device, dtype=q.dtype)
        time_diff = positions.unsqueeze(1) - positions.unsqueeze(0)
        decay_exp = time_diff.unsqueeze(0) * log_alpha.unsqueeze(-1).unsqueeze(-1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        decay_exp = decay_exp.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
        decay = torch.exp(decay_exp)

        beta_expanded = beta.transpose(1, 2).unsqueeze(2)
        weights = qk * decay.unsqueeze(0) * beta_expanded

        output = torch.matmul(weights, v_h)
        output = output.transpose(1, 2)

        with torch.no_grad():
            alpha_unsq = alpha.unsqueeze(0).unsqueeze(-1)
            accumulated_state = torch.zeros(
                batch_size, num_heads, head_dim, head_dim,
                device=q.device, dtype=q.dtype,
            )
            for t in range(seq_len):
                beta_t = beta[:, t].unsqueeze(-1).unsqueeze(-1)
                accumulated_state = alpha_unsq * accumulated_state + beta_t * torch.einsum("bhd,bhe->bhde", k[:, t], v[:, t])

        return output, accumulated_state

    def _forward_recurrent(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        alpha: Tensor,
        beta: Tensor,
        state: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        batch_size, seq_len, num_heads, head_dim = q.shape

        alpha_unsq = alpha.unsqueeze(0).unsqueeze(-1)

        if state is None:
            state = torch.zeros(
                batch_size, num_heads, head_dim, head_dim,
                device=q.device, dtype=q.dtype,
            )

        outputs = []
        for t in range(seq_len):
            beta_t = beta[:, t].unsqueeze(-1).unsqueeze(-1)
            state = alpha_unsq * state + beta_t * torch.einsum("bhd,bhe->bhde", k[:, t], v[:, t])
            o_t = torch.einsum("bhd,bhde->bhe", q[:, t], state)
            outputs.append(o_t)

        return torch.stack(outputs, dim=1), state

    def forward(
        self,
        x: Tensor,
        state: Tensor | None = None,
        offset: int = 0,
    ) -> tuple[Tensor, Tensor, dict]:
        batch_size, seq_len, _ = x.shape
        aux = {}

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        if self.use_spikes and self.q_spike is not None:
            q_flat = q.view(batch_size, seq_len, self.inner_dim)
            pre_spike_q = q_flat.clone()
            q_flat = self.q_spike(q_flat)
            aux["pre_spike_q"] = pre_spike_q
            aux["q_spikes"] = q_flat
            q = q_flat.view(batch_size, seq_len, self.num_heads, self.head_dim)

        cos, sin = self.rope(q, offset=offset)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q = apply_rotary(q.transpose(1, 2), cos, sin).transpose(1, 2)
        k = apply_rotary(k.transpose(1, 2), cos, sin).transpose(1, 2)

        if self.use_spikes and self.k_spike is not None:
            pre_spike_k = k.clone()
            k = self.k_spike(k)
            aux["pre_spike_k"] = pre_spike_k
            aux["k_spikes"] = k

        if self.use_spikes and self.v_spike is not None:
            pre_spike_v = v.clone()
            v = self.v_spike(v)
            aux["pre_spike_v"] = pre_spike_v
            aux["v_spikes"] = v

        alpha = torch.sigmoid(self.alpha_log)
        beta = torch.sigmoid(self.beta_proj(x))

        if state is not None or seq_len == 1:
            output, state = self._forward_recurrent(q, k, v, alpha, beta, state)
        else:
            output, state = self._forward_parallel(q, k, v, alpha, beta)

        output = output.reshape(batch_size, seq_len, self.inner_dim)

        if self.use_spikes and self.o_spike is not None:
            pre_spike_o = output.clone()
            output = self.o_spike(output)
            aux["pre_spike_o"] = pre_spike_o
            aux["o_spikes"] = output

        output = self.o_proj(output)

        aux["state_norm"] = state.norm().item()

        return output, state, aux
