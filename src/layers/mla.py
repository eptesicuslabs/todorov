from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MLALayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_c: int = 128,
        d_R: int = 32,
        num_heads: int = 8,
        use_spikes: bool = False,
        spike_all: bool = False,
        spike_type: str = "ternary",
        spike_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_c = d_c
        self.d_R = d_R
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_spikes = use_spikes
        self.spike_all = spike_all

        self.kv_down_proj = nn.Linear(d_model, d_c, bias=False)
        self.k_up_proj = nn.Linear(d_c, d_model, bias=False)
        self.v_up_proj = nn.Linear(d_c, d_model, bias=False)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_rope_proj = nn.Linear(d_model, d_R, bias=False)
        self.k_rope_proj = nn.Linear(d_c, d_R, bias=False)

        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d_R, 2).float() / d_R))
        self.register_buffer("inv_freq", inv_freq)

        _spike_kwargs = spike_kwargs or {}

        if use_spikes and spike_all:
            from src.spikes.atmn_spike import create_spike_module
            self.q_spike = create_spike_module(spike_type, d_features=d_model, **_spike_kwargs)
            self.kv_spike = create_spike_module(spike_type, d_features=d_model, **_spike_kwargs)
        else:
            self.q_spike = None
            self.kv_spike = None

    def _compute_rope(self, seq_len: int, offset: int, device: torch.device) -> tuple[Tensor, Tensor]:
        t = torch.arange(offset, offset + seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    def _apply_rope(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        d = x.shape[-1]
        x1 = x[..., :d // 2]
        x2 = x[..., d // 2:]
        cos = cos[..., :d // 2]
        sin = sin[..., :d // 2]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def forward(
        self,
        x: Tensor,
        kv_cache: Tensor | None = None,
        offset: int = 0,
    ) -> tuple[Tensor, Tensor, dict]:
        batch_size, seq_len, _ = x.shape
        aux = {}

        if self.use_spikes and self.q_spike is not None:
            pre_spike_q = x.clone()
            q_input = self.q_spike(x)
            aux["pre_spike_q"] = pre_spike_q
            aux["q_spikes"] = q_input
        else:
            q_input = x

        if self.use_spikes and self.kv_spike is not None:
            pre_spike_kv = x.clone()
            kv_input = self.kv_spike(x)
            aux["pre_spike_kv"] = pre_spike_kv
            aux["kv_spikes"] = kv_input
        else:
            kv_input = x

        c_kv = self.kv_down_proj(kv_input)

        k_rope_shared = self.k_rope_proj(c_kv)

        cos, sin = self._compute_rope(seq_len, offset, x.device)
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        k_rope_shared = self._apply_rope(k_rope_shared, cos, sin)

        cache_entry = torch.cat([c_kv, k_rope_shared], dim=-1)

        if kv_cache is not None:
            cache_entry = torch.cat([kv_cache, cache_entry], dim=1)

        cached_c_kv = cache_entry[..., :self.d_c]
        cached_k_rope = cache_entry[..., self.d_c:]
        total_len = cache_entry.shape[1]

        k_content = self.k_up_proj(cached_c_kv)
        v = self.v_up_proj(cached_c_kv)

        k_content = k_content.view(batch_size, total_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, total_len, self.num_heads, self.head_dim)

        q = self.q_proj(q_input).view(batch_size, seq_len, self.num_heads, self.head_dim)
        q_rope_shared = self.q_rope_proj(q_input)

        q_cos, q_sin = self._compute_rope(seq_len, offset, x.device)
        q_cos = q_cos.unsqueeze(0)
        q_sin = q_sin.unsqueeze(0)
        q_rope_shared = self._apply_rope(q_rope_shared, q_cos, q_sin)

        q = q.transpose(1, 2)
        k_content = k_content.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_content = torch.matmul(q, k_content.transpose(-2, -1))

        q_rope_expanded = q_rope_shared.unsqueeze(2).expand(-1, -1, self.num_heads, -1).transpose(1, 2)
        k_rope_expanded = cached_k_rope.unsqueeze(2).expand(-1, -1, self.num_heads, -1).transpose(1, 2)
        attn_rope = torch.matmul(q_rope_expanded, k_rope_expanded.transpose(-2, -1))

        scale = math.sqrt(self.head_dim + self.d_R)
        attn_scores = (attn_content + attn_rope) / scale

        if seq_len > 1:
            causal_mask = torch.triu(
                torch.full((seq_len, total_len), float("-inf"), device=x.device),
                diagonal=total_len - seq_len + 1,
            )
            attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)

        cache_per_token = self.d_c + self.d_R
        aux["cache_size_per_token"] = cache_per_token
        aux["total_cached_tokens"] = total_len
        aux["compression_ratio"] = (2 * self.num_heads * self.head_dim) / cache_per_token

        return output, cache_entry, aux
