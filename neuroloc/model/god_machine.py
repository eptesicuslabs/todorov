from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import random as py_random
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset


FLA_AVAILABLE = False
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    FLA_AVAILABLE = True
except Exception:
    chunk_gated_delta_rule = None

try:
    from fla.ops import chunk_simple_gla
except Exception:
    chunk_simple_gla = None

try:
    from fla.ops import fused_chunk_simple_gla
except Exception:
    fused_chunk_simple_gla = None


def _select_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        cap = torch.cuda.get_device_capability()
        if cap[0] < 7:
            return torch.device("cpu")
    except Exception:
        pass
    return torch.device("cuda")


DEVICE = _select_device()
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path("/kaggle/working") if Path("/kaggle").exists() else REPO_ROOT / "neuroloc" / "output"


def _data_root() -> Path:
    return Path(os.environ.get("NM_DATA_ROOT", str(REPO_ROOT / "data")))


def _absolute_output_path(path: Path) -> Path:
    if path.is_absolute():
        return Path(os.path.abspath(os.fspath(path)))
    return Path(os.path.abspath(os.fspath(REPO_ROOT / path)))


def _canonicalize_output_pair(path: Path) -> tuple[Path, Path]:
    lexical = _absolute_output_path(path)
    return lexical, lexical.resolve(strict=False)


def _canonical_output_path(path: Path) -> Path:
    return _canonicalize_output_pair(path)[1]


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


@dataclass
class Config:
    d_model: int = 1024
    n_layers: int = 28
    vocab_size: int = 256
    max_seq_len: int = 2048

    delta_num_heads: int = 16
    delta_head_dim: int = 64
    alpha_log_mean: float = -0.5
    alpha_log_std: float = 0.3

    attn_d_c: int = 128
    attn_d_R: int = 32
    attn_num_heads: int = 16

    mlp_ratio: float = 2.75
    num_compartments: int = 4

    layer_pattern: tuple[str, ...] = ("DELTA", "DELTA", "DELTA", "DELTA", "DELTA", "DELTA", "ATTN")

    kwta_enabled: bool = True
    kwta_fraction: float = 0.20

    delta_erasure_enabled: bool = True
    bcm_alpha_enabled: bool = True
    gamma_bcm: float = 0.3

    imagination_enabled: bool = True
    imag_gate_init: float = -2.0

    pc_diagnostic_enabled: bool = True
    pc_lambda: float = 1e-4

    multi_compartment_enabled: bool = True

    use_fla_if_available: bool = True

    slot_num_slots: int = 64
    slot_log_temperature_init: float = -2.3
    slot_surprise_tau: float = 0.1
    slot_gate_init: float = -4.0
    slot_predict_rank: int = 8
    slot_prediction_lambda: float = 1e-3

    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    batch_size: int = 16
    seq_len: int = 2048
    max_steps: int = 4000
    warmup_steps: int = 200
    val_interval: int = 200
    grad_clip: float = 1.0
    grad_accum_steps: int = 1

    grad_checkpointing: bool = True
    amp: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        if self.d_model % self.delta_num_heads != 0:
            raise ValueError(f"d_model {self.d_model} must be divisible by delta_num_heads {self.delta_num_heads}")
        if self.d_model % self.attn_num_heads != 0:
            raise ValueError(f"d_model {self.d_model} must be divisible by attn_num_heads {self.attn_num_heads}")
        if self.n_layers % len(self.layer_pattern) != 0:
            raise ValueError(
                f"n_layers {self.n_layers} must be divisible by len(layer_pattern) {len(self.layer_pattern)}"
            )
        if self.multi_compartment_enabled:
            if self.d_model % self.num_compartments != 0:
                raise ValueError(
                    f"d_model {self.d_model} must be divisible by num_compartments {self.num_compartments} "
                    f"when multi_compartment_enabled=True"
                )
            hidden_raw = int(self.d_model * self.mlp_ratio)
            hidden_rounded = ((hidden_raw + 63) // 64) * 64
            if hidden_rounded % self.num_compartments != 0:
                raise ValueError(
                    f"rounded mlp hidden {hidden_rounded} (from d_model={self.d_model} * mlp_ratio={self.mlp_ratio}) "
                    f"must be divisible by num_compartments {self.num_compartments}"
                )
        if self.kwta_enabled and not (0.0 < self.kwta_fraction <= 1.0):
            raise ValueError(f"kwta_fraction must be in (0, 1], got {self.kwta_fraction}")

    @property
    def layer_types(self) -> list[str]:
        pattern = list(self.layer_pattern)
        repeats = self.n_layers // len(pattern)
        return pattern * repeats


class RotaryPE(nn.Module):

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"rotary dim must be even, got {dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache_len = 0
        self._cache_device: torch.device | None = None
        self._cos: Tensor | None = None
        self._sin: Tensor | None = None

    def _build(self, length: int, device: torch.device) -> None:
        if (
            self._cos is not None
            and length <= self._cache_len
            and self._cache_device == device
        ):
            return
        self._cache_len = length
        self._cache_device = device
        t = torch.arange(length, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos = emb.cos()
        self._sin = emb.sin()

    def forward(self, seq_len: int, offset: int, device: torch.device) -> tuple[Tensor, Tensor]:
        self._build(seq_len + offset, device)
        return self._cos[offset:offset + seq_len], self._sin[offset:offset + seq_len]


def rotary_apply(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    cos = cos[..., : d // 2]
    sin = sin[..., : d // 2]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class TernaryQuantizer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, threshold: Tensor) -> Tensor:
        pos = x > threshold
        neg = x < -threshold
        out = torch.zeros_like(x)
        out[pos] = 1.0
        out[neg] = -1.0
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        return grad_output.clone(), None


class AdaptiveSpike(nn.Module):

    def __init__(
        self,
        alpha_init: float = 1.0,
        min_threshold: float = 0.01,
        max_threshold: float = 10.0,
    ) -> None:
        super().__init__()
        self.min_threshold = float(min_threshold)
        self.max_threshold = float(max_threshold)
        self.register_buffer("alpha", torch.tensor(float(alpha_init)), persistent=True)
        self.register_buffer("running_density", torch.tensor(0.0), persistent=True)
        self.register_buffer("running_threshold", torch.tensor(float(alpha_init)), persistent=True)
        self.register_buffer("n_updates", torch.tensor(0, dtype=torch.long), persistent=True)

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            mean_abs = x.abs().mean()
            threshold = torch.clamp(
                self.alpha * mean_abs, self.min_threshold, self.max_threshold
            )
        spikes = TernaryQuantizer.apply(x, threshold)
        if self.training:
            with torch.no_grad():
                density = (spikes != 0).float().mean()
                self.running_density.mul_(0.99).add_(density, alpha=0.01)
                self.running_threshold.mul_(0.99).add_(threshold, alpha=0.01)
                self.n_updates += 1
        return spikes


def kwta_select(x: Tensor, k_fraction: float) -> tuple[Tensor, Tensor]:
    d = x.shape[-1]
    k = max(1, int(round(d * k_fraction)))
    if k >= d:
        return x, torch.ones_like(x)
    _, topk_idx = x.abs().topk(k, dim=-1)
    mask = torch.zeros_like(x).scatter_(-1, topk_idx, 1.0)
    return x * mask, mask


class KWTAModule(nn.Module):

    def __init__(self, k_fraction: float = 0.20) -> None:
        super().__init__()
        self.k_fraction = float(k_fraction)
        self.register_buffer("last_realized_rate", torch.tensor(0.0), persistent=False)
        self.register_buffer("running_rate", torch.tensor(0.0), persistent=True)
        self.register_buffer("n_updates", torch.tensor(0, dtype=torch.long), persistent=True)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        out, mask = kwta_select(x, self.k_fraction)
        if self.training:
            with torch.no_grad():
                realized = mask.mean()
                self.last_realized_rate.copy_(realized)
                self.running_rate.mul_(0.99).add_(realized, alpha=0.01)
                self.n_updates += 1
        return out, mask


class RMSNorm(nn.Module):

    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x_fp = x.float()
        rms = (x_fp.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()
        return (x_fp * rms).to(input_dtype) * self.weight


class SwiGLU(nn.Module):

    def __init__(self, d_model: int, ratio: float = 2.75) -> None:
        super().__init__()
        hidden = int(d_model * ratio)
        hidden = ((hidden + 63) // 64) * 64
        self.hidden = hidden
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> tuple[Tensor, dict]:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up), {}


class MultiCompartmentSwiGLU(nn.Module):

    def __init__(self, d_model: int, ratio: float = 2.75, num_compartments: int = 4) -> None:
        super().__init__()
        if d_model % num_compartments != 0:
            raise ValueError(
                f"d_model {d_model} must be divisible by num_compartments {num_compartments}"
            )
        self.d_model = d_model
        self.K = num_compartments
        self.sub_d = d_model // num_compartments
        hidden = int(d_model * ratio)
        hidden = ((hidden + 63) // 64) * 64
        self.hidden = hidden
        if hidden % num_compartments != 0:
            raise ValueError(
                f"rounded hidden {hidden} must be divisible by num_compartments {num_compartments}"
            )
        self.sub_hidden = hidden // num_compartments

        self.w_gates = nn.Parameter(torch.empty(self.K, self.sub_d, self.sub_hidden))
        self.w_ups = nn.Parameter(torch.empty(self.K, self.sub_d, self.sub_hidden))
        self.w_down = nn.Linear(hidden, d_model, bias=False)

        nn.init.normal_(self.w_gates, mean=0.0, std=0.02)
        nn.init.normal_(self.w_ups, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> tuple[Tensor, dict]:
        B, T, D = x.shape
        x_split = x.view(B, T, self.K, self.sub_d)
        gate = torch.einsum("btkd,kdh->btkh", x_split, self.w_gates)
        up = torch.einsum("btkd,kdh->btkh", x_split, self.w_ups)
        inter = F.silu(gate) * up
        with torch.no_grad():
            per_compartment_l2 = inter.float().pow(2).mean(dim=(0, 1, 3)).sqrt()
        inter_flat = inter.reshape(B, T, self.hidden)
        out = self.w_down(inter_flat)
        aux = {"compartment_l2": per_compartment_l2.detach()}
        return out, aux


class DeltaRuleMemory(nn.Module):

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.nh = cfg.delta_num_heads
        self.hd = cfg.delta_head_dim
        self.inner = self.nh * self.hd

        self.q = nn.Linear(d, self.inner, bias=False)
        self.k = nn.Linear(d, self.inner, bias=False)
        self.v = nn.Linear(d, self.inner, bias=False)
        self.o = nn.Linear(self.inner, d, bias=False)
        self.beta_proj = nn.Linear(d, self.nh, bias=True)

        self.alpha_log = nn.Parameter(torch.empty(self.nh, 1))
        nn.init.normal_(self.alpha_log, mean=cfg.alpha_log_mean, std=cfg.alpha_log_std)

        self.rope = RotaryPE(self.hd)

        self.kwta_enabled = cfg.kwta_enabled
        self.kwta_k = KWTAModule(cfg.kwta_fraction) if cfg.kwta_enabled else None
        self.kwta_v = KWTAModule(cfg.kwta_fraction) if cfg.kwta_enabled else None

        self.delta_erasure = cfg.delta_erasure_enabled
        self.bcm_enabled = cfg.bcm_alpha_enabled
        self.gamma_bcm = cfg.gamma_bcm

        self.register_buffer(
            "running_state_norm",
            torch.ones(self.nh),
            persistent=True,
        )

        self.imagination_enabled = cfg.imagination_enabled
        if cfg.imagination_enabled:
            imag_rank = max(8, min(64, d // 16))
            self.imag_filter_down = nn.Linear(d, imag_rank, bias=False)
            self.imag_filter_up = nn.Linear(imag_rank, d, bias=False)
            self.imag_gate_proj = nn.Linear(d, 1, bias=True)
            nn.init.normal_(self.imag_filter_down.weight, std=0.02)
            nn.init.normal_(self.imag_filter_up.weight, std=0.02)
            nn.init.normal_(self.imag_gate_proj.weight, std=0.02)
            nn.init.constant_(self.imag_gate_proj.bias, cfg.imag_gate_init)
        else:
            self.imag_filter_down = None
            self.imag_filter_up = None
            self.imag_gate_proj = None

        self.pc_diagnostic_enabled = cfg.pc_diagnostic_enabled
        if cfg.pc_diagnostic_enabled:
            pc_rank = max(8, min(64, d // 16))
            self.pc_head_down = nn.Linear(d, pc_rank, bias=False)
            self.pc_head_up = nn.Linear(pc_rank, d, bias=False)
            nn.init.normal_(self.pc_head_down.weight, std=0.02)
            nn.init.normal_(self.pc_head_up.weight, std=0.02)
        else:
            self.pc_head_down = None
            self.pc_head_up = None

        self.use_fla = cfg.use_fla_if_available and FLA_AVAILABLE

    def _effective_log_alpha(self, device: torch.device) -> Tensor:
        alpha_log_base = self.alpha_log.squeeze(-1).to(device)
        if self.bcm_enabled:
            running = self.running_state_norm.to(device).clamp(min=1e-6)
            log_norm = torch.log(running)
            pre_sigmoid = alpha_log_base - self.gamma_bcm * log_norm
        else:
            pre_sigmoid = alpha_log_base
        return F.logsigmoid(pre_sigmoid)

    def _recurrent_with_erasure(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        beta: Tensor,
        state: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        B, T, H, D = q.shape
        if state is None:
            state = torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
        else:
            state = state.float()

        log_alpha_eff = self._effective_log_alpha(state.device)
        alpha_eff = torch.exp(log_alpha_eff).view(1, H, 1, 1)

        q_norm = F.normalize(q.float(), p=2, dim=-1)
        k_norm = F.normalize(k.float(), p=2, dim=-1)
        v_fp = v.float()

        outputs = []
        for t in range(T):
            k_t = k_norm[:, t]
            v_t = v_fp[:, t]
            q_t = q_norm[:, t]
            beta_t = beta[:, t].unsqueeze(-1).unsqueeze(-1).float()

            state_decayed = alpha_eff * state
            if self.delta_erasure:
                kT_state = torch.einsum("bhd,bhde->bhe", k_t, state_decayed)
                v_effective = v_t - kT_state
            else:
                v_effective = v_t

            state = state_decayed + beta_t * torch.einsum(
                "bhd,bhe->bhde", k_t, v_effective
            )
            o_t = torch.einsum("bhd,bhde->bhe", q_t, state)
            outputs.append(o_t.to(q.dtype))

        if self.bcm_enabled and self.training:
            with torch.no_grad():
                final_norm = state.float().norm(dim=(-1, -2)).mean(dim=0).clamp(min=1e-6)
                self.running_state_norm.mul_(0.99).add_(final_norm.detach(), alpha=0.01)

        return torch.stack(outputs, dim=1), state

    def _fla_delta_rule_path(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        beta: Tensor,
        state: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        B, T, H, D = q.shape
        target_dtype = q.dtype
        log_alpha_eff = self._effective_log_alpha(q.device)
        g = log_alpha_eff.view(1, 1, H).expand(B, T, H).contiguous().to(target_dtype)
        beta_cast = beta.to(target_dtype)

        if state is not None and state.dtype != target_dtype:
            state = state.to(target_dtype)

        out, final_state = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta_cast,
            scale=1.0,
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

        if final_state is not None and self.bcm_enabled and self.training:
            with torch.no_grad():
                current_norm = final_state.float().norm(dim=(-1, -2)).mean(dim=0).clamp(min=1e-6)
                self.running_state_norm.mul_(0.99).add_(current_norm.detach(), alpha=0.01)

        return out, final_state

    def _fla_no_erasure_path(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        beta: Tensor,
        state: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        B, T, H, D = q.shape
        target_dtype = q.dtype
        log_alpha_eff = self._effective_log_alpha(q.device)
        g = log_alpha_eff.view(1, 1, H).expand(B, T, H).contiguous().to(target_dtype)

        q_norm = F.normalize(q.float(), p=2, dim=-1).to(target_dtype)
        k_norm = F.normalize(k.float(), p=2, dim=-1).to(target_dtype)
        beta_broadcast = beta.to(target_dtype).unsqueeze(-1)
        v_scaled = (v.to(target_dtype) * beta_broadcast).contiguous()

        if state is not None and state.dtype != target_dtype:
            state = state.to(target_dtype)

        out, final_state = chunk_simple_gla(
            q=q_norm,
            k=k_norm,
            v=v_scaled,
            g=g,
            scale=1.0,
            initial_state=state,
            output_final_state=True,
        )

        if final_state is not None and self.bcm_enabled and self.training:
            with torch.no_grad():
                current_norm = final_state.float().norm(dim=(-1, -2)).mean(dim=0).clamp(min=1e-6)
                self.running_state_norm.mul_(0.99).add_(current_norm.detach(), alpha=0.01)

        return out, final_state

    def forward(
        self,
        x: Tensor,
        state: Any = None,
        offset: int = 0,
    ) -> tuple[Tensor, Any, dict]:
        B, T, _ = x.shape
        aux: dict = {}

        q = self.q(x).view(B, T, self.nh, self.hd)
        k = self.k(x).view(B, T, self.nh, self.hd)
        v = self.v(x).view(B, T, self.nh, self.hd)

        orig_dtype = q.dtype
        cos, sin = self.rope(T, offset, x.device)
        cos_h = cos.unsqueeze(0).unsqueeze(0).to(orig_dtype)
        sin_h = sin.unsqueeze(0).unsqueeze(0).to(orig_dtype)
        q = rotary_apply(q.transpose(1, 2), cos_h, sin_h).transpose(1, 2)
        k = rotary_apply(k.transpose(1, 2), cos_h, sin_h).transpose(1, 2)

        if self.kwta_k is not None:
            k_flat = k.reshape(B, T, self.inner)
            k_flat, k_mask = self.kwta_k(k_flat)
            k = k_flat.view(B, T, self.nh, self.hd)
            aux["kwta_k_rate"] = k_mask.mean().detach()
        if self.kwta_v is not None:
            v_flat = v.reshape(B, T, self.inner)
            v_flat, v_mask = self.kwta_v(v_flat)
            v = v_flat.view(B, T, self.nh, self.hd)
            aux["kwta_v_rate"] = v_mask.mean().detach()

        alpha_base = torch.sigmoid(self.alpha_log)
        beta = torch.sigmoid(self.beta_proj(x))

        with torch.no_grad():
            alpha_eff_scalar = torch.exp(self._effective_log_alpha(x.device)).mean()
        aux["alpha_base_mean"] = alpha_base.mean().detach()
        aux["alpha_eff_mean"] = alpha_eff_scalar.detach()
        aux["beta_mean"] = beta.mean().detach()

        fla_available_base = (
            self.use_fla
            and q.is_cuda
            and state is None
            and T > 1
        )
        can_use_fla_erasure = (
            fla_available_base
            and chunk_gated_delta_rule is not None
            and self.delta_erasure
        )
        can_use_fla_no_erasure = (
            fla_available_base
            and chunk_simple_gla is not None
            and not self.delta_erasure
        )

        if can_use_fla_erasure:
            out_thd, new_state = self._fla_delta_rule_path(q, k, v, beta, state)
            aux["delta_path"] = torch.tensor(0.0, device=x.device)
        elif can_use_fla_no_erasure:
            out_thd, new_state = self._fla_no_erasure_path(q, k, v, beta, state)
            aux["delta_path"] = torch.tensor(0.0, device=x.device)
        else:
            out_thd, new_state = self._recurrent_with_erasure(q, k, v, beta, state)
            aux["delta_path"] = torch.tensor(1.0, device=x.device)
        aux["delta_erasure_flag"] = torch.tensor(
            1.0 if self.delta_erasure else 0.0, device=x.device
        )

        out_flat = out_thd.reshape(B, T, self.inner)
        out = self.o(out_flat)

        if new_state is not None:
            aux["state_frobenius"] = new_state.float().norm(dim=(-1, -2)).mean().detach()

        if self.pc_head_down is not None:
            x_pred = self.pc_head_up(self.pc_head_down(out.detach()))
            pc_error = F.mse_loss(x_pred, x.detach())
            aux["pc_error_l2"] = pc_error

        if self.imag_filter_down is not None:
            with torch.no_grad():
                pre_imag_norm = out.norm(dim=-1).mean().clamp(min=1e-9)
            imag_value = self.imag_filter_up(self.imag_filter_down(out))
            gate_logit = self.imag_gate_proj(x)
            gate_val = torch.sigmoid(gate_logit)
            imag_contribution = gate_val * imag_value
            out = out + imag_contribution

            with torch.no_grad():
                imag_norm = imag_contribution.norm(dim=-1).mean()
                aux["imag_gate_mean"] = gate_val.mean().detach()
                aux["imag_contribution_l2"] = imag_norm.detach()
                aux["imag_ratio"] = (imag_norm / pre_imag_norm).detach()

        return out, new_state, aux


class SlotMemory(nn.Module):

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.nh = cfg.delta_num_heads
        self.hd = cfg.delta_head_dim
        self.num_slots = int(cfg.slot_num_slots)
        self.surprise_tau = float(cfg.slot_surprise_tau)
        self.inner = self.nh * self.hd

        self.q = nn.Linear(d, self.inner, bias=False)
        self.k = nn.Linear(d, self.inner, bias=False)
        self.v = nn.Linear(d, self.inner, bias=False)
        self.o = nn.Linear(self.inner, d, bias=False)

        self.rope = RotaryPE(self.hd)

        self.log_temperature = nn.Parameter(
            torch.full((self.nh,), float(cfg.slot_log_temperature_init))
        )
        self.output_gate_logit = nn.Parameter(
            torch.full((self.nh,), float(cfg.slot_gate_init))
        )

        self.prototype_keys = nn.Parameter(torch.randn(self.nh, self.num_slots, self.hd) * 0.02)

        self.predict_rank = max(1, int(cfg.slot_predict_rank))
        self.predict_down = nn.Linear(d, self.predict_rank, bias=False)
        self.predict_up = nn.Linear(self.predict_rank, d, bias=False)
        nn.init.normal_(self.predict_down.weight, std=0.02)
        nn.init.normal_(self.predict_up.weight, std=0.02)

        self.alpha_log = nn.Parameter(
            torch.full((self.nh,), float(cfg.alpha_log_mean))
        )

    def _empty_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> dict[str, Tensor]:
        state_dtype = torch.float32 if dtype == torch.float32 else dtype
        return {
            "slot_state": torch.zeros(batch, self.nh, self.num_slots, self.hd, device=device, dtype=state_dtype),
            "prev_hidden": torch.zeros(batch, self.cfg.d_model, device=device, dtype=dtype),
        }

    def forward(
        self,
        x: Tensor,
        state: Any = None,
        offset: int = 0,
    ) -> tuple[Tensor, Any, dict]:
        B, T, _ = x.shape
        aux: dict = {}

        if state is None:
            state = self._empty_state(B, x.device, x.dtype)
        slot_state = state["slot_state"]
        prev_hidden = state["prev_hidden"]

        q = self.q(x).view(B, T, self.nh, self.hd)
        k = self.k(x).view(B, T, self.nh, self.hd)
        v = self.v(x).view(B, T, self.nh, self.hd)

        cos, sin = self.rope(T, offset, x.device)
        cos_h = cos.unsqueeze(0).unsqueeze(0).to(x.dtype)
        sin_h = sin.unsqueeze(0).unsqueeze(0).to(x.dtype)
        q = rotary_apply(q.transpose(1, 2), cos_h, sin_h).transpose(1, 2)
        k = rotary_apply(k.transpose(1, 2), cos_h, sin_h).transpose(1, 2)

        temperature_per_head = torch.exp(self.log_temperature).clamp(min=1e-3)
        gate = torch.sigmoid(self.output_gate_logit).view(1, 1, self.nh, 1)
        log_alpha_per_head = F.logsigmoid(self.alpha_log)

        shifted_prev = torch.cat([prev_hidden.unsqueeze(1), x[:, :-1, :]], dim=1)
        predicted_hidden = self.predict_up(self.predict_down(shifted_prev))
        prediction_loss = (predicted_hidden - x).float().pow(2).mean()
        residual_sq = (x - predicted_hidden).float().pow(2).sum(dim=-1)
        hidden_sq = x.float().pow(2).sum(dim=-1).clamp(min=1e-6)
        surprise_per_token = (residual_sq / hidden_sq).clamp(min=0.0, max=1.0)

        k_logits = torch.einsum("bthd,hnd->bthn", k, self.prototype_keys) / temperature_per_head.view(1, 1, self.nh, 1)
        q_logits = torch.einsum("bthd,hnd->bthn", q, self.prototype_keys) / temperature_per_head.view(1, 1, self.nh, 1)
        w_write = F.softmax(k_logits, dim=-1)
        w_read = F.softmax(q_logits, dim=-1)
        w_write = w_write * surprise_per_token.unsqueeze(-1).unsqueeze(-1).to(w_write.dtype)

        g_per_step = log_alpha_per_head.view(1, 1, self.nh).expand(B, T, self.nh).to(x.dtype).contiguous()

        fla_fn = fused_chunk_simple_gla if fused_chunk_simple_gla is not None else chunk_simple_gla
        can_use_fla = (
            fla_fn is not None
            and q.is_cuda
            and (slot_state.abs().sum() == 0)
            and T > 1
        )

        if can_use_fla:
            read_out, final_state = fla_fn(
                q=w_read.contiguous(),
                k=w_write.contiguous(),
                v=v.contiguous(),
                g=g_per_step,
                scale=1.0,
                initial_state=None,
                output_final_state=True,
            )
        else:
            read_out, final_state = self._recurrent_slot_update(
                w_read, w_write, v, g_per_step, slot_state
            )

        gated = read_out * gate
        out = self.o(gated.reshape(B, T, self.inner))

        new_state = {
            "slot_state": final_state.detach() if not self.training else final_state,
            "prev_hidden": x[:, -1, :].detach(),
        }

        aux["slot_write_fraction"] = surprise_per_token.mean().detach()
        aux["slot_gate_mean"] = gate.mean().detach()
        aux["slot_log_temperature_mean"] = self.log_temperature.mean().detach()
        aux["slot_alpha_mean"] = torch.exp(log_alpha_per_head).mean().detach()
        aux["slot_allocation_entropy"] = (-w_write * (w_write.clamp(min=1e-9)).log()).sum(dim=-1).mean().detach()
        aux["slot_prediction_loss"] = prediction_loss
        aux["delta_path"] = torch.tensor(2.0 if can_use_fla else 3.0, device=x.device)
        aux["delta_erasure_flag"] = torch.tensor(0.0, device=x.device)
        aux["alpha_base_mean"] = torch.sigmoid(self.alpha_log).mean().detach()
        aux["alpha_eff_mean"] = torch.exp(log_alpha_per_head).mean().detach()
        aux["beta_mean"] = torch.zeros((), device=x.device)
        aux["state_frobenius"] = final_state.float().norm(dim=(-1, -2)).mean().detach()

        return out, new_state, aux

    def _recurrent_slot_update(
        self,
        w_read: Tensor,
        w_write: Tensor,
        v: Tensor,
        g_per_step: Tensor,
        initial_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        B, T, H, N = w_write.shape
        D = v.shape[-1]
        state = initial_state.to(v.dtype) if initial_state.numel() > 0 else torch.zeros(B, H, N, D, device=v.device, dtype=v.dtype)
        outputs: list[Tensor] = []
        for t in range(T):
            alpha_t = torch.exp(g_per_step[:, t]).view(B, H, 1, 1)
            k_t = w_write[:, t].unsqueeze(-1)
            v_t = v[:, t].unsqueeze(-2)
            state = state * alpha_t + k_t * v_t
            q_t = w_read[:, t].unsqueeze(-1)
            out_t = (q_t * state).sum(dim=-2)
            outputs.append(out_t.unsqueeze(1))
        out = torch.cat(outputs, dim=1)
        return out, state


class CompressedAttention(nn.Module):

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        d = cfg.d_model
        self.d_c = cfg.attn_d_c
        self.d_R = cfg.attn_d_R
        self.nh = cfg.attn_num_heads
        self.hd = d // self.nh
        if self.d_R % 2 != 0:
            raise ValueError(f"attn_d_R must be even, got {self.d_R}")

        self.kv_down = nn.Linear(d, self.d_c, bias=False)
        self.k_up = nn.Linear(self.d_c, d, bias=False)
        self.v_up = nn.Linear(self.d_c, d, bias=False)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.q_rope_proj = nn.Linear(d, self.d_R, bias=False)
        self.k_rope_proj = nn.Linear(self.d_c, self.d_R, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

        self.rope = RotaryPE(self.d_R)

    def forward(
        self,
        x: Tensor,
        state: Any = None,
        offset: int = 0,
    ) -> tuple[Tensor, Tensor, dict]:
        B, T, _ = x.shape
        aux: dict = {}

        orig_dtype = x.dtype
        c_kv = self.kv_down(x)
        k_rope_shared = self.k_rope_proj(c_kv)
        cos, sin = self.rope(T, offset, x.device)
        cos_b = cos.unsqueeze(0).to(orig_dtype)
        sin_b = sin.unsqueeze(0).to(orig_dtype)
        k_rope_shared = rotary_apply(k_rope_shared, cos_b, sin_b)

        cache_entry = torch.cat([c_kv, k_rope_shared], dim=-1)

        if state is not None:
            cache_entry = torch.cat([state, cache_entry], dim=1)

        cached_c_kv = cache_entry[..., : self.d_c]
        cached_k_rope = cache_entry[..., self.d_c :]
        total_len = cache_entry.shape[1]

        k_content = self.k_up(cached_c_kv).view(B, total_len, self.nh, self.hd)
        v = self.v_up(cached_c_kv).view(B, total_len, self.nh, self.hd)

        q_content = self.q_proj(x).view(B, T, self.nh, self.hd)
        q_rope_shared = self.q_rope_proj(x)
        q_rope_shared = rotary_apply(q_rope_shared, cos_b, sin_b)

        q_rope_per_head = q_rope_shared.unsqueeze(2).expand(-1, -1, self.nh, -1)
        k_rope_per_head = cached_k_rope.unsqueeze(2).expand(-1, -1, self.nh, -1)

        q_combined = torch.cat([q_content, q_rope_per_head], dim=-1)
        k_combined = torch.cat([k_content, k_rope_per_head], dim=-1)

        q_combined = q_combined.transpose(1, 2)
        k_combined = k_combined.transpose(1, 2)
        v_t = v.transpose(1, 2)

        is_causal_flag = (T > 1) and (total_len == T)
        attn_mask_arg: Tensor | None = None
        if T > 1 and not is_causal_flag:
            causal_mask = torch.zeros(
                (T, total_len), device=x.device, dtype=torch.bool
            )
            off = total_len - T
            for i in range(T):
                causal_mask[i, : off + i + 1] = True
            attn_mask_arg = causal_mask.unsqueeze(0).unsqueeze(0)

        scale = 1.0 / math.sqrt(self.hd + self.d_R)
        if is_causal_flag:
            attn_output = F.scaled_dot_product_attention(
                q_combined,
                k_combined,
                v_t,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=scale,
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                q_combined,
                k_combined,
                v_t,
                attn_mask=attn_mask_arg,
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
            )

        attn_output = attn_output.transpose(1, 2).reshape(B, T, -1)
        out = self.o_proj(attn_output)

        aux["cache_size_per_token"] = self.d_c + self.d_R
        aux["total_cached_tokens"] = total_len

        return out, cache_entry, aux


class Block(nn.Module):

    def __init__(self, layer_type: str, cfg: Config) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.mixer_norm = RMSNorm(cfg.d_model)
        self.mlp_norm = RMSNorm(cfg.d_model)
        if layer_type == "DELTA":
            self.mixer = DeltaRuleMemory(cfg)
        elif layer_type == "ATTN":
            self.mixer = CompressedAttention(cfg)
        elif layer_type == "SLOT":
            self.mixer = SlotMemory(cfg)
        else:
            raise ValueError(f"unknown layer type: {layer_type}")
        if cfg.multi_compartment_enabled:
            self.mlp = MultiCompartmentSwiGLU(
                cfg.d_model, cfg.mlp_ratio, cfg.num_compartments
            )
        else:
            self.mlp = SwiGLU(cfg.d_model, cfg.mlp_ratio)

    def forward(
        self,
        x: Tensor,
        state: Any = None,
        offset: int = 0,
    ) -> tuple[Tensor, Any, dict]:
        residual = x
        normed = self.mixer_norm(x)
        mixer_out, new_state, aux = self.mixer(normed, state=state, offset=offset)
        x = residual + mixer_out

        residual = x
        mlp_out, mlp_aux = self.mlp(self.mlp_norm(x))
        x = residual + mlp_out

        for k, v in mlp_aux.items():
            aux[f"mlp_{k}"] = v

        return x, new_state, aux


class GodMachine(nn.Module):

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(lt, cfg) for lt in cfg.layer_types])
        self.final_norm = RMSNorm(cfg.d_model)
        self.output_proj = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.output_proj.weight = self.embedding.weight

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self._init_linears()
        self._restore_custom_inits()
        self._gc_enabled = False

    def _init_linears(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.weight is self.embedding.weight:
                    continue
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _restore_custom_inits(self) -> None:
        for block in self.blocks:
            if isinstance(block.mixer, DeltaRuleMemory):
                if block.mixer.imag_gate_proj is not None:
                    nn.init.constant_(
                        block.mixer.imag_gate_proj.bias, self.cfg.imag_gate_init
                    )
                nn.init.normal_(
                    block.mixer.alpha_log,
                    mean=self.cfg.alpha_log_mean,
                    std=self.cfg.alpha_log_std,
                )

    def enable_gradient_checkpointing(self) -> None:
        self._gc_enabled = True

    def forward(
        self,
        input_ids: Tensor,
        states: list[Any] | None = None,
        offset: int = 0,
    ) -> tuple[Tensor, list[Any], dict]:
        x = self.embedding(input_ids)

        if states is None:
            states = [None] * len(self.blocks)

        new_states: list[Any] = []
        all_aux: dict[int, dict] = {}

        for i, block in enumerate(self.blocks):
            if self._gc_enabled and self.training and states[i] is None:
                x, new_state, aux = checkpoint(
                    block,
                    x,
                    states[i],
                    offset,
                    use_reentrant=False,
                )
            else:
                x, new_state, aux = block(x, state=states[i], offset=offset)
            new_states.append(new_state)
            all_aux[i] = aux

        x = self.final_norm(x)
        logits = self.output_proj(x)
        return logits, new_states, all_aux

    def count_params(self) -> int:
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total


class ByteDataset(Dataset):

    def __init__(self, data: Any, seq_len: int) -> None:
        if isinstance(data, torch.Tensor):
            self.data = data.to(torch.uint8) if data.dtype != torch.uint8 else data
            self._buf = None
        elif isinstance(data, np.ndarray):
            self.data = torch.from_numpy(np.ascontiguousarray(data, dtype=np.uint8))
            self._buf = None
        else:
            self._buf = bytearray(data)
            self.data = torch.frombuffer(self._buf, dtype=torch.uint8)
        self.seq_len = seq_len
        self._n = max(0, (len(self.data) - 1) // self.seq_len)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> Tensor:
        if idx >= self._n:
            raise IndexError(idx)
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        return self.data[start:end].long()


def collate_fn(batch: list[Tensor]) -> Tensor:
    return torch.stack([b for b in batch], dim=0)


def download_fineweb_edu(
    max_bytes: int = 5_000_000_000,
    allow_wikitext_fallback: bool = False,
) -> tuple[Any, Any, str]:
    data_dir = _data_root() / "fineweb"
    if (data_dir / "train.bin").exists() and (data_dir / "val.bin").exists():
        log(f"loading cached fineweb-edu from {data_dir}")
        train_np = np.memmap(data_dir / "train.bin", dtype=np.uint8, mode="r")
        val_np = np.memmap(data_dir / "val.bin", dtype=np.uint8, mode="r")
        return train_np, val_np, "fineweb-edu-cached"
    try:
        from datasets import load_dataset
        log(f"streaming fineweb-edu (target {max_bytes / 1e9:.1f}gb)")
        ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
        buf = bytearray()
        last_report = 0
        for item in ds:
            buf.extend(item["text"].encode("utf-8", errors="replace"))
            if len(buf) - last_report >= 500_000_000:
                log(f"  {len(buf) / 1e9:.1f}gb")
                last_report = len(buf)
            if len(buf) >= max_bytes:
                break
        val_size = min(len(buf) // 20, 50_000_000)
        val_data = bytes(buf[-val_size:])
        train_data = bytes(buf[:-val_size])
        data_dir.mkdir(parents=True, exist_ok=True)
        with open(data_dir / "train.bin", "wb") as f:
            f.write(train_data)
        with open(data_dir / "val.bin", "wb") as f:
            f.write(val_data)
        return train_data, val_data, "fineweb-edu"
    except Exception as e:
        if not allow_wikitext_fallback:
            raise RuntimeError(
                f"fineweb-edu failed to load ({e}) and ALLOW_WIKITEXT_FALLBACK is not set. "
                "refusing to fall open onto wikitext-2 for a run that should be comparable to fineweb-backed baselines. "
                "set ALLOW_WIKITEXT_FALLBACK=1 only for an explicitly named fallback/debug run."
            ) from e
        log(f"fineweb-edu failed: {e}, falling back to wikitext-2 (ALLOW_WIKITEXT_FALLBACK=1)")
        train, val, src = download_wikitext2()
        return train, val, f"wikitext-2-fallback-from-fineweb ({src})"


def download_wikitext2() -> tuple[bytes, bytes, str]:
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_text = "\n".join(ds["train"]["text"])
        val_text = "\n".join(ds["validation"]["text"])
        return train_text.encode("utf-8"), val_text.encode("utf-8"), "wikitext-2"
    except Exception as e:
        log(f"wikitext-2 failed: {e}")
        if not _env_flag("ALLOW_SYNTHETIC"):
            raise RuntimeError(
                f"wikitext-2 failed to load ({e}) and ALLOW_SYNTHETIC is not set. "
                "refusing to train on synthetic random bytes, which would produce a "
                "fake run indistinguishable from real data. set ALLOW_SYNTHETIC=1 "
                "only for debugging pipeline plumbing."
            )
        log("warning: using synthetic random-byte data (ALLOW_SYNTHETIC=1)")
        rng = np.random.RandomState(42)
        train_data = bytes(rng.randint(0, 256, size=2_000_000).tolist())
        val_data = bytes(rng.randint(0, 256, size=200_000).tolist())
        return train_data, val_data, "synthetic-random-bytes"


@torch.no_grad()
def run_validation(
    model: GodMachine,
    dataloader: DataLoader,
    max_batches: int = 50,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
) -> dict:
    was_training = model.training
    model.train(False)
    total_loss = 0.0
    total_tokens = 0
    device = _model_device(model)
    precision = str(amp_dtype).replace("torch.", "") if use_amp else "float32"
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        batch = batch.to(device)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits, _, _ = model(batch[:, :-1])
            target = batch[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target.reshape(-1),
                reduction="sum",
            )
        total_loss += loss.item()
        total_tokens += target.numel()
    model.train(was_training)
    if total_tokens == 0:
        return {"mean_loss": float("inf"), "bpb": float("inf"), "tokens": 0, "precision": precision}
    mean_loss = total_loss / total_tokens
    return {
        "mean_loss": mean_loss,
        "bpb": mean_loss / math.log(2),
        "tokens": total_tokens,
        "precision": precision,
    }


def collect_god_metrics(aux: dict) -> dict:
    skip_aggregation = {"delta_path", "total_cached_tokens", "cache_size_per_token"}

    layer_keys = sorted(k for k in aux.keys() if isinstance(k, int)) if isinstance(aux, dict) else []
    layer_indices: list[int] = [int(li) for li in layer_keys if isinstance(aux.get(li), dict)]
    n_layers = len(layer_indices)

    scalar_keys_seen: set[str] = set()
    vector_keys_seen: set[str] = set()
    for li in layer_indices:
        la = aux[li]
        for key, val in la.items():
            if isinstance(val, Tensor):
                if val.numel() == 1:
                    scalar_keys_seen.add(key)
                elif val.numel() <= 64:
                    vector_keys_seen.add(key)
            elif isinstance(val, (int, float)):
                scalar_keys_seen.add(key)

    scalar_by_key: dict[str, list[float | None]] = {
        k: [None] * n_layers for k in scalar_keys_seen
    }
    vector_by_key: dict[str, list[list[float] | None]] = {
        k: [None] * n_layers for k in vector_keys_seen
    }

    for pos, li in enumerate(layer_indices):
        la = aux[li]
        for key, val in la.items():
            if isinstance(val, Tensor):
                if val.numel() == 1:
                    scalar_by_key[key][pos] = float(val.item())
                elif val.numel() <= 64:
                    vector_by_key[key][pos] = val.detach().float().cpu().tolist()
            elif isinstance(val, (int, float)):
                scalar_by_key[key][pos] = float(val)

    result: dict[str, Any] = {"layer_indices": layer_indices}
    for key, per_layer in scalar_by_key.items():
        result[f"{key}_per_layer"] = per_layer
        if key in skip_aggregation:
            continue
        present = [x for x in per_layer if x is not None]
        if present:
            result[f"{key}_mean"] = float(np.mean(present))
            result[f"{key}_std"] = float(np.std(present))
            result[f"{key}_min"] = float(np.min(present))
            result[f"{key}_max"] = float(np.max(present))
    for key, per_layer in vector_by_key.items():
        result[f"{key}_per_layer"] = per_layer

    kwta_k = [x for x in scalar_by_key.get("kwta_k_rate", []) if x is not None]
    kwta_v = [x for x in scalar_by_key.get("kwta_v_rate", []) if x is not None]
    combined_rates = kwta_k + kwta_v
    result["mean_fr"] = float(np.mean(combined_rates)) if combined_rates else 0.0
    per_layer_mean_fr: list[float | None] = []
    pk = scalar_by_key.get("kwta_k_rate", [None] * n_layers)
    pv = scalar_by_key.get("kwta_v_rate", [None] * n_layers)
    for i in range(n_layers):
        a = pk[i] if i < len(pk) else None
        b = pv[i] if i < len(pv) else None
        if a is None and b is None:
            per_layer_mean_fr.append(None)
        elif a is None:
            per_layer_mean_fr.append(b)
        elif b is None:
            per_layer_mean_fr.append(a)
        else:
            per_layer_mean_fr.append((a + b) / 2.0)
    result["per_layer_mean_fr"] = per_layer_mean_fr

    return result


collect_spike_stats = collect_god_metrics


def aggregate_pc_loss(aux: dict, device: torch.device | None = None) -> Tensor:
    total = None
    count = 0
    for layer_aux in aux.values():
        if not isinstance(layer_aux, dict):
            continue
        pc = layer_aux.get("pc_error_l2")
        if isinstance(pc, Tensor) and pc.requires_grad:
            total = pc if total is None else (total + pc)
            count += 1
    if total is None or count == 0:
        return torch.zeros((), device=device if device is not None else torch.device("cpu"))
    return total / float(count)


def aggregate_slot_prediction_loss(aux: dict, device: torch.device | None = None) -> Tensor:
    total = None
    count = 0
    for layer_aux in aux.values():
        if not isinstance(layer_aux, dict):
            continue
        sp = layer_aux.get("slot_prediction_loss")
        if isinstance(sp, Tensor) and sp.requires_grad:
            total = sp if total is None else (total + sp)
            count += 1
    if total is None or count == 0:
        return torch.zeros((), device=device if device is not None else torch.device("cpu"))
    return total / float(count)


def probe_feature_assertions(aux: dict, cfg: Config) -> None:
    inner = cfg.delta_num_heads * cfg.delta_head_dim
    k_exact = max(1, int(round(inner * cfg.kwta_fraction)))
    expected_rate = k_exact / inner
    for layer_idx, layer_aux in aux.items():
        if not isinstance(layer_aux, dict):
            continue
        if cfg.kwta_enabled and "kwta_k_rate" in layer_aux:
            rate = float(layer_aux["kwta_k_rate"].item() if isinstance(layer_aux["kwta_k_rate"], Tensor) else layer_aux["kwta_k_rate"])
            if abs(rate - expected_rate) > 1e-4:
                raise AssertionError(
                    f"layer {layer_idx} kwta_k_rate {rate:.6f} deviates from exact {expected_rate:.6f} (k={k_exact}/{inner})"
                )
        if cfg.bcm_alpha_enabled and "alpha_base_mean" in layer_aux:
            alpha_val = float(layer_aux["alpha_base_mean"].item() if isinstance(layer_aux["alpha_base_mean"], Tensor) else layer_aux["alpha_base_mean"])
            if not (0.0 < alpha_val < 1.0):
                raise AssertionError(f"layer {layer_idx} alpha_base_mean out of range: {alpha_val}")
        if cfg.imagination_enabled and "imag_gate_mean" in layer_aux:
            gate_val = float(layer_aux["imag_gate_mean"].item() if isinstance(layer_aux["imag_gate_mean"], Tensor) else layer_aux["imag_gate_mean"])
            if not (0.0 <= gate_val <= 1.0):
                raise AssertionError(f"layer {layer_idx} imag_gate_mean out of range: {gate_val}")
        if cfg.pc_diagnostic_enabled and "pc_error_l2" in layer_aux:
            pc_val = float(layer_aux["pc_error_l2"].item() if isinstance(layer_aux["pc_error_l2"], Tensor) else layer_aux["pc_error_l2"])
            if pc_val < 0 or not math.isfinite(pc_val):
                raise AssertionError(f"layer {layer_idx} pc_error_l2 invalid: {pc_val}")
        if cfg.multi_compartment_enabled and "mlp_compartment_l2" in layer_aux:
            comp = layer_aux["mlp_compartment_l2"]
            if isinstance(comp, Tensor):
                if comp.numel() != cfg.num_compartments:
                    raise AssertionError(
                        f"layer {layer_idx} mlp_compartment_l2 has {comp.numel()} entries, "
                        f"expected {cfg.num_compartments}"
                    )
            else:
                raise AssertionError(
                    f"layer {layer_idx} mlp_compartment_l2 is not a Tensor: type={type(comp).__name__}"
                )


def _build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim <= 1 or "norm" in name or "bias" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _cosine_lr(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def _capture_env_metadata() -> dict:
    env: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "python_full": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        try:
            env["cuda_version"] = torch.version.cuda
            env["cudnn_version"] = torch.backends.cudnn.version()
            env["gpu_name"] = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            env["gpu_capability"] = f"sm_{cap[0]}{cap[1]}"
            props = torch.cuda.get_device_properties(0)
            env["gpu_total_memory_gb"] = props.total_memory / 1e9
            env["gpu_multi_processor_count"] = props.multi_processor_count
        except Exception as e:
            env["gpu_info_error"] = str(e)
    return env


def _capture_git_metadata() -> dict:
    result: dict[str, Any] = {"sha": None, "dirty": None, "error": None}
    try:
        sha_proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if sha_proc.returncode == 0:
            result["sha"] = sha_proc.stdout.strip()
        else:
            result["error"] = sha_proc.stderr.strip()
            return result
        dirty_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if dirty_proc.returncode == 0:
            result["dirty"] = bool(dirty_proc.stdout.strip())
    except Exception as e:
        result["error"] = str(e)
    return result


def _capture_git_state_fingerprint(ignore_paths: set[Path] | None = None) -> dict[str, Any]:
    git = _capture_git_metadata()
    result: dict[str, Any] = {
        "sha": git.get("sha"),
        "dirty": git.get("dirty"),
        "fingerprint": None,
        "error": git.get("error"),
    }
    if result["error"] is not None or result["sha"] is None or result["dirty"] is None:
        if result["error"] is None:
            result["error"] = "git metadata unavailable"
        return result
    try:
        repo_root = REPO_ROOT.resolve()
        ignored: set[str] = set()
        for path in ignore_paths or set():
            try:
                ignored.add(path.resolve().relative_to(repo_root).as_posix())
            except ValueError:
                ignored.add(path.as_posix())
        status_proc = subprocess.run(
            ["git", "status", "--porcelain=v1", "-uall"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if status_proc.returncode != 0:
            result["error"] = status_proc.stderr.strip() or "git status failed"
            return result
        diff_proc = subprocess.run(
            ["git", "diff", "HEAD", "--binary", "--no-ext-diff", "--submodule=diff"],
            cwd=repo_root,
            capture_output=True,
            timeout=10,
            check=False,
        )
        if diff_proc.returncode != 0:
            stderr = diff_proc.stderr.decode("utf-8", errors="replace") if diff_proc.stderr else ""
            result["error"] = stderr.strip() or "git diff failed"
            return result
        status_lines: list[str] = []
        untracked_entries: list[dict[str, Any]] = []
        for raw_line in status_proc.stdout.splitlines():
            if len(raw_line) < 4:
                continue
            rel_path = Path(raw_line[3:]).as_posix()
            if any(rel_path == ignored_path or rel_path.startswith(f"{ignored_path}/") for ignored_path in ignored):
                continue
            status_lines.append(raw_line)
            if raw_line.startswith("?? "):
                untracked_path = repo_root / raw_line[3:]
                entry: dict[str, Any] = {"path": rel_path}
                if untracked_path.is_file():
                    entry["sha256"] = _sha256_file(untracked_path)
                untracked_entries.append(entry)
        payload = {
            "sha": result["sha"],
            "dirty": result["dirty"],
            "status_lines": status_lines,
            "diff_hash": hashlib.sha256(diff_proc.stdout).hexdigest(),
            "untracked": untracked_entries,
        }
        result["fingerprint"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return result
    except Exception as e:
        result["error"] = str(e)
        return result


def _cfg_to_dict(cfg: Config) -> dict:
    d = asdict(cfg)
    for k, v in list(d.items()):
        if isinstance(v, tuple):
            d[k] = list(v)
    return d


def _config_hash(cfg_dict: dict) -> str:
    s = json.dumps(cfg_dict, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


class MetricsLogger:

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "a", encoding="utf-8", buffering=1)
        self._closed = False

    def log(self, record: dict) -> None:
        if self._closed:
            return
        try:
            self._f.write(json.dumps(_json_safe(record), default=str, allow_nan=False) + "\n")
            self._f.flush()
        except Exception as e:
            sys.stderr.write(f"metrics log write failed: {e}\n")

    def fsync(self) -> None:
        if self._closed:
            return
        try:
            self._f.flush()
            os.fsync(self._f.fileno())
        except Exception:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._f.flush()
            os.fsync(self._f.fileno())
            self._f.close()
        except Exception:
            pass


def _atomic_tmp_path(path: Path) -> Path:
    return path.parent / (path.name + ".tmp")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(
        f"invalid boolean env var {name}={raw!r}; expected one of 1/0, true/false, yes/no, on/off"
    )


def _apply_runtime_env_overrides(cfg: Config) -> list[str]:
    applied: list[str] = []
    int_overrides = {
        "NM_BATCH_SIZE": "batch_size",
        "NM_SEQ_LEN": "seq_len",
        "NM_MAX_STEPS": "max_steps",
        "NM_VAL_INTERVAL": "val_interval",
        "NM_WARMUP_STEPS": "warmup_steps",
        "NM_MAX_SEQ_LEN": "max_seq_len",
    }
    for env_name, attr in int_overrides.items():
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        value = int(raw)
        setattr(cfg, attr, value)
        applied.append(f"{attr}={value}")
    if cfg.max_seq_len < cfg.seq_len:
        cfg.max_seq_len = cfg.seq_len
        applied.append(f"max_seq_len={cfg.max_seq_len}")
    return applied


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _write_json_atomic(path: Path, payload: Any) -> None:
    tmp = _atomic_tmp_path(path)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_json_safe(payload), f, indent=2, default=str, allow_nan=False)
        tmp.replace(path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _device_metadata() -> dict[str, Any]:
    info: dict[str, Any] = {"device": DEVICE.type}
    if DEVICE.type == "cuda":
        try:
            cap = torch.cuda.get_device_capability()
            info["device_capability"] = [int(cap[0]), int(cap[1])]
        except Exception as e:
            info["device_capability_error"] = str(e)
        try:
            info["device_name"] = torch.cuda.get_device_name()
        except Exception as e:
            info["device_name_error"] = str(e)
        try:
            props = torch.cuda.get_device_properties(0)
            info["device_total_memory_gb"] = round(props.total_memory / 1e9, 3)
            info["device_multi_processor_count"] = int(props.multi_processor_count)
        except Exception as e:
            info["device_properties_error"] = str(e)
    return info


def _device_capability_tuple(value: Any) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return (0, 0)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _validate_run_name(name: str) -> str:
    candidate = name.strip()
    path_obj = Path(candidate)
    if not candidate or path_obj.name != candidate or path_obj.drive or path_obj.root:
        raise RuntimeError(
            f"invalid run name {name!r}. use a plain filename stem without path separators or absolute components."
        )
    return candidate


def _same_path(a: Path, b: Path) -> bool:
    return os.path.normcase(os.path.normpath(os.fspath(a))) == os.path.normcase(os.path.normpath(os.fspath(b)))


def _require_lexical_path(path: Path, label: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    resolved = path.resolve(strict=False)
    if not _same_path(lexical, resolved):
        raise RuntimeError(
            f"{label} {lexical} resolves to {resolved}; symlink and junction path aliases are forbidden on the official run1 surface."
        )
    return lexical


def _require_path_under(root: Path, path: Path, label: str) -> Path:
    root_resolved = root.resolve(strict=False)
    path_resolved = path.resolve(strict=False)
    try:
        path_resolved.relative_to(root_resolved)
    except ValueError as e:
        raise RuntimeError(f"{label} {path_resolved} escapes {root_resolved}") from e
    return path_resolved


def save_resume_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    best_val_bpb: float,
    history: dict,
    cfg_dict: dict,
    train_gen_state: Tensor | None = None,
    step_in_epoch: int = 0,
    epoch_idx: int = 0,
    launch_contract: dict[str, Any] | None = None,
) -> None:
    state: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "step_in_epoch": step_in_epoch,
        "epoch_idx": epoch_idx,
        "best_val_bpb": best_val_bpb,
        "history": history,
        "config": cfg_dict,
        "rng_state": torch.random.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": py_random.getstate(),
        "saved_at": datetime.now().isoformat(),
    }
    if launch_contract is not None:
        state["launch_contract"] = launch_contract
    if train_gen_state is not None:
        state["train_gen_state"] = train_gen_state
    if torch.cuda.is_available():
        try:
            state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    tmp = _atomic_tmp_path(path)
    torch.save(state, tmp)
    tmp.replace(path)


def load_resume_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    torch.random.set_rng_state(ckpt["rng_state"])
    np.random.set_state(ckpt["numpy_rng_state"])
    py_random.setstate(ckpt["python_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in ckpt:
        try:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state_all"])
        except Exception:
            pass
    return {
        "step": ckpt["step"],
        "step_in_epoch": ckpt.get("step_in_epoch", 0),
        "epoch_idx": ckpt.get("epoch_idx", 0),
        "best_val_bpb": ckpt["best_val_bpb"],
        "history": ckpt.get("history", {}),
        "train_gen_state": ckpt.get("train_gen_state"),
        "config": ckpt.get("config", {}),
        "launch_contract": ckpt.get("launch_contract"),
    }


def _write_results_snapshot(
    path: Path,
    name: str,
    n_params: int,
    best_val_bpb: float,
    current_step: int,
    elapsed: float,
    tokens_seen: int,
    cfg_dict: dict,
    metadata: dict,
    history: dict,
    final: bool = False,
) -> None:
    snapshot = {
        "run": name,
        "param_count": n_params,
        "best_val_bpb": best_val_bpb,
        "current_step": current_step,
        "elapsed_seconds": elapsed,
        "tokens_seen_total": tokens_seen,
        "config": cfg_dict,
        "metadata": metadata,
        "history": history,
        "final": final,
        "updated_at": datetime.now().isoformat(),
    }
    _write_json_atomic(path, snapshot)


def train_model(
    cfg: Config,
    train_data: Any,
    val_data: Any,
    name: str,
    output_dir: Path,
    dataset_source: str = "unknown",
    stop_after_steps: int | None = None,
    launch_contract: dict[str, Any] | None = None,
) -> dict:
    name = _validate_run_name(name)
    log(f"=== training {name} ===")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    py_random.seed(cfg.seed)
    effective_max = cfg.max_steps if stop_after_steps is None else min(cfg.max_steps, stop_after_steps)

    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = _cfg_to_dict(cfg)
    env = _capture_env_metadata()
    git = _capture_git_metadata()
    cfg_hash = _config_hash(cfg_dict)
    resume_path = output_dir / f"{name}_last.pt"
    resume_requested = _env_flag("NM_RESUME")
    existing_artifacts = list(output_dir.iterdir())
    if resume_requested:
        if not resume_path.exists():
            raise RuntimeError(
                f"NM_RESUME=1 was set, but no resume checkpoint exists at {resume_path}. "
                f"refusing to treat a dirty or partial directory as resumable without a last checkpoint."
            )
    elif existing_artifacts:
        joined = ", ".join(path.name for path in existing_artifacts[:5])
        raise RuntimeError(
            f"output directory {output_dir} is not fresh ({joined}). "
            f"use a fresh output directory for a fresh benchmark or launch, or set NM_RESUME=1 for an intentional continuation."
        )

    metadata = {
        "run": name,
        "config": cfg_dict,
        "config_hash": cfg_hash,
        "environment": env,
        "git": git,
        "dataset_source": dataset_source,
        "seed": cfg.seed,
        "started_at": datetime.now().isoformat(),
    }

    metadata_path = output_dir / f"{name}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    log(f"wrote metadata: {metadata_path}")
    log(f"config_hash: {cfg_hash}")
    log(f"git sha: {git.get('sha')}, dirty: {git.get('dirty')}")
    log(f"dataset: {dataset_source}")

    metrics_logger = MetricsLogger(output_dir / f"{name}_metrics.jsonl")
    metrics_logger.log({"event": "run_start", **metadata})

    model = GodMachine(cfg).to(DEVICE)
    n_params = model.count_params()
    log(f"params: {n_params:,}")
    log(f"layers: {len(cfg.layer_types)} ({cfg.layer_pattern} x {cfg.n_layers // len(cfg.layer_pattern)})")

    skip_validation = _env_flag("NM_SKIP_VALIDATION")
    train_ds = ByteDataset(train_data, cfg.seq_len)
    val_dl: DataLoader | None = None
    if skip_validation:
        log(f"train batches per epoch: {len(train_ds) // cfg.batch_size}, val batches: skipped (NM_SKIP_VALIDATION=1)")
    else:
        val_ds = ByteDataset(val_data, cfg.seq_len)
        log(f"train batches per epoch: {len(train_ds) // cfg.batch_size}, val batches: {len(val_ds) // cfg.batch_size}")

    train_generator = torch.Generator()
    train_generator.manual_seed(cfg.seed)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        generator=train_generator,
    )
    if not skip_validation:
        val_dl = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=True,
        )

    param_groups = _build_param_groups(model, cfg.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: _cosine_lr(s, cfg.warmup_steps, cfg.max_steps)
    )

    if cfg.grad_checkpointing and n_params > 100_000_000:
        model.enable_gradient_checkpointing()
        log("gradient checkpointing enabled")

    use_amp = cfg.amp and DEVICE.type == "cuda"
    amp_dtype = torch.float32
    if use_amp and torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8:
            amp_dtype = torch.bfloat16
        else:
            log(
                f"warning: sm_{cap[0]}{cap[1]} lacks bf16 tensor cores; "
                f"disabling amp (fp16 without grad scaler is unstable for this architecture)"
            )
            use_amp = False
    if use_amp:
        log(f"amp enabled ({amp_dtype})")

    history: dict[str, list] = {
        "steps": [],
        "train_loss": [],
        "grad_norm_pre_clip": [],
        "lr": [],
        "throughput_tok_per_sec": [],
        "gpu_mem_alloc_gb": [],
        "gpu_mem_max_alloc_gb": [],
        "step_duration_sec": [],
        "val_steps": [],
    }

    step = 0
    best_val_bpb = float("inf")
    tokens_seen_total = 0
    start_time = time.time()
    last_step_time = start_time

    batches_per_epoch = max(1, len(train_dl))
    epoch_idx = 0
    step_in_epoch = 0
    resume_skip_batches = 0
    train_gen_state_at_epoch_start: Tensor | None = None

    shutdown_flag = {"requested": False}

    def _handle_signal(signum: int, frame: Any) -> None:
        shutdown_flag["requested"] = True
        log(f"received signal {signum}; will flush and exit after current step")

    try:
        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)
    except (ValueError, OSError):
        pass

    if resume_path.exists():
        try:
            resumed = load_resume_checkpoint(resume_path, model, optimizer, scheduler)
            resumed_cfg = resumed.get("config", {})
            resumed_hash = _config_hash(resumed_cfg) if resumed_cfg else None
            if resumed_hash is not None and resumed_hash != cfg_hash:
                raise RuntimeError(
                    f"resume checkpoint config hash {resumed_hash} does not match active config hash {cfg_hash}. "
                    f"refusing to continue a different preset or override set."
                )
            if launch_contract is not None and resumed.get("launch_contract") != launch_contract:
                raise RuntimeError(
                    "resume checkpoint launch contract does not match the validated benchmark/full-run contract."
                )
            step = resumed["step"]
            best_val_bpb = resumed["best_val_bpb"]
            epoch_idx = resumed.get("epoch_idx", 0)
            step_in_epoch = resumed.get("step_in_epoch", 0)
            resumed_history = resumed.get("history", {})
            expected_len = len(resumed_history.get("steps", []))
            expected_val_len = len(resumed_history.get("val_steps", []))
            for k, v in resumed_history.items():
                if not isinstance(v, list):
                    history[k] = v
                    continue
                if k == "val_steps" or k.startswith("val_"):
                    if len(v) == expected_val_len:
                        history[k] = v
                elif len(v) == expected_len:
                    history[k] = v
            saved_gen_state = resumed.get("train_gen_state")
            if saved_gen_state is not None:
                train_generator.set_state(saved_gen_state)
                train_gen_state_at_epoch_start = saved_gen_state
                resume_skip_batches = step_in_epoch
                log(
                    f"resumed from step {step} (epoch={epoch_idx}, "
                    f"step_in_epoch={step_in_epoch}); restored train generator state"
                )
            else:
                log(
                    f"resumed from step {step}, but no train_gen_state in checkpoint. "
                    f"data stream will restart from epoch 0; this will not match a "
                    f"from-scratch run and batches may be revisited."
                )
            metrics_logger.log({
                "event": "resume",
                "step": step,
                "epoch_idx": epoch_idx,
                "step_in_epoch": step_in_epoch,
                "best_val_bpb": best_val_bpb,
                "gen_state_restored": saved_gen_state is not None,
            })
        except Exception as e:
            raise RuntimeError(f"resume failed: {e}") from e

    def _flush_emergency(reason: str, extra: dict | None = None) -> None:
        try:
            rec = {
                "event": "emergency_flush",
                "reason": reason,
                "step": step,
                "epoch_idx": epoch_idx,
                "step_in_epoch": step_in_epoch,
            }
            if extra:
                rec.update(extra)
            metrics_logger.log(rec)
            metrics_logger.fsync()
        except Exception:
            pass
        try:
            save_resume_checkpoint(
                output_dir / f"{name}_crash.pt",
                model, optimizer, scheduler,
                step, best_val_bpb, history, cfg_dict,
                train_gen_state=train_gen_state_at_epoch_start,
                step_in_epoch=step_in_epoch,
                epoch_idx=epoch_idx,
                launch_contract=launch_contract,
            )
        except Exception as e:
            log(f"crash checkpoint save failed: {e}")
        try:
            _write_results_snapshot(
                output_dir / f"{name}_results.json",
                name, n_params, best_val_bpb, step,
                time.time() - start_time, tokens_seen_total,
                cfg_dict, metadata, history, final=False,
            )
        except Exception as e:
            log(f"results snapshot save failed: {e}")

    model.train(True)

    try:
        while step < effective_max and not shutdown_flag["requested"]:
            if train_gen_state_at_epoch_start is None or resume_skip_batches == 0:
                train_gen_state_at_epoch_start = train_generator.get_state()
            for batch in train_dl:
                if resume_skip_batches > 0:
                    resume_skip_batches -= 1
                    continue
                if step >= effective_max or shutdown_flag["requested"]:
                    break

                batch = batch.to(DEVICE)
                input_ids = batch[:, :-1]
                target = batch[:, 1:]
                batch_tokens = int(target.numel())

                with torch.amp.autocast(device_type=DEVICE.type, dtype=amp_dtype, enabled=use_amp):
                    logits, _, aux = model(input_ids)
                    main_loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        target.reshape(-1),
                    )
                    pc_loss_val = aggregate_pc_loss(aux, device=main_loss.device)
                    slot_pred_loss_val = aggregate_slot_prediction_loss(aux, device=main_loss.device)
                    loss = main_loss + cfg.pc_lambda * pc_loss_val + cfg.slot_prediction_lambda * slot_pred_loss_val
                    loss_scaled = loss / cfg.grad_accum_steps if cfg.grad_accum_steps > 1 else loss

                loss_val = loss.detach().float().item()
                if not math.isfinite(loss_val):
                    log(f"fatal: non-finite loss at step {step}: {loss_val}")
                    _flush_emergency("non_finite_loss", {"loss": loss_val})
                    raise RuntimeError(f"non-finite loss at step {step}: {loss_val}")

                spike_stats = collect_spike_stats(aux)

                loss_scaled.backward()

                grad_norm_val: float = float("nan")
                do_step = ((step + 1) % cfg.grad_accum_steps == 0) or (step == effective_max - 1)
                if do_step:
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    grad_norm_val = gn.item() if isinstance(gn, torch.Tensor) else float(gn)
                    if not math.isfinite(grad_norm_val):
                        log(f"fatal: non-finite grad norm at step {step}: {grad_norm_val}")
                        _flush_emergency("non_finite_grad_norm", {"grad_norm": grad_norm_val})
                        raise RuntimeError(f"non-finite grad norm at step {step}: {grad_norm_val}")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                tokens_seen_total += batch_tokens
                now = time.time()
                step_duration = now - last_step_time
                last_step_time = now
                elapsed = now - start_time
                tokens_per_sec = batch_tokens / max(step_duration, 1e-9)

                mem_alloc_gb = 0.0
                mem_max_alloc_gb = 0.0
                if torch.cuda.is_available():
                    try:
                        mem_alloc_gb = torch.cuda.memory_allocated() / 1e9
                        mem_max_alloc_gb = torch.cuda.max_memory_allocated() / 1e9
                    except Exception:
                        pass

                lr_now = scheduler.get_last_lr()[0]

                history["steps"].append(step)
                history["train_loss"].append(loss_val)
                history["grad_norm_pre_clip"].append(grad_norm_val)
                history["lr"].append(lr_now)
                history["throughput_tok_per_sec"].append(tokens_per_sec)
                history["gpu_mem_alloc_gb"].append(mem_alloc_gb)
                history["gpu_mem_max_alloc_gb"].append(mem_max_alloc_gb)
                history["step_duration_sec"].append(step_duration)
                for k_stat, v_stat in spike_stats.items():
                    history.setdefault(k_stat, []).append(v_stat)

                step_record: dict[str, Any] = {
                    "event": "step",
                    "step": step,
                    "loss": loss_val,
                    "grad_norm_pre_clip": grad_norm_val,
                    "lr": lr_now,
                    "throughput_tok_per_sec": tokens_per_sec,
                    "step_duration_sec": step_duration,
                    "tokens_seen_total": tokens_seen_total,
                    "gpu_mem_alloc_gb": mem_alloc_gb,
                    "gpu_mem_max_alloc_gb": mem_max_alloc_gb,
                    "elapsed_sec": elapsed,
                    "wall_time": now,
                }
                for k, v in spike_stats.items():
                    if k not in step_record:
                        step_record[k] = v
                metrics_logger.log(step_record)

                if step % 50 == 0:
                    log(
                        f"step {step}/{cfg.max_steps} loss={loss_val:.4f} "
                        f"gn={grad_norm_val:.3f} fr={spike_stats['mean_fr']:.3f} "
                        f"tps={tokens_per_sec:.0f} mem={mem_alloc_gb:.2f}gb "
                        f"lr={lr_now:.2e} t={elapsed:.0f}s"
                    )

                step += 1
                step_in_epoch += 1

                if not skip_validation and step % cfg.val_interval == 0:
                    assert val_dl is not None
                    val_result = run_validation(model, val_dl, use_amp=use_amp, amp_dtype=amp_dtype)
                    history["val_steps"].append(step)
                    for k_v, v_v in val_result.items():
                        hk = f"val_{k_v}" if not k_v.startswith("val_") else k_v
                        history.setdefault(hk, []).append(v_v)
                    val_record: dict[str, Any] = {"event": "validation", "step": step}
                    for k_v, v_v in val_result.items():
                        val_record[f"val_{k_v}" if not k_v.startswith("val_") else k_v] = v_v
                    metrics_logger.log(val_record)
                    if val_result["bpb"] < best_val_bpb:
                        best_val_bpb = val_result["bpb"]
                        torch.save(model.state_dict(), output_dir / f"{name}_best.pt")
                        log(f"val bpb={val_result['bpb']:.4f} best={best_val_bpb:.4f} (saved)")
                    else:
                        log(f"val bpb={val_result['bpb']:.4f} best={best_val_bpb:.4f}")

                    save_resume_checkpoint(
                        output_dir / f"{name}_last.pt",
                        model, optimizer, scheduler,
                        step, best_val_bpb, history, cfg_dict,
                        train_gen_state=train_gen_state_at_epoch_start,
                        step_in_epoch=step_in_epoch,
                        epoch_idx=epoch_idx,
                        launch_contract=launch_contract,
                    )
                    _write_results_snapshot(
                        output_dir / f"{name}_results.json",
                        name, n_params, best_val_bpb, step,
                        time.time() - start_time, tokens_seen_total,
                        cfg_dict, metadata, history, final=False,
                    )
                    metrics_logger.fsync()

                    if cfg.imagination_enabled and history.get("imag_ratio_mean"):
                        recent = [
                            x for x in history["imag_ratio_mean"][-50:]
                            if x is not None and math.isfinite(x)
                        ]
                        if recent:
                            imag_avg = float(np.mean(recent))
                            if imag_avg < 0.02:
                                log(
                                    f"  warning: imag_ratio_mean={imag_avg:.4f} below 0.02 threshold "
                                    f"across last {len(recent)} steps. imag_filter may be dead weight."
                                )
                    if cfg.pc_diagnostic_enabled and history.get("pc_error_l2_mean"):
                        recent_pc = [
                            x for x in history["pc_error_l2_mean"][-50:]
                            if x is not None and math.isfinite(x)
                        ]
                        if recent_pc:
                            pc_avg = float(np.mean(recent_pc))
                            log(f"  pc_error_l2_mean={pc_avg:.4f} (diagnostic)")

                    if torch.cuda.is_available():
                        try:
                            torch.cuda.reset_peak_memory_stats()
                        except Exception:
                            pass

            if step >= effective_max or shutdown_flag["requested"]:
                break
            epoch_idx += 1
            step_in_epoch = 0
            train_gen_state_at_epoch_start = None

        already_valed_here = bool(history["val_steps"]) and history["val_steps"][-1] == step
        if shutdown_flag["requested"]:
            final_val = {
                "mean_loss": float("nan"),
                "bpb": float("nan"),
                "tokens": 0,
                "precision": "skipped_on_shutdown",
            }
            skip_record: dict[str, Any] = {
                "event": "final_validation_skipped",
                "step": step,
                "reason": "shutdown_requested",
            }
            for k_v, v_v in final_val.items():
                skip_record[f"val_{k_v}" if not k_v.startswith("val_") else k_v] = v_v
            metrics_logger.log(skip_record)
        elif skip_validation:
            final_val = {
                "mean_loss": float("nan"),
                "bpb": float("nan"),
                "tokens": 0,
                "precision": "skipped_by_flag",
            }
            skip_record = {
                "event": "final_validation_skipped",
                "step": step,
                "reason": "NM_SKIP_VALIDATION",
            }
            for k_v, v_v in final_val.items():
                skip_record[f"val_{k_v}" if not k_v.startswith("val_") else k_v] = v_v
            metrics_logger.log(skip_record)
        elif already_valed_here:
            last_bpb = history["val_bpb"][-1]
            last_mean_loss = history["val_mean_loss"][-1]
            final_val = {
                "mean_loss": last_mean_loss,
                "bpb": last_bpb,
                "tokens": 0,
                "precision": "reused_mid_loop_val",
            }
            dedup_record: dict[str, Any] = {
                "event": "final_validation_deduped",
                "step": step,
                "reused_from_mid_loop_val": True,
            }
            for k_v, v_v in final_val.items():
                dedup_record[f"val_{k_v}" if not k_v.startswith("val_") else k_v] = v_v
            metrics_logger.log(dedup_record)
        else:
            assert val_dl is not None
            final_val = run_validation(model, val_dl, use_amp=use_amp, amp_dtype=amp_dtype)
            history["val_steps"].append(step)
            for k_v, v_v in final_val.items():
                hk = f"val_{k_v}" if not k_v.startswith("val_") else k_v
                history.setdefault(hk, []).append(v_v)
            final_record: dict[str, Any] = {"event": "final_validation", "step": step}
            for k_v, v_v in final_val.items():
                final_record[f"val_{k_v}" if not k_v.startswith("val_") else k_v] = v_v
            metrics_logger.log(final_record)
            if math.isfinite(final_val["bpb"]) and final_val["bpb"] < best_val_bpb:
                best_val_bpb = final_val["bpb"]
                torch.save(model.state_dict(), output_dir / f"{name}_best.pt")

        total_time = time.time() - start_time
        log(
            f"{name} done. final_bpb={final_val['bpb']:.4f} best_bpb={best_val_bpb:.4f} "
            f"time={total_time:.0f}s tokens={tokens_seen_total:,}"
        )

        save_resume_checkpoint(
            output_dir / f"{name}_last.pt",
            model, optimizer, scheduler,
            step, best_val_bpb, history, cfg_dict,
            train_gen_state=train_gen_state_at_epoch_start,
            step_in_epoch=step_in_epoch,
            epoch_idx=epoch_idx,
            launch_contract=launch_contract,
        )
        _write_results_snapshot(
            output_dir / f"{name}_results.json",
            name, n_params, best_val_bpb, step,
            total_time, tokens_seen_total,
            cfg_dict, metadata, history, final=True,
        )
        metrics_logger.log({
            "event": "run_end",
            "total_steps": step,
            "total_time_sec": total_time,
            "best_val_bpb": best_val_bpb,
            "final_val_bpb": final_val["bpb"],
            "tokens_seen_total": tokens_seen_total,
        })

        return {
            "name": name,
            "param_count": n_params,
            "final_val_bpb": final_val["bpb"],
            "best_val_bpb": best_val_bpb,
            "total_steps": step,
            "total_time": total_time,
            "tokens_seen_total": tokens_seen_total,
            "history": history,
            "metadata": metadata,
            "dataset_source": dataset_source,
        }

    except Exception as e:
        log(f"training failed: {type(e).__name__}: {e}")
        _flush_emergency("exception", {"error_type": type(e).__name__, "error": str(e)})
        raise

    finally:
        metrics_logger.fsync()
        metrics_logger.close()


def _model_device(model: nn.Module) -> torch.device:
    for p in model.parameters():
        return p.device
    for b in model.buffers():
        return b.device
    return torch.device("cpu")


@torch.no_grad()
def greedy_generate(
    model: GodMachine,
    prompt: bytes,
    max_new_tokens: int = 64,
) -> bytes:
    was_training = model.training
    model.train(False)
    device = _model_device(model)

    states: list[Any] | None = None
    offset = 0
    logits = None
    for i, byte_val in enumerate(prompt):
        single = torch.tensor([[byte_val]], dtype=torch.long, device=device)
        logits, states, _ = model(single, states=states, offset=offset)
        offset += 1

    if logits is None:
        raise ValueError("greedy_generate requires a non-empty prompt")

    next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = [int(next_token.item())]
    for _ in range(max_new_tokens - 1):
        logits, states, _ = model(next_token, states=states, offset=offset)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(int(next_token.item()))
        offset += 1

    model.train(was_training)
    return bytes(generated)


@torch.no_grad()
def passkey_retrieval_test(
    model: GodMachine,
    context_length: int,
    num_trials: int = 10,
    passkey_length: int = 5,
    chunk_size: int = 256,
    seed: int = 0,
) -> dict:
    was_training = model.training
    model.train(False)
    device = _model_device(model)
    rng = np.random.RandomState(seed + context_length)
    correct = 0
    for trial in range(num_trials):
        passkey = rng.randint(10 ** (passkey_length - 1), 10 ** passkey_length)
        passkey_bytes = list(str(passkey).encode("ascii"))
        marker_start = [255, 254, 253]
        marker_end = [253, 254, 255]
        marker_query = [252, 251, 250]
        filler = rng.randint(32, 127, size=context_length * 2).tolist()
        overhead = len(marker_start) + len(passkey_bytes) + len(marker_end) + len(marker_query)
        max_insert = max(1, context_length - overhead - 20)
        insert_pos = rng.randint(5, max(6, min(max_insert, context_length // 3)))
        before = filler[:insert_pos]
        mid_len = max(0, context_length - overhead - insert_pos)
        middle = filler[insert_pos : insert_pos + mid_len]
        sequence = before + marker_start + passkey_bytes + marker_end + middle + marker_query
        sequence = sequence[:context_length]
        if len(sequence) < context_length:
            sequence = sequence + filler[len(sequence) : context_length]
        input_tensor = torch.tensor([sequence[:context_length]], dtype=torch.long, device=device)
        states: list[Any] | None = None
        logits = None
        for start in range(0, input_tensor.shape[1], chunk_size):
            end = min(start + chunk_size, input_tensor.shape[1])
            chunk = input_tensor[:, start:end]
            logits, states, _ = model(chunk, states=states, offset=start)
        if logits is None:
            continue
        generated = []
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(int(next_token.item()))
        offset = input_tensor.shape[1]
        for step in range(passkey_length - 1):
            logits, states, _ = model(next_token, states=states, offset=offset + step)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(int(next_token.item()))
        generated_str = bytes(generated).decode("ascii", errors="replace")
        if generated_str == str(passkey):
            correct += 1
    model.train(was_training)
    return {
        "accuracy": correct / max(num_trials, 1),
        "correct": correct,
        "total": num_trials,
        "context_length": context_length,
    }


@torch.no_grad()
def selective_copy_test(
    model: GodMachine,
    context_length: int,
    num_trials: int = 20,
    copy_length: int = 8,
    chunk_size: int = 256,
    seed: int = 7,
) -> dict:
    was_training = model.training
    model.train(False)
    device = _model_device(model)
    rng = np.random.RandomState(seed + context_length)
    correct = 0
    for trial in range(num_trials):
        target_bytes = rng.randint(48, 58, size=copy_length).tolist()
        marker = [255, 254]
        end_marker = [254, 255]
        query_marker = [253, 252]
        filler = rng.randint(65, 91, size=context_length).tolist()
        insert_pos = rng.randint(10, max(11, context_length // 3))
        sequence = filler[:insert_pos] + marker + target_bytes + end_marker
        target_query_end = context_length
        target_query_start = target_query_end - len(query_marker)
        if len(sequence) >= target_query_start:
            sequence = sequence[:target_query_start]
        else:
            sequence = sequence + filler[len(sequence) : target_query_start]
        sequence = sequence + query_marker
        if len(sequence) != context_length:
            sequence = sequence[:context_length]
        input_tensor = torch.tensor([sequence], dtype=torch.long, device=device)
        states: list[Any] | None = None
        logits = None
        for start in range(0, input_tensor.shape[1], chunk_size):
            end = min(start + chunk_size, input_tensor.shape[1])
            logits, states, _ = model(input_tensor[:, start:end], states=states, offset=start)
        if logits is None:
            continue
        generated = []
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(int(next_token.item()))
        offset = input_tensor.shape[1]
        for step in range(copy_length - 1):
            logits, states, _ = model(next_token, states=states, offset=offset + step)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(int(next_token.item()))
        if generated == target_bytes:
            correct += 1
    model.train(was_training)
    return {
        "accuracy": correct / max(num_trials, 1),
        "correct": correct,
        "total": num_trials,
        "context_length": context_length,
    }


@torch.no_grad()
def measure_perplexity_at_length(
    model: GodMachine,
    data: Any,
    context_length: int,
    num_windows: int = 5,
    chunk_size: int = 256,
) -> dict:
    was_training = model.training
    model.train(False)
    device = _model_device(model)
    total_loss = 0.0
    total_tokens = 0
    data_len = len(data) if hasattr(data, "__len__") else 0
    for window_idx in range(num_windows):
        start = window_idx * context_length
        if start + context_length + 1 > data_len:
            break
        if isinstance(data, (bytes, bytearray)):
            chunk_bytes = list(data[start : start + context_length + 1])
        else:
            chunk_bytes = [int(b) for b in data[start : start + context_length + 1]]
        input_tensor = torch.tensor([chunk_bytes], dtype=torch.long, device=device)
        states: list[Any] | None = None
        all_logits = []
        for cs in range(0, input_tensor.shape[1] - 1, chunk_size):
            ce = min(cs + chunk_size, input_tensor.shape[1] - 1)
            chunk = input_tensor[:, cs:ce]
            logits, states, _ = model(chunk, states=states, offset=cs)
            all_logits.append(logits)
        if not all_logits:
            continue
        all_logits_cat = torch.cat(all_logits, dim=1)
        targets = input_tensor[:, 1 : all_logits_cat.shape[1] + 1]
        loss = F.cross_entropy(
            all_logits_cat.reshape(-1, all_logits_cat.shape[-1]),
            targets.reshape(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += int(targets.numel())
    model.train(was_training)
    if total_tokens == 0:
        return {"bpb": float("inf"), "context_length": context_length, "total_tokens": 0}
    mean_loss = total_loss / total_tokens
    return {
        "bpb": mean_loss / math.log(2),
        "context_length": context_length,
        "total_tokens": total_tokens,
    }


@torch.no_grad()
def run_delta_state_structure_probe(
    model: GodMachine,
    val_data: Any,
    n_warmup: int = 512,
    n_probes: int = 32,
    seed: int = 2026,
) -> dict:
    was_training = model.training
    model.train(False)
    device = _model_device(model)

    if isinstance(val_data, (bytes, bytearray)):
        warmup_tokens = list(val_data[:n_warmup])
    elif isinstance(val_data, np.ndarray):
        warmup_tokens = [int(b) for b in val_data[:n_warmup]]
    else:
        warmup_tokens = [int(b) for b in val_data[:n_warmup]]

    if len(warmup_tokens) == 0:
        model.train(was_training)
        return {"error": "no warmup data", "per_layer": {}}

    warmup_input = torch.tensor([warmup_tokens], dtype=torch.long, device=device)
    logits, states, aux = model(warmup_input)

    per_layer: dict[int, dict[str, float]] = {}
    for layer_idx, (block, state) in enumerate(zip(model.blocks, states)):
        if block.layer_type != "DELTA" or state is None:
            continue

        state_f = state.float()
        if state_f.dim() != 4:
            continue
        B, H, D_k, D_v = state_f.shape

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + layer_idx)
        probe_queries = torch.randn(n_probes, H, D_k, generator=gen).to(device=device, dtype=state_f.dtype)
        probe_queries = F.normalize(probe_queries, p=2, dim=-1)

        outputs = torch.einsum("nhd,bhde->nbhe", probe_queries, state_f).squeeze(1)

        rand_gen = torch.Generator(device="cpu")
        rand_gen.manual_seed(seed + 1000 + layer_idx)
        random_state = torch.randn(B, H, D_k, D_v, generator=rand_gen).to(device=device, dtype=state_f.dtype)
        state_mean_per_head = state_f.float().mean(dim=(0, 2, 3), keepdim=True)
        state_std_per_head = state_f.float().std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-9)
        random_state = (random_state.float() * state_std_per_head + state_mean_per_head).to(state_f.dtype)
        random_outputs = torch.einsum("nhd,bhde->nbhe", probe_queries, random_state).squeeze(1)

        outputs_flat = outputs.reshape(n_probes, -1).float()
        random_flat = random_outputs.reshape(n_probes, -1).float()

        out_norm = outputs_flat.norm(dim=-1).mean().item()
        rand_norm = random_flat.norm(dim=-1).mean().item()

        outputs_normed = F.normalize(outputs_flat, p=2, dim=-1)
        random_normed = F.normalize(random_flat, p=2, dim=-1)

        pairwise_out = (outputs_normed @ outputs_normed.T)
        pairwise_rand = (random_normed @ random_normed.T)

        if n_probes > 1:
            off_diag_mask = ~torch.eye(n_probes, dtype=torch.bool, device=device)
            mean_pairwise_out = pairwise_out[off_diag_mask].mean().item()
            mean_pairwise_rand = pairwise_rand[off_diag_mask].mean().item()
        else:
            mean_pairwise_out = 0.0
            mean_pairwise_rand = 0.0

        state_norm = state_f.norm().item()
        structure_ratio = out_norm / max(rand_norm, 1e-9)

        per_layer[int(layer_idx)] = {
            "state_frobenius": state_norm,
            "probe_output_norm": out_norm,
            "random_output_norm": rand_norm,
            "structure_ratio": structure_ratio,
            "mean_pairwise_cos": mean_pairwise_out,
            "random_pairwise_cos": mean_pairwise_rand,
        }

    model.train(was_training)

    if not per_layer:
        return {"error": "no DELTA layers had populated state", "per_layer": per_layer}

    structure_vals = [r["structure_ratio"] for r in per_layer.values()]
    pairwise_vals = [r["mean_pairwise_cos"] for r in per_layer.values()]
    random_pairwise_vals = [r["random_pairwise_cos"] for r in per_layer.values()]

    return {
        "n_warmup": n_warmup,
        "n_probes": n_probes,
        "per_layer": per_layer,
        "mean_structure_ratio": float(np.mean(structure_vals)),
        "mean_pairwise_cos": float(np.mean(pairwise_vals)),
        "random_mean_pairwise_cos": float(np.mean(random_pairwise_vals)),
    }


def _wilson_upper_95(correct: int, total: int) -> float:
    if total <= 0:
        return 1.0
    p = correct / total
    z = 1.96
    denom = 1.0 + (z * z) / total
    center = p + (z * z) / (2.0 * total)
    radius = z * math.sqrt((p * (1.0 - p) / total) + (z * z) / (4.0 * total * total))
    return min(1.0, (center + radius) / denom)


def run_eval_suite(
    model: GodMachine,
    val_data: Any,
    output_dir: Path,
    name: str,
) -> dict:
    log(f"=== running eval suite for {name} ===")
    results: dict[str, Any] = {}

    log("passkey retrieval tests")
    passkey_results = {}
    for ctx_len in [256, 1024, 4096]:
        t0 = time.time()
        pr = passkey_retrieval_test(model, ctx_len, num_trials=100)
        passkey_results[str(ctx_len)] = pr
        ci_upper = _wilson_upper_95(pr["correct"], pr["total"])
        log(
            f"  passkey @{ctx_len}: {pr['accuracy']:.1%} "
            f"({pr['correct']}/{pr['total']}, 95% CI upper {ci_upper:.1%}) "
            f"[{time.time() - t0:.0f}s]"
        )
    results["passkey"] = passkey_results

    log("selective copy tests")
    copy_results = {}
    for ctx_len in [256, 512, 1024, 2048]:
        t0 = time.time()
        cr = selective_copy_test(model, ctx_len, num_trials=100)
        copy_results[str(ctx_len)] = cr
        ci_upper = _wilson_upper_95(cr["correct"], cr["total"])
        log(
            f"  copy @{ctx_len}: {cr['accuracy']:.1%} "
            f"({cr['correct']}/{cr['total']}, 95% CI upper {ci_upper:.1%}) "
            f"[{time.time() - t0:.0f}s]"
        )
    results["selective_copy"] = copy_results

    log("perplexity at length")
    ppl_results = {}
    data_len = len(val_data) if hasattr(val_data, "__len__") else 0
    for ctx_len in [256, 512, 1024, 2048, 4096]:
        if ctx_len * 6 > data_len:
            break
        t0 = time.time()
        pr = measure_perplexity_at_length(model, val_data, ctx_len, num_windows=5)
        ppl_results[str(ctx_len)] = pr
        log(f"  bpb @{ctx_len}: {pr['bpb']:.4f} [{time.time() - t0:.0f}s]")
    results["perplexity_at_length"] = ppl_results

    log("delta state structure probe")
    t0 = time.time()
    imag = run_delta_state_structure_probe(model, val_data, n_warmup=512, n_probes=32)
    log(
        f"  structure_ratio={imag.get('mean_structure_ratio', 0):.3f} "
        f"pairwise_cos={imag.get('mean_pairwise_cos', 0):.3f} "
        f"random_pairwise_cos={imag.get('random_mean_pairwise_cos', 0):.3f} "
        f"[{time.time() - t0:.0f}s]"
    )
    results["delta_state_structure_probe"] = imag

    eval_path = output_dir / f"{name}_eval_suite.json"
    _write_json_atomic(eval_path, results)
    log(f"wrote eval suite to {eval_path}")
    return results


def _run_fla_parity_variant(
    label: str,
    *,
    bcm_alpha: bool,
    seed: int,
    tol_rel: float = 1e-2,
    delta_erasure: bool = True,
) -> None:
    torch.manual_seed(seed)
    test_cfg = Config(
        d_model=64,
        n_layers=1,
        delta_num_heads=2,
        delta_head_dim=32,
        attn_d_c=16,
        attn_d_R=8,
        attn_num_heads=2,
        mlp_ratio=2.0,
        layer_pattern=("DELTA",),
        kwta_enabled=False,
        delta_erasure_enabled=delta_erasure,
        bcm_alpha_enabled=bcm_alpha,
        imagination_enabled=False,
        pc_diagnostic_enabled=False,
        multi_compartment_enabled=False,
        use_fla_if_available=True,
        batch_size=2,
        seq_len=32,
        max_steps=1,
        warmup_steps=1,
        val_interval=100,
        grad_checkpointing=False,
        amp=False,
    )
    layer = DeltaRuleMemory(test_cfg).to("cuda")
    layer.train(False)
    if bcm_alpha:
        with torch.no_grad():
            layer.running_state_norm.fill_(2.5)

    B, T = 2, 32
    x = torch.randn(B, T, test_cfg.d_model, device="cuda")

    with torch.no_grad():
        out_fla, _, _ = layer(x)

        layer.use_fla = False
        chunks: list[Tensor] = []
        state = None
        for t in range(T):
            single = x[:, t : t + 1, :]
            out_t, state, _ = layer(single, state=state, offset=t)
            chunks.append(out_t)
        out_rec = torch.cat(chunks, dim=1)
        layer.use_fla = True

    diff = (out_fla - out_rec).abs()
    max_abs = diff.max().item()
    ref = out_fla.abs().max().item() + 1e-9
    rel = max_abs / ref
    log(f"  {label}: max_abs={max_abs:.4e}, rel={rel:.4e}")
    if rel >= tol_rel:
        raise AssertionError(
            f"fla and recurrent paths diverge ({label}): "
            f"max_abs={max_abs:.4e}, rel={rel:.4e}. expected rel < {tol_rel}. "
            f"this indicates the CPU recurrence form does not match FLA's gated delta rule."
        )


def _test_fla_vs_recurrent_parity() -> None:
    if not FLA_AVAILABLE or not torch.cuda.is_available():
        log("fla-vs-recurrent parity test skipped (fla not available or no cuda)")
        return
    log("fla-vs-recurrent parity test (four variants, erasure-on and erasure-off)")
    _run_fla_parity_variant("fla vs recurrent, erasure=on, bcm=off", bcm_alpha=False, seed=54321, delta_erasure=True)
    _run_fla_parity_variant("fla vs recurrent, erasure=on, bcm=on", bcm_alpha=True, seed=54322, delta_erasure=True)
    if chunk_simple_gla is not None:
        _run_fla_parity_variant("fla vs recurrent, erasure=off, bcm=off", bcm_alpha=False, seed=54323, delta_erasure=False)
        _run_fla_parity_variant("fla vs recurrent, erasure=off, bcm=on", bcm_alpha=True, seed=54324, delta_erasure=False)
    else:
        log("  skipping no-erasure parity variants: chunk_simple_gla not available")


def _test_slot_memory_parity() -> None:
    log("slot memory parity test (cpu, no cuda required)")
    torch.manual_seed(98765)
    test_cfg = Config(
        d_model=64,
        n_layers=1,
        delta_num_heads=2,
        delta_head_dim=32,
        attn_d_c=16,
        attn_d_R=8,
        attn_num_heads=2,
        mlp_ratio=2.0,
        layer_pattern=("SLOT",),
        kwta_enabled=False,
        delta_erasure_enabled=False,
        bcm_alpha_enabled=False,
        imagination_enabled=False,
        pc_diagnostic_enabled=False,
        multi_compartment_enabled=False,
        use_fla_if_available=False,
        slot_num_slots=8,
        slot_log_temperature_init=-2.3,
        slot_gate_init=0.0,
        alpha_log_mean=3.0,
        batch_size=1,
        seq_len=8,
        max_steps=1,
        warmup_steps=1,
        val_interval=100,
        grad_checkpointing=False,
        amp=False,
    )
    layer = SlotMemory(test_cfg)
    layer.train(False)
    T_seq = 5
    D = test_cfg.d_model
    unique_patterns = torch.randn(T_seq, D)
    unique_patterns = unique_patterns / unique_patterns.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    x = unique_patterns.unsqueeze(0)
    with torch.no_grad():
        out_write, state_after_write, aux_write = layer(x)
        state_norm = float(state_after_write["slot_state"].norm().item())
        log(
            "  write pass: "
            f"slot_state_norm={state_norm:.3f}, "
            f"slot_write_fraction={float(aux_write['slot_write_fraction'].item()):.3f}, "
            f"allocation_entropy={float(aux_write['slot_allocation_entropy'].item()):.3f}, "
            f"slot_gate_mean={float(aux_write['slot_gate_mean'].item()):.3f}"
        )
        out_read, _, aux_read = layer(x, state=state_after_write, offset=T_seq)
    if state_norm <= 1e-4:
        raise AssertionError(
            f"slot memory parity: state did not accumulate any content (norm={state_norm:.4e})"
        )
    read_norm = float(out_read.norm(dim=-1).mean().item())
    if read_norm <= 1e-4:
        raise AssertionError(
            f"slot memory parity: read output norm collapsed to {read_norm:.4e}; slot memory returning zeros"
        )
    log(
        "  read pass: "
        f"read_out mean norm={read_norm:.4f}, "
        f"state_frobenius={float(aux_read['state_frobenius'].item()):.3f}"
    )
    log("slot memory parity ok (state accumulated, read produces non-zero output)")


def _test_alpha_eff_math() -> None:
    log("alpha_eff math sanity test (cpu, no fla required)")
    torch.manual_seed(77)
    test_cfg = Config(
        d_model=64,
        n_layers=1,
        delta_num_heads=2,
        delta_head_dim=32,
        attn_d_c=16,
        attn_d_R=8,
        attn_num_heads=2,
        mlp_ratio=2.0,
        layer_pattern=("DELTA",),
        kwta_enabled=False,
        delta_erasure_enabled=True,
        bcm_alpha_enabled=True,
        imagination_enabled=False,
        pc_diagnostic_enabled=False,
        multi_compartment_enabled=False,
        use_fla_if_available=False,
        batch_size=2,
        seq_len=8,
        max_steps=1,
        warmup_steps=1,
        val_interval=100,
        grad_checkpointing=False,
        amp=False,
    )
    layer = DeltaRuleMemory(test_cfg)
    layer.train(False)
    with torch.no_grad():
        layer.running_state_norm.fill_(2.5)

    log_alpha = layer._effective_log_alpha(torch.device("cpu"))
    from_exp = torch.exp(log_alpha)
    from_sigmoid = torch.sigmoid(log_alpha)
    pre_sigmoid = layer.alpha_log.squeeze(-1).float() - layer.gamma_bcm * torch.log(
        layer.running_state_norm.clamp(min=1e-6)
    )
    expected = torch.sigmoid(pre_sigmoid)

    diff_exp = (from_exp - expected).abs().max().item()
    diff_sig = (from_sigmoid - expected).abs().max().item()
    log(f"  |exp(logsigmoid) - sigmoid(pre)| = {diff_exp:.3e} (should be ~0)")
    log(f"  |sigmoid(logsigmoid) - sigmoid(pre)| = {diff_sig:.3e} (should be non-zero)")
    if diff_exp >= 1e-5:
        raise AssertionError(
            f"alpha_eff math regression: exp(logsigmoid) should equal sigmoid(pre), "
            f"got diff {diff_exp:.3e}. this is the F1 bug - recurrent path should use exp() "
            f"not sigmoid() on the log-sigmoid output of _effective_log_alpha."
        )
    if diff_sig < 1e-5:
        raise AssertionError(
            f"alpha_eff math sanity: sigmoid(logsigmoid) should NOT equal sigmoid(pre) "
            f"(they differ by ~30 percent at typical init). got diff {diff_sig:.3e}. "
            f"test itself is broken."
        )


def _test_delta_equivalence() -> None:
    log("recurrent self-consistency test (full-T vs chunked stepping)")
    torch.manual_seed(12345)
    test_cfg = Config(
        d_model=64,
        n_layers=1,
        delta_num_heads=2,
        delta_head_dim=32,
        attn_d_c=16,
        attn_d_R=8,
        attn_num_heads=2,
        mlp_ratio=2.0,
        layer_pattern=("DELTA",),
        kwta_enabled=False,
        delta_erasure_enabled=False,
        bcm_alpha_enabled=False,
        imagination_enabled=False,
        pc_diagnostic_enabled=False,
        multi_compartment_enabled=False,
        use_fla_if_available=False,
        batch_size=2,
        seq_len=32,
        max_steps=1,
        warmup_steps=1,
        val_interval=100,
        grad_checkpointing=False,
        amp=False,
    )
    layer = DeltaRuleMemory(test_cfg)
    layer.train(False)

    B, T = 2, 64
    x = torch.randn(B, T, test_cfg.d_model)

    with torch.no_grad():
        out_full, _, _ = layer(x)

        chunks: list[Tensor] = []
        state = None
        for t in range(T):
            single = x[:, t : t + 1, :]
            out_t, state, _ = layer(single, state=state, offset=t)
            chunks.append(out_t)
        out_step = torch.cat(chunks, dim=1)

    diff = (out_full - out_step).abs()
    max_abs = diff.max().item()
    ref = out_full.abs().max().item() + 1e-9
    rel = max_abs / ref
    if rel >= 1e-3:
        raise AssertionError(
            f"full-T and chunked recurrence paths diverge: max_abs={max_abs:.3e}, rel={rel:.3e}. "
            f"expected rel < 1e-3"
        )
    log(f"  full-T vs chunked: max_abs={max_abs:.2e}, rel={rel:.2e} ok")


def _test_resume_correctness() -> None:
    log("resume correctness test (train N steps, resume, verify batch sequence)")
    import shutil
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="god_resume_"))
    try:
        test_cfg = Config(
            d_model=32,
            n_layers=2,
            vocab_size=256,
            max_seq_len=32,
            delta_num_heads=2,
            delta_head_dim=16,
            attn_d_c=16,
            attn_d_R=8,
            attn_num_heads=2,
            mlp_ratio=2.0,
            layer_pattern=("DELTA", "ATTN"),
            kwta_enabled=False,
            delta_erasure_enabled=False,
            bcm_alpha_enabled=False,
            imagination_enabled=False,
            pc_diagnostic_enabled=False,
            multi_compartment_enabled=False,
            use_fla_if_available=False,
            batch_size=2,
            seq_len=16,
            max_steps=4,
            warmup_steps=1,
            val_interval=2,
            grad_checkpointing=False,
            amp=False,
            seed=999,
        )
        synth = bytes(range(256)) * 32
        train_data = synth[: len(synth) * 3 // 4]
        val_data = synth[len(synth) * 3 // 4 :]

        torch.manual_seed(test_cfg.seed)
        r1 = train_model(
            test_cfg,
            train_data,
            val_data,
            name="resume_a",
            output_dir=tmp_dir / "a",
            dataset_source="synthetic",
        )

        torch.manual_seed(test_cfg.seed)
        _ = train_model(
            test_cfg,
            train_data,
            val_data,
            name="resume_b",
            output_dir=tmp_dir / "b",
            dataset_source="synthetic",
            stop_after_steps=2,
        )
        os.environ["NM_RESUME"] = "1"
        try:
            r2 = train_model(
                test_cfg,
                train_data,
                val_data,
                name="resume_b",
                output_dir=tmp_dir / "b",
                dataset_source="synthetic",
            )
        finally:
            os.environ.pop("NM_RESUME", None)

        bpb_from_scratch = r1["best_val_bpb"]
        bpb_from_resume = r2["best_val_bpb"]
        if not math.isfinite(bpb_from_scratch) or not math.isfinite(bpb_from_resume):
            raise AssertionError(
                f"resume correctness test: bpb not finite "
                f"(scratch={bpb_from_scratch}, resume={bpb_from_resume})"
            )
        if abs(bpb_from_scratch - bpb_from_resume) > 1e-4:
            raise AssertionError(
                f"resume correctness test: bpb mismatch. scratch={bpb_from_scratch:.6f}, "
                f"resume={bpb_from_resume:.6f}, diff={abs(bpb_from_scratch - bpb_from_resume):.6f}. "
                f"expected scratch == resume within 1e-4. this indicates resume does not replay "
                f"the training trajectory deterministically (rng, data loader, or optimizer "
                f"state drift)."
            )
        scratch_best = torch.load(tmp_dir / "a" / "resume_a_best.pt", weights_only=True)
        resume_best = torch.load(tmp_dir / "b" / "resume_b_best.pt", weights_only=True)
        max_param_diff = 0.0
        for k in scratch_best:
            if k not in resume_best:
                raise AssertionError(f"resume correctness: key {k} missing from resume checkpoint")
            d = (scratch_best[k].float() - resume_best[k].float()).abs().max().item()
            if d > max_param_diff:
                max_param_diff = d
        if max_param_diff > 1e-4:
            raise AssertionError(
                f"resume correctness: max parameter diff {max_param_diff:.6e} > 1e-4. "
                f"resume does not produce bit-identical weights to from-scratch."
            )
        log(
            f"  resume correctness: scratch_bpb={bpb_from_scratch:.4f} "
            f"resume_bpb={bpb_from_resume:.4f} max_param_diff={max_param_diff:.2e}"
        )
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def smoke_test() -> None:
    log("running cpu smoke test")
    cfg = Config(
        d_model=64,
        n_layers=4,
        vocab_size=256,
        max_seq_len=64,
        delta_num_heads=2,
        delta_head_dim=32,
        attn_d_c=32,
        attn_d_R=8,
        attn_num_heads=2,
        mlp_ratio=2.0,
        layer_pattern=("DELTA", "DELTA", "DELTA", "ATTN"),
        batch_size=2,
        seq_len=32,
        max_steps=5,
        warmup_steps=2,
        val_interval=100,
        grad_checkpointing=False,
        amp=False,
    )

    torch.manual_seed(cfg.seed)
    model = GodMachine(cfg)
    n_params = model.count_params()
    log(f"smoke model params: {n_params:,}")

    input_ids = torch.randint(0, cfg.vocab_size, (2, 32))
    targets = torch.randint(0, cfg.vocab_size, (2, 32))

    model.train(True)
    logits, states, aux = model(input_ids)
    assert logits.shape == (2, 32, cfg.vocab_size), f"bad logits shape {logits.shape}"
    assert len(states) == cfg.n_layers, f"bad states length {len(states)}"
    assert len(aux) == cfg.n_layers, f"bad aux length {len(aux)}"

    main_loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
    pc_loss = aggregate_pc_loss(aux, device=main_loss.device)
    loss = main_loss + cfg.pc_lambda * pc_loss
    loss.backward()

    missing = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is None:
            missing.append(name)
    if missing:
        raise RuntimeError(f"no grad for params: {missing[:5]} ({len(missing)} total)")
    log(
        f"smoke test forward+backward ok. main_loss={main_loss.item():.4f} "
        f"pc_loss={pc_loss.item() if isinstance(pc_loss, Tensor) else float(pc_loss):.4f}"
    )

    probe_feature_assertions(aux, cfg)
    log("smoke test per-feature probes ok")

    logged_metrics = collect_spike_stats(aux)
    expected_aux_keys: set[str] = set()
    for la in aux.values():
        if isinstance(la, dict):
            for k_a, v_a in la.items():
                if isinstance(v_a, Tensor):
                    if v_a.numel() == 1 or v_a.numel() <= 64:
                        expected_aux_keys.add(k_a)
                elif isinstance(v_a, (int, float)):
                    expected_aux_keys.add(k_a)
    required_god_keys: list[str] = ["mean_fr", "per_layer_mean_fr", "layer_indices"]
    for k_a in expected_aux_keys:
        required_god_keys.append(f"{k_a}_per_layer")
    missing_god = [k for k in required_god_keys if k not in logged_metrics]
    if missing_god:
        raise RuntimeError(
            f"collect_spike_stats dropped enabled-feature metrics: {missing_god}. "
            f"available keys: {sorted(logged_metrics.keys())}"
        )
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tf:
        tmp_path = Path(tf.name)
    try:
        tmp_logger = MetricsLogger(tmp_path)
        step_record: dict[str, Any] = {"event": "step", "step": 0}
        for k, v in logged_metrics.items():
            if k not in step_record:
                step_record[k] = v
        tmp_logger.log(step_record)
        tmp_logger.close()
        with open(tmp_path) as f:
            roundtrip = json.loads(f.readline())
        missing_disk = [k for k in required_god_keys if k not in roundtrip]
        if missing_disk:
            raise RuntimeError(
                f"god metrics computed but not written to jsonl: {missing_disk}. "
                f"disk keys: {sorted(roundtrip.keys())}"
            )
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass
    log(
        f"smoke test jsonl round-trip ok "
        f"({len(required_god_keys)} keys derived from aux, verified on disk)"
    )

    model.train(False)
    single = torch.randint(0, cfg.vocab_size, (1, 1))
    logits_s, states_s, _ = model(single)
    assert logits_s.shape == (1, 1, cfg.vocab_size), f"bad single logits {logits_s.shape}"
    for i, (layer_type, s) in enumerate(zip(cfg.layer_types, states_s)):
        if layer_type == "DELTA":
            assert s is not None, f"delta layer {i} returned None state in recurrent mode"
        elif layer_type == "ATTN":
            assert s is not None, f"attn layer {i} returned None cache"
        elif layer_type == "SLOT":
            assert isinstance(s, dict), f"slot layer {i} returned non-dict state {type(s)}"
            for required_key in ("slot_state", "prev_hidden"):
                assert required_key in s, f"slot layer {i} missing state key {required_key}"
    log("smoke test single-token recurrent ok")

    prompt = b"hello"
    gen = greedy_generate(model, prompt, max_new_tokens=8)
    assert len(gen) == 8, f"greedy_generate returned {len(gen)} bytes, expected 8"
    log(f"smoke test greedy_generate ok (len={len(gen)})")

    pk = passkey_retrieval_test(model, context_length=64, num_trials=2, passkey_length=3)
    assert 0.0 <= pk["accuracy"] <= 1.0, f"passkey accuracy out of range: {pk}"
    log(f"smoke test passkey_retrieval ok (untrained acc={pk['accuracy']:.2f})")

    cp = selective_copy_test(model, context_length=64, num_trials=2, copy_length=4)
    assert 0.0 <= cp["accuracy"] <= 1.0, f"copy accuracy out of range: {cp}"
    log(f"smoke test selective_copy ok (untrained acc={cp['accuracy']:.2f})")

    synth_bytes = bytes(range(64)) * 5
    ppl = measure_perplexity_at_length(model, synth_bytes, context_length=64, num_windows=3)
    assert math.isfinite(ppl["bpb"]), f"bpb not finite: {ppl}"
    log(f"smoke test measure_perplexity_at_length ok (bpb={ppl['bpb']:.2f})")

    probe = run_delta_state_structure_probe(model, synth_bytes, n_warmup=32, n_probes=8)
    assert "per_layer" in probe and isinstance(probe["per_layer"], dict), f"probe missing per_layer: {probe}"
    if probe["per_layer"]:
        log(
            f"smoke test delta_state_structure_probe ok "
            f"(structure_ratio={probe['mean_structure_ratio']:.3f} "
            f"pairwise_cos={probe['mean_pairwise_cos']:.3f} "
            f"random_pairwise_cos={probe['random_mean_pairwise_cos']:.3f})"
        )
    else:
        log("smoke test delta_state_structure_probe ok (no delta state populated in tiny config)")

    _test_alpha_eff_math()
    _test_delta_equivalence()
    _test_fla_vs_recurrent_parity()
    _test_slot_memory_parity()
    _test_resume_correctness()

    _smoke_preset_baseline("run4_erasure_ablation")
    _smoke_preset_baseline("run1_baseline_noerasure")
    _smoke_preset_baseline("run1a_retention_ablation")
    _smoke_preset_baseline("run2_slot_memory")

    log("all smoke checks passed")


def _resolve_preset(preset: str) -> tuple[dict[str, Any], str, str]:
    if preset == "run4_erasure_ablation":
        return (
            {
                "kwta_enabled": False,
                "delta_erasure_enabled": True,
                "use_fla_if_available": False,
                "bcm_alpha_enabled": False,
                "multi_compartment_enabled": False,
                "imagination_enabled": False,
                "pc_diagnostic_enabled": False,
            },
            "preset: run4_erasure_ablation (dense k/v, delta erasure on, non-FLA path, all other features off)",
            "run4_erasure_ablation",
        )
    if preset == "run1_baseline_noerasure":
        return (
            {
                "kwta_enabled": False,
                "delta_erasure_enabled": False,
                "use_fla_if_available": True,
                "bcm_alpha_enabled": False,
                "multi_compartment_enabled": False,
                "imagination_enabled": False,
                "pc_diagnostic_enabled": False,
            },
            "preset: run1_baseline_noerasure (dense k/v, no overwrite subtraction, FLA simple_gla path, all other features off)",
            "run1_baseline_noerasure",
        )
    if preset == "run1a_retention_ablation":
        return (
            {
                "kwta_enabled": False,
                "delta_erasure_enabled": False,
                "use_fla_if_available": False,
                "bcm_alpha_enabled": False,
                "multi_compartment_enabled": False,
                "imagination_enabled": False,
                "pc_diagnostic_enabled": False,
                "alpha_log_mean": 2.2,
            },
            "preset: run1a_retention_ablation (dense k/v, no overwrite subtraction, non-FLA path, slower static retention target ~0.90, all other features off)",
            "run1a_retention_ablation",
        )
    if preset == "run2_slot_memory":
        slot_pattern = tuple(
            "SLOT" if lt == "DELTA" else lt
            for lt in ("DELTA", "DELTA", "DELTA", "DELTA", "DELTA", "DELTA", "ATTN")
        )
        return (
            {
                "kwta_enabled": False,
                "delta_erasure_enabled": False,
                "use_fla_if_available": False,
                "bcm_alpha_enabled": False,
                "multi_compartment_enabled": False,
                "imagination_enabled": False,
                "pc_diagnostic_enabled": False,
                "layer_pattern": slot_pattern,
            },
            "preset: run2_slot_memory (slot memory with softmax addressing replacing delta layers, compressed attention preserved, all other features off)",
            "run2_slot_memory",
        )
    if preset == "god":
        return ({}, "preset: god (default god_machine config)", "god_run")
    raise ValueError(f"unknown NM_PRESET: {preset}")


def _run1_benchmark_manifest_path(output_dir: Path) -> Path:
    return output_dir.parent / "run1_baseline_noerasure_benchmark_manifest.json"


def _run1_benchmark_contract_cfg() -> Config:
    preset_overrides, _, _ = _resolve_preset("run1_baseline_noerasure")
    cfg = Config(**preset_overrides)
    cfg.batch_size = 16
    cfg.seq_len = 2048
    cfg.max_seq_len = 2048
    cfg.max_steps = 20
    return cfg


def _run1_benchmark_contract_hash() -> str:
    return _config_hash(_cfg_to_dict(_run1_benchmark_contract_cfg()))


def _run1_full_contract_hash() -> str:
    preset_overrides, _, _ = _resolve_preset("run1_baseline_noerasure")
    return _config_hash(_cfg_to_dict(Config(**preset_overrides)))


def _load_json_dict(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a json object at {path}, got {type(payload).__name__}")
    return payload


def _require_run1_benchmark_manifest(output_dir: Path) -> dict[str, Any]:
    manifest_path = _run1_benchmark_manifest_path(output_dir)
    if not manifest_path.exists():
        raise RuntimeError(
            f"full run1_baseline_noerasure launch requires benchmark manifest {manifest_path}. "
            "run the documented 20-step FineWeb benchmark first in the same output root."
        )
    payload = _load_json_dict(manifest_path)
    if payload.get("preset") != "run1_baseline_noerasure":
        raise RuntimeError(f"benchmark manifest {manifest_path} does not belong to run1_baseline_noerasure")
    if payload.get("status") != "completed":
        raise RuntimeError(f"benchmark manifest {manifest_path} is not marked completed")
    if payload.get("config_hash") != _run1_benchmark_contract_hash():
        raise RuntimeError(
            f"benchmark manifest {manifest_path} does not match the required run1 benchmark config hash"
        )
    if payload.get("dataset_mode") != "fineweb":
        raise RuntimeError(f"benchmark manifest {manifest_path} was not produced with NM_DATASET=fineweb")
    dataset_source = str(payload.get("dataset_source", ""))
    if not dataset_source.startswith("fineweb-edu"):
        raise RuntimeError(
            f"benchmark manifest {manifest_path} recorded dataset_source={dataset_source!r}, expected fineweb-edu"
        )
    if int(payload.get("total_steps", -1)) != 20:
        raise RuntimeError(f"benchmark manifest {manifest_path} did not complete the required 20 steps")
    if payload.get("device") != "cuda":
        raise RuntimeError(f"benchmark manifest {manifest_path} was not produced on cuda")
    cap_raw = payload.get("device_capability")
    if not isinstance(cap_raw, list) or len(cap_raw) != 2:
        raise RuntimeError(f"benchmark manifest {manifest_path} is missing a valid cuda capability")
    cap = _device_capability_tuple(cap_raw)
    if cap < (9, 0):
        raise RuntimeError(
            f"benchmark manifest {manifest_path} was produced on sm_{cap[0]}{cap[1]}, expected sm_90 or newer"
        )
    benchmark_output_dir = Path(str(payload.get("output_dir", "")))
    benchmark_output_dir_lexical = _require_lexical_path(benchmark_output_dir, "benchmark output dir")
    benchmark_output_dir_resolved = benchmark_output_dir_lexical.resolve(strict=False)
    manifest_root = _require_lexical_path(manifest_path.parent, "shared output root")
    if not _same_path(benchmark_output_dir_lexical.parent, manifest_root):
        raise RuntimeError(
            f"benchmark manifest {manifest_path} references output dir {benchmark_output_dir_lexical} outside the shared output root"
        )
    run_name = str(payload.get("run_name", ""))
    artifact_hashes = payload.get("artifact_hashes")
    if not run_name:
        raise RuntimeError(f"benchmark manifest {manifest_path} is missing run_name")
    _validate_run_name(run_name)
    if not isinstance(artifact_hashes, dict):
        raise RuntimeError(f"benchmark manifest {manifest_path} is missing artifact hashes")
    metadata_path = _require_path_under(benchmark_output_dir_resolved, benchmark_output_dir_lexical / f"{run_name}_metadata.json", "benchmark metadata")
    results_path = _require_path_under(benchmark_output_dir_resolved, benchmark_output_dir_lexical / f"{run_name}_results.json", "benchmark results")
    metrics_path = _require_path_under(benchmark_output_dir_resolved, benchmark_output_dir_lexical / f"{run_name}_metrics.jsonl", "benchmark metrics")
    for suffix in ("metadata", "results", "metrics"):
        expected_hash = artifact_hashes.get(suffix)
        if not isinstance(expected_hash, str) or not expected_hash:
            raise RuntimeError(f"benchmark manifest {manifest_path} is missing {suffix} hash")
        artifact_path = metrics_path if suffix == "metrics" else (metadata_path if suffix == "metadata" else results_path)
        if not artifact_path.exists():
            raise RuntimeError(
                f"benchmark manifest {manifest_path} references missing artifact {artifact_path}"
            )
        if _sha256_file(artifact_path) != expected_hash:
            raise RuntimeError(
                f"benchmark manifest {manifest_path} hash mismatch for {artifact_path}"
            )
    metadata_payload = _load_json_dict(metadata_path)
    results_payload = _load_json_dict(results_path)
    if metadata_payload.get("run") != run_name:
        raise RuntimeError(f"benchmark metadata {metadata_path} does not match run_name {run_name}")
    if metadata_payload.get("config_hash") != payload.get("config_hash"):
        raise RuntimeError(f"benchmark metadata {metadata_path} does not match manifest config hash")
    if metadata_payload.get("dataset_source") != payload.get("dataset_source"):
        raise RuntimeError(f"benchmark metadata {metadata_path} does not match manifest dataset source")
    if results_payload.get("run") != run_name:
        raise RuntimeError(f"benchmark results {results_path} do not match run_name {run_name}")
    if results_payload.get("current_step") != payload.get("total_steps"):
        raise RuntimeError(f"benchmark results {results_path} do not match manifest total_steps")
    if results_payload.get("final") is not True:
        raise RuntimeError(f"benchmark results {results_path} are not a final snapshot")
    results_metadata = results_payload.get("metadata")
    if not isinstance(results_metadata, dict) or results_metadata.get("config_hash") != payload.get("config_hash"):
        raise RuntimeError(f"benchmark results {results_path} do not embed the manifest config hash")
    manifest_fingerprint = payload.get("git_state_fingerprint")
    current_git = _capture_git_state_fingerprint(
        ignore_paths={
            manifest_path,
            output_dir,
            output_dir.parent / f"{output_dir.name}.stdout.log",
            benchmark_output_dir.parent / f"{run_name}.stdout.log",
        }
    )
    if current_git.get("error") is not None or current_git.get("fingerprint") is None:
        raise RuntimeError(
            f"benchmark manifest {manifest_path} cannot validate current git state: {current_git.get('error')}"
        )
    if not isinstance(manifest_fingerprint, str) or not manifest_fingerprint:
        raise RuntimeError(f"benchmark manifest {manifest_path} is missing git_state_fingerprint")
    if manifest_fingerprint != current_git.get("fingerprint"):
        raise RuntimeError(
            f"benchmark manifest {manifest_path} does not match the current git working-tree fingerprint"
        )
    return payload


def _require_run1_full_output_dir(output_dir: Path, benchmark_manifest: dict[str, Any]) -> Path:
    shared_root = _require_lexical_path(_run1_benchmark_manifest_path(output_dir).parent, "shared output root")
    try:
        output_dir_lexical = _require_lexical_path(output_dir, "full-run output dir")
        output_dir_resolved = _require_path_under(shared_root, output_dir_lexical, "full-run output dir")
    except RuntimeError as e:
        raise RuntimeError(
            f"full-run output dir {Path(os.path.abspath(os.fspath(output_dir)))} is outside the shared output root {shared_root}"
        ) from e
    benchmark_output_dir = _require_lexical_path(Path(str(benchmark_manifest.get("output_dir", ""))), "benchmark output dir")
    if not _same_path(output_dir_lexical.parent, benchmark_output_dir.parent):
        raise RuntimeError(
            f"full-run output dir {output_dir_resolved} is outside the shared output root {benchmark_output_dir.parent}"
        )
    return output_dir_lexical


def _require_run1_full_launch_device(benchmark_manifest: dict[str, Any]) -> None:
    current_device = _device_metadata()
    current_cap = _device_capability_tuple(current_device.get("device_capability"))
    manifest_cap = _device_capability_tuple(benchmark_manifest.get("device_capability"))
    if current_device.get("device") != benchmark_manifest.get("device"):
        raise RuntimeError(
            "full run1_baseline_noerasure launch must run on the same device class recorded by the benchmark manifest."
        )
    if current_device.get("device") != "cuda" or current_cap < (9, 0):
        raise RuntimeError(
            "full run1_baseline_noerasure launch requires an sm_90-or-newer cuda device."
        )
    if current_cap != manifest_cap:
        raise RuntimeError(
            f"full run1_baseline_noerasure launch device sm_{current_cap[0]}{current_cap[1]} does not match "
            f"benchmark manifest sm_{manifest_cap[0]}{manifest_cap[1]}."
        )
    current_memory = current_device.get("device_total_memory_gb")
    manifest_memory = benchmark_manifest.get("device_total_memory_gb")
    if current_memory != manifest_memory:
        raise RuntimeError(
            f"full run1_baseline_noerasure launch memory profile {current_memory} does not match benchmark manifest {manifest_memory}."
        )
    current_sms = current_device.get("device_multi_processor_count")
    manifest_sms = benchmark_manifest.get("device_multi_processor_count")
    if current_sms != manifest_sms:
        raise RuntimeError(
            f"full run1_baseline_noerasure launch sm count {current_sms} does not match benchmark manifest {manifest_sms}."
        )


def _write_run1_benchmark_manifest(
    output_dir: Path,
    run_name: str,
    cfg: Config,
    dataset_mode: str,
    dataset_source: str,
    total_steps: int,
) -> Path:
    manifest_path = _run1_benchmark_manifest_path(output_dir)
    metadata_path = output_dir / f"{run_name}_metadata.json"
    results_path = output_dir / f"{run_name}_results.json"
    metrics_path = output_dir / f"{run_name}_metrics.jsonl"
    git = _capture_git_metadata()
    git_state = _capture_git_state_fingerprint(
        ignore_paths={
            manifest_path,
            output_dir.parent / f"{run_name}.stdout.log",
        }
    )
    if git_state.get("error") is not None or git_state.get("fingerprint") is None:
        raise RuntimeError(f"unable to capture git working-tree fingerprint for benchmark manifest: {git_state.get('error')}")
    payload = {
        "preset": "run1_baseline_noerasure",
        "status": "completed",
        "config_hash": _config_hash(_cfg_to_dict(cfg)),
        "dataset_mode": dataset_mode,
        "dataset_source": dataset_source,
        "total_steps": int(total_steps),
        "output_dir": str(output_dir),
        "run_name": run_name,
        "completed_at": datetime.now().isoformat(),
        "git_sha": git.get("sha"),
        "git_dirty": git.get("dirty"),
        "git_state_fingerprint": git_state.get("fingerprint"),
        "artifact_hashes": {
            "metadata": _sha256_file(metadata_path),
            "results": _sha256_file(results_path),
            "metrics": _sha256_file(metrics_path),
        },
        **_device_metadata(),
    }
    _write_json_atomic(manifest_path, payload)
    return manifest_path


def _smoke_preset_baseline(preset_name: str) -> None:
    preset_overrides, _, _ = _resolve_preset(preset_name)
    erasure_enabled = bool(preset_overrides.get("delta_erasure_enabled", False))
    preset_desc = "erasure on" if erasure_enabled else "no erasure"
    log(f"smoke test {preset_name} preset (all features off, dense k/v, {preset_desc})")
    base_kwargs: dict[str, Any] = dict(
        d_model=64,
        n_layers=4,
        vocab_size=256,
        max_seq_len=64,
        delta_num_heads=2,
        delta_head_dim=32,
        attn_d_c=32,
        attn_d_R=8,
        attn_num_heads=2,
        mlp_ratio=2.0,
        batch_size=2,
        seq_len=32,
        max_steps=5,
        warmup_steps=2,
        val_interval=100,
        grad_checkpointing=False,
        amp=False,
        slot_num_slots=8,
    )
    if "layer_pattern" not in preset_overrides:
        base_kwargs["layer_pattern"] = ("DELTA", "DELTA", "DELTA", "ATTN")
    base_kwargs.update(preset_overrides)
    preset_pattern = base_kwargs["layer_pattern"]
    base_kwargs["n_layers"] = len(preset_pattern)
    cfg = Config(**base_kwargs)
    torch.manual_seed(cfg.seed)
    model = GodMachine(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 32))
    targets = torch.randint(0, cfg.vocab_size, (2, 32))
    model.train(True)
    logits, states, aux = model(input_ids)
    assert logits.shape == (2, 32, cfg.vocab_size)
    main_loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
    pc_loss = aggregate_pc_loss(aux, device=main_loss.device)
    assert pc_loss.device == main_loss.device, (
        f"aggregate_pc_loss device mismatch when pc_diagnostic disabled: "
        f"pc_loss={pc_loss.device} vs main_loss={main_loss.device}"
    )
    loss = main_loss + cfg.pc_lambda * pc_loss
    loss.backward()
    stats = collect_spike_stats(aux)
    if stats.get("mean_fr", None) != 0.0:
        raise AssertionError(
            f"{preset_name}: mean_fr should be 0.0 with kwta disabled, got {stats.get('mean_fr')}"
        )
    for forbidden_key in ("kwta_k_rate_per_layer", "kwta_v_rate_per_layer", "imag_gate_mean_per_layer", "imag_contribution_l2_per_layer", "imag_ratio_per_layer", "pc_error_l2_per_layer", "mlp_compartment_l2_per_layer"):
        if forbidden_key in stats:
            raise AssertionError(
                f"{preset_name}: aux should not contain {forbidden_key} when the "
                f"corresponding feature is disabled. found: {stats[forbidden_key]}"
            )
    required_keys = ("alpha_base_mean_per_layer", "alpha_eff_mean_per_layer", "beta_mean_per_layer", "state_frobenius_per_layer", "delta_path_per_layer", "delta_erasure_flag_per_layer")
    for k in required_keys:
        if k not in stats:
            raise AssertionError(
                f"{preset_name}: expected stats key {k} not present. "
                f"available: {sorted(stats.keys())[:20]}"
            )
    erasure_flags = stats["delta_erasure_flag_per_layer"]
    expected_flag = 1 if erasure_enabled else 0
    realized_flags = [flag for flag in erasure_flags if flag is not None]
    if not realized_flags:
        raise AssertionError(f"{preset_name}: no delta erasure flags were recorded")
    if any(flag != expected_flag for flag in realized_flags):
        raise AssertionError(
            f"{preset_name}: expected delta_erasure_flag_per_layer={expected_flag}, got {erasure_flags}"
        )
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tf:
        tmp_path = Path(tf.name)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as rf:
        results_path = Path(rf.name)
    try:
        tmp_logger = MetricsLogger(tmp_path)
        step_record: dict[str, Any] = {"event": "step", "step": 0}
        for k, v in stats.items():
            if k not in step_record:
                step_record[k] = v
        tmp_logger.log(step_record)
        tmp_logger.close()
        with open(tmp_path) as f:
            roundtrip = json.loads(f.readline())
        for k in required_keys:
            if k not in roundtrip:
                raise AssertionError(
                    f"{preset_name}: key {k} computed but not written to jsonl. "
                    f"disk keys: {sorted(roundtrip.keys())[:20]}"
                )
        gate_record = {
            "metric": "passkey_256",
            "threshold": "> 0.0",
            "is_hard_gate": True,
            "preset": preset_name,
            "status": "pass",
            "passkey_256_accuracy": 0.25,
        }
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({"run": preset_name}, f, indent=2)
        _persist_retrieval_gate(results_path, gate_record)
        with open(results_path, encoding="utf-8") as f:
            results_payload = json.load(f)
        if results_payload.get("retrieval_gate") != gate_record:
            raise AssertionError(
                f"{preset_name}: retrieval_gate did not roundtrip through results.json. "
                f"payload keys: {sorted(results_payload.keys())}"
            )
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass
        try:
            results_path.unlink()
        except Exception:
            pass
    log(f"smoke test {preset_name} preset ok (mean_fr=0.0, {len(required_keys)} keys roundtripped, pc_loss device={pc_loss.device})")


def _persist_retrieval_gate(results_path: Path, gate_record: dict[str, Any]) -> None:
    payload: dict[str, Any]
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {}
    payload["retrieval_gate"] = _json_safe(gate_record)
    _write_json_atomic(results_path, payload)


def _require_completed_retrieval_gate(
    gate_record: dict[str, Any] | None,
    skip_eval: bool,
    require_pass: bool = False,
) -> None:
    if gate_record is None or skip_eval:
        return
    status = str(gate_record.get("status", "not_evaluated"))
    if status == "pass":
        return
    if status == "fail":
        if not require_pass:
            return
        metric = str(gate_record.get("metric", "retrieval gate"))
        accuracy = gate_record.get("passkey_256_accuracy")
        detail = metric
        if isinstance(accuracy, (int, float)):
            detail = f"{metric}={float(accuracy):.1%}"
        raise RuntimeError(
            f"retrieval gate failed ({detail}). official run1_baseline_noerasure full launches require retrieval_gate.status == 'pass'."
        )
    detail = gate_record.get("reason") or gate_record.get("error") or status
    raise RuntimeError(
        f"retrieval gate did not complete ({status}: {detail}). "
        "official baseline-style launches require a loadable best checkpoint and a completed eval suite."
    )


def main() -> None:
    if _env_flag("SMOKE_TEST"):
        smoke_test()
        return

    preset = os.environ.get("NM_PRESET", "run1_baseline_noerasure")
    preset_overrides, preset_log, default_run_name = _resolve_preset(preset)
    output_dir_env = os.environ.get("NM_OUTPUT_DIR")
    output_dir_lexical, output_dir = _canonicalize_output_pair(Path(output_dir_env or str(DEFAULT_OUTPUT_ROOT / default_run_name)))
    cfg = Config(**preset_overrides)
    run_name_env = os.environ.get("NM_RUN_NAME")
    run_name = _validate_run_name(run_name_env or default_run_name)
    runtime_overrides = _apply_runtime_env_overrides(cfg)
    log(preset_log)
    if runtime_overrides:
        log(f"runtime overrides: {', '.join(runtime_overrides)}")
    official_run1 = preset == "run1_baseline_noerasure"
    gated_baseline_presets = {"run1_baseline_noerasure", "run1a_retention_ablation", "run4_erasure_ablation"}
    launch_contract: dict[str, Any] | None = None
    benchmark_mode = (
        official_run1
        and cfg.max_steps <= 20
        and _env_flag("NM_SKIP_VALIDATION")
        and _env_flag("NM_SKIP_EVAL")
    )
    if benchmark_mode:
        if run_name_env is None:
            run_name = "run1_baseline_noerasure_bench"
        if output_dir_env is None:
            output_dir_lexical, output_dir = _canonicalize_output_pair(DEFAULT_OUTPUT_ROOT / run_name)
    if benchmark_mode and _env_flag("NM_RESUME"):
        raise RuntimeError(
            "benchmark mode requires a fresh directory and does not allow NM_RESUME=1. "
            "rerun the documented 20-step benchmark from step 0 before authorizing the full launch."
        )
    if official_run1 and _env_flag("ALLOW_WIKITEXT_FALLBACK"):
        raise RuntimeError(
            "ALLOW_WIKITEXT_FALLBACK=1 is forbidden on the official run1_baseline_noerasure surface. "
            "use a separately named fallback or debug run instead of the canonical FineWeb branch."
        )
    if preset in gated_baseline_presets and not benchmark_mode and (
        _env_flag("NM_SKIP_VALIDATION") or _env_flag("NM_SKIP_EVAL")
    ):
        raise RuntimeError(
            "NM_SKIP_VALIDATION and NM_SKIP_EVAL are benchmark-only flags on the baseline launch surface. "
            "full launches must keep validation and eval enabled so the retrieval gate is real."
        )
    if official_run1 and not benchmark_mode and not _env_flag("NM_AUTHORIZE_FULL_RUN"):
        raise RuntimeError(
            "full run1_baseline_noerasure launch requires NM_AUTHORIZE_FULL_RUN=1 after the benchmark gate has been reviewed."
        )
    if official_run1 and not benchmark_mode:
        if _config_hash(_cfg_to_dict(cfg)) != _run1_full_contract_hash():
            raise RuntimeError(
                "official run1_baseline_noerasure full launches must use the canonical 4000-step config with no runtime overrides."
            )
        manifest_path = _run1_benchmark_manifest_path(output_dir_lexical)
        benchmark_manifest = _require_run1_benchmark_manifest(output_dir_lexical)
        output_dir_lexical = _require_run1_full_output_dir(output_dir_lexical, benchmark_manifest)
        output_dir = output_dir_lexical.resolve(strict=False)
        _require_run1_full_launch_device(benchmark_manifest)
        launch_contract = {
            "preset": preset,
            "benchmark_manifest_sha256": _sha256_file(manifest_path),
            "git_state_fingerprint": benchmark_manifest.get("git_state_fingerprint"),
            "device": benchmark_manifest.get("device"),
            "device_capability": benchmark_manifest.get("device_capability"),
            "device_total_memory_gb": benchmark_manifest.get("device_total_memory_gb"),
            "device_multi_processor_count": benchmark_manifest.get("device_multi_processor_count"),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"device: {DEVICE}")
    log(f"output: {output_dir_lexical}")
    log(
        f"config: d_model={cfg.d_model} n_layers={cfg.n_layers} "
        f"seq_len={cfg.seq_len} batch_size={cfg.batch_size}"
    )
    log(
        f"features: kwta={cfg.kwta_enabled} erasure={cfg.delta_erasure_enabled} "
        f"bcm={cfg.bcm_alpha_enabled} multicomp={cfg.multi_compartment_enabled} "
        f"imag={cfg.imagination_enabled} pc_diag={cfg.pc_diagnostic_enabled}"
    )
    log(f"layer pattern: {cfg.layer_pattern}")

    log("loading training data")
    dataset_mode = os.environ.get("NM_DATASET", "auto").strip().lower()
    if dataset_mode == "auto":
        dataset_mode = "wikitext2" if cfg.max_steps <= 50 else "fineweb"
    if official_run1 and dataset_mode != "fineweb":
        raise RuntimeError(
            "official run1_baseline_noerasure launches must use the FineWeb path. set NM_DATASET=fineweb."
        )
    if benchmark_mode:
        if dataset_mode != "fineweb":
            raise RuntimeError(
                "benchmark mode requires NM_DATASET=fineweb so the prelaunch gate exercises the real corpus path."
            )
        if cfg.batch_size != 16 or cfg.seq_len != 2048 or cfg.max_seq_len != 2048:
            raise RuntimeError(
                "benchmark mode requires the full launch shape: batch_size=16, seq_len=2048, max_seq_len=2048."
            )
        if _config_hash(_cfg_to_dict(cfg)) != _run1_benchmark_contract_hash():
            raise RuntimeError(
                "benchmark mode requires the exact documented run1 benchmark config; remove extra runtime overrides."
            )
        device_info = _device_metadata()
        cap = _device_capability_tuple(device_info.get("device_capability"))
        if device_info.get("device") != "cuda" or cap < (9, 0):
            raise RuntimeError(
                "benchmark mode requires an sm_90-or-newer cuda device so the prelaunch gate measures the real h200 path."
            )
    if dataset_mode == "wikitext2":
        train_data, val_data, dataset_source = download_wikitext2()
    elif dataset_mode == "fineweb":
        train_data, val_data, dataset_source = download_fineweb_edu(
            max_bytes=5_000_000_000,
            allow_wikitext_fallback=_env_flag("ALLOW_WIKITEXT_FALLBACK"),
        )
    else:
        raise ValueError(f"unknown NM_DATASET: {dataset_mode}")
    log(f"dataset: {dataset_source}  train: {len(train_data):,} bytes, val: {len(val_data):,} bytes")

    result = train_model(
        cfg,
        train_data,
        val_data,
        name=run_name,
        output_dir=output_dir,
        dataset_source=dataset_source,
        launch_contract=launch_contract,
    )

    log(f"results.json snapshot at {output_dir / f'{run_name}_results.json'}")
    log(f"metrics stream at {output_dir / f'{run_name}_metrics.jsonl'}")
    log(f"metadata at {output_dir / f'{run_name}_metadata.json'}")
    log(
        f"final best_val_bpb={result['best_val_bpb']:.4f} "
        f"steps={result['total_steps']} tokens={result['tokens_seen_total']:,}"
    )

    gate_record: dict[str, Any] | None = None
    pending_eval_failure: RuntimeError | None = None
    if preset in ("run1_baseline_noerasure", "run1a_retention_ablation", "run4_erasure_ablation"):
        gate_record = {
            "metric": "passkey_256",
            "threshold": "> 0.0",
            "is_hard_gate": True,
            "preset": preset,
            "status": "not_evaluated",
        }

    if _env_flag("NM_SKIP_EVAL"):
        log("NM_SKIP_EVAL set, skipping eval suite")
        if gate_record is not None:
            gate_record["reason"] = "NM_SKIP_EVAL"
    else:
        best_path = output_dir / f"{run_name}_best.pt"
        if not best_path.exists():
            log(f"eval suite skipped: {best_path} does not exist (no best checkpoint written)")
            if gate_record is not None:
                gate_record["reason"] = "best_checkpoint_missing"
        else:
            log("loading best checkpoint for eval suite")
            eval_model = GodMachine(cfg).to(DEVICE)
            checkpoint_stage = "load"
            try:
                best_state = torch.load(best_path, map_location=DEVICE, weights_only=True)
                incompat = eval_model.load_state_dict(best_state, strict=False)
                missing = list(incompat.missing_keys) if hasattr(incompat, "missing_keys") else []
                unexpected = list(incompat.unexpected_keys) if hasattr(incompat, "unexpected_keys") else []
                if missing or unexpected:
                    log(
                        f"checkpoint load mismatch: "
                        f"missing={missing[:10]}{'...' if len(missing) > 10 else ''} "
                        f"unexpected={unexpected[:10]}{'...' if len(unexpected) > 10 else ''}"
                    )
                    message = (
                        f"checkpoint load mismatch: {len(missing)} missing keys and {len(unexpected)} unexpected keys. "
                        f"refusing to compute retrieval_gate on a checkpoint that does not exactly match the active preset."
                    )
                    if gate_record is not None:
                        gate_record["status"] = "checkpoint_mismatch"
                        gate_record["error"] = message
                    pending_eval_failure = RuntimeError(message)
                log(f"loaded {best_path}")
                checkpoint_stage = "eval"
                if pending_eval_failure is None:
                    eval_results = run_eval_suite(eval_model, val_data, output_dir, run_name)
                    if gate_record is not None:
                        passkey_256 = float(eval_results["passkey"]["256"]["accuracy"])
                        gate_record["passkey_256_accuracy"] = passkey_256
                        gate_record["status"] = "pass" if passkey_256 > 0.0 else "fail"
                        log(f"retrieval gate: {gate_record['status']} (passkey@256={passkey_256:.1%})")
                    log(
                        f"eval suite: passkey@256={eval_results['passkey']['256']['accuracy']:.1%} "
                        f"copy@256={eval_results['selective_copy']['256']['accuracy']:.1%} "
                        f"imag_structure={eval_results['delta_state_structure_probe'].get('mean_structure_ratio', 0):.3f}"
                    )
            except Exception as e:
                log(f"eval suite runtime failure: {type(e).__name__}: {e}")
                if gate_record is not None:
                    gate_record["status"] = "checkpoint_load_error" if checkpoint_stage == "load" else "eval_error"
                    gate_record["error"] = f"{type(e).__name__}: {e}"
                pending_eval_failure = RuntimeError(f"eval suite runtime failure: {type(e).__name__}: {e}")
            finally:
                del eval_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if gate_record is not None:
        _persist_retrieval_gate(output_dir / f"{run_name}_results.json", gate_record)

    if pending_eval_failure is not None:
        raise pending_eval_failure

    _require_completed_retrieval_gate(
        gate_record,
        skip_eval=_env_flag("NM_SKIP_EVAL"),
        require_pass=official_run1 and not benchmark_mode,
    )

    if benchmark_mode:
        manifest_path = _write_run1_benchmark_manifest(
            output_dir=output_dir_lexical,
            run_name=run_name,
            cfg=cfg,
            dataset_mode=dataset_mode,
            dataset_source=dataset_source,
            total_steps=result["total_steps"],
        )
        log(f"benchmark manifest at {manifest_path}")

    log("done")


if __name__ == "__main__":
    main()
