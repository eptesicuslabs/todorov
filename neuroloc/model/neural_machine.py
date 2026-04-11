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
from dataclasses import asdict, dataclass, field
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
SEED = 42
DEFAULT_OUTPUT = Path("/kaggle/working") if Path("/kaggle").exists() else Path("neuroloc/output/run1_baseline")


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
    alpha_log_mean: float = -2.0
    alpha_log_std: float = 0.5

    attn_d_c: int = 128
    attn_d_R: int = 32
    attn_num_heads: int = 16

    mlp_ratio: float = 2.75
    layer_pattern: tuple[str, ...] = ("DELTA", "DELTA", "DELTA", "DELTA", "DELTA", "DELTA", "ATTN")

    use_spikes: bool = True
    spike_alpha_init: float = 1.0
    spike_min_threshold: float = 0.01
    spike_max_threshold: float = 10.0

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

    def forward(self, x: Tensor) -> Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)


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

        if cfg.use_spikes:
            self.k_spike = AdaptiveSpike(
                cfg.spike_alpha_init, cfg.spike_min_threshold, cfg.spike_max_threshold
            )
            self.v_spike = AdaptiveSpike(
                cfg.spike_alpha_init, cfg.spike_min_threshold, cfg.spike_max_threshold
            )
        else:
            self.k_spike = None
            self.v_spike = None

    def _parallel(
        self, q: Tensor, k: Tensor, v: Tensor, alpha: Tensor, beta: Tensor
    ) -> Tensor:
        B, T, H, D = q.shape
        q_h = q.transpose(1, 2)
        k_h = k.transpose(1, 2)
        v_h = v.transpose(1, 2)

        qk = torch.matmul(q_h, k_h.transpose(-2, -1))

        alpha_per_head = alpha.squeeze(-1)
        log_alpha = torch.log(alpha_per_head.clamp(min=1e-6)).float()

        positions = torch.arange(T, device=q.device, dtype=torch.float32)
        time_diff = positions.unsqueeze(1) - positions.unsqueeze(0)
        decay_exp = time_diff.unsqueeze(0) * log_alpha.unsqueeze(-1).unsqueeze(-1)
        causal_mask = torch.triu(
            torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1
        )
        decay_exp = decay_exp.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
        decay = torch.exp(decay_exp).to(qk.dtype)

        beta_expanded = beta.transpose(1, 2).unsqueeze(2)
        weights = qk * decay.unsqueeze(0) * beta_expanded

        out = torch.matmul(weights, v_h)
        return out.transpose(1, 2)

    def _recurrent(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        alpha: Tensor,
        beta: Tensor,
        state: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        B, T, H, D = q.shape
        alpha_unsq = alpha.unsqueeze(0).unsqueeze(-1).float()
        if state is None:
            state = torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
        else:
            state = state.float()
        outputs = []
        for t in range(T):
            k_t = k[:, t].float()
            v_t = v[:, t].float()
            q_t = q[:, t].float()
            beta_t = beta[:, t].unsqueeze(-1).unsqueeze(-1).float()
            state = alpha_unsq * state + beta_t * torch.einsum(
                "bhd,bhe->bhde", k_t, v_t
            )
            o_t = torch.einsum("bhd,bhde->bhe", q_t, state)
            outputs.append(o_t.to(q.dtype))
        return torch.stack(outputs, dim=1), state

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

        if self.k_spike is not None:
            k_flat = k.reshape(B, T, self.inner)
            k_flat = self.k_spike(k_flat)
            k = k_flat.view(B, T, self.nh, self.hd)
            aux["k_fr_per_neuron"] = (
                (k_flat != 0).float().mean(dim=(0, 1)).detach()
            )
        if self.v_spike is not None:
            v_flat = v.reshape(B, T, self.inner)
            v_flat = self.v_spike(v_flat)
            v = v_flat.view(B, T, self.nh, self.hd)
            aux["v_fr_per_neuron"] = (
                (v_flat != 0).float().mean(dim=(0, 1)).detach()
            )

        alpha = torch.sigmoid(self.alpha_log)
        beta = torch.sigmoid(self.beta_proj(x))

        if state is not None or T == 1:
            out, new_state = self._recurrent(q, k, v, alpha, beta, state)
        else:
            out = self._parallel(q, k, v, alpha, beta)
            new_state = None

        out = out.reshape(B, T, self.inner)
        out = self.o(out)
        return out, new_state, aux


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

        q = self.q_proj(x).view(B, T, self.nh, self.hd)
        q_rope_shared = self.q_rope_proj(x)
        q_rope_shared = rotary_apply(q_rope_shared, cos_b, sin_b)

        q_t = q.transpose(1, 2)
        k_t = k_content.transpose(1, 2)
        v_t = v.transpose(1, 2)

        attn_content = torch.matmul(q_t, k_t.transpose(-2, -1))
        q_rope_exp = q_rope_shared.unsqueeze(2).expand(-1, -1, self.nh, -1).transpose(1, 2)
        k_rope_exp = cached_k_rope.unsqueeze(2).expand(-1, -1, self.nh, -1).transpose(1, 2)
        attn_rope = torch.matmul(q_rope_exp, k_rope_exp.transpose(-2, -1))

        scale = math.sqrt(self.hd + self.d_R)
        attn_scores = (attn_content + attn_rope) / scale

        if T > 1:
            causal_mask = torch.triu(
                torch.full((T, total_len), float("-inf"), device=x.device, dtype=attn_scores.dtype),
                diagonal=total_len - T + 1,
            )
            attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_t)
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
        else:
            raise ValueError(f"unknown layer type: {layer_type}")
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
        mlp_out = self.mlp(self.mlp_norm(x))
        x = residual + mlp_out

        return x, new_state, aux


class NeuralMachine(nn.Module):

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
        self._gc_enabled = False

    def _init_linears(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.weight is self.embedding.weight:
                    continue
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

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
            self.data = data.long()
        elif isinstance(data, np.ndarray):
            self.data = torch.from_numpy(np.asarray(data, dtype=np.uint8)).long()
        else:
            self.data = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
        self.seq_len = seq_len
        self._n = max(0, (len(self.data) - 1) // self.seq_len)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> Tensor:
        if idx >= self._n:
            raise IndexError(idx)
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        return self.data[start:end]


def collate_fn(batch: list[Tensor]) -> Tensor:
    return torch.stack([b for b in batch], dim=0)


def download_fineweb_edu(max_bytes: int = 5_000_000_000) -> tuple[Any, Any, str]:
    data_dir = Path("data/fineweb")
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
        log(f"fineweb-edu failed: {e}, falling back to wikitext-2")
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
        if not os.environ.get("ALLOW_SYNTHETIC"):
            raise RuntimeError(
                f"wikitext-2 failed to load ({e}) and ALLOW_SYNTHETIC is not set. "
                "refusing to train on synthetic random bytes, which would produce a "
                "fake run indistinguishable from real data. set ALLOW_SYNTHETIC=1 "
                "only for debugging pipeline plumbing."
            )
        log("WARNING: using synthetic random-byte data (ALLOW_SYNTHETIC=1)")
        rng = np.random.RandomState(42)
        train_data = bytes(rng.randint(0, 256, size=2_000_000).tolist())
        val_data = bytes(rng.randint(0, 256, size=200_000).tolist())
        return train_data, val_data, "synthetic-random-bytes"


@torch.no_grad()
def run_validation(
    model: NeuralMachine,
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


def collect_spike_stats(aux: dict) -> dict:
    per_layer_k_fr: list[float] = []
    per_layer_v_fr: list[float] = []
    per_layer_mean_fr: list[float] = []
    per_layer_dead_count: list[int] = []
    per_layer_neuron_count: list[int] = []
    all_per_neuron: list[Tensor] = []
    layer_indices: list[int] = []

    for layer_idx in sorted(aux.keys()) if isinstance(aux, dict) else []:
        layer_aux = aux[layer_idx]
        if not isinstance(layer_aux, dict):
            continue
        k_fr_val: float | None = None
        v_fr_val: float | None = None
        dead_here = 0
        neurons_here = 0
        if "k_fr_per_neuron" in layer_aux:
            fr_vec = layer_aux["k_fr_per_neuron"]
            k_fr_val = float(fr_vec.mean().item())
            all_per_neuron.append(fr_vec)
            dead_here += int((fr_vec < 0.05).sum().item())
            neurons_here += int(fr_vec.numel())
        if "v_fr_per_neuron" in layer_aux:
            fr_vec = layer_aux["v_fr_per_neuron"]
            v_fr_val = float(fr_vec.mean().item())
            all_per_neuron.append(fr_vec)
            dead_here += int((fr_vec < 0.05).sum().item())
            neurons_here += int(fr_vec.numel())
        if k_fr_val is None and v_fr_val is None:
            continue
        layer_indices.append(int(layer_idx))
        per_layer_k_fr.append(k_fr_val if k_fr_val is not None else float("nan"))
        per_layer_v_fr.append(v_fr_val if v_fr_val is not None else float("nan"))
        vals = [v for v in (k_fr_val, v_fr_val) if v is not None]
        per_layer_mean_fr.append(float(np.mean(vals)) if vals else 0.0)
        per_layer_dead_count.append(dead_here)
        per_layer_neuron_count.append(neurons_here)

    if all_per_neuron:
        rates_tensor = torch.cat(all_per_neuron).detach().cpu().numpy()
        dead_pct = float((rates_tensor < 0.05).mean())
        saturated_pct = float((rates_tensor > 0.95).mean())
    else:
        dead_pct = 0.0
        saturated_pct = 0.0

    return {
        "mean_fr": float(np.mean(per_layer_mean_fr)) if per_layer_mean_fr else 0.0,
        "dead_pct": dead_pct,
        "saturated_pct": saturated_pct,
        "layer_indices": layer_indices,
        "per_layer_k_fr": per_layer_k_fr,
        "per_layer_v_fr": per_layer_v_fr,
        "per_layer_mean_fr": per_layer_mean_fr,
        "per_layer_dead_count": per_layer_dead_count,
        "per_layer_neuron_count": per_layer_neuron_count,
    }


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
            self._f.write(json.dumps(record, default=str) + "\n")
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
    tmp = _atomic_tmp_path(path)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, default=str)
        tmp.replace(path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def train_model(
    cfg: Config,
    train_data: Any,
    val_data: Any,
    name: str,
    output_dir: Path,
    dataset_source: str = "unknown",
) -> dict:
    log(f"=== training {name} ===")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    py_random.seed(cfg.seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = _cfg_to_dict(cfg)
    env = _capture_env_metadata()
    git = _capture_git_metadata()
    cfg_hash = _config_hash(cfg_dict)

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

    model = NeuralMachine(cfg).to(DEVICE)
    n_params = model.count_params()
    log(f"params: {n_params:,}")
    log(f"layers: {len(cfg.layer_types)} ({cfg.layer_pattern} x {cfg.n_layers // len(cfg.layer_pattern)})")

    train_ds = ByteDataset(train_data, cfg.seq_len)
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
        "spike_fr": [],
        "spike_fr_per_layer": [],
        "dead_pct": [],
        "saturated_pct": [],
        "dead_count_per_layer": [],
        "throughput_tok_per_sec": [],
        "gpu_mem_alloc_gb": [],
        "gpu_mem_max_alloc_gb": [],
        "step_duration_sec": [],
        "val_steps": [],
        "val_bpb": [],
        "val_mean_loss": [],
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

    resume_path = output_dir / f"{name}_last.pt"
    if resume_path.exists():
        try:
            resumed = load_resume_checkpoint(resume_path, model, optimizer, scheduler)
            step = resumed["step"]
            best_val_bpb = resumed["best_val_bpb"]
            epoch_idx = resumed.get("epoch_idx", 0)
            step_in_epoch = resumed.get("step_in_epoch", 0)
            for k, v in resumed.get("history", {}).items():
                if k in history:
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
            log(f"resume failed: {e}; starting fresh")
            step = 0
            best_val_bpb = float("inf")
            epoch_idx = 0
            step_in_epoch = 0

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
        while step < cfg.max_steps and not shutdown_flag["requested"]:
            if train_gen_state_at_epoch_start is None or resume_skip_batches == 0:
                train_gen_state_at_epoch_start = train_generator.get_state()
            for batch in train_dl:
                if resume_skip_batches > 0:
                    resume_skip_batches -= 1
                    continue
                if step >= cfg.max_steps or shutdown_flag["requested"]:
                    break

                batch = batch.to(DEVICE)
                input_ids = batch[:, :-1]
                target = batch[:, 1:]
                batch_tokens = int(target.numel())

                with torch.amp.autocast(device_type=DEVICE.type, dtype=amp_dtype, enabled=use_amp):
                    logits, _, aux = model(input_ids)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        target.reshape(-1),
                    )
                    loss_scaled = loss / cfg.grad_accum_steps if cfg.grad_accum_steps > 1 else loss

                loss_val = loss.detach().float().item()
                if not math.isfinite(loss_val):
                    log(f"FATAL: non-finite loss at step {step}: {loss_val}")
                    _flush_emergency("non_finite_loss", {"loss": loss_val})
                    raise RuntimeError(f"non-finite loss at step {step}: {loss_val}")

                spike_stats = collect_spike_stats(aux)

                loss_scaled.backward()

                grad_norm_val: float = float("nan")
                do_step = ((step + 1) % cfg.grad_accum_steps == 0) or (step == cfg.max_steps - 1)
                if do_step:
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    grad_norm_val = gn.item() if isinstance(gn, torch.Tensor) else float(gn)
                    if not math.isfinite(grad_norm_val):
                        log(f"FATAL: non-finite grad norm at step {step}: {grad_norm_val}")
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
                history["spike_fr"].append(spike_stats["mean_fr"])
                history["spike_fr_per_layer"].append(list(spike_stats["per_layer_mean_fr"]))
                history["dead_pct"].append(spike_stats["dead_pct"])
                history["saturated_pct"].append(spike_stats["saturated_pct"])
                history["dead_count_per_layer"].append(list(spike_stats["per_layer_dead_count"]))
                history["throughput_tok_per_sec"].append(tokens_per_sec)
                history["gpu_mem_alloc_gb"].append(mem_alloc_gb)
                history["gpu_mem_max_alloc_gb"].append(mem_max_alloc_gb)
                history["step_duration_sec"].append(step_duration)

                metrics_logger.log({
                    "event": "step",
                    "step": step,
                    "loss": loss_val,
                    "grad_norm_pre_clip": grad_norm_val,
                    "lr": lr_now,
                    "mean_fr": spike_stats["mean_fr"],
                    "dead_pct": spike_stats["dead_pct"],
                    "saturated_pct": spike_stats["saturated_pct"],
                    "per_layer_mean_fr": spike_stats["per_layer_mean_fr"],
                    "per_layer_dead_count": spike_stats["per_layer_dead_count"],
                    "throughput_tok_per_sec": tokens_per_sec,
                    "step_duration_sec": step_duration,
                    "tokens_seen_total": tokens_seen_total,
                    "gpu_mem_alloc_gb": mem_alloc_gb,
                    "gpu_mem_max_alloc_gb": mem_max_alloc_gb,
                    "elapsed_sec": elapsed,
                    "wall_time": now,
                })

                if step % 50 == 0:
                    log(
                        f"step {step}/{cfg.max_steps} loss={loss_val:.4f} "
                        f"gn={grad_norm_val:.3f} fr={spike_stats['mean_fr']:.3f} "
                        f"dead={spike_stats['dead_pct']:.3f} "
                        f"tps={tokens_per_sec:.0f} mem={mem_alloc_gb:.2f}gb "
                        f"lr={lr_now:.2e} t={elapsed:.0f}s"
                    )

                step += 1
                step_in_epoch += 1

                if step % cfg.val_interval == 0:
                    val_result = run_validation(model, val_dl, use_amp=use_amp, amp_dtype=amp_dtype)
                    history["val_steps"].append(step)
                    history["val_bpb"].append(val_result["bpb"])
                    history["val_mean_loss"].append(val_result["mean_loss"])
                    metrics_logger.log({
                        "event": "validation",
                        "step": step,
                        "val_bpb": val_result["bpb"],
                        "val_mean_loss": val_result["mean_loss"],
                        "val_tokens": val_result["tokens"],
                        "val_precision": val_result.get("precision", "unknown"),
                    })
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
                    )
                    _write_results_snapshot(
                        output_dir / f"{name}_results.json",
                        name, n_params, best_val_bpb, step,
                        time.time() - start_time, tokens_seen_total,
                        cfg_dict, metadata, history, final=False,
                    )
                    metrics_logger.fsync()

                    if torch.cuda.is_available():
                        try:
                            torch.cuda.reset_peak_memory_stats()
                        except Exception:
                            pass

            if step >= cfg.max_steps or shutdown_flag["requested"]:
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
            metrics_logger.log({
                "event": "final_validation_skipped",
                "step": step,
                "reason": "shutdown_requested",
            })
        elif already_valed_here:
            last_bpb = history["val_bpb"][-1]
            last_mean_loss = history["val_mean_loss"][-1]
            final_val = {
                "mean_loss": last_mean_loss,
                "bpb": last_bpb,
                "tokens": 0,
                "precision": "reused_mid_loop_val",
            }
            metrics_logger.log({
                "event": "final_validation_deduped",
                "step": step,
                "reused_from_mid_loop_val": True,
                "val_bpb": last_bpb,
            })
        else:
            final_val = run_validation(model, val_dl, use_amp=use_amp, amp_dtype=amp_dtype)
            history["val_steps"].append(step)
            history["val_bpb"].append(final_val["bpb"])
            history["val_mean_loss"].append(final_val["mean_loss"])
            metrics_logger.log({
                "event": "final_validation",
                "step": step,
                "val_bpb": final_val["bpb"],
                "val_mean_loss": final_val["mean_loss"],
                "val_tokens": final_val["tokens"],
                "val_precision": final_val.get("precision", "unknown"),
            })
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
    model: NeuralMachine,
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


def _test_delta_equivalence() -> None:
    log("delta parallel/recurrent equivalence test (no spikes, channel_wise_gate=False)")
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
        use_spikes=False,
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
        out_par, _, _ = layer(x)

        chunks: list[Tensor] = []
        state = None
        for t in range(T):
            single = x[:, t : t + 1, :]
            out_t, state, _ = layer(single, state=state, offset=t)
            chunks.append(out_t)
        out_rec = torch.cat(chunks, dim=1)

    diff = (out_par - out_rec).abs()
    max_abs = diff.max().item()
    ref = out_par.abs().max().item() + 1e-9
    rel = max_abs / ref
    if rel >= 1e-3:
        raise AssertionError(
            f"delta parallel/recurrent paths diverge: max_abs={max_abs:.3e}, rel={rel:.3e}. "
            f"expected rel < 1e-3 with channel_wise_gate=False"
        )
    log(f"  parallel vs recurrent: max_abs={max_abs:.2e}, rel={rel:.2e} ok")


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
    model = NeuralMachine(cfg)
    n_params = model.count_params()
    log(f"smoke model params: {n_params:,}")

    input_ids = torch.randint(0, cfg.vocab_size, (2, 32))
    targets = torch.randint(0, cfg.vocab_size, (2, 32))

    model.train(True)
    logits, states, aux = model(input_ids)
    assert logits.shape == (2, 32, cfg.vocab_size), f"bad logits shape {logits.shape}"
    assert len(states) == cfg.n_layers, f"bad states length {len(states)}"
    assert len(aux) == cfg.n_layers, f"bad aux length {len(aux)}"

    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
    loss.backward()

    missing = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is None:
            missing.append(name)
    if missing:
        raise RuntimeError(f"no grad for params: {missing[:5]} ({len(missing)} total)")
    log(f"smoke test forward+backward ok. loss={loss.item():.4f}")

    model.train(False)
    single = torch.randint(0, cfg.vocab_size, (1, 1))
    logits_s, states_s, _ = model(single)
    assert logits_s.shape == (1, 1, cfg.vocab_size), f"bad single logits {logits_s.shape}"
    for i, (layer_type, s) in enumerate(zip(cfg.layer_types, states_s)):
        if layer_type == "DELTA":
            assert s is not None, f"delta layer {i} returned None state in recurrent mode"
        elif layer_type == "ATTN":
            assert s is not None, f"attn layer {i} returned None cache"
    log("smoke test single-token recurrent ok")

    prompt = b"hello"
    gen = greedy_generate(model, prompt, max_new_tokens=8)
    assert len(gen) == 8, f"greedy_generate returned {len(gen)} bytes, expected 8"
    log(f"smoke test greedy_generate ok (len={len(gen)})")

    _test_delta_equivalence()

    log("all smoke checks passed")


def main() -> None:
    if os.environ.get("SMOKE_TEST"):
        smoke_test()
        return

    output_dir = Path(os.environ.get("NM_OUTPUT_DIR", str(DEFAULT_OUTPUT)))
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    log(f"device: {DEVICE}")
    log(f"output: {output_dir}")
    log(
        f"config: d_model={cfg.d_model} n_layers={cfg.n_layers} "
        f"seq_len={cfg.seq_len} batch_size={cfg.batch_size}"
    )
    log(f"layer pattern: {cfg.layer_pattern}")

    log("loading training data")
    if cfg.max_steps <= 50:
        train_data, val_data, dataset_source = download_wikitext2()
    else:
        train_data, val_data, dataset_source = download_fineweb_edu(max_bytes=5_000_000_000)
    log(f"dataset: {dataset_source}  train: {len(train_data):,} bytes, val: {len(val_data):,} bytes")

    result = train_model(
        cfg,
        train_data,
        val_data,
        name="run1_baseline",
        output_dir=output_dir,
        dataset_source=dataset_source,
    )

    log(f"results.json snapshot at {output_dir / 'run1_baseline_results.json'}")
    log(f"metrics stream at {output_dir / 'run1_baseline_metrics.jsonl'}")
    log(f"metadata at {output_dir / 'run1_baseline_metadata.json'}")
    log(
        f"final best_val_bpb={result['best_val_bpb']:.4f} "
        f"steps={result['total_steps']} tokens={result['tokens_seen_total']:,}"
    )
    log("done")


if __name__ == "__main__":
    main()
