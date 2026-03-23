from __future__ import annotations

import gc
import json
import math
import os
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

def _select_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        cap = torch.cuda.get_device_capability()
        if cap[0] < 7:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU sm_{cap[0]}{cap[1]} not supported by this PyTorch, falling back to CPU", flush=True)
            return torch.device("cpu")
    except Exception:
        pass
    return torch.device("cuda")


DEVICE = _select_device()
SEED = 42
PHASE = 3
OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle").exists() else Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

FLA_AVAILABLE = False
if DEVICE.type == "cuda":
    os.environ.setdefault("TRITON_CACHE_DIR", str(OUTPUT_DIR / ".triton_cache"))
    try:
        from fla.ops.kda import chunk_kda as _chunk_kda
        FLA_AVAILABLE = True
    except ImportError:
        try:
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                                   "flash-linear-attention"], stdout=subprocess.DEVNULL)
            from fla.ops.kda import chunk_kda as _chunk_kda
            FLA_AVAILABLE = True
        except Exception:
            pass

print(f"[{datetime.now().strftime('%H:%M:%S')}] device: {DEVICE}")
print(f"[{datetime.now().strftime('%H:%M:%S')}] fla available: {FLA_AVAILABLE}")
print(f"[{datetime.now().strftime('%H:%M:%S')}] output: {OUTPUT_DIR}")


@dataclass
class TrainConfig:
    d_model: int = 256
    n_layers: int = 8
    vocab_size: int = 256
    max_seq_len: int = 512
    kda_num_heads: int = 4
    kda_head_dim: int = 64
    m3_d_state: int = 16
    m3_expand: int = 2
    mla_d_c: int = 64
    mla_d_R: int = 16
    mla_num_heads: int = 4
    mlp_ratio: float = 2.25
    layer_pattern: tuple[str, ...] = ("KDA", "KDA", "KDA", "Mamba3", "KDA", "KDA", "KDA", "MLA")
    lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    max_steps: int = 500
    warmup_steps: int = 100
    eval_interval: int = 100
    gradient_clip: float = 1.0
    seq_len: int = 256
    use_spikes: bool = True
    spike_alpha: float = 1.0
    spatial_mode: bool = False

    @property
    def layer_types(self) -> list[str]:
        pattern = list(self.layer_pattern)
        repeats = self.n_layers // len(pattern)
        return pattern * repeats


BASELINE_CONFIG = TrainConfig(
    d_model=256,
    n_layers=8,
    vocab_size=256,
    layer_pattern=("Transformer",) * 8,
    use_spikes=False,
)


class RotaryPE(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._cache_len = 0
        self._cos: Tensor | None = None
        self._sin: Tensor | None = None

    def _build(self, length: int, device: torch.device) -> None:
        if length <= self._cache_len and self._cos is not None:
            return
        self._cache_len = length
        t = torch.arange(length, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos = emb.cos()
        self._sin = emb.sin()

    def forward(self, x: Tensor, offset: int = 0) -> tuple[Tensor, Tensor]:
        self._build(x.shape[-2] + offset, x.device)
        sl = x.shape[-2]
        return self._cos[offset:offset+sl], self._sin[offset:offset+sl]


def rotary_apply(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    c, s = cos[..., :d//2], sin[..., :d//2]
    return torch.cat([x1*c - x2*s, x2*c + x1*s], dim=-1)


class TernaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, threshold: Tensor, training: bool = True) -> Tensor:
        ctx.save_for_backward(x, threshold)
        out = torch.zeros_like(x)
        out[x > threshold] = 1.0
        out[x < -threshold] = -1.0
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None, None]:
        return grad_output.clone(), None, None


class AdaptiveSpike(nn.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.register_buffer("running_density", torch.tensor(0.0))
        self.register_buffer("n_updates", torch.tensor(0))

    def forward(self, x: Tensor) -> Tensor:
        threshold = torch.clamp(self.alpha * x.abs().mean(), 0.01, 10.0)
        spikes = TernaryQuantizer.apply(x, threshold, self.training)
        if self.training:
            with torch.no_grad():
                d = (spikes != 0).float().mean()
                self.running_density = 0.99 * self.running_density + 0.01 * d
                self.n_updates += 1
        return spikes


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = (x.float().pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


NUM_MV_COMPONENTS = 16
_GRADE_RANGES = [(0, 1), (1, 5), (5, 11), (11, 15), (15, 16)]


def _build_cayley_table() -> tuple[Tensor, Tensor]:
    basis_blades = [
        (),
        (0,), (1,), (2,), (3,),
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
        (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3),
        (0, 1, 2, 3),
    ]
    metric_sq = {0: 0, 1: 1, 2: 1, 3: 1}
    blade_to_idx = {b: i for i, b in enumerate(basis_blades)}

    def _mul(a, b):
        combined = list(a) + list(b)
        sign = 1
        n = len(combined)
        for i in range(n):
            for j in range(i + 1, n):
                if combined[j] < combined[i]:
                    combined[i], combined[j] = combined[j], combined[i]
                    sign *= -1
        result = []
        i = 0
        while i < len(combined):
            if i + 1 < len(combined) and combined[i] == combined[i + 1]:
                sq = metric_sq[combined[i]]
                if sq == 0:
                    return 0, ()
                sign *= sq
                i += 2
            else:
                result.append(combined[i])
                i += 1
        return sign, tuple(result)

    indices = torch.zeros(16, 16, dtype=torch.long)
    signs = torch.zeros(16, 16)
    for i, ba in enumerate(basis_blades):
        for j, bb in enumerate(basis_blades):
            s, rb = _mul(ba, bb)
            if s != 0 and rb in blade_to_idx:
                indices[i, j] = blade_to_idx[rb]
                signs[i, j] = float(s)
    return indices, signs


def _build_sparse_gp_tables(idx: Tensor, sgn: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    si, sj, tk, sg = [], [], [], []
    for i in range(16):
        for j in range(16):
            if sgn[i, j] != 0:
                si.append(i)
                sj.append(j)
                tk.append(idx[i, j].item())
                sg.append(sgn[i, j].item())
    return (
        torch.tensor(si, dtype=torch.long),
        torch.tensor(sj, dtype=torch.long),
        torch.tensor(tk, dtype=torch.long),
        torch.tensor(sg, dtype=torch.float),
    )


_CAY_IDX, _CAY_SGN = _build_cayley_table()
_GP_SI, _GP_SJ, _GP_TK, _GP_SG = _build_sparse_gp_tables(_CAY_IDX, _CAY_SGN)


def _gp_ensure_device(device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    global _GP_SI, _GP_SJ, _GP_TK, _GP_SG
    if _GP_SI.device != device:
        _GP_SI = _GP_SI.to(device)
        _GP_SJ = _GP_SJ.to(device)
        _GP_TK = _GP_TK.to(device)
        _GP_SG = _GP_SG.to(device)
    return _GP_SI, _GP_SJ, _GP_TK, _GP_SG


def sparse_gp(a: Tensor, b: Tensor) -> Tensor:
    si, sj, tk, sg = _gp_ensure_device(a.device)
    sg = sg.to(dtype=a.dtype)
    batch_shape = torch.broadcast_shapes(a.shape[:-1], b.shape[:-1])
    a = a.expand(*batch_shape, 16)
    b = b.expand(*batch_shape, 16)
    products = a[..., si] * b[..., sj] * sg
    flat = products.reshape(-1, products.shape[-1])
    out = torch.zeros(flat.shape[0], 16, device=a.device, dtype=a.dtype)
    out.index_add_(-1, tk, flat)
    return out.reshape(*batch_shape, 16)


_REVERSE_SIGNS = torch.ones(16)
for _k, (_gs, _ge) in enumerate(_GRADE_RANGES):
    if _k * (_k - 1) // 2 % 2 == 1:
        _REVERSE_SIGNS[_gs:_ge] = -1.0


def reverse_mv(x: Tensor) -> Tensor:
    global _REVERSE_SIGNS
    if _REVERSE_SIGNS.device != x.device:
        _REVERSE_SIGNS = _REVERSE_SIGNS.to(x.device)
    return x * _REVERSE_SIGNS.to(dtype=x.dtype)


def sandwich_gp(rotor: Tensor, x: Tensor) -> Tensor:
    return sparse_gp(sparse_gp(rotor, x), reverse_mv(rotor))


class KDALayer(nn.Module):
    def __init__(self, d: int, nh: int, hd: int, spikes: bool = False, alpha: float = 1.0) -> None:
        super().__init__()
        self.nh, self.hd = nh, hd
        inner = nh * hd
        self.q = nn.Linear(d, inner, bias=False)
        self.k = nn.Linear(d, inner, bias=False)
        self.v = nn.Linear(d, inner, bias=False)
        self.o = nn.Linear(inner, d, bias=False)
        self.beta_proj = nn.Linear(d, nh, bias=True)
        self.alpha_log = nn.Parameter(torch.zeros(nh, hd) - 2.0)
        self.rope = RotaryPE(hd)
        self.k_spike = AdaptiveSpike(alpha) if spikes else None
        self.v_spike = AdaptiveSpike(alpha) if spikes else None

    def forward(self, x: Tensor, state: Tensor | None = None, offset: int = 0) -> tuple[Tensor, Tensor, dict]:
        B, T, _ = x.shape
        aux: dict = {}
        qr = self.q(x).view(B, T, self.nh, self.hd)
        kr = self.k(x).view(B, T, self.nh, self.hd)
        vr = self.v(x).view(B, T, self.nh, self.hd)
        cos, sin = self.rope(qr, offset)
        c, s = cos.unsqueeze(1), sin.unsqueeze(1)
        qr = rotary_apply(qr.transpose(1,2), c, s).transpose(1,2)
        kr = rotary_apply(kr.transpose(1,2), c, s).transpose(1,2)
        if self.k_spike:
            aux["pre_k"] = kr.detach()
            kr = self.k_spike(kr)
            aux["k_spikes"] = kr.detach()
        if self.v_spike:
            aux["pre_v"] = vr.detach()
            vr = self.v_spike(vr)
            aux["v_spikes"] = vr.detach()
        alpha = torch.sigmoid(self.alpha_log)
        beta = torch.sigmoid(self.beta_proj(x))
        if state is not None or T == 1:
            alpha_u = alpha.unsqueeze(0).unsqueeze(-1)
            if state is None:
                state = torch.zeros(B, self.nh, self.hd, self.hd, device=x.device, dtype=x.dtype)
            outs = []
            for t in range(T):
                bt = beta[:, t].unsqueeze(-1).unsqueeze(-1)
                state = alpha_u * state + bt * torch.einsum("bhd,bhe->bhde", kr[:,t], vr[:,t])
                outs.append(torch.einsum("bhd,bhde->bhe", qr[:,t], state))
            out = torch.stack(outs, 1).reshape(B, T, -1)
        elif FLA_AVAILABLE and T >= 512:
            from fla.ops.kda import chunk_kda
            g = F.logsigmoid(self.alpha_log)
            g = g.unsqueeze(0).unsqueeze(1).expand(B, T, -1, -1).contiguous()
            kda_scale = 1.0 / math.sqrt(self.hd)
            o, state = chunk_kda(
                q=qr.half(), k=kr.half(), v=vr.half(),
                g=g.float(), beta=beta.float(),
                scale=kda_scale, output_final_state=True,
            )
            out = o.to(x.dtype).reshape(B, T, -1)
        else:
            qh = qr.transpose(1,2)
            kh = kr.transpose(1,2)
            vh = vr.transpose(1,2)
            qk = torch.matmul(qh, kh.transpose(-2,-1))
            alpha_mean = alpha.mean(dim=-1) if alpha.shape[-1] > 1 else alpha.squeeze(-1)
            log_a = torch.log(alpha_mean.clamp(min=1e-6))
            pos = torch.arange(T, device=x.device, dtype=x.dtype)
            time_diff = pos.unsqueeze(1) - pos.unsqueeze(0)
            decay_exp = time_diff.unsqueeze(0) * log_a.unsqueeze(-1).unsqueeze(-1)
            cmask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            decay_exp = decay_exp.masked_fill(cmask.unsqueeze(0), float("-inf"))
            decay = torch.exp(decay_exp)
            beta_e = beta.transpose(1,2).unsqueeze(2)
            weights = qk * decay.unsqueeze(0) * beta_e
            out = torch.matmul(weights, vh).transpose(1,2).reshape(B, T, -1)
            with torch.no_grad():
                alpha_u = alpha.unsqueeze(0).unsqueeze(-1)
                state = torch.zeros(B, self.nh, self.hd, self.hd, device=x.device, dtype=x.dtype)
                for t in range(T):
                    bt = beta[:, t].unsqueeze(-1).unsqueeze(-1)
                    state = alpha_u * state + bt * torch.einsum("bhd,bhe->bhde", kr[:,t], vr[:,t])
        aux["state_norm"] = state.norm().item()
        return self.o(out), state, aux


class Mamba3Layer(nn.Module):
    def __init__(self, d: int, ds: int = 16, expand: int = 2) -> None:
        super().__init__()
        di = d * expand
        self.in_proj = nn.Linear(d, di*2, bias=False)
        self.A_log = nn.Parameter(torch.randn(di, ds).uniform_(-4, -1))
        self.B_proj = nn.Linear(d, ds, bias=False)
        self.C_proj = nn.Linear(d, ds, bias=False)
        self.dt_proj = nn.Linear(d, di, bias=True)
        self.dt_bias = nn.Parameter(torch.zeros(di) - 4.0)
        self.norm = nn.LayerNorm(di)
        self.out_proj = nn.Linear(di, d, bias=False)
        self.di, self.ds = di, ds

    def forward(self, x: Tensor, state: Tensor | None = None, offset: int = 0) -> tuple[Tensor, Tensor, dict]:
        B, T, _ = x.shape
        xz = self.in_proj(x)
        xi, z = xz.chunk(2, dim=-1)
        z = F.silu(z)
        A = -torch.exp(self.A_log)
        Bm = self.B_proj(x)
        C = self.C_proj(x)
        dt = F.softplus(self.dt_proj(x) + self.dt_bias)
        dtA = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        A_bar = (1 + dtA/2) / (1 - dtA/2)
        B_bar = dt.unsqueeze(-1) / (1 - dtA/2)
        if state is None:
            state = torch.zeros(B, self.di, self.ds, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            Bt = Bm[:,t].unsqueeze(1).expand(-1, self.di, -1)
            xt = xi[:,t].unsqueeze(-1)
            state = A_bar[:,t] * state + B_bar[:,t] * Bt * xt
            Ct = C[:,t].unsqueeze(1).expand(-1, self.di, -1)
            outs.append((state * Ct).sum(-1))
        out = torch.stack(outs, 1) * z
        return self.out_proj(self.norm(out)), state, {"state_util": (state.abs()>1e-6).float().mean().item()}


class MLALayer(nn.Module):
    def __init__(self, d: int, dc: int = 64, dr: int = 16, nh: int = 4) -> None:
        super().__init__()
        self.dc, self.dr, self.nh = dc, dr, nh
        self.hd = d // nh
        self.kv_down = nn.Linear(d, dc, bias=False)
        self.k_up = nn.Linear(dc, d, bias=False)
        self.v_up = nn.Linear(dc, d, bias=False)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.q_rope = nn.Linear(d, dr, bias=False)
        self.k_rope = nn.Linear(dc, dr, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        inv = 1.0 / (10000.0 ** (torch.arange(0, dr, 2).float() / dr))
        self.register_buffer("inv_freq", inv)

    def _rope(self, sl: int, off: int, dev: torch.device) -> tuple[Tensor, Tensor]:
        t = torch.arange(off, off+sl, device=dev, dtype=self.inv_freq.dtype)
        f = torch.outer(t, self.inv_freq)
        e = torch.cat([f, f], -1)
        return e.cos(), e.sin()

    def _rap(self, x: Tensor, c: Tensor, s: Tensor) -> Tensor:
        d = x.shape[-1]
        x1, x2 = x[..., :d//2], x[..., d//2:]
        return torch.cat([x1*c[...,:d//2]-x2*s[...,:d//2], x2*c[...,:d//2]+x1*s[...,:d//2]], -1)

    def forward(self, x: Tensor, kv_cache: Tensor | None = None, offset: int = 0) -> tuple[Tensor, Tensor, dict]:
        B, T, _ = x.shape
        ckv = self.kv_down(x)
        kr_shared = self.k_rope(ckv)
        cos, sin = self._rope(T, offset, x.device)
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
        kr_shared = self._rap(kr_shared, cos, sin)
        entry = torch.cat([ckv, kr_shared], -1)
        if kv_cache is not None:
            entry = torch.cat([kv_cache, entry], 1)
        cached_ckv = entry[..., :self.dc]
        cached_kr = entry[..., self.dc:]
        tl = entry.shape[1]
        k = self.k_up(cached_ckv).view(B, tl, self.nh, self.hd).transpose(1,2)
        v = self.v_up(cached_ckv).view(B, tl, self.nh, self.hd).transpose(1,2)
        q = self.q_proj(x).view(B, T, self.nh, self.hd).transpose(1,2)
        qr_shared = self.q_rope(x)
        qc, qs = self._rope(T, offset, x.device)
        qc, qs = qc.unsqueeze(0), qs.unsqueeze(0)
        qr_shared = self._rap(qr_shared, qc, qs)
        ac = torch.matmul(q, k.transpose(-2,-1))
        qr_exp = qr_shared.unsqueeze(2).expand(-1,-1,self.nh,-1).transpose(1,2)
        kr_exp = cached_kr.unsqueeze(2).expand(-1,-1,self.nh,-1).transpose(1,2)
        ar = torch.matmul(qr_exp, kr_exp.transpose(-2,-1))
        scale = math.sqrt(self.hd + self.dr)
        scores = (ac + ar) / scale
        if T > 1:
            mask = torch.triu(torch.full((T, tl), float("-inf"), device=x.device), diagonal=tl-T+1)
            scores = scores + mask.unsqueeze(0).unsqueeze(0)
        out = torch.matmul(F.softmax(scores, -1), v).transpose(1,2).reshape(B, T, -1)
        return self.o_proj(out), entry, {"compression": (2*self.nh*self.hd)/(self.dc+self.dr)}


class TransformerLayer(nn.Module):
    def __init__(self, d: int, nh: int = 4) -> None:
        super().__init__()
        self.nh = nh
        self.hd = d // nh
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.rope = RotaryPE(d // nh)

    def forward(self, x: Tensor, state: Tensor | None = None, offset: int = 0) -> tuple[Tensor, None, dict]:
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.nh, self.hd)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        cos, sin = self.rope(q, offset)
        c, s = cos.unsqueeze(1), sin.unsqueeze(1)
        q = rotary_apply(q.transpose(1,2), c, s)
        k = rotary_apply(k.transpose(1,2), c, s)
        v = v.transpose(1,2)
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.hd)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        scores = scores + mask
        out = torch.matmul(F.softmax(scores, -1), v).transpose(1,2).reshape(B, T, -1)
        return self.o_proj(out), None, {}


class SwiGLU(nn.Module):
    def __init__(self, d: int, ratio: float = 2.75, spatial_mode: bool = False) -> None:
        super().__init__()
        h = int(d * ratio)
        h = ((h + 63) // 64) * 64
        self.gate = nn.Linear(d, h, bias=False)
        self.up = nn.Linear(d, h, bias=False)
        self.down = nn.Linear(h, d, bias=False)
        self.spatial_mode = spatial_mode
        if spatial_mode:
            self.w_left = nn.Linear(d, 16, bias=False)
            self.w_right = nn.Linear(d, 16, bias=False)
            self.gp_proj = nn.Linear(16, d, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        h = self.down(F.silu(self.gate(x)) * self.up(x))
        if self.spatial_mode:
            left = self.w_left(x)
            right = self.w_right(x)
            gp_out = sparse_gp(left, right)
            h = h + self.gp_proj(gp_out)
        return h


class Block(nn.Module):
    def __init__(self, lt: str, cfg: TrainConfig, idx: int) -> None:
        super().__init__()
        self.lt = lt
        self.n1 = RMSNorm(cfg.d_model)
        self.n2 = RMSNorm(cfg.d_model)
        self._sk = "kv_cache" if lt == "MLA" else "state"
        if lt == "KDA":
            self.attn = KDALayer(cfg.d_model, cfg.kda_num_heads, cfg.kda_head_dim, cfg.use_spikes, cfg.spike_alpha)
        elif lt == "Mamba3":
            self.attn = Mamba3Layer(cfg.d_model, cfg.m3_d_state, cfg.m3_expand)
        elif lt == "MLA":
            self.attn = MLALayer(cfg.d_model, cfg.mla_d_c, cfg.mla_d_R, cfg.mla_num_heads)
        elif lt == "Transformer":
            self.attn = TransformerLayer(cfg.d_model, cfg.kda_num_heads)
            self._sk = "state"
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_ratio, spatial_mode=cfg.spatial_mode)

    def forward(self, x: Tensor, state=None, offset: int = 0) -> tuple[Tensor, any, dict]:
        r = x
        kwargs = {self._sk: state, "offset": offset}
        a, ns, aux = self.attn(self.n1(r), **kwargs)
        x = r + a
        x = x + self.mlp(self.n2(x))
        return x, ns, aux


class LM(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.emb.weight, std=0.02)
        lts = cfg.layer_types
        self.blocks = nn.ModuleList([Block(lts[i], cfg, i) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, ids: Tensor, states=None, offset: int = 0) -> tuple[Tensor, list, dict]:
        x = self.emb(ids)
        if states is None:
            states = [None] * self.cfg.n_layers
        ns, all_aux = [], {}
        for i, b in enumerate(self.blocks):
            x, s, aux = b(x, states[i], offset)
            ns.append(s)
            all_aux[i] = aux
        return self.head(self.norm(x)), ns, all_aux

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ByteTextDataset(Dataset):
    def __init__(self, data: bytes, seq_len: int) -> None:
        self.data = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(1, (len(self.data) - 1) // self.seq_len)

    def __getitem__(self, idx: int) -> Tensor:
        start = idx * self.seq_len
        end = min(start + self.seq_len + 1, len(self.data))
        return self.data[start:end]


def collate_fn(batch: list[Tensor]) -> Tensor:
    max_len = max(b.shape[0] for b in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        padded[i, :b.shape[0]] = b
    return padded


def download_wikitext2() -> tuple[bytes, bytes]:
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_text = "\n".join(ds["train"]["text"])
        val_text = "\n".join(ds["validation"]["text"])
        return train_text.encode("utf-8"), val_text.encode("utf-8")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] datasets load failed: {e}, using synthetic data")
        rng = np.random.RandomState(42)
        train_data = bytes(rng.randint(0, 256, size=2_000_000).tolist())
        val_data = bytes(rng.randint(0, 256, size=200_000).tolist())
        return train_data, val_data


SHAPE_MARKER_START = 0xFE
SHAPE_MARKER_END = 0xFD
NBODY_MARKER_START = 0xFC
NBODY_MARKER_END = 0xFB
SHAPE_CLASSES = 4
SHAPE_N_POINTS = 16
NBODY_N_PARTICLES = 4
NBODY_N_STEPS = 8


def _random_rotation_matrix(rng: np.random.RandomState) -> np.ndarray:
    q = rng.randn(4)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def _sphere_points(n: int, rng: np.random.RandomState) -> np.ndarray:
    pts = rng.randn(n, 3)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _cube_points(n: int, rng: np.random.RandomState) -> np.ndarray:
    pts = np.zeros((n, 3))
    for i in range(n):
        face = rng.randint(6)
        axis = face // 2
        side = 1.0 if face % 2 == 0 else -1.0
        coords = rng.uniform(-1, 1, size=3)
        coords[axis] = side
        pts[i] = coords
    return pts


def _tetrahedron_points(n: int, rng: np.random.RandomState) -> np.ndarray:
    verts = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ], dtype=np.float64) / np.sqrt(3)
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    pts = np.zeros((n, 3))
    for i in range(n):
        fi = rng.randint(len(faces))
        a, b, c = verts[faces[fi][0]], verts[faces[fi][1]], verts[faces[fi][2]]
        u, v = rng.uniform(0, 1, 2)
        if u + v > 1:
            u, v = 1 - u, 1 - v
        pts[i] = a + u * (b - a) + v * (c - a)
    return pts


def _torus_points(n: int, rng: np.random.RandomState, R: float = 0.7, r: float = 0.3) -> np.ndarray:
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return np.stack([x, y, z], axis=1)


def _quantize_coords(pts: np.ndarray, lo: float = -1.5, hi: float = 1.5) -> list[int]:
    scaled = (pts - lo) / (hi - lo) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8).flatten().tolist()


def generate_shape_data(n_samples: int, seed: int = 42) -> bytes:
    rng = np.random.RandomState(seed)
    generators = [_sphere_points, _cube_points, _tetrahedron_points, _torus_points]
    data = []
    for i in range(n_samples):
        cls = i % SHAPE_CLASSES
        pts = generators[cls](SHAPE_N_POINTS, rng)
        rot = _random_rotation_matrix(rng)
        pts = pts @ rot.T
        coord_bytes = _quantize_coords(pts)
        sample = [SHAPE_MARKER_START] + coord_bytes + [SHAPE_MARKER_END, cls]
        data.extend(sample)
    return bytes(data)


def _nbody_simulate(n_particles: int, n_steps: int, rng: np.random.RandomState,
                    dt: float = 0.02, G: float = 1.0, eps: float = 0.1) -> np.ndarray:
    pos = rng.uniform(-1, 1, (n_particles, 2))
    vel = rng.uniform(-0.3, 0.3, (n_particles, 2))
    mass = np.ones(n_particles)
    trajectory = np.zeros((n_steps + 1, n_particles, 4))
    trajectory[0, :, :2] = pos
    trajectory[0, :, 2:] = vel
    for t in range(n_steps):
        acc = np.zeros_like(pos)
        for i in range(n_particles):
            for j in range(n_particles):
                if i == j:
                    continue
                diff = pos[j] - pos[i]
                r2 = np.sum(diff ** 2) + eps ** 2
                acc[i] += G * mass[j] * diff / (r2 ** 1.5)
        vel = vel + acc * dt
        pos = pos + vel * dt
        pos = np.clip(pos, -2, 2)
        vel = np.clip(vel, -1, 1)
        trajectory[t + 1, :, :2] = pos
        trajectory[t + 1, :, 2:] = vel
    return trajectory


def _quantize_nbody(traj: np.ndarray) -> list[int]:
    result = []
    for t in range(traj.shape[0]):
        for p in range(traj.shape[1]):
            px = int(np.clip((traj[t, p, 0] + 2) / 4 * 255, 0, 255))
            py = int(np.clip((traj[t, p, 1] + 2) / 4 * 255, 0, 255))
            vx = int(np.clip((traj[t, p, 2] + 1) / 2 * 255, 0, 255))
            vy = int(np.clip((traj[t, p, 3] + 1) / 2 * 255, 0, 255))
            result.extend([px, py, vx, vy])
    return result


def generate_nbody_data(n_samples: int, seed: int = 42) -> bytes:
    rng = np.random.RandomState(seed)
    data = []
    for _ in range(n_samples):
        traj = _nbody_simulate(NBODY_N_PARTICLES, NBODY_N_STEPS, rng)
        history = _quantize_nbody(traj[:NBODY_N_STEPS])
        target = _quantize_nbody(traj[NBODY_N_STEPS:NBODY_N_STEPS + 1])
        sample = [NBODY_MARKER_START] + history + [NBODY_MARKER_END] + target
        data.extend(sample)
    return bytes(data)


def make_mixed_data(lang_data: bytes, shape_data: bytes, nbody_data: bytes,
                    chunk_size: int = 256) -> bytes:
    lang_chunks = [lang_data[i:i+chunk_size] for i in range(0, len(lang_data) - chunk_size, chunk_size)]
    shape_chunks = [shape_data[i:i+chunk_size] for i in range(0, len(shape_data) - chunk_size, chunk_size)]
    nbody_chunks = [nbody_data[i:i+chunk_size] for i in range(0, len(nbody_data) - chunk_size, chunk_size)]
    result = []
    li, si, ni = 0, 0, 0
    while li < len(lang_chunks) or si < len(shape_chunks) or ni < len(nbody_chunks):
        if li < len(lang_chunks):
            result.append(lang_chunks[li])
            li += 1
        if li < len(lang_chunks):
            result.append(lang_chunks[li])
            li += 1
        if si < len(shape_chunks):
            result.append(shape_chunks[si])
            si += 1
        if ni < len(nbody_chunks):
            result.append(nbody_chunks[ni])
            ni += 1
    return b"".join(result)


@torch.no_grad()
def evaluate(model: LM, dataloader: DataLoader, max_batches: int = 50) -> dict:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    losses = []
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        batch = batch.to(DEVICE)
        logits, _, _ = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1), reduction="sum")
        n = batch[:, 1:].numel()
        total_loss += loss.item()
        total_tokens += n
        losses.append(loss.item() / n)
    model.train()
    mean_loss = total_loss / max(total_tokens, 1)
    return {"mean_loss": mean_loss, "bpb": mean_loss / math.log(2), "n_batches": len(losses)}


def collect_spike_stats(aux: dict) -> dict:
    stats = {"firing_rates": [], "dead_pct": 0.0, "saturated_pct": 0.0}
    all_spikes = []
    for layer_aux in aux.values():
        if isinstance(layer_aux, dict):
            for key in ("k_spikes", "v_spikes"):
                if key in layer_aux:
                    s = layer_aux[key]
                    fr = (s != 0).float().mean().item()
                    stats["firing_rates"].append(fr)
                    all_spikes.append(s)
    if all_spikes:
        combined = torch.cat([s.reshape(-1) for s in all_spikes])
        per_neuron = []
        for s in all_spikes:
            flat = s.reshape(-1, s.shape[-1])
            per_neuron.append((flat != 0).float().mean(0))
        if per_neuron:
            rates = torch.cat(per_neuron).cpu().numpy()
            stats["dead_pct"] = float((rates < 0.05).mean())
            stats["saturated_pct"] = float((rates > 0.95).mean())
    return stats


def train_model(cfg: TrainConfig, train_data: bytes, val_data: bytes, name: str, init_checkpoint: str | None = None) -> dict:
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === Training {name} ===")
    model = LM(cfg).to(DEVICE)
    if init_checkpoint and Path(init_checkpoint).exists():
        state_dict = torch.load(init_checkpoint, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] loaded checkpoint: {init_checkpoint}")
    n_params = model.count_params()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] params: {n_params:,}")

    train_ds = ByteTextDataset(train_data, cfg.seq_len)
    val_ds = ByteTextDataset(val_data, cfg.seq_len)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    decay_params, no_decay_params = [], []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or "norm" in pname or "bias" in pname:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=cfg.lr, betas=(0.9, 0.95))

    def lr_fn(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    history = {"train_loss": [], "val_bpb": [], "steps": [], "spike_stats": [], "state_norms": []}
    step = 0
    best_val_bpb = float("inf")
    start_time = time.time()

    for epoch in range(100):
        if step >= cfg.max_steps:
            break
        for batch in train_dl:
            if step >= cfg.max_steps:
                break
            batch = batch.to(DEVICE)
            logits, _, aux = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()
            scheduler.step()

            if step % 50 == 0:
                elapsed = time.time() - start_time
                print(f"[{datetime.now().strftime('%H:%M:%S')}] step {step}/{cfg.max_steps} loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e} t={elapsed:.0f}s")
                history["train_loss"].append(loss.item())
                history["steps"].append(step)

                spike_stats = collect_spike_stats(aux)
                history["spike_stats"].append(spike_stats)

                norms = []
                for la in aux.values():
                    if isinstance(la, dict) and "state_norm" in la:
                        norms.append(la["state_norm"])
                history["state_norms"].append(norms)

            if step > 0 and step % cfg.eval_interval == 0:
                val_result = evaluate(model, val_dl)
                history["val_bpb"].append(val_result["bpb"])
                if val_result["bpb"] < best_val_bpb:
                    best_val_bpb = val_result["bpb"]
                    torch.save(model.state_dict(), OUTPUT_DIR / f"{name}_best.pt")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] val bpb={val_result['bpb']:.4f} best={best_val_bpb:.4f}")

            step += 1

    final_val = evaluate(model, val_dl)
    history["val_bpb"].append(final_val["bpb"])
    if final_val["bpb"] < best_val_bpb:
        best_val_bpb = final_val["bpb"]

    total_time = time.time() - start_time
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {name} done. final_bpb={final_val['bpb']:.4f} best_bpb={best_val_bpb:.4f} time={total_time:.0f}s")

    return {
        "name": name,
        "param_count": n_params,
        "final_val_bpb": final_val["bpb"],
        "best_val_bpb": best_val_bpb,
        "total_steps": step,
        "total_time": total_time,
        "history": history,
        "spike_firing_rate": float(np.mean([np.mean(s["firing_rates"]) for s in history["spike_stats"] if s.get("firing_rates")])) if any(s.get("firing_rates") for s in history["spike_stats"]) else 0,
        "spike_dead_ratio": history["spike_stats"][-1].get("dead_pct", 0) if history["spike_stats"] else 0,
    }


def plot_results(todorov_result: dict, baseline_result: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(todorov_result["history"]["steps"], todorov_result["history"]["train_loss"], label="Todorov", alpha=0.7)
    ax.plot(baseline_result["history"]["steps"], baseline_result["history"]["train_loss"], label="Transformer", alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    t_bpb = todorov_result["history"]["val_bpb"]
    b_bpb = baseline_result["history"]["val_bpb"]
    ax.plot(range(len(t_bpb)), t_bpb, "o-", label="Todorov")
    ax.plot(range(len(b_bpb)), b_bpb, "s-", label="Transformer")
    ax.set_xlabel("Eval Checkpoint")
    ax.set_ylabel("BPB")
    ax.set_title("Validation BPB")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    spike_stats = todorov_result["history"]["spike_stats"]
    if spike_stats and any(s.get("firing_rates") for s in spike_stats):
        frs = [np.mean(s["firing_rates"]) if s["firing_rates"] else 0 for s in spike_stats]
        ax.plot(todorov_result["history"]["steps"][:len(frs)], frs)
        ax.axhline(y=0.3, color="r", linestyle="--", alpha=0.5, label="Target min (30%)")
        ax.axhline(y=0.6, color="r", linestyle="--", alpha=0.5, label="Target max (60%)")
        ax.legend()
    ax.set_xlabel("Step")
    ax.set_ylabel("Firing Rate")
    ax.set_title("Spike Firing Rate")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    state_norms = todorov_result["history"]["state_norms"]
    if state_norms and any(n for n in state_norms):
        mean_norms = [np.mean(n) if n else 0 for n in state_norms]
        ax.plot(todorov_result["history"]["steps"][:len(mean_norms)], mean_norms)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean State Norm")
    ax.set_title("KDA State Norm Evolution")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
    plt.close()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] saved training_curves.png")


@torch.no_grad()
def validate_spikes(model: LM, dataloader: DataLoader, max_batches: int = 20) -> dict:
    model.eval()
    all_spikes: list[torch.Tensor] = []
    all_pre: list[torch.Tensor] = []

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        batch = batch.to(DEVICE)
        _, _, aux = model(batch[:, :-1])
        for layer_aux in aux.values():
            if not isinstance(layer_aux, dict):
                continue
            for key in ("k_spikes", "v_spikes"):
                if key in layer_aux:
                    s = layer_aux[key]
                    all_spikes.append(s.reshape(-1, s.shape[-1]).cpu())
            pre_key_map = {"k_spikes": "pre_k", "v_spikes": "pre_v"}
            for skey, pkey in pre_key_map.items():
                if pkey in layer_aux:
                    p = layer_aux[pkey]
                    all_pre.append(p.reshape(-1, p.shape[-1]).cpu())

    if not all_spikes or not all_pre:
        model.train()
        return {"mi": 0.0, "cka": 0.0, "n_samples": 0}

    spikes = torch.cat(all_spikes, dim=0).float().numpy()
    pre = torch.cat(all_pre, dim=0).float().numpy()
    n = min(spikes.shape[0], pre.shape[0], 10000)
    spikes = spikes[:n]
    pre = pre[:n]
    n_dims = min(8, spikes.shape[1], pre.shape[1])

    mi_values = []
    for dim in range(n_dims):
        s_col = spikes[:, dim]
        r_col = pre[:, dim]
        r_min, r_max = float(r_col.min()), float(r_col.max())
        if abs(r_max - r_min) < 1e-12:
            continue
        n_bins = 32
        r_bins = np.digitize(r_col, np.linspace(r_min, r_max, n_bins + 1)[1:-1])
        s_disc = np.ones_like(s_col, dtype=np.int32)
        s_disc[s_col > 1e-6] = 2
        s_disc[s_col < -1e-6] = 0
        joint = np.zeros((3, n_bins), dtype=np.float64)
        for idx in range(n):
            r_bin = max(0, min(int(r_bins[idx]), n_bins - 1))
            joint[s_disc[idx], r_bin] += 1.0
        joint = joint / (joint.sum() + 1e-12)
        p_s = joint.sum(axis=1, keepdims=True) + 1e-12
        p_r = joint.sum(axis=0, keepdims=True) + 1e-12
        mi = float(np.sum(joint * np.log2((joint + 1e-12) / (p_s * p_r))))
        mi_values.append(max(0.0, mi))

    mi_mean = float(np.mean(mi_values)) if mi_values else 0.0

    sx = spikes[:n] - spikes[:n].mean(axis=0, keepdims=True)
    sy = pre[:n] - pre[:n].mean(axis=0, keepdims=True)
    hsic_xy = np.linalg.norm(sx.T @ sy, ord="fro") ** 2
    hsic_xx = np.linalg.norm(sx.T @ sx, ord="fro") ** 2
    hsic_yy = np.linalg.norm(sy.T @ sy, ord="fro") ** 2
    denom = math.sqrt(float(hsic_xx * hsic_yy)) + 1e-12
    cka = float(hsic_xy / denom)

    model.train()
    return {"mi": mi_mean, "cka": cka, "n_samples": n}


@torch.no_grad()
def passkey_retrieval_test(
    model: LM,
    context_length: int,
    num_trials: int = 10,
    passkey_length: int = 5,
    chunk_size: int = 256,
) -> dict:
    model.eval()
    rng = np.random.RandomState(SEED + context_length)
    correct = 0
    for trial in range(num_trials):
        passkey = rng.randint(10**(passkey_length-1), 10**passkey_length)
        passkey_bytes = list(str(passkey).encode("ascii"))
        marker_start = [255, 254, 253]
        marker_end = [253, 254, 255]
        marker_query = [252, 251, 250]
        filler = rng.randint(32, 127, size=context_length * 2).tolist()
        overhead = len(marker_start) + len(passkey_bytes) + len(marker_end) + len(marker_query)
        max_insert = max(1, context_length - overhead - 20)
        insert_pos = rng.randint(5, min(max_insert, context_length // 3))
        before = filler[:insert_pos]
        middle = filler[insert_pos:insert_pos + context_length - overhead - insert_pos]
        sequence = before + marker_start + passkey_bytes + marker_end + middle + marker_query
        sequence = sequence[:context_length]
        if len(sequence) < context_length:
            sequence = sequence + filler[len(sequence):context_length]
        input_tensor = torch.tensor([sequence[:context_length]], dtype=torch.long, device=DEVICE)
        states = None
        for start in range(0, input_tensor.shape[1], chunk_size):
            end = min(start + chunk_size, input_tensor.shape[1])
            chunk = input_tensor[:, start:end]
            logits, states, _ = model(chunk, states=states, offset=start)
        generated = []
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())
        for step in range(passkey_length - 1):
            logits, states, _ = model(next_token, states=states, offset=input_tensor.shape[1] + step)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(next_token.item())
        generated_str = bytes(generated).decode("ascii", errors="replace")
        if generated_str == str(passkey):
            correct += 1
    return {"accuracy": correct / num_trials, "correct": correct, "total": num_trials, "context_length": context_length}


@torch.no_grad()
def measure_perplexity_at_length(
    model: LM,
    data: bytes,
    context_length: int,
    num_windows: int = 5,
    chunk_size: int = 256,
) -> dict:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for window_idx in range(num_windows):
        start = window_idx * context_length
        if start + context_length + 1 > len(data):
            break
        chunk_data = list(data[start:start + context_length + 1])
        input_tensor = torch.tensor([chunk_data], dtype=torch.long, device=DEVICE)
        states = None
        all_logits = []
        for cs in range(0, input_tensor.shape[1] - 1, chunk_size):
            ce = min(cs + chunk_size, input_tensor.shape[1] - 1)
            chunk = input_tensor[:, cs:ce]
            logits, states, _ = model(chunk, states=states, offset=cs)
            all_logits.append(logits)
        all_logits_cat = torch.cat(all_logits, dim=1)
        targets = input_tensor[:, 1:all_logits_cat.shape[1]+1]
        loss = F.cross_entropy(all_logits_cat.reshape(-1, all_logits_cat.shape[-1]), targets.reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += targets.numel()
    mean_loss = total_loss / max(total_tokens, 1)
    return {"bpb": mean_loss / math.log(2), "context_length": context_length, "total_tokens": total_tokens}


@torch.no_grad()
def selective_copy_test(
    model: LM,
    context_length: int,
    num_trials: int = 20,
    copy_length: int = 8,
    chunk_size: int = 256,
) -> dict:
    model.eval()
    rng = np.random.RandomState(SEED + context_length + 7)
    correct = 0
    for trial in range(num_trials):
        target_bytes = rng.randint(48, 58, size=copy_length).tolist()
        marker = [255, 254]
        end_marker = [254, 255]
        query_marker = [253, 252]
        filler = rng.randint(65, 91, size=context_length).tolist()
        insert_pos = rng.randint(10, context_length // 3)
        query_pos = context_length - copy_length - len(query_marker) - 5
        sequence = filler[:insert_pos] + marker + target_bytes + end_marker
        sequence = sequence + filler[len(sequence):query_pos] + query_marker
        sequence = sequence[:context_length]
        input_tensor = torch.tensor([sequence], dtype=torch.long, device=DEVICE)
        states = None
        for start in range(0, input_tensor.shape[1], chunk_size):
            end = min(start + chunk_size, input_tensor.shape[1])
            logits, states, _ = model(input_tensor[:, start:end], states=states, offset=start)
        generated = []
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())
        for step in range(copy_length - 1):
            logits, states, _ = model(next_token, states=states, offset=input_tensor.shape[1] + step)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(next_token.item())
        if generated == target_bytes:
            correct += 1
    return {"accuracy": correct / num_trials, "correct": correct, "total": num_trials, "context_length": context_length}


def run_phase2_evaluation(model: LM, val_data: bytes) -> dict:
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === PHASE 2: CONTEXT EXTENSION ===", flush=True)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] selective copy tests...", flush=True)
    copy_results = {}
    for ctx_len in [256, 512, 1024, 2048]:
        t0 = time.time()
        cr = selective_copy_test(model, ctx_len, num_trials=20)
        elapsed = time.time() - t0
        copy_results[ctx_len] = cr
        print(f"  copy @{ctx_len}: {cr['accuracy']:.1%} ({cr['correct']}/{cr['total']}) [{elapsed:.0f}s]", flush=True)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] passkey retrieval tests...", flush=True)
    passkey_results = {}
    for ctx_len in [256, 1024, 4096]:
        t0 = time.time()
        pr = passkey_retrieval_test(model, ctx_len, num_trials=20)
        elapsed = time.time() - t0
        passkey_results[ctx_len] = pr
        print(f"  passkey @{ctx_len}: {pr['accuracy']:.1%} ({pr['correct']}/{pr['total']}) [{elapsed:.0f}s]", flush=True)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] perplexity scaling...", flush=True)
    ppl_results = {}
    for ctx_len in [256, 512, 1024, 2048, 4096]:
        if ctx_len * 6 > len(val_data):
            break
        t0 = time.time()
        pr = measure_perplexity_at_length(model, val_data, ctx_len, num_windows=5)
        elapsed = time.time() - t0
        ppl_results[ctx_len] = pr
        print(f"  BPB @{ctx_len}: {pr['bpb']:.4f} [{elapsed:.0f}s]", flush=True)

    return {"copy": copy_results, "passkey": passkey_results, "perplexity": ppl_results}


@torch.no_grad()
def shape_classification_test(model: LM, n_trials: int = 200, seed: int = 999) -> dict:
    model.eval()
    rng = np.random.RandomState(seed)
    generators = [_sphere_points, _cube_points, _tetrahedron_points, _torus_points]
    correct = 0
    per_class = {c: {"correct": 0, "total": 0} for c in range(SHAPE_CLASSES)}
    for trial in range(n_trials):
        cls = trial % SHAPE_CLASSES
        pts = generators[cls](SHAPE_N_POINTS, rng)
        rot = _random_rotation_matrix(rng)
        pts = pts @ rot.T
        coord_bytes = _quantize_coords(pts)
        seq = [SHAPE_MARKER_START] + coord_bytes + [SHAPE_MARKER_END]
        input_tensor = torch.tensor([seq], dtype=torch.long, device=DEVICE)
        logits, _, _ = model(input_tensor)
        pred = logits[0, -1, :SHAPE_CLASSES].argmax().item()
        per_class[cls]["total"] += 1
        if pred == cls:
            correct += 1
            per_class[cls]["correct"] += 1
    model.train()
    return {
        "accuracy": correct / n_trials,
        "correct": correct,
        "total": n_trials,
        "per_class": {k: v["correct"] / max(v["total"], 1) for k, v in per_class.items()},
    }


@torch.no_grad()
def nbody_prediction_test(model: LM, n_trials: int = 200, seed: int = 999) -> dict:
    model.eval()
    rng = np.random.RandomState(seed)
    total_mae = 0.0
    for _ in range(n_trials):
        traj = _nbody_simulate(NBODY_N_PARTICLES, NBODY_N_STEPS, rng)
        history = _quantize_nbody(traj[:NBODY_N_STEPS])
        target = _quantize_nbody(traj[NBODY_N_STEPS:NBODY_N_STEPS + 1])
        seq = [NBODY_MARKER_START] + history + [NBODY_MARKER_END]
        input_tensor = torch.tensor([seq], dtype=torch.long, device=DEVICE)
        logits, states, _ = model(input_tensor)
        n_target = len(target)
        predicted = []
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        predicted.append(next_token.item())
        for step in range(n_target - 1):
            logits, states, _ = model(next_token, states=states, offset=input_tensor.shape[1] + step)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            predicted.append(next_token.item())
        mae = np.mean(np.abs(np.array(predicted) - np.array(target)))
        total_mae += mae
    model.train()
    return {"mae": total_mae / n_trials, "n_trials": n_trials}


def equivariance_test(n_samples: int = 200, seed: int = 42) -> dict:
    rng_t = torch.Generator()
    rng_t.manual_seed(seed)
    results = {}
    for angle_name, angle in [("30deg", math.pi / 6), ("60deg", math.pi / 3), ("90deg", math.pi / 2)]:
        rotor = torch.zeros(16)
        rotor[0] = math.cos(angle / 2)
        rotor[8] = math.sin(angle / 2)
        errors = []
        for _ in range(n_samples):
            x = torch.randn(16, generator=rng_t)
            gp_xx = sparse_gp(x.unsqueeze(0), x.unsqueeze(0)).squeeze(0)
            rotate_after = sandwich_gp(rotor.unsqueeze(0), gp_xx.unsqueeze(0)).squeeze(0)
            x_rot = sandwich_gp(rotor.unsqueeze(0), x.unsqueeze(0)).squeeze(0)
            gp_rotrot = sparse_gp(x_rot.unsqueeze(0), x_rot.unsqueeze(0)).squeeze(0)
            norm_a = torch.norm(rotate_after)
            if norm_a > 1e-8:
                err = torch.norm(rotate_after - gp_rotrot) / norm_a
                errors.append(err.item())
        results[angle_name] = float(np.mean(errors)) if errors else 0.0
    return results


def run_phase3_evaluation(
    model_gp: LM, model_nogp: LM, model_baseline: LM,
    val_data: bytes,
) -> dict:
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === PHASE 3: SPATIAL MODULE VALIDATION ===", flush=True)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] shape classification (todorov+gp)...", flush=True)
    t0 = time.time()
    shape_gp = shape_classification_test(model_gp, n_trials=200, seed=999)
    print(f"  accuracy: {shape_gp['accuracy']:.1%} [{time.time()-t0:.0f}s]", flush=True)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] shape classification (transformer)...", flush=True)
    t0 = time.time()
    shape_bl = shape_classification_test(model_baseline, n_trials=200, seed=999)
    print(f"  accuracy: {shape_bl['accuracy']:.1%} [{time.time()-t0:.0f}s]", flush=True)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] n-body dynamics (todorov+gp)...", flush=True)
    t0 = time.time()
    nbody_gp = nbody_prediction_test(model_gp, n_trials=200, seed=999)
    print(f"  mae: {nbody_gp['mae']:.2f} [{time.time()-t0:.0f}s]", flush=True)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] n-body dynamics (transformer)...", flush=True)
    t0 = time.time()
    nbody_bl = nbody_prediction_test(model_baseline, n_trials=200, seed=999)
    print(f"  mae: {nbody_bl['mae']:.2f} [{time.time()-t0:.0f}s]", flush=True)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] equivariance test...", flush=True)
    t0 = time.time()
    equiv = equivariance_test(n_samples=200, seed=42)
    print(f"  60deg error: {equiv['60deg']:.6f} [{time.time()-t0:.0f}s]", flush=True)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] language degradation check...", flush=True)
    val_ds_lang = ByteTextDataset(val_data, 256)
    val_dl_lang = DataLoader(val_ds_lang, batch_size=32, shuffle=False, collate_fn=collate_fn)
    bpb_gp = evaluate(model_gp, val_dl_lang)["bpb"]
    bpb_nogp = evaluate(model_nogp, val_dl_lang)["bpb"]
    degrade_pct = (bpb_gp - bpb_nogp) / max(bpb_nogp, 1e-6) * 100
    print(f"  BPB with GP: {bpb_gp:.4f}, without GP: {bpb_nogp:.4f}, degradation: {degrade_pct:.1f}%", flush=True)

    gate_classify = shape_gp["accuracy"] > shape_bl["accuracy"]
    gate_dynamics = nbody_gp["mae"] < nbody_bl["mae"]
    gate_equiv = equiv["60deg"] < 0.05
    gate_lang = degrade_pct <= 10.0
    spatial_pass_count = sum([gate_classify, gate_dynamics, gate_equiv])
    phase3_pass = spatial_pass_count >= 2 and gate_lang

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === PHASE 3 GATES ===")
    print(f"  spatial_classify:    {'PASS' if gate_classify else 'FAIL'} (GP={shape_gp['accuracy']:.1%} vs BL={shape_bl['accuracy']:.1%})")
    print(f"  spatial_dynamics:    {'PASS' if gate_dynamics else 'FAIL'} (GP mae={nbody_gp['mae']:.2f} vs BL mae={nbody_bl['mae']:.2f})")
    print(f"  equivariance_test:   {'PASS' if gate_equiv else 'FAIL'} (error={equiv['60deg']:.6f})")
    print(f"  language_no_degrade: {'PASS' if gate_lang else 'FAIL'} (degradation={degrade_pct:.1f}%)")
    print(f"  OVERALL:             {'PASS' if phase3_pass else 'FAIL'} ({spatial_pass_count}/3 spatial + lang={'ok' if gate_lang else 'fail'})")

    return {
        "shape_gp": shape_gp,
        "shape_baseline": shape_bl,
        "nbody_gp": nbody_gp,
        "nbody_baseline": nbody_bl,
        "equivariance": equiv,
        "bpb_with_gp": bpb_gp,
        "bpb_without_gp": bpb_nogp,
        "language_degradation_pct": degrade_pct,
        "gate_classify": bool(gate_classify),
        "gate_dynamics": bool(gate_dynamics),
        "gate_equiv": bool(gate_equiv),
        "gate_lang": bool(gate_lang),
        "spatial_pass_count": spatial_pass_count,
        "phase3_pass": bool(phase3_pass),
    }


def main() -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] downloading WikiText-2...")
    train_data, val_data = download_wikitext2()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] train: {len(train_data):,} bytes, val: {len(val_data):,} bytes")

    if PHASE >= 3:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Phase 3: spatial module validation", flush=True)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] generating spatial data...", flush=True)
        shape_train = generate_shape_data(4000, seed=SEED)
        shape_test = generate_shape_data(800, seed=SEED + 100)
        nbody_train = generate_nbody_data(2000, seed=SEED)
        nbody_test = generate_nbody_data(400, seed=SEED + 100)
        mixed_train = make_mixed_data(train_data, shape_train, nbody_train)
        mixed_val = make_mixed_data(val_data, shape_test, nbody_test)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] mixed train: {len(mixed_train):,} bytes, mixed val: {len(mixed_val):,} bytes", flush=True)

        gp_cfg = TrainConfig(spatial_mode=True, max_steps=500)
        gp_result = train_model(gp_cfg, mixed_train, mixed_val, "todorov_gp")
        gp_checkpoint = str(OUTPUT_DIR / "todorov_gp_best.pt")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        nogp_cfg = TrainConfig(spatial_mode=False, max_steps=200)
        nogp_result = train_model(nogp_cfg, train_data, val_data, "todorov_nogp")
        nogp_checkpoint = str(OUTPUT_DIR / "todorov_nogp_best.pt")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        bl_cfg = TrainConfig(
            layer_pattern=("Transformer",) * 8, use_spikes=False,
            spatial_mode=False, max_steps=200,
        )
        bl_result = train_model(bl_cfg, mixed_train, mixed_val, "transformer")
        bl_checkpoint = str(OUTPUT_DIR / "transformer_best.pt")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[{datetime.now().strftime('%H:%M:%S')}] loading models for evaluation...", flush=True)
        model_gp = LM(gp_cfg).to(DEVICE)
        if Path(gp_checkpoint).exists():
            model_gp.load_state_dict(torch.load(gp_checkpoint, map_location=DEVICE, weights_only=True), strict=False)
        model_nogp = LM(nogp_cfg).to(DEVICE)
        if Path(nogp_checkpoint).exists():
            model_nogp.load_state_dict(torch.load(nogp_checkpoint, map_location=DEVICE, weights_only=True), strict=False)
        model_bl = LM(bl_cfg).to(DEVICE)
        if Path(bl_checkpoint).exists():
            model_bl.load_state_dict(torch.load(bl_checkpoint, map_location=DEVICE, weights_only=True), strict=False)

        phase3_results = run_phase3_evaluation(model_gp, model_nogp, model_bl, val_data)

        del model_bl
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] running SpikingBrain validation...")
        val_ds_sb = ByteTextDataset(val_data, gp_cfg.seq_len)
        val_dl_sb = DataLoader(val_ds_sb, batch_size=gp_cfg.batch_size, shuffle=False, collate_fn=collate_fn)
        spike_val = validate_spikes(model_gp, val_dl_sb)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] MI={spike_val['mi']:.4f} CKA={spike_val['cka']:.4f} (n={spike_val['n_samples']})")

        del model_gp, model_nogp
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results = {
            "phase": 3,
            "best_val_bpb": gp_result["best_val_bpb"],
            "baseline_best_bpb": bl_result["best_val_bpb"],
            "param_count_gp": gp_result["param_count"],
            "param_count_nogp": nogp_result["param_count"],
            "param_count_baseline": bl_result["param_count"],
            "spike_firing_rate": gp_result["spike_firing_rate"],
            "spike_dead_ratio": gp_result["spike_dead_ratio"],
            "spike_mi": spike_val["mi"],
            "spike_cka": spike_val["cka"],
            "training_time_gp": gp_result["total_time"],
            "training_time_nogp": nogp_result["total_time"],
            "training_time_baseline": bl_result["total_time"],
            "phase3": phase3_results,
            "timestamp": datetime.now().isoformat(),
            "device": str(DEVICE),
            "seed": SEED,
        }

    elif PHASE >= 2:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Phase 2: progressive context training (fla={FLA_AVAILABLE})", flush=True)
        stages = [
            ("s256",  256,  200,  32, 3e-4, 50),
            ("s512",  512,  200,  16, 1e-4, 30),
            ("s1024", 1024, 200,  16, 1e-4, 30),
            ("s2048", 2048, 200,  8,  1e-4, 30),
        ]
        prev_checkpoint = None
        final_stage_name = ""
        stage_summaries = []
        for stage_name, seq_len, steps, bs, lr, warmup in stages:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            stage_cfg = TrainConfig(seq_len=seq_len, max_steps=steps, batch_size=bs, lr=lr, warmup_steps=warmup)
            todorov_result = train_model(stage_cfg, train_data, val_data, f"todorov_{stage_name}", init_checkpoint=prev_checkpoint)
            prev_checkpoint = str(OUTPUT_DIR / f"todorov_{stage_name}_best.pt")
            final_stage_name = stage_name
            todorov_cfg = stage_cfg
            stage_summary = {
                "stage": stage_name, "seq_len": seq_len, "steps": steps,
                "best_bpb": todorov_result["best_val_bpb"],
                "final_bpb": todorov_result["final_val_bpb"],
                "time": todorov_result["total_time"],
                "s_per_step": todorov_result["total_time"] / max(steps, 1),
            }
            stage_summaries.append(stage_summary)
            with open(OUTPUT_DIR / f"stage_{stage_name}.json", "w") as f:
                json.dump(stage_summary, f, indent=2, default=str)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {stage_name} DONE: BPB={todorov_result['best_val_bpb']:.4f} time={todorov_result['total_time']:.0f}s ({todorov_result['total_time']/max(steps,1):.1f}s/step)", flush=True)
        final_checkpoint = prev_checkpoint
        baseline_cfg = TrainConfig(
            seq_len=2048, max_steps=200, batch_size=8, lr=3e-4,
            layer_pattern=("Transformer",)*8, use_spikes=False,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        baseline_result = train_model(baseline_cfg, train_data, val_data, "transformer")

        plot_results(todorov_result, baseline_result)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] running SpikingBrain validation...")
        todorov_model = LM(todorov_cfg).to(DEVICE)
        if Path(final_checkpoint).exists():
            todorov_model.load_state_dict(torch.load(final_checkpoint, map_location=DEVICE, weights_only=True), strict=False)
        val_ds_sb = ByteTextDataset(val_data, todorov_cfg.seq_len)
        val_dl_sb = DataLoader(val_ds_sb, batch_size=todorov_cfg.batch_size, shuffle=False, collate_fn=collate_fn)
        spike_val = validate_spikes(todorov_model, val_dl_sb)
        del todorov_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] MI={spike_val['mi']:.4f} CKA={spike_val['cka']:.4f} (n={spike_val['n_samples']})")

        bpb_ratio = todorov_result["best_val_bpb"] / max(baseline_result["best_val_bpb"], 1e-6)

        results = {
            "phase": 2,
            "final_val_bpb": todorov_result["final_val_bpb"],
            "best_val_bpb": todorov_result["best_val_bpb"],
            "baseline_best_bpb": baseline_result["best_val_bpb"],
            "bpb_ratio": bpb_ratio,
            "total_steps": todorov_result["total_steps"],
            "param_count": todorov_result["param_count"],
            "spike_firing_rate": todorov_result["spike_firing_rate"],
            "spike_dead_ratio": todorov_result["spike_dead_ratio"],
            "spike_mi": spike_val["mi"],
            "spike_cka": spike_val["cka"],
            "training_time_todorov": todorov_result["total_time"],
            "training_time_baseline": baseline_result["total_time"],
            "timestamp": datetime.now().isoformat(),
            "device": str(DEVICE),
            "seed": SEED,
        }

        todorov_model_p2 = LM(todorov_cfg).to(DEVICE)
        if Path(final_checkpoint).exists():
            todorov_model_p2.load_state_dict(torch.load(final_checkpoint, map_location=DEVICE, weights_only=True), strict=False)
        phase2_results = run_phase2_evaluation(todorov_model_p2, val_data)
        del todorov_model_p2
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        results["phase2"] = {
            "copy": {str(k): v for k, v in phase2_results.get("copy", {}).items()},
            "passkey": {str(k): v for k, v in phase2_results.get("passkey", {}).items()},
            "perplexity": {str(k): v for k, v in phase2_results.get("perplexity", {}).items()},
        }
    else:
        todorov_cfg = TrainConfig()
        baseline_cfg = BASELINE_CONFIG
        todorov_result = train_model(todorov_cfg, train_data, val_data, "todorov")
        final_checkpoint = str(OUTPUT_DIR / "todorov_best.pt")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        baseline_result = train_model(baseline_cfg, train_data, val_data, "transformer")

        plot_results(todorov_result, baseline_result)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] running SpikingBrain validation...")
        todorov_model = LM(todorov_cfg).to(DEVICE)
        if Path(final_checkpoint).exists():
            todorov_model.load_state_dict(torch.load(final_checkpoint, map_location=DEVICE, weights_only=True), strict=False)
        val_ds_sb = ByteTextDataset(val_data, todorov_cfg.seq_len)
        val_dl_sb = DataLoader(val_ds_sb, batch_size=todorov_cfg.batch_size, shuffle=False, collate_fn=collate_fn)
        spike_val = validate_spikes(todorov_model, val_dl_sb)
        del todorov_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] MI={spike_val['mi']:.4f} CKA={spike_val['cka']:.4f} (n={spike_val['n_samples']})")

        bpb_ratio = todorov_result["best_val_bpb"] / max(baseline_result["best_val_bpb"], 1e-6)

        results = {
            "phase": 1,
            "final_val_bpb": todorov_result["final_val_bpb"],
            "best_val_bpb": todorov_result["best_val_bpb"],
            "baseline_best_bpb": baseline_result["best_val_bpb"],
            "bpb_ratio": bpb_ratio,
            "param_count": todorov_result["param_count"],
            "spike_firing_rate": todorov_result["spike_firing_rate"],
            "spike_mi": spike_val["mi"],
            "spike_cka": spike_val["cka"],
            "timestamp": datetime.now().isoformat(),
            "device": str(DEVICE),
            "seed": SEED,
        }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    bundle_path = OUTPUT_DIR / "evidence_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w") as zf:
        for fp in OUTPUT_DIR.iterdir():
            if fp.name != "evidence_bundle.zip" and fp.is_file():
                zf.write(fp, fp.name)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] evidence bundle: {bundle_path}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] FATAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        error_results = {"status": "error", "error": str(e), "timestamp": datetime.now().isoformat()}
        try:
            with open(OUTPUT_DIR / "results.json", "w") as f:
                json.dump(error_results, f, indent=2, default=str)
        except Exception:
            pass
        raise
