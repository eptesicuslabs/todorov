from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class DecodeHead(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.output_proj(self.norm(x))

    def tie_weights(self, embedding_weight: Tensor) -> None:
        self.output_proj.weight = embedding_weight


class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(Tensor(d_model).fill_(1.0))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = (x.float().pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight
