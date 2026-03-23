from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class TodorovEmbedding(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 131072,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, std=0.02)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.token_embedding(input_ids)
