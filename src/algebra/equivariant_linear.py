from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.algebra.multivector import GRADE_RANGES, NUM_COMPONENTS


class EquivariantLinear(nn.Module):

    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out

        self.grade_weights = nn.ParameterList()
        self.grade_biases = nn.ParameterList()

        for start, end in GRADE_RANGES:
            grade_dim = end - start
            weight = nn.Parameter(torch.randn(channels_out, channels_in, grade_dim, grade_dim) * 0.02)
            bias = nn.Parameter(torch.zeros(channels_out, grade_dim))
            self.grade_weights.append(weight)
            self.grade_biases.append(bias)

    def forward(self, x: Tensor) -> Tensor:
        batch_dims = x.shape[:-2]
        channels_in = x.shape[-2]
        assert channels_in == self.channels_in
        assert x.shape[-1] == NUM_COMPONENTS

        result = torch.zeros(
            *batch_dims, self.channels_out, NUM_COMPONENTS,
            device=x.device, dtype=x.dtype,
        )

        for k, (start, end) in enumerate(GRADE_RANGES):
            grade_input = x[..., start:end]
            weight = self.grade_weights[k]
            bias = self.grade_biases[k]

            grade_output = torch.einsum(
                "...ci, ocij -> ...oj",
                grade_input, weight,
            ) + bias

            result[..., start:end] = grade_output

        return result
