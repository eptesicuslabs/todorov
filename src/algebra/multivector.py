from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


GRADE_RANGES: list[tuple[int, int]] = [
    (0, 1),
    (1, 5),
    (5, 11),
    (11, 15),
    (15, 16),
]

BASIS_LABELS: list[str] = [
    "s",
    "e0", "e1", "e2", "e3",
    "e01", "e02", "e03", "e12", "e13", "e23",
    "e012", "e013", "e023", "e123",
    "e0123",
]

NUM_COMPONENTS: int = 16


class Multivector:

    def __init__(self, values: Tensor) -> None:
        assert values.shape[-1] == NUM_COMPONENTS
        self.values = values

    @property
    def shape(self) -> torch.Size:
        return self.values.shape

    @property
    def device(self) -> torch.device:
        return self.values.device

    @property
    def dtype(self) -> torch.dtype:
        return self.values.dtype

    def grade(self, k: int) -> Tensor:
        start, end = GRADE_RANGES[k]
        return self.values[..., start:end]

    def grade_project(self, k: int) -> Multivector:
        result = torch.zeros_like(self.values)
        start, end = GRADE_RANGES[k]
        result[..., start:end] = self.values[..., start:end]
        return Multivector(result)

    def grade_select(self, grades: list[int]) -> Multivector:
        result = torch.zeros_like(self.values)
        for k in grades:
            start, end = GRADE_RANGES[k]
            result[..., start:end] = self.values[..., start:end]
        return Multivector(result)

    def reverse(self) -> Multivector:
        signs = torch.ones(NUM_COMPONENTS, device=self.device, dtype=self.dtype)
        for k in range(5):
            if k * (k - 1) // 2 % 2 == 1:
                start, end = GRADE_RANGES[k]
                signs[start:end] = -1.0
        return Multivector(self.values * signs)

    def grade_involution(self) -> Multivector:
        signs = torch.ones(NUM_COMPONENTS, device=self.device, dtype=self.dtype)
        for k in range(5):
            if k % 2 == 1:
                start, end = GRADE_RANGES[k]
                signs[start:end] = -1.0
        return Multivector(self.values * signs)

    def dual(self) -> Multivector:
        result = torch.zeros_like(self.values)
        result[..., 0] = self.values[..., 15]
        result[..., 1] = self.values[..., 14]
        result[..., 2] = -self.values[..., 13]
        result[..., 3] = self.values[..., 12]
        result[..., 4] = -self.values[..., 11]
        result[..., 5] = self.values[..., 10]
        result[..., 6] = -self.values[..., 9]
        result[..., 7] = self.values[..., 8]
        result[..., 8] = self.values[..., 7]
        result[..., 9] = -self.values[..., 6]
        result[..., 10] = self.values[..., 5]
        result[..., 11] = -self.values[..., 4]
        result[..., 12] = self.values[..., 3]
        result[..., 13] = -self.values[..., 2]
        result[..., 14] = self.values[..., 1]
        result[..., 15] = self.values[..., 0]
        return Multivector(result)

    def norm_squared(self) -> Tensor:
        from src.algebra.geometric_product import geometric_product
        product = geometric_product(self, self.reverse())
        return product.values[..., 0]

    def norm(self) -> Tensor:
        return self.norm_squared().abs().sqrt()

    def __add__(self, other: Multivector) -> Multivector:
        return Multivector(self.values + other.values)

    def __sub__(self, other: Multivector) -> Multivector:
        return Multivector(self.values - other.values)

    def __mul__(self, scalar: float | Tensor) -> Multivector:
        if isinstance(scalar, (int, float)):
            return Multivector(self.values * scalar)
        return Multivector(self.values * scalar.unsqueeze(-1))

    def __rmul__(self, scalar: float | Tensor) -> Multivector:
        return self.__mul__(scalar)

    def __neg__(self) -> Multivector:
        return Multivector(-self.values)

    def __repr__(self) -> str:
        return f"Multivector(shape={self.shape})"

    @staticmethod
    def zeros(*shape: int, device: torch.device | None = None) -> Multivector:
        return Multivector(torch.zeros(*shape, NUM_COMPONENTS, device=device))

    @staticmethod
    def random(*shape: int, device: torch.device | None = None) -> Multivector:
        return Multivector(torch.randn(*shape, NUM_COMPONENTS, device=device))


def grade_rms_norm(mv: Multivector, eps: float = 1e-8) -> Multivector:
    result = torch.zeros_like(mv.values)
    for k in range(len(GRADE_RANGES)):
        start, end = GRADE_RANGES[k]
        components = mv.values[..., start:end]
        rms = torch.sqrt(torch.mean(components ** 2, dim=-1, keepdim=True) + eps)
        result[..., start:end] = components / rms
    return Multivector(result)
