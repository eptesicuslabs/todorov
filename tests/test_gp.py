import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from src.algebra.multivector import Multivector, NUM_COMPONENTS, GRADE_RANGES, grade_rms_norm
from src.algebra.geometric_product import (
    geometric_product, outer_product, inner_product, sandwich_product,
    _build_cayley_table,
)


class TestCayleyTable:

    def test_e1_squared_equals_1(self) -> None:
        e1 = Multivector(torch.zeros(NUM_COMPONENTS))
        e1.values[2] = 1.0
        result = geometric_product(e1, e1)
        assert abs(result.values[0].item() - 1.0) < 1e-6

    def test_e0_squared_equals_0(self) -> None:
        e0 = Multivector(torch.zeros(NUM_COMPONENTS))
        e0.values[1] = 1.0
        result = geometric_product(e0, e0)
        assert abs(result.values.sum().item()) < 1e-6

    def test_e1_e2_equals_e12(self) -> None:
        e1 = Multivector(torch.zeros(NUM_COMPONENTS))
        e1.values[2] = 1.0
        e2 = Multivector(torch.zeros(NUM_COMPONENTS))
        e2.values[3] = 1.0
        result = geometric_product(e1, e2)
        assert abs(result.values[8].item() - 1.0) < 1e-6

    def test_e2_e1_equals_neg_e12(self) -> None:
        e1 = Multivector(torch.zeros(NUM_COMPONENTS))
        e1.values[2] = 1.0
        e2 = Multivector(torch.zeros(NUM_COMPONENTS))
        e2.values[3] = 1.0
        result = geometric_product(e2, e1)
        assert abs(result.values[8].item() + 1.0) < 1e-6


class TestGeometricProduct:

    def test_associativity(self) -> None:
        a = Multivector.random(4)
        b = Multivector.random(4)
        c = Multivector.random(4)

        ab_c = geometric_product(geometric_product(a, b), c)
        a_bc = geometric_product(a, geometric_product(b, c))

        assert torch.allclose(ab_c.values, a_bc.values, atol=1e-4)

    def test_scalar_multiplication(self) -> None:
        a = Multivector.random(4)
        scalar = Multivector(torch.zeros(4, NUM_COMPONENTS))
        scalar.values[..., 0] = 3.0

        result = geometric_product(scalar, a)
        expected = a * 3.0
        assert torch.allclose(result.values, expected.values, atol=1e-5)

    def test_batch_processing(self) -> None:
        a = Multivector.random(2, 8)
        b = Multivector.random(2, 8)
        result = geometric_product(a, b)
        assert result.shape == (2, 8, NUM_COMPONENTS)

    def test_grade_structure(self) -> None:
        a = Multivector.random(4)
        b = Multivector.random(4)
        result = geometric_product(a, b)
        assert result.values.shape[-1] == NUM_COMPONENTS


class TestOuterProduct:

    def test_grade_raising(self) -> None:
        e1 = Multivector(torch.zeros(NUM_COMPONENTS))
        e1.values[2] = 1.0
        e2 = Multivector(torch.zeros(NUM_COMPONENTS))
        e2.values[3] = 1.0
        result = outer_product(e1, e2)
        assert abs(result.values[8].item() - 1.0) < 1e-6


class TestSandwichProduct:

    def test_rotation_preserves_norm(self) -> None:
        angle = torch.tensor(1.0471)
        rotor = Multivector(torch.zeros(NUM_COMPONENTS))
        rotor.values[0] = torch.cos(angle / 2)
        rotor.values[8] = torch.sin(angle / 2)

        x = Multivector(torch.zeros(NUM_COMPONENTS))
        x.values[2] = 1.0

        rotated = sandwich_product(rotor, x)
        original_norm = x.norm()
        rotated_norm = rotated.norm()
        assert abs(original_norm.item() - rotated_norm.item()) < 1e-4


class TestGradeRMSNorm:

    def test_normalized_grades(self) -> None:
        mv = Multivector(torch.randn(4, 8, NUM_COMPONENTS) * 10)
        normed = grade_rms_norm(mv)
        for k in range(len(GRADE_RANGES)):
            start, end = GRADE_RANGES[k]
            grade_vals = normed.values[..., start:end]
            rms = torch.sqrt(torch.mean(grade_vals ** 2, dim=-1))
            assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_preserves_shape(self) -> None:
        mv = Multivector(torch.randn(2, 4, NUM_COMPONENTS))
        normed = grade_rms_norm(mv)
        assert normed.shape == mv.shape
