import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import torch
import pytest
from src.algebra.multivector import Multivector, NUM_COMPONENTS, GRADE_RANGES
from src.algebra.geometric_product import geometric_product, sandwich_product
from src.algebra.equivariant_linear import EquivariantLinear


def make_rotor_e12(angle: float) -> Multivector:
    values = torch.zeros(NUM_COMPONENTS)
    values[0] = math.cos(angle / 2)
    values[8] = math.sin(angle / 2)
    return Multivector(values)


class TestEquivariance:

    def test_rotation_60_degrees(self) -> None:
        angle = math.pi / 3
        rotor = make_rotor_e12(angle)

        x = Multivector(torch.zeros(NUM_COMPONENTS))
        x.values[2] = 1.0

        rotated = sandwich_product(rotor, x)

        expected_e1 = math.cos(angle)
        expected_e2 = -math.sin(angle)

        assert abs(rotated.values[2].item() - expected_e1) < 1e-4
        assert abs(rotated.values[3].item() - expected_e2) < 1e-4

    def test_rotation_preserves_scalar(self) -> None:
        rotor = make_rotor_e12(math.pi / 4)
        x = Multivector(torch.zeros(NUM_COMPONENTS))
        x.values[0] = 5.0
        rotated = sandwich_product(rotor, x)
        assert abs(rotated.values[0].item() - 5.0) < 1e-4

    def test_rotation_preserves_grade_structure(self) -> None:
        rotor = make_rotor_e12(math.pi / 3)
        x = Multivector.random(4)

        rotated = sandwich_product(rotor, x)

        for k in range(5):
            start, end = GRADE_RANGES[k]
            original_grade_norm = torch.norm(x.values[..., start:end])
            rotated_grade_norm = torch.norm(rotated.values[..., start:end])
            if original_grade_norm > 1e-6:
                relative_error = abs(original_grade_norm - rotated_grade_norm) / original_grade_norm
                assert relative_error < 0.1, f"Grade {k} norm not preserved: {relative_error}"

    def test_equivariant_linear_grade_structure(self) -> None:
        layer = EquivariantLinear(4, 8)
        x = torch.randn(2, 4, NUM_COMPONENTS)
        output = layer(x)
        assert output.shape == (2, 8, NUM_COMPONENTS)

    def test_equivariant_linear_batch(self) -> None:
        layer = EquivariantLinear(2, 3)
        x = torch.randn(4, 8, 2, NUM_COMPONENTS)
        output = layer(x)
        assert output.shape == (4, 8, 3, NUM_COMPONENTS)

    def test_gp_self_interaction_equivariance(self) -> None:
        torch.manual_seed(42)
        angle = math.pi / 3
        rotor = make_rotor_e12(angle)

        x = Multivector.random(8)

        gp_then_rotate = sandwich_product(rotor, geometric_product(x, x))

        x_rotated = sandwich_product(rotor, x)
        rotate_then_gp = geometric_product(x_rotated, x_rotated)

        assert torch.allclose(gp_then_rotate.values, rotate_then_gp.values, atol=1e-3)
