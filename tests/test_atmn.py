import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from src.spikes.atmn_spike import ATMNSpike, ATMNQuantizer, create_spike_module


class TestATMNQuantizer:

    def test_output_values(self) -> None:
        h = torch.randn(100)
        threshold = torch.tensor(0.5)
        result = ATMNQuantizer.apply(h, threshold)
        unique = torch.unique(result)
        for val in unique:
            assert val.item() in (-1.0, 0.0, 1.0)

    def test_ste_gradient(self) -> None:
        h = torch.randn(32, 64, requires_grad=True)
        threshold = torch.tensor(0.5)
        result = ATMNQuantizer.apply(h, threshold)
        loss = result.sum()
        loss.backward()
        assert h.grad is not None
        assert torch.allclose(h.grad, torch.ones_like(h))


class TestATMNSpike:

    def test_output_ternary(self) -> None:
        spike = ATMNSpike(d_features=32)
        x = torch.randn(4, 32)
        result = spike(x)
        unique = torch.unique(result)
        for val in unique:
            assert val.item() in (-1.0, 0.0, 1.0)

    def test_temporal_accumulation(self) -> None:
        spike = ATMNSpike(d_features=16, tau=2.0, threshold_init=1.0)
        spike.eval()
        spike.reset_state(1)

        small_input = torch.ones(1, 16) * 0.5
        r1 = spike(small_input)
        r2 = spike(small_input)
        r3 = spike(small_input)

        density_1 = (r1 != 0).float().mean().item()
        density_3 = (r3 != 0).float().mean().item()
        assert density_3 >= density_1

    def test_reset_clears_state(self) -> None:
        spike = ATMNSpike(d_features=16)
        x = torch.randn(2, 16)
        spike(x)
        assert spike._state_initialized

        spike.reset_state(2)
        assert spike.membrane_potential.abs().sum().item() == 0.0

    def test_threshold_always_positive(self) -> None:
        spike = ATMNSpike(d_features=32, threshold_init=-5.0)
        assert (spike.threshold > 0).all()

    def test_per_neuron_threshold(self) -> None:
        spike = ATMNSpike(d_features=32)
        assert spike.threshold_log.shape == (32,)

    def test_gradient_flow(self) -> None:
        spike = ATMNSpike(d_features=32)
        x = torch.randn(4, 32, requires_grad=True)
        result = spike(x)
        loss = result.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_multidim_input(self) -> None:
        spike = ATMNSpike(d_features=32)
        x = torch.randn(2, 8, 32)
        result = spike(x)
        assert result.shape == (2, 8, 32)

    def test_running_density(self) -> None:
        spike = ATMNSpike(d_features=32)
        spike.train()
        for _ in range(10):
            x = torch.randn(4, 32)
            spike(x)
        assert spike.num_updates.item() == 10
        assert spike.running_spike_density.item() > 0


class TestCreateSpikeModule:

    def test_create_ternary(self) -> None:
        module = create_spike_module("ternary")
        from src.spikes.ternary_spike import AdaptiveTernarySpike
        assert isinstance(module, AdaptiveTernarySpike)

    def test_create_atmn(self) -> None:
        module = create_spike_module("atmn", d_features=64)
        assert isinstance(module, ATMNSpike)

    def test_invalid_type(self) -> None:
        with pytest.raises(ValueError):
            create_spike_module("invalid")
