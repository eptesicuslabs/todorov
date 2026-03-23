import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from src.spikes.ternary_spike import TernaryQuantizer, AdaptiveTernarySpike, compute_spike_statistics


class TestTernaryQuantizer:

    def test_output_values(self) -> None:
        x = torch.randn(100)
        threshold = torch.tensor(0.5)
        result = TernaryQuantizer.apply(x, threshold, True)
        unique = torch.unique(result)
        for val in unique:
            assert val.item() in (-1.0, 0.0, 1.0)

    def test_ste_gradient_flow(self) -> None:
        x = torch.randn(32, 64, requires_grad=True)
        threshold = torch.tensor(0.5)
        result = TernaryQuantizer.apply(x, threshold, True)
        loss = result.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones_like(x))

    def test_threshold_effect(self) -> None:
        x = torch.randn(1000)
        low_t = TernaryQuantizer.apply(x, torch.tensor(0.1), True)
        high_t = TernaryQuantizer.apply(x, torch.tensor(2.0), True)
        low_density = (low_t != 0).float().mean()
        high_density = (high_t != 0).float().mean()
        assert low_density > high_density


class TestAdaptiveTernarySpike:

    def test_output_ternary(self) -> None:
        spike = AdaptiveTernarySpike()
        x = torch.randn(4, 32)
        result = spike(x)
        unique = torch.unique(result)
        for val in unique:
            assert val.item() in (-1.0, 0.0, 1.0)

    def test_adaptive_threshold_tracks_stats(self) -> None:
        spike = AdaptiveTernarySpike()
        for _ in range(10):
            x = torch.randn(4, 32)
            spike(x)
        assert spike.num_updates.item() == 10
        assert spike.running_spike_density.item() > 0

    def test_gradient_through_spike(self) -> None:
        spike = AdaptiveTernarySpike()
        x = torch.randn(4, 32, requires_grad=True)
        result = spike(x)
        loss = result.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_learnable_alpha(self) -> None:
        spike = AdaptiveTernarySpike(learnable_alpha=True)
        assert isinstance(spike.alpha, torch.nn.Parameter)

    def test_fixed_alpha(self) -> None:
        spike = AdaptiveTernarySpike(learnable_alpha=False)
        assert not isinstance(spike.alpha, torch.nn.Parameter)


class TestSpikeStatistics:

    def test_density_computation(self) -> None:
        spikes = torch.tensor([1.0, 0.0, -1.0, 0.0, 1.0])
        stats = compute_spike_statistics(spikes)
        assert abs(stats["density"] - 0.6) < 1e-6
        assert abs(stats["positive_fraction"] - 0.4) < 1e-6
        assert abs(stats["negative_fraction"] - 0.2) < 1e-6
        assert abs(stats["zero_fraction"] - 0.4) < 1e-6

    def test_flicker_rate(self) -> None:
        spikes = torch.tensor([1.0, 0.0, -1.0])
        prev = torch.tensor([1.0, 1.0, -1.0])
        stats = compute_spike_statistics(spikes, prev)
        assert "flicker_rate" in stats
        assert abs(stats["flicker_rate"] - 1.0 / 3.0) < 1e-6
