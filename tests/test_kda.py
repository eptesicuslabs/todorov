import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from src.layers.kda import KDALayer, RotaryPositionEncoding, apply_rotary


class TestKDALayer:

    def setup_method(self) -> None:
        self.d_model = 64
        self.num_heads = 2
        self.head_dim = 32
        self.batch_size = 2
        self.seq_len = 16

    def test_output_shape(self) -> None:
        layer = KDALayer(self.d_model, self.num_heads, self.head_dim)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output, state, aux = layer(x)
        assert output.shape == (self.batch_size, self.seq_len, self.d_model)

    def test_state_shape(self) -> None:
        layer = KDALayer(self.d_model, self.num_heads, self.head_dim)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        _, state, _ = layer(x)
        assert state.shape == (self.batch_size, self.num_heads, self.head_dim, self.head_dim)

    def test_recurrence_matches_sequential(self) -> None:
        layer = KDALayer(self.d_model, self.num_heads, self.head_dim, use_spikes=False)
        layer.eval()
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)

        full_output, full_state, _ = layer(x)

        state = None
        step_outputs = []
        for t in range(self.seq_len):
            step_out, state, _ = layer(x[:, t:t+1], state=state, offset=t)
            step_outputs.append(step_out)
        sequential_output = torch.cat(step_outputs, dim=1)

        max_diff = (full_output - sequential_output).abs().max().item()
        assert max_diff < 0.2, f"Max difference {max_diff} exceeds tolerance"

    def test_channel_wise_gate_shape(self) -> None:
        layer = KDALayer(self.d_model, self.num_heads, self.head_dim, channel_wise_gate=True)
        assert layer.alpha_log.shape == (self.num_heads, self.head_dim)

    def test_scalar_gate_shape(self) -> None:
        layer = KDALayer(self.d_model, self.num_heads, self.head_dim, channel_wise_gate=False)
        assert layer.alpha_log.shape == (self.num_heads, 1)

    def test_gradient_flow(self) -> None:
        layer = KDALayer(self.d_model, self.num_heads, self.head_dim)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True)
        output, _, _ = layer(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_causal_no_future_leak(self) -> None:
        layer = KDALayer(self.d_model, self.num_heads, self.head_dim, use_spikes=False)
        layer.eval()
        x = torch.randn(1, 8, self.d_model)

        full_out, _, _ = layer(x)
        prefix_out, _, _ = layer(x[:, :4])

        assert torch.allclose(full_out[:, :4], prefix_out, atol=1e-5)

    def test_with_spikes(self) -> None:
        from src.spikes.ternary_spike import AdaptiveTernarySpike
        spike = AdaptiveTernarySpike()
        layer = KDALayer(self.d_model, self.num_heads, self.head_dim, use_spikes=True, spike_module=spike)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output, state, aux = layer(x)
        assert output.shape == (self.batch_size, self.seq_len, self.d_model)
        assert "k_spikes" in aux


class TestRotaryPositionEncoding:

    def test_output_shapes(self) -> None:
        rope = RotaryPositionEncoding(32)
        x = torch.randn(2, 16, 32)
        cos, sin = rope(x)
        assert cos.shape == (16, 32)
        assert sin.shape == (16, 32)

    def test_apply_rotary_preserves_norm(self) -> None:
        x = torch.randn(2, 4, 8, 32)
        cos = torch.ones(8, 32)
        sin = torch.zeros(8, 32)
        rotated = apply_rotary(x, cos, sin)
        assert torch.allclose(x, rotated, atol=1e-6)
