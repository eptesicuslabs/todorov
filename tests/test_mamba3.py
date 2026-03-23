import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from src.layers.mamba3 import Mamba3Layer


class TestMamba3Layer:

    def setup_method(self) -> None:
        self.d_model = 64
        self.d_state = 8
        self.expand = 2
        self.batch_size = 2
        self.seq_len = 16

    def test_output_shape(self) -> None:
        layer = Mamba3Layer(self.d_model, self.d_state, self.expand)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output, state, aux = layer(x)
        assert output.shape == (self.batch_size, self.seq_len, self.d_model)

    def test_state_shape(self) -> None:
        layer = Mamba3Layer(self.d_model, self.d_state, self.expand)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        _, state, _ = layer(x)
        d_inner = self.d_model * self.expand
        assert state.shape == (self.batch_size, d_inner, self.d_state)

    def test_trapezoidal_discretization(self) -> None:
        layer = Mamba3Layer(self.d_model, self.d_state, self.expand, use_trapezoidal=True)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output, _, _ = layer(x)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_exponential_discretization(self) -> None:
        layer = Mamba3Layer(self.d_model, self.d_state, self.expand, use_trapezoidal=False)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output, _, _ = layer(x)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self) -> None:
        layer = Mamba3Layer(self.d_model, self.d_state, self.expand)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True)
        output, _, _ = layer(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_state_utilization_in_aux(self) -> None:
        layer = Mamba3Layer(self.d_model, self.d_state, self.expand)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        _, _, aux = layer(x)
        assert "state_utilization" in aux
        assert 0.0 <= aux["state_utilization"] <= 1.0

    def test_state_continuity(self) -> None:
        layer = Mamba3Layer(self.d_model, self.d_state, self.expand)
        layer.eval()
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        _, state1, _ = layer(x[:, :8])
        _, state2, _ = layer(x[:, 8:], state=state1, offset=8)
        assert state2 is not None
        assert not torch.isnan(state2).any()
