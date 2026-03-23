import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from src.layers.mla import MLALayer


class TestMLALayer:

    def setup_method(self) -> None:
        self.d_model = 64
        self.d_c = 32
        self.d_R = 8
        self.num_heads = 2
        self.batch_size = 2
        self.seq_len = 16

    def test_output_shape(self) -> None:
        layer = MLALayer(self.d_model, self.d_c, self.d_R, self.num_heads)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output, cache, aux = layer(x)
        assert output.shape == (self.batch_size, self.seq_len, self.d_model)

    def test_cache_compression(self) -> None:
        layer = MLALayer(self.d_model, self.d_c, self.d_R, self.num_heads)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        _, cache, aux = layer(x)
        cache_per_token = self.d_c + self.d_R
        assert cache.shape == (self.batch_size, self.seq_len, cache_per_token)
        assert aux["compression_ratio"] > 1.0

    def test_cache_accumulation(self) -> None:
        layer = MLALayer(self.d_model, self.d_c, self.d_R, self.num_heads)
        layer.eval()
        x1 = torch.randn(self.batch_size, 8, self.d_model)
        x2 = torch.randn(self.batch_size, 8, self.d_model)

        _, cache1, _ = layer(x1)
        _, cache2, _ = layer(x2, kv_cache=cache1, offset=8)

        cache_per_token = self.d_c + self.d_R
        assert cache2.shape == (self.batch_size, 16, cache_per_token)

    def test_gradient_flow(self) -> None:
        layer = MLALayer(self.d_model, self.d_c, self.d_R, self.num_heads)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True)
        output, _, _ = layer(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_cache_size_vs_standard(self) -> None:
        layer = MLALayer(self.d_model, self.d_c, self.d_R, self.num_heads)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        _, _, aux = layer(x)
        standard_kv_per_token = 2 * self.num_heads * (self.d_model // self.num_heads)
        compressed_per_token = aux["cache_size_per_token"]
        assert compressed_per_token < standard_kv_per_token

    def test_single_token_inference(self) -> None:
        layer = MLALayer(self.d_model, self.d_c, self.d_R, self.num_heads)
        layer.eval()
        x = torch.randn(self.batch_size, 1, self.d_model)
        output, cache, _ = layer(x)
        assert output.shape == (self.batch_size, 1, self.d_model)
