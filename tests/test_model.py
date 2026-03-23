import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from config import TINY_CONFIG, TodorovConfig
from src.model.todorov import Todorov


class TestTodorovModel:

    def setup_method(self) -> None:
        self.config = TINY_CONFIG
        self.batch_size = 2
        self.seq_len = 16

    def test_forward_shape(self) -> None:
        model = Todorov(self.config)
        input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len))
        logits, states, aux = model(input_ids)
        assert logits.shape == (self.batch_size, self.seq_len, self.config.vocab_size)

    def test_layer_pattern(self) -> None:
        model = Todorov(self.config)
        layer_types = [block.layer_type for block in model.blocks]
        expected = list(self.config.layer_pattern) * (self.config.n_layers // len(self.config.layer_pattern))
        assert layer_types == expected

    def test_layer_count(self) -> None:
        model = Todorov(self.config)
        assert len(model.blocks) == self.config.n_layers

    def test_parameter_count_reasonable(self) -> None:
        model = Todorov(self.config)
        param_count = model.count_parameters()
        assert param_count > 0
        assert param_count < 100_000_000

    def test_gradient_flow_full_model(self) -> None:
        model = Todorov(self.config)
        input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len))
        logits, _, _ = model(input_ids)
        loss = logits.sum()
        loss.backward()
        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad

    def test_autoregressive_generation(self) -> None:
        model = Todorov(self.config)
        model.eval()
        input_ids = torch.randint(0, self.config.vocab_size, (1, 1))
        states = None
        generated = [input_ids[0, 0].item()]

        for step in range(4):
            logits, states, _ = model(input_ids, states=states, offset=step)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(next_token.item())
            input_ids = next_token

        assert len(generated) == 5

    def test_layer_type_counts(self) -> None:
        model = Todorov(self.config)
        counts = model.layer_type_counts()
        assert "KDA" in counts
        assert "Mamba3" in counts
        assert "MLA" in counts
        total = sum(counts.values())
        assert total == self.config.n_layers
