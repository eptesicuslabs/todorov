from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from config import TodorovConfig
from src.layers.kda import KDALayer
from src.layers.mamba3 import Mamba3Layer
from src.layers.mla import MLALayer
from src.layers.swiglu import SwiGLU
from src.model.decode_head import DecodeHead, RMSNorm
from src.model.embedding import TodorovEmbedding


class TodorovBlock(nn.Module):

    def __init__(
        self,
        layer_type: str,
        config: TodorovConfig,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.layer_idx = layer_idx

        self.attn_norm = RMSNorm(config.d_model)
        self.mlp_norm = RMSNorm(config.d_model)
        self._state_key = "kv_cache" if layer_type == "MLA" else "state"

        spike_kwargs = {
            "alpha_init": config.spike.alpha_init,
            "learnable_alpha": config.spike.learnable_alpha,
            "min_threshold": config.spike.min_threshold,
            "max_threshold": config.spike.max_threshold,
            "tau": config.spike.atmn_tau,
            "threshold_init": config.spike.atmn_threshold_init,
        }

        spike_all = config.spike.spike_all_projections

        if layer_type == "KDA":
            self.attn = KDALayer(
                d_model=config.d_model,
                num_heads=config.kda.num_heads,
                head_dim=config.kda.head_dim,
                channel_wise_gate=config.kda.channel_wise_gate,
                use_spikes=config.kda.use_spikes,
                spike_all=spike_all,
                spike_type=config.spike.spike_type,
                spike_kwargs=spike_kwargs,
            )
        elif layer_type == "Mamba3":
            self.attn = Mamba3Layer(
                d_model=config.d_model,
                d_state=config.mamba3.d_state,
                expand=config.mamba3.expand,
                use_rope=config.mamba3.use_rope,
                use_trapezoidal=config.mamba3.use_trapezoidal,
                use_spikes=config.kda.use_spikes,
                spike_all=spike_all,
                spike_type=config.spike.spike_type,
                spike_kwargs=spike_kwargs,
            )
        elif layer_type == "MLA":
            self.attn = MLALayer(
                d_model=config.d_model,
                d_c=config.mla.d_c,
                d_R=config.mla.d_R,
                num_heads=config.mla.num_heads,
                use_spikes=config.kda.use_spikes,
                spike_all=spike_all,
                spike_type=config.spike.spike_type,
                spike_kwargs=spike_kwargs,
            )

        mlp_use_spikes = config.kda.use_spikes and spike_all
        self.mlp = SwiGLU(
            d_model=config.d_model,
            ratio=config.mlp.ratio,
            spatial_mode=config.mlp.spatial_mode,
            use_spikes=mlp_use_spikes,
            spike_type=config.spike.spike_type,
            spike_kwargs=spike_kwargs,
        )

    def forward(
        self,
        x: Tensor,
        state: Any = None,
        offset: int = 0,
    ) -> tuple[Tensor, Any, dict]:
        residual = x
        normed = self.attn_norm(x)
        kwargs = {self._state_key: state, "offset": offset}
        attn_out, new_state, aux = self.attn(normed, **kwargs)
        x = residual + attn_out

        residual = x
        mlp_out, mlp_aux = self.mlp(self.mlp_norm(x))
        x = residual + mlp_out

        aux.update(mlp_aux)

        return x, new_state, aux


class Todorov(nn.Module):

    def __init__(self, config: TodorovConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = TodorovEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
        )

        layer_types = config.layer_types
        self.blocks = nn.ModuleList([
            TodorovBlock(layer_types[i], config, i)
            for i in range(config.n_layers)
        ])

        self.head = DecodeHead(config.d_model, config.vocab_size)
        self.head.tie_weights(self.embedding.token_embedding.weight)

    def forward(
        self,
        input_ids: Tensor,
        states: list[Any] | None = None,
        offset: int = 0,
        return_aux: bool = False,
    ) -> tuple[Tensor, list[Any], dict]:
        x = self.embedding(input_ids)

        if states is None:
            states = [None] * self.config.n_layers

        new_states = []
        all_aux: dict[str, Any] = {"layers": {}}

        for i, block in enumerate(self.blocks):
            x, new_state, aux = block(x, state=states[i], offset=offset)
            new_states.append(new_state)
            all_aux["layers"][i] = aux

        logits = self.head(x)

        spike_data = {}
        pre_spike_data = {}
        spike_keys = [
            ("k_spikes", "pre_spike_k"),
            ("v_spikes", "pre_spike_v"),
            ("q_spikes", "pre_spike_q"),
            ("o_spikes", "pre_spike_o"),
            ("in_spikes", "pre_spike_in"),
            ("out_spikes", "pre_spike_out"),
            ("kv_spikes", "pre_spike_kv"),
            ("gate_spikes", "pre_spike_gate"),
            ("up_spikes", "pre_spike_up"),
        ]
        for i, aux in all_aux["layers"].items():
            for spike_key, pre_key in spike_keys:
                if spike_key in aux:
                    if i not in spike_data:
                        spike_data[i] = []
                        pre_spike_data[i] = []
                    spike_data[i].append(aux[spike_key])
                    pre_spike_data[i].append(aux[pre_key])

        all_aux["spike_data"] = spike_data
        all_aux["pre_spike_data"] = pre_spike_data

        return logits, new_states, all_aux

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def layer_type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for block in self.blocks:
            lt = block.layer_type
            counts[lt] = counts.get(lt, 0) + 1
        return counts
