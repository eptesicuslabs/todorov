from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KDAConfig:
    num_heads: int = 8
    head_dim: int = 128
    channel_wise_gate: bool = True
    use_spikes: bool = True
    spike_alpha_init: float = 1.0


@dataclass
class Mamba3Config:
    d_state: int = 32
    expand: int = 2
    use_rope: bool = True
    use_trapezoidal: bool = True


@dataclass
class MLAConfig:
    d_c: int = 128
    d_R: int = 32
    num_heads: int = 8


@dataclass
class MLPConfig:
    ratio: float = 2.25
    spatial_mode: bool = False


@dataclass
class SpikeConfig:
    spike_type: str = "ternary"
    spike_all_projections: bool = False
    alpha_init: float = 1.0
    learnable_alpha: bool = True
    min_threshold: float = 0.01
    max_threshold: float = 10.0
    atmn_tau: float = 2.0
    atmn_threshold_init: float = 0.0


@dataclass
class AlgebraConfig:
    num_components: int = 16
    grade_dims: tuple[int, ...] = (1, 4, 6, 4, 1)


@dataclass
class TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    max_steps: int = 100_000
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    residual_penalty_weight: float = 0.01
    seed: int = 42
    min_lr_ratio: float = 0.1


@dataclass
class TodorovConfig:
    d_model: int = 1024
    n_layers: int = 24
    vocab_size: int = 32_000
    max_seq_len: int = 131_072
    weight_precision: str = "INT8"
    layer_pattern: tuple[str, ...] = ("KDA", "KDA", "KDA", "Mamba3", "KDA", "KDA", "KDA", "MLA")
    kda: KDAConfig = field(default_factory=KDAConfig)
    mamba3: Mamba3Config = field(default_factory=Mamba3Config)
    mla: MLAConfig = field(default_factory=MLAConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)
    spike: SpikeConfig = field(default_factory=SpikeConfig)
    algebra: AlgebraConfig = field(default_factory=AlgebraConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @property
    def layer_types(self) -> list[str]:
        pattern = list(self.layer_pattern)
        repeats = self.n_layers // len(pattern)
        return pattern * repeats


TINY_CONFIG = TodorovConfig(
    d_model=64,
    n_layers=8,
    vocab_size=256,
    max_seq_len=512,
    kda=KDAConfig(num_heads=2, head_dim=32),
    mamba3=Mamba3Config(d_state=8, expand=2),
    mla=MLAConfig(d_c=32, d_R=8, num_heads=2),
    mlp=MLPConfig(ratio=2.0),
)

SMALL_CONFIG = TodorovConfig(
    d_model=256,
    n_layers=16,
    vocab_size=8_000,
    max_seq_len=4_096,
    kda=KDAConfig(num_heads=4, head_dim=64),
    mamba3=Mamba3Config(d_state=16, expand=2),
    mla=MLAConfig(d_c=64, d_R=16, num_heads=4),
)

BASE_CONFIG = TodorovConfig()
