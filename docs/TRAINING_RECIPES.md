# Training Recipes

No validated recipes yet. Initial configuration:

## Default Config

    learning_rate: 3e-4
    weight_decay: 0.01
    batch_size: 32
    max_steps: 100000
    warmup_steps: 1000
    gradient_clip: 1.0
    residual_penalty_weight: 0.01
    optimizer: AdamW (betas=0.9, 0.95)
    scheduler: cosine with warmup
    min_lr_ratio: 0.1

## Phase 1 Hyperparameter Search Space

    LR: [1e-4, 3e-4, 1e-3]
    Batch: [16, 32, 64]
    Residual penalty: [0.001, 0.01, 0.1]
    Spike alpha: [0.5, 1.0, 2.0]
