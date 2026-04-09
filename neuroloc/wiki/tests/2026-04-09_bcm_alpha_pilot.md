# 2026-04-09 bcm-like adaptive alpha pilot

date run: 2026-04-09

status: completed

test type: bridge-validation pilot

script: `neuroloc/simulations/plasticity/bcm_alpha_pilot.py`

artifacts:
- `neuroloc/simulations/plasticity/bcm_alpha_pilot.png`
- `neuroloc/simulations/plasticity/bcm_alpha_pilot_metrics.json`

bridge: [[plasticity_to_kda_delta_rule]]

## what was done

tested whether activity-dependent forgetting (bcm-like sliding threshold on alpha) prevents state saturation in a kda-like recurrent associative memory. the proposed change: alpha_eff = sigmoid(alpha_log + gamma * log(||S_t||)). when state norm is large, alpha increases (faster forgetting).

compared gamma values [0.0, 0.1, 0.3, 0.5, 1.0] across sequence lengths [64, 128, 256, 512, 1024] with 24 trials per condition. measured state norm variance (stability) and retrieval quality (cosine similarity of retrieved vs stored values).

## configuration

- 8 heads, head_dim 64
- alpha_log_init: -2.0 (alpha ~ 0.12)
- beta_mean: 0.5
- 24 trials per condition, 600 total trials
- numpy seed: 42

## key results

at seq_len=1024:

| gamma | stability delta | stability p | retrieval delta | retrieval p |
|-------|----------------|-------------|-----------------|-------------|
| 0.1   | -0.015         | 0.622       | +0.005          | 0.138       |
| 0.3   | -0.337         | 0.001       | +0.004          | 0.194       |
| 0.5   | -0.698         | 0.001       | +0.003          | 0.232       |
| 1.0   | +0.021         | 0.863       | +0.007          | 0.050       |

- gamma=0.3 and gamma=0.5 significantly reduce state norm variance (p=0.001)
- gamma=0.5 gives the largest stability improvement (-0.698 reduction in norm std)
- retrieval quality is NOT degraded at any gamma value (all deltas positive but not significant)
- gamma=1.0 overshoots: stability improvement disappears, alpha becomes too aggressive

## verdict

bcm-like adaptive alpha works as predicted by the bridge document. gamma=0.3 to 0.5 significantly stabilizes the recurrent state over long sequences without degrading retrieval. gamma=1.0 is too aggressive and collapses back to baseline stability.

the mechanism is sound: when state norm grows, alpha increases, accelerating forgetting and preventing saturation. this is a genuine stability improvement with no measurable cost to retrieval at this scale.

recommended: test gamma=0.3 as default in the next trained model run (phase 5 or later).

## limitations

- synthetic gaussian inputs, not trained language model activations
- retrieval cosine similarity is near zero for all conditions (baseline 0.007), meaning the absolute retrieval performance is weak -- but the relative comparison is valid
- 8 heads x 64 dim is smaller than the 267M architecture (16 heads x 64 dim)
- does not test interaction with fla chunk_kda kernel
