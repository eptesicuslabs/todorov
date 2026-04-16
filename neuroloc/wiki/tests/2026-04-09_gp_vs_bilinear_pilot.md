# 2026-04-09 gp vs bilinear pilot

status: historical context only. frozen as of 2026-04-09. do not edit.

test type: bridge-validation pilot (the critical missing experiment)

script: `neuroloc/simulations/spatial/gp_vs_bilinear_pilot.py`

artifacts:
- `neuroloc/simulations/spatial/gp_vs_bilinear_pilot.png`
- `neuroloc/simulations/spatial/gp_vs_bilinear_pilot_metrics.json`

bridge: [[spatial_computation_to_pga]]

## what was done

tested whether G(3,0,1) projective geometric algebra structure provides any advantage over alternative bilinear maps at matched parameter count. compared four methods:
- **pga**: G(3,0,1) geometric product via cayley table (192 non-zero entries)
- **quaternion**: quaternion product applied blockwise (4 components at a time)
- **random_bilinear**: random dense 16x16->16 bilinear tensor
- **elementwise**: simple element-wise product (baseline)

all methods: W_left projects x to 16-dim, bilinear interaction, W_out projects back. same random projections across methods within each trial. measured MI (input to output) and CKA (representational similarity) across noise levels [0.0, 0.2, 0.5, 1.0].

## configuration

- d_model: 32, d_mv: 16, n_samples: 256
- 16 trials per condition, 256 total trials
- noise levels: [0.0, 0.2, 0.5, 1.0]
- numpy seed: 42

## key results

at noise=0:

| method | MI (bits) | CKA |
|--------|----------|-----|
| pga | 3.605 | 1.000 |
| quaternion | 3.617 | 1.000 |
| random_bilinear | 3.658 | 1.000 |
| elementwise | 3.646 | 1.000 |

pga vs others at noise=0:
- pga vs quaternion: MI delta = -0.012, p = 0.321 (not significant)
- pga vs random_bilinear: MI delta = -0.053, p = 0.001 (random bilinear WINS)
- pga vs elementwise: MI delta = -0.041, p = 0.001 (elementwise WINS)

## verdict

at random initialization with untrained projections, the geometric structure of G(3,0,1) provides NO advantage over random bilinear maps or elementwise products. random bilinear and elementwise actually show significantly higher MI than PGA (p=0.001).

this does NOT mean PGA is useless. it means:
1. the geometric structure does not help at initialization -- the advantage (MI=1.311 in trained todorov) must come from the interaction between trained weights and the algebra's structure
2. the cayley table's sparse structure (192 of 256 entries) may constrain the function space in a way that is beneficial under gradient optimization but not at random init
3. the critical experiment remains partially open: this pilot tests random projections, not trained weights. a trained-weight comparison is needed to close the question.

the bridge document's recommendation ("do nothing, run the control experiment first") is validated in the sense that the control has been run. the result is: geometric structure does not help at random init, so if it helps in the trained model, the explanation is weight-algebra interaction, not raw algebraic structure.

## limitations

- random projections, not trained weights -- the most important caveat
- small scale (d_model=32, 256 samples) due to numpy stability issues on this platform
- needs replication at d_model=256+ on stable compute (kaggle T4)
- does not test the additive residual connection (x + GP(W_left(x), W_right(x)) @ W_out) that todorov uses
- does not measure the spatial reasoning tasks (n-body, shape classification) where GP showed 29% improvement

## see also

- `wiki/tests/index.md` — tests/ catalog
- `wiki/PROJECT_PLAN.md` — canonical project state
- `wiki/INDEX.md` — full wiki navigation map
- `wiki/OPERATING_DIRECTIVE.md` — rules governing this article
