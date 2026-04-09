# results

centralized index of all experiment results. raw data (json metrics + figures) lives with the simulation scripts. this directory collects summaries and cross-experiment analysis.

## experiment registry

| id | date | experiment | script | result | status |
|---|---|---|---|---|---|
| 001 | 2026-04-07 | pattern completion baseline | memory/pattern_completion.py | hebbian 0.982 vs shuffled 0.083 (p=0.001) | validated |
| 002 | 2026-04-07 | k-wta vs threshold | lateral_inhibition/wta_dynamics.py | k-wta 0.925 vs threshold 0.500 (p=0.001) | validated |
| 003 | 2026-04-08 | leak vs carry | single_neuron/lif_fi_curve.py | tradeoff: leak retains more, carry more stable | validated |
| 004 | 2026-04-09 | bcm adaptive alpha | plasticity/bcm_alpha_pilot.py | gamma=0.5 reduces norm std by 0.70 (p=0.001) | validated |
| 005 | 2026-04-09 | gp vs bilinear | spatial/gp_vs_bilinear_pilot.py | pga no advantage at random init (p=0.001 against) | validated |
| 006 | 2026-04-09 | capacity scaling | memory/capacity_scaling.py | binary saturates at alpha=0.08-0.10 with 5% corruption | running |
| 007 | 2026-04-09 | hierarchical ternary | sparse_coding/hierarchical_ternary.py | 10% k-wta = 0.56 bits/dim, CKA 0.71 | validated |
| 008 | 2026-04-09 | imagination recombination | memory/imagination_recombination.py | recombined overlap 0.93, random -0.02, novelty 7.5% | validated |
| 009 | 2026-04-09 | sparse topology | cortical_microcircuit/sparse_topology.py | random sparse beats small-world for retrieval | validated |

## metrics locations

all json metrics follow the neuroloc.sim.metrics/v1 schema (defined in simulations/shared.py).

- memory/pattern_completion_metrics.json
- lateral_inhibition/wta_dynamics_metrics.json
- single_neuron/lif_leak_validation_metrics.json
- plasticity/bcm_alpha_pilot_metrics.json
- spatial/gp_vs_bilinear_pilot_metrics.json
- memory/capacity_scaling_metrics.json (pending completion)
- sparse_coding/hierarchical_ternary_metrics.json
- memory/imagination_recombination_metrics.json
- cortical_microcircuit/sparse_topology_metrics.json

## standalone publishable components

these results may stand as independent contributions:

1. **hierarchical ternary compression** -- k-wta selection + ternary quantization as two-stage pipeline. 0.32-0.56 bits/dim with 63-71% structural preservation. needs novelty verification.
2. **bcm-like adaptive forgetting** -- activity-dependent alpha for recurrent state stabilization. gamma=0.3-0.5 validated.
3. **k-wta spike selection** -- competitive selection outperforms threshold spiking by 0.425 on exact support recovery.
4. **imagination via hopfield recombination** -- novel queries to outer-product memory produce structured interpolation, not noise.
