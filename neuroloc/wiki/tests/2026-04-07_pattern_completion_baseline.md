# 2026-04-07 pattern completion baseline

status: historical context only. frozen as of 2026-04-07. do not edit.

test type: simulation retrofit and validation run

script:
- `neuroloc/simulations/memory/pattern_completion.py`

artifacts:
- `neuroloc/simulations/memory/pattern_completion.png`
- `neuroloc/simulations/memory/pattern_completion_metrics.json`

evolution:
- this is an evolution of the earlier memory simulation baseline documented in [[pattern_completion]] and [[memory_systems_to_kda_mla]]

## what was done

- repaired the update rule to true in-place asynchronous hopfield updates
- added a shuffled-weight control that preserves the weight histogram while removing learned structure
- added corruption, capacity, and scaling sweeps
- added trial-level metrics, confidence intervals, paired comparison statistics, run metadata, and a machine-readable json artifact
- reran the simulation after fixing scaling aggregation and degenerate-confidence-interval edge cases

## configuration

- neurons: 200
- stored patterns in baseline corruption sweep: 5
- corruption sweep: 0.00 to 0.50
- corruption trials per level: 30
- capacity sweep pattern counts: 2, 8, 14, 20, 28, 36, 44, 52, 60
- capacity sweep trials per point: 20
- scaling network sizes: 100, 200, 300
- scaling load factors: 0.05, 0.10, 0.138, 0.18, 0.22
- scaling trials per point: 12
- max iterations: 100
- numpy seed: 42

## key results

- hebbian overlap at 30% corruption: 0.982 with 95% ci [0.946, 1.018]
- shuffled-control overlap at 30% corruption: 0.083 with 95% ci [0.056, 0.110]
- paired overlap delta at 30% corruption: 0.899
- permutation p-value for the paired delta: 0.001
- exact retrieval remained at 1.0 through 25% corruption in this run and dropped to 0.0 by 50% corruption

## verdict

the repaired recurrent weights produce strong pattern cleanup relative to the shuffled control, so this run is a valid ca3-like attractor-memory baseline.

this is evidence for nonlinear recurrent cleanup in the simulation, not evidence that todorov already implements hippocampal pattern completion and not evidence for complementary learning systems.

## limitations

- this is still a classical hopfield-style toy model, not a full dg-ca3-ca1 circuit
- the 0.138n reference is theoretical context, not a theorem proved by this run
- the run was executed in a dirty git worktree, so the artifact records local state rather than a pristine revision
- the simulation is cpu-friendly and intentionally small, so the result is useful for mechanistic contrast, not scale claims

## see also

- `wiki/tests/index.md` — tests/ catalog
- `wiki/PROJECT_PLAN.md` — canonical project state
- `wiki/INDEX.md` — full wiki navigation map
- `wiki/OPERATING_DIRECTIVE.md` — rules governing this article
