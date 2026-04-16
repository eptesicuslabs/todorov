# overwrite sweep results

status: historical context only. frozen as of 2026-04-16. do not edit.

last updated: 2026-04-12

## what this is

this note records the focused overwrite sweep for the asymmetric matrix-memory simulation in `neuroloc/simulations/memory/asymmetric_outer_product_recall.py`. the decay sweep identified `decay=0.90` as the first useful retention knee at `d_head=64`, so this follow-up asks a single question at that operating point: once recall becomes viable, does the overwrite subtraction help or hurt?

## run configuration

the sweep fixed the state width and retention at the first useful knee and increased trial count to reduce noise in the erasure delta:

- head dimension `64`
- pattern counts `{8, 16, 32, 48, 64}`
- query noise `{0.0, 0.1}`
- decay fixed at `0.90`
- erasure off and on
- 128 trials per cell

the generated artifacts live at:

- `neuroloc/output/overwrite_sweep/overwrite_sweep_summary.json`
- `neuroloc/output/overwrite_sweep/pattern_delta_summary.json`
- `neuroloc/output/overwrite_sweep/overwrite_delta_heatmap.png`

## the headline result

the exact-query mean overwrite delta is defined here as:

$$
\Delta = \text{mean raw cosine with erasure on} - \text{mean raw cosine with erasure off}
$$

averaged across the seven tested encodings, the sweep gives:

| pattern count | exact-query mean $\Delta$ across encodings | exact-query sign split | noisy-query mean $\Delta$ across encodings |
|---|---:|---|---:|
| 8 | `+0.0063` | 6 positive, 1 negative | `+0.0012` |
| 16 | `-0.0336` | 0 positive, 7 negative | `-0.0533` |
| 32 | `-0.0882` | 0 positive, 7 negative | `-0.0433` |
| 48 | `-0.0043` | 3 positive, 4 negative | `-0.0487` |
| 64 | `-0.0044` | 1 positive, 6 negative | `-0.0434` |

the important line is the first useful retention knee, `pattern_count=32`. once the memory has enough retention to reopen 32-pattern recall, erasure hurts every tested encoding.

## the key cells

at exact query, `pattern_count=32`, and `decay=0.90`:

- `bounded_all`: `0.6009 -> 0.5020`, delta `-0.0988`
- `identity_all`: `0.6005 -> 0.5021`, delta `-0.0984`
- `ternary_per_dim_all`: `0.5486 -> 0.4478`, delta `-0.1007`

the pattern is not restricted to one encoding family. the best exact-query help case at 32 patterns is still negative: `dense_key_topk20_value` falls by `-0.0717`.

the low-load `pattern_count=8` case is the only place where overwrite is modestly positive on average, but that is not the planning bottleneck. the run-planning question is whether overwrite helps once retrieval first becomes useful at realistic load. at that point, it does not.

## what this says about the mechanism

three conclusions are justified by the sweep:

1. overwrite is not a generic rescue for the dense-key asymmetric memory. at the first useful retention knee it is directionally wrong across every tested encoding.
2. the earlier "read the erasure-on cells before concluding" warning from round B is now resolved for the most relevant baseline condition. the answer is negative, not ambiguous.
3. any future erasure claim has to be conditional. it may still become helpful in a different learned regime, but it is no longer evidence-backed as the default dense baseline.

## planning implication

the scheduled dense baseline should not launch with erasure on. the evidence-backed baseline is now the `god_machine.py` preset `run1_baseline_noerasure`: dense keys, dense values, no overwrite subtraction, and the previously confounding auxiliary mechanisms removed.

if erasure returns later, it should return only as an explicit post-baseline ablation after dense-key retrieval has been validated. this sweep moves overwrite from "baseline assumption" to "later comparison branch."

## limitations

1. this is still a synthetic-gaussian simulation, not a trained-activation replay.
2. the sweep tests one retention point, `decay=0.90`, because that is the first exact-query reopening point for 32 patterns at `d_head=64`.
3. the conclusion is strongest for the baseline load regime around 16-32 patterns. at 48-64 exact-query patterns, the effect is near-neutral to mildly negative rather than catastrophically negative.
4. this note does not prove erasure is always harmful after training. it proves only that the standalone dense-key baseline should not assume erasure is helpful.

## sources

- `neuroloc/simulations/memory/asymmetric_outer_product_recall.py`
- `neuroloc/output/overwrite_sweep/asymmetric_outer_product_recall/asymmetric_outer_product_recall_metrics.json`
- `neuroloc/output/overwrite_sweep/overwrite_sweep_summary.json`
- `neuroloc/output/overwrite_sweep/pattern_delta_summary.json`
- `neuroloc/output/overwrite_sweep/overwrite_delta_heatmap.png`
- `neuroloc/wiki/PROJECT_PLAN.md`

## related docs

- `neuroloc/wiki/tests/encoding_simulation_round_b.md`
- `neuroloc/wiki/tests/decay_sweep_results.md`

## update history

- **2026-04-12** — deyan todorov — file created. records the focused overwrite sweep at `decay=0.90` and 128 trials per cell, shows that erasure hurts all seven encodings at the first useful 32-pattern retention knee, and moves overwrite from the scheduled baseline into a later explicit ablation.

## see also

- `wiki/tests/index.md` — tests/ catalog
- `wiki/PROJECT_PLAN.md` — canonical project state
- `wiki/INDEX.md` — full wiki navigation map
- `wiki/OPERATING_DIRECTIVE.md` — rules governing this article
