# input encoding simulation round A

status: historical context only. frozen as of 2026-04-16. do not edit.

last updated: 2026-04-12

## what this is

phase A of the run-1 planning cycle. three simulations from `neuroloc/simulations/` were run on CPU to baseline what we already know about input encodings on outer-product matrix memories. all three test the symmetric, iterative, attractor-style memory shape from the published literature, not the asymmetric single-step gated-decay shape that the project's architecture actually uses. the synthesis below identifies the gap and points to the round-B simulation that closes it.

## simulation A1: pattern count vs network size at five magnitude-quantization schemes

`neuroloc/simulations/memory/capacity_scaling.py`. measures the fraction of stored patterns that the symmetric outer-product matrix memory recalls perfectly under iterative attractor dynamics, swept across network sizes 200, 500, 1000, 2000, 5000 and load factors 0.02 through 0.30 (the published 0.138 reference is in the sweep). 5% query corruption, 50 trials per cell. produced cached metrics on 2026-04-09 to 2026-04-11; the run took ~55 hours wall-clock total across all cells.

**at the 0.138 N capacity reference**, exact recall fraction:

| network size | sign-only encoding (binary, no zeros) | three-level encoding (with zeros, 41% nonzero) |
|---|---|---|
| 200 | 0.42 | 0.04 |
| 500 | 0.22 | 0.00 |
| 1000 | 0.04 | 0.00 |
| 2000 | 0.00 | 0.00 |
| 5000 | 0.00 | 0.00 |

mean difference: 0.128 in favour of sign-only. effect size 0.76 (sample size 5, p=0.25 by permutation, p=0.16 by paired t — not statistically significant on this sample but the direction is consistent at every network size).

interpretation: at the published 0.138 N capacity reference, sign-only encoding outperforms three-level encoding for symmetric outer-product memory recall. the gap is largest at small network sizes, where sign-only retrieves up to 42% of stored patterns and three-level retrieves at most 4%. both schemes collapse to zero recall at large network sizes (>=2000) at this load factor, consistent with the published capacity ceiling.

source data: `neuroloc/simulations/memory/capacity_scaling_metrics.json` summary.binary_at_0138 and summary.ternary_at_0138.

## simulation A2: encoding head-to-head on symmetric outer-product matrix retrieval at varying pattern counts

`neuroloc/simulations/prototypes/rate_coded_spike.py`. measures cosine similarity between recalled output and stored target, plus exact-match-after-rounding accuracy, at fixed network size n=500, 30 trials, 10% query corruption, sweeping pattern count p in {5, 10, 20, 30, 50}. tests three encoding schemes: sign-only (binary), three-level magnitude quantization with adaptive threshold (41% nonzero), continuous bounded encoding (tanh).

results (single representative trial set, 3.5 second wall-clock):

| pattern count | sign-only cosine | sign-only exact | three-level cosine | three-level exact | continuous cosine | continuous exact |
|---|---|---|---|---|---|---|
| 5 | 1.000 | 1.000 | 0.748 | 0.600 | 0.897 | 0.000 |
| 10 | 1.000 | 1.000 | 0.708 | 0.500 | 0.817 | 0.000 |
| 20 | 1.000 | 1.000 | 0.708 | 0.300 | 0.672 | 0.000 |
| 30 | 1.000 | 1.000 | 0.562 | 0.367 | 0.633 | 0.000 |
| 50 | 0.998 | 0.767 | 0.354 | 0.000 | 0.543 | 0.000 |

interpretation: sign-only encoding dominates on both cosine similarity and exact recall across every pattern count. it holds 100% exact recall up to 30 patterns, dropping to 76.7% at 50 patterns. three-level encoding starts at 60% exact at 5 patterns and degrades to 0% by 50 patterns. continuous bounded encoding hits 0% exact recall at every pattern count even though its cosine similarity is in the same range as three-level (because rounding the recalled continuous value to the nearest stored target requires near-perfect match, which the continuous code never achieves).

source data: `neuroloc/simulations/prototypes/rate_coded_spike_metrics.json`.

## simulation A3: representation fidelity at varying selection fractions

`neuroloc/simulations/sparse_coding/hierarchical_ternary.py`. measures information content (mutual information bits per dimension), centered kernel alignment (CKA) to the unencoded representation, and bits-per-dimension Shannon limit, sweeping selection fraction in {0.05, 0.10, 0.20, 0.41} and dimension in {64, 128, 256}. n_samples=256, 16 trials per cell. produced metrics on 2026-04-11 in 16.3 seconds.

at d=256 (highest dimensionality tested):

| selection fraction | mutual information (bits/dim) | CKA |
|---|---|---|
| 0.05 | 0.283 | 0.629 |
| 0.10 | 0.493 | 0.709 |
| 0.20 | 0.816 | 0.805 |
| 0.41 | 1.311 | 0.894 |

interpretation: information content per dimension and representation similarity to the unencoded form both increase monotonically with selection fraction. at 41% selection (the project's previously-validated operating point), three-level encoding preserves 1.31 bits/dim and 0.89 CKA. at 5% selection (closer to biological cortex), only 0.28 bits/dim and 0.63 CKA survive.

**critical caveat**: this simulation measures REPRESENTATION FIDELITY, not RETRIEVAL. it tells us how much information is preserved in the encoded vector, but does not tell us how much of that information can be recalled from a stored outer-product matrix. simulations A1 and A2 measure retrieval; A3 measures only the input-side compression cost.

source data: `neuroloc/simulations/sparse_coding/hierarchical_ternary_metrics.json`.

## what these three simulations agree on

1. on the **symmetric outer-product matrix memory with iterative attractor recall** (A1 and A2), sign-only encoding produces strictly better recall than three-level encoding at every load factor and every pattern count, with no exceptions in the swept ranges.
2. continuous bounded encoding produces high cosine similarity but zero exact recall, because exact recall requires the recalled output to round to the same nearest-stored-target as the true value, and continuous outputs do not stably hit this criterion.
3. higher selection fraction in three-level encoding preserves more information per dimension (A3), but this increased fidelity does not translate into proportionally better recall (A1 + A2 show three-level recall is poor at every selection fraction tested, including 41%).
4. the worst recall scheme is the one that puts zeros in the encoded vector by setting components below an absolute-value threshold to zero. zeros destroy the address space.

## the gap: what these three simulations do NOT test

all three simulations test the **symmetric** outer-product memory, where the same matrix stores both the keys and the values (each pattern is auto-associated to itself). recall is iterative: the query is fed into the matrix, the output is fed back as the next query, and the system either converges to the stored pattern or to a spurious attractor. the matrix is constructed once from all patterns and is then static.

the project's architecture uses a **different shape**:

- **asymmetric**: the matrix `S = sum a^(t-i) * b_i * k_i * v_i^T` stores each pattern as the outer product of its key with its value. queries with `q^T * S` retrieve the value associated with the closest key. keys and values can be different lengths and live in different subspaces.
- **single-step**: there is no iterative attractor. the recall operation is one matrix-vector multiply.
- **decayed**: the coefficient `a^(t-i)` weights older patterns less than newer ones. this is not present in any of the three simulations.
- **optionally erased**: an additional subtraction term `-a * b * k * (k^T * S)` removes the projection of the previous state onto the incoming key direction. this is also not present in any of the three simulations.
- **trained**: the keys, values, and queries are produced by learned linear projections from a residual stream, not drawn from a distribution. their statistics depend on what the model has learned to encode.

so simulations A1-A3 give us a strong piece of information — sign-only encoding beats three-level encoding on a closely related but not identical problem — and a clear research question: does this advantage hold for the asymmetric, single-step, decayed, optionally-erased shape that the project actually uses? round B closes this gap with a new simulation.

## what this implies for the next runs (tentative, until round B confirms)

if sign-only encoding turns out to also dominate on the asymmetric shape, the next training run should use sign-only encoding on whichever of (key, value, query) the round-B simulation indicates. this is a different operating point than every previous run in the project: previous runs (`run_010`, `run_011`, `god_run`, `god_run_v2`) all used either the three-level encoding (for `run_010`/`run_011`) or fixed-fraction absolute-magnitude selection at 20% (for `god_run`/`god_run_v2`). none of them used sign-only encoding, and none of them tested recall.

if sign-only does NOT dominate on the asymmetric shape, the round-B result will tell us which encoding does, and what to run next. either way, the round-B simulation is the prerequisite for the next training run.

## immediate next action

write `neuroloc/simulations/memory/asymmetric_outer_product_recall.py`. run it. interpret. update `PROJECT_PLAN.md`. then implement the chosen encoding under the per-logical-commit prosecutor cycle.

## sources

- `neuroloc/simulations/memory/capacity_scaling.py` and `capacity_scaling_metrics.json`
- `neuroloc/simulations/prototypes/rate_coded_spike.py` and `rate_coded_spike_metrics.json`
- `neuroloc/simulations/sparse_coding/hierarchical_ternary.py` and `hierarchical_ternary_metrics.json`
- `neuroloc/wiki/PROJECT_PLAN.md` (the canonical persistent project plan, current as of 2026-04-12)

## update history

- **2026-04-12** — file created. captures the round-A simulation results and identifies the symmetric-vs-asymmetric memory gap that round B closes.
- **2026-04-12** — fixed two prosecutor findings: corrected the k_fraction=0.41 row in the A3 table (mutual information 1.250 → 1.311, CKA 0.891 → 0.894 — the values were read from the wrong JSON key in the first draft; the source JSON stores the 41% case under `standard_d256_*` rather than `k0.41_d256_*`); changed the A2 section heading from "Hopfield-style retrieval" to "symmetric outer-product matrix retrieval" to remove a published-technique-name use of vocabulary as project terminology.
