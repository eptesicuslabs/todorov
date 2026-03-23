# phase 3: lessons, mistakes, and knowledge reference

Deyan Todorov, Eptesicus Labs
2026-03-23

## knowledge files used

| file | role in phase 3 |
|------|----------------|
| knowledge/geometric_algebra.md | primary reference for gp implementation: g(3,0,1) basis, cayley table construction, sparse einsum approach, gatr architecture patterns, pin-equivariant linear maps |
| knowledge/ternary_spikes.md | spike health thresholds (mi > 0.1, cka > 0.3, fr 30-60%), used to verify gp does not degrade spike encoding |
| knowledge/hybrid_architectures.md | 3:1 layer ratio validation, confirmed gp should be added to mlp (swiglu) not attention layers |
| knowledge/kda_channel_gating.md | state monitoring patterns used in post-run spike analysis |
| knowledge/mamba3_architecture.md | confirmed mamba3 layers unaffected by gp addition (gp only in swiglu) |
| knowledge/mla_compression.md | confirmed mla layers unaffected by gp addition |
| knowledge/context_extension.md | not directly used in phase 3, but informed understanding of state quality monitoring |
| knowledge/training_efficiency.md | not read during phase 3 (should have been checked for mixed-data training patterns) |
| knowledge/papers.md | referenced gatr (arxiv 2305.18415) for n-body benchmark comparison |

## mistakes

### 1. skipped mandatory pre-push subagents (steps 2-4)

the agent skipped the research agent, plan compliance agent, and
self-critique agent steps of the autoresearch loop. it rationalized
this by thinking "i already explored the codebase" and "the smoke
test will catch everything." this is the exact failure mode the
protocol was designed to prevent.

the shape mismatch bug in swiglu (gp output d_model added to
hidden_dim) would have been caught by the self-critique agent's
shape tracing. the step count mismatch would have been caught by
the plan compliance agent. you cannot rely on discipline alone --
you need mechanical enforcement.

fix: updated program.md and scripts/autoresearch_loop.md with push
gate verification, named artifacts, anti-rationalization rules, and
structural blocking (no path to push without all 4 subagent artifacts).

### 2. step count mismatch between models

the gp model trained for 500 steps while the nogp and transformer
baselines trained for only 200 steps. this confounds the language
degradation comparison: the -18.8% "improvement" is almost certainly
training budget, not gp.

the gate was designed to catch regressions. a false positive (gp
appears to help when it does not) is low-harm, but the gate did not
measure what it was supposed to measure.

fix: train all comparison models at the same step count. add a 4th
model (nogp on mixed data, same steps) to isolate gp contribution
from multitask regularization.

### 3. swiglu shape bug in library (now fixed)

src/layers/swiglu.py line 80 did `hidden = hidden + gp_out` where
hidden is (b, t, hidden_dim) and gp_out is (b, t, d_model). these
dimensions differ when mlp_ratio != 1.0. the bug was never caught
because spatial_mode was "implemented but not yet validated" -- unit
tests only tested the gp algebra, not the swiglu integration.

the review correctly identified this as technical debt with a timer:
"the moment someone imports the library code with spatial_mode=True,
it crashes. fix it now, not before phase 5."

fix applied to both train.py and src/layers/swiglu.py: gp residual
applied after down projection, so both tensors are in d_model space.

### 4. equivariance gate is vacuous

the equivariance gate tests gp(rx, rx) = r * gp(x, x) on raw
multivectors. this validates the cayley table implementation, not
the trained model. it will always pass at float32 precision if the
algebra is correct. it is a unit test masquerading as a phase gate.

a meaningful equivariance test would check the full swiglu pathway:
rotate input, run through w_left/w_right/gp/gp_proj, check that
output transforms predictably. this tests whether the learned
projections exploit equivariance for downstream tasks.

fix for phase 5: replace the raw-algebra equivariance gate with a
full-pathway equivariance test.

### 5. shape classification is not yet real

30% vs 25% on 200 samples has p~0.07 (binomial test). does not reach
p<0.05. the qualitative story (transformer collapses, gp generalizes)
is interesting but 60 vs 50 correct out of 200 is within noise.

at 1000 trials with a true 5pp difference, p<0.01. worth doing in
phase 5.

### 6. did not read training_efficiency.md

knowledge/training_efficiency.md (77.6 kb) was not read during phase 3
planning. it likely contains information about mixed-data training
patterns, multitask learning, and data mixing ratios that would have
informed the experimental design and prevented the step count confound.

## what is real

### n-body dynamics: genuine geometric inductive bias

29% mae improvement (51.55 vs 72.70) is the strongest phase 3 result.
the gp self-interaction encodes pairwise force structure -- relative
positions, distances, angles -- through the algebra itself, not through
learned weights. a transformer has to discover these relationships from
data. at 500 steps and 6m params, it cannot.

this is the echoloc thesis validated in production architecture: gp as
self-interaction works, gp as neighbor interaction does not. comparable
to eqmotion's 24% improvement on particle dynamics.

### zero computational overhead

2.9 s/step with and without spatial_mode. the 98,304 gp parameters
(1.7% of model) and 192 multiply-accumulate operations per token per
layer are invisible at this scale. gp can be on by default with no cost.
the "optional spatial module" design from chimera-edge is validated: you
do not need to choose between text and spatial performance.

this is the most important practical result. it means gp is free to
include in every configuration going forward.

### spike mi at all-time high under gp

mi=1.311 in run_009 exceeds all prior runs (previous best 1.275 in
run_002). the mixed training data creates richer activation patterns,
and ternary spikes preserve them better than on language-only data.

this is unexpected and important. it suggests that multimodal training
might actually help spike information preservation, not hurt it. if
this holds at 300m, it is a finding worth reporting independently.

### transformer collapse vs gp generalization

on shape classification, the transformer collapsed to a constant
predictor (always sphere, 100% class-0 accuracy, 0% on other classes).
the gp model achieved non-trivial discrimination on 3 of 4 classes
(sphere 80%, tetrahedron 16%, torus 24%). this qualitative difference
is stronger evidence than the raw accuracy numbers, even though the
quantitative result is not statistically significant.

## key numbers

| metric | value | context |
|--------|-------|---------|
| total kaggle runs | 9 | phases 1-3 |
| budget spent | $0 | of $500 |
| gp param overhead | 98,304 | 1.7% of 5.9m base |
| gp time overhead | 0% | 2.9 s/step with and without |
| n-body improvement | 29% | mae 51.55 vs 72.70, real |
| shape accuracy delta | +5pp | 30% vs 25%, p~0.07, not yet real |
| equivariance error | 1.34e-07 | validates algebra, not model |
| language degradation | -18.8% | confounded by step count mismatch |
| spike mi | 1.311 | all-time high across 9 runs |
| phase 3 wall clock | 38 min | kaggle t4 |

## recommendations for phase 5

1. replace vacuous equivariance gate with full-pathway swiglu test
2. add nogp-mixed-data ablation (same steps) to isolate gp contribution
3. increase shape classification to 1000 trials for p<0.01
4. read knowledge/training_efficiency.md before designing training schedule
5. consider 60-100m params as a stepping stone before 300m
6. never skip pre-push subagents -- the push gate now blocks this structurally
7. if spike mi remains elevated at 300m under mixed training, report it
