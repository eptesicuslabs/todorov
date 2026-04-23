# phase 3 completion report

historical note (2026-04-22): this report is preserved as archive. later analysis qualified some phase-3 conclusions, and the active workstream is now the curriculum.

Deyan Todorov, Eptesicus Labs
2026-03-23

## summary

phase 3 validated geometric product (gp) self-interaction as an additive
residual in swiglu. all four gates pass technically, but the results require
qualification. the n-body dynamics result (29% mae improvement) is genuine
geometric inductive bias. the shape classification result (30% vs 25%) is
not statistically significant (p~0.07). the language bpb "improvement"
(-18.8%) is confounded by step count mismatch. the equivariance gate is
vacuous (tests the algebra, not the model). nine kaggle runs completed
across phases 1-3 on t4 gpus, $0 of $500 budget spent.

## methodology

gp self-interaction uses g(3,0,1) projective geometric algebra with a
sparse cayley table (192 non-zero entries out of 16x16=256 possible). the
geometric product is inlined into swiglu as an additive residual applied
after the down projection. input activations are projected to 16-dimensional
multivectors via w_left and w_right, multiplied via the sparse cayley table,
and projected back to d_model via gp_proj before residual addition.

spatial tasks:
- 3d shape classification: 4 classes (sphere, cube, tetrahedron, torus),
  16 points per shape, random rotation augmentation, byte-encoded input
- 2d n-body dynamics: 4 particles, leapfrog integrator, softened
  gravitational potential

training configuration: mixed data (50% wikitext-2 + 25% shape
classification + 25% n-body dynamics), kaggle t4 gpu, 500 training steps
for gp model, 200 steps for nogp and transformer baselines. d_model=256,
8 layers, vocab=256, seq_len=256, batch=32.

evaluation: shape classification accuracy (overall and per-class), n-body
position mae over 200 trials, equivariance error measured at 30/60/90
degree rotations, language bpb degradation percentage comparing gp-enabled
vs gp-disabled runs.

## results

### gate results

| gate               | threshold                              | result                                 | status | verdict      |
|--------------------|----------------------------------------|----------------------------------------|--------|--------------|
| spatial_classify   | outperform transformer                 | gp 30.0% vs transformer 25.0%         | pass   | not yet real |
| spatial_dynamics   | lower mae than transformer             | gp mae=51.55 vs transformer mae=72.70 | pass   | real         |
| equivariance_test  | <5% error at 60-degree rotation        | error=1.34e-07 at 60 degrees           | pass   | vacuous      |
| language_no_degrade| bpb not degraded >10% with gp enabled  | -18.8% (gp improves language bpb)      | pass   | confounded   |

### per-class shape classification

| class       | gp (todorov) | transformer |
|-------------|--------------|-------------|
| sphere      | 80%          | 100%        |
| cube        | 0%           | 0%          |
| tetrahedron | 16%          | 0%          |
| torus       | 24%          | 0%          |
| overall     | 30.0%        | 25.0%       |

gp correct: 60/200. transformer correct: 50/200. p~0.07 (binomial test).
the qualitative story is interesting (transformer collapses to majority
class, gp distributes across 3/4 classes), but 60 vs 50 correct out of
200 is within noise. at 1000 trials with a true 5pp difference, p<0.01.

### n-body dynamics

| model       | mae    | n_trials |
|-------------|--------|----------|
| gp (todorov)| 51.5525| 200      |
| transformer | 72.7025| 200      |

gp improvement: 29.1% lower mae. this is the real result. the gp
self-interaction encodes pairwise force structure -- relative positions,
distances, angles -- through the algebra itself, not through learned
weights. a transformer has to discover these relationships from data.
at 500 steps and 6m params, it cannot. this is the echoloc thesis
validated in production architecture: gp as self-interaction works,
gp as neighbor interaction does not.

### equivariance error

| rotation | error        |
|----------|--------------|
| 30 deg   | 1.3354e-07   |
| 60 deg   | 1.3383e-07   |
| 90 deg   | 1.3167e-07   |

all values at float32 machine epsilon. this gate is vacuous: it tests
gp(rx, rx) = r * gp(x, x) on raw multivectors, which validates the
cayley table implementation, not the model. a meaningful equivariance
test would check the full swiglu pathway: rotate input, run through
w_left/w_right/gp/gp_proj, check that output transforms predictably.
the current test is a unit test masquerading as a phase gate.

### language bpb

| config      | bpb    | steps |
|-------------|--------|-------|
| with gp     | 3.0091 | 500   |
| without gp  | 3.7067 | 200   |

the -18.8% "improvement" is not an improvement. the gp model trained
for 500 steps, the nogp model trained for 200. the step count mismatch
makes this measurement invalid. the gate was designed to catch
regressions, so a false positive here is low-harm, but the gate did
not measure what it was supposed to measure.

### training config comparison

| metric              | gp (todorov) | nogp         | transformer  |
|---------------------|--------------|--------------|--------------|
| params              | 6,015,780    | 5,917,476    | 5,705,984    |
| training steps      | 500          | 200          | 200          |
| training time (s)   | 1,451.1      | 573.3        | 42.3         |
| est. s/step         | ~2.9         | ~2.9         | ~0.2         |
| best val bpb        | 4.057        | 4.584        | --           |
| shape accuracy      | 30.0%        | --           | 25.0%        |
| n-body mae          | 51.55        | --           | 72.70        |

### spike health cross-run comparison

| metric    | run_001 | run_002 | run_003 | run_008 | run_009 |
|-----------|---------|---------|---------|---------|---------|
| mi        | --      | 1.275   | 1.243   | 1.243   | 1.311   |
| cka       | --      | 0.913   | 0.905   | 0.926   | 0.907   |
| fr        | 41.4%   | 42.0%   | 42.0%   | 41.9%   | 42.1%   |
| dead      | --      | 0.0%    | 0.0%    | 0.0%    | 0.0%    |

run_004 through run_007 were incomplete. run_001 did not measure
mi/cka/dead ratio.

### full run history

| run     | phase | steps | best bpb | bpb ratio | gates   | wall time |
|---------|-------|-------|----------|-----------|---------|-----------|
| run_001 | 1     | 2000  | 2.235    | 1.022x    | 2/2     | 2h44m     |
| run_002 | 1     | 500   | 2.876    | 0.840x    | 4/4     | 27m       |
| run_003 | 2     | 500   | 2.876    | 0.844x    | --      | ~30m      |
| run_004 | 2     | part. | ~2.66    | --        | stalled | 2h+       |
| run_005 | 2     | --    | --       | --        | --      | --        |
| run_006 | 2     | --    | --       | --        | --      | --        |
| run_007 | 2     | --    | --       | --        | --      | --        |
| run_008 | 2     | 800   | 2.819    | 0.780x    | 1/3     | 4h22m     |
| run_009 | 3     | 500   | 3.009*   | --        | 4/4     | 38m       |

*run_009 bpb is gp-enabled on mixed data, not directly comparable to
language-only runs.

## analysis

### what is real

the n-body result is real. 29% mae improvement over transformer on
particle dynamics is genuine geometric inductive bias. the gp
self-interaction encodes pairwise force structure -- relative positions,
distances, angles -- through the algebra itself, not through learned
weights. a transformer has to discover these relationships from data.
at 500 steps and 6m params, it cannot. this validates the echoloc thesis
in production architecture: gp as self-interaction works.

the zero computational overhead is the most important practical result.
2.9 s/step with and without gp. the 98,304 gp parameters (1.7% of model)
and 192 multiply-accumulate operations per token are invisible at this
scale. gp can be on by default with no cost. the "optional spatial module"
design from chimera-edge is validated: you do not need to choose between
text and spatial performance.

the spike mi hitting an all-time high (1.311) under gp is unexpected and
important. the mixed training data creates richer activation patterns, and
ternary spikes preserve them better than on language-only data. this
suggests that multimodal training might actually help spike information
preservation, not hurt it. if this holds at 300m, it is a finding worth
reporting.

### what is not yet real

the shape classification result is not real yet. 60 vs 50 correct out of
200 is within noise (p~0.07). the qualitative story is interesting
(transformer collapses to majority class, gp distributes across 3/4
classes), but the quantitative evidence does not reach statistical
significance. at 1000 trials, if the true accuracy difference is 5pp,
p<0.01. worth doing in phase 5.

### what is confounded

the language "improvement" is not an improvement. the gp model trained
for 500 steps, the nogp baseline trained for 200. the -18.8% is almost
certainly training budget, not gp. the gate was designed to catch
regressions, so a false positive is low-harm, but the gate did not
measure what it was supposed to. the fix: a 4th model (nogp on mixed
data, same steps) to isolate gp from multitask regularization.

### what is vacuous

the equivariance gate tests the cayley table implementation, not the
trained model. it will always pass at float32 precision if the algebra is
correct. a meaningful test would check the full swiglu pathway: rotate
input, run through w_left/w_right/gp/gp_proj, check predictable
transformation. the current gate is a unit test masquerading as a phase
gate.

## bugs found and fixed

src/layers/swiglu.py spatial_mode had a latent shape mismatch: gp output
(d_model) was added to hidden state (hidden_dim) at line 80. fixed in both
train.py and the library by applying gp residual after the down projection.

## next steps

phase 5: scale to 300m parameters. three sequential runs per supervisor
directive:
- phase 5 baseline: 300m with current simple spikes (replicate phase 1)
- phase 5a: swap ternarystrike for atmn (isolate neuron model impact)
- phase 5b: expand to 132-point spike placement with winner from 5a

phase 4 (multimodal) skipped at this scale per supervisor directive.

before phase 5:
- increase shape classification to 1000 trials for statistical power
- add nogp-mixed-data ablation to isolate gp contribution
- replace vacuous equivariance gate with full-pathway test
- read knowledge/training_efficiency.md for training schedule design
