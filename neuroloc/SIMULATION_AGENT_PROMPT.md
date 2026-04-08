# simulation agent briefing

you are continuing work on the neuroloc project inside the todorov repository
at `C:\Users\deyan\Projects\todorov\neuroloc\`. read this entire document
before taking any action.

## what you are building toward

todorov is a hybrid spiking neural architecture that achieves 0.663x BPB at
267M parameters (33.7% better than transformer baseline). it uses ternary
spikes, recurrent associative memory, and geometric algebra.

todorov is a proof-of-concept. it will be replaced by an actual neural
computer -- a digital brain with inference capabilities that computes through
real spiking dynamics, temporal integration, and local learning rules, not
through matrix multiplications and backpropagation. general cognition and
pattern recognition, not a language model.

the neuroloc project is the research phase that informs the design of this
digital brain. it maps biological neural computation to todorov's architecture,
identifies which biological principles are genuine vs superficial, and tests
whether those principles can actually compute at scale.

the simulations are the experimental arm of this research. they are not
educational demos. they are prototype components and validation experiments
for the digital brain.

## the llm-wiki

neuroloc uses the llm-wiki pattern (inspired by karpathy's april 2026 gist).
the wiki is an obsidian vault at `neuroloc/wiki/` with this structure:

```
neuroloc/wiki/
  mechanisms/     61 articles on biological neural computation
  bridge/         18 documents mapping biology to todorov source code
  comparisons/    13 side-by-side analyses (biological vs artificial)
  concepts/       7 introductory and reference articles
  entities/       33 researcher and lab notes
  knowledge/      13 ML architecture research files
  synthesis/      5 cross-domain integration articles
  Home.md         vault home page with navigation links
  index.md        flat reference catalog of all articles
  log.md          chronological operation log (append-only)
```

open as an obsidian vault at `neuroloc/wiki/`. all internal cross-references
use `[[wikilink]]` format with shortest-path resolution.

**how to contribute to the wiki**: if your research produces findings worth
preserving, write them as articles following the existing format. mechanism
articles go in `mechanisms/`, bridge analyses go in `bridge/`. every article
must have a "why this matters" section, a challenges section with at least 3
criticisms, and inline ML analog notes. update `index.md` and `log.md` after
adding articles. read existing articles for format reference before writing.

the wiki is the project's long-term memory. what you learn goes into the wiki
so future agents and sessions can build on it.

## your primary mission: simulations

there are currently 18 brian2 + python simulations at `neuroloc/simulations/`,
organized by domain. each runs standalone on CPU in under 10 minutes, outputs
PNG figures, and has a README.md. they are toy demos -- 20-200 neurons, basic
matplotlib plots, no quantitative analysis, no parameter sweeps, no scaling
studies, no connection to todorov's evaluation metrics.

your job is to make them dramatically better across three tracks:

### track A: depth (upgrade existing simulations)

take each of the 18 existing simulations from toy demo to research-grade:

- **parameter sweeps**: vary key parameters across biologically meaningful
  ranges. plot results as heatmaps or line plots with error bars.
- **scaling studies**: test if the principle holds at 100, 1K, 10K+ neurons.
  does the computation scale or collapse?
- **information-theoretic metrics**: measure mutual information (MI), centered
  kernel alignment (CKA), firing rate, and capacity -- the same metrics todorov
  uses (see `neuroloc/wiki/bridge/population_coding_to_spike_health.md`).
- **statistical rigor**: multiple runs, confidence intervals, p-values < 0.01
  for claims. no eyeballing.
- **automated quantitative output**: every simulation produces a JSON metrics
  file alongside its figures. metrics are machine-comparable across runs.

### track B: integration (multi-mechanism circuits)

build 3-5 new simulations that combine multiple biological mechanisms into
integrated circuits:

- **cortical microcircuit**: excitatory/inhibitory balance + STDP + lateral
  inhibition + dendritic gating, all interacting. does the circuit self-
  organize? does it compute useful representations?
- **associative memory circuit**: hebbian storage (outer product) + pattern
  separation (sparse coding) + pattern completion (recurrent dynamics).
  test capacity, noise robustness, interference.
- **timescale circuit**: fast dynamics (gamma-like, ~40 Hz) nested within
  slow dynamics (theta-like, ~6 Hz). does cross-frequency coupling emerge?
  does it improve information coding?
- **sensory processing pipeline**: input encoding -> lateral inhibition ->
  sparse representation -> temporal integration -> readout. end-to-end
  information flow through a neural circuit.
- **learning circuit**: three-factor learning rule + eligibility traces +
  neuromodulatory gating. can a circuit learn a simple association without
  backpropagation?

these test whether mechanisms COMPOSE or INTERFERE. individual demos prove
nothing about composition.

### track C: bridge validation (design experiments)

one simulation per top-5 intervention from the bridge documents:

1. **ATMN leak vs no leak** (`bridge/neuron_models_to_atmn.md`):
   h_t = (1-alpha)*u_{t-1} + x_t vs h_t = x_t + (1/tau)*u_{t-1}.
   measure: temporal integration fidelity, drift over long sequences.

2. **k-WTA vs threshold spiking** (`bridge/lateral_inhibition_to_adaptive_threshold.md`):
   top-k selection vs alpha*mean(|x|) threshold.
   measure: MI, gradient flow proxy, capacity.

3. **progressive spike activation** (`bridge/development_to_training_curriculum.md`):
   no spikes during warmup, linear ramp over 10% of training.
   measure: representation quality before/during/after spike onset.

4. **BCM-like adaptive alpha** (`bridge/plasticity_to_kda_delta_rule.md`):
   alpha_eff = sigmoid(alpha_log + gamma*log(||S_t||)).
   measure: state norm dynamics, saturation prevention, long-sequence stability.

5. **GP vs random bilinear vs quaternion product** (`bridge/spatial_computation_to_pga.md`):
   G(3,0,1) geometric product vs random 16x16->16 bilinear map vs quaternion product.
   measure: MI improvement, representation geometry, whether structure matters.

each of these directly answers a design question for the digital brain.

## how to work

### subagents

you MUST use subagents extensively and in parallel. specifically, use the
non-claude-suffix variants:

- **researcher** subagents: for web research on simulation methods, brian2
  best practices, spiking neural network benchmarks, neuromorphic computing
  techniques, and relevant papers.
- **prosecutor** subagents: for adversarial review of simulation designs
  before implementation. does the simulation actually test what it claims?
  are there confounds? is the analysis rigorous?
- **documentator** subagents: for auditing that simulation READMEs, wiki
  articles, and metrics outputs are accurate and complete.

launch multiple subagents in parallel whenever possible. never do substantial
work inline when a subagent can do it better.

### research before execution

a lot of planning needs to be done before you write any code. for each
simulation you plan to build or improve:

1. **research phase**: launch researcher subagents to find the best
   simulation methodology. what do published papers use? what are the
   standard benchmarks for this mechanism? what parameter ranges are
   biologically meaningful? what are the known failure modes?

2. **design phase**: write a simulation design document describing:
   what question it answers, network architecture, parameters and their
   ranges, metrics to measure, expected results, controls, failure criteria.

3. **review phase**: launch prosecutor subagents to critique the design.
   does it actually test the hypothesis? are there confounds? is it
   statistically powered?

4. **implementation**: only after research, design, and review.

5. **validation**: run the simulation, check results against expectations,
   launch documentator to verify outputs.

### kaggle execution

all brian2 simulations run on CPU and fit within kaggle's free tier. for
pytorch-based experiments (BCM-like alpha, GP vs bilinear, integration
circuits), use the kaggle T4 GPU.

**kaggle setup:**
- kernel: `dttdrv/todorov-autoresearch`
- auth: read `~/.kaggle/kaggle.json`, set `KAGGLE_API_TOKEN` env var
  BEFORE importing the kaggle package (it auto-authenticates on import)
- always use `acc="NvidiaTeslaT4"` in kernels_push
- use `scripts/kaggle_exec.py` for push/poll/pull (the mcp-server-kaggle
  package is BROKEN -- do not use it)

**working pattern:**
```
KAGGLE_API_TOKEN="KGAT_..." python -u scripts/kaggle_exec.py [script.py] [timeout_min]
```

**kaggle quirks:**
- `kernels_status()` returns 404 for ~60s after push on new kernels
- `kernels_output()` returns the LAST COMPLETED run, not the current one
- no streaming log API -- view live output on kaggle.com UI only
- always reuse the existing kernel `dttdrv/todorov-autoresearch`

**what runs where:**
- brian2 CPU simulations (track A upgrades): kaggle CPU, no GPU needed
- pytorch bridge experiments (track C: BCM alpha, GP control): kaggle T4
- integration circuits (track B): depends on framework -- brian2 on CPU,
  pytorch on T4
- the brian2 constraint: < 10 min per simulation, pip install brian2 only

**t4 limitations (from prior runs):**
- triton 3.3+ dropped sm_75 (T4) support. pin torch==2.6.0 + triton==3.2.0
  if using flash-linear-attention
- T4 lacks bf16 tensor cores: use fp16 for all fla kernels
- 16GB VRAM. budget accordingly for pytorch experiments

### eara ultra loop protocol

when running experiment sequences (particularly track C bridge validations),
you MUST follow the eara ultra loop protocol defined in
`scripts/autoresearch_loop.md`. this is the same protocol used for all
todorov architecture experiments.

**what eara ultra is:** an autonomous experiment loop with mandatory
subagent verification at every step. it prevents the agent from pushing
broken code, running confounded experiments, or wasting GPU time on
untested changes. the protocol exists because a previous agent (run_009)
skipped pre-push checks and pushed code with a shape mismatch bug and a
confounded experimental design.

**the loop state machine:**
```
INIT -> ANALYZE -> HYPOTHESIZE -> IMPLEMENT -> PRE_CHECK ->
MEASURE -> GATE_CHECK -> DECIDE -> KEEP or DISCARD ->
LOG -> POST_ANALYSIS -> TERMINATE_CHECK -> (ANALYZE or DONE)
```

**pre-push phase (before ANY kaggle push):**
1. REPORT: document the change, hypothesis, expected outcome
2. RESEARCH AGENT [separate subagent]: verify technical assumptions,
   API shapes, GPU compatibility, published benchmarks
3. PLAN COMPLIANCE AGENT [separate subagent]: verify changes follow
   CLAUDE.md spec and current phase plan
4. SELF-CRITIQUE AGENT [separate subagent]: review for crash risks
   (shapes, imports, memory, API breaking changes)
5. SMOKE TEST AGENT [separate subagent]: write and run a targeted
   test for the changed code paths
6. CODE QUALITY AGENT [separate subagent]: review code quality,
   spec compliance, evidence requirements
7. FIX: apply all fixes from agents 2-6
8. REVIEW GATE: all 5 agent artifacts must exist and pass
9. AGENT COUNT GATE: verify exactly 5 subagent dispatches occurred

**post-completion phase (after results return):**
- launch 4 PARALLEL agents: analysis, research grounding, plan
  compliance, documentation
- INVESTIGATION agents if ANY anomaly found
- update all docs (STATUS_BOARD.md, program_status.yaml, wiki)

**critical rules:**
- steps 2-6 are each a SEPARATE SUBAGENT. you cannot do them inline.
  you cannot combine them. each produces a named artifact.
- never push without all pre-push agents passing
- never queue multiple kaggle runs simultaneously
- always run investigation agents on anomalies
- the protocol config is in `eara.yaml` at the repo root

**rationalization halt signals (if you think any of these, STOP):**
- "this change is too small to need verification"
- "I already tested this manually"
- "the subagents will just slow things down"
- "I can combine steps 2 and 3 to save time"
- "this obviously works"

these thoughts are protocol violations. the protocol exists because
agents who thought these things pushed broken code.

### secondary mission: project research

your primary job is simulations. however, if your research uncovers findings
relevant to the wiki -- new papers, new mechanisms, corrections to existing
articles -- you are welcome to contribute them. this is not expected but is
appreciated. follow the wiki contribution format described above.

## project rules (non-negotiable)

read `CLAUDE.md` at the repo root. the critical rules:

- zero comments in code. no inline, no block, no docstrings, no TODOs.
- zero emojis anywhere.
- zero AI attribution. no Co-authored-by. sole author: Deyan Todorov.
- lowercase in all docs, commits, and readme.
- the adversarial thinking rule applies to every claim.

## quality bar for simulations

every simulation must:

- run standalone on CPU in under 10 minutes (brian2 constraint)
- produce publication-quality figures (not matplotlib defaults -- proper
  axis labels, consistent color schemes, appropriate font sizes)
- output a JSON metrics file with quantitative results
- have a README.md explaining: what it tests, the biological basis, the
  experimental design, and the quantitative verdict
- connect explicitly to a wiki article or bridge document
- include at least one control condition
- report statistical significance for any claimed effect

## key files to read first

1. this file (SIMULATION_AGENT_PROMPT.md)
2. `HANDOFF.md` in this directory (full project state)
3. `NEUROLOC.md` at repo root (component registry, adversarial thinking rule)
4. `CLAUDE.md` at repo root (project rules, architecture, bug history)
5. `wiki/concepts/start_here.md` (wiki entry point)
6. `wiki/index.md` (flat catalog of all articles)
7. all 18 bridge documents in `wiki/bridge/` (these define what to test)
8. existing simulations in `simulations/` (understand the current baseline)

## the 15 adversarial findings

the wiki stress-tested whether todorov's biological analogies hold. 14 of 15
are superficial. the one genuine correspondence: KDA's outer product (k*v^T)
IS hebbian learning, mathematically identical.

your simulations should test the GENUINE principles, not validate the
superficial analogies. the bridge documents tell you which is which.

additionally: KDA's "delta rule" name is aspirational -- the implementation
has no targeted erasure term (just alpha * S + beta * k * v^T). this is a
decaying hebbian memory, not an error-correcting delta rule.

## what success looks like

when your work is done, the simulation suite should be able to answer:

- does hebbian outer-product memory scale beyond 200 neurons?
- does adding a leak term to LIF neurons improve temporal integration?
- does k-WTA produce different representations than threshold spiking?
- do biological timescales (theta/gamma) improve information coding?
- can a circuit learn associations without backpropagation?
- does geometric algebra structure matter, or does any bilinear map work?
- do multiple mechanisms compose constructively or destructively?

these answers determine what goes into the digital brain and what gets left
out. the simulations are the evidence. make them good enough to trust.
