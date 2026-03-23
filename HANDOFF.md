# todorov handoff: phase 3 and beyond

## what this project is

todorov is a 24-layer hybrid neural architecture (kda + mamba-3 + mla in 3:1
ratio) with ternary spikes. built from scratch in one session. all code is at
eptesicuslabs/todorov on github.

## where we are

phase 0 (foundation): complete. 76/76 tests passing. 312m params full scale,
5.9m training config.

phase 1 (language modeling): complete. all 4 gates pass. bpb ratio 0.84x
(todorov beats transformer by 16%). spike mi 1.275, cka 0.913, firing rate
42%. these numbers are 29x and 46x better than the predecessor project
(gerhard) -- the architecture was the bottleneck, not the spike mechanism.

phase 2 (context extension): complete. 2/3 gates pass. perplexity stable at
+4% over 16x context extrapolation. mla cache scales linearly. selective copy
0% is expected at 6m params (research confirms 130m+ needed) and is deferred
to phase 5.

8 kaggle runs completed. $0 of $500 budget spent.

## what comes next

phase 3: spatial module validation. enable gp self-interaction in swiglu.
test on 3d shape classification, n-body dynamics, equivariance at 60-degree
rotations. gates: outperform transformer on 2/3 spatial tasks, language bpb
not degraded >10% with gp enabled.

phase 5: scale to 300m parameters on 10-20b tokens. the supervisor mandated
a strict sequencing:
  - phase 5 baseline: 300m with current simple spikes (replicate phase 1)
  - phase 5a: swap ternarystrike for atmn (isolate neuron model impact)
  - phase 5b: expand to 132-point placement with winner from 5a (isolate placement)
  - sbds distillation only if bpb ratio degrades at 300m scale
three runs, not one. each produces an evidence-backed claim. never activate
multiple untested features simultaneously.

phase 4 (multimodal) is skipped at this scale. supervisor confirmed it would
repeat echoloc's failure.

## how to continue

this project uses eara (eptesicus autonomous research agent). the protocol
is in program.md at the project root. the project-specific config is in
eara.yaml.

to start a new experiment session:

1. read program.md -- it contains the full autonomous loop protocol
2. read eara.yaml -- it defines the training script, metric, compute backend,
   and gates
3. read state/program_status.yaml -- current phase, run history, gate results
4. read docs/STATUS_BOARD.md -- what's passing, what's failing, what's next
5. read docs/EXPERIMENT_LOG.md -- detailed analysis of every prior run
6. read the relevant knowledge files in knowledge/ for the phase you're working on

then follow the eara loop: modify train.py, pre-run checks, push to kaggle,
wait, pull results, analyze, keep or discard, repeat. never stop. never ask
for permission.

## critical files

notebooks/autoresearch/train.py -- the file you modify. 1090 lines.
self-contained kaggle training script with all model code inlined.
progressive training (256->512->1024->2048), fla chunk_kda integration,
spiking brain validation, passkey retrieval tests, perplexity scaling.

src/ -- the library code. do NOT modify during experiment loops. contains
the canonical implementations of kda, mamba3, mla, swiglu, ternary spike,
atmn spike, spiking brain, geometric algebra.

config.py -- model configuration dataclasses. tiny/small/base configs.

## things to know

kaggle api: the token must be set as KAGGLE_API_TOKEN env var BEFORE
importing the kaggle package. the package auto-authenticates on import.
always use acc="NvidiaTeslaT4" in kernels_push.

fla on t4: triton 3.3+ dropped sm_75 support. if fla fails to install,
the training script falls back to the o(t^2) matmul path. the hybrid
threshold is T >= 512 (matmul below, fla above).

state bug history: the matmul path state_approx previously only captured
the last timestep, making all chunked evaluation silently wrong for
context > 256. this was fixed in run_008 with full state accumulation.
never simplify recurrent state computation -- it is the memory, not
diagnostics.

fla at small scale: chunk_kda is designed for 1b+ params, head_dim 128+,
seq 2k+, and a100/h100. at our 6m param operating point, fla provides
~2x speedup at seq=2048 but is slower than matmul at seq=512 due to
kernel launch overhead.

spike health: mi > 0.1, cka > 0.3, firing rate 30-60%. the adaptive
threshold (alpha * mean(|x|)) with alpha=1.0 gives 42% firing rate at
initialization. this is validated.

## implemented but not yet validated

these are built and tested (unit tests pass) but never used in a training run:
- atmn spike neurons (membrane potential dynamics, per-neuron v_th = exp(a))
- 132-point expanded spike placement (spike_all_projections=True in config)
- sbds bidirectional distillation loss (0.2 fwd kl + 0.7 rev kl + 0.1 feature)

do not claim these as validated features. the phase 5 sequencing protocol
defines when and how to test each one in isolation.

## memory files

persistent context is stored in the claude memory system at
~/.claude/projects/C--Users-deyan-Projects-todorov/memory/. key files:
- user_profile.md -- deyan todorov, eptesicus labs
- feedback_code_rules.md -- zero comments, zero emojis, zero ai attribution
- feedback_autoresearch_loop.md -- mandatory loop protocol
- feedback_supervisor_phase2.md -- phase 5 sequencing directive, lessons learned
- project_todorov_phase0.md -- current project state and next steps
- reference_kaggle.md -- kaggle api pitfalls, fla on t4, working patterns

## run the loop

point your agent at this repo. say:

"read program.md and eara.yaml. check state/program_status.yaml for current
phase. read docs/STATUS_BOARD.md for gate status. begin the eara experiment
loop for the next phase. be fully autonomous."
