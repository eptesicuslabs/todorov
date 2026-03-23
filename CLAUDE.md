# what this is

todorov is a proof that biological neural computation principles -- ternary
spikes, adaptive thresholds, recurrent state accumulation -- can outperform
standard transformers at matched scale, not despite the biological constraints
but because of them.

the architecture combines three sequence mechanisms (kda + mamba-3 + mla in
3:1 ratio) with ternary spike quantization ({-1, 0, +1}) on key/value paths.
the spikes are not a compression trick. they are the point. the question is
whether sparse, quantized, biologically-motivated activation patterns can
carry enough information for competitive language modeling, spatial reasoning,
and eventually multimodal processing.

phase 1 answered yes: ternary spikes preserve 29x more mutual information
than gerhard's predecessor architecture (mi 1.275 vs 0.044). the architecture
was the bottleneck, not the spike mechanism. todorov beats a same-size
transformer by 16% on byte-level language modeling.

phase 3 validated the geometric algebra module: g(3,0,1) projective geometric
product as self-interaction in swiglu adds zero overhead and provides genuine
spatial inductive bias (29% n-body improvement). gp as self-interaction works;
gp as neighbor interaction does not (echoloc thesis, now validated in
production architecture).

the open question is whether these advantages hold at 300m scale, and whether
atmn membrane-potential neurons (temporal state, per-neuron thresholds) offer
further improvement over the simpler ternary spikes. phase 5 tests this with
isolated ablations.

three prior eptesicus projects led here: gerhard (ternary spikes on rwkv,
proved spikes work but architecture bottlenecked them), echoloc (geometric
algebra, proved gp self-interaction but neighbor-interaction failed), and
chimera-edge (3:1 hybrid ratio, proved the layer mix). todorov unifies all
three.

# rules

zero comments in code. no inline, no block, no docstrings, no TODOs.
zero emojis. anywhere. code, docs, commits, prints, logs, filenames.
zero AI attribution. no Co-authored-by. no "written with AI." sole author: Deyan Todorov.
lowercase in all docs, commits, and readme. no tables in readme.

these rules apply to every file you create or modify. no exceptions.

# autoresearch loop

this project uses eara (program.md + eara.yaml). when running the experiment
loop, you MUST follow scripts/autoresearch_loop.md LITERALLY. the protocol
has 7 pre-push steps and 4 post-completion agents.

steps 2-5 (research, plan compliance, self-critique, smoke test) MUST each be
launched as SEPARATE SUBAGENTS using the Agent tool. you may not do them
inline. you may not combine them. the push gate at step 7 blocks any push
without all 4 subagent artifacts.

this is non-negotiable. the protocol exists because an agent skipped steps
2-4 in run_009 and pushed code with a shape mismatch bug and a confounded
experimental design. read the anti-rationalization rules in the loop protocol
before you convince yourself you can skip a step.

# phase sequencing

phases 0-3 are complete. phase 4 (multimodal) is skipped at this scale.

phase 5 sequencing is MANDATORY (supervisor directive):
1. phase 5 baseline: 300m params with current simple spikes (replicate phase 1)
2. phase 5a: swap ternarystrike for atmn (isolate neuron model impact)
3. phase 5b: expand to 132-point placement with winner from 5a (isolate placement)
4. sbds distillation only if bpb ratio degrades at 300m scale

three runs, not one. each produces an evidence-backed claim. NEVER activate
multiple untested features simultaneously. always distinguish "implemented"
from "validated" in reports.

implemented but not yet validated (unit tests pass, never trained):
- atmn spike neurons (membrane potential dynamics, per-neuron v_th = exp(a))
- 132-point expanded spike placement (spike_all_projections=True in config)
- sbds bidirectional distillation loss (0.2 fwd kl + 0.7 rev kl + 0.1 feature)

# architecture rules

never simplify recurrent state computation. it is the memory, not diagnostics.
the matmul path state_approx bug (run_003-007) was caused by taking only the
last timestep instead of accumulating full state. this made all chunked
evaluation silently wrong for context > 256.

gp self-interaction is additive after the swiglu down projection. it adds zero
computational overhead. it can be on by default. the "optional spatial module"
from chimera-edge is validated: gp as self-interaction works, gp as neighbor
interaction does not (echoloc thesis).

spike health thresholds: mi > 0.1, cka > 0.3, firing rate 30-60%. the
adaptive threshold (alpha * mean(|x|)) with alpha=1.0 gives ~42% firing rate.
stable across 9 runs. mi reached all-time high of 1.311 with gp active.

layer pattern: 3:1 ratio (75% kda, 25% mamba3/mla). this ratio converges
independently across kimi, qwen3, olmo, and systematic analysis papers.

# kaggle

kernel: dttdrv/todorov-autoresearch
auth: read ~/.kaggle/kaggle.json, set KAGGLE_API_TOKEN env var BEFORE
importing the kaggle package. the package auto-authenticates on import.
always use acc="NvidiaTeslaT4" in kernels_push.

fla on t4: triton 3.3+ dropped sm_75. pin torch==2.6.0 + triton==3.2.0.
t4 lacks bf16 cores: use fp16 for all fla kernels. use chunk_kda (not
chunk_gated_delta_rule). hybrid threshold: matmul for seq<512, fla for
seq>=512.

api quirks: kernels_status() 404s for ~60s after push on new kernels.
kernels_output() returns last COMPLETED run, not current. no streaming
log api. the mcp-server-kaggle-exec package is BROKEN -- use
scripts/kaggle_exec.py instead.

kaggle t4 is not viable for 300m params. phase 5 needs runpod or similar.

# critical files

notebooks/autoresearch/train.py -- the sole file modified during experiment
loops. ~1550 lines. self-contained for kaggle. all model code inlined.

scripts/autoresearch_loop.md -- the MANDATORY loop protocol. read this
before every experiment session.

program.md -- universal eara agent instructions.
eara.yaml -- project-specific config (metric, gates, compute backend).
state/program_status.yaml -- current phase, run history, gate results.
docs/STATUS_BOARD.md -- what is passing, what is failing, what is next.

src/ -- library code. do NOT modify during experiment loops. the library
is the source of truth. if train.py and src/ diverge, fix src/.

# bug history

run_003-007: matmul path state_approx only captured last timestep. fixed
in run_008 with full accumulation loop.

run_009: src/layers/swiglu.py spatial_mode added gp output (d_model) to
hidden state (hidden_dim). fixed in both train.py and library: gp residual
applied after down projection.

run_009: step count mismatch (gp=500, nogp=200, transformer=200) confounded
language degradation measurement. the -18.8% "improvement" is training
budget, not gp. fix: always train comparison models at matched step counts.

run_009: equivariance gate tested the cayley table algebra, not the trained
model. this gate is vacuous (always passes at float32 precision). fix: test
full swiglu pathway equivariance, not raw multivector algebra.

run_009: shape classification p~0.07 on 200 trials. not statistically
significant. fix: increase to 1000 trials for p<0.01.

# results summary

phase 1: bpb 0.84x (16% better than transformer). mi 1.275 (29x gerhard).
cka 0.913 (46x gerhard). all 4 gates pass.

phase 2: perplexity +4% over 16x extrapolation (pass). mla cache linear
(pass). selective copy 0% (deferred, needs 130m+). fla o(t) scaling
confirmed (1.94-1.97x per 2x length).

phase 3: n-body 29% mae improvement (real). shape classification 30% vs
25% (not yet real, p~0.07). equivariance 1.34e-07 (vacuous gate). language
bpb not degraded (confounded by step count mismatch). gp overhead: zero.
spike mi all-time high (1.311).

9 kaggle runs. $0 of $500 budget. ~13 gpu hours total.

# keeping this file current

this file MUST be updated after every completed run. the documentation agent
(step 9d of the autoresearch loop) is responsible for updating CLAUDE.md
along with all other docs and state files.

what to update after each run:
- results summary: add new phase results, update run count and budget
- bug history: add any new bugs found, with root cause and fix
- phase sequencing: advance completed phases, update "what comes next"
- architecture rules: add any new rules learned from the run
- kaggle section: update if compute backend changes for phase 5

if CLAUDE.md and the actual project state ever disagree, the project state
(state/program_status.yaml, docs/STATUS_BOARD.md) is authoritative. update
CLAUDE.md to match, not the other way around.

# read order for new sessions

1. this file (CLAUDE.md)
2. state/program_status.yaml
3. docs/STATUS_BOARD.md
4. eara.yaml
5. program.md
6. scripts/autoresearch_loop.md
7. knowledge/ files relevant to current phase
