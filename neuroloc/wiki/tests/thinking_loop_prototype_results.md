# thinking loop prototype (recurrent hidden-state refinement)

status: historical context only. frozen as of 2026-04-16. do not edit.

last updated: 2026-04-14

## what this is

a cpu simulation that tests whether a minimal recurrent-refinement loop ("thinking")
on a small sequence model improves accuracy on an iterative-reasoning toy task, and
whether that improvement justifies adding a thinking mode to the architecture before
the first training run (run 1 baseline).

thinking here means: after a standard forward pass, feed the final hidden state back
into the model as an extra appended state-slot token and run the forward pass again.
repeat k times. emit the k-th pass output. this is intermediate hidden-state
refinement, not decoded chain-of-output-tokens. the recurrence happens inside a
single decision, not across multiple generated outputs.

## the simulation

`neuroloc/simulations/reasoning/thinking_loop_prototype.py`. pure pytorch cpu
implementation.

### task

modular arithmetic in `z_{mod 7}`. each sample is a length-n sequence of
`(operation, constant)` pairs encoded as one-hot tokens (vocab size 22: 1 reserved +
3 operations * 7 constants). the operation vocab has three kinds: add, sub, mul. the
accumulator starts at 0 and each token applies `(acc op const) mod 7` left to right.
the model must predict the final value `in {0..6}` after n tokens. one sample = one
sequence, trained with cross-entropy on the 7-class final accumulator.

### model

two layer blocks of pre-norm self-attention plus a two-layer mlp, residual on both,
layernorm. token embedding + sinusoidal-free learned position embedding. a shared
parameter `state_slot` is added when the refined hidden state is fed back as an
extra appended token with its own position id. `d_model=48`, depth `d=4`, head uses
a single attention pathway. ~78k parameters.

two operating modes:

- **no-thinking (k=1)**: one forward pass, emit. equivalent to a standard
  feedforward classifier.
- **thinking (k>=1)**: run one forward pass, take the final-token hidden
  representation, re-embed it as an additional token at a dedicated state slot with
  a shared learned bias, run the forward pass again. repeat k times.

both modes share the same architecture and parameter count. the only difference is
the recurrence loop at forward time. the model is trained with `k=4` (thinking mode)
and evaluated at `k in {1, 2, 4, 8, 16}`.

### baselines

a deeper feedforward variant with `depth=12` (~228k parameters, roughly 3x the
thinking model) is trained from scratch in the same regime. this is the
parameter-unmatched control: if thinking at `k=8` beats a 3x-deeper network, that
is the strongest possible case for thinking. if thinking does not beat it, the
result is negative.

### sweeps

- task depth `n in {2, 4, 8, 16, 32}`, trained on `n<=8` only, tested on all.
- thinking steps `k in {1, 2, 4, 8, 16}` at evaluation.
- 3 seeds per cell. `n_train=4096`, `n_val=1024`, `n_test=1024`. `15` epochs.
- 95 percent cis from a student-t on 3 seeds.

## the run that produced the metrics this article cites

source: `neuroloc/output/simulation_runs/reasoning/thinking_loop_prototype/thinking_loop_prototype_metrics.json`,
run id `thinking_loop_prototype_20260414_121149`. 3 seeds, wall-clock 242 seconds.
78199 parameters for thinking and shallow-feedforward models, 228343 for the deep
feedforward baseline.

## results: accuracy at k=1 vs k=4 (train-time setting)

chance for a 7-way classifier is 0.143.

- `n=2`: k=1 = 0.241; k=4 = 0.225 (no improvement, gain within noise)
- `n=4`: k=1 = 0.200; k=4 = 0.197 (no improvement)
- `n=8`: k=1 = 0.199; k=4 = 0.249 (+0.050 absolute, +25 percent relative, ci excludes zero for k=1 to k=2)
- `n=16`: k=1 = 0.171; k=4 = 0.150 (slight degradation)
- `n=32`: k=1 = 0.183; k=4 = 0.140 (degradation beyond training cap)

## results: per-step gain plateau

marginal accuracy gain from `k` to `2k`, averaged across 3 seeds. only gains with
their 95 percent ci excluding zero count as real.

- `n=8` k=1->k=2 gain = 0.0371 ci [0.0160, 0.0583] (real, positive)
- `n=8` k=2->k=4 gain = 0.0133 ci [-0.0100, 0.0367] (positive but noisy)
- `n=8` k=4->k=8 gain = 0.0003 ci [-0.0111, 0.0118] (flat)
- `n=8` k=8->k=16 gain = -0.0013 ci [-0.0050, 0.0024] (flat)

the only step-pair with a ci that excludes zero on the positive side is `k=1` to
`k=2` at `n=8`. all further steps are within noise. the gain plateau sits at `k<=4`
in every condition tested.

## results: fixed-point convergence

fraction of test samples whose final hidden-state delta (norm of change relative
to previous step) drops below `tol=1e-3` by `k=32`.

- `n=2`: 0.987
- `n=4`: 0.996
- `n=8`: 0.993
- `n=16`: 0.991
- `n=32`: 0.993

essentially every sample reaches a fixed point in the hidden state by the tested
horizon. the refinement loop is a contraction mapping under this parameterization.
this is a strong positive result for the "thinking is stable" claim and a neutral
result for the "thinking improves accuracy" claim: the fixed point is reached fast,
but the fixed point is not the right answer.

## results: generalization beyond training depth

trained on `n<=8`, tested on `n>8`. thinking evaluation at `k=16` (the largest
tested). deep feedforward baseline evaluated at its single forward pass. both at 3
seeds.

- `n=16`: thinking k=16 = 0.150 ci [0.137, 0.163]; deep feedforward = 0.201 ci [0.164, 0.237]; delta = -0.050.
- `n=32`: thinking k=16 = 0.137 ci [0.108, 0.167]; deep feedforward = 0.198 ci [0.176, 0.221]; delta = -0.061.

the parameter-inflated deep feedforward baseline beats the thinking model by 5 to 6
percentage points on out-of-distribution task depth. the thinking model does not
generalize better than just stacking more layers under matched-compute-per-step
assumptions.

## what these results say

four findings emerge from this cpu pilot. none of them is strong enough to justify
adding the recurrence as a run-1 requirement.

1. **thinking produces one step of real improvement, then plateaus.** the only
   step-pair where the 95 percent ci of the per-step gain excludes zero is
   `k=1` to `k=2` at `n=8`. beyond `k=2` the gains are within noise. this matches
   the expected behavior of a contraction loop whose stable point carries only
   one additional layer of compositional reasoning relative to the unrolled
   feedforward pass. it is not the "deep recursion into arbitrary task length"
   the architecture would need.

2. **the refinement loop reaches a fixed point on nearly every sample.** the
   trace of successive hidden states contracts to a stable representation within
   the first 4 to 8 iterations. thinking is a stable operation, not an
   oscillating one. this is neutral information: stability is cheap, but the
   fixed point itself is not discriminative enough to solve the task.

3. **thinking does not beat a deeper feedforward of matched compute-per-step
   budget.** at out-of-distribution task depth, a 3x deeper feedforward network
   trained with the same data outperforms `k=16` thinking by 5 to 6 percentage
   points. for the same forward-pass compute cost, more depth beats more
   iterations. this is the strongest finding against the proposal to promote
   thinking to a run-1 requirement.

4. **the generalization gap between thinking and feedforward grows with task
   depth.** at `n=16` the gap is -0.050, at `n=32` it is -0.061. the recurrence
   loop is not compensating for out-of-distribution depth; it is falling behind
   a deeper feedforward as the gap widens.

## limitations of this pilot

the pilot is a cpu prototype, not a full-scale validation. limitations are
significant enough to weaken the negative conclusion from first principles.

1. **the toy task is modular arithmetic with mul by 0, not natural-language
   reasoning.** the model sees `(op, const)` token pairs and must predict the
   scalar accumulator. this task has an adversarial structural feature: any
   `mul by 0` token collapses the accumulator to zero regardless of prior
   history. at `n=8`, the probability a `mul by 0` appears is roughly 32 percent,
   so the optimal policy is closer to "find the last mul-by-0 and process the
   suffix" than "integrate all tokens uniformly." natural-language reasoning has
   different structural properties and the thinking loop's suitability there is
   not tested by this pilot.

2. **the model size and training budget are tiny.** 78k parameters and 15 epochs
   on a 7-way classification task with 63 possible (op, const) pairs per position
   is not enough to reach the task's ceiling. every reported accuracy is close to
   chance (0.143). the signal-to-noise ratio is low. a cleaner pilot would require
   training to convergence on a larger model and report the ceiling behavior.

3. **the deep feedforward baseline is not matched-parameter.** the comparison is
   matched-compute-per-step (both do a single forward pass of 12-layer depth in
   the thinking k=1 case at the deep baseline), which is the correct comparison
   for a real-time inference budget. but it is not matched-parameter. a smaller
   deep baseline at the same parameter count as thinking would be a complementary
   comparison. at the current budget, the parameter-matched comparison (k=1
   shallow) is just the no-thinking case, which loses to k=4 thinking at n=8.

4. **only one architectural variant of thinking was tested.** the state is fed
   back as an extra appended token at a dedicated state slot. it is not fed back
   via gated residual injection on every token position, and it is not fed back
   via a cross-attention-style recurrence. other parameterizations may converge
   faster or produce larger plateaus.

5. **three seeds per cell is barely enough to exclude zero from the cis.** the
   cis are wide at small ns and the only statistically clean signal is the
   `k=1 -> k=2` gain at `n=8`. a cleaner replication would require at least 10
   seeds per cell.

## what this means for run 1

the pilot does not support promoting thinking from a phase-7 item to a run-1
requirement. the evidence from the pilot is:

- thinking gives one step of real improvement then plateaus. the plateau is at
  `k=4`.
- a deeper feedforward of 3x parameters beats `k=16` thinking at out-of-distribution
  task depth.
- the refinement loop is stable but stable at the wrong point for harder tasks.

the supporting claim ("thinking is a first-model requirement") is not falsified,
but it is not supported either. before run 1, a cheaper test with the following
properties is needed:

- model size large enough that the baseline is not at chance (try 500k params
  with 50 epochs on this task and watch the curve).
- a second toy task that is not adversarial in the same way (for example, a
  copy-or-reorder task instead of scalar accumulation).
- parameter-matched deep baseline (same parameter budget, all spent on depth).
- at least 10 seeds per cell so the cis are tight at `k=2` and beyond.

run 1 should proceed as the dense-baseline replication without a thinking loop.
if the run 1 baseline clears the passkey gate, thinking can re-enter the scope
as a phase-2 addition. if the run 1 baseline fails the passkey gate for
structural reasons that a thinking loop would plausibly address (such as not
enough effective depth of reasoning per token), the argument for a
cleaner-pilot-then-run-2-thinking reopens.

## sources

- `neuroloc/simulations/reasoning/thinking_loop_prototype.py` (the simulation
  script)
- `neuroloc/output/simulation_runs/reasoning/thinking_loop_prototype/thinking_loop_prototype_metrics.json`
  (the metrics for the 3-seed sweep across `n in {2,4,8,16,32}` and
  `k in {1,2,4,8,16}`)
- `neuroloc/output/simulation_runs/reasoning/thinking_loop_prototype/thinking_loop_prototype.png`
  (4-panel figure: accuracy-by-k-and-n, per-step gain, fixed-point fraction,
  thinking vs deep feedforward generalization)
- `tests/test_thinking_loop_prototype.py` (8 unit tests, all passing)

## update history

- **2026-04-14** — deyan todorov — file created. captures the cpu pilot result
  for the recurrent hidden-state refinement loop on a modular-arithmetic toy
  task. the pilot shows one step of real gain at `n=8`, a plateau by `k=4`, a
  refinement loop that reaches a fixed point on 99 percent of samples, and a
  generalization gap that favors a 3x deeper feedforward baseline at
  out-of-distribution task depth. the pilot does not support promoting thinking
  to a run-1 requirement; a cleaner pilot at larger model size and with a
  non-adversarial second task is recommended before reconsidering.
