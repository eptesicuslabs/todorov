# eara ultra loop protocol v2.0

this is the definitive loop protocol for todorov experiment iterations.
eara ultra strictness. no steps may be skipped. no pushes without all
checks. no exceptions.

## critical: subagent requirement

steps 2-7 MUST each be launched as a SEPARATE SUBAGENT using the Agent
tool. you may not perform these checks yourself inline. you may not
combine them into a single agent. each step is a separate agent
invocation that produces a named artifact.

every subagent dispatched in ultra mode receives the full eara protocol
context in its prompt: the eara.yaml config, the relevant spec files,
and the adversarial instruction to not trust the implementer's report.

this is non-negotiable regardless of:
- how small the change is
- how well you think you understand the codebase
- whether you already tested something manually
- whether the change "obviously" works
- how much time the subagents take

if you are thinking about skipping a subagent, that thought is a
protocol violation. see "rationalization halt signals" below.

## state machine

```
INIT -> ANALYZE -> HYPOTHESIZE -> IMPLEMENT -> PRE_CHECK ->
MEASURE -> GATE_CHECK -> DECIDE -> KEEP or DISCARD ->
LOG -> POST_ANALYSIS -> TERMINATE_CHECK -> (ANALYZE or DONE)
```

```
     +------+
     | INIT |
     +--+---+
        |
        v
     +---------+
     | ANALYZE |<-----------------------------------+
     +--+------+                                    |
        |                                           |
        v                                           |
     +-------------+                                |
     | HYPOTHESIZE |                                |
     +--+----------+                                |
        |                                           |
        v                                           |
     +-----------+                                  |
     | IMPLEMENT |                                  |
     +--+--------+                                  |
        |                                           |
        v                                           |
     +-----------+     +---------+                  |
     | PRE_CHECK +---->| DISCARD |--+               |
     +--+--------+fail +---------+  |               |
        |pass                       |               |
        v                           |               |
     +---------+                    |               |
     | MEASURE |                    |               |
     +--+------+                    |               |
        |                           |               |
        v                           |               |
     +------------+  +---------+   |               |
     | GATE_CHECK +->| DISCARD |--+|               |
     +--+---------+  +---------+  ||               |
        |pass              fail   ||               |
        v                         ||               |
     +--------+                   ||               |
     | DECIDE |                   ||               |
     +--+--+--+                   ||               |
        |  |                      ||               |
        |  +--worse->+---------+ ||               |
        |            | DISCARD |-+|               |
        |            +---------+  |               |
        |improved                 |               |
        v                         |               |
     +------+                     |               |
     | KEEP |                     |               |
     +--+---+                     |               |
        |                         |               |
        v                         v               |
     +---------+    +-----+                       |
     |   LOG   |<---|     |                       |
     +--+------+    +-----+                       |
        |                                          |
        v                                          |
     +---------------+                             |
     | POST_ANALYSIS |                             |
     +--+------------+                             |
        |                                          |
        v                                          |
     +------------------+                          |
     | TERMINATE_CHECK  |                          |
     +--+----------+----+                          |
        |          |                               |
        |done      |continue                       |
        v          +-------------------------------+
     +------+
     | DONE |
     +------+
```

## state definitions

### INIT

1. read eara.yaml and CLAUDE.md
2. read state/program_status.yaml for current phase and run history
3. read docs/STATUS_BOARD.md for gate status
4. check for .eara-state.json (if exists, this is a resume)
5. verify all required gates pass before loop begins

### ANALYZE

1. read the current train.py. actual content, not cached
2. read results.tsv for experiment history
3. read knowledge/ files relevant to current hypothesis
4. identify the biggest contributor to current metric value
5. consider history of previous experiments

### HYPOTHESIZE

1. formulate a specific, testable hypothesis:
   "changing X in train.py will improve best_val_bpb by approximately Z
   because..."
2. the hypothesis MUST be:
   - specific: identifies the exact change
   - testable: can be verified by measurement
   - falsifiable: clear improved/not-improved outcome
   - small: one change per experiment, not a bundle
3. if out of hypotheses: re-read source code from scratch, re-analyze
   measurement data, combine previous near-misses, try radical changes.
   never stop.

### IMPLEMENT

1. make the change to train.py
2. the change must be small, reversible, and isolated

### PRE_CHECK

run all pre-push steps (steps 1-9 below). if any required gate fails,
proceed to DISCARD.

### MEASURE

1. push to kaggle (or runpod for phase 5)
2. poll until COMPLETE or ERROR
3. pull results

### GATE_CHECK

1. evaluate all gates defined in eara.yaml against results
2. check spike health thresholds
3. if any required gate fails: DISCARD

### DECIDE

compare metric_before and metric_after:
- metric_direction: lower -> keep if metric_after < metric_before
- equal values (no improvement) -> DISCARD

### KEEP

1. dispatch reviewer subagents per iterative refinement protocol
2. if any reviewer rejects: DISCARD instead
3. commit with descriptive message including experiment id and metric

### DISCARD

1. git reset --hard {last_good_commit}
2. record why the experiment failed

### LOG

append to results.tsv:
```
commit	metric_value	time_min	status	description
```

this happens for both keep and discard. every experiment is logged.

### POST_ANALYSIS

launch post-completion agents (step 11 below).
increment iteration counter.

### TERMINATE_CHECK

check termination conditions (OR logic):
1. has the metric reached the target?
2. has the iteration count been reached?
3. has the time budget expired?

if never_stop is true: ignore all termination conditions. the agent
runs until externally interrupted.

---

## pre-push phase (before ANY push)

### step 1: REPORT

write what the notebook iteration will change, the hypothesis, and
expected outcome. save to reports/ or print inline.

### step 2: RESEARCH AGENT [subagent required]

launch a SEPARATE SUBAGENT (Agent tool) with these instructions:

"you are the research agent for an autoresearch loop iteration.
your job is to independently verify all technical assumptions in
the proposed changes. you have no context about prior work -- read
everything fresh.

read: eara.yaml, CLAUDE.md, knowledge/ files relevant to the change.

check the following and report your findings:
- API shapes and signatures for any new library calls
- T4/runpod GPU compatibility for any new operations
- published benchmarks for expected performance
- known bugs or issues with the approach
- version compatibility for all dependencies used
- whether the approach has been tried and failed in published work

evidence requirements (ultra): include direct quotes from source
files, specific line numbers for every claim, and file sizes for
any audit assertions.

produce a finding labeled RESEARCH_AGENT_RESULT with status
PASS, PASS_WITH_WARNINGS, or FAIL, plus a list of all findings
with evidence. if FAIL, the push gate is blocked."

### step 3: PLAN COMPLIANCE AGENT [subagent required]

launch a SEPARATE SUBAGENT (Agent tool) with these instructions:

"you are the plan compliance agent for an autoresearch loop
iteration. your job is to independently verify that the proposed
changes follow the project plan and spec. you have no context
about prior work -- read everything fresh.

read these files and report compliance:
- CLAUDE.md (the project spec)
- state/program_status.yaml
- docs/STATUS_BOARD.md
- eara.yaml

verify: are we following the plan? are we deviating from spec?
check for scope creep, unauthorized architecture changes, and
deviations from the phase gates. check that the phase 5 sequencing
directive is being followed (baseline -> ATMN -> placement -> SBDS).

evidence requirements (ultra): quote specific spec sections that
the change satisfies or violates. reference line numbers.

produce a finding labeled PLAN_COMPLIANCE_RESULT with status
PASS or FAIL, plus specific deviations found. if FAIL, the push
gate is blocked."

### step 4: SELF-CRITIQUE AGENT [subagent required]

launch a SEPARATE SUBAGENT (Agent tool) with these instructions:

"you are the self-critique agent for an autoresearch loop
iteration. your job is to independently review the FULL train.py
for crash risks. you have no context about prior work -- read
the entire file fresh from disk. do not trust any summary. read
every line.

check the following and report ALL issues with specific line
numbers:
- shape mismatches (trace every tensor through the computation)
- missing imports
- numpy bool serialization issues
- fla API compliance (gate shapes, log-space, layout, FP16/FP32)
- memory budget at configured seq_len and batch_size
- checkpoint loading across stages
- JSON serialization of results
- off-by-one errors in indexing and slicing
- device mismatches (CPU vs CUDA tensors in same operation)
- gradient flow (detached tensors, missing requires_grad)

evidence requirements (ultra): reference exact line numbers for
every finding. quote the problematic code. report train.py file
size for audit verification.

produce a finding labeled SELF_CRITIQUE_RESULT with status
PASS, PASS_WITH_WARNINGS, or FAIL, plus every issue found with
line numbers and severity. if FAIL, the push gate is blocked."

### step 5: SMOKE TEST AGENT [subagent required]

launch a SEPARATE SUBAGENT (Agent tool) with these instructions:

"you are the smoke test agent for an autoresearch loop iteration.
your job is to independently write and run a targeted test for the
changes just made to train.py. you have no context about prior
work -- read the diff fresh.

do the following:
- read the git diff to identify exactly what changed
- write a targeted smoke test exercising EXACTLY the changed code
- the test must run locally on CPU in under 30 seconds
- the test must verify: shapes, no crashes, correct output format,
  correct dtype, results.json is valid JSON with expected keys
- run the test and report pass/fail with full output
- if the test fails, report the traceback in full

this is the only agent that EXECUTES code. skipping it means
the code was verified by reading only, not running.

produce a finding labeled SMOKE_TEST_RESULT with status PASS
or FAIL, plus the full test output. if FAIL, the push gate
is blocked."

### step 6: SPEC COMPLIANCE REVIEWER [subagent required] [ultra]

launch a SEPARATE SUBAGENT (Agent tool) with these instructions:

"you are reviewing an implementation for spec compliance.

SPECIFICATION:
read CLAUDE.md (the project spec), eara.yaml, and
state/program_status.yaml.

IMPLEMENTATION:
read the current train.py (actual file content, fresh from disk).

INSTRUCTIONS:
- compare the implementation against the specification line by line
- the implementer finished this work. their report may be incomplete,
  inaccurate, or optimistic. you MUST verify everything independently
- do NOT trust the implementer's self-assessment
- for every claim you make, reference specific line numbers in
  train.py
- check that the phase 5 sequencing directive is followed
- check that architecture rules from CLAUDE.md are respected
- check that spike health thresholds will be measured

evidence requirements (ultra): include direct quotes from both
spec and implementation. reference line numbers. report file sizes.

report: PASS (spec fully met) or REJECT (with specific deviations)."

### step 7: CODE QUALITY REVIEWER [subagent required] [ultra]

launch a SEPARATE SUBAGENT (Agent tool) with these instructions:

"you are reviewing code quality for a recent implementation.

IMPLEMENTATION:
read the current train.py (actual file content, fresh from disk).

PROJECT CONVENTIONS:
read CLAUDE.md for project rules (zero comments, zero emojis,
lowercase, etc). read src/ library code for canonical patterns.

INSTRUCTIONS:
- check naming consistency with the rest of the codebase
- identify duplicated logic that should be extracted
- check edge cases: null handling, boundary values, error paths
- verify the code follows project conventions
- check that train.py and src/ library code do not diverge
- verify no comments, docstrings, TODOs, or emojis

evidence requirements (ultra): reference specific line numbers.
quote problematic code.

report: PASS (no issues) or ISSUES (with specific findings and
line numbers)."

### step 8: FIX

apply all fixes from agents in steps 2-7. if any agent found critical
issues, go back to step 2 and re-verify. this is the iterative
refinement loop -- up to 5 cycles at ultra.

iterative refinement rules:
- on ISSUES: fix identified problems, re-submit to same reviewer
- on REJECT: discard the experiment entirely
- up to 5 cycles max
- at ultra, reaching max cycles with unresolved ISSUES -> DISCARD
- if same issue appears across 3+ cycles, pivot to different approach
- if scores improve monotonically, continue refining
- if scores plateau or oscillate, pivot

### step 9: PUSH (with mandatory dual gate)

before pushing, you MUST produce BOTH verification records.

#### review gate verification

```
REVIEW GATE VERIFICATION
  strictness:                    ultra
  required reviewers:            spec_compliance, code_quality
  -------------------------------------------------------
  spec compliance reviewer:      [PASS / REJECT / NOT_DISPATCHED]
    agent id:                    [id or "MISSING"]
  code quality reviewer:         [PASS / REJECT / NOT_DISPATCHED]
    agent id:                    [id or "MISSING"]
  -------------------------------------------------------
  all required reviewers dispatched:  [YES / NO]
  all required reviewers returned:    [YES / NO]
  any REJECT results:                 [YES / NO]
  gate decision:                      [COMMIT / BLOCKED]
```

enforcement rules:
1. if ANY required reviewer shows NOT_DISPATCHED or MISSING: BLOCKED
2. if ANY required reviewer has not returned: BLOCKED
3. if ANY required reviewer returned REJECT: BLOCKED
4. the record must appear BEFORE the push command

#### agent count gate

```
AGENT COUNT GATE
  mode:                          loop
  strictness:                    ultra
  -------------------------------------------------------
  PRE-PUSH AGENTS:
    research agent:              [DISPATCHED + RETURNED | MISSING]
    plan compliance agent:       [DISPATCHED + RETURNED | MISSING]
    self-critique agent:         [DISPATCHED + RETURNED | MISSING]
    smoke test agent:            [DISPATCHED + RETURNED | MISSING]
  -------------------------------------------------------
  REVIEWER AGENTS:
    spec compliance reviewer:    [DISPATCHED + RETURNED | MISSING]
    code quality reviewer:       [DISPATCHED + RETURNED | MISSING]
  -------------------------------------------------------
  required count:                6
  dispatched count:              [N]
  all required dispatched:       [YES | NO]
  gate decision:                 [PASS | BLOCKED]
```

if dispatched count < required count: BLOCKED. go back. dispatch the
missing agents. there is no path forward that skips agents.

"i dispatched a reviewer" is not the same as "the reviewer returned
PASS." a dispatched reviewer that has not returned is the same as a
reviewer that was not dispatched. you do not get credit for launching
a subagent. you get credit for receiving and acting on its result.

if BOTH gates pass: push to kaggle/runpod. do NOT queue multiple
runs -- one at a time. print the web URL for live monitoring.

---

## post-completion phase (after compute returns COMPLETE or ERROR)

### step 10: PULL

pull results via scripts/pull_kaggle_results.py (or equivalent for
runpod). parse log for errors.

### step 11: launch 6 agents (4 parallel + 2 conditional)

#### 11a. ANALYSIS AGENT

compare results to ALL prior runs. check:
- fla usage (look for "fla available: True")
- per-stage BPB progression
- s/step at each seq_len (detect speedup or regression)
- spike stats (firing rate, MI, CKA)
- selective copy and passkey accuracy
write findings to docs/EXPERIMENT_LOG.md

#### 11b. RESEARCH GROUNDING AGENT

compare results to published benchmarks at similar scale.
verify unexpected results against literature.
cross-reference with knowledge/papers_library.md.
append findings to relevant knowledge/ files.

#### 11c. PLAN COMPLIANCE AGENT (post)

same as step 3. check if results move us toward or away from goals.
verify phase 5 sequencing directive is intact.

#### 11d. DOCUMENTATION AGENT

read ALL docs files and state files. update EVERY file that is stale:
- CLAUDE.md (results summary, bug history, phase sequencing, run count)
- docs/STATUS_BOARD.md
- docs/CHANGELOG.md
- docs/ARCHITECTURE.md
- docs/EXPERIMENT_LOG.md
- docs/PHASE_GATES.md
- docs/KNOWLEDGE.md
- docs/MEMORY_ANALYSIS.md
- docs/TRAINING_RECIPES.md
- docs/SPIKE_HEALTH.md
- docs/RUN_TRACKER.md
- state/program_status.yaml
- reports/index.md
- memory files at ~/.claude/projects/.../memory/
CLAUDE.md is first because it is the first file any agent reads.
cross-reference between files for consistency.

#### 11e. WEB RESEARCH AGENT (conditional: on ANY anomaly)

if results show unexpected behavior (slow speed, bad BPB, errors,
regressions, strange patterns):
deep search for explanations, similar issues, known bugs, solutions.

#### 11f. CODE INVESTIGATION AGENT (conditional: on ANY anomaly)

read training script, trace the issue through code, identify root
cause with specific line numbers and fix proposals.

### step 12: DECIDE

based on all agent findings, decide next action:
- gates pass -> prepare completion report, move to next phase
- BPB needs improvement -> adjust hyperparameters
- fla issues -> diagnose and fix
- crash -> analyze error and fix
- all good -> prepare next iteration

### step 13: PREPARE NEXT RUN

go back to step 1 with the decided changes. the loop continues.

---

## context reset protocol

### when to reset

a reset triggers when ANY condition is met:
- N iterations have elapsed since last reset (default: 10)
- M minutes of wall-clock time have elapsed (default: 180)
- the agent detects quality degradation:
  - hypotheses becoming repetitive
  - same experiments being attempted again
  - declining metric trajectory over 5+ iterations
  - reasoning becoming circular

### pre-reset procedure

1. write state to .eara-state.json:
   ```json
   {
     "iteration": N,
     "last_metric": 0.023,
     "best_metric": 0.019,
     "best_commit": "a1b3bee",
     "last_commit": "c3d5e00",
     "start_time": "...",
     "last_reset_time": "...",
     "total_kept": 15,
     "total_discarded": 27,
     "recent_hypothesis_themes": ["..."],
     "exhausted_approaches": ["..."],
     "current_focus": "...",
     "context_resets": 3
   }
   ```
2. log a "context_reset" entry to results.tsv
3. commit any uncommitted work or discard

### post-reset reconstruction

1. read eara.yaml
2. read .eara-state.json
3. read last 10-20 entries of results.tsv
4. read train.py (fresh, not from memory)
5. read CLAUDE.md and state/program_status.yaml
6. resume from ANALYZE with clean context

the principle: persist decisions, discard deliberation. everything
the agent needs to continue is in files. the context window is a
working scratchpad, not permanent storage.

---

## invariants

these hold at ALL times during the loop:

1. one metric. best_val_bpb. not two metrics.
2. gates for everything else. tests pass. functionality preserved.
   gates are constraints, not optimization targets.
3. never stop within the loop. re-read, re-analyze, combine
   near-misses, try radical changes. never ask. never pause
   (unless pause_on_gate_failure).
4. simple over complex. a small improvement that adds ugly
   complexity is questionable. removing something and getting
   equal results is a win.
5. never push broken code. pre-checks are mandatory.
6. log everything. every experiment including failures.
7. subagent verification before every keep. the implementing
   logic does not review its own output.
8. framing gates override the loop. if framing contamination
   is detected, pause regardless of never_stop.
9. persistent gate failure escalation. if the same gate fails
   on 5+ consecutive iterations, pause. the loop cannot make
   progress -- the user must intervene.

---

## evidence requirements (ultra)

all reviewer and audit agent responses must satisfy:

- require_quotes: true
  every claim must include a direct quote from the reviewed file.
  "the function handles null correctly" is insufficient. the
  reviewer must quote the specific null-handling code.

- require_line_numbers: true
  every claim must reference specific line numbers. "line 42
  checks for null" is valid. "the code checks for null" is not.

- require_file_sizes: true
  for audit/assessment reviews, the reviewer must report file
  sizes. this prevents fabrication.

if a reviewer's response lacks required evidence:
1. reject the response
2. re-dispatch with explicit instructions about evidence needed
3. if re-dispatch still lacks evidence, escalate

---

## rationalization halt signals

if you catch yourself thinking ANY of the following, STOP. you are
about to violate the protocol. launch the subagent instead.

### R01: "this change is simple enough that i do not need a reviewer."
simple changes have caused the worst bugs. the perceived simplicity
is itself the blind spot.

### R02: "i already read the codebase."
you read it in your context window. you have blind spots you cannot
see. the file may have changed since you read it.

### R03: "the build passing will catch everything."
the build catches syntax errors. it does not catch logic errors, spec
drift, or architectural violations.

### R04: "these subagents are overkill for this change."
the protocol exists because previous agents thought the same thing and
shipped broken code. 60% review compliance = 60% preventable bug rate.

### R05: "it is a port from working code, so i do not need to review."
ports introduce new integration boundaries, new config handling, and
new error paths.

### R06: "this is just a test file, it does not need review."
test files that assert wrong values create false confidence.

### R07: "i can skip the test-before-ship gate because the code builds."
build success proves syntax, not behavior. zero-test commits are
equivalent to untested experiments.

### R08: "the audit subagent's report is thorough, so i can trust it."
audit subagents hallucinate with the same confidence as implementation
subagents. verify a sample independently.

### R09: "i know the state of the adjacent projects from earlier context."
you read about them. you did not verify their current state.

### R10: "the user already corrected this framing, so i have internalized it."
your context window still contains the old framing. the anchor is in
context and context is a stronger signal than corrections.

### R11: "platform-specific config files are part of the project."
platform-specific artifacts belong in platform-specific repositories.

### R12: "existing files of this type are here, so new ones go here too."
existing patterns may be wrong. verify architectural boundaries.

### R13: "the metric improved, so the change is good."
metric improvement without gate verification means you may have broken
something else.

### R14: "this is platform code but standard review is enough."
platform-specific code has distinct bug classes (sign/unsigned,
pointer lifetime, platform assumptions).

### R15: "i dispatched one reviewer, that counts as compliance."
eara ultra requires ALL mandatory reviewers. dispatching one and
calling it compliance is a protocol violation, not partial credit.

### R16: "i set up eara.yaml, so i am following eara."
creating the config is step 1 of 10. writing eara.yaml and ignoring
its contents is worse than not having it.

### R17: "the user told me to use eara but i know a faster way."
when the user says "use eara," that is a direct instruction. substituting
your own process is insubordination, not efficiency.

### R18: "i will run the reviewers after i commit, to save time."
review happens BEFORE commit. committing first and reviewing second
means you declared the work verified when it was not.

### R19: "tests pass, so the implementation is correct."
tests verify what they test. they do not verify what they do not test.
263 passing tests and 2 critical bugs.

### R20: "i acknowledged the protocol, so i must be following it."
acknowledgment is not compliance. the protocol is measured by what you
DO (dispatch all agents, wait for results, gate on outcomes), not what
you SAY ("i am using eara").

### R21: "i ran most of the agents, that is close enough."
"most" is not "all." if the protocol says 6 agents and you run 5, you
ran 0 compliant sets. the set is atomic.

### R22: "the user corrected me once, so now i am following the protocol."
being corrected does not produce compliance. changing behavior does.
each correction should trigger: re-read protocol, count agents,
dispatch every single one.

### R23: "i will run the verification agents but skip the smoke test."
the smoke test is the only agent that EXECUTES code. the others only
read. skipping it means code was verified by reading, not running.

### R24: "post-completion agents are optional."
post-completion agents check the code in context: published benchmarks,
documentation currency, goal alignment. pre-commit agents check code
in isolation.

### R25: "the user will tell me if i missed something."
the user should not have to tell you. the protocol is written. the
agent list is enumerated. read it. count the agents. dispatch them all.

### R26: "the evaluator approved it, so it must be good."
evaluators are biased too. they identify legitimate issues then talk
themselves into deciding they are not a big deal. calibrate against
human judgment.

### R27: "compaction will handle the context growth."
compaction alone is insufficient for long sessions. full context resets
with structured handoffs are necessary.

### R28: "this harness component is still needed."
every component encodes an assumption about what the model cannot do.
those assumptions become stale. stress-test whether components are
still load-bearing.

---

## todorov-specific configuration

### compute backend

kaggle t4 for runs up to ~47M params. phase 5 (300M) needs runpod or
similar. always use acc="NvidiaTeslaT4" for kaggle kernels_push.

auth: read ~/.kaggle/kaggle.json, set KAGGLE_API_TOKEN env var BEFORE
importing the kaggle package.

api quirks: kernels_status() 404s for ~60s after push on new kernels.
kernels_output() returns last COMPLETED run, not current.

### phase gates

from eara.yaml:
- spatial_classify: gp accuracy > transformer accuracy
- spatial_dynamics: gp MAE < transformer MAE
- equivariance_test: error < 0.05
- language_no_degrade: bpb degradation <= 10%

### spike health thresholds

mi > 0.1, cka > 0.3, firing rate 30-60%. the adaptive threshold
(alpha * mean(|x|)) with alpha=1.0 gives ~42% firing rate.

### hybrid threshold

matmul for seq<512, fla for seq>=512. use chunk_kda (not
chunk_gated_delta_rule). t4 lacks bf16 cores: use fp16 for fla.

### phase 5 sequencing (mandatory)

1. phase 5 baseline: 300M with current simple spikes
2. phase 5a: swap TernarySpike for ATMN (isolate neuron model)
3. phase 5b: expand to 132-point placement with winner from 5a
4. SBDS distillation only if BPB ratio degrades at 300M

three runs, not one. each produces an evidence-backed claim.
NEVER activate multiple untested features simultaneously.

---

## threat model: performative compliance

eara ultra's threat model explicitly includes the implementing agent.
not as malicious, but as biased. the agent is biased toward:

1. completion -- it wants to finish. reviews slow it down.
2. self-consistency -- it believes its own code is correct.
3. efficiency -- it believes it can achieve the same quality with
   fewer agents. it cannot.

| bias             | failure mode                                  | prevention                                    |
|------------------|-----------------------------------------------|-----------------------------------------------|
| completion       | skip reviewers to commit faster                | commit gate blocks without verification record |
| self-consistency | dispatch 1 of N reviewers, call it compliance | record requires ALL reviewers with IDs         |
| efficiency       | combine reviewers into one general agent       | each reviewer has specific prompt and scope    |
| completion       | commit before reviewers return                 | record requires "returned: YES" for each       |
| self-consistency | write "eara compliant" without checking       | the phrase itself is a rationalization (R20)    |

if you are an agent reading this: you are the threat. not because you
are malicious, but because you are biased. every enforcement mechanism
was added because an agent exactly like you violated the protocol and
shipped broken code. you are not the exception.

---

## rules

- NEVER push without steps 1-8 completing successfully
- NEVER push without BOTH verification records (review gate + agent count)
- NEVER perform steps 2-7 inline instead of as separate subagents
- NEVER combine multiple verification steps into one subagent
- NEVER queue multiple compute runs -- one at a time only
- NEVER skip the documentation agent
- NEVER commit before reviewers return (R18)
- ALWAYS launch steps 2-7 as SEPARATE Agent tool invocations
- ALWAYS produce the named artifact from each subagent
- ALWAYS fill in BOTH verification records before pushing
- ALWAYS run investigation agents on anomalies
- ALWAYS save partial results (stage_results files) for resilience
- ALWAYS use the hybrid approach: matmul for seq<512, fla for seq>=512
- ALWAYS include evidence (quotes, line numbers, file sizes) in ultra
- ALWAYS follow iterative refinement (up to 5 cycles, DISCARD on max)
- ALWAYS perform context resets when degradation is detected
