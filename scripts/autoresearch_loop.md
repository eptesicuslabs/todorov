# Autoresearch Loop Protocol

This is the DEFINITIVE loop that Claude Code must follow for every
notebook iteration. No steps may be skipped. No pushes without all
checks passing. No exceptions.

## PRE-PUSH PHASE (before ANY push to Kaggle)

### Step 1: REPORT
Write what the notebook iteration will change, the hypothesis, and
expected outcome. Save to reports/ or print inline.

### Step 2: RESEARCH AGENT
Launch a web research agent to verify all technical assumptions:
- API shapes and signatures for any new library calls
- T4 GPU compatibility for any new operations
- Published benchmarks for expected performance
- Known bugs or issues with the approach

### Step 3: PLAN COMPLIANCE AGENT
Launch an agent that reads:
- The plan at C:\Users\deyan\.claude\plans\bubbly-nibbling-hanrahan.md
- docs/STATUS_BOARD.md
- state/program_status.yaml
- The CLAUDE.md project spec
And verifies: are we following the plan? Are we deviating from spec?

### Step 4: SELF-CRITIQUE AGENT
Launch an agent that reads the FULL train.py and checks:
- Shape mismatches (trace every tensor through the computation)
- Missing imports
- numpy bool serialization issues
- fla API compliance (gate shapes, log-space, layout, FP16/FP32)
- Memory budget at configured seq_len and batch_size
- Checkpoint loading across stages
- JSON serialization of results
Report ALL crash risks with line numbers.

### Step 5: CUSTOM SMOKE TEST AGENT
Launch an agent that:
- Reads the changes just made to train.py
- Writes a targeted smoke test exercising EXACTLY the changed code paths
- The test runs locally on CPU in under 30 seconds
- Verifies shapes, no crashes, correct output format
- Runs the test and reports pass/fail

### Step 6: FIX
Apply all fixes from agents in steps 2-5. If any agent found critical
issues, go back to step 2 and re-verify.

### Step 7: PUSH
Push to Kaggle. Do NOT queue multiple runs -- only ONE run at a time.
Print the Kaggle web URL for live monitoring.

## POST-COMPLETION PHASE (after Kaggle returns COMPLETE or ERROR)

### Step 8: PULL
Pull results via scripts/pull_kaggle_results.py. Parse log for errors.

### Step 9: Launch 4 PARALLEL AGENTS

#### 9a. ANALYSIS AGENT
Compare results to ALL prior runs. Check:
- fla usage (look for "fla available: True")
- Per-stage BPB progression
- s/step at each seq_len (detect speedup or regression)
- Spike stats (firing rate, MI, CKA)
- Selective copy and passkey accuracy
Write findings to docs/EXPERIMENT_LOG.md

#### 9b. RESEARCH GROUNDING AGENT
Compare results to published benchmarks at similar scale.
Verify unexpected results against literature.
Append findings to relevant knowledge/ files.

#### 9c. PLAN COMPLIANCE AGENT
Same as step 3. Check if results move us toward or away from goals.

#### 9d. DOCUMENTATION AGENT
Read ALL docs files and state files. Update EVERY file that is stale:
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
- state/gate_results.yaml
- reports/index.md
- Memory files at ~/.claude/projects/.../memory/
Cross-reference between files for consistency.

### Step 10: INVESTIGATION AGENTS (if ANY anomaly)
If results show unexpected behavior (slow speed, bad BPB, errors,
regressions, strange patterns):

#### 10a. WEB RESEARCH AGENT
Deep search for explanations, similar issues, known bugs, solutions.

#### 10b. CODE INVESTIGATION AGENT
Read training script, trace the issue through code, identify root
cause with specific line numbers and fix proposals.

### Step 11: DECIDE
Based on all agent findings, decide next action:
- Gates pass -> prepare completion report, move to next phase
- BPB needs improvement -> adjust hyperparameters
- fla issues -> diagnose and fix
- Crash -> analyze error and fix
- All good -> prepare next iteration

### Step 12: PREPARE NEXT RUN
Go back to Step 1 with the decided changes. The loop continues.

## RULES

- NEVER push without steps 1-6 completing successfully
- NEVER queue multiple Kaggle runs -- one at a time only
- NEVER skip the documentation agent
- ALWAYS run investigation agents on anomalies
- ALWAYS save partial results (stage_results files) for resilience
- ALWAYS use the hybrid approach: matmul for seq<512, fla for seq>=512
