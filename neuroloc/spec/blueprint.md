# neural machine specification

see also: `neuroloc/wiki/PROJECT_PLAN.md` is the canonical persistent project state. that file holds the current run, current question, hypothesis under test, decision rules, prior runs index, open research questions, and project glossary. this blueprint covers the design intent and feature definitions; PROJECT_PLAN.md covers what is being run right now and what we have learned. update both when the project state changes.

## context

todorov proved that biological computation math outperforms standard
approaches at matched scale (33.7% better at 267M parameters). neuroloc
mapped 61 biological mechanisms, validated 5 bridge interventions, and
identified which principles are genuine vs superficial. this spec defines
what comes next: a unified neural machine that implements the validated
math as one operation, not a collection of named modules.

this is not a language model. it is a general-purpose neural computation
engine that recognizes, remembers, predicts, and imagines.

## design decisions (from user)

- input: mixed continuous signals (text, image, sensor, any modality)
- output: structured responses in the same signal space
- two operating modes: on-demand (process, respond, stop) and persistent
  (always-on with background processing between active tasks)
- one core operation with mode signals (not separate modules)
- prediction-error-driven processing (only process what surprises you)
- eidetic memory (everything stored, older things fade extremely slowly)
- novel compression to make eidetic storage feasible
- self-directed imagination (the machine decides when to simulate)
- automatic fast/slow path switching (machine detects its own uncertainty)
- first milestone: recognition + memory + prediction unified
- learning method: open research question (discover, don't prescribe)
- compression depth: open research question (find the best way)
- self-monitoring: separate inspectable system (medical telemetry)
- name: unnamed until it works

## the core operation

one mathematical operation, repeated. each step:

    z = Q(R(B(C(x), C(h))))

where:
- x is the current input signal
- h is the accumulated memory state
- C is compression: select the important parts, discard the rest.
  implemented as competitive selection (top-k by magnitude) producing
  sparse rate-coded values (survivors keep their continuous magnitude,
  the rest are zero). this is NOT ternary -- values are continuous.
- B is mixing: combine the compressed input with compressed memory.
  outer product (creates associations), element-wise product (gates),
  or structured bilinear map (geometric relationships)
- R is rotation: encode position, phase, and temporal order through
  rotation in the signal space
- Q is output shaping: compress the result back to a transmittable form

the same equation runs at every step. what changes is the MODE SIGNAL
that shifts which variant of each operator is active:

- **remember mode**: C selects what to store, B writes it to memory
  (outer product), R encodes when it happened, Q shapes the memory
  update. memory state h is updated.
- **retrieve mode**: C compresses the query, B matches it against
  memory (dot product retrieval), R aligns temporal position, Q
  shapes the retrieved content.
- **imagine mode**: C suppresses external input (gate closed), B
  recombines stored fragments (outer product on memory contents),
  R generates novel temporal sequences, Q evaluates plausibility.
- **predict mode**: C processes input normally, B compares input
  against prediction from previous step, R maintains sequence
  position, Q produces the prediction error (what was surprising).

the mode signal is not a hard switch. it is a continuous blend
determined by the machine's own state (how much external input vs
how much internal processing is needed at this moment).

## memory architecture

eidetic memory with slow fade:

the memory state h is a matrix per processing unit (like the current
outer-product state). UNLIKE current design:

- decay rate is near-zero for important memories (alpha close to 1.0)
- decay rate is activity-dependent (BCM-like: busy memories fade
  faster to prevent saturation, quiet memories persist). validated
  by pilot: gamma=0.3-0.5 stabilizes without degrading retrieval.
- targeted erasure: when writing a new value for an existing key,
  ERASE the old value first, then write the new one. this is the
  actual delta rule, not just decaying accumulation.
- periodic consolidation: every N steps, compress the memory state
  into a summary snapshot and store it permanently. the fast state
  can then be partially cleared. old snapshots are addressable by
  the retrieval operation.

compression for eidetic storage:
- hierarchical ternary: top-k selection + ternary values = 0.37 bits/dim
- consolidated snapshots: rank-r SVD every 64 steps = 64x more compact
- content-addressable: hash-based deduplication, store by concept not token
- further compression methods are an open research question

## imagination

the machine can run internal simulations on demand.

mechanism: the same core operation runs, but with external input GATED
(suppressed). the compressed memory fragments are recombined through
the mixing operation to produce novel configurations that have never
been directly experienced.

control:
- imagination is OFF by default (suppression gate closed)
- the machine activates imagination when it detects high uncertainty
  (prediction errors are large and increasing) or when explicitly
  asked to "think about" something
- the suppression gate prevents uncontrolled generation (no intrusive
  thoughts). the gate is a learned binary control signal.
- during imagination, the machine evaluates each generated scenario
  against its learned model of plausibility. implausible scenarios
  are discarded. plausible ones inform the next action.

## fast and slow processing

the machine automatically detects when it needs more computation:

- **fast path**: one pass through the core operation. good for
  familiar inputs where predictions match reality (small errors).
  cheap. this is the default.
- **slow path**: multiple iterations of the core operation on the
  same input. each iteration refines the result. triggered when
  prediction errors exceed a threshold (the machine is surprised
  or uncertain). more expensive but more accurate.

the switching is automatic: the prediction error magnitude at each
step determines whether to continue iterating or commit to the output.
this is the biological equivalent of the ACC conflict monitor -- it
detects when more processing is needed and triggers it.

## processing units (not "neurons")

each processing unit is NOT a point. it has multiple independent
sub-operations (like dendritic compartments):

- K = 4-8 independent sub-gates per unit (block-diagonal)
- each sub-gate receives from a DIFFERENT source (some from input,
  some from memory, some from neighboring units)
- the sub-gates perform independent competitive selection (k-WTA)
  before their outputs are combined
- this gives each unit ~4-8x the computational capacity of a point
  unit at the same parameter count

the multi-compartment structure is validated by Beniaguev 2021
(single neuron = 5-8 layer deep network, depth from NMDA nonlinearity).

## connectivity

sparse, structured, not fully connected:

- each unit connects to ~1-20% of other units (not all)
- connections are structured: high local connectivity (nearby units
  connect densely), sparse long-range connections (distant units
  connect selectively through learned hubs)
- hub units (rich club) are a small fraction (~5-10%) that connect
  widely and mediate global integration
- this small-world topology is validated by the connectomics research
  (watts & strogatz 1998, van den heuvel & sporns 2011)

## self-monitoring (telemetry)

a separate, lightweight monitoring system observes the main computation:

- tracks: memory state norms, prediction error magnitude, spike
  patterns, imagination gate state, fast/slow mode, which sub-gates
  are active
- produces: a real-time dashboard of the machine's internal state
- enables: sampling data at any point for analysis
- does NOT interfere with the main computation
- this is the "medical system" -- see everything as it happens

planned for future: built-in self-awareness where the monitoring
signals feed back into the main computation (the machine reasons
about its own uncertainty).

## two operating modes

**on-demand mode**: receives input, processes it through enough
iterations to produce a confident output, emits the output, stops.
memory state is preserved between calls but no background processing
runs. like current AI chatbots.

**persistent mode**: always running. between active tasks, the machine
runs at reduced priority:
- consolidation: compresses and reorganizes memory
- imagination: generates and evaluates hypothetical scenarios
- maintenance: cleans up memory, removes redundant entries
- monitoring: self-checks, confidence calibration

the machine works in both modes. persistent is better but on-demand
is cheaper.

## what the first version must do

recognition: given an input signal (text, image, pattern), identify
what it is by matching against stored memory.

memory: store new experiences with full detail. retrieve them by
content (what was it?), by time (when did it happen?), or by
association (what else was happening?).

prediction: given a sequence of signals, predict what comes next.
measure prediction quality. use prediction errors to drive learning
and allocate processing.

these three capabilities emerge from the same core operation:
- recognition = retrieve mode (match input against memory)
- memory = remember mode (write input to memory state)
- prediction = predict mode (compare input against expectation)

## research questions -- ANSWERED by experiments (2026-04-09)

1. **learning rule**: ANSWERED. backprop for v1. one-shot hebbian gives
   90% of optimal for free (experiment 012). delta rule = gradient descent
   for linear layers. the unified learning hypothesis (forward-pass-as-
   learning) is an untested conjecture requiring prediction error and
   runtime-adaptive thresholds not yet in the architecture.

2. **compression level**: ANSWERED. rate-coded k-WTA at 10% = 0.56 bits/dim,
   CKA 0.71. at 20% = 0.92 bits/dim, CKA 0.81. the tradeoff is smooth
   and predictable (experiment 007). novel: no prior work on k-WTA +
   rate-coded activation compression with CKA validation.

3. **memory capacity**: ANSWERED. binary hopfield saturates at alpha=0.08-0.10
   with 5% corruption. ternary has ~5x lower practical capacity (experiment
   006). delta rule erasure is mandatory for practical capacity. recurrent
   state alone cannot provide eidetic memory -- exact retrieval mode (MLA
   function) needed for long-range recall.

4. **imagination quality**: ANSWERED. recombined queries produce structured
   interpolation (cosine 0.93), not noise. random queries produce -0.02.
   novelty rate 7.5% (experiment 008). imagination via outer-product
   recombination is functional.

5. **connectivity**: ANSWERED. small-world beats random sparse at 50%
   density when connection counts are matched (experiment 009, corrected).
   at lower densities, random sparse shows higher overlap. use small-world
   BETWEEN processing stages, random sparse WITHIN each memory unit.

## exact retrieval as a mode (replacing separate MLA module)

the formal impossibility result (wen et al. ICLR 2025) proves pure recurrence
cannot solve associative recall perfectly beyond a retrieval horizon. the
solution is NOT a separate MLA module but a MODE of the same operation:

- when the mode signal indicates retrieval is needed and the recurrent state
  cannot provide a confident answer (high prediction error on retrieval),
  the operation switches to exact attention over consolidated snapshots
- consolidated snapshots are compressed summaries of past recurrent state,
  stored periodically (every N steps)
- this is functionally equivalent to MLA but implemented as a mode, not
  a separate layer type
- the retrieval mode uses dot-product attention over the snapshot bank,
  which is O(T/N) in the number of snapshots, not O(T) per token

## implementation sequencing (mandatory -- do NOT bundle)

each feature is validated in isolation before combination. this follows
the phase 5 sequencing protocol from CLAUDE.md. **the ordering below has
been revised on 2026-04-12 after god_run_v2's 0% passkey confirmed that
the prior ordering (k-WTA on both keys AND values) destroys verbatim
memory. see `memory/project_v2_diagnosis.md` for the 8 root causes.**

**critical invariant (post-god_run_v2)**: keys must stay dense. k-WTA,
sparsification, and any form of address-space truncation must be applied
only to values, never to keys. a 20%-sparse key in a 64-dim head carries
~13 effective dimensions, which drops Hopfield interference-free capacity
from ~9 patterns to ~2 patterns per head. delta rule erasure with sparse
keys also reads only the 13 surviving dimensions of the state, leaving
"ghost" content in the zeroed dimensions that accumulates and interferes.
both failure modes are explicit in the empirical data from god_run_v2.

**run 1 (ternary-spike baseline at 350M)**: use `neural_machine.py` as-is.
ternary spikes (adaptive threshold, not k-WTA), no delta erasure, no BCM,
standard SwiGLU (not multi-compartment), no imagination probe, no PC
diagnostic head. target: replicate run_010's 2.375 BPB at 267M scaled up
to 350M. **critical success metric: nonzero passkey at 256 or 512 tokens.**
if this run produces 0% passkey, the problem is deeper than the bundle
and the entire delta-rule architecture needs rethinking.

**run 2 (k-WTA 50% on values only, dense keys)**: replace ternary spikes
with rate-coded k-WTA at 50% selection applied ONLY to values. keys remain
dense (no k-WTA on k). apply to the neural_machine.py baseline from run 1,
not god_machine. target: preserve run 1's retrieval accuracy, achieve some
compression benefit from value sparsity. if 50% works, follow-up run tries
30% on values; NEVER sparsify keys below 100%.

**run 3 (delta erasure with dense keys)**: add delta-rule erasure
`state -= beta * k * (k^T @ state)` on top of run 2. dense keys allow
erasure to see the full state and correctly remove old associations.
target: preserve or improve retrieval over run 2, add targeted unlearning
capability useful for safety/provenance.

**run 4 (BCM alpha with faster EMA)**: add BCM-adapted alpha on top of
run 3. use momentum 0.95 instead of 0.99 (half-life ~14 steps vs ~69).
initialize `running_state_norm` to 0.01 instead of 1.0 so the BCM
correction is active from step 0 rather than kicking in at step ~69 after
the state has already saturated. target: activity-dependent forgetting
without catastrophic saturation.

**what is explicitly NOT added unless validated in a dedicated run:**
- imagination probe (god_run_v2 confirmed it creates a gradient bypass
  that the model uses instead of learning content-addressable storage,
  gate rose from 0.12 to 0.22 during training)
- multi-compartment SwiGLU K=4 (halves effective FFN width; responsible
  for 283M vs 350M param gap and +0.050 bpb regression in god_run_v2)
- PC diagnostic head (loss drag, no retrieval benefit)
- retrieval mode / consolidated snapshots (deferred to next_gen.md phase 6
  tiered architecture — not a bolt-on to the single-tier design)

each run that improves or maintains quality AND retrieval is KEPT.
each run that degrades is DROPPED. retrieval is a HARD gate: a run that
drops passkey to 0% is rejected regardless of BPB improvement.

## verification

each run is verified by:
1. prosecutor audit of the code before execution, with every finding fixed to zero
   (no cherry-picking; each finding is a bug class, grep analogous sites and fix them all)
2. eara ultra loop pre-push checks (5 subagents)
3. comparison against the baseline (run 1) on matched token budget
4. statistical significance (p < 0.01) for any claimed improvement
5. documentation update after each run
6. end-to-end telemetry round-trip: every metric computed in code must be written to the
   on-disk jsonl and verified by reading it back. in-memory assertions alone do not count
   (god_run lost 35 probe metrics because this guard was missing)

## empirical status

**god_run (2026-04-11)**: the first direct test of the 5-feature bundle. 283m params on
fineweb-edu, 131m tokens, 4000 steps on h200. bypassed the sequential-isolation protocol
below (activated all 5 features at once with always-on imagination probe and pc diagnostic
head). result: val_bpb 1.3950 (best in project history, 2.57x vs transformer baseline) but
retrieval FAILED at every tested length (passkey 0/20 and selective copy 0/20 at 256, 512,
1024, 2048, 4096). delta state structure probe shows the delta memory is statistically
indistinguishable from high-dimensional noise (mean_structure_ratio 0.981, mean_pairwise_cos
-0.003). perplexity-at-length decreases monotonically with context, proving the
compressed-attention path uses context correctly; the delta-rule memory is the channel
that failed.

interpretation: the compressed-attention + mlp path fit the next-byte distribution while
the delta-rule memory never became content-addressable. this is the lossy-mechanism
failure mode predicted by `wiki/synthesis/compression_beyond_quantization.md`. a
confounder remains: the BCM train/eval path divergence (prosecutor finding F1) meant the
model was trained under one dynamical rule and evaluated under a different one for
everything beyond the first chunk. F1 and 16 related prosecutor findings have been fixed,
and the re-run will determine whether F1 was the primary cause or whether the 5-feature
bundle itself destroys verbatim memory.

**decision rule for post-re-run analysis**:
- if re-run passkey > 0 at any length: F1+F2 was the culprit. proceed with the sequential
  isolation protocol below as originally planned, using the 5-feature bundle as validated.
- if re-run passkey still 0 at all lengths: the bundle destroys verbatim memory regardless
  of path alignment. fall back to run 1 (below) as the honest starting point and introduce
  one feature at a time, validating retrieval after each.

**god_run_v2 (2026-04-12)**: the re-run. all 17 F1-F17 + 14 G1-G14 prosecutor findings
fixed, including the critical F1 `torch.sigmoid → torch.exp` math correction for the
log-space gate and F2 `running_state_norm` training-only gating. 283m params, same
pipeline, 4000 steps, 131m tokens, 59 min runtime. **result: val_bpb 1.4453 (+0.050 vs
v1), passkey 0/100 at 256/1024/4096 (Wilson 95% CI upper 3.7%), copy 0/100 at every
length. F1 math fix did NOT recover retrieval.** the 5-feature bundle is intrinsically
broken: dense-key requirement for content-addressable storage is violated by k-WTA at
20% on keys. decision rule hits the bundle-is-broken branch: fall back to run 1.

external review (2026-04-12) identified 8 root causes. the two critical ones are k-WTA
on keys (destroys address space, drops per-head capacity from ~9 patterns to ~2) and
delta erasure with sparse keys (ghost content accumulates in zeroed dimensions). the
six supporting causes are listed in `memory/project_v2_diagnosis.md` and inform the
revised run ordering below.

see `neuroloc/output/god_run/run_card.md`, `neuroloc/output/god_run_v2/run_card.md`,
and `neuroloc/wiki/tests/god_run_findings.md` for full detail.
