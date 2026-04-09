# neural machine specification

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
the phase 5 sequencing protocol from CLAUDE.md.

**run 1 (baseline)**: standard todorov at 350M with current ternary spikes,
standard alpha, no delta rule erasure, standard SwiGLU. replicate the
0.663x BPB ratio at larger scale. this is the comparison target.

**run 2 (rate-coded k-WTA)**: replace ternary spikes with rate-coded k-WTA
at 20% selection. measure: BPB, MI, CKA, firing rate. if gradient flow
breaks at 20%, fallback to 30-40% and anneal.

**run 3 (delta rule erasure + BCM alpha)**: add targeted overwrite and
activity-dependent forgetting (gamma=0.3). measure: BPB, state norm
dynamics, long-sequence retrieval quality.

**run 4 (multi-compartment SwiGLU)**: replace standard SwiGLU with K=4
block-diagonal sub-gates. measure: BPB change, per-layer representation
diversity. if it hurts, drop it.

**run 5 (retrieval mode)**: add the exact-retrieval mode with consolidated
snapshots. measure: long-range retrieval accuracy, BPB on documents
>1024 tokens.

each run that improves or maintains quality is KEPT. each run that
degrades is DROPPED. the final architecture is the composition of
all surviving features.

## verification

each run is verified by:
1. prosecutor audit of the code before execution
2. eara ultra loop pre-push checks (5 subagents)
3. comparison against the baseline (run 1) on matched token budget
4. statistical significance (p < 0.01) for any claimed improvement
5. documentation update after each run
