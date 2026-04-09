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
  sparse ternary values {-1, 0, +1}
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

## open research questions (to be resolved by experimentation)

1. what learning rule? backprop works but isn't the final answer.
   local rules (three-factor, eligibility traces) are biological but
   don't scale yet. hybrid? something new? resolve by testing.

2. what compression level? ternary activations are proven. ternary
   weights need testing. hierarchical ternary (0.37 bits/dim) needs
   testing. how much compression can you do before quality degrades?

3. what is the right memory capacity? how large should the state
   matrix be? how often should consolidation run? what rank for
   SVD snapshots? resolve by scaling experiments.

4. how does imagination quality scale? does the recombination
   operation produce useful novel scenarios at scale, or does it
   produce noise? test by measuring whether imagined scenarios
   improve prediction accuracy.

5. optimal sparse connectivity pattern? random sparsity, structured
   sparsity, learned sparsity? what fraction of connections is
   optimal for the compute-communication tradeoff?

## resolution plan for open questions

each open question gets a kaggle experiment + research subagent in parallel.
answers fold back into this blueprint, converting it to a working spec.

**Q1: learning rule**
- experiment: implement three-factor learning on the pattern completion
  simulation. compare learning speed and final quality vs backprop at
  matched parameter count. kaggle CPU.
- research: search for 2024-2026 papers on scalable local learning rules.
  eligibility traces at 100M+ params. any breakthroughs?

**Q2: compression level**
- experiment: hierarchical ternary (0.37 bits/dim) on the existing
  todorov 6M architecture. measure BPB degradation vs standard ternary.
  kaggle T4.
- research: search for ternary weight papers at LLM scale. bitnet,
  1-bit LLMs, any quality-preserving full-ternary results?

**Q3: memory capacity**
- experiment: scale the pattern completion pilot from 200 to 10K neurons.
  measure capacity curve. when does the outer-product memory saturate?
  kaggle CPU.
- research: modern hopfield network capacity results. exponential vs
  polynomial capacity. what determines the practical limit?

**Q4: imagination quality**
- experiment: implement controlled recombination on the associative
  memory pilot. store 100 patterns. recombine fragment pairs. measure
  whether recombinations are plausible (high cosine with nearest stored
  pattern) vs noise. kaggle CPU.
- research: search for computational creativity papers. generative
  recombination quality metrics. how do you measure if an imagined
  scenario is useful?

**Q5: connectivity pattern**
- experiment: test random vs structured sparsity in the cortical
  microcircuit simulation. does small-world topology improve information
  flow vs random sparse connectivity? kaggle CPU.
- research: search for sparse neural network architecture papers.
  lottery ticket hypothesis. structured pruning. what patterns survive
  training?

## files to create

- `neuroloc/spec/core_operation.md` -- the math, in detail
- `neuroloc/spec/memory.md` -- eidetic memory architecture
- `neuroloc/spec/compression.md` -- novel compression proposals
- `neuroloc/spec/imagination.md` -- controlled generation
- `neuroloc/spec/monitoring.md` -- telemetry system
- `neuroloc/spec/v1_requirements.md` -- first version capabilities

## verification

test the core operation components individually:
- k-WTA selection: already validated (pilot showed 0.925 vs 0.500)
- BCM-like decay: already validated (gamma=0.5, p=0.001)
- delta rule erasure: implement and test on selective copy task
- multi-compartment units: test 4-gate vs 1-gate on pattern classification
- imagination gate: test controlled recombination on associative tasks
- consolidation: test memory retention over 10K+ steps with and without

then integrate and test the unified operation on:
- sequence prediction (text, byte-level)
- pattern classification (images, synthetic)
- associative retrieval (store N items, retrieve by partial cue)
- novel scenario generation (recombine stored fragments, evaluate)
