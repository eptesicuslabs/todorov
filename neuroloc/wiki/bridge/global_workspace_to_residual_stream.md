# bridge: global workspace to todorov residual stream

status: current (as of 2026-04-16).

## the biological mechanism

the global neuronal workspace (Baars 1988, Dehaene & Naccache 2001) is a cognitive architecture for consciousness. the brain contains many specialized processors (visual cortex, auditory cortex, motor areas, memory systems) that operate in parallel and largely unconsciously. conscious access occurs through ignition: a nonlinear, all-or-none broadcast from a local processor to a global workspace that makes information available to ALL other processors. see [[global_workspace_theory]], [[ignition_dynamics]], [[thalamocortical_loops]].

the neural substrate is a network of large pyramidal neurons in cortical layers II/III and V with long-range axonal connections, concentrated in prefrontal and parietal cortex. these workspace neurons are densely interconnected and, when activated above threshold, produce a self-sustaining reverberant activation (ignition) that broadcasts information globally. the thalamocortical system (particularly higher-order thalamic relays) provides additional routing and gating infrastructure: the thalamic reticular nucleus (TRN) selectively inhibits thalamic relay neurons, controlling which information flows to cortex.

key properties of the biological workspace:
- **ignition threshold**: information enters the workspace only through a nonlinear threshold-crossing event. sub-threshold signals remain unconscious (subliminal processing). the threshold depends on signal strength, attentional state, and ongoing brain activity
- **selectivity**: only one coherent representation occupies the workspace at a time. processors compete for workspace access. ignition of one representation suppresses competitors through lateral inhibition
- **capacity limits**: the workspace can hold ~4 items (related to working memory capacity, theta-gamma coupling). this severe limitation forces the system to select
- **active maintenance**: workspace representations are sustained by recurrent NMDA-mediated feedback loops (~200-300 ms), not passive accumulation
- **broadcast**: information in the workspace is simultaneously available to all processors (language, motor, memory, evaluation), enabling flexible combination and verbal report

## the current todorov implementation

the residual stream in todorov:
- a d_model-dimensional vector (d_model=1024 at 267M scale) that flows through all 24 layers
- each layer reads from it (via RMSNorm) and writes to it (via addition)
- the stream is the ONLY channel of inter-layer communication
- information is linearly superposed: x_{l+1} = x_l + f_l(RMSNorm(x_l))
- every layer can read everything (no selective gating of reads)
- every layer writes everything (no selective gating of writes)
- there is no threshold for stream access: every layer's output is unconditionally added
- there is no capacity limit: the full d_model vector flows unchanged through all layers

comparison with GWT's global workspace:

**the residual stream IS a shared workspace accessible by all "processors" (layers)**. this is the one dimension where the analogy holds. the residual stream is the sole communication channel, accessible to all layers, and information deposited by any layer is available to all subsequent layers. this matches GWT's core requirement: a shared medium that enables inter-processor communication.

**but the residual stream LACKS ignition**: there is no nonlinear threshold for broadcast. every layer always writes to the stream. in GWT, the threshold is the defining feature -- it creates the qualitative distinction between conscious (broadcast) and unconscious (local) processing. the residual stream has no such distinction. all processing is "conscious" (broadcast) in GWT terms, which is equivalent to saying none of it is.

**the residual stream LACKS selectivity**: everything is broadcast, not just "attended" information. in GWT, processors compete for workspace access and only the winner broadcasts. in todorov, every layer unconditionally contributes. there is no competition for stream access, no suppression of losing representations.

**the residual stream LACKS capacity limits**: the full d_model vector (1024 dimensions at 267M) flows unchanged. GWT's workspace has severe capacity constraints (~4 items) that force selection. the residual stream can carry superposed information from all 24 layers simultaneously.

**the residual stream is passive**: it is a vector being linearly modified by successive additions. GWT's workspace is actively maintained by recurrent amplification through NMDA-mediated feedback loops. the stream has no self-sustaining dynamics. (KDA layers have recurrence WITHIN their state, but this does not operate at the stream level.)

**MLA layers (3/24) are the closest analog to workspace broadcast**: MLA performs global attention (softmax over all past tokens), selectively retrieving information from the shared workspace based on query-key matching. this IS selective access to a shared representation. but MLA reads selectively -- it does not write selectively. and it operates on the inter-token dimension (selecting which past tokens to attend to), not the inter-layer dimension (selecting which features to broadcast).

**KDA layers (18/24) are more like specialized processors**: KDA accumulates information recurrently in a matrix-valued state (S_t). each KDA layer maintains its own private state, not accessible to other layers except through what it writes to the residual stream. this is analogous to GWT's specialized processors that operate in their own state space. but unlike biological processors, KDA layers always broadcast everything they compute to the stream.

## argument FOR: the residual stream as global workspace

Goyal & Bengio (2022) explicitly draw the comparison between transformer residual streams and GWT. their strongest arguments:

1. **shared medium**: the residual stream is functionally a global workspace -- the only communication channel between processing units. this is not a metaphor. it is a literal architectural property: remove the residual stream and inter-layer communication ceases entirely.

2. **linear superposition enables broadcast**: information deposited by layer 3 is available to layers 4-24. the superposition of features from multiple layers in a single vector is a form of simultaneous availability -- the defining property of broadcast. Elhage et al. (2021) showed that transformers learn to use the residual stream as a shared communication channel where features from different layers coexist.

3. **MLA as selective retrieval**: MLA layers perform content-based retrieval from the global representation -- the closest analog to "reading from the workspace." the softmax attention mechanism selectively weights past contributions, implementing a form of the competitive access that GWT requires.

4. **KDA state as unconscious processing**: the recurrent states within KDA layers are private -- not directly accessible by other layers. this mirrors GWT's specialized processors that operate unconsciously. information enters the global workspace (residual stream) only through each layer's output projection, which is a form of access control.

## argument AGAINST: the residual stream is NOT a global workspace

the strongest argument against the analogy rests on five structural differences:

1. **no ignition (severity: CRITICAL)**: GWT is defined by a nonlinear threshold-crossing event that creates a qualitative distinction between broadcast and non-broadcast states. the residual stream has NO such threshold. every layer ALWAYS broadcasts. this is not a minor difference -- ignition is the central mechanism of GWT. a workspace without ignition is not a GWT workspace. it is a shared bus.

2. **no write selectivity (severity: HIGH)**: in GWT, processors compete for workspace access and only the winner enters. in todorov, every layer writes unconditionally. the layer cannot choose NOT to broadcast. there is no competition, no suppression of losers. the stream accumulates EVERYTHING.

3. **no capacity limit (severity: HIGH)**: GWT's workspace holds ~4 items. this capacity limit is not an implementation detail -- it is a core theoretical prediction. the workspace's value comes from its scarcity: by limiting what can be broadcast, it forces the system to select the most relevant information. the residual stream has capacity proportional to d_model (1024 dimensions x theoretical unlimited number of superposed features). this is the opposite of a bottleneck.

4. **passive vs active maintenance (severity: MODERATE)**: GWT's workspace representations are actively maintained by recurrent feedback loops. the residual stream is a passive accumulator -- add and move on. without active maintenance, there is no mechanism for the workspace to sustain a representation against decay or interference. (KDA layers have recurrence, but within their private state, not at the stream level.)

5. **no gating (severity: MODERATE)**: a layer cannot choose to not broadcast, and a layer cannot choose to not read. in GWT, the TRN selectively gates thalamic relay, and attentional mechanisms selectively gate workspace access. the residual stream has no gates -- it is an open channel.

## the proposed change

### option 1: bottleneck layer (ignition analog)

add a learned gating layer at one or more points in the 24-layer stack that compresses the residual stream through a bottleneck:

    gate = sigmoid(W_gate @ RMSNorm(x_l))
    x_l_gated = gate * x_l

this implements a soft threshold: only features with high gate values pass through. features with low gate values are suppressed. this is not true ignition (no all-or-none bifurcation), but it introduces selectivity into the stream.

placement: after every 6 layers (4 bottleneck points in a 24-layer stack), or at a single midpoint (after layer 12). the midpoint placement mirrors the "bow-tie" architecture of the GNW: convergence in early layers, bottleneck, divergence in later layers.

**expected impact**: uncertain. the bottleneck restricts information flow, which could help (forcing selection of relevant features, improving coherence) or hurt (discarding useful information, increasing loss). the Goyal & Bengio result suggests bottlenecks help on multi-specialist coordination tasks, but next-token prediction may not require this kind of coordination.

**risk**: information loss. the residual stream carries superposed features from many layers, and a gate that suppresses features at layer 12 permanently removes them from all subsequent layers. if the gate learns to pass everything (sigmoid -> 1 for all features), it becomes a no-op and adds parameters without benefit. if it learns to aggressively filter (sigmoid -> 0 for many features), it could severely degrade BPB.

### option 2: competitive write gating

modify each layer's write to the stream to be gated by the layer's confidence:

    confidence = sigmoid(W_conf @ h_l)
    x_{l+1} = x_l + confidence * f_l(RMSNorm(x_l))

this allows layers to modulate HOW MUCH they write to the stream. layers that have little to add (confidence near 0) write weakly. layers with important updates (confidence near 1) write strongly. this does not implement competition between layers (no suppression of losers), but it adds selectivity.

**expected impact**: low risk, low expected benefit. most layers will learn confidence near 1 (write everything), reducing to the current architecture. the gradient through the confidence gate may create optimization difficulties. this is essentially a learned residual scaling factor, which has been explored (ReZero, FixUp) without dramatic benefit.

### option 3: sparse broadcast with dimensional gating

instead of broadcasting the full d_model vector, each layer selects a SUBSET of dimensions to write to:

    mask = top_k(sigmoid(W_mask @ h_l), k=d_model/4)
    x_{l+1} = x_l + mask * f_l(RMSNorm(x_l))

this implements dimensional selectivity: each layer broadcasts only its top-k most activated features. this is analogous to GWT's selectivity requirement: not everything gets broadcast. and it introduces an implicit capacity limit: only d_model/4 features can be updated per layer.

**expected impact**: moderate risk. this interacts with the superposition hypothesis -- if features are superposed across dimensions, sparse masking could corrupt features that share dimensions. top-k selection adds discontinuities that may complicate optimization. but if it works, it would be the strongest architectural analog to GWT's selective broadcast.

## implementation spec

all three options are implementable as modifications to the layer block in train.py. option 1 is simplest (one additional linear layer + sigmoid at designated points). option 2 is a per-layer scalar or per-dimension gate. option 3 requires a top-k selection mechanism.

parameter overhead:
- option 1: W_gate is d_model x d_model = ~1M params per bottleneck point, ~4M total for 4 bottlenecks. <2% of 267M model
- option 2: W_conf is d_model x 1 = 1024 params per layer, ~25K total. negligible
- option 3: W_mask is d_model x d_model = ~1M per layer, ~24M total for all layers. ~9% overhead

## risk assessment

- option 1 (bottleneck): MODERATE risk. information loss is the primary danger. could be catastrophic if the gate learns the wrong filtering policy. mitigated by residual bypass: x_l_gated = gate * x_l + (1 - gate) * x_l_detached, ensuring gradient flow. phase 6+ at earliest
- option 2 (confidence gating): LOW risk. likely learns identity (confidence = 1). may become a no-op. harmless but useless. testable at any phase
- option 3 (sparse broadcast): HIGH risk. top-k discontinuities, superposition interference, significant parameter overhead. highest potential benefit if it works. phase 6+ or later

estimated probability of meaningful BPB improvement at matched parameters: 10-20% for any option. the residual stream works well precisely because it broadcasts everything. removing information rarely helps in language modeling.

## verdict

the residual stream is topologically a workspace (shared medium, accessible by all layers) but dynamically not (no ignition, no selectivity, no capacity limits, passive accumulation). the strongest argument for the analogy is that the residual stream IS the sole communication channel. the strongest argument against is that a communication channel without access control is a bus, not a workspace. the GWT workspace's value comes from what it EXCLUDES, not what it includes. the residual stream excludes nothing.

the most honest assessment: todorov's residual stream is a SHARED BUS, not a GLOBAL WORKSPACE. a shared bus is a perfectly good architecture for language modeling. whether adding workspace-like properties (bottleneck, selectivity, capacity limits) would improve performance is an open empirical question that the Goyal & Bengio result suggests is worth testing -- but at later phases, not now.

## see also

- [[global_workspace_theory]]
- [[ignition_dynamics]]
- [[thalamocortical_loops]]
- [[gwt_vs_transformer]]
- [[integrated_information_theory]]
