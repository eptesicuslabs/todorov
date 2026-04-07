# GWT vs transformer architecture

## what is being compared

[[global_workspace_theory]] (GWT) proposes that consciousness arises from a functional architecture: specialized unconscious processors + a shared global workspace + ignition-mediated broadcast. the transformer architecture (Vaswani et al. 2017, and its derivatives including todorov) has a superficially similar structure: layers (specialized processors) + residual stream (shared workspace) + attention (selective access). Goyal & Bengio (2022, ICLR) explicitly built a neural network architecture inspired by GWT, implementing specialist modules that compete for access to a bandwidth-limited shared workspace.

the comparison matters because if the residual stream IS a global workspace, then transformer-derived architectures already implement a key principle of conscious processing, and todorov inherits this property. if it is NOT, then the differences identify specific architectural features that would need to be added.

## dimension-by-dimension comparison

### shared workspace

**GWT**: the global workspace is a limited-capacity communication medium accessible by all processors. information in the workspace is globally available -- every processor can read it. the workspace has severe capacity constraints (~4 items, related to working memory, see [[theta_oscillations]]).

**transformer**: the residual stream is a d_model-dimensional vector that flows through all layers. each layer reads from it (via normalization) and writes to it (via addition). it is the ONLY channel of inter-layer communication. information is linearly superposed: x_{l+1} = x_l + f_l(norm(x_l)). every layer can read everything. every layer writes everything.

**verdict**: the residual stream IS a shared workspace in the functional sense -- it is a shared medium accessible by all processing units. but it lacks capacity limits: the full d_model vector flows through every layer with no bandwidth restriction. the Goyal & Bengio workspace explicitly adds a bottleneck (fewer workspace slots than specialist states), which the residual stream does not have.

### broadcast mechanism

**GWT**: broadcast is an active, selective process. information enters the workspace through competitive ignition (see [[ignition_dynamics]]). once in the workspace, it is actively broadcast to all processors via long-range feedback connections. the broadcast is sustained by recurrent amplification through NMDA-mediated feedback loops.

**transformer**: there is no explicit broadcast event. each layer adds its output to the residual stream, and all subsequent layers can read it. this is passive broadcasting: information accumulates in the stream through linear superposition. there is no ignition event, no threshold, no active maintenance. information does not need to "win" a competition to be written to the stream -- every layer always writes.

**verdict**: the mechanisms are fundamentally different. GWT broadcast is active, selective, threshold-gated, and sustained by recurrence. residual stream accumulation is passive, non-selective, always-on, and requires no maintenance. the transformer has broadcast-by-default (everything is always available), while GWT has broadcast-by-selection (only ignited content is available).

### selectivity

**GWT**: the defining feature of the workspace is selectivity. most processing remains unconscious (local, modular, never broadcast). only the information that wins the competition and triggers ignition enters the workspace. the TRN (see [[thalamocortical_loops]]) provides subcortical gating. attention (see [[selective_attention]]) provides the top-down bias that determines what ignites. the workspace is a bottleneck: it forces the system to select.

**transformer**: the residual stream has no selectivity for writes. every layer contributes to the stream unconditionally. for reads, attention layers (MLA in todorov, 6/24 layers) perform selective retrieval via softmax attention -- they selectively weight past information. KDA layers (18/24) perform content-addressable retrieval from their recurrent state. but neither mechanism prevents a layer from writing its output to the stream.

**verdict**: the transformer has selective reading (attention) but non-selective writing. GWT requires selective writing (only ignited content enters the workspace). this is a fundamental architectural difference. in the transformer, the residual stream accumulates everything -- there is no mechanism to keep information OUT of the stream.

### nonlinearity and threshold

**GWT**: ignition is a nonlinear, all-or-none event -- a bifurcation in network dynamics. below threshold: local processing, decaying activity, no broadcast. above threshold: sudden global activation, sustained broadcast, suppression of competitors. this threshold creates a qualitative distinction between conscious and unconscious processing.

**transformer**: the residual stream path is linear (x_{l+1} = x_l + f_l(...)). the layer computation f_l contains nonlinearities (SwiGLU activation, softmax attention), but the stream-level dynamics are purely additive. there is no threshold for stream access. there is no bifurcation between broadcast and non-broadcast states. every layer's output, regardless of magnitude or relevance, is added to the stream.

**verdict**: the transformer lacks the ignition threshold that defines GWT. every computation contributes to the stream with no all-or-none gating. this is the most important difference: GWT is defined by the existence of a nonlinear broadcast threshold, and the transformer does not have one.

### information integration

**GWT**: the workspace enables flexible integration of information from different processors. a visual percept, an emotional evaluation, and a motor plan can be combined in the workspace to produce coherent behavior. this integration is what [[integrated_information_theory]] formalizes as Phi.

**transformer**: the residual stream enables information integration through superposition. features from different layers coexist in the same d_model-dimensional vector. attention layers can attend to and combine features deposited by different earlier layers. but the integration is implicit (linear superposition) rather than explicit (active combination in a workspace).

**verdict**: both architectures enable information integration, but through different mechanisms. GWT uses active combination in a capacity-limited workspace (forced selection). the transformer uses passive superposition in an unlimited stream (no forced selection). the transformer may actually integrate MORE information (it keeps everything), but this may be less useful than the selective integration GWT provides.

### recurrence and sustained activity

**GWT**: workspace activity is sustained by recurrent amplification. NMDA-mediated feedback loops between workspace neurons and sensory cortex create self-sustaining reverberant activity that persists for ~200-300 ms after stimulus offset. this active maintenance is crucial for the workspace to function as a shared communication medium.

**transformer**: in a standard transformer, there is no recurrence -- information flows forward through layers. in todorov, KDA layers (18/24) have recurrent state: S_t = diag(alpha) * S_{t-1} + beta_t * k_t * v_t^T. this provides a form of sustained memory within each layer. but this recurrence operates within individual layers, not at the level of the residual stream. the stream itself has no maintenance mechanism.

**verdict**: todorov has recurrence within layers (KDA state), but not at the stream level. GWT has recurrence at the workspace level (reverberant loops between workspace and processors). the architectural location of recurrence differs: local (within-layer) in todorov vs global (workspace-level) in GWT.

## the Goyal & Bengio architecture

Goyal, Didolkar et al. (2022, ICLR) built an explicit GWT-inspired architecture:
- specialist modules process different aspects of the input
- a shared workspace consists of a small number of slot vectors (e.g., 4 slots)
- specialists write to the workspace via cross-attention (competing for limited slots)
- specialists read from the workspace via cross-attention
- the workspace acts as a bandwidth-limited bottleneck: only the most relevant specialist outputs enter the shared representation

this architecture outperformed pairwise specialist communication on tasks requiring multi-specialist coordination. the bandwidth limitation (fewer workspace slots than specialist states) forced selection and improved generalization.

the Goyal & Bengio workspace differs from the transformer residual stream in the key ways that GWT predicts: it has capacity limits, competitive write access, and explicit selection of what gets broadcast.

## verdict

the transformer residual stream shares the TOPOLOGY of GWT's workspace -- it is a shared medium accessible by all processing units -- but lacks the DYNAMICS:

1. **no ignition**: there is no nonlinear threshold for workspace access. every layer always broadcasts. GWT is defined by the existence of this threshold.
2. **no selectivity for writes**: every layer unconditionally adds to the stream. GWT requires competitive selection for workspace entry.
3. **no capacity limit**: the full d_model vector flows unchanged. GWT's workspace has severe capacity constraints that force selection.
4. **no active maintenance**: the stream is passive (additive accumulation). GWT's workspace is actively maintained by recurrent amplification.
5. **no suppression**: writing to the stream does not suppress competitors. GWT's ignition actively inhibits competing representations.

the residual stream is a shared workspace without ignition. this makes it a fundamentally different computational architecture from GWT, despite the superficial similarity.

## dissenting argument

the comparison may be too literal. GWT describes a biological system optimized by evolution for survival, where information bandwidth is severely constrained by metabolic costs (see [[brain_energy_budget]], [[metabolic_constraints_on_computation]]) and physical wiring constraints (long-range axons are expensive in space and energy). the capacity limits, selectivity, and ignition threshold may be solutions to biological engineering constraints, not computational principles.

the transformer operates in silicon with no metabolic cost per information bit. if the function of the GWT workspace is to enable inter-processor communication, then the residual stream achieves this function MORE EFFICIENTLY by removing the bottleneck. the capacity limits of consciousness may be bugs (engineering constraints), not features (computational advantages). under this view, the transformer is not a failed implementation of GWT but a SUPERIOR implementation that removes the bottleneck.

this argument has force for tasks where information bandwidth is not a constraint (next-token prediction on sequences). it has less force for tasks where selection and focus are valuable (planning, reasoning, sustained attention to a single problem). the Goyal & Bengio result -- that adding a bottleneck IMPROVES performance on multi-specialist coordination -- is evidence against the "bottleneck as bug" interpretation.

## key references

- Baars, B. J. (1988). A Cognitive Theory of Consciousness. Cambridge University Press.
- Goyal, A., Didolkar, A. et al. (2022). Coordination Among Neural Modules Through a Shared Global Workspace. ICLR 2022.
- Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS 2017.
- Elhage, N. et al. (2021). A Mathematical Framework for Transformer Circuits. Anthropic.
- Mashour, G. A. et al. (2020). Conscious Processing and the Global Neuronal Workspace Hypothesis. Neuron, 105(5), 776-798.

## see also

- [[global_workspace_theory]]
- [[ignition_dynamics]]
- [[integrated_information_theory]]
- [[selective_attention]]
