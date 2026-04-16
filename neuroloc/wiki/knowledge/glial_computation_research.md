# glial computation research

status: current (as of 2026-04-16).

curated peer-reviewed research on glial cells as active participants in neural computation. glia were long dismissed as passive support cells ("glue"), but the last two decades have revealed that astrocytes modulate synaptic transmission, oligodendrocytes implement activity-dependent myelination (a form of learning), and microglia prune synapses and mediate forgetting. the glia:neuron ratio is 1:1 (not the mythical 10:1), and every synapse is physically enwrapped by astrocytic processes, making the tripartite synapse -- presynaptic neuron, postsynaptic neuron, astrocyte -- the fundamental unit of synaptic computation.

## glia:neuron ratio

### the 1:1 ratio

azevedo, f. a. c., carvalho, l. r. b., grinberg, l. T., farfel, j. M., ferretti, r. E. L., leite, r. E. P., jacob filho, w., lent, r., & herculano-houzel, s. (2009). equal numbers of neuronal and nonneuronal cells make the human brain an isometrically scaled-up primate brain. *journal of comparative neurology*, 513(5), 532-541.

key finding: using the isotropic fractionator method (dissolving brain tissue into a suspension of free nuclei and counting neuronal vs nonneuronal nuclei using neuron-specific markers), herculano-houzel's group determined that the adult human brain contains approximately 86 billion neurons and 85 billion nonneuronal cells (glia + vascular cells), yielding a ratio very close to 1:1. the oft-cited "10:1 glia-to-neuron ratio" was a myth perpetuated through textbooks without empirical basis. the ratio varies by brain region: cerebral cortex ~1.5 glia per neuron, cerebellum ~0.2 glia per neuron (because cerebellar granule cells are tiny and densely packed).

relevance to neural computer: the 1:1 ratio means that the brain dedicates as much cellular machinery to glial computation as to neuronal computation. this doubles the relevant computational substrate from ~86 billion units to ~170 billion units. todorov models only the neuronal component (spiking, recurrence, attention); the glial component (synaptic modulation, myelination, pruning) is entirely absent. any comparison between biological and artificial computational capacity must account for this -- the brain has 2x more processing units than neuron-count estimates suggest.

confidence: high. the isotropic fractionator is a validated counting method replicated across species and labs. caveat: nonneuronal cells include vascular cells (endothelial, pericytes) in addition to glia; the purely glial count is somewhat less than 85 billion.

## astrocytes and the tripartite synapse

### the tripartite synapse

araque, a., parpura, v., sanzgiri, r. p., & haydon, p. g. (1999). tripartite synapses: glia, the unacknowledged partner. *trends in neurosciences*, 22(5), 208-215.

key finding: araque et al. formalized the concept of the tripartite synapse: the functional synaptic unit consists of the presynaptic terminal, the postsynaptic element, and the perisynaptic astrocyte process. astrocytic processes enwrap >90% of all synapses and can detect neurotransmitter release (via metabotropic receptors), respond with intracellular calcium signals, and release gliotransmitters (glutamate, d-serine, atp) that modulate synaptic transmission. this means synaptic transmission is not a two-party conversation but a three-party negotiation, with the astrocyte providing contextual modulation based on the history and state of the local network.

relevance to neural computer: the tripartite synapse adds a modulatory third party to every connection. in todorov, each weight is a fixed scalar (or quantized value) with no per-synapse modulatory state. the astrocyte component adds a slow (seconds to minutes) state variable to each synapse that integrates local network activity and modulates transmission strength accordingly. this is functionally similar to per-parameter learning rate adaptation (like adam's moment estimates) but implemented at the synapse level rather than as a global optimizer state.

confidence: high. the anatomical relationship is well-established. the functional significance -- whether astrocytic modulation is essential for normal synaptic function or merely modulatory -- is debated. caveat: the term "tripartite synapse" may overstate the astrocyte's role at some synapses where astrocytic coverage is thin or absent.

### astrocytes gate ltp via d-serine

henneberger, c., papouin, t., oliet, s. h. r., & rusakov, d. a. (2010). long-term potentiation depends on release of d-serine from astrocytes. *nature*, 463, 232-236.

key finding: long-term potentiation (ltp) at hippocampal ca3-ca1 synapses requires d-serine released by astrocytes. d-serine is a co-agonist at the nmda receptor -- the receptor cannot open (and therefore ltp cannot be induced) without both glutamate (from the presynaptic neuron) and d-serine (from the astrocyte). clamping astrocytic calcium (which prevents d-serine release) blocked ltp. exogenous d-serine rescued ltp when astrocytic signaling was blocked. this means that the astrocyte effectively gates whether a synapse can undergo long-term plasticity.

relevance to neural computer: this is a biological permission signal for learning. the astrocyte determines not whether a synapse is activated but whether it can be modified by that activation. in todorov, every weight is equally modifiable during training -- there is no per-synapse gating of plasticity. the d-serine mechanism suggests that a neural computer could benefit from a learned plasticity gate: a per-parameter signal that determines whether a given parameter should be updated on this training step, based on local network state. this is related to but distinct from gradient masking or selective unfreezing.

confidence: high. clear causal demonstration with rescue experiment. the nmda receptor's requirement for a co-agonist is well-established. caveat: the relative contributions of d-serine (from astrocytes) vs glycine (from neurons) as nmda co-agonist vary by brain region and synapse type.

## gliotransmitter debate

### subpopulation-specific gliotransmission

de ceglia, r. et al. (2023). specialized astrocytes mediate glutamatergic gliotransmission in the cns. *nature*, 622, 120-129.

key finding: the longstanding debate about whether astrocytes release gliotransmitters (some groups could replicate it, others could not) was partially resolved by the discovery that only specific subpopulations of astrocytes in specific brain regions are gliotransmission-competent. using single-cell rna sequencing and calcium imaging, this study identified molecularly distinct astrocyte subtypes with different gliotransmitter release capabilities. this explains the field's replication problems: results depend on which astrocyte subpopulation is studied, and bulk approaches average across functionally distinct types.

relevance to neural computer: the astrocyte subpopulation finding parallels the observation that not all neurons are functionally equivalent -- there are excitatory, inhibitory, and modulatory subtypes with different computational roles. extending this to the glial level means the brain has at least 2x more cell type diversity than neuron-only catalogs suggest. todorov uses a single neuron model (the ternary spike) applied uniformly. the biological reality of multiple specialized cell types performing different modulatory functions suggests that architectural diversity -- different processing modes at different locations in the network -- may be more important than uniform application of a single mechanism.

confidence: medium-high. the subpopulation specificity is a new finding that resolves prior contradictions, but the full catalog of astrocyte subtypes and their functions is still being mapped. caveat: the study focused on sensory cortex; whether the same subtypes exist in other regions is being investigated.

## myelin as learning

### activity-dependent myelination

mckenzie, i. a., ohayon, d., li, h., de faria, j. p., emery, b., tohyama, k., & richardson, w. D. (2014). motor skill learning requires active central nervous system myelination. *science*, 346(6207), 318-322.

key finding: blocking new myelin formation (by preventing oligodendrocyte precursor cell differentiation) in adult mice prevented motor skill learning without affecting previously learned skills or general motor function. this demonstrates that new myelination is required for new learning, not just for maintaining existing function. the learning-associated myelination was specific to the circuits involved in the learned skill.

relevance to neural computer: myelination adjusts signal conduction velocity along axons, which changes the timing of signal arrival at downstream neurons. this is a form of learning that operates on the temporal domain rather than the synaptic strength domain -- the brain learns not just what to transmit but when signals should arrive. todorov has no analog of conduction velocity adjustment. all signals propagate instantaneously through matrix multiplications. the biological finding that timing adjustment is essential for learning suggests that temporal precision in signal propagation may be computationally important in ways that synchronous forward passes cannot capture.

confidence: high. causal manipulation (genetic block of new myelination) with clear behavioral effect (learning impaired, existing skills preserved). caveat: the genetic manipulation was global, not circuit-specific; the learning deficit may partly reflect non-specific effects of blocked oligodendrogenesis.

### myelin and brain function

fields, r. D. (2015). a new mechanism of nervous system plasticity: activity-dependent myelination. *nature reviews neuroscience*, 16, 756-767.

key finding: comprehensive review establishing that myelination is an activity-dependent form of plasticity. action potentials in axons signal to oligodendrocyte precursor cells (opcs) via vesicular release of glutamate, prompting opcs to differentiate and begin myelinating active axons. the resulting myelin sheath increases conduction velocity ~100x (from ~1 m/s unmyelinated to ~100 m/s myelinated) and adjusts signal timing with sub-millisecond precision. this creates a feedback loop: neural activity drives myelination, which changes conduction velocity, which changes spike timing, which changes neural activity patterns. fields argues this constitutes a second learning system operating on a slower timescale (days to weeks) than synaptic plasticity (seconds to hours).

relevance to neural computer: the two-timescale learning system (fast synaptic plasticity + slow myelination) suggests that optimal neural computation requires adjusting both connection strength and connection timing. todorov's two-timescale architecture (fast kda recurrence + slow mamba3 continuous dynamics) captures the multi-timescale principle but implements it through different recurrence equations rather than through synaptic strength vs conduction velocity. the 100x speed increase from myelination is also relevant to the architecture's efficiency: biology invests heavily in making frequently used pathways faster.

confidence: high. review synthesizing extensive evidence for activity-dependent myelination. the basic mechanisms (axonal signaling to opcs, activity-dependent differentiation) are well-established. caveat: the computational significance of myelination timing adjustments relative to synaptic plasticity is difficult to quantify -- both contribute to learning, but their relative importance varies by task and circuit.

## microglial synapse pruning

### complement-mediated synapse elimination

stevens, b. et al. (2007). the classical complement cascade mediates cns synapse elimination. *cell*, 131(6), 1164-1178.

key finding: microglia eliminate synapses during development via the classical complement cascade. weak or inactive synapses are tagged with complement proteins (c1q, c3), which serve as "eat me" signals for microglia. microglia recognize these tags via complement receptor cr3 and phagocytose the tagged synapses. this molecular mechanism explains how the brain selectively eliminates ~50% of synapses during postnatal development while preserving the strongest and most active connections. knockout of c1q or c3 results in excessive synaptic connectivity and epileptic activity.

relevance to neural computer: complement-mediated pruning is a biological implementation of magnitude-based weight pruning: weak connections are tagged for removal, strong connections are preserved. todorov's ternary spike quantization performs a related function -- activations near zero are quantized to zero, effectively pruning weak signals. but this operates on activations, not weights. a weight-side analog would involve identifying and removing weights that contribute minimally to the network's output, which is standard network pruning. the biological mechanism adds specificity: pruning is not just magnitude-based but also activity-dependent (inactive synapses are pruned regardless of their weight).

confidence: high. the complement-microglia pruning pathway is well-established and has been linked to neurodevelopmental and neurodegenerative disorders. caveat: complement-mediated pruning is best characterized during development; its role in adult synaptic maintenance is less clear.

### microglia refine circuits through activity-dependent pruning

schafer, d. p., lehrman, e. k., kautzman, a. g., et al. (2012). microglia sculpt postnatal neural circuits in an activity and complement-dependent manner. *neuron*, 74(4), 691-705.

key finding: using high-resolution imaging of the developing visual system, this study showed that microglia preferentially engulf synapses from the less active eye during ocular dominance development. when neural activity was blocked in one eye, microglia increased phagocytosis of that eye's synapses while sparing the active eye's synapses. this demonstrates that microglial pruning is not random but precisely targeted based on relative neural activity -- it implements a competitive, activity-dependent selection process.

relevance to neural computer: the activity-dependent targeting of microglial pruning is a competitive mechanism: synapses from different inputs compete for survival based on their activity levels. this is structurally similar to attention mechanisms that weight inputs by their relevance (activity). in todorov, the mla mechanism selects which tokens to attend to based on query-key similarity. microglial pruning selects which connections to retain based on activity history. the difference is timescale: attention operates per-token, pruning operates over developmental time.

confidence: high. elegant experimental design combining activity manipulation with high-resolution imaging of microglial engulfment. caveat: the retinogeniculate system is a model system; the degree to which this mechanism generalizes to cortical synapse pruning is being investigated.

### microglia mediate forgetting

wang, c. et al. (2020). microglia mediate forgetting via complement-dependent synaptic elimination. *science*, 367(6478), 688-694.

key finding: microglia actively mediate forgetting of established memories in adult mice through complement-dependent synapse elimination. inhibiting microglial phagocytosis or complement signaling in the hippocampus preserved memories that would otherwise be forgotten over time. this demonstrates that forgetting is not simply passive decay of synaptic connections but an active, microglia-driven process that removes specific synapses encoding specific memories. the same complement pathway used for developmental pruning is repurposed in adults for memory clearance.

relevance to neural computer: active forgetting -- the deliberate removal of stored information -- has no analog in todorov. model weights accumulate information during training and never actively discard it (except through interference from new learning). the biological finding that forgetting is an active, targeted process suggests that a neural computer could benefit from a mechanism that identifies and removes outdated or irrelevant stored associations from the kda state. this connects to the neurogenesis-forgetting tradeoff (see [[neurogenesis_and_plasticity_research]]): both neurogenesis and microglial pruning contribute to active forgetting, suggesting that controlled information removal is a design feature, not a failure mode.

confidence: high. causal manipulation (microglial inhibition, complement knockout) with clear memory preservation effect. caveat: the specificity of microglial targeting -- how microglia identify which synapses encode which memories -- is not fully understood.

## oligodendrocyte precursor cells

### opcs receive synaptic input

bergles, d. e., roberts, j. d. b., somogyi, p., & jahr, c. e. (2000). glutamatergic synapses on oligodendrocyte precursor cells in the hippocampus. *nature*, 405, 187-191.

key finding: oligodendrocyte precursor cells (opcs) -- the progenitor cells that differentiate into myelinating oligodendrocytes -- receive direct synaptic input from neurons. these are bona fide synapses with vesicular release, postsynaptic receptors, and measurable synaptic currents. this was unexpected because opcs were considered passive precursors waiting to differentiate, not active participants in neural circuits. the synaptic input from neurons to opcs provides a mechanism for activity-dependent myelination: opcs can "listen" to neural activity and preferentially myelinate active axons.

relevance to neural computer: opcs represent a class of cells that are not currently computing but are monitoring the network to decide where to deploy new resources (myelin). this is a form of meta-computation: computing about computation. todorov has no analog -- there is no monitoring system that observes which pathways are most active and allocates additional resources to them. a biologically inspired approach might include a monitoring mechanism that tracks which layers or attention heads are most heavily used and dynamically allocates more compute (e.g., more heads, wider layers) to bottlenecked pathways.

confidence: high. electrophysiological recording from identified opcs in hippocampal slices. replicated across brain regions. caveat: the functional significance of neuron-to-opc synapses for myelination decisions in vivo is established but the signaling pathway is still being mapped.

## relevance to todorov

### validated connections
- the 1:1 glia:neuron ratio means biological computation uses 2x the cellular machinery that neuron-only models assume
- complement-mediated synapse pruning is a biological implementation of magnitude-based and activity-dependent weight pruning, paralleling ternary spike quantization of weak activations to zero
- active forgetting (microglial memory clearance) validates the idea that controlled information removal improves system performance

### challenged assumptions
- todorov has no tripartite synapse analog -- every weight is a simple scalar, not a three-party interaction with modulatory state
- no per-synapse plasticity gating (astrocytic d-serine control of ltp has no analog in uniform gradient updates)
- no conduction velocity learning (myelination adjusts timing, which has no analog in synchronous matrix multiplications)
- no meta-computation (opc monitoring of network activity and resource allocation)

### future phases
- per-parameter plasticity gating: learned gates that control which parameters are updated on each training step (inspired by astrocytic d-serine)
- active forgetting mechanism: targeted removal of low-utility associations from kda state
- timing-aware computation: mechanisms that model signal propagation delays (phase 6+, requires fundamental architecture changes)
- resource monitoring: tracking which layers/heads are bottlenecked and dynamically reallocating capacity

## see also

- [[synaptic_pruning]]
- [[critical_periods]]
- [[homeostatic_plasticity]]
- [[excitatory_inhibitory_balance]]
- [[brain_energy_budget]]
- [[neurogenesis_and_plasticity_research]]
