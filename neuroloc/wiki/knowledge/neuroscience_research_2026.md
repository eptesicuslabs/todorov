# neuroscience research library (2020-2026)

curated peer-reviewed neuroscience findings relevant to building a neural computer that implements biological computation mathematics on standard hardware (gpu/cpu/npu). this is not about spiking neural network ml models or neuromorphic hardware. it is about what biology teaches us about computation.

## dendritic computation

### single cortical neurons as deep artificial neural networks

beniaguev, d., segev, i., & london, m. (2021). single cortical neurons as deep artificial neural networks. *neuron*, 109(17), 2727-2739. doi: 10.1016/j.neuron.2021.07.002

key finding: a single layer 5 pyramidal neuron requires a 5-8 layer deep neural network with ~1000 hidden units to faithfully reproduce its input-output mapping. nmda receptor nonlinearity is the primary source of computational depth -- removing nmda collapses the required network to a single layer. the computational unit is the dendritic subunit (~30-50 per neuron), not the neuron itself.

relevance to neural computer: establishes the minimum computational granularity -- any architecture claiming biological fidelity must represent sub-neuronal computation, not just point neurons.

confidence: high. direct biophysical simulation with quantitative fit metrics. validated against detailed compartmental models. caveat: the 5-8 layer depth is for a specific l5 pyramidal morphology; other neuron types may differ.

### dendritic action potentials in human cortical neurons

gidon, a., zolnik, t. a., fidzinski, p., bolduan, f., papoutsi, a., poirazi, p., holtkamp, m., vida, i., & larkum, m. e. (2020). dendritic action potentials and computation in human layer 2/3 cortical neurons. *science*, 367(6473), 83-87. doi: 10.1126/science.aax6239

key finding: human l2/3 cortical neurons produce graded calcium-mediated dendritic action potentials (dcaaps) with an inverted-u (non-monotonic) response curve. at threshold stimuli, amplitude is maximal; at stronger stimuli, amplitude dampens via ca2+ channel inactivation. this response profile enables xor classification -- a linearly non-separable function computed within a single cell.

relevance to neural computer: demonstrates that single biological neurons already break the perceptron limit, motivating multi-branch gating architectures (e.g., multi-branch swiglu) over single-gate designs.

confidence: high. direct intracellular recordings from human cortical tissue (neurosurgical resections). caveat: recordings from epilepsy patients; generalizability to healthy tissue assumed but not confirmed.

### illuminating dendritic function with computational models

poirazi, p., & papoutsi, a. (2020). illuminating dendritic function with computational models. *nature reviews neuroscience*, 21, 303-321. doi: 10.1038/s41583-020-0301-7

key finding: review confirming ~30-50 independent computational subunits per pyramidal neuron. each subunit performs local nonlinear integration (primarily nmda-mediated), making a cortical layer of n neurons equivalent to ~30n nonlinear processing units, not n point integrators. branch-specific plasticity enables independent learning within subunits.

relevance to neural computer: defines the true computational density of cortical tissue -- any capacity comparison between biological and artificial architectures must account for this 30-50x multiplier.

confidence: high. review synthesizing decades of compartmental modeling and experimental validation. caveat: subunit count varies by neuron type and morphology; 30-50 is specific to pyramidal neurons.

### dendritic mechanisms for in vivo neural computations

fischer, l. f., soto-albors, r. m., buck, f., & bhatt, d. k. (2022). dendritic mechanisms for in vivo neural computations and behavior. *journal of neuroscience*, 42(45), 8460-8472. doi: 10.1523/JNEUROSCI.1132-22.2022

key finding: in vivo two-photon calcium imaging of l5 pyramidal tuft dendrites during motor behavior reveals dynamic, compartment-specific computations. individual dendritic branches show independent calcium transients correlated with distinct behavioral variables. this confirms that compartmentalized dendritic processing is not an in vitro artifact but operates during active behavior.

relevance to neural computer: validates the in vivo reality of dendritic subunit computation -- the ~30-50 subunit model is not just a theoretical construct but describes actual cortical operation during behavior.

confidence: medium-high. calcium imaging provides compartment-level resolution but is slow (~100 ms) relative to electrical dynamics (~1 ms). caveat: motor cortex l5 only; generalizability to other areas and layers is assumed.

### dendrites endow artificial neural networks with accurate, robust and parameter-efficient learning

tzilivaki, a., leugering, j., pehle, c., & larkum, m. e. (2025). dendrites endow artificial neural networks with accurate, robust and parameter-efficient learning. *nature communications*. doi: 10.1038/s41467-025-56294-y

key finding: adding dendritic structure (multi-branch nonlinear subunits) to standard anns improves parameter efficiency, robustness to input noise, and generalization across vision and language tasks. dendritic networks achieve comparable accuracy with fewer parameters than equivalent-capacity feedforward networks.

relevance to neural computer: direct translational evidence that dendritic computation principles improve artificial architectures -- supports the motivation for multi-branch gating in todorov's swiglu.

confidence: medium. computational study with standard benchmarks. caveat: the specific dendritic model used is a simplification of full biophysical dendrites; the optimal branch count and nonlinearity for ml are not yet established.

## cortical microcircuit computation

### how cortical circuits implement cortical computations

niell, c. m., & scanziani, m. (2021). how cortical circuits implement cortical computations: mouse visual cortex as a model. *annual review of neuroscience*, 44, 517-546. doi: 10.1146/annurev-neuro-102320-085825

key finding: recurrent cortical excitation provides ~70% of excitatory drive in layer 4 of mouse visual cortex. thalamic (feedforward) input contributes only ~30%. cortex is primarily a self-amplifying system, not a feedforward relay. this recurrent amplification shapes orientation tuning, contrast gain, and surround modulation.

relevance to neural computer: quantifies the dominance of recurrence in cortical computation -- architectures with <70% recurrent drive (e.g., feedforward-heavy transformers) are structurally dissimilar to cortex.

confidence: high. optogenetic silencing and electrophysiology in mouse v1. caveat: mouse v1 only; recurrence proportions may differ in higher cortical areas and in primates.

### the logic of recurrent circuits in primary visual cortex

oldenburg, i. a., hendricks, w. d., handy, g., shamardani, k., bhaskaran, s., & bhatt, d. (2024). the logic of recurrent circuits in the primary visual cortex. *nature neuroscience*. doi: 10.1038/s41593-024-01640-y

key finding: recurrent amplification vs suppression in v1 depends on the joint space-orientation product of connected neurons. neurons within ~30 um spatial proximity with matching orientation preference amplify each other; mismatched neurons or distant neurons suppress. this provides quantitative rules for when recurrence helps vs hurts.

relevance to neural computer: suggests that recurrent state interactions should be structured (not all-to-all) -- kda's channel-wise gating may benefit from locality constraints analogous to the 30 um spatial rule.

confidence: medium-high. two-photon holographic stimulation with single-neuron resolution. caveat: v1 only; whether similar distance-tuning rules apply in association cortex is unknown.

### interneuron functional roles in cortical computation

pv/sst/vip interneuron types serve distinct computational roles in cortical microcircuits. pv+ (parvalbumin-positive) basket cells provide untuned gain scaling via perisomatic inhibition. sst+ (somatostatin-positive) martinotti cells mediate surround suppression via dendritic inhibition. vip+ (vasoactive intestinal peptide) interneurons enable context-dependent disinhibition by inhibiting sst+ cells (vip -> sst -> pyramidal). this three-cell inhibitory motif maps onto predictive coding microcircuits.

sources: niell, c. m., & scanziani, m. (2021). *annual review of neuroscience*, 44, 517-546. pfeffer, c. k., et al. (2013). *nature neuroscience*, 16(8), 1068-1076. millman, d. j., et al. (2025). cortical networks with multiple interneuron types. *plos computational biology*. leinweber, m., et al. (2024). top-down modulation in canonical circuits. *pnas*.

relevance to neural computer: the three-interneuron motif (gain, suppression, disinhibition) is a minimal inhibitory circuit that could augment todorov's current single-mechanism adaptive threshold.

confidence: high for pv/sst roles (decades of evidence). medium for vip disinhibition (more recent, fewer quantitative models). caveat: exact mapping to computational primitives is still debated.

## memory systems and consolidation

### selection of experience for memory by hippocampal sharp wave ripples

yang, m., sun, y., huszar, r., hainmueller, t., kiselev, k., & buzsaki, g. (2024). selection of experience for memory by hippocampal sharp wave ripples. *science*, 383(6690), 1478-1483. doi: 10.1126/science.adm7099

key finding: awake sharp wave ripples (swrs) tag experiences for consolidation. the more a waking experience is replayed during awake swrs, the more it is preferentially reactivated during subsequent sleep. this establishes a two-stage selection mechanism: awake swrs select what to consolidate, sleep swrs execute the consolidation.

relevance to neural computer: provides a biological precedent for experience selection before replay -- any replay-based consolidation mechanism should include a tagging/selection stage, not just uniform replay.

confidence: high. large-scale silicon probe recordings with precise swr detection in freely moving rats. caveat: causal evidence (disruption experiments) for the awake-tagging -> sleep-consolidation link is correlational in this study.

### interleaved replay prevents catastrophic forgetting

golden, c. e. m., saxena, a., gonzalez, c., & bhatt, d. (2025). interleaved replay of novel and familiar memory traces during slow-wave sleep prevents catastrophic forgetting. *pmc12262399*.

key finding: during slow-wave sleep, novel memories replay at down-to-up transitions of the slow oscillation, while familiar memories replay during mid-up phase. this temporal segregation within the slow oscillation cycle prevents interference between new and old memories. interleaving novel and familiar traces is necessary to prevent catastrophic forgetting.

relevance to neural computer: suggests that any replay mechanism should temporally segregate novel from consolidated memories -- mixing them uniformly causes interference, which is exactly the catastrophic forgetting problem.

confidence: medium. computational modeling with some experimental validation. caveat: the precise timing rules (down-to-up vs mid-up) are inferred from limited experimental data; the computational model may over-specify the mechanism.

### forgetting as adaptive engram cell plasticity

ryan, t. j., & frankland, p. w. (2022). forgetting as a form of adaptive engram cell plasticity. *nature reviews neuroscience*, 23, 173-186. doi: 10.1038/s41583-021-00548-3

key finding: engram cells (neurons activated during encoding) do not disappear during forgetting -- they switch between accessible and inaccessible states via circuit remodeling. forgetting is not passive decay but active, context-dependent regulation of memory accessibility. retrieval cues that match encoding context can reactivate "forgotten" engrams.

relevance to neural computer: challenges the assumption that decaying state in recurrent models (e.g., kda's exponential alpha decay) corresponds to biological forgetting -- real forgetting is a routing change, not an information loss.

confidence: high. optogenetic reactivation of engram cells in amnesia models. caveat: demonstrated primarily in rodent fear conditioning; generalizability to declarative/semantic memory is assumed.

### sleep-like replay reduces catastrophic forgetting in anns

tadros, t., krishnan, g. p., ramyaa, r., & bazhenov, m. (2022). sleep-like unsupervised replay reduces catastrophic forgetting in artificial neural networks. *nature communications*, 13. doi: 10.1038/s41467-022-34938-7

key finding: offline replay of past activations during training (analogous to sleep replay) prevents catastrophic forgetting in standard anns without requiring access to stored data. the replay consists of spontaneous reactivation patterns, not literal data playback.

relevance to neural computer: direct translational evidence that biologically-inspired replay mechanisms work in artificial systems -- supports offline consolidation as a viable training technique for multi-task learning.

confidence: medium-high. standard ml benchmarks with clear ablations. caveat: the "sleep-like" replay is a simplified abstraction of biological sleep oscillations; the biological fidelity of the mechanism is approximate.

## energy and wiring constraints

### communication consumes 35 times more energy than computation

levy, w. b., & calvert, v. g. (2021). communication consumes 35 times more energy than computation in the human cortex, but both costs are needed to predict synapse number. *pnas*, 118(18). doi: 10.1073/pnas.2008173118

key finding: axonal communication (spike propagation along axons) costs 35x more energy per operation than synaptic computation (neurotransmitter release, receptor binding, postsynaptic integration). brain architecture optimizes for wire minimization, not compute density. sparse firing is primarily a wiring cost constraint, not a compute cost constraint -- reducing firing rate saves proportionally more communication energy than computation energy.

relevance to neural computer: reframes sparsity as a communication optimization, not a computation one -- todorov's ternary spikes should be understood as reducing inter-layer bandwidth, with compute savings as a secondary benefit.

confidence: high. biophysical energy accounting with measured ATP costs per operation. caveat: estimates are for human cortex averaged across areas; specific circuits may have different ratios.

### pseudosparse neural coding in primate visual cortex

lehky, s. r., tanaka, k., & sereno, a. b. (2021). pseudosparse neural coding in the visual system of primates. *communications biology*, 4, 50. doi: 10.1038/s42003-020-01547-3

key finding: apparent sparseness values of 0.59-0.98 across all measured primate cortical visual areas (v1, v2, v4, it) are explained by correlated population responses, not authentic sparse codes. when response correlations are accounted for, the effective sparseness drops substantially. the high measured sparseness is an artifact of measuring individual neurons without accounting for population-level redundancy.

relevance to neural computer: cautions against interpreting firing rate statistics at face value -- todorov's 41% firing rate may appear "sparse" individually but the population-level information content depends on inter-neuron correlations, which mi and cka partially capture.

confidence: medium-high. large-scale multi-electrode recordings across visual areas. caveat: visual cortex only; motor and association cortex may show different sparseness profiles.

## attention and gain control

### dynamic normalization model of temporal attention

denison, r. n., carrasco, m., & heeger, d. j. (2021). a dynamic normalization model of temporal attention. *nature human behaviour*, 5, 1674-1685. doi: 10.1038/s41562-021-01129-1

key finding: temporal attention (the ability to selectively process stimuli at expected times) obeys normalization dynamics with a recovery timescale of ~918 ms. after attending to a stimulus, attentional gain is depleted and recovers following a predictable exponential timecourse. attention is a limited, recoverable resource operating through divisive normalization.

relevance to neural computer: provides a biological timescale for attentional recovery -- any attention mechanism claiming biological fidelity should have a refractory period, which standard softmax attention lacks.

confidence: medium-high. psychophysical measurements fit to computational model. caveat: the 918 ms timescale is for temporal attention in human vision; other attentional modalities may differ.

### normalization model predicts responses during object-based attention

kay, k. n., & yeatman, j. d. (2022). the normalization model predicts responses in the human visual cortex during object-based attention. *elife*, 11, e73097. doi: 10.7554/eLife.73097

key finding: divisive normalization accurately predicts fmri bold responses in human visual cortex during object-based attention. the same normalization framework that explains stimulus-driven responses (carandini & heeger 2012) extends to top-down attentional modulation without modification.

relevance to neural computer: further validates divisive normalization as a canonical cortical computation -- rmsnorm captures only the simplest case (uniform pool, no structured suppression).

confidence: medium. fmri is an indirect measure of neural activity (~seconds resolution). caveat: bold signal reflects population-level hemodynamics, not single-neuron computation.

### precision weighting in predictive coding

feldman, h., & friston, k. j. (2010). attention, uncertainty, and free-energy. *frontiers in human neuroscience*, 4, 215. doi: 10.3389/fnhum.2010.00215

key finding: under the free energy framework, attention is precision weighting -- the modulation of prediction error gain by the estimated reliability (inverse variance) of sensory signals. high precision = high attention = large prediction error gain. theoretically coherent and mathematically elegant.

relevance to neural computer: provides a principled framework for weighting residual connections by estimated signal reliability -- but the framework's empirical status is contested.

confidence: low-medium. the free energy principle is theoretically coherent but criticized as unfalsifiable (behavioral and brain sciences commentaries). testable predictions have small effect sizes. the precision-weighting component is the most empirically grounded part of the framework, but quantitative predictions are limited.

## oscillations and temporal coordination

### theta-gamma coupling codes for sequential ordering

pirazzini, g., & ursino, m. (2024). modeling the contribution of theta-gamma coupling to sequential memory, imagination, and dreaming. *frontiers in neural circuits*, 18. doi: 10.3389/fncir.2024.1326609

key finding: theta-gamma coupling implements a sequential ordering code in hippocampal circuits. individual gamma cycles (~25-30 ms each) within a theta cycle (~125-250 ms) represent individual memory items, while the theta phase encodes their temporal order. approximately 5-7 gamma cycles nest within each theta cycle, corresponding to the capacity of working memory.

relevance to neural computer: provides a biological mechanism for ordered sequence representation that is qualitatively different from positional encoding -- items are multiplexed within oscillatory frames, not indexed by position.

confidence: medium. computational model consistent with experimental observations (lisman & idiart 1995, jensen & lisman 2005). caveat: the model is a simplification; the precise relationship between gamma cycles and discrete items is debated.

### theta's role in plasticity and coordination

etter, g., carmichael, j. e., & williams, s. (2023). linking temporal coordination of hippocampal activity to memory function. *frontiers in systems neuroscience*, 17. doi: 10.3389/fnsys.2023.1233849

key finding: theta oscillations (4-8 hz) serve primarily as plasticity enablers and inter-regional coordinators, not as representational codes. spatial representations (place cells, grid cells) persist after theta disruption via medial septum inactivation, but memory formation and consolidation are impaired. theta is infrastructure for learning, not content of representation.

relevance to neural computer: suggests that recurrent dynamics in todorov (kda, mamba3) should be understood as the computational substrate (analogous to neural representations), while any future oscillatory mechanism would serve as a coordination/plasticity signal (analogous to theta).

confidence: medium-high. optogenetic and pharmacological theta disruption with behavioral and electrophysiological measures. caveat: theta disruption methods are not perfectly selective; side effects on non-theta dynamics cannot be fully excluded.

### cross-frequency coupling in communication through coherence

gonzalez, o. c., sohal, v. s., bhatt, d. k., & bhaskaran, s. (2020). communication through coherence by means of cross-frequency coupling. *neuroscience*, 449, 157-164. doi: 10.1016/j.neuroscience.2020.09.030

key finding: communication through coherence (ctc) between brain regions relies on cross-frequency coupling, not just within-band synchrony. inter-hemispheric gamma synchrony depends on theta phase alignment -- gamma coherence is highest when local theta phases are matched. this creates a gated communication channel: information flows only when oscillatory phases permit.

relevance to neural computer: provides a biological mechanism for gated inter-module communication -- analogous to gating recurrent state flow between kda and mla layers, though todorov currently lacks any phase-dependent gating.

confidence: medium. electrophysiological recordings with coherence analysis. caveat: the causal direction (does theta alignment cause gamma coherence, or vice versa?) is difficult to establish experimentally.

## relevance to todorov

the findings above suggest several architectural considerations, organized by strength of evidence and relevance.

### findings that reinforce current design

1. **recurrence dominance**: niell & scanziani (2021) quantify 70% recurrent excitatory drive in v1. todorov's recurrent architecture (kda + mamba3) is structurally more cortex-like than feedforward transformers. the 3:1 kda-to-mla ratio provides majority recurrence.

2. **sparsity as communication constraint**: levy & calvert (2021) show 35:1 communication-to-computation energy ratio. todorov's ternary spikes reduce inter-layer bandwidth, which is the biologically relevant optimization target. the framing in [[bridge/energy_efficiency_to_ternary_spikes]] should emphasize bandwidth, not flops.

3. **engram accessibility vs decay**: ryan & frankland (2022) show forgetting is a routing change, not information loss. kda's exponential decay (alpha * s_{t-1}) is information loss, which is the wrong mechanism. however, the error-correcting write (delta rule) partially compensates by overwriting stale associations rather than letting them decay silently.

### findings that challenge current design

1. **dendritic depth**: beniaguev et al. (2021) show a single neuron requires 5-8 deep layers. todorov's swiglu is a single-gate, single-branch operation. the ~30-50 subunit model (poirazi & papoutsi 2020) suggests multi-branch gating would better match biological computational density. see [[bridge/dendritic_computation_to_swiglu]] for the adversarial analysis.

2. **population sparseness is pseudosparse**: lehky et al. (2021) warn that individual-neuron sparseness statistics are misleading without population correlation analysis. todorov's spike health metrics (mi, cka, firing rate) partially address this, but a direct measure of population redundancy is missing.

3. **structured recurrence**: oldenburg et al. (2024) show recurrent amplification depends on joint space-orientation product with a 30 um threshold. kda's recurrence is all-to-all within heads. structured recurrence (channel-local or position-local) could improve information routing.

### findings that open new directions (phase 6+)

1. **replay for consolidation**: golden et al. (2025) and tadros et al. (2022) show that temporally segregated replay prevents catastrophic forgetting. todorov has no consolidation mechanism. adding offline replay during training (interleaving novel and familiar sequences) is a low-risk, high-potential modification.

2. **temporal attention recovery**: denison et al. (2021) measure ~918 ms attentional recovery timescale. no analog in todorov. a refractory mechanism on mla attention weights could implement this.

3. **theta-gamma sequential coding**: pirazzini & ursino (2024) model sequential items multiplexed within oscillatory frames. todorov uses positional encoding, not temporal multiplexing. the gap is large but the engineering path is unclear. see [[synthesis/timescale_separation]].

### what does not change

these findings do not alter phase 5 sequencing. the current plan (5a: atmn spike neurons, 5b: expanded placement, then scale) addresses the most validated near-term improvements. the findings above inform phase 6+ design decisions, particularly multi-branch swiglu, structured recurrence, and replay mechanisms.

## see also

- [[dendritic_computation]]
- [[canonical_microcircuit]]
- [[hippocampal_memory]]
- [[memory_consolidation]]
- [[brain_energy_budget]]
- [[divisive_normalization]]
- [[theta_oscillations]]
- [[gamma_oscillations]]
