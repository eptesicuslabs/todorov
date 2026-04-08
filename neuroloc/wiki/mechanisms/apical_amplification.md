# apical amplification

**why this matters**: apical amplification is a biological implementation of context-dependent gating -- bottom-up input specifies WHAT a neuron responds to while top-down input specifies WHETHER the response is amplified. this is functionally equivalent to multiplicative attention and motivates architectures with separate content and context pathways.

## status
[DRAFT]
last updated: 2026-04-06
sources: 7 papers

## biological description

**layer 5 (L5) pyramidal neurons** (the largest excitatory neurons in cortex, serving as the primary output to subcortical structures) have the longest and most elaborate **apical dendrites** (the main upward-extending dendritic branch of pyramidal neurons) in the cortex, extending from the soma in deep layers (L5) up through the cortical column to reach layer 1 (L1) at the cortical surface. this morphological fact has a profound computational consequence. the **apical tuft** (the terminal branching of the apical dendrite in L1) receives top-down feedback from higher cortical areas. the **basal dendrites** (short dendrites extending from the base of the soma) and soma in L5 receive bottom-up feedforward input from L4 and thalamus. the neuron physically spans the cortical column, with its two ends receiving fundamentally different information streams.

Larkum et al. (1999) discovered that L5 pyramidal neurons have TWO distinct integration zones:

1. **the somatic/perisomatic zone.** receives feedforward synaptic input to basal dendrites. integrates this input using standard mechanisms (temporal summation, threshold crossing, Na+ spike generation). this is the "bottom-up" integration site.

2. **the calcium spike initiation zone (CaIZ).** located at the top of the apical trunk, ~300-800 um from the soma, at the junction between the main trunk and the apical tuft branches. this zone has a high density of voltage-gated Ca2+ channels (L-type and T-type). it integrates synaptic input arriving at the apical tuft from higher cortical areas (top-down feedback) and from nonspecific thalamic nuclei. this is the "top-down" integration site.

ML analog: the two-zone architecture is analogous to a cross-attention mechanism where queries come from one pathway (bottom-up) and keys/values from another (top-down).

### BAC firing

backpropagation-activated calcium spike (BAC) firing is the mechanism by which these two integration zones communicate. the process has four steps:

1. a somatic action potential (driven by bottom-up input) is generated at the axon hillock.
2. this action potential propagates backward into the apical dendrite as a backpropagating action potential (bAP). the bAP is actively propagated by dendritic Na+ channels but attenuates with distance.
3. the bAP arrives at the CaIZ. if, AND ONLY IF, the CaIZ is simultaneously depolarized by top-down synaptic input to the apical tuft, the bAP and the top-down depolarization COINCIDE to trigger a dendritic calcium spike.
4. the calcium spike propagates back toward the soma, producing a large, sustained depolarization (~30-50 mV, lasting ~50-100 ms) that triggers a BURST of somatic action potentials (typically 2-5 spikes in rapid succession).

the critical insight: neither the bottom-up input alone nor the top-down input alone produces a burst. the bottom-up input produces a single action potential. the top-down input to the apical tuft produces subthreshold depolarization at the soma (too far away, too much attenuation). only when BOTH arrive within a narrow temporal window (~10-100 ms) does the neuron fire a burst.

### apical amplification vs apical drive

Larkum (2013) distinguished two modes of apical-somatic interaction:

**apical amplification:** top-down apical input is subthreshold on its own but amplifies the somatic response to bottom-up input. the neuron responds to feedforward features, but its response is modulated (amplified) by context. this is the default mode during normal sensory processing.

**apical drive:** top-down apical input is strong enough on its own to trigger a calcium spike and somatic burst, even without bottom-up input. this represents an internally generated signal overriding sensory input. Larkum proposed this as the mechanism for hallucinations, mental imagery, and dreaming.

### inhibitory control

BAC firing is tightly regulated by inhibitory interneurons:

- **SST+ (somatostatin) interneurons** target the apical dendrites specifically. they can gate BAC firing by inhibiting the CaIZ, preventing top-down input from triggering calcium spikes even when bottom-up input is present. this is a veto mechanism for context (see [[inhibitory_interneurons]]).

- **VIP+ (vasoactive intestinal peptide) interneurons** inhibit SST+ interneurons (disinhibition). when VIP+ neurons are active, SST+ inhibition of apical dendrites is released, ENABLING BAC firing. this creates a three-level control: VIP+ -> SST+ -> apical dendrite -> BAC firing.

- **PV+ (parvalbumin) interneurons** target the soma and axon initial segment. they control the bottom-up firing rate without directly affecting the apical integration zone.

this inhibitory microcircuit (VIP -> SST -> dendrite) provides precise subcellular control over the coincidence detection mechanism, allowing the network to gate which neurons are allowed to perform apical amplification.

## mathematical formulation

a minimal two-compartment model of BAC firing:

somatic compartment:

    C_s * dV_s/dt = -g_L * (V_s - V_rest) - I_Na(V_s) - I_K(V_s) + g_c * (V_d - V_s) + I_basal

apical compartment:

    C_d * dV_d/dt = -g_L * (V_d - V_rest) - I_Ca(V_d) + g_c * (V_s - V_d) + I_apical

where:
- V_s, V_d = somatic and apical membrane potentials
- g_c = coupling conductance between compartments (depends on apical trunk diameter and length)
- I_basal = bottom-up synaptic input to soma/basal dendrites
- I_apical = top-down synaptic input to apical tuft
- I_Na, I_K = somatic Na+ and K+ currents (for action potential generation)
- I_Ca = dendritic Ca2+ current (for calcium spike generation)

the Ca2+ current follows:

    I_Ca = g_Ca * s^2 * r * (V_d - E_Ca)

    ds/dt = (s_inf(V_d) - s) / tau_s
    dr/dt = (r_inf(V_d) - r) / tau_r

where s is fast activation (tau_s ~ 5 ms) and r is slow inactivation (tau_r ~ 50-200 ms). the BAC condition is met when V_d exceeds the Ca2+ spike threshold due to COMBINED effects of (1) passive depolarization from bAP (arriving via g_c coupling) and (2) synaptic depolarization from I_apical.

the output is detected by the pattern of somatic action potentials:
- isolated spike: bottom-up alone, no apical coincidence
- burst (2-5 spikes in ~10-30 ms): BAC firing, coincident bottom-up and top-down

## evidence strength

STRONG. the BAC firing mechanism has been demonstrated by:

1. dual soma-dendrite recordings in vitro (Larkum et al. 1999). simultaneous patch-clamp recordings from the soma and apical dendrite of L5 pyramidal neurons directly measured the bAP arriving at the CaIZ and the resulting calcium spike. this is the foundational experiment.

2. calcium imaging in vivo (Xu et al. 2012, Takahashi et al. 2016). two-photon calcium imaging of apical dendrites in awake mice shows dendritic calcium events correlated with specific behavioral states.

3. pharmacological dissection. blocking Ca2+ channels in the apical tuft with cadmium or nifedipine abolishes BAC firing while preserving isolated somatic spikes. blocking Na+ channels in the apical trunk with TTX prevents bAP propagation and eliminates BAC firing triggered by bottom-up input.

4. SST+ interneuron inhibition of BAC firing (Murayama et al. 2009, Gentet et al. 2012). optogenetic activation of SST+ interneurons suppresses dendritic calcium spikes in L5 pyramidal neurons, confirming the inhibitory gating mechanism.

## challenges and counter-arguments

1. **BAC firing is best characterized in L5 pyramidal neurons.** L2/3 pyramidal neurons have shorter apical dendrites and different Ca2+ channel distributions. the two-compartment model with distinct integration zones may not apply equally to all pyramidal neuron types. Gidon et al. (2020) showed that human L2/3 neurons have a different type of dendritic Ca2+ spike (graded dCaAPs, not BAC firing), suggesting that the specific mechanism differs across cell types even if the principle of dendritic computation generalizes.

2. **the timing window for BAC firing is debated.** Larkum et al. (1999) reported effective BAC firing with bAP-apical input delays of ~5-50 ms. later studies (Shai et al. 2015) found longer effective windows (~100 ms). the functional timing window in vivo, where background activity fluctuates the membrane potential, is not precisely established.

3. **apical amplification vs predictive coding.** Larkum (2013) proposed apical amplification as "the missing cellular mechanism for cortical associations" and connected it to predictive coding (see [[predictive_coding]]). the mapping is: top-down predictions arrive at apical dendrites in L1, bottom-up sensory input arrives at basal dendrites, and BAC firing signals a match (prediction confirmed). this is elegant but speculative -- the same mechanism could implement attention (apical input = attentional gain signal) without invoking predictive coding.

4. **thalamic contribution to L1 input.** not all L1 input is top-down cortical feedback. nonspecific thalamic nuclei also project to L1, providing arousal-related and salience signals. the computational interpretation of apical input depends on which source dominates in different behavioral states.

## computational implications

1. **BAC firing is a cellular implementation of context-dependent gating.** the bottom-up input specifies WHAT the neuron responds to (its receptive field features). the top-down input specifies WHETHER the neuron's response is amplified (contextual relevance). this is functionally equivalent to multiplicative attention: output = bottom_up * f(top_down).

ML analog: BAC firing maps to SwiGLU's multiplicative gating (silu(gate) * value), but with a critical difference -- in biology, the gate and value paths receive DIFFERENT inputs (top-down and bottom-up), while SwiGLU receives the same input on both paths.

2. **bursts carry a different message than isolated spikes.** Lisman (1997) proposed that bursts and isolated spikes encode different information channels. in the BAC firing framework, isolated spikes represent feedforward sensory evidence; bursts represent feedforward evidence PLUS top-down confirmation. downstream neurons can read bursts and single spikes differently, enabling multiplexed communication (see [[dendritic_spikes]]).

3. **apical amplification provides a mechanism for binding.** the binding problem asks how distributed features (color, shape, motion) are bound into coherent percepts. BAC firing provides a cellular answer: features represented by basal dendrite input are "bound" to context/predictions represented by apical input. when they coincide, the burst signals a bound representation. this does not solve the binding problem entirely but provides a cellular mechanism that could contribute.

4. **the VIP -> SST -> dendrite pathway is a learned attention gate.** if VIP+ interneurons learn to respond to specific contextual signals (which brain region provides the relevant context), they can selectively enable BAC firing in specific pyramidal neurons, implementing learned selective attention at the cellular level.

## bridge to todorov

todorov does not implement apical amplification or two-source input integration. the residual stream provides a single input to each layer; there is no distinction between feedforward and feedback pathways within a layer.

the closest analog in todorov:

SwiGLU's multiplicative gating (silu(W_gate(x)) * W_up(x)) has the formal structure of coincidence detection -- two paths must both produce high values for the output to be large. however, both paths receive the SAME input x (the residual stream). in BAC firing, the two integration zones receive DIFFERENT inputs (bottom-up and top-down). this is a fundamental structural difference: SwiGLU is self-gating, while BAC firing is cross-source gating.

the most direct way to implement apical-like computation in todorov would be to provide each SwiGLU layer with TWO inputs: one from the residual stream (bottom-up) and one from a later layer's output (top-down feedback). this would require feedback connections, breaking the purely feedforward architecture. see [[dendritic_computation_to_swiglu]] for full analysis.

what exists: multiplicative gating (SwiGLU) with single-source input.
what is missing: dual-source input (feedforward + feedback), burst-coding output, inhibitory gating of the coincidence mechanism.
what matters: the apical amplification framework suggests that context-dependent gating with separate context and content pathways could improve representation quality. whether this matters more than simply adding parameters to a feedforward architecture is the empirical question.

## related mechanisms

- [[dendritic_computation]] -- the broader framework of dendritic information processing
- [[dendritic_spikes]] -- the three types of dendritic spikes, including the Ca2+ spikes underlying BAC firing
- [[two_layer_neuron]] -- how dendritic computation makes a neuron a two-layer network
- [[predictive_coding]] -- the theoretical framework that apical amplification may implement
- [[inhibitory_interneurons]] -- the SST+ and VIP+ interneurons that gate BAC firing
- [[laminar_processing]] -- the cortical layer organization that makes apical amplification possible
- [[canonical_microcircuit]] -- the broader cortical circuit in which apical amplification operates

## open questions

1. could a two-input SwiGLU (one from residual stream, one from a later layer's output) capture the computational benefit of apical amplification without full feedback connections? this would add ~d_model^2 parameters per layer for the feedback pathway and create a dependency between layers that complicates training.

2. Larkum's framework predicts that anesthesia (which suppresses top-down feedback while preserving feedforward processing) should selectively abolish BAC firing. electrophysiological evidence supports this (Suzuki & Larkum 2020). does this mean a model without feedback connections is "anesthetized" -- processing feedforward input without contextual modulation?

3. the burst vs single spike distinction could be mapped to a multi-valued output in todorov: ternary spikes (+1, 0, -1) could be extended to a quinary code (+2, +1, 0, -1, -2) where +2/-2 represents a "burst" (strong confidence, context-confirmed) and +1/-1 represents a single spike (weak evidence). would this improve information capacity?

4. SST+ interneurons targeting apical dendrites create layer-type-specific inhibition (they inhibit top-down integration while leaving feedforward processing intact). could a similar mechanism in todorov -- where different mask values apply to different subspaces of the hidden dimension -- implement source-specific gating?

## source bibliography

- Larkum, M.E., Zhu, J.J., and Sakmann, B. (1999). A new cellular mechanism for coupling inputs arriving at different cortical layers. Nature, 398(6725), 338-341.
- Larkum, M. (2013). A cellular mechanism for cortical associations: an organizing principle for the cerebral cortex. Trends in Neurosciences, 36(3), 141-151.
- Lisman, J.E. (1997). Bursts as a unit of neural information: making unreliable synapses reliable. Trends in Neurosciences, 20(3), 38-43.
- Shai, A.S., Anastassiou, C.A., Larkum, M.E., and Koch, C. (2015). Physiology of layer 5 pyramidal neurons in mouse primary visual cortex: coincidence detection through bursting. PLoS Computational Biology, 11(3), e1004090.
- Murayama, M., Perez-Garci, E., Nevian, T., Bock, T., Senn, W., and Larkum, M.E. (2009). Dendritic encoding of sensory stimuli controlled by deep cortical interneurons. Nature, 457(7233), 1137-1141.
- Suzuki, M. and Larkum, M.E. (2020). General anesthesia decouples cortical pyramidal neurons. Cell, 180(4), 666-676.
- Xu, N.L., Harnett, M.T., Williams, S.R., Huber, D., O'Connor, D.H., Svoboda, K., and Magee, J.C. (2012). Nonlinear dendritic integration of sensory and motor input during an active sensing task. Nature, 492(7428), 247-251.
