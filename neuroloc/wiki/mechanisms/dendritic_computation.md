# dendritic computation

**why this matters**: dendrites turn each biological neuron into a multi-branch nonlinear processor -- not a point neuron. this means a cortical layer has 30-50x more nonlinear subunits than point neuron models assume, with direct implications for how much computational capacity each layer of an artificial network should have.

## status
[DRAFT]
last updated: 2026-04-06
sources: 8 papers, 1 textbook

## biological description

**dendrites** (the branching input structures of neurons that receive and process synaptic signals) are not passive cables. for decades, the dominant view treated dendrites as simple conduits that collect synaptic input and deliver it to the soma for integration. this view is wrong. dendrites are active computational elements that perform local, nonlinear transformations of their inputs before the soma ever sees the result. a single pyramidal neuron is not a point processor -- it is a spatially distributed, multi-compartment computer.

London and Hausser (2005) systematically reviewed the evidence that dendrites implement elementary computations including:

1. **supralinear summation** (a response that exceeds the linear sum of individual inputs)**.** when synaptic inputs arrive on the same dendritic branch within a narrow spatial and temporal window (~20-50 um, ~5-10 ms), the resulting **depolarization** (a shift in membrane voltage toward positive values, increasing the probability of firing) can exceed the linear sum of individual inputs. this is caused by regenerative dendritic spikes (see [[dendritic_spikes]]) -- local, active events mediated by voltage-gated Na+, Ca2+, and NMDA receptor channels. supralinear summation turns each thin dendrite into a nonlinear integration unit, equivalent to a hidden unit in a neural network (see [[two_layer_neuron]]).

2. **sublinear summation.** when many inputs activate the same dendritic region simultaneously, local conductance increases can reduce driving force (shunting), producing a net depolarization smaller than the linear sum. this implements a form of divisive normalization (see [[divisive_normalization]]) at the single-branch level, providing automatic gain control for dense input regimes.

3. **coincidence detection.** the narrow temporal window for triggering dendritic spikes (~5-10 ms for Na+/NMDA, ~10-100 ms for Ca2+) means dendrites naturally detect temporal coincidences among their inputs. this is a form of AND gate: two inputs arriving separately produce subthreshold responses, but arriving together they trigger a supralinear spike. the BAC firing mechanism in L5 pyramidal neurons (see [[apical_amplification]]) is the most dramatic example: coincident bottom-up and top-down inputs produce a burst, while either alone produces only single spikes or subthreshold responses.

4. **directional selectivity.** the passive cable properties of dendrites (spatial decay, temporal filtering) combined with the distribution of active conductances create sensitivity to the direction and sequence of input activation along a dendritic branch. inputs sweeping from distal to proximal can generate different responses than proximal-to-distal, because the propagation of dendritic spikes interacts differently with the voltage gradient in each case. this has been demonstrated in retinal ganglion cells and cortical pyramidal neurons.

5. **compartmentalized processing.** the thin diameter of terminal dendrites (~0.5-2 um) creates high input resistance, which means that even small synaptic currents produce large local voltage changes. combined with the spatial attenuation of voltage along dendrites (length constant ~200-400 um for passive propagation), this creates semi-independent processing compartments. a pyramidal neuron with 10-40 basal and oblique dendrites operates as 10-40 parallel processing units, each performing its own nonlinear integration before contributing to the somatic sum.

6. **local plasticity.** different dendritic branches can undergo independent synaptic plasticity (see [[stdp]], [[hebbian_learning]]). the local voltage, calcium concentration, and neuromodulatory environment at each branch determine whether synapses are strengthened or weakened. this means learning is compartmentalized: one branch can potentiate while another depresses, based on the correlation structure of their respective inputs.

the computational implication is that a cortical layer of N pyramidal neurons, each with ~30 dendritic branches, provides not N processing units but ~30N nonlinear subunits feeding into N output units. this is the basis of the two-layer neuron model (see [[two_layer_neuron]]).

## mathematical formulation

the passive dendritic cable is described by the cable equation:

    tau_m * dV/dt = lambda^2 * d^2V/dx^2 - (V - V_rest) + r_m * I_syn(x, t)

where:
- V(x, t) = membrane potential at position x along the dendrite
- tau_m = R_m * C_m (membrane time constant, ~10-30 ms)
- lambda = sqrt(R_m * d / (4 * R_i)) (electrotonic length constant, ~200-400 um)
- R_m = specific membrane resistance (~10,000-40,000 ohm*cm^2)
- R_i = axial resistance (~100-200 ohm*cm)
- C_m = specific membrane capacitance (~1 uF/cm^2)
- d = dendrite diameter (~0.5-2 um for terminal branches, ~3-8 um for main trunk)
- I_syn(x, t) = synaptic current density at position x

the cable equation gives purely passive (linear) integration. active dendritic computation arises from adding voltage-gated conductances:

    tau_m * dV/dt = lambda^2 * d^2V/dx^2 - (V - V_rest) + r_m * I_syn(x, t) + r_m * I_active(V, x, t)

where I_active includes Na+, Ca2+, K+, and NMDA conductances with their own gating dynamics (see [[hodgkin_huxley]] for the general framework of voltage-gated conductance).

for the two-layer neuron approximation (see [[two_layer_neuron]]):

    f_soma = sigma_out(sum_j sigma_j(w_j^T * x_j))

where:
- x_j = vector of synaptic inputs to dendritic branch j
- sigma_j = dendritic branch nonlinearity (sigmoidal or rectified, depending on channel types)
- w_j = synaptic weights on branch j
- sigma_out = somatic output nonlinearity (spike threshold)

this reduces a full multi-compartment biophysical model to a two-layer neural network with branch-specific nonlinearities as hidden units.

## evidence strength

STRONG. dendritic computation is supported by:

1. direct dendritic recordings (patch-clamp from individual dendrites, pioneered by Stuart & Sakmann 1994, Stuart et al. 1997). these demonstrate local dendritic spikes, supralinear summation, and compartmentalized integration in vitro and in vivo.

2. calcium imaging of dendritic activity (Helmchen et al. 1999, Svoboda et al. 1997). two-photon microscopy shows branch-specific calcium transients during sensory stimulation, confirming compartmentalized processing in intact animals.

3. glutamate uncaging experiments (Losonczy & Magee 2006, Branco & Hausser 2011). targeted activation of specific synaptic inputs on individual branches demonstrates supralinear summation with ~3-5 input spatial requirement and ~5 ms temporal window.

4. computational modeling (Poirazi et al. 2003, Mel 1994, Polsky et al. 2004). detailed biophysical models of pyramidal neurons reproduce the two-layer network prediction and confirm that dendritic nonlinearities expand computational capacity beyond point neuron models.

5. in vivo demonstrations (Smith et al. 2013, Xu et al. 2012). dendritic calcium spikes correlated with behavior have been recorded in awake, behaving animals.

## challenges and counter-arguments

1. **most evidence comes from in vitro preparations.** the majority of dendritic recording studies use brain slices, where the network context, neuromodulatory environment, and ongoing activity are absent. in vivo dendritic recordings are technically difficult and relatively sparse. the computational principles demonstrated in vitro may be modulated or overridden by network-level dynamics in the intact brain.

2. **the computational role of dendritic spikes in vivo remains debated.** while dendritic calcium events have been observed in vivo (Xu et al. 2012, Smith et al. 2013), their causal contribution to behavior is not always clear. optogenetic suppression of dendritic spikes in specific behavioral contexts is needed to establish necessity, and such experiments are technically challenging.

3. **the two-layer neuron model is an approximation.** the mapping from biophysical compartments to abstract neural network layers assumes that dendritic branches are independent subunits. in reality, branches share voltage through the main trunk, interact through passive coupling, and are subject to global somatic feedback (backpropagating action potentials). the degree to which branches are truly independent depends on dendritic morphology, active conductance density, and input statistics.

4. **dendritic morphology is highly variable.** different cell types (L2/3 pyramidal, L5 pyramidal, L6 corticothalamic, interneurons) have radically different dendritic trees. the computational principles derived from L5 pyramidal neurons (the most studied type) may not generalize to all cell types. L2/3 pyramidal neurons have shorter apical dendrites and different channel distributions.

5. **point neuron models often suffice for network-level predictions.** despite the richness of dendritic computation, many network phenomena (oscillations, working memory, decision-making) can be reproduced with point neuron models. this does not mean dendritic computation is unimportant, but it does mean the additional complexity is not always necessary for the phenomena of interest.

## computational implications

1. **the neuron is not a perceptron.** a perceptron computes a single weighted sum followed by a threshold. a pyramidal neuron with active dendrites computes multiple nonlinear weighted sums (one per branch) followed by a linear combination and threshold. this is strictly more powerful -- a single neuron can solve linearly non-separable problems like XOR (see [[dendritic_spikes]] on Gidon et al. 2020).

2. **dendritic gating is a form of multiplicative interaction.** the coincidence detection at dendritic branch points (requiring both presynaptic glutamate AND sufficient local depolarization for NMDA unblock or dendritic spike initiation) is a biological AND gate. mathematically, this is a multiplicative interaction: output = f(input_1 * input_2).

ML analog: this is the same class of operation as gating in LSTM gates, GRU gates, and SwiGLU multiplicative gating. see [[dendritic_computation_to_swiglu]] for the bridge analysis.

3. **dendritic computation enables credit assignment.** Sacramento et al. (2018) showed that cortical microcircuits with segregated apical and basal dendritic compartments can approximate the backpropagation algorithm using only local learning rules. the apical dendrite receives top-down error signals, the basal dendrite receives feedforward input, and the mismatch between them drives synaptic plasticity. this provides a biologically plausible solution to the credit assignment problem.

4. **compartmentalized processing enables multiplexing.** because different dendritic branches can process different inputs independently, a single neuron can simultaneously represent information about multiple features or input streams. the somatic output can encode one type of information (e.g., firing rate for feedforward features) while bursts (driven by dendritic calcium spikes) encode another (e.g., match between top-down prediction and bottom-up input). see [[apical_amplification]].

## bridge to todorov

todorov's architecture does not explicitly implement multi-compartment dendritic computation. the closest mechanism is SwiGLU's multiplicative gating:

    out = W_down(silu(W_gate(x)) * W_up(x))

where the gate path (silu(W_gate(x))) and value path (W_up(x)) are analogous to two dendritic branches performing independent linear transformations, with the multiplication acting as a form of coincidence detection. however, this analogy has significant limitations:

- SwiGLU uses ONE gate for the entire hidden dimension; dendrites have 10-40 INDEPENDENT branches
- SwiGLU receives one input (the residual stream x); dendrites receive inputs from DIFFERENT sources (feedforward, feedback, lateral)
- SwiGLU's nonlinearity (silu) is smooth and monotonic; dendritic spikes are regenerative, threshold-crossing events

see [[dendritic_computation_to_swiglu]] for the full adversarial analysis.

what exists: multiplicative gating in SwiGLU approximates single-branch coincidence detection.
what is missing: multi-branch independence, source-segregated inputs, regenerative nonlinearities, branch-specific plasticity.
what matters for 300M scale: the two-layer neuron model suggests that adding multiple independent nonlinear pathways per "neuron" could increase expressiveness without proportional parameter cost. this is a phase 6+ consideration.

## related mechanisms

- [[dendritic_spikes]] -- the active mechanisms (Na+, Ca2+, NMDA) that generate dendritic nonlinearities
- [[apical_amplification]] -- BAC firing as coincidence detection between bottom-up and top-down signals
- [[two_layer_neuron]] -- Poirazi et al. 2003, the pyramidal neuron as a two-layer neural network
- [[hodgkin_huxley]] -- the general framework for voltage-gated conductances
- [[divisive_normalization]] -- sublinear summation as local gain control
- [[inhibitory_interneurons]] -- SST+ interneurons target dendrites, controlling dendritic computation

## open questions

1. does multi-branch dendritic computation provide a measurable advantage over point neuron models for language modeling? the two-layer neuron theory predicts yes (more nonlinear capacity per parameter), but language may not require the spatial/temporal coincidence detection that dendrites are optimized for.

2. could todorov's SwiGLU be extended to multiple independent gating branches (multi-head SwiGLU) to capture the computational benefit of dendritic branching? this would increase the hidden dimension but also increase parameter count. the cost-benefit tradeoff is unclear.

3. the dendritic credit assignment framework (Sacramento et al. 2018) shows that segregated dendritic compartments enable backprop-like learning with local rules. could this inform a new training procedure for todorov that is more biologically plausible than standard backpropagation? the Millidge et al. result (see [[predictive_coding_to_training_objective]]) suggests backprop already computes equivalent gradients, so the benefit would be architectural, not algorithmic.

4. the NMDA receptor's coincidence detection (requiring both presynaptic glutamate AND postsynaptic depolarization to relieve the Mg2+ block) is the canonical biological AND gate. SwiGLU's silu*linear multiplication is a softer version of this. would a harder gate (e.g., threshold on both paths) improve spike information content?

## source bibliography

- London, M. and Hausser, M. (2005). Dendritic computation. Annual Review of Neuroscience, 28, 503-532.
- Stuart, G.J. and Sakmann, B. (1994). Active propagation of somatic action potentials into neocortical pyramidal cell dendrites. Nature, 367(6458), 69-72.
- Stuart, G., Spruston, N., Sakmann, B., and Hausser, M. (1997). Action potential initiation and backpropagation in neurons of the mammalian CNS. Trends in Neurosciences, 20(3), 125-131.
- Losonczy, A. and Magee, J.C. (2006). Integrative properties of radial oblique dendrites in hippocampal CA1 pyramidal neurons. Neuron, 50(2), 291-307.
- Branco, T. and Hausser, M. (2011). Synaptic integration gradients in single cortical pyramidal cell dendrites. Neuron, 69(5), 885-892.
- Mel, B.W. (1994). Information processing in dendritic trees. Neural Computation, 6(6), 1031-1085.
- Sacramento, J., Costa, R.P., Bengio, Y., and Senn, W. (2018). Dendritic cortical microcircuits approximate the backpropagation algorithm. NeurIPS, 8721-8732.
- Larkum, M. (2013). A cellular mechanism for cortical associations: an organizing principle for the cerebral cortex. Trends in Neurosciences, 36(3), 141-151.
