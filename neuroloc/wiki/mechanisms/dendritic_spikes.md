# dendritic spikes

status: definitional. last fact-checked 2026-04-16.

**why this matters**: dendritic spikes enable single neurons to compute XOR and other linearly non-separable functions -- breaking the perceptron limit. this means biological neurons are strictly more powerful than the point neurons used in standard neural networks.

## status
[DRAFT]
last updated: 2026-04-06
sources: 10 papers

## biological description

**dendritic spikes** (regenerative, active electrical events generated locally within dendrites by **voltage-gated ion channels** (ion channels that open and close in response to membrane voltage changes)) are NOT propagated from the soma -- they originate in the dendrites themselves and travel toward the soma (or remain local). three types exist, each with different biophysical mechanisms and computational properties.

### sodium spikes (dendritic Na+ spikes)

location: primarily in basal dendrites and proximal apical trunk, near the soma.
mechanism: voltage-gated Na+ channels (same molecular type as axonal channels, Nav1.2/Nav1.6, but at lower density).
duration: fast (~1-2 ms), similar to axonal action potentials.
amplitude: ~40-80 mV locally, attenuated by distance from initiation site.
propagation: can propagate toward soma but often fail at branch points (branch point failure).
threshold: relatively low (~-50 to -40 mV local depolarization).
computational role: fast amplification of synaptic inputs, sharpening of temporal coincidence detection. because Na+ spikes are fast, they respond to inputs that are near-synchronous within ~1-2 ms. this makes them the fastest coincidence detectors in dendrites.

**backpropagating action potentials** (**bAPs**, somatic Na+ spikes that propagate backward from the soma into the dendrites) are related but distinct. they do not originate in dendrites but they modulate dendritic processing by depolarizing the dendritic tree, lowering the threshold for dendritic Ca2+ spikes (see BAC firing in [[apical_amplification]]).

### calcium spikes (dendritic Ca2+ spikes)

location: primarily in the apical tuft and the calcium spike initiation zone at the top of the apical trunk (~300-800 um from soma in L5 pyramidal neurons).
mechanism: voltage-gated Ca2+ channels (L-type Cav1.2/Cav1.3, T-type Cav3.1/Cav3.2, R-type Cav2.3).
duration: long (~10-100 ms), much slower than Na+ spikes.
amplitude: ~30-60 mV at the initiation zone, substantially attenuated at the soma (~5-15 mV somatic depolarization depending on distance and active conductances).
propagation: broad, can invade the entire apical tuft. does not propagate well in the forward direction (toward soma) through passive cable, but can trigger burst firing through BAC mechanism.
threshold: high -- requires either strong synaptic input to the tuft OR a coincident backpropagating action potential (BAC firing).
computational role: the primary mechanism for associating top-down feedback (arriving in L1 at the apical tuft) with bottom-up sensory drive (arriving at basal dendrites and soma). the long duration of Ca2+ spikes creates a sustained depolarization that can trigger multiple somatic action potentials (a burst), encoding a qualitatively different signal from single spikes. see [[apical_amplification]].

Gidon et al. (2020) discovered a novel type of dendritic Ca2+ action potential (**dCaAP**) in human L2/3 cortical neurons that has graded amplitude. unlike typical all-or-none action potentials, dCaAP amplitude is MAXIMAL at threshold-level stimuli and DAMPENED for stronger stimuli. this creates an inverted-U response curve (XOR-like): weak input produces no spike, moderate input produces a maximal spike, and strong input suppresses the spike through Ca2+ channel inactivation. this enables a single neuron to compute XOR, a **linearly non-separable function** (a classification problem that cannot be solved by a single hyperplane). standard neural networks require at least two layers to solve XOR. the biophysical mechanism is Ca2+ channel inactivation at high depolarization levels.

### NMDA spikes (NMDA receptor-mediated plateau potentials)

location: primarily in thin basal and oblique dendrites (terminal branches ~50-200 um from soma).
mechanism: NMDA receptors (NMDARs, specifically GluN2A/GluN2B subunits). NMDARs are both ligand-gated AND voltage-gated: they require glutamate binding AND sufficient local depolarization to relieve the Mg2+ block of the channel pore. this dual requirement makes them natural coincidence detectors.
duration: very long (~50-100 ms for pure NMDA spike, up to ~200-500 ms when Ca2+ channels are co-activated, creating plateau potentials).
amplitude: ~40-50 mV locally, with a sevenfold spatial gradient in somatic amplitude from distal (~3 mV) to proximal (~23 mV) inputs (Nevian et al. 2007).
propagation: strictly local -- the highest spatial confinement of any dendritic spike type. NMDA spikes in one thin dendrite do NOT propagate to neighboring branches. this is the physical basis for branch-independent computation.
threshold: ~10-50 simultaneously active synapses within ~20-50 um of dendritic length, arriving within ~5-10 ms.
computational role: the NMDA spike is the primary mechanism for dendritic subunit computation. each thin dendrite can function as an independent sigmoidal unit (see [[two_layer_neuron]]), and NMDA spikes are the nonlinearity. because NMDA receptors require both presynaptic glutamate (evidence of presynaptic activity) and postsynaptic depolarization (evidence of local dendritic activity), they implement a biological AND gate. the long duration of NMDA plateau potentials also supports working memory: a single dendritic plateau can sustain depolarization for hundreds of milliseconds, maintaining a local representation without ongoing input.

Major & Tank (2004) and Schiller et al. (2000) first characterized NMDA spikes in basal dendrites. the decade 2004-2014 established NMDA spikes as the primary mechanism for multisite independent subunit computation in pyramidal neurons (Major et al. 2013, "The decade of the dendritic NMDA spike").

## mathematical formulation

### Na+ dendritic spike (simplified)

uses Hodgkin-Huxley gating variables adapted for dendritic Na+ channels:

    I_Na_dend = g_Na_dend * m^3 * h * (V_dend - E_Na)

where g_Na_dend ~ 10-40 mS/cm^2 (lower density than axon's 120 mS/cm^2), and m, h follow voltage-dependent kinetics (see [[hodgkin_huxley]]).

### Ca2+ dendritic spike (simplified)

    I_Ca = g_Ca * s^2 * r * (V_dend - E_Ca)

where g_Ca ~ 1-5 mS/cm^2, s is activation (slow, tau_s ~ 5-20 ms), r is inactivation (very slow, tau_r ~ 50-200 ms), and E_Ca ~ +120 mV. the slow inactivation of r underlies the graded dCaAP discovered by Gidon et al. (2020): at high depolarization, r decreases, dampening the spike amplitude.

### NMDA spike (simplified)

    I_NMDA = g_NMDA * B(V) * (V - E_NMDA)

where:
- g_NMDA = total NMDA conductance (depends on number of glutamate-bound receptors)
- B(V) = 1 / (1 + [Mg2+]_o/3.57 * exp(-0.062 * V)) is the voltage-dependent Mg2+ block
- E_NMDA ~ 0 mV (mixed cation reversal potential)

the B(V) function creates a region of negative slope conductance: as V increases from resting, the Mg2+ block is relieved, increasing conductance, which further depolarizes the dendrite, which further relieves the block. this positive feedback loop is what makes NMDA spikes regenerative. the spike terminates when glutamate unbinds or when K+ channels activate.

### somatic integration of branch outputs

    V_soma = V_rest + sum_j g_transfer_j * f_j(I_syn_j)

where:
- j indexes dendritic branches
- g_transfer_j = transfer conductance from branch j to soma (depends on cable properties)
- f_j = branch nonlinearity (sigmoidal for NMDA-rich branches, step-like for Na+-rich branches, broad for Ca2+-rich branches)
- I_syn_j = total synaptic input to branch j

## evidence strength

STRONG. all three types of dendritic spikes have been directly recorded and characterized:

1. dendritic Na+ spikes: Stuart et al. (1997), Golding & Spruston (1998). direct dendritic patch-clamp recordings demonstrate fast, local Na+ spikes in apical and basal dendrites.

2. dendritic Ca2+ spikes: Schiller et al. (1997), Larkum et al. (1999), Larkum (2013). dendritic Ca2+ imaging and dual soma-dendrite recordings demonstrate sustained Ca2+ events in the apical tuft, triggered by coincident back-propagating APs and distal input (BAC firing).

3. NMDA spikes: Schiller et al. (2000), Major et al. (2004, 2008, 2013), Polsky et al. (2004). glutamate uncaging and dendritic recordings demonstrate local NMDA-dependent plateau potentials in basal and oblique dendrites. the decade of the dendritic NMDA spike (Major et al. 2013) reviews the full evidence.

4. human cortical neurons: Gidon et al. (2020). dendritic recordings from human L2/3 pyramidal neurons (from surgical tissue) demonstrate a novel class of calcium-mediated dendritic action potentials with graded, dampened amplitude enabling XOR computation.

## challenges and counter-arguments

1. **dendritic spike types interact.** the three types are not independent -- a local NMDA spike can trigger a dendritic Na+ spike, which can interact with a backpropagating AP to trigger a Ca2+ spike in the apical tuft. this cascade of events makes it difficult to attribute specific computations to specific spike types in natural conditions.

2. **channel distributions vary across cell types and species.** the prominence of each spike type depends on the density and distribution of the underlying ion channels, which varies across cortical layers, brain regions, and species. L5 pyramidal neurons have the most prominent Ca2+ spikes; L2/3 pyramidal neurons have different Ca2+ channel distributions (as shown by Gidon et al. 2020). generalizing from one cell type or species to another is risky.

3. **the Gidon et al. XOR result is from human tissue only.** the graded dCaAP enabling XOR computation was demonstrated in human L2/3 neurons but has not been replicated in other species. it may reflect a species-specific adaptation of human cortical neurons rather than a general principle. however, independent modeling studies have confirmed the biophysical plausibility of the mechanism.

4. **in vivo relevance.** the sharp dendritic spike thresholds measured in vitro may be blurred by ongoing synaptic bombardment in vivo. background synaptic activity can both lower thresholds (through tonic depolarization) and increase variability (through membrane potential fluctuations), potentially reducing the reliability of dendritic spike-based computation.

## computational implications

1. **each dendritic branch is a hidden unit.** NMDA spikes convert thin dendrites into sigmoidal nonlinear units. a pyramidal neuron with 30 thin dendrites is computationally equivalent to a two-layer network with 30 hidden units (see [[two_layer_neuron]]). this means a single neuron has more computational power than a **perceptron** (a single-layer linear classifier followed by a threshold).

ML analog: each dendritic branch functions like a separate head in multi-head attention -- an independent nonlinear computation on a subset of inputs, whose outputs are combined at the soma.

2. **three spike types = three temporal filters.** Na+ spikes detect fast coincidences (~1 ms), NMDA spikes detect intermediate coincidences (~10-50 ms), and Ca2+ spikes detect slow coincidences (~50-100 ms). this multi-timescale coincidence detection is analogous to multi-scale attention in transformers, but operating within a single neuron rather than across a layer.

3. **the XOR capability (Gidon et al. 2020) breaks the single-neuron perceptron limit.** a single human L2/3 neuron can solve problems that require a hidden layer in a perceptron. this has immediate relevance for understanding the computational capacity of cortical circuits: if each neuron is already a two-layer network, then a cortical layer of N neurons has the capacity of a 2*N-layer network (in a loose sense). this suggests cortical circuits may be far more computationally powerful than point neuron models predict.

4. **NMDA plateau potentials are a form of working memory.** a dendritic plateau lasting ~100-500 ms can maintain a local depolarization without ongoing input, effectively storing information at the single-dendrite level. this is distinct from somatic working memory (sustained firing) and network working memory (attractor dynamics). it operates at a timescale intermediate between synaptic transmission (~1-10 ms) and short-term plasticity (~100-1000 ms).

## bridge to todorov

todorov does not implement dendritic spikes. the ternary spike mechanism (see [[sparse_coding_to_ternary_spikes]]) is a somatic-level threshold operation that does not distinguish between dendritic sources or implement branch-specific nonlinearities.

the closest analog to dendritic spike computation in todorov is the multiplicative gating in SwiGLU (see [[dendritic_computation_to_swiglu]]):
- silu(W_gate(x)) * W_up(x) is a multiplicative interaction, like NMDA coincidence detection
- but SwiGLU applies ONE gate to the entire hidden dimension, not independent gates per branch

what exists: threshold-based ternary quantization (somatic level).
what is missing: branch-level nonlinearities, multi-type spike dynamics, spatially structured input integration.
what matters: the XOR result (Gidon et al. 2020) suggests that adding branch-level nonlinearities could increase per-neuron computational capacity. whether this matters at 300M scale (where the network has millions of "neurons") is an open question.

## related mechanisms

- [[dendritic_computation]] -- the broader framework of dendritic information processing
- [[apical_amplification]] -- BAC firing and the role of Ca2+ spikes in cortical associations
- [[two_layer_neuron]] -- Poirazi et al. 2003, how dendritic spikes make a neuron a two-layer network
- [[hodgkin_huxley]] -- the general framework for voltage-gated ion channels
- [[canonical_microcircuit]] -- the cortical circuit context in which dendritic spikes operate

## open questions

1. do the three dendritic spike types map to distinct computational roles in neural coding, or are they simply different implementations of the same basic nonlinearity (threshold followed by saturation)? the timescale differences (1 ms vs 50 ms vs 100 ms) suggest distinct functional roles, but this has not been definitively established.

2. the Gidon et al. (2020) dCaAP result suggests human neurons may have unique computational properties not present in rodent neurons. are there other species-specific dendritic mechanisms that expand computational capacity?

3. could a hierarchical spike model (NMDA spike -> Na+ spike -> Ca2+ spike -> burst) be captured by a cascade of activation functions in an artificial neuron? this would map naturally to a multi-layer transformation within a single "neuron unit."

4. NMDA plateau potentials last ~100-500 ms. in a language model processing tokens at ~50 ms intervals, this corresponds to 2-10 tokens of persistent dendritic memory. would implementing per-branch persistent state at this timescale improve context-dependent processing in todorov?

## source bibliography

- Schiller, J., Schiller, Y., Stuart, G., and Sakmann, B. (1997). Calcium action potentials restricted to distal apical dendrites of rat neocortical pyramidal neurons. Journal of Physiology, 505(3), 605-616.
- Schiller, J., Major, G., Koester, H.J., and Schiller, Y. (2000). NMDA spikes in basal dendrites of cortical pyramidal neurons. Nature, 404(6775), 285-289.
- Major, G. and Tank, D. (2004). Persistent neural activity: prevalence and mechanisms. Current Opinion in Neurobiology, 14(6), 675-684.
- Major, G., Larkum, M.E., and Schiller, J. (2013). Active properties of neocortical pyramidal neuron dendrites. Annual Review of Neuroscience, 36, 1-24.
- Polsky, A., Mel, B.W., and Schiller, J. (2004). Computational subunits in thin dendrites of pyramidal cells. Nature Neuroscience, 7(6), 621-627.
- Gidon, A., Zolnik, T.A., Fidzinski, P., Bolduan, F., Papoutsi, A., Poirazi, P., Holtkamp, M., Vida, I., and Larkum, M.E. (2020). Dendritic action potentials and computation in human layer 2/3 cortical neurons. Science, 367(6473), 83-87.
- Larkum, M.E., Zhu, J.J., and Sakmann, B. (1999). A new cellular mechanism for coupling inputs arriving at different cortical layers. Nature, 398(6725), 338-341.
- Stuart, G., Spruston, N., Sakmann, B., and Hausser, M. (1997). Action potential initiation and backpropagation in neurons of the mammalian CNS. Trends in Neurosciences, 20(3), 125-131.
- Golding, N.L. and Spruston, N. (1998). Dendritic sodium spikes are variable triggers of axonal action potentials in hippocampal CA1 pyramidal neurons. Neuron, 21(5), 1189-1200.
- Nevian, T., Larkum, M.E., Polsky, A., Schiller, J. (2007). Properties of basal dendrites of layer 5 pyramidal neurons: a direct patch-clamp recording study. Nature Neuroscience, 10(2), 206-214.
