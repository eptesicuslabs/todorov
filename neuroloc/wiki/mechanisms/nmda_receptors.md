# nmda receptors

**why this matters**: the nmda receptor is the brain's native multiplicative gate -- it requires simultaneous presynaptic glutamate AND postsynaptic depolarization to open, implementing a biological AND operation. this coincidence detection is the biophysical basis of hebbian learning ("fire together, wire together") and maps directly to the multiplicative gating used in swiglu and kda channel-wise gates in todorov.

## status
[DRAFT]
last updated: 2026-04-07
sources: 12 papers

## biological description

**nmda receptors** (**N-methyl-D-aspartate receptors**, a class of ionotropic glutamate receptors named after their selective agonist) are ligand-gated and voltage-gated ion channels that require two simultaneous conditions to conduct current: presynaptic glutamate binding and postsynaptic depolarization. this dual requirement makes them the primary coincidence detectors in the mammalian brain.

### structure

nmda receptors are **heterotetramers** (protein complexes composed of four subunits of at least two different types). the obligatory subunit is **GluN1**, which binds the co-agonist glycine (or D-serine). the second subunit type is **GluN2** (subtypes A, B, C, D), which binds glutamate and determines the receptor's kinetic properties. the standard synaptic nmda receptor is a diheteromeric GluN1/GluN2 assembly -- two GluN1 subunits plus two GluN2 subunits arranged around a central ion pore. **triheteromeric** receptors (GluN1/GluN2A/GluN2B) also exist and are increasingly recognized as the dominant form at mature synapses.

the GluN2 subunit identity determines the receptor's biophysical fingerprint: decay time, Mg2+ block affinity, calcium permeability, and sensitivity to modulators. this makes GluN2 subunit composition a tuning parameter for synaptic computation.

### voltage-dependent Mg2+ block: the biological AND gate

at resting membrane potential (~-70 mV), a **Mg2+ ion** (magnesium) physically occludes the nmda receptor channel pore, preventing ion flow even when glutamate is bound. depolarization of the postsynaptic membrane to approximately -40 to -20 mV electrostatically repels the Mg2+ ion, unblocking the channel. the voltage dependence follows:

    B(V) = 1 / (1 + [Mg2+]_o / 3.57 * exp(-0.062 * V))

this creates a multiplicative gate: current flows only when glutamate is present (presynaptic activity) AND the membrane is depolarized (postsynaptic activity). neither condition alone is sufficient. this is the biophysical implementation of hebbian coincidence detection -- the receptor literally detects when pre and post are active together.

ML analog: the Mg2+ block functions as a multiplicative gate analogous to swiglu's silu(W_gate(x)) * W_up(x). but unlike swiglu, which is self-gating (both inputs derive from x), the nmda gate is cross-source: one input (glutamate) comes from the presynaptic neuron and the other (depolarization) comes from the postsynaptic neuron's own activity. this is closer to cross-attention gating than self-attention gating.

### calcium permeability and plasticity cascades

once unblocked, nmda receptors are highly permeable to **Ca2+** (calcium ions), in addition to Na+ and K+. the Ca2+ influx through nmda receptors is the primary trigger for synaptic plasticity. the calcium signal activates a cascade:

1. **CaMKII** (calcium/calmodulin-dependent protein kinase II) -- autophosphorylates and remains active after Ca2+ dissipates, providing a molecular memory trace lasting minutes to hours
2. **CREB** (cAMP response element-binding protein) -- a transcription factor that activates gene expression for long-term structural changes
3. **protein synthesis** -- local dendritic translation of mRNAs for structural plasticity proteins (AMPA receptor insertion, spine growth)

the magnitude and temporal pattern of Ca2+ influx determines the direction of plasticity: high-frequency stimulation produces large, fast Ca2+ transients that trigger **LTP** (long-term potentiation, a lasting increase in synaptic strength), while low-frequency stimulation produces smaller, sustained Ca2+ levels that trigger **LTD** (long-term depression, a lasting decrease). this is the **BCM rule** in biophysical form (see [[bcm_theory]]).

ML analog: Ca2+ influx gating plasticity cascades maps to gated gradient flow. the nmda receptor decides not just whether to transmit a signal but whether to update the weights. in standard backpropagation, gradient always flows; in the nmda system, gradient flow (plasticity) is gated by the coincidence detector. this is conceptually similar to gated update rules in adaptive optimizers.

### slow kinetics

nmda receptors have dramatically slower kinetics than **AMPA receptors** (the other major glutamate receptor type, which mediates fast excitatory transmission). AMPA receptors activate in <1 ms and decay in ~1-5 ms. nmda receptors rise in ~10 ms and decay in ~50-300 ms depending on GluN2 subunit:

- GluN2A: decay ~50-100 ms (faster, predominant at mature synapses)
- GluN2B: decay ~200-300 ms (slower, predominant at immature synapses and extrasynaptic locations)
- GluN2C/D: intermediate kinetics, primarily in cerebellum and diencephalon

the slow kinetics mean that nmda receptors integrate inputs over a much wider temporal window than AMPA receptors. a single nmda receptor activation outlasts the original synaptic event by 10-100x, creating a sustained depolarization that bridges discrete inputs into a continuous signal.

### nmda spikes in dendritic computation

in thin basal and oblique dendrites, clustered synaptic inputs can trigger **nmda spikes** -- regenerative plateau potentials sustained by positive feedback between nmda receptor activation and local depolarization (see [[dendritic_spikes]]). the sequence: synaptic glutamate activates nmda receptors -> local depolarization relieves Mg2+ block -> more current flows -> more depolarization -> more Mg2+ relief. this positive feedback loop produces a sustained plateau of ~40-50 mV lasting 50-500 ms.

nmda spikes are spatially confined to individual dendritic branches, enabling each branch to function as an independent computational subunit (Polsky et al. 2004, see [[two_layer_neuron]]). this is the substrate for branch-specific nonlinear integration: ~10-50 co-active synapses within ~20-50 um of dendritic length can trigger an nmda spike that is invisible to neighboring branches (Schiller et al. 2000).

### nmda in plasticity: the hebbian coincidence detector

the requirement for nmda receptors in LTP induction was established by Collingridge et al. (1983), who showed that the nmda antagonist APV blocks LTP without affecting baseline synaptic transmission. Morris et al. (1986) demonstrated that nmda blockade impairs spatial learning in rats (Morris water maze). these experiments established the nmda receptor as the molecular implementation of the [[hebbian_learning]] rule: synapses are strengthened when pre and post are active together, and the nmda receptor is the detector that identifies this conjunction.

the link between [[stdp]] (spike-timing-dependent plasticity) and nmda receptors is direct: the relative timing of pre (glutamate) and post (backpropagating action potential, which provides depolarization) determines the magnitude of Mg2+ unblock and thus Ca2+ influx. pre-before-post produces maximal Ca2+ influx (LTP); post-before-pre produces minimal Ca2+ influx (LTD).

### nmda plateau potentials and persistent activity

Antic et al. (2010) and others demonstrated that nmda plateau potentials can sustain depolarization for 100-500 ms without ongoing synaptic input. this creates a form of working memory at the dendritic level: a single plateau event can maintain a local representation across multiple input cycles. in prefrontal cortex, nmda plateau potentials have been proposed as a substrate for persistent activity underlying working memory, complementing network-level attractor models (see [[dendritic_computation]]).

ML analog: nmda plateau potentials are the closest biological analog to the ATMN membrane potential in todorov. both maintain a persistent state without explicit recurrence -- the nmda plateau through regenerative channel dynamics, the ATMN through temporal integration with adaptive threshold (note: ATMN currently lacks a leak term -- see [[neuron_models_to_atmn]]). the key difference: nmda plateaus are branch-local, while ATMN membrane state is per-neuron.

### GluN2B to GluN2A developmental switch

during early postnatal development, nmda receptors undergo a subunit composition switch from GluN2B-dominant to GluN2A-dominant. this transition has profound computational consequences:

- **immature (GluN2B)**: slower decay (~300 ms), larger Ca2+ influx per event, enhanced plasticity, wider temporal integration window. this supports the high plasticity of [[critical_periods]].
- **mature (GluN2A)**: faster decay (~50-100 ms), smaller Ca2+ influx per event, reduced plasticity, tighter temporal precision. this supports stable, precise computation in the adult brain.

the switch is activity-dependent: sensory experience drives GluN2A expression. dark-rearing delays the switch in visual cortex; light exposure accelerates it. this means the receptor itself encodes the transition from learning mode to inference mode.

ML analog: the GluN2B-to-GluN2A switch maps to learning rate decay or critical period scheduling. early training with high learning rate (GluN2B, high plasticity, wide integration) transitions to lower learning rate (GluN2A, stable weights, precise timing). the biological system implements this transition at the receptor level, not as a global hyperparameter.

### nmda hypofunction and schizophrenia

pharmacological blockade of nmda receptors by **PCP** (phencyclidine) or **ketamine** produces symptoms closely resembling schizophrenia in healthy subjects: positive symptoms (hallucinations, delusions), negative symptoms (social withdrawal, flat affect), and cognitive symptoms (working memory deficits, disorganized thinking). the **nmda hypofunction hypothesis** (Olney and Farber 1995, Javitt and Zukin 1991) proposes that reduced nmda receptor function, particularly on GABAergic interneurons, disinhibits cortical pyramidal neurons, disrupting the excitatory-inhibitory balance (see [[excitatory_inhibitory_balance]]).

this has computational significance: if nmda receptors are the coincidence detectors that enforce hebbian learning rules, then nmda hypofunction means the network can no longer properly associate correlated inputs. the result is a network that forms spurious associations (positive symptoms) and fails to form valid ones (cognitive symptoms).

## mathematical formulation

    I_NMDA = g_NMDA * B(V) * (V - E_NMDA)

where:
- g_NMDA = total nmda conductance (proportional to glutamate-bound receptor count)
- B(V) = 1 / (1 + [Mg2+]_o / 3.57 * exp(-0.062 * V)) is the Mg2+ block function
- E_NMDA ~ 0 mV (reversal potential for mixed Na+/K+/Ca2+ current)

Ca2+ fraction of total current:

    I_Ca_NMDA ~ 0.1 * I_NMDA (approximately 10% of total nmda current is carried by Ca2+)

## evidence strength

STRONG. nmda receptors are among the most extensively studied molecules in neuroscience.

1. Mg2+ block: Mayer et al. (1984), Nowak et al. (1984). single-channel recordings demonstrated voltage-dependent block with millisecond kinetics.
2. LTP requirement: Collingridge et al. (1983). APV blocks LTP induction but not expression or baseline transmission.
3. spatial learning: Morris et al. (1986). nmda antagonists impair learning in Morris water maze.
4. nmda spikes: Schiller et al. (2000), Polsky et al. (2004). glutamate uncaging demonstrates regenerative nmda-dependent potentials in thin dendrites.
5. plateau potentials: Antic et al. (2010). dendritic nmda plateaus sustain depolarization for hundreds of milliseconds.

## challenges and counter-arguments

1. **the coincidence detector framing oversimplifies.** nmda receptors do not implement a clean AND gate. the Mg2+ block is graded, not binary -- partial depolarization produces partial unblock. the receptor also requires glycine/D-serine as a co-agonist, making it a three-input gate in practice. furthermore, polyamines, zinc, pH, and redox state all modulate nmda function, meaning the "AND gate" has at least six modulatory inputs. the clean computational narrative obscures this biochemical complexity.

2. **nmda-independent LTP exists.** while nmda-dependent LTP at Schaffer collateral-CA1 synapses is the canonical form, nmda-independent LTP has been demonstrated at mossy fiber-CA3 synapses (Harris and Cotman 1986), corticothalamic synapses, and various other circuits. the universality of nmda receptors as THE mechanism for associative learning is overstated -- multiple plasticity mechanisms coexist, and nmda-dependent LTP may be a special case rather than the general rule.

3. **the schizophrenia model is incomplete.** while nmda antagonists produce schizophrenia-like symptoms, schizophrenia is not simply nmda hypofunction. genetic studies implicate hundreds of loci, and the nmda hypofunction model does not explain why symptoms emerge in late adolescence, why there are sex differences, or why antipsychotics (which primarily target dopamine D2 receptors, not nmda receptors) are partially effective. the model is a useful entry point but not a complete explanation.

4. **extrapolation from slice to behaving brain.** most detailed biophysical characterization of nmda receptors comes from in vitro slice preparations at room temperature with artificial extracellular solutions. in vivo, ongoing synaptic bombardment, neuromodulatory tone, and temperature (37C vs 22C) all change nmda receptor kinetics. decay times measured in vivo are consistently faster than in vitro estimates, and the effective temporal integration window may be narrower than commonly cited values suggest.

## computational implications

1. **multiplicative gating as a fundamental computational primitive.** the nmda Mg2+ block implements gating at the biophysical level. this same operation appears in swiglu (silu(gate) * value), in kda (channel-wise gating of delta-rule updates), and in lstm (input/forget/output gates). biology discovered multiplicative gating ~500 million years before deep learning.

2. **gated plasticity separates inference from learning.** the nmda receptor separates signal transmission (AMPA) from weight update (nmda -> Ca2+ -> CaMKII). in standard neural networks, every forward pass is also a potential backward pass. the biological separation suggests architectures where the decision to update weights is itself a learned, gated computation.

3. **persistent state without recurrence.** nmda plateau potentials maintain information for 100-500 ms through local channel dynamics, not through recurrent connectivity. this is the biological precedent for the ATMN membrane potential in todorov, which maintains per-neuron state through leaky integration rather than network-level recurrence.

## bridge to todorov

todorov implements multiplicative gating at two levels: swiglu (feed-forward gating) and kda (channel-wise gating of recurrent state updates). the nmda receptor's Mg2+ block is the biological precedent for both, but with a critical difference: nmda gating is cross-source (pre x post), while swiglu and kda gating are self-gating (input gates itself).

the ATMN membrane potential mechanism is the closest todorov analog to nmda plateau potentials. both maintain persistent state through local dynamics (leaky integration / regenerative channel activation) rather than recurrent connectivity. the ATMN adaptive threshold (v_th = exp(a)) parallels the activity-dependent modulation of nmda receptor properties.

what exists: multiplicative gating (swiglu, kda), persistent membrane state (ATMN).
what is missing: cross-source gating (pre x post coincidence detection), gated plasticity (learning rate controlled by signal content), developmental subunit switching.
what matters: cross-source gating could improve context-dependent processing by allowing one stream to gate another, rather than relying on self-gating alone.

## related mechanisms

- [[dendritic_spikes]] -- nmda spikes as one of three dendritic spike types
- [[dendritic_computation]] -- the broader framework for dendritic information processing
- [[hebbian_learning]] -- nmda receptors as the molecular implementation of the hebb rule
- [[stdp]] -- spike-timing-dependent plasticity mediated by nmda timing sensitivity
- [[critical_periods]] -- the GluN2B-to-GluN2A switch as a critical period mechanism
- [[two_layer_neuron]] -- nmda spikes enabling branch-independent subunit computation
- [[apical_amplification]] -- nmda receptors in apical tuft integration

## source bibliography

- Mayer, M.L., Westbrook, G.L., and Guthrie, P.B. (1984). Voltage-dependent block by Mg2+ of NMDA responses in spinal cord neurones. Nature, 309(5965), 261-263.
- Nowak, L., Bregestovski, P., Ascher, P., Herbet, A., and Prochiantz, A. (1984). Magnesium gates glutamate-activated channels in mouse central neurones. Nature, 307(5950), 462-465.
- Collingridge, G.L., Kehl, S.J., and McLennan, H. (1983). Excitatory amino acids in synaptic transmission in the Schaffer collateral-commissural pathway of the rat hippocampus. Journal of Physiology, 334, 33-46.
- Morris, R.G., Anderson, E., Lynch, G.S., and Bhatt, M. (1986). Selective impairment of learning and blockade of long-term potentiation by an N-methyl-D-aspartate receptor antagonist, AP5. Nature, 319(6056), 774-776.
- Schiller, J., Major, G., Koester, H.J., and Schiller, Y. (2000). NMDA spikes in basal dendrites of cortical pyramidal neurons. Nature, 404(6775), 285-289.
- Polsky, A., Mel, B.W., and Schiller, J. (2004). Computational subunits in thin dendrites of pyramidal cells. Nature Neuroscience, 7(6), 621-627.
- Antic, S.D., Zhou, W.L., Moore, A.R., Short, S.M., and Bhatt, D. (2010). The decade of the dendritic NMDA spike. Journal of Neuroscience Research, 88(14), 2991-3001.
- Olney, J.W. and Farber, N.B. (1995). Glutamate receptor dysfunction and schizophrenia. Archives of General Psychiatry, 52(12), 998-1007.
- Javitt, D.C. and Zukin, S.R. (1991). Recent advances in the phencyclidine model of schizophrenia. American Journal of Psychiatry, 148(10), 1301-1308.
- Harris, E.W. and Cotman, C.W. (1986). Long-term potentiation of guinea pig mossy fiber responses is not blocked by N-methyl-D-aspartate antagonists. Neuroscience Letters, 70(1), 132-137.
- Major, G., Larkum, M.E., and Schiller, J. (2013). Active properties of neocortical pyramidal neuron dendrites. Annual Review of Neuroscience, 36, 1-24.
- Paoletti, P., Bellone, C., and Zhou, Q. (2013). NMDA receptor subunit diversity: impact on receptor properties, synaptic plasticity and disease. Nature Reviews Neuroscience, 14(6), 383-400.
