# acetylcholine system

status: definitional. last fact-checked 2026-04-16.

**why this matters**: acetylcholine implements a biological learning rate -- switching between encoding new information and retrieving stored patterns. this directly addresses the stability-plasticity dilemma that causes catastrophic forgetting in neural networks.

## summary

the **basal forebrain** (a group of subcortical nuclei that project widely to cortex and hippocampus) cholinergic system modulates the balance between encoding new information and retrieving stored patterns. Hasselmo (2006) proposed that **acetylcholine** (**ACh**, a neuromodulatory neurotransmitter released by basal forebrain and brainstem nuclei) acts as a switch between two cortical operating modes:

**high ACh (encoding mode)**: enhances **afferent** (feedforward, incoming) input via **nicotinic receptors** (fast ionotropic receptors gated by ACh). suppresses recurrent (feedback) connections via **muscarinic** (slow metabotropic receptors gated by ACh) presynaptic inhibition. net effect: the cortex prioritizes incoming sensory data over stored representations. this mode favors learning new associations.

ML analog: high ACh is analogous to a high learning rate with strong input weighting -- the model updates aggressively from new data rather than relying on cached representations.

**low ACh (retrieval/consolidation mode)**: releases the suppression of recurrent connections. stored patterns dominate cortical dynamics via recurrent excitation. net effect: the cortex prioritizes pattern completion from memory. this mode favors recall and memory consolidation (especially during slow-wave sleep).

ACh is thus a neuromodulatory learning rate: high ACh increases the rate at which new information overwrites old patterns; low ACh protects existing memories from interference. in Doya's (2002) metalearning framework, ACh corresponds to the learning rate meta-parameter.

## anatomy

### source nuclei

**nucleus basalis of Meynert (NBM)**: the primary source of cortical ACh. located in the basal forebrain. projects diffusely to all neocortical areas. degeneration of NBM neurons is a hallmark of Alzheimer's disease and correlates with the severity of cognitive decline.

**medial septum / diagonal band of Broca**: projects primarily to hippocampus. critical for generating theta oscillations (4-8 Hz) during active exploration and encoding. septal cholinergic input to hippocampus is required for spatial learning and episodic memory formation.

**pedunculopontine and laterodorsal tegmental nuclei**: brainstem cholinergic nuclei projecting to thalamus and basal ganglia. involved in arousal, REM sleep, and sensory gating.

### receptor subtypes

**nicotinic receptors (nAChRs)**: ionotropic (ligand-gated ion channels). fast action (~milliseconds). alpha7 subtype dominant in cortex: high calcium permeability, rapid desensitization. enhance presynaptic glutamate release at afferent synapses, increasing the magnitude of thalamocortical and other feedforward inputs. also modulate interneuron activity, increasing the temporal precision of pyramidal cell firing during theta oscillations.

**muscarinic receptors (mAChRs)**: metabotropic (G-protein coupled). slow action (~hundreds of milliseconds). five subtypes (M1-M5). M1 and M3 (excitatory, Gq-coupled): enhance pyramidal cell excitability, enable persistent spiking, facilitate LTP. M2 and M4 (inhibitory, Gi-coupled): presynaptic inhibition of glutamate release at recurrent and associational synapses. M2 presynaptic inhibition is the key mechanism for suppressing recurrent activity during encoding.

the nicotinic/muscarinic dichotomy creates the encoding switch: nicotinic enhancement of feedforward + muscarinic suppression of feedback = prioritize new input over old patterns.

## mechanism

### encoding vs retrieval

Hasselmo's model of the hippocampal CA3 region provides the clearest formulation:

**encoding (high ACh)**:
- muscarinic M2 receptors suppress Schaffer collateral and recurrent CA3 synapses (feedback)
- nicotinic alpha7 receptors enhance mossy fiber and perforant path input (feedforward)
- net effect: new patterns from entorhinal cortex drive CA3 activity; old stored patterns cannot compete via recurrent connections
- synaptic plasticity (LTP) is enhanced, so the new pattern is stored

**retrieval (low ACh)**:
- muscarinic suppression is released; Schaffer collateral and recurrent synapses operate at full strength
- a partial cue activates a subset of the stored pattern
- recurrent excitation completes the full pattern via attractor dynamics
- synaptic plasticity (LTP) is reduced, protecting the stored pattern from overwriting

this two-phase cycle operates on multiple timescales:
- theta cycle (~125 ms): encoding on the trough (high ACh), retrieval on the peak (low ACh). Hasselmo proposed that theta oscillations implement a rapid alternation between encoding and retrieval within each cycle.
- sleep-wake cycle (~hours): high ACh during waking (encoding), low ACh during slow-wave sleep (consolidation). cholinergic neurons in the basal forebrain are active during waking and silent during slow-wave sleep.

### attention and uncertainty

Yu and Dayan (2005) proposed that ACh signals EXPECTED uncertainty (known unreliability of sensory cues), while norepinephrine signals UNEXPECTED uncertainty (surprising events that indicate environmental change). in this framework:

- high ACh = "sensory input is unreliable; increase gain on feedforward processing to gather more information"
- this maps onto the Hasselmo model: when input is uncertain, the brain should weight new data more heavily (encoding mode) rather than rely on stored patterns (retrieval mode)

the computational role of ACh is thus dual: it is both a learning rate modulator (how fast to update) and a precision signal (how reliable is the current input). see [[precision_weighting]] for the Feldman & Friston (2010) interpretation where ACh modulates sensory precision.

### cholinergic modulation of plasticity

ACh enhances LTP at feedforward synapses through M1 receptor activation of PKC and CaMKII signaling. simultaneously, ACh suppresses LTP at recurrent synapses (via M2 presynaptic inhibition reducing postsynaptic depolarization). this selective plasticity enhancement is critical: without it, both old and new patterns would be modified simultaneously, creating catastrophic interference.

the molecular pathway: M1 activation -> Gq -> PLC -> IP3 + DAG -> PKC activation -> enhanced AMPA receptor insertion. this occurs at the same synapses that are active during encoding, implementing a form of gated [[hebbian_learning]]:

    Delta_w_feedforward = eta_high * x_pre * x_post * ACh
    Delta_w_recurrent = eta_low * x_pre * x_post * (1 - ACh)

where ACh modulates the effective learning rate differently for feedforward and recurrent connections.

## relationship to todorov

### beta gate as ACh analog

KDA's write gate beta_t = sigmoid(beta_proj(x_t)) is the closest functional analog to cholinergic modulation in todorov:

- high beta: write strongly to KDA state S_t (encoding mode)
- low beta: preserve existing state (retrieval/consolidation mode)
- beta is data-dependent: some tokens trigger strong writes, others weak writes

the correspondence is real but limited:
1. beta is computed from the CURRENT TOKEN only. biological ACh is modulated by global brain state (arousal, task demands, uncertainty). beta has no access to global context.
2. beta gates ALL connections equally (one scalar per head). biological ACh differentially modulates feedforward vs recurrent connections.
3. beta does not modulate plasticity (weight updates). it modulates the forward computation only. biological ACh modulates BOTH online processing AND synaptic plasticity.

### missing mechanisms

todorov has no encoding/retrieval separation. KDA simultaneously reads from (retrieval) and writes to (encoding) the state S_t on every token. biological cholinergic modulation would suggest separating these phases: some tokens should primarily read without writing (low ACh, retrieval), others should primarily write without reading (high ACh, encoding). the current beta gate partially achieves this (low beta = mostly read, high beta = write + read), but the read operation is not modulated by beta.

see [[neuromodulation_to_learning_and_gating]] for a proposed neuromodulator network that could implement context-dependent modulation of beta.

## challenges

1. **ACh does more than learning rate**: cholinergic modulation affects attention, arousal, sensory processing, motor control, and autonomic function. the "ACh = learning rate" mapping (Doya 2002) is a useful simplification, not a complete account.

2. **timing ambiguity**: the theta-phase encoding/retrieval model predicts that encoding happens at specific phases of the theta cycle. experimental evidence is mixed: some studies confirm phase-dependent plasticity, others find no phase dependence.

3. **Alzheimer's paradox**: if ACh primarily facilitates encoding, cholinergic degeneration should impair new learning while sparing retrieval. Alzheimer's patients show BOTH encoding and retrieval deficits, suggesting ACh plays a more general role than Hasselmo's model predicts (though the deficits may also reflect broader neurodegeneration beyond the cholinergic system).

4. **cholinergic diversity**: the basal forebrain contains both cholinergic AND GABAergic projection neurons. GABAergic projections from the basal forebrain to cortex have distinct effects that are often confounded with cholinergic effects in lesion studies.

## key references

- Hasselmo, M. E. (2006). The role of acetylcholine in learning and memory. Current Opinion in Neurobiology, 16(6), 710-715.
- Hasselmo, M. E. & McGaughy, J. (2004). High acetylcholine levels set circuit dynamics for attention and encoding and low acetylcholine levels set dynamics for consolidation. Progress in Brain Research, 145, 207-231.
- Yu, A. J. & Dayan, P. (2005). Uncertainty, neuromodulation, and attention. Neuron, 46(4), 681-692.
- Hasselmo, M. E. (1999). Neuromodulation: acetylcholine and memory consolidation. Trends in Cognitive Sciences, 3(9), 351-359.
- Moran, R. J. et al. (2013). Free energy, precision and learning: the role of cholinergic neuromodulation. Journal of Neuroscience, 33(19), 8227-8236.
- Ballinger, E. C. et al. (2016). Basal forebrain cholinergic circuits and signaling in cognition and cognitive decline. Neuron, 91(6), 1199-1218.

## see also

- [[dopamine_system]]
- [[norepinephrine_system]]
- [[neuromodulatory_framework]]
- [[neuromodulation_to_learning_and_gating]]
- [[precision_weighting]]
- [[hebbian_learning]]
- [[homeostatic_plasticity]]
