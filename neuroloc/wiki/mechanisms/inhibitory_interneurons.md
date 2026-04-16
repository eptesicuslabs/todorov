# inhibitory interneurons

status: definitional. last fact-checked 2026-04-16.

**why this matters**: the three cardinal interneuron types (PV+, SST+, VIP+) implement distinct computational operations -- temporal gating, dendritic input selection, and disinhibition -- that map to specific ML primitives: PV+ to softmax competition, SST+ to attention masking, and VIP+ to gating mechanisms that route information based on context.

## the 80/20 ratio

approximately 80% of cortical neurons are excitatory (pyramidal cells, stellate cells) and 20% are inhibitory (interneurons). despite being a minority, inhibitory interneurons exert disproportionate influence because:

1. individual inhibitory synapses are stronger than excitatory ones
2. inhibitory neurons fire at higher rates than excitatory neurons
3. each inhibitory neuron contacts many excitatory neurons (divergent connectivity)
4. inhibitory neurons are positioned to control the timing and magnitude of excitatory activity

this 80/20 ratio is remarkably consistent across cortical areas, layers, and species. it is not arbitrary: computational modeling shows that an 80/20 **E/I ratio** (excitation-to-inhibition balance) optimally stabilizes a dynamical regime characterized by intermittent, burst-like activity. this state is associated with maximal information capacity. near the 80/20 ratio, this regime emerges robustly across a wide range of parameters and with low energy cost.

**Dale's law** (the principle that each neuron releases the same neurotransmitter at all its synapses) constrains the circuit: each neuron is either excitatory or inhibitory, never both. a neuron releases the same neurotransmitter (glutamate or GABA) at all its synapses. this means inhibition must be implemented by dedicated cells, not by dual-function neurons.

## the three cardinal interneuron types

cortical inhibitory interneurons are classified into three non-overlapping molecular classes, each with distinct morphology, connectivity, electrophysiology, and computational role.

### PV+ (parvalbumin-positive): fast-spiking perisomatic inhibition

**morphology:** basket cells and chandelier cells. basket cells form synapses on the soma and proximal dendrites (perisomatic region) of pyramidal cells. chandelier cells form rows of boutons (cartridges) on the axon initial segment (AIS) of pyramidal cells.

**electrophysiology:** fast-spiking (FS) phenotype. short action potential duration (~0.3 ms half-width), high maximum firing rate (up to 300-800 Hz in some species), minimal spike-frequency adaptation, low input resistance, short membrane time constant. PV+ neurons can follow high-frequency input without failing, making them ideal for temporal precision.

**connectivity:**
- receive strong, convergent excitation from nearby pyramidal cells
- send divergent inhibition to many pyramidal cells (each PV+ basket cell contacts ~100-1000 pyramidal cells)
- receive inhibition from VIP+ interneurons (disinhibitory pathway)
- strong reciprocal connections between PV+ cells (network of mutual inhibition)

**computational role: timing and gating.**

PV+ perisomatic inhibition controls WHEN a pyramidal cell fires, not WHETHER it fires. by targeting the soma and AIS, PV+ neurons can veto action potential generation with millisecond precision. this enables:

1. **temporal precision:** the brief window of PV+ inhibition constrains the timing of the first spike evoked by sensory input, enforcing synchrony across the population.
2. **gamma oscillations:** reciprocal interactions between PV+ interneurons and pyramidal cells generate ~30-80 Hz gamma oscillations. the cycle of excitation-inhibition-recovery creates a periodic window for spiking, implementing a temporal [[winner_take_all]]: neurons that reach threshold earliest in each gamma cycle fire; the rest are suppressed until the next cycle.
3. **gain control:** strong PV+ inhibition scales down the overall activity of a pyramidal cell population without changing its selectivity. optogenetic activation of PV+ neurons produces divisive gain modulation (scaling the response amplitude).
4. **feedforward inhibition:** in thalamocortical and corticocortical pathways, PV+ neurons are recruited by the same input that drives pyramidal cells, creating a brief excitatory window followed by inhibition. this sharpens temporal responses.

**chandelier cells** specifically target the AIS, the site of action potential initiation. their effect is debated: in adults, they may be depolarizing (due to the high Cl- reversal potential at the AIS) and could play a role in excitation rather than inhibition, though this is contested.

### SST+ (somatostatin-positive): dendritic inhibition

**morphology:** Martinotti cells (layer 2/3 and layer 5), with ascending axons that arborize in layer 1. also includes non-Martinotti SST+ cells in layer 4 and deeper layers.

**electrophysiology:** regular-spiking or low-threshold-spiking. moderate firing rates, some spike-frequency adaptation. lower input resistance than PV+ cells. their IPSPs show little short-term depression with repeated activation, enabling sustained inhibitory influence.

**connectivity:**
- receive excitation from local pyramidal cells (feedback connection)
- target the distal dendrites of pyramidal cells (especially in layer 1 apical tufts)
- receive inhibition from VIP+ interneurons (strong, primary target of VIP+ disinhibition)
- sparse connections to other SST+ cells

**computational role: gating of dendritic input.**

SST+ dendritic inhibition controls WHAT a pyramidal cell integrates. by targeting the distal dendrites, SST+ neurons can selectively block specific synaptic inputs (top-down feedback, long-range lateral connections) without affecting the soma.

1. **dendritic gate:** SST+ inhibition controls whether top-down signals (from higher cortical areas) reach the soma. when SST+ neurons are active, the apical dendritic tuft is shunted, and top-down input is suppressed. when SST+ neurons are silenced (e.g., by VIP+ disinhibition), top-down signals pass through.
2. **surround suppression:** SST+ neurons contribute to surround suppression in V1: they are recruited by stimuli extending beyond the receptive field center and inhibit pyramidal cell dendrites, reducing the response to the center stimulus. this connects SST+ activity to [[divisive_normalization]] at the circuit level.
3. **sublinear response control:** optogenetic activation of SST+ neurons produces primarily divisive inhibitory gain control, reducing the magnitude of supralinear dendritic responses without affecting their threshold.
4. **sustained inhibition:** unlike PV+ IPSPs (which are brief and depress rapidly), SST+ IPSPs are sustained and non-depressing. this enables tonic inhibition that gates ongoing integration rather than brief temporal selection.

### VIP+ (vasoactive intestinal peptide-positive): disinhibition

**morphology:** bipolar and bitufted cells with vertically oriented dendrites and axons. small somata, vertically restricted dendritic arbors.

**electrophysiology:** irregular-spiking or adapting. moderate firing rates. some VIP+ subtypes show burst firing.

**connectivity:**
- receive strong input from long-range corticocortical projections (feedback from higher areas)
- receive strong neuromodulatory input (acetylcholine, serotonin)
- primarily inhibit SST+ interneurons (strong, monosynaptic)
- weakly inhibit PV+ interneurons
- minimal direct inhibition of pyramidal cells

**computational role: disinhibition and top-down gating.**

VIP+ neurons implement **disinhibition** (inhibition of inhibitory neurons, resulting in net excitation): VIP+ inhibits SST+, which inhibits pyramidal cells. the net effect of VIP+ activation is increased pyramidal cell activity -- inhibition of inhibition equals excitation. ML analog: disinhibition is functionally equivalent to a gating mechanism like the sigmoid gate in LSTMs or the beta gate in KDA -- a signal that selectively unblocks information flow through a pathway.

Pi et al. (2013) demonstrated that VIP+ interneurons mediate disinhibitory circuits: activation of VIP+ neurons transiently suppresses SST+ and a fraction of PV+ interneurons, releasing pyramidal cells from inhibition.

1. **behavioral state gating:** VIP+ neurons are activated during locomotion, whisking, arousal, and attention. during these states, VIP+ disinhibition opens the dendritic gate (by silencing SST+ neurons), allowing top-down and long-range lateral signals to influence pyramidal cell responses. this modulates cortical processing based on behavioral context.
2. **reinforcement signaling:** in auditory cortex, VIP+ neurons are strongly and uniformly activated by reinforcement signals (reward and punishment). punishment produces sharp activation (peak at ~50 ms latency); reward produces more sustained activation. this disinhibition during reinforcement could gate plasticity in the underlying pyramidal population.
3. **gain modulation:** VIP+-mediated disinhibition produces different transformations depending on target neuron properties: divisive gain modulation (scaling response amplitude) in some pyramidal cells, and additive baseline shifts in others. this enables sophisticated, cell-type-specific modulation from a single disinhibitory signal.
4. **conserved circuit motif:** the VIP+ -> SST+ -> pyramidal cell disinhibitory pathway is found across cortical areas (visual, auditory, somatosensory, prefrontal) and species, suggesting it is a canonical circuit motif for routing external signals through local circuits.

## E/I balance

the balance between excitation and inhibition is critical for cortical function. in the balanced state:

- excitatory and inhibitory synaptic currents in a pyramidal cell are approximately equal in magnitude but opposite in sign
- the net input is a small difference between two large quantities, making the neuron sensitive to fluctuations
- firing is driven by temporal fluctuations in the balance, not by the mean input
- this produces irregular firing patterns resembling those observed in vivo

E/I balance is maintained by:
- [[homeostatic_plasticity]]: synaptic scaling adjusts excitatory and inhibitory synaptic strengths to maintain a target firing rate
- inhibitory plasticity: inhibitory synapses undergo activity-dependent plasticity (both Hebbian and anti-Hebbian) to track changes in excitatory drive
- the balanced network regime: networks with strong, random E/I connectivity self-organize into a balanced state (van Vreeswijk and Sompolinsky, 1996)

disruption of E/I balance is implicated in epilepsy (excess excitation), autism spectrum disorders (altered E/I ratio), schizophrenia (reduced PV+ interneuron function), and Alzheimer's disease.

## challenges

- the three-class taxonomy (PV/SST/VIP) is a simplification. within each class, there are subtypes with different properties (e.g., PV+ basket cells vs chandelier cells, SST+ Martinotti vs non-Martinotti). whether these subtypes serve distinct computational roles or are variations on a theme is unresolved.
- the disinhibitory circuit (VIP -> SST -> pyramidal) is well-characterized anatomically, but its functional role in vivo is still debated. VIP+ activation correlates with behavioral state changes (locomotion, attention), but whether disinhibition is the cause or a correlate of these state changes is unclear.
- the computational modeling of inhibitory interneuron circuits is hindered by the large number of cell types, connection types, and dynamic properties. simplified models (e.g., a single inhibitory pool) capture WTA and E/I balance but miss the distinct contributions of PV+, SST+, and VIP+ interneurons.
- the 80/20 ratio is a cortex-wide average. layer-specific E/I ratios vary: layer 1 is almost entirely inhibitory, layer 4 has a higher proportion of inhibitory neurons than layer 2/3. the functional implications of these layer-specific ratios are not fully understood.
- inhibitory interneurons are a minority of cortical neurons, but they are the majority of the complexity. understanding how three (or more) types of inhibition interact to produce stable, flexible, and computationally powerful circuits remains one of the central challenges of systems neuroscience.

## key references

- markram, h. et al. (2004). interneurons of the neocortical inhibitory system. nature reviews neuroscience, 5(10), 793-807.
- pi, h. j. et al. (2013). cortical interneurons that specialize in disinhibitory control. nature, 503(7477), 521-524.
- tremblay, r., lee, s. & rudy, b. (2016). GABAergic interneurons in the neocortex: from cellular properties to circuits. neuron, 91(3), 521-539.
- van vreeswijk, c. & sompolinsky, h. (1996). chaos in neuronal networks with balanced excitatory and inhibitory activity. science, 274(5293), 1724-1726.
- pfeffer, c. k., xue, m., he, m., huang, z. j. & scanziani, m. (2013). inhibition of inhibition in visual cortex: the logic of connections between molecularly distinct interneurons. nature neuroscience, 16(8), 1068-1076.

## see also

- [[lateral_inhibition]]
- [[divisive_normalization]]
- [[winner_take_all]]
- [[homeostatic_plasticity]]
