# neural synchrony

**why this matters**: neural synchrony implements information routing without physical rewiring -- the same connection can transmit or block depending on phase alignment. this is the biological solution to the routing problem that attention mechanisms solve in transformers.

## definition

**neural synchrony** (the temporally coordinated activity of neurons or neural populations, where neurons fire within narrow time windows relative to each other or relative to an ongoing oscillation) is not merely a by-product of shared input. it is a computational mechanism that gates communication, routes information, and encodes variables through timing.

the central theoretical framework is **communication through coherence** (**CTC**, proposed by Fries 2005, 2015): two neural populations communicate effectively only when they are synchronized at appropriate frequencies. synchrony is the mechanism by which the brain selectively routes information without dedicated physical channels.

ML analog: CTC is analogous to attention-based routing in transformers -- attention weights determine which source tokens communicate with each target position, just as phase coherence determines which neural populations communicate with each other.

## communication through coherence (CTC)

### the core mechanism

consider two presynaptic populations (A and B) that both project to a postsynaptic population (C). both A and B are active and sending spikes to C. how does C selectively listen to A while ignoring B?

the CTC hypothesis: C communicates with whichever presynaptic population it is gamma-coherent with.

step-by-step mechanism:

1. each population generates its own [[gamma_oscillations]]: cyclic excitation-inhibition sequences at ~30-90 Hz
2. within each gamma cycle, the population has a brief excitatory window (~3-5 ms) followed by a longer inhibitory window (~10-20 ms)
3. if population A is gamma-coherent with C, then A's excitatory volleys arrive during C's excitatory window. the spikes from A arrive when C's neurons are maximally excitable and before C's perisomatic inhibition kicks in. the spikes from A therefore effectively drive C's neurons
4. if population B is NOT gamma-coherent with C (wrong phase, different frequency), then B's excitatory volleys arrive during C's inhibitory window. the spikes from B are suppressed by ongoing GABAA inhibition and fail to drive C's neurons
5. result: C selectively receives input from A and ignores B, based purely on the phase relationship of their gamma oscillations

the key insight: gamma coherence gates communication. no physical rewiring is needed -- the same synaptic connections can transmit or fail to transmit depending on the temporal relationship between sender and receiver.

### the unequal duty cycle

why gamma, specifically? the mechanism depends on an asymmetric **duty cycle** (the ratio of excitatory to inhibitory phases within one oscillation period): the excitatory window is SHORT (~3-5 ms) and the inhibitory window is LONG (~10-20 ms). this means:

- inputs arriving during the brief excitatory phase are effective (they arrive when neurons are excitable and escape inhibition)
- inputs arriving at any other time are suppressed (they hit the inhibitory phase)

the unequal duty cycle arises naturally from PING/ING dynamics (see [[gamma_oscillations]]): excitation from pyramidal cells activates PV+ interneurons with a ~3-5 ms delay, which then impose ~10-15 ms of GABAA-mediated inhibition. the excitatory "escape" window before inhibition arrives is narrow.

if the duty cycle were 50/50, the mechanism would not work -- random inputs would have equal probability of hitting excitatory or inhibitory phases. the asymmetry is essential.

### selective routing via coherence

attention modulates communication by adjusting coherence:

1. when attention is directed to stimulus A, the gamma oscillation in A's cortical representation increases in power and frequency (~3 Hz upward shift)
2. this stronger, faster gamma from A entrains the postsynaptic population C, establishing coherence
3. the competing gamma from B (representing the unattended stimulus) fails to entrain C because:
   - it arrives at the wrong phase (not coherent with C)
   - it is weaker and slower than A's gamma
   - C's perisomatic inhibition (entrained by A's gamma) actively suppresses B's inputs
4. result: attended information (A) flows to downstream areas; unattended information (B) is gated out

this has been demonstrated in macaque V1-V4: when attention is directed to one of two stimuli, V4 gamma becomes selectively coherent with the V1 representation of the attended stimulus. the V1 representation of the ignored stimulus generates gamma locally but fails to entrain V4 (Fries et al. 2001, Bosman et al. 2012).

## frequency-specific communication channels

CTC is not limited to gamma. different frequency bands serve different communication roles:

### gamma (30-90 Hz): feedforward communication

gamma coherence is primarily observed between areas in the feedforward direction (e.g., V1 to V4, V4 to IT). gamma originates from superficial cortical layers (L2/3), which are the source of feedforward projections. gamma-band coherence carries sensory information from lower to higher cortical areas.

### alpha-beta (8-20 Hz): feedback communication

alpha and beta oscillations are associated with feedback (top-down) communication from higher to lower cortical areas. they originate from deep cortical layers (L5/L6), which are the source of feedback projections. anatomical tracing confirms: projections originating from superficial layers correlate with feedforward designation (gamma), while projections from deep layers correlate with feedback designation (alpha-beta).

alpha-beta feedback performs two functions:
1. **suppression of irrelevant areas:** increased alpha power in cortical areas representing unattended locations/features. alpha is an "inhibitory" rhythm that suppresses gamma and reduces excitability
2. **modulation of gamma:** beta-band signals from higher areas can modulate the frequency and strength of gamma in lower areas, adjusting the "tuning" of the feedforward channel

### theta (4-8 Hz): attentional sampling

Fries (2015) proposed that attention operates as a theta-rhythmic sampling process:

- a single attended stimulus is sampled at ~7-8 Hz
- two competing stimuli are sampled at ~4 Hz each (alternating on successive theta cycles)
- three stimuli at ~2.6 Hz each

this theta-rhythmic sampling explains behavioral oscillations in attention (Landau and Fries 2012): detection performance fluctuates at ~4 Hz for competing stimuli, with peaks alternating between the two locations. attention does not lock onto one stimulus continuously but samples between alternatives at theta frequency.

## cross-frequency coupling

**cross-frequency coupling** (**CFC**, the mechanism by which slow and fast oscillations interact) enables coordination across temporal scales. the most computationally significant form is **phase-amplitude coupling** (**PAC**, where the phase of a slow oscillation modulates the amplitude of a fast oscillation).

### phase-amplitude coupling (PAC)

in PAC, the phase of a slow oscillation modulates the amplitude (power) of a fast oscillation. the canonical example is theta-gamma PAC:

    gamma_amplitude(t) ~ f(theta_phase(t))

gamma power is maximal at specific theta phases (typically near the trough) and minimal at others (typically near the peak). this creates a periodic modulation of local computational activity: gamma-mediated local processing is rhythmically gated by the theta cycle.

PAC is measured by the modulation index (MI):

    MI = (H_max - H_observed) / H_max

where H is the entropy of the gamma amplitude distribution across phase bins. uniform distribution (no coupling) gives MI = 0. complete coupling (gamma only at one phase) gives MI = 1.

### other forms of CFC

- **phase-phase coupling (PPC):** the phase of a fast oscillation is locked to specific phases of a slow oscillation. n:m phase locking, where n fast cycles occur for every m slow cycles. this is the mechanism underlying the theta-gamma code ([[theta_oscillations]])
- **phase-frequency coupling (PFC):** the frequency of a fast oscillation varies with the phase of a slow oscillation. this could implement frequency modulation as an information channel
- **amplitude-amplitude coupling (AAC):** the amplitudes of two oscillations co-vary. this is the weakest form of CFC and may reflect shared input rather than genuine coupling

### computational significance of CFC

CFC solves a fundamental scale problem: slow oscillations coordinate across large brain regions (long-range coherence) but are too slow for fine-grained computation. fast oscillations support local computation with high temporal precision but cannot synchronize over long distances (because of conduction delays). CFC bridges these scales:

- slow oscillations (theta, alpha) provide a large-scale temporal framework that coordinates distant regions
- fast oscillations (gamma) provide local computational windows within the framework
- PAC couples the two: global coordination from slow oscillations, local computation from fast oscillations

this is analogous to a clocking hierarchy in digital systems: a slow master clock synchronizes subsystems, while fast local clocks drive computation within each subsystem.

## temporal coding: information in spike timing

neural synchrony enables temporal coding: information is carried not only by which neurons fire (rate coding) and how many spikes they produce, but by WHEN they fire relative to the ongoing oscillation.

### phase coding

the phase at which a neuron fires within an oscillatory cycle carries information beyond the firing rate. the canonical example is theta phase precession in hippocampal place cells ([[theta_oscillations]]): the firing phase encodes position within the place field, independent of firing rate.

phase coding increases the information capacity of the neural code. if firing rate alone provides R bits/second, adding phase information within a gamma cycle (with ~4-5 discriminable phases) multiplies the capacity by log2(4-5) ~ 2 bits per spike.

### spike timing relative to gamma

within a gamma cycle, the relative timing of spikes encodes input strength and priority:

- strongly driven neurons fire early in the gamma cycle (immediately after inhibition lifts)
- weakly driven neurons fire later (need more time to reach threshold)
- the weakest neurons fail to fire before the next inhibitory volley

this temporal ordering converts a rate code (how many spikes) into a latency code (how fast the spike arrives). latency codes are faster (require only one spike per gamma cycle for readout) and more metabolically efficient (each neuron fires at most once per cycle).

### phase-of-firing code

Montemurro et al. (2008) showed that phase-of-firing carries ~54% additional information beyond firing rate in rat hippocampal CA1 during theta oscillations. this demonstrates that the brain genuinely uses phase as an information channel, not merely as a side effect of oscillatory processing.

## challenges and criticisms

### correlation vs causation

most evidence for CTC is correlational: coherence between areas increases during effective communication. but this does not prove that coherence CAUSES communication. alternative: coherence is a consequence of communication (successful transmission naturally synchronizes receiver to sender), not a prerequisite.

counter-evidence: optogenetic and electrical stimulation studies can establish or disrupt coherence and observe corresponding changes in communication effectiveness. Fries (2015) argues that the millisecond-scale predictive relationship (phase relations precede interactions by a few milliseconds) supports a causal role.

### is synchrony necessary or sufficient?

synchrony may be neither necessary nor sufficient for communication:
- **not necessary:** strong synaptic connections can drive postsynaptic neurons regardless of phase alignment (rate coding works even without synchrony)
- **not sufficient:** two areas can be coherent without exchanging meaningful information (common input can produce coherence without communication)

the strongest version of CTC is that synchrony is MODULATORY: it does not create or destroy communication channels but adjusts their gain. this weaker claim is more defensible but less theoretically distinctive.

### the traveling wave problem

oscillations are often traveling waves, not standing waves. this means that "the phase" at a recording site depends on the wave's propagation direction and speed. interpreting phase relationships between distant sites requires accounting for wave propagation, which complicates CTC claims.

### replication and generalization

much of the CTC evidence comes from the macaque visual system (V1, V2, V4). whether the same mechanisms operate in other cortical areas, other modalities, and other species (including humans, where invasive recordings are limited to clinical contexts) is an open question.

## key references

- Fries, P. (2005). a mechanism for cognitive dynamics: neuronal communication through neuronal coherence. trends in cognitive sciences, 9(10), 474-480.
- Fries, P. (2015). rhythms for cognition: communication through coherence. neuron, 88(1), 220-235.
- Womelsdorf, T. et al. (2007). modulation of neuronal interactions through neuronal synchronization. science, 316(5831), 1609-1612.
- Bosman, C. A. et al. (2012). attentional stimulus selection through selective synchronization between monkey visual areas. neuron, 75(5), 875-888.
- Canolty, R. T. and Knight, R. T. (2010). the functional role of cross-frequency coupling. trends in cognitive sciences, 14(11), 506-515.
- Landau, A. N. and Fries, P. (2012). attention samples stimuli rhythmically. current biology, 22(11), 1000-1004.
- Montemurro, M. A. et al. (2008). phase-of-firing coding of natural visual stimuli in primary visual cortex. current biology, 18(5), 375-380.

## see also

- [[gamma_oscillations]]
- [[theta_oscillations]]
- [[inhibitory_interneurons]]
- [[precision_weighting]]
- [[lateral_inhibition]]
