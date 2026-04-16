# ignition dynamics

status: definitional. last fact-checked 2026-04-16.

**why this matters**: ignition is a nonlinear all-or-none activation threshold that determines whether information gets globally broadcast -- a biological mechanism analogous to gating in mixture-of-experts and the threshold crossing in ternary spike activations that determines whether a signal propagates or is zeroed out.

## overview

**ignition** is the neural implementation of conscious access in the [[global_workspace_theory|global neuronal workspace]] framework. it refers to a sudden, self-sustaining, all-or-none activation of a widespread cortical network. ignition is triggered when a local sensory signal exceeds a threshold and activates long-range recurrent feedback loops. it is the transition from unconscious local processing to conscious global broadcasting.

ML analog: ignition is analogous to the "activation threshold" in sparse activation functions. in ternary spikes, signals below alpha * mean(|x|) are zeroed -- they never propagate. signals above threshold pass at full magnitude. the biological version adds self-sustaining recurrence (the signal amplifies itself once threshold is crossed), which has no direct ML analog in feedforward architectures.

the term was introduced by Dehaene & Changeux (2005) and formalized computationally by Dehaene, Charles, King & Marti (2014). the key insight: there is no gradual transition from unconscious to conscious processing. instead, there is a nonlinear bifurcation -- a phase transition -- where local activity either dies out (subliminal processing) or suddenly explodes into a sustained, globally distributed activation pattern (conscious ignition).

## the mechanism

### local processing phase (0-200 ms)

sensory input enters primary sensory cortex and propagates through a **feedforward sweep** (the initial wave of activation flowing from lower to higher cortical areas without feedback):
- stimulus arrives at sensory receptors
- fast AMPA-mediated feedforward connections carry the signal through the cortical hierarchy (V1 -> V2 -> V4 -> IT for vision)
- processing is local: each area performs its specialized computation
- the signal is a decaying wave -- without amplification, it diminishes at each relay
- this phase is largely identical for conscious and unconscious stimuli
- ERP components during this phase (C1, P1, N1, N170) are present for both seen and unseen stimuli
- semantic processing can occur: masked words prime associated words, demonstrating feedforward processing without consciousness

### threshold crossing

for ignition to occur, two conditions must be met:

1. **sufficient signal strength**: the feedforward signal must be strong enough to reach workspace neurons in prefrontal and parietal cortex. masking, brief presentation, or low contrast can prevent this.

2. **available attention**: top-down attentional amplification (see [[selective_attention]]) must be available. if the workspace is already occupied by a previous ignition (attentional blink), a new signal cannot ignite even if it is strong enough.

the threshold is not fixed -- it depends on ongoing brain state, arousal, attentional allocation, and competing signals. this is why the same stimulus can sometimes be consciously perceived and sometimes not (near-threshold experiments).

### ignition phase (~200-300 ms)

when both conditions are met, a sudden state transition occurs:

1. the feedforward signal reaches prefrontal and parietal workspace neurons
2. these neurons activate and send feedback signals via long-range NMDA-mediated connections back to the sensory cortex that originated the signal
3. the returning feedback signal amplifies the original sensory representation
4. the amplified sensory representation drives workspace neurons even more strongly
5. a self-sustaining reverberant loop forms: sensory cortex <-> workspace neurons
6. the reverberant activity spreads to encompass a distributed network of cortical areas
7. lateral inhibition suppresses competing representations -- ignition is exclusive (one representation broadcasts, others are suppressed)

this process is:
- **nonlinear**: there is no gradual increase in "consciousness." it is a bifurcation -- the system jumps from one stable state (no ignition) to another (full ignition)
- **all-or-none**: partial ignition does not occur in the standard model. a representation either fully ignites or it does not
- **self-sustaining**: once ignited, the reverberant activity persists for ~200-300 ms without continued sensory input
- **exclusive**: only one coherent representation can occupy the workspace at a time. ignition of one representation suppresses competitors

### cellular substrate

the neurons that implement ignition are primarily:
- large pyramidal cells in cortical layer II/III: these have long-range horizontal connections that link distant cortical areas within the same hierarchical level
- large pyramidal cells in cortical layer V: these have long-range feedback projections to earlier areas and projections to subcortical structures (thalamus, basal ganglia, brainstem)
- workspace neurons are densely interconnected by reciprocal excitatory connections, creating the conditions for reverberant activity
- feedforward connections are primarily fast AMPA-mediated
- feedback connections are primarily slower NMDA-mediated, which is critical: NMDA receptors have voltage-dependent Mg2+ block that creates a nonlinear activation threshold and sustained depolarization once unblocked

the [[thalamocortical_loops|thalamus]] participates in ignition through cortico-thalamo-cortical loops: cortical workspace neurons project to higher-order thalamic nuclei, which project back to cortex, adding an additional reverberant loop that sustains ignition.

## neural signatures

### P3b / late positive component

the most robust ERP marker of conscious access. the P3b is a positive deflection over parietal electrodes peaking at ~300-500 ms post-stimulus. source modeling locates its generators in inferior prefrontal cortex, with simultaneous reactivation of early visual areas. the P3b is:
- present for consciously perceived stimuli, absent for unperceived stimuli
- all-or-none: its amplitude does not scale gradually with stimulus visibility but shows a bimodal distribution (present or absent)
- modality-independent: it occurs for visual, auditory, and somatosensory conscious access

### late gamma burst

enhanced high-frequency gamma-band power (~30-100 Hz) accompanies ignition. this gamma activity is:
- late-onset (~200-300 ms), distinguishing it from early stimulus-evoked gamma
- distributed across prefrontal and parietal cortex, not localized to sensory cortex
- accompanied by enhanced long-range gamma-band synchrony (phase coherence) between distant cortical areas, consistent with a broadcast mechanism
- see [[gamma_oscillations]] for the cellular mechanisms underlying gamma generation

### long-range synchrony

ignition produces enhanced synchrony between prefrontal/parietal workspace areas and the sensory cortex encoding the conscious content:
- beta-band (~15-30 Hz) synchrony in feedback direction (prefrontal -> sensory)
- gamma-band (~30-100 Hz) synchrony in feedforward direction (sensory -> prefrontal)
- this bidirectional frequency-specific synchrony is consistent with the communication through coherence framework (Fries 2015, see [[neural_synchrony]])

### sustained prefrontal activation

fMRI studies show that conscious perception is associated with sustained activation of prefrontal cortex that outlasts the stimulus (Dehaene et al. 2001; Lau & Passingham 2006). this sustained activity is interpreted as the reverberant loop maintaining information in the workspace. unconscious stimuli produce only transient, quickly decaying activation that does not reach prefrontal cortex or does not sustain.

## computational models

### the Dehaene-Changeux model

the most developed computational model of ignition. a large-scale cortical network with:
- modular structure: multiple "areas" representing sensory processors and workspace
- within-area recurrent excitation with inhibitory stabilization (see [[excitatory_inhibitory_balance]])
- sparse long-range excitatory connections between areas (workspace connections)
- NMDA-like slow excitation for feedback connections
- AMPA-like fast excitation for feedforward connections

the model reproduces:
- threshold behavior: weak inputs produce only local activation, strong inputs trigger ignition
- all-or-none dynamics: a bimodal distribution of late cortical activity
- attentional modulation: top-down bias lowers the ignition threshold for attended stimuli
- attentional blink: an ongoing ignition prevents new ignitions for ~200-500 ms
- masking: a second stimulus that arrives before ignition can abort the first signal's access

### bifurcation analysis

mathematically, ignition is a **saddle-node bifurcation** (a critical point where a stable and an unstable fixed point collide and annihilate, causing the system to jump to a different state) in the network's dynamics. the system has two stable fixed points: a low-activity state (no ignition) and a high-activity state (full ignition). the input strength acts as a **bifurcation parameter** (the control variable that triggers the state transition). below threshold, the system remains at the low fixed point. above threshold, the low fixed point disappears and the system jumps to the high fixed point. this explains the all-or-none character: near the bifurcation point, small changes in input produce large changes in network state.

ML analog: the bifurcation is analogous to the phase transition in training dynamics, where a small change in learning rate or batch size can cause a qualitative shift in what the model learns. more directly, the hard threshold in ternary spikes implements a step-function version of this bifurcation -- no gradual transition, just on or off.

the bifurcation threshold depends on:
- strength of long-range recurrent connections (stronger = lower threshold)
- level of top-down attentional modulation (more attention = lower threshold)
- ongoing spontaneous activity (can assist or interfere with ignition)
- inhibitory tone (excessive inhibition raises the threshold, see [[excitatory_inhibitory_balance]])

## subliminal processing

stimuli that fail to trigger ignition can still undergo substantial processing:
- feedforward semantic processing: masked words prime associated words (Marcel 1983)
- feedforward emotional processing: masked fearful faces activate the amygdala (Whalen et al. 1998)
- motor preparation: masked arrows can activate motor cortex (Dehaene et al. 1998)
- numerical processing: masked numbers activate parietal number areas (Naccache & Dehaene 2001)

this processing is:
- feedforward only (no recurrent amplification)
- short-lived (decays within ~500 ms)
- not reportable (not accessible to language or voluntary action)
- not flexible (cannot be combined with other information)
- not memorizable (does not enter long-term memory)

the contrast between rich subliminal processing and the additional capabilities that consciousness provides is one of the strongest arguments for GWT: consciousness adds broadcast, not computation.

## criticisms

### is ignition the cause or consequence of consciousness?

ignition could be a neural correlate of consciousness without being consciousness itself. the late (~300 ms) timing of ignition suggests it may reflect post-perceptual processing (decision, report preparation) rather than the moment of conscious experience. no-report paradigms (where subjects do not need to report their experience) show reduced prefrontal ignition markers, suggesting some of the signal reflects report demands.

### is ignition truly all-or-none?

some recent evidence suggests graded levels of conscious access rather than a sharp threshold (Windey et al. 2014; Overgaard et al. 2006). forced-choice visibility ratings show a continuous distribution in some paradigms, challenging the strictly bimodal prediction. ignition may be more accurately described as a steep sigmoidal function rather than a perfect step function.

### relationship to recurrent processing theory

Lamme (2006) proposed that consciousness requires recurrent processing but not necessarily global ignition. local recurrent loops within visual cortex may be sufficient for phenomenal consciousness (seeing), while global ignition is required for access consciousness (reporting). this view predicts a richer unconscious experience than GWT allows.

## relationship to other mechanisms

- [[global_workspace_theory]]: ignition is the mechanism that implements workspace access
- [[thalamocortical_loops]]: the thalamus participates in reverberant ignition loops
- [[selective_attention]]: attention modulates the ignition threshold
- [[gamma_oscillations]]: late gamma bursts are a signature of ignition
- [[neural_synchrony]]: long-range synchrony reflects the broadcast component of ignition
- [[excitatory_inhibitory_balance]]: E/I balance determines whether the network can support bistability (the prerequisite for all-or-none ignition)
- [[canonical_microcircuit]]: recurrent amplification within the canonical microcircuit is the local building block of ignition
- [[apical_amplification]]: BAC firing in L5 pyramidal cells provides a cellular mechanism for the coincidence between bottom-up input and top-down feedback that triggers ignition

## key references

- Dehaene, S. & Changeux, J. P. (2005). Ongoing spontaneous activity controls access to consciousness: a neuronal model for inattentional blindness. PLOS Biology, 3(5), e141.
- Dehaene, S., Charles, L., King, J. R. & Marti, S. (2014). Toward a computational theory of conscious processing. Current Opinion in Neurobiology, 25, 76-84.
- Dehaene, S. & Changeux, J. P. (2011). Experimental and theoretical approaches to conscious processing. Neuron, 70(2), 200-227.
- Mashour, G. A., Roelfsema, P., Changeux, J. P. & Dehaene, S. (2020). Conscious Processing and the Global Neuronal Workspace Hypothesis. Neuron, 105(5), 776-798.
- Deco, G. & Kringelbach, M. L. (2017). Hierarchy of information processing in the brain: a novel 'intrinsic ignition' framework. Neuron, 94(5), 961-968.
- Lamme, V. A. (2006). Towards a true neural stance on consciousness. Trends in Cognitive Sciences, 10(11), 494-501.

## see also

- [[global_workspace_theory]]
- [[integrated_information_theory]]
- [[thalamocortical_loops]]
- [[selective_attention]]
- [[gamma_oscillations]]
- [[apical_amplification]]
