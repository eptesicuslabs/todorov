# autonomic and interoception research

curated peer-reviewed research on the body-brain interface: interoception (sensing the body's internal state), autonomic regulation, allostasis, and how visceral signals shape perception, decision-making, and cognition. the brain is not a disembodied computer -- it continuously monitors and predicts the body's physiological state, and this body-state information modulates virtually every cognitive process. these findings constrain claims about what a disembodied neural computer can and cannot capture.

## interoception and the insula

### the insular cortex as body-state map

craig, a. D. (2002). how do you feel? interoception: the sense of the physiological condition of the body. *nature reviews neuroscience*, 3, 655-666.

craig, a. D. (2009). how do you feel -- now? the anterior insula and human awareness. *nature reviews neuroscience*, 10, 59-70.

key finding: craig established that the insular cortex contains a topographic map of the body's internal state -- a moment-by-moment representation of temperature, pain, itch, visceral sensations, muscle tension, hunger, thirst, and autonomic arousal. sensory afferents from the body project via the lamina i spinothalamocortical pathway and vagus nerve to the posterior insula, which constructs a primary interoceptive map. this map is progressively re-represented anteriorly, integrating with emotional, cognitive, and contextual information, until the anterior insula generates a unified "feeling state" -- craig's proposed substrate for subjective awareness. the right anterior insula is particularly associated with awareness of bodily states.

relevance to neural computer: todorov has no interoceptive channel -- no representation of system state (temperature, memory utilization, computational load) that feeds back into processing. biological cognition is continuously modulated by body-state signals. the insular body map suggests that even a disembodied system could benefit from a "system state" representation that monitors internal metrics (state saturation, gradient magnitudes, attention entropy) and modulates processing accordingly. this would be a form of meta-cognition grounded in the system's own operational state rather than external input.

confidence: high. the insular cortex's role in interoception is well-established across neuroimaging, lesion, and electrophysiological studies. caveat: craig's strong claim that the anterior insula is the seat of subjective awareness is debated; other structures (especially thalamus and prefrontal cortex) are also implicated.

## the vagus nerve

### vagal afference and brain modulation

the vagus nerve (cranial nerve x) is 80% afferent -- 80% of its fibers carry information from the body to the brain, not from the brain to the body. this makes the vagus primarily a sensory nerve that reports visceral state (gut distension, cardiac rhythm, respiratory status, inflammation markers) to the brainstem (nucleus of the solitary tract), which relays to hypothalamus, amygdala, insula, and prefrontal cortex.

### gut microbiome modulates brain gaba via vagus

bravo, j. a., forsythe, p., chew, m. v., escaravage, e., savignac, h. m., dinan, t. g., bienenstock, j., & cryan, j. f. (2011). ingestion of lactobacillus strain regulates emotional behavior and central gaba receptor expression in a mouse via the vagus nerve. *proceedings of the national academy of sciences*, 108(38), 16050-16055.

key finding: oral administration of a specific lactobacillus strain to mice changed gaba receptor expression in multiple brain regions (increased gaba-b in cingulate and prefrontal cortex, decreased in hippocampus and amygdala) and reduced anxiety and depression-like behaviors. critically, vagotomy (cutting the vagus nerve) completely abolished both the neurochemical and behavioral effects, demonstrating that the gut-brain communication was mediated entirely through the vagus nerve. this establishes a direct causal pathway from gut microbiome composition to brain neurotransmitter receptor expression and behavior.

relevance to neural computer: the gut-brain axis demonstrates that the brain's computational parameters (receptor expression, which determines sensitivity to neurotransmitters) are modulated by signals from an entirely different organ system. this is an extreme form of context-dependent processing: the brain's transfer function changes based on what is in the gut. todorov's parameters are fixed after training -- there is no mechanism for environmental signals to modulate the model's sensitivity or processing mode. the vagus pathway suggests that a truly biological-like system would have external modulatory channels that continuously adjust internal processing based on the broader context.

confidence: high. clean experimental design with vagotomy control establishing causal necessity of the vagal pathway. caveat: the specific lactobacillus strain and its effects may not generalize to all gut bacteria or to humans. the gut-brain field is rapidly evolving with many overstated claims; this study is one of the more rigorous.

## cardiac cycle and perception

### heartbeat timing gates threat perception

garfinkel, s. n., minati, l., gray, m. a., seth, a. k., dolan, r. j., & critchley, h. D. (2014). fear from the heart: sensitivity to fear stimuli depends on individual heartbeats. *journal of neuroscience*, 34(19), 6573-6582.

key finding: the perception of fear-related stimuli (fearful faces) is modulated by the cardiac cycle on a beat-by-beat basis. fearful faces presented during cardiac systole (when the heart is contracting and baroreceptors are firing) are perceived as more intense and are detected more accurately than the same faces presented during diastole (between beats). neutral faces showed no cardiac cycle modulation. the effect was specific to fear and correlated with individual differences in interoceptive sensitivity (ability to count one's own heartbeats). this demonstrates that basic threat perception is temporally gated by visceral afferent signals on a sub-second timescale.

relevance to neural computer: the cardiac gating of fear perception shows that the body's rhythmic physiological signals create temporal windows that modulate cognitive processing. this is a form of oscillatory gating (see [[neural_synchrony]]) driven by a peripheral oscillator (the heart) rather than a central one (cortical oscillations). todorov processes each token uniformly -- there is no periodic modulation of processing sensitivity. a biological analog would involve cyclically varying the gain or threshold of the ternary spike mechanism, creating temporal windows of heightened and reduced sensitivity.

confidence: high. well-controlled psychophysics with cardiac cycle measurement. the cardiac-systole fear enhancement has been replicated. caveat: the effect sizes are modest (d~0.3-0.5), and the specificity to fear (vs other emotions) is debated in subsequent studies.

## allostasis and predictive regulation

### allostatic prediction

sterling, p. & eyer, j. (1988). allostasis: a new paradigm to explain arousal pathology. in s. fisher & j. reason (eds.), *handbook of life stress, cognition and health*. john wiley & sons.

barrett, l. f. (2017). the theory of constructed emotion: an active inference account of interoception and categorization. *social cognitive and affective neuroscience*, 12(1), 1-23.

key finding: allostasis (sterling & eyer) replaces the homeostasis concept: instead of reactively correcting deviations from a setpoint, the brain predictively regulates the body by anticipating needs before they arise. the brain maintains a predictive model of the body's metabolic and physiological requirements and proactively adjusts autonomic, endocrine, and behavioral responses. barrett extends this to emotion: emotions are the brain's categorization of predicted body-state changes. a feeling of "anxiety" is the brain's prediction that the body will need to mobilize energy for an uncertain threat, not a reactive response to a detected threat. allostatic regulation is the brain's primary function -- all other cognition (perception, memory, reasoning) serves the goal of keeping the body alive.

relevance to neural computer: allostasis frames the brain as fundamentally a body-regulation system that has been co-opted for cognition. this is the strongest argument against disembodied ai capturing the full scope of biological intelligence: if the brain's core function is body regulation, and cognition is a derivative of that function, then cognition without a body may be missing the foundational layer on which it is built. for todorov specifically, the allostatic framework suggests that the model's predictions should be evaluated not just for accuracy but for their utility in serving a goal -- which requires having a goal. next-token prediction is a goal, but it is not grounded in survival needs the way biological prediction is.

confidence: medium-high for allostasis as a useful framework. the specific claim that all cognition serves body regulation is debated -- some researchers view cognition as having evolved additional functions beyond body management. barrett's constructed emotion theory is influential but contested. caveat: allostasis is more a framework than a falsifiable theory; it reframes existing findings rather than making novel predictions.

## pain as prediction

### gate control theory

melzack, r. & wall, p. d. (1965). pain mechanisms: a new theory. *science*, 150(3699), 971-979.

key finding: melzack and wall proposed that pain perception is not a direct readout of nociceptor activation but is gated by a spinal mechanism that integrates nociceptive input (small-diameter c and a-delta fibers) with non-nociceptive input (large-diameter a-beta fibers) and descending signals from the brain. the "gate" in the spinal cord dorsal horn can be opened (enhancing pain) or closed (reducing pain) by these competing inputs. this explained why rubbing an injury reduces pain (a-beta activation closes the gate) and why psychological state modulates pain intensity (descending brain signals adjust the gate). the theory was the first to propose that pain is a computed percept, not a sensory readout.

relevance to neural computer: gate control theory established the principle that even at the lowest level of sensory processing, the brain computes percepts rather than passively receiving them. the spinal gate is a simple competitive circuit (excitatory and inhibitory inputs competing to determine output) that prefigures the gain control and gating mechanisms throughout the brain. todorov's ternary spike mechanism implements a simple gate (values below threshold are zeroed), but without the competitive modulation that makes biological gating context-dependent.

confidence: high for the principle of pain modulation by competing inputs. the specific circuitry proposed by melzack and wall has been revised substantially, but the core concept (pain is gated, not direct) is universally accepted.

### the neuromatrix theory

melzack, r. (1999). from the gate to the neuromatrix. *pain*, 82 (suppl 1), s121-s126.

key finding: melzack extended gate control to the neuromatrix theory: pain (and body awareness generally) is generated by a widely distributed brain network (the "neuromatrix") that constructs a body image from multiple inputs -- somatosensory, limbic-emotional, and cognitive-evaluative. phantom limb pain demonstrates that the neuromatrix can generate vivid pain experiences without any peripheral input, proving that pain is a brain-generated output, not a peripheral signal. the neuromatrix continuously generates a "neurosignature" -- a characteristic pattern of neural activity that defines the felt body -- which is modulated but not determined by sensory input.

relevance to neural computer: the neuromatrix demonstrates that the brain generates representations (body image, pain) even in the absence of corresponding input. this is relevant to todorov's generative capabilities: the model generates text without corresponding perceptual input, which is computationally analogous to phantom limb generation -- constructing output from internal state rather than external stimulus. the neuromatrix suggests that robust internal models require distributed representation across multiple processing streams, not localized to a single module.

confidence: medium-high. phantom limb phenomena are well-documented and support the brain-generated nature of body representation. the neuromatrix concept is more a framework than a specific circuit model. caveat: the term "neuromatrix" is vague and has not led to specific testable predictions beyond what existing neuroscience already addresses.

## somatic markers and decision-making

### somatic marker hypothesis

damasio, a. R. (1994). *descartes' error: emotion, reason, and the human brain*. putnam.

damasio, a. R. (1996). the somatic marker hypothesis and the possible functions of the prefrontal cortex. *philosophical transactions of the royal society of london b*, 351(1346), 1413-1420.

key finding: damasio proposed that decision-making in uncertain or complex situations relies on "somatic markers" -- body-state signals (gut feelings, skin conductance changes, heart rate shifts) that tag decision options with positive or negative valence based on prior experience. when facing a decision, the brain reactivates the body-state patterns associated with previous outcomes of similar decisions, and these reactivated body states bias the decision process before conscious deliberation. the ventromedial prefrontal cortex (vmpfc) is critical for generating and using somatic markers; patients with vmpfc damage make catastrophically poor decisions despite intact intellectual abilities.

relevance to neural computer: somatic markers are a form of learned bias that operates through body-state simulation rather than explicit reasoning. todorov generates decisions (next-token predictions) based on learned weight patterns without any affective or body-state component. the somatic marker hypothesis suggests that decision quality in complex/ambiguous situations may depend on an additional signal channel (valence/confidence) beyond the primary prediction. this connects to the idea of confidence calibration: a model that knows when it is uncertain (a form of metacognition) may make better decisions than one that always outputs maximum-confidence predictions.

confidence: medium. the iowa gambling task (igt), the primary experimental test of the somatic marker hypothesis, has been criticized for methodological confounds. vmpfc patients do show decision-making deficits, but whether somatic markers per se are the mechanism is debated. caveat: the somatic marker hypothesis may conflate multiple functions of vmpfc (value computation, emotional regulation, contextual memory) under a single label.

## polyvagal theory

### polyvagal theory: clinical utility vs empirical support

porges, s. v. (1995). orienting in a defensive world: mammalian modifications of our evolutionary heritage. a polyvagal theory. *psychophysiology*, 32(4), 301-318.

grossman, p. (2023). fundamental challenges and likely refutations of the five basic premises of the polyvagal theory. *biological psychology*, 180, 108589.

key finding: porges' polyvagal theory proposes three phylogenetically ordered autonomic subsystems: (1) the dorsal vagal complex (reptilian, mediating immobilization/freezing), (2) the sympathetic nervous system (mobilization/fight-or-flight), and (3) the ventral vagal complex (mammalian, mediating social engagement and calm states). the theory is widely used in clinical psychology and trauma therapy. however, grossman's systematic critique challenges all five core premises: the phylogenetic ordering is not supported by comparative anatomy, the dorsal vs ventral vagal distinction is oversimplified, the claimed link between vagal tone and social engagement is correlational at best, cardiac vagal tone measurement conflates multiple autonomic processes, and the "neuroception" concept (unconscious threat detection driving autonomic shifts) is unfalsifiable.

relevance to neural computer: polyvagal theory is included here as a cautionary example of a framework that is clinically useful but empirically weak. the mapping of physiological states to computational modes (freeze, fight/flight, social engagement) is appealing but oversimplified. for todorov, the lesson is that intuitively appealing biological analogies (e.g., "the model has different operating modes like the autonomic nervous system") must be validated empirically, not just theoretically. the three-mode framework does not survive scrutiny even in its biological domain.

confidence: low for polyvagal theory's specific neuroanatomical claims. the clinical utility is pragmatic, not scientific. grossman's critique is well-supported by comparative physiology evidence.

## relevance to todorov

### validated connections
- gate control theory (competing inputs modulating output) is structurally similar to ternary spike gating and attention mechanisms
- allostatic prediction (brain predicts body needs) aligns with todorov's predictive architecture -- both systems generate predictions about future states
- somatic markers as learned decision biases parallel learned weight patterns that bias next-token predictions

### challenged assumptions
- todorov has no interoceptive channel -- no monitoring of its own internal state (memory utilization, gradient health, state saturation)
- no body-state modulation of processing: every token is processed with the same gain regardless of system state
- no cardiac-like oscillatory gating: processing is uniform across time, not periodically modulated
- no allostatic goal: predictions serve next-token accuracy, not survival or body regulation
- polyvagal-style mode-switching is not supported even in biology -- simple multi-mode models are too coarse

### future phases
- system-state monitoring: internal metrics (state saturation, attention entropy, gradient magnitude) fed back as auxiliary input to modulate processing
- confidence gating: somatic marker-like mechanism that tags predictions with certainty estimates and adjusts behavior in low-confidence regimes
- oscillatory processing modulation: periodic variation in spike threshold or attention temperature (phase 6+)

## see also

- [[free_energy_principle]]
- [[predictive_coding]]
- [[precision_weighting]]
- [[dopamine_system]]
- [[decision_and_emotion_research]]
- [[perception_and_consciousness_research]]
