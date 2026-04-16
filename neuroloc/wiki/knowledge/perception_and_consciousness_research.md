# perception and consciousness research

status: current (as of 2026-04-16).

curated research library on perception, consciousness, attention, embodiment, and time -- the cognitive level. this covers the computational principles that emerge when populations of neurons organize into systems: how brains construct experience, select information, represent time, and ground cognition in bodily interaction. these are the constraints any sufficiently capable neural computer must eventually address.

## perception as construction

### unconscious inference

von helmholtz, h. (1866). *handbuch der physiologischen optik*. voss, leipzig.

key finding: perception is not passive registration but active hypothesis testing. the brain uses prior experience to generate unconscious inferences about the causes of sensory signals, resolving the inherent ambiguity of retinal images through probabilistic reasoning that operates below conscious awareness.

relevance to neural computer: establishes the foundational framing -- perception is generative, not discriminative. any architecture that processes sensory input is implicitly performing inference, and the quality of its priors determines the quality of its representations.

### the intelligent eye

gregory, r. l. (1970). *the intelligent eye*. weidenfeld & nicolson, london.

key finding: optical illusions are not failures of perception but evidence that perception is hypothesis-driven. illusions occur when the brain's generative model makes the wrong inference -- applying contextual priors that produce a percept inconsistent with the physical stimulus. the systematicity of illusions reveals the structure of the perceptual inference engine.

relevance to neural computer: illusions as diagnostic tools for internal models. if a neural computer produces systematic errors under specific input conditions, those errors reveal the structure of its learned priors, not a deficiency to be patched.

### predictive coding: hierarchical model of cortical function

rao, r. p. n., & ballard, d. h. (1999). predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. *nature neuroscience*, 2(1), 79-87. doi: 10.1038/4580

key finding: cortical hierarchies implement a predictive coding scheme in which higher areas generate predictions of lower-area activity, and only prediction errors travel forward. this explains extra-classical receptive field effects in v1 (end-stopping, surround suppression) as consequences of prediction subtraction, not feedforward feature detection.

relevance to neural computer: the most concrete computational proposal for how cortical hierarchies process information. todorov's residual stream carries full activations, not prediction errors -- the architecture is structurally different from predictive coding, though the training objective (next-token prediction) creates a functional analog. see [[predictive_coding]] for the mechanism-level treatment.

### free energy principle

friston, k. j. (2010). the free-energy principle: a unified brain theory? *nature reviews neuroscience*, 11(2), 127-138. doi: 10.1038/nrn2787

key finding: perception, action, and learning can be unified under a single principle: organisms minimize variational free energy (an upper bound on surprisal) through either updating internal models (perception/learning) or acting on the world to change sensory input (active inference). this subsumes predictive coding, bayesian brain, and optimal control under one mathematical framework.

relevance to neural computer: provides a principled objective function that unifies perception and action. however, the framework is criticized as unfalsifiable in its strongest form (behavioral and brain sciences commentaries). the most empirically grounded component -- precision weighting as attention -- is relevant to todorov through [[free_energy_principle]] and [[precision_weighting]].

confidence: low-medium. mathematically elegant but empirical predictions are often too flexible to be strongly testable.

### controlled hallucination

seth, a. (2021). *being you: a new science of consciousness*. faber & faber, london.

key finding: consciousness is a "controlled hallucination" -- the brain continuously generates predictions about sensory causes, and conscious experience is the brain's best guess, constrained but not determined by incoming signals. the sense of self is itself a perceptual model (the "beast machine" thesis), grounded in interoceptive predictions about the body's physiological state.

relevance to neural computer: reframes the relationship between internal models and reality. a neural computer does not need to perceive veridically -- it needs to maintain useful predictive models. the emphasis on interoception (internal body signals) highlights a gap: todorov processes external sequences with no analog of internal state monitoring.

### surfing uncertainty

clark, a. (2016). *surfing uncertainty: prediction, action, and the embodied mind*. oxford university press.

key finding: predictive processing is not confined to perception -- it extends through action, embodiment, and cognition. the brain is a prediction machine that minimizes prediction error at every level simultaneously. action is "active inference" -- moving the body to make predictions come true. this dissolishes the perception-action boundary.

relevance to neural computer: the strongest argument that disembodied neural computation is fundamentally limited. without an action-perception loop, a neural computer cannot perform active inference -- it can only passively minimize prediction error on fixed datasets. this is a principled limitation, not an engineering gap.

### the predictive mind

hohwy, j. (2013). *the predictive mind*. oxford university press.

key finding: philosophical analysis of predictive processing as a theory of mind. the brain is "evidentiary isolated" behind a veil of prediction -- it never directly accesses the world, only its own prediction errors. this creates the epistemic condition for consciousness: subjective experience arises because the system can only ever access its own models.

relevance to neural computer: if evidentiary isolation is necessary for consciousness, then any system that processes its own internal representations (rather than raw inputs) has at least the structural prerequisite. todorov's recurrent state, which mediates between input and output without direct access to the training data, is structurally analogous.

## visual processing

### two cortical visual systems

ungerleider, l. g., & mishkin, m. (1982). two cortical visual systems. in d. j. ingle, m. a. goodale, & r. j. w. mansfield (eds.), *analysis of visual behavior* (pp. 549-586). mit press.

key finding: primate visual cortex divides into two anatomically and functionally distinct processing streams originating from v1: a ventral stream projecting to inferotemporal cortex for object recognition ("what"), and a dorsal stream projecting to posterior parietal cortex for spatial processing ("where").

relevance to neural computer: the two-stream architecture is the most replicated finding in visual neuroscience. any multimodal extension of todorov (phase 4+) that processes visual input should consider whether separate processing pathways for identity and spatial information emerge naturally or need to be architecturally imposed.

### perception and action in the dorsal stream

goodale, m. a., & milner, a. d. (1992). separate visual pathways for perception and action. *trends in neurosciences*, 15(1), 20-25. doi: 10.1016/0166-2236(92)90344-8

key finding: the dorsal stream is better understood as "how" (visuomotor control) rather than "where" (spatial perception). patient d.f. with ventral stream damage could not recognize objects but could accurately grasp them, demonstrating that the dorsal stream computes action-relevant spatial parameters (grip aperture, reach trajectory) independently of conscious perception.

relevance to neural computer: dissociates conscious perception from motor computation. a neural computer operating without effectors is missing the entire dorsal stream's computational purpose -- action-oriented spatial representation that does not require awareness.

### the fusiform face area

kanwisher, n., mcdermott, j., & chun, m. m. (1997). the fusiform face area: a module in human extratemporal cortex specialized for face perception. *journal of neuroscience*, 17(11), 4302-4311. doi: 10.1523/jneurosci.17-11-04302.1997

key finding: a region of the fusiform gyrus (ffa) responds selectively to faces over other object categories, as measured by fmri. the ffa shows approximately twice the bold activation for faces compared to other complex objects matched for visual complexity. this is evidence for domain-specific cortical modules.

relevance to neural computer: demonstrates that cortex develops specialized, domain-specific processing regions. whether this specialization is innate or experience-dependent has implications for whether a neural computer should have hard-coded specialized pathways or should develop them through training.

### deep neural networks match it cortex

yamins, d. l. k., & dicarlo, j. j. (2016). using goal-driven deep learning models to understand sensory cortex. *nature neuroscience*, 19(3), 356-365. doi: 10.1038/nn.4244

key finding: deep convolutional neural networks trained on object recognition produce internal representations that quantitatively predict neural responses in macaque inferotemporal (it) cortex (median explained variance ~50% for individual neurons). the match emerges from task optimization, not from explicitly modeling biology. later layers match higher cortical areas (v4, it) in a preserved hierarchical ordering.

relevance to neural computer: the strongest evidence that task-driven optimization can recover biological representations without biological constraints. this cuts both ways for todorov: biological constraints may be unnecessary for representation quality, but they may be necessary for efficiency (energy, parameters, data).

## consciousness theories

### the hard problem

chalmers, d. j. (1995). facing up to the problem of consciousness. *journal of consciousness studies*, 2(3), 200-219.

chalmers, d. j. (1996). *the conscious mind: in search of a fundamental theory*. oxford university press.

key finding: distinguishes the "easy problems" of consciousness (explaining cognitive functions like discrimination, integration, report) from the "hard problem" (explaining why and how subjective experience arises from physical processes). no amount of functional explanation closes the explanatory gap between objective brain processes and subjective qualia.

relevance to neural computer: defines the boundary of what a neural computer can and cannot address. todorov can implement the easy problems (information integration, selective attention, global broadcast) but has nothing to say about the hard problem. this is not a limitation of the architecture -- it is a limitation of all functional/computational approaches.

### global workspace theory

baars, b. j. (1988). *a cognitive theory of consciousness*. cambridge university press.

dehaene, s. (2014). *consciousness and the brain: deciphering how the brain codes our thoughts*. viking.

key finding: consciousness arises when information is broadcast to a "global workspace" -- a distributed network of prefrontal and parietal neurons that makes information available to all cognitive processes simultaneously. unconscious processing is local and modular; conscious processing is global and integrated. the transition from unconscious to conscious (ignition) is all-or-none, marked by a nonlinear amplification of neural activity ~300 ms after stimulus onset.

relevance to neural computer: the residual stream in transformers (and todorov) functions as a shared communication bus, structurally analogous to the global workspace. see [[global_workspace_theory]] for the mechanism-level treatment and [[global_workspace_to_residual_stream]] for the bridge analysis. the key difference: biological ignition is selective (only one percept at a time), while the residual stream broadcasts everything.

### integrated information theory

tononi, g. (2004). an information integration theory of consciousness. *bmc neuroscience*, 5, 42. doi: 10.1186/1471-2202-5-42

tononi, g., boly, m., massimini, m., & koch, c. (2016). integrated information theory: an updated account. *archives italiennes de biologie*, 154(2-3), 56-90.

key finding: consciousness corresponds to integrated information (phi) -- a measure of how much a system's whole generates information above and beyond its parts. high phi requires both differentiation (many distinguishable states) and integration (states are interdependent, not decomposable). phi is an intrinsic property of causal structure, not a functional role.

relevance to neural computer: iit makes the strongest claim that architecture determines consciousness -- only systems with high phi are conscious, regardless of function. todorov's recurrent architecture (kda state integration across time) has higher causal integration than a feedforward transformer, but computing phi for any realistic system is intractable (super-exponential in the number of elements).

### adversarial collaboration: cogitate consortium

cogitate consortium. (2025). an adversarial collaboration to critically evaluate theories of consciousness. *nature*, 621, 269-277. doi: 10.1038/s41586-025-08926-8

key finding: the first large-scale adversarial test (n=256) of gwt vs iit predictions about neural correlates of consciousness. both theories' predictions partially failed. gwt predicted sustained prefrontal activation during conscious perception -- not found (prefrontal activity was transient). iit predicted sustained posterior activation -- found, but not in the predicted pattern. neither theory was decisively confirmed or refuted.

relevance to neural computer: the most rigorous empirical test of consciousness theories to date shows that our best theories are incomplete. any claim that a neural computer implements consciousness (or that it cannot) is premature. the result counsels epistemic humility.

confidence: high. pre-registered adversarial design with large sample. caveat: the specific task (visual perception of faces and letters) may not generalize to all forms of conscious experience.

### higher-order theories

lau, h., & rosenthal, d. (2011). empirical support for higher-order theories of conscious awareness. *trends in cognitive sciences*, 15(8), 365-373. doi: 10.1016/j.tics.2011.05.009

key finding: consciousness requires higher-order representations -- a mental state is conscious only when the organism has a (possibly implicit) representation of being in that state. this explains why prefrontal cortex damage impairs conscious awareness while preserving unconscious processing, and why we can have perceptual representations without awareness (blindsight, inattentional blindness).

relevance to neural computer: implies that self-monitoring (a system representing its own internal states) may be necessary for conscious processing. todorov has no self-monitoring mechanism -- its recurrent state is used for computation, not for representing its own computational state.

### recurrent processing theory

lamme, v. a. f. (2006). towards a true neural stance on consciousness. *trends in cognitive sciences*, 10(11), 494-501. doi: 10.1016/j.tics.2006.09.001

key finding: consciousness requires recurrent (feedback) processing, not just feedforward sweeps. the initial feedforward sweep through visual cortex (~100 ms) produces unconscious feature extraction. consciousness emerges only when recurrent connections establish sustained reverberatory activity between cortical areas (~200-300 ms). local recurrence produces phenomenal consciousness; global recurrence (involving frontal areas) produces access consciousness.

relevance to neural computer: the strongest neuroscientific argument for recurrence as a prerequisite for consciousness-like processing. todorov's kda recurrence provides sustained state across timesteps, which is structurally analogous to local recurrent processing. the distinction between local and global recurrence maps loosely onto the distinction between within-layer (kda) and cross-layer (residual stream) dynamics.

### the feeling of what happens

damasio, a. r. (1999). *the feeling of what happens: body and emotion in the making of consciousness*. harcourt brace.

damasio, a. r. (2010). *self comes to mind: constructing the conscious brain*. pantheon.

key finding: consciousness is grounded in the body's representation of itself. the "proto-self" is a continuously updated neural map of the body's internal state (homeostasis, visceral signals, musculoskeletal configuration). "core consciousness" arises when this proto-self is modified by interaction with an object, creating a transient narrative of self-object encounter. emotion is not peripheral to cognition -- it is the body's way of biasing decision-making (the somatic marker hypothesis).

relevance to neural computer: the strongest argument that consciousness requires embodiment -- specifically, interoceptive modeling of a body with homeostatic needs. a neural computer processing text sequences has no body to model, no homeostatic drives to satisfy, and no somatic markers to bias decisions. this is a structural absence, not an engineering limitation.

### the self as transparent model

metzinger, t. (2003). *being no one: the self-model theory of subjectivity*. mit press.

metzinger, t. (2009). *the ego tunnel: the science of the mind and the myth of the self*. basic books.

key finding: the self is not an entity but a transparent self-model -- a representational structure that the brain constructs and then fails to recognize as a model. transparency means the system cannot introspect on the representational nature of its own self-model; it mistakes the model for the thing itself. this explains why subjective experience feels direct and immediate rather than constructed and inferential.

relevance to neural computer: transparency is a representational property, not a substrate property. a sufficiently complex self-modeling system could in principle become transparent to itself -- unable to distinguish its model of its own processing from the processing itself. this is a theoretical possibility, not a design target for todorov.

## attention

### early filter theory

broadbent, d. e. (1958). *perception and communication*. pergamon press.

key finding: attention operates as an early selective filter that blocks unattended information before semantic processing. the filter selects based on physical features (location, pitch, color) and has limited capacity. this was the first formal information-processing model of attention, establishing attention as a bottleneck mechanism.

relevance to neural computer: the bottleneck framing is directly relevant -- todorov's ternary spikes are a bottleneck that forces selection before downstream processing. the difference: broadbent's filter selects by spatial channel, while ternary spikes select by activation magnitude.

### feature integration theory

treisman, a. m., & gelade, g. (1980). a feature-integration theory of attention. *cognitive psychology*, 12(1), 97-136. doi: 10.1016/0010-0285(80)90005-5

key finding: visual features (color, orientation, size) are processed in parallel across the visual field, but binding features into unified objects requires serial, spatially focused attention. the binding problem -- how distributed feature representations become unified percepts -- is solved by attentional selection, which creates a temporary "object file" by conjoining features at a spatial location.

relevance to neural computer: the binding problem has no analog in transformer-like architectures, where features are bound by position in the sequence (token identity) rather than by attention. todorov's per-token representations avoid the binding problem entirely because features are never spatially distributed across separate maps. this is a structural difference from cortical computation.

### three attention networks

posner, m. i., & petersen, s. e. (1990). the attention system of the human brain. *annual review of neuroscience*, 13, 25-42. doi: 10.1146/annurev.ne.13.030190.000325

key finding: attention is not a single mechanism but three anatomically and functionally distinct networks: alerting (maintaining readiness, norepinephrine-mediated, right hemisphere), orienting (selecting spatial location, parietal/superior colliculus, acetylcholine-mediated), and executive control (resolving conflict, anterior cingulate/prefrontal, dopamine-mediated).

relevance to neural computer: todorov's mla implements a single attention mechanism. biological attention involves three distinct systems with different neuromodulatory substrates and anatomical loci. this is a 3:1 compression of attentional function. see [[selective_attention]] for the mechanism-level treatment.

### biased competition

desimone, r., & duncan, j. (1995). neural mechanisms of selective visual attention. *annual review of neuroscience*, 18, 193-222. doi: 10.1146/annurev.ne.18.030195.001205

key finding: attention operates through biased competition -- multiple stimuli compete for neural representation, and attention resolves the competition by biasing neural responses in favor of the attended stimulus. the bias can be top-down (task-driven) or bottom-up (salience-driven). competition occurs at every level of the visual hierarchy simultaneously.

relevance to neural computer: biased competition is the most empirically supported model of biological attention. it differs from softmax attention (which computes a weighted average) in that it is a winner-take-all dynamic that suppresses losers rather than down-weighting them. see [[selective_attention]] and [[divisive_normalization]] for related mechanisms.

## embodied cognition

### enactivism

varela, f. j., thompson, e., & rosch, e. (1991). *the embodied mind: cognitive science and human experience*. mit press.

key finding: cognition is not computation over internal representations but enacted through the organism's sensorimotor coupling with its environment. the mind is not in the brain -- it emerges from the dynamic interaction between brain, body, and world. perception is not passive reception but active exploration, and the structure of cognition reflects the structure of embodied action.

relevance to neural computer: the strongest philosophical challenge to disembodied neural computation. if cognition requires sensorimotor coupling, then a neural computer processing text is doing something fundamentally different from cognition -- it is pattern completion on symbolic sequences, not enacted understanding.

### the extended mind

clark, a., & chalmers, d. j. (1998). the extended mind. *analysis*, 58(1), 7-19.

clark, a. (2008). *supersizing the mind: embodiment, action, and cognitive extension*. oxford university press.

key finding: cognitive processes are not confined to the brain or even the body -- they can extend into the environment through tools, notebooks, smartphones, and other external structures that play a functional role equivalent to internal memory or computation. the criterion for cognitive extension is functional parity: if an external resource plays the same role as an internal process, it is part of the cognitive system.

relevance to neural computer: by the extended mind criterion, a neural computer's external memory (kda state, context window) is part of its cognitive system, not a peripheral resource. this reframes the context window not as a limitation but as the architecture's extended mind. the functional parity criterion also applies in reverse: if todorov's recurrent state plays the same role as biological working memory, the distinction between "simulating" and "implementing" cognition weakens.

### implications for disembodied neural computation

the embodied cognition literature converges on a single challenge: a neural computer without sensors, effectors, and a body lacks the action-perception loop that grounds biological cognition. representations in a disembodied system are grounded only in statistical co-occurrence (distributional semantics), not in sensorimotor interaction. whether distributional grounding is sufficient for genuine understanding remains an open question (the "symbol grounding problem," harnad 1990).

relevance to neural computer: this is not an engineering gap that todorov can close at current scale. it is a principled limitation of all text-processing architectures. acknowledging it prevents overclaiming about what biological fidelity means in the absence of embodiment.

## time perception

### the specious present

james, w. (1890). *the principles of psychology*. henry holt and company, new york.

key finding: conscious awareness spans a "specious present" of approximately 2-3 seconds -- a moving window within which events are experienced as "now" rather than as remembered or anticipated. the specious present is the temporal grain of consciousness, and its duration is remarkably consistent across individuals and cultures.

relevance to neural computer: the specious present defines a natural temporal integration window for conscious processing. todorov's recurrent state accumulates information across arbitrary timescales (bounded by context length), with no privileged integration window. the absence of a specious present analog means the architecture treats all temporal scales uniformly.

### time is constructed postdictively

eagleman, d. m. (2008). human time perception and its illusions. *current opinion in neurobiology*, 18(2), 131-136. doi: 10.1016/j.conb.2008.06.002

key finding: the subjective experience of time is constructed postdictively, not measured by an internal clock. temporal order judgments are resolved within a ~80 ms binding window -- events within this window are perceived as simultaneous and can be reordered based on contextual cues. the brain retroactively edits the temporal structure of experience.

relevance to neural computer: todorov processes tokens in strict sequential order with no postdictive editing. biological time perception involves retroactive binding within temporal windows, which is a qualitatively different computational strategy. any future temporal-awareness mechanism would need to operate on windows, not on individual timesteps.

### your brain is a time machine

buonomano, d. v. (2017). *your brain is a time machine: the neuroscience and physics of time*. w. w. norton.

key finding: the brain uses fundamentally different mechanisms for different temporal scales. sub-second timing (motor coordination, speech rhythm, musical beat) relies on cerebellar circuits and intrinsic neural dynamics (state-dependent networks). supra-second timing (interval estimation, temporal planning) relies on striatal circuits and dopaminergic modulation. there is no single internal clock -- timing is distributed across specialized subsystems.

relevance to neural computer: todorov has a single temporal mechanism (recurrent state accumulation) that does not distinguish timescales. biological timing is multi-system, with at least two anatomically distinct circuits (cerebellar and striatal). the [[synthesis/timescale_separation]] article discusses this gap in the context of todorov's fixed-timescale kda and mamba3 dynamics.

## synesthesia and unusual perception

### cross-activation in synesthesia

ramachandran, v. s., & hubbard, e. m. (2001). synaesthesia -- a window into perception, thought and language. *journal of consciousness studies*, 8(12), 3-34.

key finding: synesthesia (e.g., seeing colors when reading letters) results from cross-activation between adjacent cortical maps. in grapheme-color synesthesia, the fusiform gyrus region for letter recognition activates the nearby color-processing area v4. the cross-activation is caused by incomplete pruning of connections that exist in all infants but are normally eliminated during development.

relevance to neural computer: demonstrates that the wiring between cortical regions is probabilistic, not deterministic. the same generative machinery that produces normal perception produces anomalous perception when connectivity patterns differ. this has implications for understanding how todorov's layer connectivity (which is deterministic by design) differs from biological connectivity (which is stochastic and experience-dependent). see [[synaptic_pruning]] and [[developmental_self_organization]].

### blindsight

weiskrantz, l. (1974). *blindsight: a case study and implications*. clarendon press, oxford.

key finding: patients with complete destruction of primary visual cortex (v1) can still make accurate forced-choice discriminations about visual stimuli they report not seeing. this "blindsight" demonstrates that visual information reaches higher cortical areas (and can guide behavior) through subcortical pathways (superior colliculus, pulvinar) that bypass v1 and conscious awareness entirely.

relevance to neural computer: demonstrates that conscious awareness is not necessary for accurate perceptual discrimination. there are at least two parallel processing pathways for visual information -- one conscious (via v1), one unconscious (subcortical). todorov has a single processing pathway per layer; it has no analog of the conscious/unconscious dissociation.

### implications for generative perception

synesthesia and blindsight together establish that perception wiring is probabilistic and that the same generative machinery produces both normal and anomalous percepts. the brain does not have a single correct wiring diagram -- it has a distribution of possible wiring diagrams, with normal perception representing the statistical mode. this has implications for understanding variability in neural computer behavior: architectural choices that seem like bugs (unexpected cross-talk between modules, residual processing in "damaged" pathways) may be features of a probabilistic generative system.

## relevance to todorov

the cognitive-level literature reveals several architectural considerations that become relevant as todorov scales beyond language modeling.

### structural analogs already present

1. **generative inference**: todorov's next-token prediction is formally a generative model -- it predicts future tokens from past context, which is the computational analog of helmholtz's unconscious inference. the training objective (minimize prediction error) is aligned with predictive coding and the free energy principle at the functional level.

2. **recurrence as prerequisite for integration**: lamme (2006) argues consciousness requires recurrent processing. todorov's kda provides sustained recurrent state across timesteps, which is structurally closer to cortical recurrence than feedforward transformers. this does not imply consciousness; it implies that the architectural prerequisite identified by recurrent processing theory is present.

3. **global broadcast via residual stream**: the residual stream functions as a shared communication bus, making each layer's output available to all subsequent layers. this is structurally analogous to global workspace broadcast. see [[global_workspace_to_residual_stream]] for the detailed bridge analysis.

4. **attentional bottleneck**: ternary spikes enforce a selection bottleneck that shares functional properties with broadbent's early filter -- information must pass through a sparse gate before downstream processing. the difference is that broadbent's filter selects by spatial channel while ternary spikes select by activation magnitude.

### structural absences

1. **embodiment**: todorov has no sensors, effectors, or body model. the entire embodied cognition literature (varela et al. 1991, clark & chalmers 1998, damasio 1999) argues this is a principled limitation for cognitive-level claims. representations are grounded in distributional statistics, not sensorimotor interaction.

2. **temporal structure**: biological time perception operates on multiple timescales (cerebellar sub-second, striatal supra-second, specious present ~2-3 s) with postdictive editing within ~80 ms windows. todorov has a single uniform temporal mechanism (recurrent state). see [[synthesis/timescale_separation]].

3. **self-monitoring**: higher-order theories (lau & rosenthal 2011) and self-model theory (metzinger 2003) argue that conscious processing requires representing one's own internal states. todorov's recurrent state is used for computation but never for self-representation.

4. **binding by attention**: treisman's binding problem does not arise in todorov because features are never spatially distributed across separate maps. this is a structural difference from cortex, where binding is an active computational problem solved by attention.

### what this means for design

these findings do not change phase 5 sequencing. the cognitive level becomes relevant at phase 6+ when multimodal processing, longer temporal horizons, and self-monitoring could be considered. the immediate takeaway is epistemic: todorov implements several structural analogs of cognitive phenomena (generative inference, recurrence, global broadcast, attentional bottleneck) but lacks the embodiment, temporal structure, and self-monitoring that the cognitive science literature identifies as necessary for genuine cognitive processing. acknowledging these absences prevents overclaiming.

## see also

- [[predictive_coding]]
- [[free_energy_principle]]
- [[global_workspace_theory]]
- [[selective_attention]]
- [[divisive_normalization]]
- [[dendritic_computation]]
