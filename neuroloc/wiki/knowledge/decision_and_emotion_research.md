# decision making, emotion, reward, and social cognition research

curated research library on decision making, emotion, reward, intuition, social cognition, and rationality. these systems shape how neural circuits evaluate, select, and commit to actions -- processes that any neural computer architecture must eventually address.

## evidence accumulation and decision

### a theory of memory retrieval / the diffusion decision model

ratcliff, r. (1978). a theory of memory retrieval. *psychological review*, 85(2), 59-108. doi: 10.1037/0033-295X.85.2.59

ratcliff, r., & mckoon, g. (2008). the diffusion decision model: theory and data for two-choice decision tasks. *neural computation*, 20(4), 873-922. doi: 10.1162/neco.2008.12-06-420

key insight: decisions arise from noisy accumulation of evidence toward one of two boundaries. the drift-diffusion model (ddm) captures reaction time distributions, accuracy, and speed-accuracy tradeoffs with four parameters: drift rate (evidence quality), boundary separation (caution), starting point (bias), and non-decision time. this is the dominant quantitative framework for perceptual and cognitive decision making.

relevance to neural computer: the ddm formalizes decision as thresholded accumulation -- structurally identical to leaky integration in recurrent state. the ternary spike threshold in todorov is a degenerate case: accumulate, then fire at criterion.

### neural basis of a perceptual decision in the parietal cortex

shadlen, m. n., & newsome, w. t. (2001). neural basis of a perceptual decision in the parietal cortex (area lip) of the rhesus monkey. *journal of neurophysiology*, 86(4), 1916-1936. doi: 10.1152/jn.2001.86.4.1916

key insight: neurons in lateral intraparietal area (lip) ramp their firing rates during perceptual decisions, reflecting the temporal accumulation of sensory evidence toward a decision bound. the ramping rate correlates with stimulus strength (motion coherence), and the response is triggered when activity crosses a threshold.

relevance to neural computer: direct neural evidence that decision circuits implement drift-diffusion dynamics. the ramp-to-threshold pattern parallels recurrent state accumulation in kda layers.

### the neural basis of decision making

gold, j. i., & shadlen, m. n. (2007). the neural basis of decision making. *annual review of neuroscience*, 30, 535-574. doi: 10.1146/annurev.neuro.29.051605.113038

key insight: comprehensive review establishing that decision-related neural signals are distributed across sensory, association, and motor areas. evidence accumulation occurs in association cortex (lip, dlpfc), with the basal ganglia and superior colliculus implementing the commitment/threshold mechanism. the framework unifies perceptual, value-based, and motor decisions under a single accumulate-to-bound model.

relevance to neural computer: maps the full decision circuit -- accumulation (recurrent layers), threshold (spike quantization), commitment (output gating). the distributed nature suggests decisions emerge from layer interactions, not any single module.

### the urgency-gating model

thura, d., & cisek, p. (2017). the basal ganglia do not select reach targets but control the urgency of commitment. *neuron*, 95(5), 1160-1170. doi: 10.1016/j.neuron.2017.07.039

key insight: the basal ganglia do not encode which action to select but control when to commit by imposing a time-varying urgency signal that lowers the decision threshold. this separates evidence quality (cortical) from commitment timing (subcortical), making the speed-accuracy tradeoff dynamic rather than a fixed parameter.

relevance to neural computer: the urgency signal is a gain modulation on the decision threshold -- functionally analogous to adaptive threshold scaling. separating "what" (layer content) from "when" (gating/commitment) is a design principle worth preserving.

### the speed-accuracy tradeoff is dynamic, not fixed

the ddm boundary separation is typically fit as a constant, but biological data show that urgency rises over time within a trial (thura & cisek 2014, 2017). this means the effective threshold drops as deliberation continues, forcing commitment even when evidence is weak. the tradeoff is not a knob set before the trial -- it is an evolving computation shaped by context, reward rate, and metabolic cost.

relevance to neural computer: a fixed spike threshold (alpha * mean(|x|)) captures the static case but not urgency dynamics. a time-varying or context-dependent threshold could improve decision-like computations in recurrent layers.

## emotion as computation

### somatic marker hypothesis

damasio, a. r. (1994). *descartes' error: emotion, reason, and the human brain*. putnam.

key insight: the somatic marker hypothesis proposes that emotions are bodily states (heart rate, gut feeling, skin conductance) that bias decision making before conscious deliberation. patients with ventromedial prefrontal cortex (vmpfc) damage make catastrophically poor real-world decisions despite intact iq, because they cannot generate these somatic priors. emotion is not the opposite of rationality -- it is a prerequisite.

relevance to neural computer: somatic markers are fast learned priors that prune the decision space. in an architecture, this maps to state-dependent biases on gating or threshold -- a form of context that shapes computation without being the computation itself.

### the feeling of what happens

damasio, a. r. (1999). *the feeling of what happens: body and emotion in the making of consciousness*. harcourt brace.

key insight: damasio proposes three nested levels of self: proto-self (brainstem homeostatic maps of body state), core self (transient second-order maps of organism-object interaction), and autobiographical self (extended memory-based narrative). consciousness arises from the organism's ability to represent its own state changes in response to objects, not from the objects themselves.

relevance to neural computer: the layered self-model suggests that any system processing temporal sequences needs at minimum a state representation (proto-self analog) and a mechanism to represent changes to that state (core self analog). the recurrent hidden state in kda/mamba3 is a proto-self at best.

### the emotional brain and survival circuits

ledoux, j. e. (1996). *the emotional brain: the mysterious underpinnings of emotional life*. simon & schuster.

ledoux, j. e. (2015). *anxious: using the brain to understand and treat fear and anxiety*. viking.

key insight: the original model proposed two pathways for threat processing: a fast "low road" (thalamus to amygdala, ~150 ms) and a slow "high road" (thalamus to cortex to amygdala, ~300 ms). the low road enables rapid defensive responses before conscious recognition. in the 2015 revision, ledoux reframed: the amygdala runs nonconscious "survival circuits" for behavioral and physiological responses, while conscious fear is a separate cognitive construction requiring cortical involvement.

relevance to neural computer: dual-pathway processing (fast approximate vs slow precise) is a recurring architectural motif. the hybrid kda/mla design echoes this: kda provides recurrent context (fast, always-on), mla provides precise retrieval (slower, selective).

### emotions as predictions

barrett, l. f. (2017). *how emotions are made: the secret life of the brain*. houghton mifflin harcourt.

key insight: the theory of constructed emotion proposes that emotions are not triggered by dedicated circuits but are predictions constructed by the brain to categorize interoceptive signals using learned concepts. there are no emotion fingerprints in the brain or body -- the same physiological state can be categorized as anger, excitement, or illness depending on context and prior experience. emotion categories are culturally learned, not innate.

relevance to neural computer: challenges the idea that distinct computational modules map to distinct functional states. the same recurrent dynamics, with different contextual priors, can produce qualitatively different outputs -- a property that emerges naturally from context-dependent recurrent processing.

### interoceptive representation in the anterior insula

craig, a. d. (2003). interoception: the sense of the physiological condition of the body. *current opinion in neurobiology*, 13(4), 500-505. doi: 10.1016/S0959-4388(03)00090-4

key insight: the anterior insula maintains a real-time representation of the body's internal state (temperature, pain, itch, visceral sensations, hunger, thirst). this interoceptive map is the neural substrate for subjective feelings and forms the basis of emotional awareness. the right anterior insula is particularly involved in conscious access to body state.

relevance to neural computer: interoception is the body's internal state monitoring channel -- analogous to a system that reads its own hidden state to modulate behavior. this is a form of meta-computation: using state about state to adjust processing.

## reward and motivation

### dopamine as temporal difference prediction error

schultz, w., dayan, p., & montague, p. r. (1997). a neural substrate of prediction and reward. *science*, 275(5306), 1593-1599. doi: 10.1126/science.275.5306.1593

key insight: midbrain dopamine neurons encode the temporal difference (td) prediction error: they fire at unexpected rewards (positive pe), are suppressed at unexpected omissions (negative pe), and shift their response from reward to the earliest predictive cue as associations are learned. this is the most successful quantitative mapping between a computational algorithm (td learning) and single-neuron physiology.

relevance to neural computer: td error is the canonical learning signal in biological reinforcement learning. see [[dopamine_system]] for full treatment. the three-factor learning rule (pre * post * modulator) uses this signal to gate plasticity.

### wanting vs liking

berridge, k. c., & robinson, t. e. (1998). what is the role of dopamine in reward: hedonic impact, reward learning, or incentive salience? *brain research reviews*, 28(3), 309-369. doi: 10.1016/S0165-0173(98)00019-8

key insight: dopamine mediates "wanting" (incentive salience, motivational drive) but not "liking" (hedonic pleasure, which depends on opioid and endocannabinoid systems). these two components of reward are neurochemically and anatomically dissociable. an animal can "want" something intensely without "liking" it, and vice versa. this dissociation explains addiction: dopamine-driven wanting persists and escalates even as liking diminishes.

relevance to neural computer: reward is not a single scalar -- it has at least two dissociable components. any architecture that uses a single reward signal (as in standard rl) is collapsing a biologically multidimensional signal. the wanting/liking split maps roughly to value prediction vs hedonic evaluation.

### subjective value encoding in ofc

padoa-schioppa, c., & assad, j. a. (2006). neurons in the orbitofrontal cortex encode economic value. *nature*, 441(7090), 223-226. doi: 10.1038/nature04676

key insight: neurons in the orbitofrontal cortex (ofc) encode subjective value on a common scale, independent of the specific goods being compared. when monkeys choose between different juice types and quantities, ofc neurons represent the value of each option in a currency that allows direct comparison. this is the neural implementation of a "common value currency" for economic decision making.

relevance to neural computer: value computation requires a shared representation space where heterogeneous inputs can be compared. this is analogous to projecting diverse features into a common latent space for comparison -- which is what the compression stage (C) in crbr achieves.

### basal ganglia action selection

the basal ganglia implement action selection through a direct pathway (go, d1 receptors) that facilitates desired actions and an indirect pathway (nogo, d2 receptors) that suppresses competing alternatives. this creates a center-surround organization in action space: the selected action is disinhibited while competitors are actively suppressed. dopamine biases this competition by strengthening d1 (go) and weakening d2 (nogo) activity.

relevance to neural computer: selective disinhibition is the biological gating mechanism -- tonic inhibition from gpi/snr is released for the winning action. this is the biological implementation of gating that todorov's spike quantization approximates. see [[basal_ganglia]] for detailed pathway analysis.

## intuition and insight

### neural basis of insight

jung-beeman, m., bowden, e. m., haberman, j., frymiare, j. l., arambel-liu, s., greenblatt, r., reber, p. j., & kounios, j. (2004). neural activity when people solve verbal problems with insight. *plos biology*, 2(4), e97. doi: 10.1371/journal.pbio.0020097

key insight: the "aha moment" of insight is preceded by a burst of gamma-band (40 hz) activity in the right anterior superior temporal gyrus approximately 300 ms before the conscious solution. this burst is preceded by alpha-band (10 hz) suppression over right posterior cortex, suggesting gating of visual input to facilitate internal processing. insight is not random -- it has a measurable neural preparation phase.

relevance to neural computer: insight involves a detectably different computational mode -- internal recurrent processing dominates over external input. the alpha suppression followed by gamma burst suggests a gating mechanism that shifts between input-driven and state-driven computation.

### coarse semantic coding in the right hemisphere

bowden, e. m., & jung-beeman, m. (1998). getting the right idea: semantic activation in the right hemisphere may help solve insight problems. *psychological science*, 9(6), 435-440. doi: 10.1111/1467-9280.00082

key insight: the right hemisphere maintains broader, more diffuse semantic activation than the left hemisphere's focused activation. this "coarse coding" keeps distant semantic associations weakly active, enabling the detection of remote connections that underlie insight. solution primes activate right-hemisphere representations before conscious awareness of the solution.

relevance to neural computer: coarse coding is a form of maintaining multiple weak hypotheses simultaneously -- analogous to a state representation that preserves distributed, low-confidence associations rather than committing to a single interpretation.

### unconscious thought theory

dijksterhuis, a., & nordgren, l. f. (2006). a theory of unconscious thought. *perspectives on psychological science*, 1(2), 95-109. doi: 10.1111/j.1745-6916.2006.00007.x

key insight: unconscious thought (incubation) can outperform conscious deliberation on complex decisions with many attributes. conscious thought is rule-based and limited by working memory capacity (~4 items). unconscious thought integrates information in parallel without capacity limits but produces only a rough "feeling" rather than a precise calculation. the theory predicts that simple decisions benefit from conscious analysis while complex decisions benefit from a period of distraction.

relevance to neural computer: parallel integration without capacity limits describes what recurrent state accumulation does -- it integrates over all tokens without the bottleneck of attention's explicit comparison. the rough "feeling" output maps to a compressed representation that captures gist rather than detail.

### two-stage model of intuition

bowers, k. s., regehr, g., balthazard, c., & parker, k. (1990). intuition in the context of discovery. *cognitive psychology*, 22(1), 72-110. doi: 10.1016/0010-0285(90)90004-N

key insight: intuition operates in two stages. the guiding stage generates a "feeling of knowing" that directs attention and search toward promising regions of the problem space, without specifying a solution. the integrative stage produces the actual insight when accumulated clues coalesce into a coherent pattern. the guiding stage is metacognitive -- it monitors the accumulation process itself.

relevance to neural computer: two-stage intuition parallels the distinction between state accumulation (guiding, building a representation) and threshold crossing (integrative, committing to an output). the metacognitive guiding stage implies a need for monitoring the quality of accumulated state.

## social cognition

### mirror neurons

rizzolatti, g., fadiga, l., gallese, v., & fogassi, l. (1996). premotor cortex and the recognition of motor actions. *cognitive brain research*, 3(2), 131-141. doi: 10.1016/0926-6410(95)00038-0

key insight: neurons in macaque premotor cortex (area f5) fire both when the monkey performs an action and when it observes the same action performed by another. these "mirror neurons" provide a direct neural mechanism for mapping observed actions onto the observer's own motor repertoire, potentially enabling action understanding through motor simulation.

relevance to neural computer: mirror neurons suggest that perception and production share representations -- the same weights are used for generating and interpreting. this is the biological case for weight sharing between encoding and decoding pathways.

### the mirror neuron critique

hickok, g. (2014). *the myth of mirror neurons: the real neuroscience of communication and cognition*. norton.

key insight: the mirror neuron theory of understanding is overclaimed. motor simulation is neither necessary nor sufficient for action understanding -- patients with motor cortex damage can still understand actions, and mirror neurons fire for actions the monkey does not understand. the system likely supports sensorimotor integration (mapping perception to action for imitation or prediction) rather than semantic comprehension.

relevance to neural computer: caution against equating shared representations with understanding. weight sharing between encoder and decoder does not automatically confer comprehension -- it confers efficient mapping between input and output domains, which is useful but distinct.

### theory of mind and the right temporoparietal junction

saxe, r., & kanwisher, n. (2003). people thinking about thinking people: the role of the temporo-parietal junction in "theory of mind." *neuroimage*, 19(4), 1835-1842. doi: 10.1016/S1053-8119(03)00230-1

key insight: the right temporoparietal junction (rtpj) is selectively activated when people reason about others' mental states (beliefs, desires, intentions) -- a capacity called theory of mind (tom). rtpj responds to mental state attribution even when matched for narrative complexity, linguistic difficulty, and social content. this selectivity suggests a dedicated computational resource for modeling other minds.

relevance to neural computer: theory of mind requires maintaining and manipulating a model of another agent's state separate from one's own -- a form of multi-state tracking. current recurrent architectures maintain a single state per layer; modeling other agents would require parallel or partitioned state representations.

### the origin of theory of mind

premack, d., & woodruff, g. (1978). does the chimpanzee have a theory of mind? *behavioral and brain sciences*, 1(4), 515-526. doi: 10.1017/S0140525X00076512

key insight: coined the term "theory of mind" to describe the ability to attribute mental states (intentions, beliefs, desires) to others. the question of whether non-human primates possess this ability initiated decades of comparative cognition research. the original evidence from chimpanzees was ambiguous -- they may use behavioral rules rather than genuine mental state attribution.

relevance to neural computer: theory of mind is the capacity to simulate another agent's internal state -- a recursive modeling problem. implementing this requires architectures that can maintain nested representations (my model of your model of the world).

### the social brain hypothesis

dunbar, r. i. m. (1992). neocortex size as a constraint on group size in primates. *journal of human evolution*, 22(6), 469-493. doi: 10.1016/0047-2484(92)90081-J

dunbar, r. i. m. (1998). the social brain hypothesis. *evolutionary anthropology*, 6(5), 178-190. doi: 10.1002/(SICI)1520-6505(1998)6:5<178::AID-EVAN5>3.0.CO;2-8

key insight: neocortex size in primates correlates with social group size, not ecological complexity (habitat, diet, range). the "social brain hypothesis" proposes that the computational demands of tracking relationships, alliances, and social hierarchies drove the expansion of primate neocortex. for humans, the predicted natural group size (~150, "dunbar's number") matches ethnographic data on clan sizes, military units, and social network layers.

relevance to neural computer: if social computation drove cortical expansion, then modeling other agents is one of the hardest computational problems biology solves. this is a capacity challenge (how much state can you maintain) rather than an algorithmic one -- relevant to scaling arguments for recurrent state dimensionality.

## the predictive brain

### predictive coding

rao, r. p. n., & ballard, d. h. (1999). predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. *nature neuroscience*, 2(1), 79-87. doi: 10.1038/4580

key insight: cortical hierarchies implement prediction: top-down connections carry predictions of expected input, bottom-up connections carry prediction errors (the difference between predicted and actual activity). only the unpredicted residuals propagate upward, implementing efficient coding by transmitting only the surprising component. this explains extra-classical receptive field effects like end-stopping.

relevance to neural computer: predictive coding is the biological framework for error-driven learning. the residual stream in transformers carries full representations rather than prediction errors, which is less efficient but more robust. see [[predictive_coding]] for detailed analysis.

### the free energy principle

friston, k. (2010). the free-energy principle: a unified brain theory? *nature reviews neuroscience*, 11(2), 127-138. doi: 10.1038/nrn2787

key insight: the free energy principle (fep) proposes that all brain function can be understood as minimizing variational free energy -- an information-theoretic bound on surprise. perception minimizes free energy by updating internal models (inference), action minimizes it by changing sensory input to match predictions (active inference), and learning minimizes it by adjusting model parameters. this unifies perception, action, and learning under a single objective.

relevance to neural computer: fep provides a theoretical framework for unifying the training objective (minimize prediction error), gating (precision weighting), and architecture (hierarchical generative model). see [[free_energy_principle]] for full treatment. the practical question is whether minimizing fep-derived objectives produces better language models than cross-entropy.

### anterior cingulate cortex and conflict monitoring

botvinick, m. m., braver, t. s., barch, d. m., carter, c. s., & cohen, j. d. (2001). conflict monitoring and cognitive control. *psychological review*, 108(3), 624-652. doi: 10.1037/0033-295X.108.3.624

key insight: the anterior cingulate cortex (acc) monitors for response conflict -- situations where multiple competing responses are simultaneously activated. conflict detection triggers increased allocation of top-down control from the dorsolateral prefrontal cortex (dlpfc), explaining post-error slowing and the gratton effect. the acc does not resolve conflict; it signals the need for control.

relevance to neural computer: conflict monitoring is a form of meta-computation -- detecting when the system's own outputs are ambiguous or contradictory. this maps to detecting high-entropy output distributions and dynamically adjusting processing (e.g., increasing recurrent depth or lowering threshold).

### active inference

active inference extends the free energy principle to action: organisms act to fulfill their predictions rather than passively updating them. expected free energy (the variational bound on future surprise) drives policy selection, combining epistemic value (information gain) with pragmatic value (goal achievement). this reframes motor control as inference: movements are predictions of proprioceptive trajectories, and spinal reflexes minimize the prediction error between predicted and actual body state.

relevance to neural computer: active inference dissolves the perception-action boundary -- both are inference under the same objective. for a language model, this suggests that generation (action) and comprehension (perception) should share the same objective function and state representation, which autoregressive models already approximate.

## rationality and heuristics

### judgment under uncertainty: heuristics and biases

tversky, a., & kahneman, d. (1974). judgment under uncertainty: heuristics and biases. *science*, 185(4157), 1124-1131. doi: 10.1126/science.185.4157.1124

key insight: human judgment under uncertainty relies on a small number of heuristics: representativeness (judging probability by similarity to prototypes), availability (judging frequency by ease of recall), and anchoring (insufficient adjustment from an initial value). these heuristics are computationally efficient but produce systematic, predictable errors (biases). the errors are not random noise -- they reveal the algorithms the brain actually uses.

relevance to neural computer: heuristics are the brain's compression of optimal inference into fast, approximate computations. the biases they produce are the cost of compression -- structurally analogous to the information loss from ternary spike quantization. understanding which biases emerge from ternary spikes could predict failure modes.

### system 1 and system 2

kahneman, d. (2011). *thinking, fast and slow*. farrar, straus and giroux.

key insight: cognition operates through two systems. system 1 is fast, automatic, parallel, effortless, and always on -- it produces intuitive judgments, emotional reactions, and skilled actions. system 2 is slow, deliberate, serial, effortful, and lazy -- it performs logical reasoning, planning, and self-monitoring. system 2 often endorses system 1's outputs without checking, creating the illusion of deliberation. most cognition is system 1; system 2 is the exception, not the default.

relevance to neural computer: the system 1/2 distinction maps loosely onto recurrent state (fast, always-on, accumulated) vs attention (selective, effortful, capacity-limited). kda layers provide system-1-like continuous processing; mla layers provide system-2-like selective retrieval. the 3:1 ratio means the architecture is dominated by system 1, which matches biology.

### ecological rationality

gigerenzer, g., todd, p. m., & the abc research group. (1999). *simple heuristics that make us smart*. oxford university press.

key insight: heuristics are not cognitive defects but evolved adaptations matched to environmental structure. "fast-and-frugal" heuristics (take-the-best, recognition heuristic, 1/n) exploit information structure in ecological niches and often outperform optimal statistical methods on out-of-sample prediction. less information and computation can yield more accurate decisions when the environment is uncertain and sample sizes are small.

relevance to neural computer: ecological rationality is the argument for why compression (ternary spikes, low-rank projection) can improve generalization rather than degrade it. dropping information is beneficial when the dropped information is noise. the question is whether the ternary spike threshold selectively drops noise or signal.

### less-is-more effects

gigerenzer, g., & brighton, h. (2009). homo heuristicus: why biased minds make better inferences. *topics in cognitive science*, 1(1), 107-143. doi: 10.1111/j.1756-8765.2008.01006.x

key insight: the bias-variance tradeoff explains why heuristics outperform complex models: in noisy, small-sample environments, the variance reduction from using a simple model more than compensates for its increased bias. this "less-is-more" effect means that ignoring information (cue directions, correlations, base rates) can systematically improve prediction accuracy. the effect disappears with large samples and low noise.

relevance to neural computer: directly relevant to ternary spike quantization. by forcing information through a {-1, 0, +1} bottleneck, the architecture trades bias for variance reduction. the less-is-more framework predicts this should help most when data is limited or noisy -- exactly the small-to-medium scale regime where todorov operates.

### model-based vs model-free control

dayan, p., & daw, n. d. (2008). decision theory, reinforcement learning, and the brain. *cognitive, affective, & behavioral neuroscience*, 8(4), 429-453. doi: 10.3758/CABN.8.4.429

key insight: the brain implements two parallel reinforcement learning systems. model-free (habitual) learning caches values of states and actions through repeated experience -- fast to execute but slow to update when contingencies change. model-based (deliberative) learning simulates outcomes using an internal model of the environment -- flexible but computationally expensive. the arbitration between these systems depends on uncertainty, computational cost, and the reliability of each system's predictions.

relevance to neural computer: model-free maps to recurrent state (cached, always available, slow to update) and model-based maps to attention (flexible, computationally expensive, simulates relationships). the 3:1 kda:mla ratio biases toward cached computation, consistent with the biological dominance of habitual over deliberative processing.

## relevance to todorov

### what these literatures reinforce

- **accumulate-to-bound is fundamental**: decision making, intuition, and even emotion depend on evidence accumulation followed by thresholded commitment. todorov's recurrent state accumulation + ternary spike threshold implements this pattern at every layer.

- **compression improves generalization**: the heuristics literature (gigerenzer, kahneman) provides theoretical grounding for why ternary quantization can outperform full-precision computation. the bias-variance tradeoff predicts benefits at small-to-medium scale, which matches todorov's operating regime.

- **fast/slow duality is architectural**: system 1/system 2 (kahneman), model-free/model-based (dayan & daw), low road/high road (ledoux), habitual/deliberative all point to the same design pattern: a fast always-on system paired with a slow selective system. todorov's 3:1 kda:mla ratio implements this with the correct biological weighting toward the fast system.

- **reward is multidimensional**: wanting vs liking (berridge), subjective value (padoa-schioppa), and td error (schultz) show that reward processing involves multiple signals, not a single scalar loss. current training uses cross-entropy; richer reward signals are a phase 6+ consideration.

- **state monitoring enables metacognition**: conflict monitoring (botvinick), interoception (craig), and the guiding stage of intuition (bowers) all require the system to monitor its own internal state. recurrent hidden state provides the substrate; the question is whether todorov's architecture can read its own state to modulate processing.

### what these literatures challenge

- **fixed thresholds miss urgency dynamics**: the urgency-gating model (thura & cisek) shows biological thresholds are time-varying, not fixed. todorov's alpha * mean(|x|) threshold is static within a forward pass. context-dependent or time-varying thresholds could improve decision-like computations.

- **single-state limits social computation**: theory of mind (premack, saxe) and social brain scaling (dunbar) require maintaining models of other agents' states. a single recurrent state per layer cannot represent nested beliefs (my model of your model) without partitioning.

- **emotion as prediction challenges modular design**: barrett's constructed emotion theory argues that functional states are not modular -- the same circuit produces different "emotions" depending on context. this supports flexible, context-dependent computation but challenges any attempt to design dedicated emotion modules.

### what is not yet relevant (phase 6+)

- somatic markers as learned decision biases (requires rl training)
- theory of mind / multi-agent modeling (requires multi-turn interaction)
- active inference as a training objective (requires replacing cross-entropy)
- metacognitive conflict monitoring (requires output uncertainty estimation)
- unconscious parallel integration as a computational advantage of recurrence over attention

## see also

- [[dopamine_system]]
- [[basal_ganglia]]
- [[predictive_coding]]
- [[free_energy_principle]]
- [[selective_attention]]
- [[neuromodulatory_framework]]
