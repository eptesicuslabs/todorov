# neuromodulatory framework

status: definitional. last fact-checked 2026-04-16.

**why this matters**: Doya's framework reveals that the brain has a complete online hyperparameter tuning system -- four neuromodulators controlling learning rate, reward signal, discount factor, and exploration. current ML architectures fix all these before training, missing an entire level of adaptive control.

## summary

Doya (2002) proposed a computational framework mapping each major **neuromodulatory system** (a set of diffusely projecting neurons that release modulatory neurotransmitters, adjusting the operating parameters of target circuits rather than directly driving them) to a **meta-parameter** (a parameter that controls other parameters) of reinforcement learning:

| neuromodulator | meta-parameter | RL analog |
|---|---|---|
| dopamine (DA) | reward signal | TD error delta |
| serotonin (5-HT) | temporal discount | discount factor gamma |
| norepinephrine (NE) | exploration rate | temperature / epsilon |
| acetylcholine (ACh) | learning speed | learning rate eta |

the framework reframes neuromodulation as **metalearning** (learning to learn -- adjusting the parameters of the learning process itself): the neuromodulatory systems do not themselves learn or compute; they adjust the parameters that govern how other circuits learn and compute. this is a second-order control system operating on a slower timescale than the primary learning loop.

ML analog: metalearning corresponds to online hyperparameter optimization -- adjusting learning rate, exploration, and discount factor during training based on training dynamics, rather than fixing them beforehand.

    learning: Delta_w = eta * delta * eligibility
    metalearning: eta = f(ACh), delta = f(DA), gamma = f(5-HT), exploration = f(NE)

this is an elegant and influential framework, but it must be treated as a computational hypothesis, not established biology. each mapping has supporting evidence and significant counter-evidence.

## the four mappings

### dopamine -> reward signal (delta)

**the claim**: dopamine encodes the reward prediction error delta_t = r_t + gamma * V(s_{t+1}) - V(s_t), serving as the teaching signal for reinforcement learning.

**evidence for**: Schultz et al. (1997) demonstrated that phasic dopamine activity in VTA/SNc neurons matches the TD error signal. this is the most experimentally validated mapping in the framework. dopamine bursts for unexpected reward (positive RPE), silence for expected reward (zero RPE), depression for omitted reward (negative RPE). D1/D2 receptor pathways implement sign-dependent plasticity in the striatum. see [[dopamine_system]] for full details.

**evidence against**: dopamine neurons also respond to salient non-rewarding stimuli (loud sounds, aversive stimuli). some researchers argue dopamine encodes SALIENCE (unsigned prediction error) rather than reward specifically. ramping dopamine signals during approach behavior do not fit the phasic RPE model. distributional coding (Dabney et al. 2020) complicates the simple delta interpretation.

**verdict**: the strongest mapping in the framework. correct in essence for phasic VTA dopamine, but oversimplified -- dopamine is not purely a reward signal.

### serotonin -> temporal discount (gamma)

**the claim**: serotonin modulates the temporal horizon of reward prediction. high serotonin = high gamma = patient, long-horizon evaluation. low serotonin = low gamma = impulsive, short-horizon evaluation.

**evidence for**: serotonin depletion (via tryptophan depletion or 5-HT lesions) increases impulsive choice on delay discounting tasks. SSRIs (which increase serotonin) reduce impulsivity in some clinical populations. dorsal raphe serotonin neurons show sustained activity during waiting periods, potentially encoding the "cost of waiting" (Miyazaki et al. 2011). computational models (Tanaka et al. 2007) show that serotonin modulation of discount factor gamma captures observed behavioral effects.

**evidence against**: serotonin has at least 14 receptor subtypes with diverse and sometimes opposing effects. serotonin regulates mood, aggression, sleep, appetite, pain, and sexual function -- mapping this entire system to a single scalar (gamma) is a drastic oversimplification. the behavioral evidence for serotonin and temporal discounting is correlational, not causal at the circuit level. deVito et al. (2015) found that serotonin depletion affects motor impulsivity but not choice impulsivity (delay discounting), dissociating the behavioral and computational predictions.

**verdict**: the weakest mapping in the framework. plausible as one function of serotonin among many, but the "serotonin = gamma" reduction is too strong for the biological evidence.

### norepinephrine -> exploration rate (temperature / epsilon)

**the claim**: NE controls the randomness of action selection. tonic NE mode = high exploration (broad attention, probabilistic action selection). phasic NE mode = low exploration (focused attention, deterministic action selection).

**evidence for**: Aston-Jones and Cohen (2005) demonstrated that LC-NE tonic/phasic modes correspond to exploration/exploitation behavior in macaques. the adaptive gain mechanism provides a principled computational account: NE modulates the gain of neural transfer functions, which is equivalent to modulating the temperature of a softmax decision rule. see [[norepinephrine_system]] for full details.

**evidence against**: the gain modulation account conflates exploration (sampling from a broader distribution of actions) with distractibility (responding to task-irrelevant stimuli). these are computationally different: exploration is about the ACTION space; distractibility is about the INPUT space. also, the LC-NE system has strong arousal functions that are not well captured by "exploration rate."

**verdict**: a strong mapping with solid experimental support from Aston-Jones's lab. the gain modulation mechanism is computationally precise. main limitation: conflation of exploration and distractibility.

### acetylcholine -> learning rate (eta)

**the claim**: ACh controls the speed of memory update. high ACh = fast learning, new input overwrites old patterns. low ACh = slow learning, existing patterns are preserved.

**evidence for**: Hasselmo (2006) showed that ACh enhances encoding (feedforward input, synaptic plasticity) while suppressing retrieval (recurrent connections). scopolamine (muscarinic antagonist) impairs new learning while partially sparing recall. acetylcholinesterase inhibitors (donepezil) improve learning in early Alzheimer's disease. computational models of hippocampal encoding/retrieval switching depend on cholinergic modulation. see [[acetylcholine_system]] for full details.

**evidence against**: ACh also functions as an attentional signal (Yu and Dayan 2005: ACh = expected uncertainty), which is distinct from learning rate. cholinergic basal forebrain lesions impair attention even in tasks with minimal memory demands. the encoding/retrieval framework applies cleanly to hippocampus but less clearly to neocortex, where ACh effects are more heterogeneous.

**verdict**: a good mapping with strong support from Hasselmo's work. the learning rate interpretation captures the dominant effect of ACh on hippocampal dynamics. the attention/uncertainty role is complementary rather than contradictory.

## the framework as metalearning

the deepest insight in Doya (2002) is not the specific neuromodulator-to-parameter mappings, but the META-LEARNING STRUCTURE: a two-level control system where:

- **level 1 (learning)**: cortical and striatal circuits update representations and policies based on experience. this is fast, local, and operates on individual trials.
- **level 2 (metalearning)**: neuromodulatory systems adjust the parameters of level 1 learning based on higher-order statistics (reward rate, prediction accuracy, volatility). this is slower, global, and operates on blocks of trials.

this architecture has clear advantages:
1. **adaptability**: the same learning circuit can operate in different regimes (fast/slow learning, patient/impulsive evaluation, focused/exploratory attention) depending on context.
2. **stability**: metalearning parameters change slowly, preventing catastrophic oscillation of the primary learning system.
3. **modularity**: each meta-parameter can be adjusted independently (at least approximately), enabling fine-grained control.

in modern machine learning terms, this is analogous to hyperparameter optimization -- but done ONLINE during learning, not offline before training.

## evidence for and against the framework

### strengths

1. **parsimony**: four neuromodulators, four meta-parameters. a clean computational story that organizes a vast amount of pharmacological and behavioral data.
2. **testable predictions**: each mapping makes specific predictions about the behavioral effects of pharmacological manipulation (e.g., serotonin depletion should increase temporal discounting).
3. **theoretical foundation**: the meta-parameters are derived from reinforcement learning theory, not ad hoc. each parameter has a clear computational role.
4. **influence**: the framework has been highly influential (2000+ citations), spawning research programs in computational psychiatry and biologically-inspired AI.

### weaknesses

1. **many-to-many mapping**: each neuromodulator affects multiple brain functions, and each function is affected by multiple neuromodulators. the clean one-to-one mapping is a useful fiction. recent work (Marder 2012) emphasizes the combinatorial complexity of neuromodulatory interactions.

2. **no serotonin-gamma mechanism**: unlike the other three mappings, there is no clear circuit-level mechanism by which serotonin would modulate the discount factor in a TD learning circuit. the behavioral evidence (serotonin depletion -> impulsivity) has multiple possible explanations.

3. **missing neuromodulators**: the framework excludes histamine, orexin/hypocretin, endocannabinoids, neuropeptides, and other neuromodulatory systems that affect learning and cognition. are they redundant? do they control meta-parameters not captured by RL theory?

4. **static mapping**: the framework treats each neuromodulator as controlling a fixed meta-parameter. but the SAME neuromodulator (e.g., dopamine) has different functional roles in different brain regions (reward in striatum, working memory in PFC, motor control in motor cortex). the mapping is region-dependent, not system-wide.

5. **independence assumption**: the four neuromodulatory systems are heavily interconnected. dopamine and serotonin interact in the basal ganglia (opponent processing). NE modulates dopamine release. ACh and NE interact in the cortex. the "four independent knobs" picture is a simplification.

## relationship to todorov

todorov's meta-parameters are:

- **learning rate**: cosine decay with warmup. fixed schedule, not adaptive.
- **no reward signal**: the loss function is cross-entropy, applied uniformly. no RPE-like modulation.
- **no discount factor**: no temporal discounting of any kind. all tokens in a sequence contribute equally to the loss.
- **no exploration**: inference is deterministic (argmax or temperature-scaled sampling with fixed temperature). no adaptive exploration during training.

the Doya framework suggests that todorov is missing an entire level of control. all meta-parameters are fixed before training begins (or follow a predetermined schedule). there is no online metalearning: no mechanism to increase the learning rate when the model encounters novel distribution, no mechanism to shift from exploitation to exploration when loss plateaus, no mechanism to adjust the temporal horizon based on task demands.

whether this matters depends on scale. at the scale of biological learning (a lifetime of non-stationary experience), online metalearning is essential. at the scale of training a language model (a fixed dataset, processed once), fixed meta-parameters may be sufficient. the question is whether online metalearning provides benefits WITHIN a single training run -- e.g., by adapting the learning rate to local loss landscape curvature, or by modulating exploration during training on diverse data.

see [[neuromodulation_to_learning_and_gating]] for a concrete proposal.

## key references

- Doya, K. (2002). Metalearning and neuromodulation. Neural Networks, 15(4-6), 495-506.
- Doya, K. (2000). Complementary roles of basal ganglia and cerebellum in learning and motor control. Current Opinion in Neurobiology, 10(6), 732-739.
- Marder, E. (2012). Neuromodulation of neuronal circuits: back to the future. Neuron, 76(1), 1-11.
- Tanaka, S. C. et al. (2007). Serotonin differentially regulates short- and long-term prediction of rewards in the ventral and dorsal striatum. PLoS ONE, 2(12), e1333.
- Miyazaki, K. W. et al. (2011). Optogenetic activation of dorsal raphe serotonin neurons enhances patience for future rewards. Current Biology, 21(5), 396-401.
- Schweighofer, N. et al. (2008). Low-serotonin levels increase delayed reward discounting in humans. Journal of Neuroscience, 28(17), 4528-4532.
- Doya, K. (2008). Modulators of decision making. Nature Neuroscience, 11(4), 410-416.

## see also

- [[dopamine_system]]
- [[acetylcholine_system]]
- [[norepinephrine_system]]
- [[neuromodulation_to_learning_and_gating]]
- [[precision_weighting]]
- [[plasticity_to_matrix_memory_delta_rule]]
