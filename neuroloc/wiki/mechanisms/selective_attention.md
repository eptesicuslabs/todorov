# selective attention

**why this matters**: biological attention is the original gating mechanism -- it determines which information gets amplified and which gets suppressed, directly inspiring the query-key-value attention in transformers and the competitive selection in mixture-of-experts routing.

## the biased competition model

in 1995, Robert Desimone and John Duncan published "neural mechanisms of selective visual attention" in Annual Review of Neuroscience, proposing that attention is not a spotlight or a filter but the outcome of competitive interactions among **neural populations** (groups of neurons with overlapping response properties). the central claim: multiple objects in the visual field compete for neural representation, and attention is what happens when this competition is biased.

ML analog: biased competition is the biological version of softmax attention. in both systems, multiple inputs compete for representation, and a bias signal (the query vector in transformers, top-down feedback in cortex) determines the winner.

see [[desimone_duncan]] for biographical context.

the theory rests on five principles:

1. simultaneous objects compete. when multiple stimuli fall within a single neuron's **receptive field** (the region of sensory space that activates a given neuron), their neural representations interact competitively. the response to two stimuli presented together is less than the sum of responses to each presented alone. this is mutual suppression, not independent coding.

2. competition is strongest at close spatial proximity. stimuli sharing the same cortical territory (overlapping receptive fields) compete more intensely than distant stimuli. competition scales with cortical magnification: stimuli close in retinotopic space share more neurons and therefore suppress each other more.

3. **top-down feedback** (signals flowing from higher to lower cortical areas, carrying task goals and expectations) biases the competition. signals from prefrontal cortex, parietal cortex, and working memory structures modulate the competition in favor of task-relevant stimuli. the prefrontal cortex does not "select" the attended object directly -- it changes the competitive landscape so that the attended object wins the competition on its own.

4. features can bias competition. bottom-up stimulus properties (salience, luminance, motion onset, color contrast) and top-down feature templates (stored in working memory) both bias the competition. a target defined by its color will receive a competitive advantage over distractors that do not share that color.

5. the winner suppresses the losers. the winning representation gains enhanced firing rate, reduced noise correlations, increased gamma-band synchrony, and expanded effective receptive field. the losers are suppressed below their baseline response. this is a neural implementation of [[winner_take_all]]. biological competition is rarely perfectly winner-take-all -- residual representations of suppressed stimuli persist.

## neural evidence

### receptive field effects

the strongest evidence comes from single-unit recordings in macaque visual cortex (areas V2, V4, IT) by Moran and Desimone (1985), Reynolds, Chelazzi, and Desimone (1999), and Luck, Chelazzi, Hillyard, and Desimone (1997).

when a single stimulus falls in a neuron's receptive field, the neuron responds at a rate determined by the stimulus features and the neuron's tuning preferences. when a second stimulus is added inside the same receptive field, two effects occur:

1. the neuron's response shifts toward the average of the two individual responses (sensory interaction). this is the baseline competition: neither stimulus fully controls the neuron.

2. if the monkey attends to one of the two stimuli, the neuron's response shifts toward what it would have been if only the attended stimulus were present. the unattended stimulus loses its influence on the neuron's firing rate. this is the resolution of competition by attention.

Reynolds, Chelazzi, and Desimone (1999) showed this quantitatively in V4: when a preferred and a non-preferred stimulus both fell in a neuron's receptive field, the response was intermediate. attending to the preferred stimulus increased the response toward the preferred-alone level. attending to the non-preferred stimulus decreased the response toward the non-preferred-alone level. attention pushed the competition toward one or the other stimulus.

### receptive field shrinkage

a remarkable consequence of biased competition is that attention effectively shrinks the neuron's receptive field around the attended stimulus. Womelsdorf, Anton-Erxleben, Piber, and Treue (2006) demonstrated this in macaque MT: spatial attention shifted the receptive field center toward the attended location and contracted the receptive field size. the attended stimulus effectively "captured" the receptive field. unattended stimuli were excluded from influencing the neuron.

this is not a change in the neuron's intrinsic properties -- it reflects the competitive suppression of inputs from non-attended locations within the same receptive field.

### firing rate modulation

attention increases the firing rate of neurons representing the attended stimulus by 20-50% in V4 and IT (McAdams & Maunsell 1999, Treue & Martinez-Trujillo 1999). this increase is typically multiplicative: the entire tuning curve scales up by a constant factor (**response gain**), without sharpening or broadening. the Reynolds and Heeger normalization model (see [[normalization_model_of_attention]]) explains when this multiplicative scaling transitions to **contrast gain**. the transition depends on the relative sizes of the attention field and the normalization pool.

### noise correlation reduction

attention does not only increase signal strength. Cohen and Maunsell (2009) showed that attention reduces **noise correlations** (correlated trial-to-trial fluctuations in firing rate between pairs of neurons) in V4. when the neurons' shared receptive field contains an attended stimulus, shared variability decreases. this matters for population coding: correlated noise limits the information that a population can carry (see [[population_coding]]). reducing noise correlations increases the population's discriminability of the attended stimulus.

ML analog: noise correlation reduction is analogous to decorrelation techniques in ML (batch normalization, whitening). both improve the effective information capacity of the representation by reducing redundancy between units.

Mitchell, Sundberg, and Reynolds (2009) confirmed this: attention-related reductions in noise correlations accounted for more of the improvement in population discriminability than the firing rate increases did.

### gamma-band synchronization

Fries, Reynolds, Rorie, and Desimone (2001) showed that attention increases **gamma-band** (30-80 Hz) synchronization among neurons representing the attended stimulus in macaque V4. simultaneously, **alpha-band** (8-14 Hz) power decreases for the attended location and increases for the unattended location. this links biased competition to the [[neural_synchrony]] framework. the attended stimulus gains access to downstream processing through enhanced coherence (**communication through coherence** -- the principle that neural groups communicate effectively when their oscillations are phase-aligned, Fries 2015). the unattended stimulus is gated out by alpha suppression.

## types of attention

### spatial attention

spatial attention enhances processing at a specific location. the classic demonstration is Posner's (1980) cueing paradigm: a cue indicating where a target will appear speeds reaction time (valid cue, ~50 ms faster) and slows reaction time when the cue is invalid (~30 ms slower). this establishes a cost-benefit asymmetry: attention both helps the attended location and hurts the unattended location.

spatial attention is often described with the spotlight metaphor (Posner, Snyder & Davidson 1980): a region of enhanced processing that can be moved across the visual field. the spotlight has a variable size (the **zoom lens model**, Eriksen & St. James 1986) and a gradient of enhancement that falls off with distance from the focus. but the spotlight metaphor is misleading. attention is not a single mechanism that sweeps across space. it is a pattern of competitive biases that can be applied to any spatial location, or to multiple locations simultaneously (split attention).

spatial attention is controlled by two systems:

1. endogenous (top-down, voluntary): driven by task goals, engages prefrontal and parietal cortex, takes ~200 ms to deploy, sustained as long as the task demands.

2. exogenous (bottom-up, stimulus-driven): triggered by salient events (sudden onset, luminance change, motion), involves the superior colliculus and pulvinar, takes ~50 ms to deploy, transient (~200 ms), followed by inhibition of return (slowed responses at the cued location after ~300 ms).

### feature-based attention

**feature-based attention** enhances processing of a specific feature (color, orientation, motion direction) across all spatial locations simultaneously. Treue and Martinez-Trujillo (1999) showed that attending to upward motion in one location enhances responses of MT neurons tuned to upward motion everywhere in the visual field, even in the opposite hemifield. this is global feature-based enhancement: the competitive bias is applied to a feature dimension, not a spatial location.

ML analog: feature-based attention is the biological version of content-based retrieval. in transformers, a query vector selects keys by feature similarity regardless of position -- the same global, content-addressed selection that feature-based attention implements across the visual field.

the feature-similarity gain principle (Treue & Martinez-Trujillo 1999): the attentional modulation of a neuron's response is proportional to the similarity between the attended feature and the neuron's preferred feature. neurons tuned to the attended feature are enhanced; neurons tuned to the opposite feature are suppressed.

see [[feature_vs_spatial_attention]] for extended treatment.

### object-based attention

object-based attention enhances processing of all features belonging to a single object, even features that are not task-relevant. Roelfsema, Lamme, and Spekreijse (1998) showed that when a monkey traces a curve from a fixation point to a target, neurons along the entire attended curve show enhanced responses, not just those near the fixation point or the target. the object (the curve) receives enhanced processing as a whole.

O'Craven, Downing, and Kanwisher (2000) used overlapping face/house stimuli and showed that attending to the face enhanced processing of the house if they were part of the same object (transparent overlapping images), demonstrating object-based spread of attention in human fMRI.

## subcortical attention

Krauzlis, Lovejoy, and Zenon (2014) argued that selective attention is not exclusively a neocortical function. non-mammalian vertebrates (birds, reptiles, amphibians, fish) that completely lack a neocortex still exhibit selective attention: pigeons show spatial cueing effects, chickens perform target localization while ignoring distractors, zebrafish demonstrate attentional learning and reversal.

the critical structure is the optic tectum (superior colliculus in mammals), which implements a saliency map with winner-take-all competition mediated by the isthmic nuclei. the basal ganglia modulate tectal competition through inhibitory pathways, linking attention to value-based selection. the neocortex expanded the feature space over which attention operates and added flexible top-down control, but the fundamental competitive mechanism is subcortical and evolutionarily ancient.

this has important implications for computational models: attention-like selection does not require the full machinery of cortical biased competition. a competition mechanism with top-down modulation is sufficient.

## relationship to normalization

the biased competition model and the [[normalization_model_of_attention]] (Reynolds & Heeger 2009) are complementary, not competing. biased competition describes the computational principle (stimuli compete, attention biases the competition). normalization provides the mathematical framework (attention acts as a multiplicative gain on the input to a divisive normalization circuit). the normalization equation:

    R = (A * E)^n / (sigma^n + sum(w * (A * E)^n))

instantiates biased competition: the attention field A biases the stimulus drive E in the numerator and the suppressive pool in the denominator, and the normalization implements the competition.

## relationship to precision weighting

[[precision_weighting]] (Feldman & Friston 2010) reframes attention in the [[predictive_coding]] framework: attention is the optimization of precision (inverse variance) of prediction errors. attending to a stimulus means expecting high precision (low noise) from that sensory channel, which increases the gain of prediction error signals from that channel.

the two frameworks are not incompatible. precision weighting describes WHY the brain deploys attention (to optimize inference by weighting reliable information sources). biased competition describes HOW attention is implemented in neural circuits (through competitive interactions modulated by top-down bias). precision is the computational objective; biased competition is the neural mechanism.

## challenges

the biased competition model faces several open problems. first, the source of the top-down bias signal is underspecified. the model states that prefrontal cortex biases the competition, but the mechanism by which abstract task goals are converted into specific gain modulations on individual neurons in visual cortex remains unknown. the computational gap between "attend to the red object" and the specific synaptic weight changes in V4 is not bridged.

second, the relationship between biased competition and reward-based selection is unclear. attention is not purely driven by task relevance -- rewarding stimuli capture attention even when task-irrelevant (Anderson, Laurent & Yantis 2011). the biased competition model does not naturally accommodate value-driven attention, which requires integration with reinforcement learning circuits (basal ganglia, dopaminergic projections).

third, the temporal dynamics of competition resolution are poorly characterized. the model describes a steady-state outcome (the winner suppresses the losers) but not the dynamics of how long the competition takes to resolve, how the brain handles rapidly changing stimuli, or what happens when two competing representations are equally strong. oscillatory models (communication through coherence) attempt to address the temporal dimension, but the integration of oscillatory dynamics with biased competition remains incomplete.

## key references

- desimone, r. & duncan, j. (1995). neural mechanisms of selective visual attention. annual review of neuroscience, 18(1), 193-222.
- reynolds, j. h., chelazzi, l. & desimone, r. (1999). competitive mechanisms subserve attention in macaque areas V2 and V4. journal of neuroscience, 19(5), 1736-1753.
- moran, j. & desimone, r. (1985). selective attention gates visual processing in the extrastriate cortex. science, 229(4715), 782-784.
- cohen, m. r. & maunsell, j. h. r. (2009). attention improves performance primarily by reducing interneuronal correlations. nature neuroscience, 12(12), 1594-1600.
- womelsdorf, t., anton-erxleben, k., piber, f. & treue, s. (2006). dynamic shifts of visual receptive fields in cortical area MT by spatial attention. nature neuroscience, 9(9), 1156-1160.
- fries, p., reynolds, j. h., rorie, a. e. & desimone, r. (2001). modulation of oscillatory neuronal synchronization by selective visual attention. science, 291(5508), 1560-1563.
- krauzlis, r. j., lovejoy, l. p. & zenon, a. (2014). selective attention without a neocortex. cortex, 22, 103-128.
- posner, m. i. (1980). orienting of attention. quarterly journal of experimental psychology, 32(1), 3-25.
- treue, s. & martinez-trujillo, j. c. (1999). feature-based attention influences motion processing gain in macaque visual cortex. nature, 399(6736), 575-579.

## see also

- [[normalization_model_of_attention]]
- [[feature_vs_spatial_attention]]
- [[divisive_normalization]]
- [[winner_take_all]]
- [[neural_synchrony]]
- [[precision_weighting]]
- [[inhibitory_interneurons]]
- [[desimone_duncan]]
