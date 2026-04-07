# feature-based vs spatial attention

**why this matters**: the distinction between attending to "where" (spatial) vs "what" (feature-based) maps directly onto the difference between positional encoding and content-based attention in transformers, and reveals why the binding problem does not arise in architectures that never separate features into independent maps.

## overview

attention operates along multiple dimensions. the three primary modes -- spatial, feature-based, and object-based -- differ in what they select (locations, features, or objects), how broadly they modulate neural activity, and what **binding problem** (the question of how the brain combines separately encoded features into unified object representations) they solve. this binding problem is one of the oldest open questions in cognitive neuroscience.

## spatial attention

### the spotlight metaphor

spatial attention enhances processing at a specific location in the visual field. Posner (1980) established the foundational paradigm: a cue indicates where a target will appear. reaction times are faster for valid cues (~50 ms benefit) and slower for invalid cues (~30 ms cost). this cost-benefit asymmetry demonstrates that attention both enhances the attended location and suppresses unattended locations.

ML analog: spatial attention is analogous to positional masking in transformers -- selectively attending to tokens at specific positions rather than based on content.

the metaphor of a "spotlight" (Posner, Snyder & Davidson 1980) captures the intuition: attention illuminates a region of space, making stimuli there easier to detect and discriminate. the zoom lens model (Eriksen & St. James 1986) extends this: the spotlight has a variable diameter, with a tradeoff between size and resolution. a tightly focused beam provides maximal enhancement; a broadly deployed field provides less enhancement per location.

### neural mechanisms

spatial attention modulates the gain of neural responses in retinotopically organized visual areas. the effects increase along the visual hierarchy:

- V1: modest effects (~10-20% firing rate modulation), primarily in feedback-driven layers (L1, L5/6)
- V2/V3: intermediate effects (~20-30%)
- V4: strong effects (~30-50% firing rate increase, noise correlation reduction, gamma synchronization)
- IT: strongest effects (~50-80%), with attention determining which of multiple objects in the receptive field controls the neuron's response

the receptive field effects described in [[selective_attention]] -- shrinkage around the attended stimulus, shift of RF center toward the attended location -- are consequences of spatial attention applied within the framework of biased competition.

### control circuits

two parallel networks control spatial attention:

the **dorsal attention network** (DAN): bilateral **intraparietal sulcus** (IPS) and **frontal eye fields** (FEF). mediates voluntary (endogenous) spatial attention. IPS maintains a topographic map of attention priorities. FEF generates attention-directing signals through its connections with visual cortex and the superior colliculus.

the ventral attention network (VAN): right-lateralized temporoparietal junction (TPJ) and ventral frontal cortex (VFC). mediates involuntary (exogenous) reorienting. TPJ generates a "reorienting signal" when a salient or unexpected stimulus appears outside the current focus of attention. VFC interrupts the current attentional state.

Corbetta and Shulman (2002) proposed this two-network model, which has been confirmed by lesion studies (hemispatial neglect from right TPJ damage), TMS studies (disruption of IPS impairs voluntary attention), and functional connectivity analysis.

## feature-based attention

### global feature enhancement

feature-based attention enhances processing of a specific **feature value** (a particular color, orientation, or motion direction) across all spatial locations simultaneously. Treue and Martinez-Trujillo (1999) provided the key demonstration: while a monkey attended to upward-moving dots in one hemifield, neurons in area MT of the opposite hemisphere also showed enhanced responses if they were tuned to upward motion. the enhancement was global -- it affected neurons across the entire visual field, not just at the attended location.

ML analog: feature-based attention is directly analogous to content-based attention in transformers, where the query selects tokens by semantic similarity (feature match) regardless of position.

this global enhancement follows the **feature-similarity gain principle**: the modulation of a neuron's response is proportional to the cosine similarity between the attended feature and the neuron's preferred feature.

ML analog: the cosine similarity gain field is mathematically identical to the dot-product attention score in transformers: score = Q * K^T, where the query Q encodes the "attended feature" and the key K encodes each neuron's "preferred feature." neurons tuned to the attended feature are enhanced; neurons tuned to the opposite feature are suppressed; neurons tuned to orthogonal features are unaffected.

mathematically, for a neuron with preferred direction theta_pref and attended direction theta_att:

    gain = 1 + lambda * cos(theta_pref - theta_att)

where lambda is the modulation depth. this produces a gain field that smoothly varies across the feature space.

### feature attention across dimensions

feature-based attention has been demonstrated for:
- motion direction (Treue & Martinez-Trujillo 1999, in MT)
- color (Bichot, Rossi & Desimone 2005, in V4/IT)
- orientation (Scolari, Byers & Serences 2012, in V1-V4)
- spatial frequency (Serences & Boynton 2007, in V1)

feature attention and spatial attention combine multiplicatively: the enhancement from attending to a feature at the attended location is the product of the spatial attention gain and the feature attention gain (Hayden & Gallant 2009). this multiplicative interaction is consistent with the [[normalization_model_of_attention]], where the attention field A(x, theta) is separable into spatial and feature components.

### differences from spatial attention

| dimension | spatial attention | feature-based attention |
|---|---|---|
| scope | local (one or few locations) | global (all locations simultaneously) |
| time course | ~200 ms to deploy endogenously | ~100-200 ms, slightly faster for pop-out features |
| neural substrate | retinotopic maps (V1-V4, IPS, FEF) | feature-selective areas (V4/color, MT/motion) |
| capacity limit | ~4 items at distinct locations (Pylyshyn & Storm 1988) | one feature value per dimension |
| competition level | within spatial receptive field | within feature map |
| interaction with normalization | shifts attention field over space | shifts attention field over feature dimension |

## object-based attention

### spreading within objects

object-based attention enhances processing of all features belonging to an attended object, even features that are task-irrelevant. Duncan (1984) showed that two features of the same object can be reported more accurately than two features of different objects, even when spatial separation is controlled. this is the "same-object advantage."

neural evidence: O'Craven, Downing, and Kanwisher (1999) used overlapping transparent images (a face and a house, drifting in different directions). attending to the face also enhanced activity in motion-selective area MT if the face was moving, even though the task required judging the face's identity, not its motion. attention spread from the task-relevant feature (face identity) to the task-irrelevant feature (motion direction) because they belonged to the same object.

Roelfsema, Lamme, and Spekreijse (1998) showed a similar effect in V1: neurons along an entire attended curve showed enhanced activity, demonstrating that object-based attention can spread along connected visual contours.

### the binding problem

the binding problem (Treisman 1996, von der Malsburg 1981) asks: how does the brain know which features belong to which object? if separate populations of neurons encode color (red, green), shape (circle, square), and location (left, right), how does the brain represent "a red circle on the left and a green square on the right" without confusing the alternative combination?

ML analog: transformers avoid the binding problem entirely by representing each token as a single dense vector that encodes all features jointly. there are no separate feature maps to bind. this architectural choice trades the brain's modular feature extraction for a simpler but higher-dimensional representation.

Treisman and Gelade's (1980) feature integration theory proposed the foundational framework. they distinguished two stages:

1. preattentive stage: basic features (color, orientation, size, motion) are registered automatically, in parallel, across the visual field. these features are represented in separate feature maps. no attention is required.

2. focused attention stage: attention is required to bind features from different maps into a unified object representation. binding occurs serially -- one object at a time -- at the attended location.

the behavioral evidence is the conjunction search asymmetry:

- feature search (find the red item among green items): reaction time is independent of the number of distractors. search is parallel.
- conjunction search (find the red circle among red squares and green circles): reaction time increases linearly with the number of distractors. search is serial.

this asymmetry demonstrates that individual features can be detected preattentively, but conjunctions of features require attention. without attention, features can be misbound, producing "illusory conjunctions" -- perceiving a red circle and a blue triangle as a blue circle and a red triangle (Treisman & Schmidt 1982).

### binding mechanisms

three mechanisms have been proposed for how attention solves the binding problem:

**temporal synchrony hypothesis** (Singer & Gray 1995): features of the same object are bound by temporal synchrony -- neurons encoding different features of the same object fire in synchrony (within the same gamma cycle, ~25 ms window), while neurons encoding features of different objects fire at different phases. this provides a temporal code for binding that does not require dedicated "binding neurons." the hypothesis is supported by evidence of stimulus-dependent gamma synchronization in cat and monkey visual cortex, but remains controversial: whether synchrony is a cause or consequence of binding is debated.

**spatial coincidence**: features are bound at the attended location because spatial attention gates which feature representations reach the binding mechanism. this is essentially Treisman's original proposal: attention selects a location, and features at that location are bound by virtue of being simultaneously active in their respective feature maps.

**feedback binding** (Roelfsema 2006): top-down feedback from object representations in IT cortex to feature-selective areas in V1-V4 establishes which features belong to the current object. this is a hierarchical binding mechanism that does not require synchrony or spatial gating, but instead relies on the recurrent connections between cortical areas.

## relationship to todorov

todorov's architecture has no explicit feature-based or spatial attention mechanisms. the MLA layers perform content-based retrieval (what is relevant to the current query), which is closer to feature-based attention than spatial attention: MLA selects tokens by content similarity, not by position. the KDA layers perform position-independent state accumulation, which has no direct attentional analog.

the binding problem does not arise in the same form: token representations are dense vectors that encode all "features" (semantic, syntactic, positional) in a single embedding. there is no separate binding step because features are never separated into independent maps in the first place. this is a fundamental architectural difference from the biological visual system.

## challenges

the feature-vs-spatial attention distinction faces several unresolved problems. first, the two systems may not be as independent as the framework suggests. recent evidence shows that feature-based attention has spatial biases (stronger enhancement near the attended location) and spatial attention has feature biases (stronger enhancement for task-relevant features) (Maunsell & Treue 2006; Martinez-Trujillo & Treue 2004). the clean separation into independent systems may be a methodological artifact of experiments designed to isolate one mode at a time.

second, the binding problem remains unsolved despite decades of research. the three proposed mechanisms (temporal synchrony, spatial coincidence, feedback binding) each have empirical support and empirical problems. none has been shown to be necessary and sufficient for binding. it is possible that binding uses all three mechanisms in different contexts, but this makes the theory difficult to falsify.

third, the translation to ML architectures is incomplete. transformers use content-based attention (feature-like) with positional encoding (spatial-like), but the interaction between these two signals is additive and fixed. biological systems dynamically switch between spatial and feature modes depending on task demands. whether a dynamic attention mode-switching mechanism would benefit ML architectures is unexplored.

## key references

- posner, m. i. (1980). orienting of attention. quarterly journal of experimental psychology, 32(1), 3-25.
- treisman, a. m. & gelade, g. (1980). a feature-integration theory of attention. cognitive psychology, 12(1), 97-136.
- treue, s. & martinez-trujillo, j. c. (1999). feature-based attention influences motion processing gain in macaque visual cortex. nature, 399(6736), 575-579.
- duncan, j. (1984). selective attention and the organization of visual information. journal of experimental psychology: general, 113(4), 501-517.
- o'craven, k. m., downing, p. e. & kanwisher, n. (1999). fMRI evidence for objects as the units of attentional selection. nature, 401(6753), 584-587.
- roelfsema, p. r., lamme, v. a. f. & spekreijse, h. (1998). object-based attention in the primary visual cortex of the macaque monkey. nature, 395(6700), 376-381.
- singer, w. & gray, c. m. (1995). visual feature integration and the temporal correlation hypothesis. annual review of neuroscience, 18(1), 555-586.
- corbetta, m. & shulman, g. l. (2002). control of goal-directed and stimulus-driven attention in the brain. nature reviews neuroscience, 3(3), 201-215.

## see also

- [[selective_attention]]
- [[normalization_model_of_attention]]
- [[gamma_oscillations]]
- [[neural_synchrony]]
- [[winner_take_all]]
