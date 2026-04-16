# normalization model of attention

status: definitional. last fact-checked 2026-04-16.

**why this matters**: divisive normalization with attention-modulated gain is the biological precursor to softmax attention in transformers -- both divide a signal by the sum of competing signals, and understanding the biological version reveals when contrast gain vs response gain emerges.

## overview

in 2009, John Reynolds and David Heeger published "the normalization model of attention" in Neuron. the paper provides a unified mathematical framework that explains the diverse effects of attention on neural responses in visual cortex. the model extends Heeger's (1992) **divisive normalization** model (see [[divisive_normalization]]) -- a canonical computation where a neuron's response is divided by the pooled activity of its neighbors -- by introducing an "attention field." this attention field is a multiplicative gain applied to the stimulus drive before it enters the normalization circuit. the key insight: whether attention produces **contrast gain** (increased sensitivity), **response gain** (increased maximum response), or intermediate effects depends on the relative sizes of the attention field, the stimulus, and the **normalization pool** (the set of neurons whose activity contributes to the divisive denominator).

ML analog: the normalization equation R = (A * E) / (sigma + sum(w * A * E)) is structurally identical to softmax attention, where the numerator is the query-key product for one token and the denominator sums over all tokens. both implement competitive normalization with a bias term.

see [[reynolds_heeger]] for biographical context.

## the normalization equation without attention

the baseline model of neural responses follows divisive normalization:

    R(x, theta) = E(x, theta) / (S(x, theta) + sigma)

where:
- R(x, theta) is the response of a neuron at position x with feature preference theta
- E(x, theta) is the **excitatory stimulus drive** (the tuned input to the neuron, analogous to the pre-activation value in an ML layer)
- S(x, theta) is the **suppressive drive** (the pooled activity of the normalization pool)
- sigma is the **semi-saturation constant** (determines the contrast at half-maximum response, analogous to a temperature parameter in softmax)

the suppressive drive is computed by convolving the stimulus drive with a suppressive field s(x, theta) that defines the normalization pool:

    S(x, theta) = s(x, theta) * E(x, theta)

where * denotes convolution over space and feature dimensions. the suppressive field is typically broader than the excitatory receptive field -- it pools activity over a wider region of space and over all feature values (e.g., all orientations in V1). the integral of s normalizes to 1.

for a single stimulus at contrast c, the response simplifies to the **Naka-Rushton equation** (a hyperbolic ratio function that describes the saturating contrast response):

    r(c) = alpha * c / (c + sigma)

where alpha is the maximum response (response gain) and sigma is the semi-saturation contrast (contrast gain). this hyperbolic ratio captures the saturating contrast response function observed in visual cortex.

## adding attention: the attention field

the model adds attention as a multiplicative gain field A(x, theta) applied to the stimulus drive BEFORE normalization:

    R(x, theta) = A(x, theta) * E(x, theta) / (S(x, theta) + sigma)

    S(x, theta) = s(x, theta) * [A(x, theta) * E(x, theta)]

critically, A modulates both the numerator (the neuron's own response) and the denominator (the suppressive pool). the net effect depends on how much of the suppressive pool falls inside vs outside the attended region.

the attention field A(x, theta) is a weighting function with values >= 1 in the attended region and = 1 elsewhere. it can be:
- purely spatial: A(x) enhances a region of space regardless of features
- purely feature-based: A(theta) enhances a feature value at all locations
- combined: A(x, theta) enhances a specific feature at a specific location

## contrast gain vs response gain

the model's central prediction: the type of attentional modulation (contrast gain, response gain, or a mixture) depends on the relative sizes of the attention field, the stimulus, and the suppressive field.

### contrast gain (small stimulus, large attention field)

when the stimulus is smaller than the suppressive field and the attention field is broader than the suppressive field:

    r(c) = alpha * (gamma * c) / (gamma * c + sigma) = alpha * c / (c + sigma/gamma)

where gamma is the attention gain factor (the value of A at the attended location). the effect: sigma is replaced by sigma/gamma. the contrast response function shifts leftward along the contrast axis. the neuron responds at lower contrasts as if they were higher contrasts. the maximum response (alpha) is unchanged.

this is contrast gain: attention increases sensitivity to low-contrast stimuli without increasing the response to high-contrast stimuli. the percentage modulation by attention is largest at intermediate contrasts (near the original sigma) and diminishes at high contrasts where the response is already saturated.

this condition produces contrast gain because the attention field covers the entire suppressive pool. both the numerator and the full denominator are multiplied by gamma, and the gamma terms partially cancel at high contrast (where c >> sigma), leaving the maximum response unchanged.

### response gain (large stimulus, small attention field)

when the stimulus fills the suppressive field and the attention field is smaller than or equal to the stimulation field:

    r(c) = alpha * gamma * c / (gamma * c + beta * c + sigma)

where beta represents the contribution of unattended stimuli to the suppressive pool. at high contrast (c >> sigma):

    r(c) ~ alpha * gamma / (gamma + beta)

the effect: the maximum response scales with gamma. the contrast response function shifts upward. the percentage modulation by attention is roughly constant across all contrasts.

this is response gain: attention increases the neuron's maximum response without changing its contrast sensitivity. this condition produces response gain because the attention field covers only part of the suppressive pool. the numerator is multiplied by gamma, but most of the denominator is not, producing a net increase in gain.

### intermediate effects

most experimental conditions produce effects that are intermediate between pure contrast gain and pure response gain. the model predicts a continuous transition as the relative sizes of attention field, stimulus, and suppressive field change. this explains why different laboratories report different types of attentional modulation: they are studying different points on the same continuum.

## predictions

### unification of 30 years of data

the model unifies previously conflicting findings:

1. Luck, Chelazzi, Hillyard, and Desimone (1997) found contrast gain in V4 (small stimuli, large receptive fields = large attention field relative to stimulus). the model predicts contrast gain for this condition.

2. McAdams and Maunsell (1999) found response gain in V4 (larger stimuli). the model predicts response gain when the stimulus fills the suppressive field.

3. Reynolds, Pasternak, and Desimone (2000) found that the type of modulation in V4 depended on stimulus conditions: contrast gain with small stimuli, response gain with large stimuli. this is the model's central prediction.

### attention and surround suppression

the model predicts interactions between attention and surround suppression. attending to a center stimulus reduces the suppressive effect of surrounding stimuli (the attention field boosts the center relative to the surround). attending to a surround stimulus increases its suppressive effect on the center. Sundberg, Mitchell, and Reynolds (2009) confirmed both predictions.

### feature-based attention

the attention field can operate over feature dimensions:

- broad feature attention (A constant across orientations, selective for space) produces multiplicative scaling of tuning curves without sharpening. this matches McAdams and Maunsell (1999).

- narrow feature attention (A selective for a specific orientation) produces apparent sharpening of tuning curves: the preferred orientation is enhanced more than flanking orientations. Martinez-Trujillo and Treue (2004) found feature-based sharpening in MT, consistent with this prediction.

- the feature-similarity gain principle (Treue & Martinez-Trujillo 1999) emerges naturally: the attention field has higher values for neurons whose preferred feature matches the attended feature and lower values for those with dissimilar preferences.

### two-stimulus conditions

when two stimuli fall within a neuron's receptive field, the model produces the biased competition effects described by [[selective_attention]]: the response to the pair is intermediate between the individual responses, and attention shifts the response toward the attended stimulus. the Reynolds-Heeger model provides the quantitative framework for the Desimone-Duncan biased competition principle.

## relationship to biased competition

the normalization model and the biased competition model (see [[selective_attention]]) are complementary:

- biased competition describes the computational principle: stimuli compete, attention biases the competition.
- normalization provides the mathematical implementation: the attention field A biases the stimulus drive, and divisive normalization implements the competition.

every prediction of biased competition (winner suppresses losers, top-down bias resolves competition, receptive field shrinkage) can be derived from the normalization equation with appropriate attention field configurations. the normalization model adds quantitative precision: it predicts not just which stimulus wins, but how the response depends on contrast, stimulus size, and attention field geometry.

## extensions and critiques

### extensions

Ni, Ray, and Maunsell (2012) showed that attentional modulation in V4 depends on stimulus contrast in ways consistent with the normalization model. Verhoef and Maunsell (2017) extended the model to explain attention-related changes in response variability (Fano factor), not just mean firing rate.

the model has been extended to object-based attention (Torralbo, Beck, and Kastner 2015), where the attention field conforms to the shape of the attended object rather than being a simple spatial blob.

### critiques

the model has many free parameters (the shapes of the attention field and suppressive field, sigma, the gain factor gamma). fitting these to data requires careful experimental control. the model can potentially fit any pattern of results by adjusting these parameters. this weakens its falsifiability.

the model assumes that the attention field is a static multiplicative gain. in reality, attention is deployed dynamically over time, interacts with **saccade** (rapid eye movement) planning, and is modulated by task demands. the temporal dynamics of attention deployment are not captured.

the model does not specify the neural circuit that generates the attention field. the source of the top-down bias (prefrontal cortex, frontal eye fields, pulvinar) and how it is converted into a multiplicative gain on the stimulus drive remain open questions.

## key references

- reynolds, j. h. & heeger, d. j. (2009). the normalization model of attention. neuron, 61(2), 168-185.
- heeger, d. j. (1992). normalization of cell responses in cat striate cortex. visual neuroscience, 9(2), 181-197.
- carandini, m. & heeger, d. j. (2012). normalization as a canonical neural computation. nature reviews neuroscience, 13(1), 51-62.
- ni, a. m., ray, s. & maunsell, j. h. r. (2012). tuned normalization explains the size of attention modulations. neuron, 73(4), 803-813.
- reynolds, j. h., pasternak, t. & desimone, r. (2000). attention increases sensitivity of V4 neurons. neuron, 26(3), 703-714.
- martinez-trujillo, j. c. & treue, s. (2004). feature-based attention increases the selectivity of population responses in primate visual cortex. current biology, 14(9), 744-751.

## see also

- [[selective_attention]]
- [[divisive_normalization]]
- [[feature_vs_spatial_attention]]
- [[precision_weighting]]
- [[reynolds_heeger]]
- [[carandini_heeger]]
