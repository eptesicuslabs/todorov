# precision weighting

status: definitional. last fact-checked 2026-04-16.

**why this matters**: precision weighting reframes attention as inference about reliability -- attending to a stimulus means believing its prediction errors are precise (low noise). in ML, this maps directly to the scaling factors in attention mechanisms, the beta gate in KDA, and more broadly to any architecture that modulates signal gain based on estimated confidence.

## summary

**precision** (the inverse variance of a probability distribution: Pi = Sigma^(-1), measuring how reliable or concentrated the distribution is) determines, in the [[predictive_coding]] and [[free_energy_principle]] frameworks, how much influence a prediction error has on updating representations. a precision-weighted prediction error is:

    epsilon = Pi * (x - mu)

where x is the observed input, mu is the predicted input, and Pi is the precision. high precision means the prediction error is reliable and should strongly drive the update. low precision means the error is noisy and should be downweighted.

the central claim of Feldman & Friston (2010): attention IS precision optimization. attending to a stimulus means increasing the precision (gain) of the prediction errors associated with that stimulus. this is implemented neurally through **synaptic gain modulation** (changing the amplification factor of a neuron's response to its inputs) of superficial pyramidal cells (error units) by neuromodulatory systems (dopamine, acetylcholine, noradrenaline). the formalism reframes attention not as a selection mechanism but as an inference problem: the brain must infer not only the causes of its sensory data but also the RELIABILITY of that data. ML analog: this is equivalent to the 1/sqrt(d_k) scaling in transformer attention, or more generally to any learned temperature parameter that controls how sharply attention scores are distributed.

## mechanism

### mathematical formulation

in the [[free_energy_principle]] framework, the generative model at each hierarchical level has the form:

    x_l = g_l(theta_{l+1}) + noise_l

where noise_l ~ N(0, Sigma_l). the precision at level l is:

    Pi_l = Sigma_l^(-1)

for the multivariate case, Pi_l is a precision MATRIX (inverse covariance matrix). precision-weighted prediction errors are:

    epsilon_l = Pi_l * (x_l - g_l(theta_{l+1}))
    epsilon_l = Pi_l * e_l

the variational free energy at level l is:

    F_l = 1/2 * e_l^T * Pi_l * e_l + 1/2 * log |Sigma_l|
    F_l = 1/2 * e_l^T * Pi_l * e_l - 1/2 * log |Pi_l| + const

the first term penalizes large prediction errors, weighted by precision. the second term penalizes high precision itself (it is "expensive" to be confident). this prevents the trivial solution of setting precision to zero (ignoring all errors).

### representation update with precision

the update rule for the representation mu_l (the conditional expectation of the hidden cause at level l) becomes:

    dmu_l/dt = (dg_{l-1}/dmu_l)^T * Pi_{l-1} * e_{l-1} - Pi_l * e_l

compare with the unweighted [[predictive_coding]] update:

    dmu_l/dt = (dg_{l-1}/dmu_l)^T * e_{l-1} - e_l

the precision matrices Pi_{l-1} and Pi_l act as gain factors. a level with high precision dominates the update; a level with low precision is ignored.

### precision optimization

precision is itself inferred by minimizing free energy:

    dPi_l/dt = -dF_l/dPi_l

for scalar precision (pi_l = sigma_l^(-2)):

    dpi_l/dt = 1/2 * (pi_l^(-1) - e_l^2)

at equilibrium: pi_l = 1/e_l^2, meaning the estimated precision equals the inverse squared prediction error. this is the brain learning how reliable its predictions are at each level.

for the precision to be state-dependent (which is required for attention), the precision is parameterized as a function of the hidden states:

    pi_l = f(theta_{l+1})

this means that higher-level representations can modulate the precision of lower-level prediction errors. top-down attention is exactly this: higher cortical areas adjusting the gain (precision) of error signals in lower areas.

### attention as precision optimization

Feldman & Friston (2010) formalized attention as follows:

1. the brain has a generative model where sensory noise variance DEPENDS on hidden states: Sigma(x, theta)
2. during inference, the brain optimizes not only its beliefs about hidden causes (mu) but also its beliefs about precision (pi)
3. "attending to a feature" means expecting high precision (low noise) for prediction errors about that feature
4. this expectation increases the GAIN of error units encoding that feature, amplifying their influence on the representation update

the key insight: attention is not a separate mechanism that "selects" inputs. it is an integral part of Bayesian inference. whenever the brain infers the causes of its sensory data, it MUST also infer the reliability (precision) of that data. the result is differential gain control over error signals -- which is functionally identical to selective attention.

mathematical formulation of the attentional gain:

    epsilon_attended = Pi_high * e
    epsilon_unattended = Pi_low * e

the prediction errors are the same, but their influence on the representation update differs by a factor of Pi_high / Pi_low. this is equivalent to multiplicative gain modulation.

## neural implementation

### synaptic gain as precision

precision is implemented as the SYNAPTIC GAIN of error-encoding neurons (superficial pyramidal cells). high precision = high gain = large postsynaptic response to a given input. low precision = low gain = small response.

two neurobiological mechanisms implement gain modulation:

**neuromodulatory gain**: dopamine, acetylcholine, and noradrenaline modulate the excitability of pyramidal cells through metabotropic receptors. this is slow (hundreds of milliseconds) and diffuse (affects entire cortical columns or regions).

**oscillatory gain**: gamma-band oscillations (30-100 Hz) modulate the effective gain of neural populations by synchronizing or desynchronizing their firing. synchronized populations have higher effective precision because their signals sum coherently, while desynchronized populations have lower effective precision. this is fast (tens of milliseconds) and can be spatially specific.

### neuromodulatory specificity

different neuromodulatory systems are proposed to modulate precision at different levels of the hierarchy:

**acetylcholine** (basal forebrain): modulates sensory precision. cholinergic projections to primary sensory cortex increase the gain of bottom-up (feedforward) prediction errors. this explains why acetylcholine enhances sensory processing and why cholinergic agonists improve detection of weak stimuli. experimental evidence: ACh enhances the precision of bottom-up synaptic transmission in cortical hierarchies by optimizing the gain of supragranular pyramidal cells (eLife, 2024).

**dopamine** (VTA/SNc): modulates precision of predictions about rewards and action outcomes. dopaminergic projections to prefrontal cortex and basal ganglia adjust the confidence in expected outcomes. this connects the free energy framework to reinforcement learning: dopamine prediction errors are precision-weighted prediction errors about expected rewards. experimental evidence: precision-weighted cortical prediction error signals are mediated by dopamine and impaired in psychosis (Molecular Psychiatry, 2020).

**noradrenaline** (locus coeruleus): modulates GLOBAL precision, signaling unexpected changes in environmental volatility. high noradrenaline = low global precision = "everything is uncertain, pay attention to everything." this explains the alerting and arousal functions of the noradrenergic system. the relationship to the Yerkes-Dodson inverted-U: moderate noradrenaline optimizes precision weighting; too much or too little degrades it.

**serotonin** (raphe nuclei): proposed to modulate the temporal precision of predictions -- how confident the brain is about WHEN expected events will occur. this connects to temporal discounting and the role of serotonin in impulsivity.

## relationship to todorov

todorov does not implement precision weighting. the cross-entropy loss treats all token predictions equally -- there is no per-token or per-layer confidence weighting of the error signal.

however, several architectural features have precision-like properties:

**RMSNorm**: applied before each sublayer, RMSNorm scales the representation by the inverse root-mean-square of its elements. this is a form of gain normalization that stabilizes the variance of the representation. it does not modulate precision adaptively based on context, but it prevents runaway activation magnitudes.

**the KDA beta gate**: beta_t = sigmoid(beta_proj(x_t)) is a data-dependent scalar that modulates the write strength to the KDA state. high beta = "this token is important, write it strongly." low beta = "this token is unimportant, write it weakly." this is functionally analogous to precision weighting of input: the gate determines how much influence each token has on the recurrent state. unlike biological precision, beta is computed from the input alone, not from a comparison between input and prediction.

**attention scores (in MLA layers)**: softmax attention scores weight the contribution of each key-value pair by relevance to the current query. this is equivalent to position-specific precision weighting: high attention = high precision (reliable association), low attention = low precision (irrelevant association). the mathematical form is different (softmax vs inverse variance), but the computational role is identical: differential gain control over information sources.

**alpha (KDA decay rate)**: alpha determines how quickly old state is erased. this is analogous to the TEMPORAL precision of the recurrent state. high alpha (slow decay) = high temporal precision = "old information is reliable, keep it." low alpha (fast decay) = low temporal precision = "old information is unreliable, erase it." alpha is fixed per channel, not adaptively modulated by context, which limits the analogy.

none of these features implement precision in the full predictive coding sense (context-dependent inference about the reliability of prediction errors). they are engineering approximations that serve similar functional roles.

## challenges

1. **circularity problem.** to compute the precision of a prediction error, you need the prediction error (to estimate its variance). but the prediction error itself depends on precision (because precision-weighted errors drive the representation update). the brain must solve this chicken-and-egg problem through iterative inference, oscillatory dynamics, or separate fast/slow timescales for precision and representation estimation.

2. **neuromodulatory implementation is too slow for fast attention.** cholinergic effects take ~100 ms and dopaminergic effects take ~seconds, which limits the temporal resolution of precision modulation. fast attentional shifts (<100 ms) likely rely on oscillatory gain modulation rather than neuromodulatory precision, suggesting that multiple distinct mechanisms implement what the theory describes as a single operation.

3. **precision estimation requires meta-knowledge.** to infer the precision of prediction errors at a given level, the system must have a model of its own noise statistics. this meta-cognitive requirement adds a layer of complexity that is difficult to implement in biological circuits and raises the question of where the precision priors come from. the theory pushes this back to evolutionary priors or higher hierarchical levels, which risks infinite regress.

## key references

- Feldman, H. & Friston, K. (2010). Attention, uncertainty, and free-energy. Frontiers in Human Neuroscience, 4, 215.
- Friston, K. (2009). The free-energy principle: a rough guide to the brain? Trends in Cognitive Sciences, 13(7), 293-301.
- Kanai, R., Komura, Y., Shipp, S. & Friston, K. (2015). Cerebral hierarchies: predictive processing, precision and the pulvinar. Philosophical Transactions of the Royal Society B, 370(1668), 20140169.
- Parr, T. & Friston, K. J. (2019). Attention or salience? Current Opinion in Psychology, 29, 1-7.
- Marshall, L. et al. (2016). Pharmacological fingerprints of contextual uncertainty. PLoS Biology, 14(11), e1002575.
- Moran, R. J. et al. (2013). Free energy, precision and learning: the role of cholinergic neuromodulation. Journal of Neuroscience, 33(19), 8227-8236.
- Bogacz, R. (2017). A tutorial on the free-energy framework for modelling perception and learning. Journal of Mathematical Psychology, 76, 198-211.

## see also

- [[predictive_coding]]
- [[free_energy_principle]]
- [[predictive_coding_vs_next_token]]
- [[predictive_coding_to_training_objective]]
