# free energy principle

**why this matters**: the FEP reframes all neural computation as variational inference -- minimizing an upper bound on surprise. this is the same mathematical framework as the variational autoencoders (VAEs) and ELBO optimization used throughout modern ML, making the FEP a bridge between neuroscience theory and practical generative modeling.

## summary

the free energy principle (FEP) is Karl Friston's proposal that all adaptive systems -- from single cells to entire brains -- can be described as minimizing a quantity called **variational free energy** (an upper bound on surprise, or negative log-evidence, from variational inference). minimizing it is equivalent to maximizing the evidence for the system's internal model of its environment. the principle subsumes [[predictive_coding]] as a special case (linear-Gaussian generative model) and extends it to arbitrary probability distributions. it also adds **active inference**: the idea that organisms not only update their beliefs to match sensory data (perception) but also act on the world to make sensory data match their beliefs (action). ML analog: variational free energy is the same quantity as the negative ELBO (evidence lower bound) minimized in variational autoencoders. the FEP claims that the brain performs the same optimization that VAE training performs.

the mathematical formulation:

    F = E_q[log q(theta) - log p(y, theta)]

where F is variational free energy, q(theta) is the recognition density (the brain's approximate posterior over hidden causes theta), p(y, theta) is the generative model (the joint distribution over sensory data y and causes theta). minimizing F with respect to q makes q approximate the true posterior p(theta|y), which is Bayes-optimal perception. minimizing F with respect to action a changes the sensory data y to reduce surprise, which is active inference.

this is the most ambitious unifying theory in neuroscience. it claims that ALL neural computation -- perception, action, learning, attention, emotion -- is free energy minimization. the claim is either trivially true (any system that persists must avoid surprising states) or profoundly false (the framework is unfalsifiable because any behavior can be redescribed as free energy minimization). the truth is contested.

## mechanism

### variational free energy

for a system receiving sensory data y and maintaining an internal model with hidden variables theta, the variational free energy is:

    F = E_q[log q(theta)] - E_q[log p(y, theta)]

where q(theta) is the **recognition density** (the brain's approximate posterior, equivalent to the encoder network in a VAE) and p(y, theta) is the **generative model** (the joint distribution over observations and hidden causes, equivalent to the decoder in a VAE). this decomposes in two useful ways.

**decomposition 1 (energy minus entropy)**:

    F = E_q[-log p(y, theta)] - H[q(theta)]
    F = "energy" - "entropy"

the first term is the expected negative log-joint (the "energy"). the second term is the entropy of the recognition density. minimizing F means finding a q that assigns high probability to theta values that have high joint probability p(y, theta), while remaining as broad (entropic) as possible. this is the standard variational inference objective.

**decomposition 2 (surprise plus KL divergence)**:

    F = -log p(y) + D_KL[q(theta) || p(theta|y)]
    F = "surprise" + "divergence"

since D_KL >= 0, free energy is always >= surprise (-log p(y)). minimizing F with respect to q reduces D_KL, making q a better approximation to the true posterior. minimizing F with respect to action (which changes y) reduces surprise directly.

### the generative model

the brain's generative model has hierarchical structure:

    p(y, theta) = p(y | theta_1) * p(theta_1 | theta_2) * ... * p(theta_L)

level 1 generates sensory data from low-level causes. level 2 generates low-level causes from higher-level causes. the top level has a prior. this is hierarchical Bayesian inference.

for the Gaussian case (which reduces to [[predictive_coding]]):

    p(theta_l | theta_{l+1}) = N(theta_l; g_l(theta_{l+1}), Sigma_l)

where g_l is the generative function at level l and Sigma_l is the noise covariance. the variational free energy becomes:

    F = sum_l [ 1/2 * e_l^T * Sigma_l^(-1) * e_l + 1/2 * log |Sigma_l| ]

where e_l = theta_l - g_l(theta_{l+1}) is the prediction error at level l. this is the sum of precision-weighted squared errors -- exactly the objective that [[predictive_coding]] minimizes.

### inference dynamics

the brain minimizes F with respect to the **sufficient statistics** (summary values that capture all the information in a distribution) of q (denoted mu, the conditional expectations). using a **Laplace approximation** (approximating q as a Gaussian centered on the mode of the posterior):

    dmu_l/dt = -dF/dmu_l

for the hierarchical Gaussian model, this yields:

    dmu_l/dt = -Sigma_l^(-1) * (mu_l - g_l(mu_{l+1})) + (dg_{l-1}/dmu_l)^T * Sigma_{l-1}^(-1) * (mu_{l-1} - g_{l-1}(mu_l))

this is precisely the predictive coding update rule: the representation at each level adjusts to balance the top-down prediction error and the bottom-up prediction error, weighted by their respective precisions.

### learning (parameter optimization)

the generative model parameters Theta (the synaptic weights) are optimized by gradient descent on F:

    dTheta_l/dt = -eta * dF/dTheta_l

for a linear generative model g_l(mu_{l+1}) = Theta_l * mu_{l+1}:

    dTheta_l/dt = eta * epsilon_l * mu_{l+1}^T

where epsilon_l = Sigma_l^(-1) * e_l is the precision-weighted prediction error. this is a Hebbian learning rule: the weight change is the outer product of the (precision-weighted) error and the presynaptic activity. it is local and biologically plausible.

### precision optimization

the noise covariance Sigma_l (or equivalently, the precision Pi_l = Sigma_l^(-1)) is also optimized:

    dSigma_l/dt = -eta * dF/dSigma_l = eta * (epsilon_l * epsilon_l^T - Sigma_l^(-1))

at equilibrium, Sigma_l = <e_l * e_l^T>, the empirical covariance of the prediction errors. precision optimization is how the brain learns the reliability of its predictions at each level. see [[precision_weighting]] for the relationship between precision and attention.

### active inference

action minimizes F by changing sensory data:

    da/dt = -dF/da = -dF/dy * dy/da

the organism acts to make its sensory inputs match its predictions. this is not the same as reward maximization: the organism seeks to confirm its expectations, not to maximize pleasure. however, if the generative model encodes "preferred observations" (homeostatic set points), then active inference reduces to goal-directed behavior.

the controversial claim: active inference replaces reward-based reinforcement learning. organisms do not maximize reward; they minimize surprise relative to their prior expectations about what they should observe. reward signals are reinterpreted as prior beliefs about the precision of sensory prediction errors.

## biological implementation

### cortical microcircuit

the free energy principle maps to cortical circuits as follows:

- **representation units** (deep pyramidal cells, layers 5/6): encode the conditional expectations mu_l, send top-down predictions to lower areas
- **error units** (superficial pyramidal cells, layers 2/3): encode precision-weighted prediction errors epsilon_l, send bottom-up errors to higher areas
- **precision units** (inhibitory interneurons, neuromodulatory systems): encode and modulate precision Pi_l, gate the gain of error units

### neuromodulatory systems as precision signals

- **dopamine**: modulates precision of predictions about rewards and action outcomes (basal ganglia, prefrontal cortex)
- **acetylcholine**: modulates precision of sensory prediction errors (cortical gain control, basal forebrain projections)
- **noradrenaline**: modulates precision globally, signaling unexpected uncertainty (locus coeruleus, arousal system)
- **serotonin**: modulates precision of long-term predictions (raphe nuclei, temporal discounting)

## criticisms

### unfalsifiability

the most serious criticism: any behavior can be redescribed as free energy minimization by choosing an appropriate generative model. if the organism does X, one can always construct a generative model under which X minimizes free energy. this makes the FEP unfalsifiable in principle (Andrews 2021, Colombo & Wright 2021). defenders argue that the FEP is a framework (like Bayesian inference), not a specific model, and that specific instantiations of FEP (e.g., particular generative models) are falsifiable.

### computational intractability

exact Bayesian inference is intractable for most generative models. the FEP relies on variational approximations (Laplace, mean-field), which introduce approximation errors. the brain may not have the computational resources to perform even approximate variational inference in real time, especially for deep hierarchies with nonlinear generative models.

### the dark room problem

if organisms minimize surprise, why don't they seek out maximally predictable environments (a dark, silent room)? the standard answer is that the generative model includes prior beliefs about expected sensory states (e.g., "I expect to be fed"), and a dark room violates these priors. but this answer moves the explanatory burden to the prior, raising the question of where priors come from. evolution provides some, but not all.

### relationship to reward

active inference claims to replace reward-based learning, but it smuggles reward back in through "prior preferences" -- prior beliefs about what sensory states should be observed. whether this is a genuine theoretical advance or a relabeling of reward as "expected observation" remains debated.

## relationship to todorov

todorov does not implement the free energy principle. the architecture minimizes cross-entropy loss on next-token prediction, not variational free energy. the training objective is global (loss computed at the output, gradients backpropagated through all layers), not local (per-layer free energy minimization).

however, two observations connect the FEP to todorov's architecture:

1. **Millidge et al. (2021, 2022) showed that predictive coding (and thus FEP under Gaussian assumptions) converges to exact backpropagation gradients.** this means that a network trained by backpropagation is, in a precise mathematical sense, computing the same weight updates that predictive coding would compute at convergence. todorov's backpropagation-trained parameters may therefore approximate the parameters that a predictive coding network would learn.

2. **the KDA delta rule decay (alpha) can be interpreted as a precision parameter.** when alpha is close to 0, old state is aggressively erased (high "surprise" overwrites everything). when alpha is close to 1, old state persists (low surprise, stable prediction). alpha is not learned to minimize free energy, but its functional role -- gating the persistence of internal state based on a learned reliability estimate -- is analogous to precision optimization.

these connections are suggestive but not sufficient to claim that todorov implements the FEP. the critical missing element is LOCAL error computation: in the FEP, each layer computes its own prediction error and updates independently. in todorov, errors are only computed at the output and propagated backward globally.

## key references

- Friston, K. (2005). A theory of cortical responses. Philosophical Transactions of the Royal Society B, 360(1456), 815-836.
- Friston, K. (2010). The free-energy principle: a unified brain theory? Nature Reviews Neuroscience, 11(2), 127-138.
- Bogacz, R. (2017). A tutorial on the free-energy framework for modelling perception and learning. Journal of Mathematical Psychology, 76, 198-211.
- Friston, K. (2006). A free energy principle for the brain. Journal of Physiology-Paris, 100(1-3), 70-87.
- Andrews, M. (2021). The math is not the territory: navigating the free energy principle. Biology & Philosophy, 36, 30.
- Colombo, M. & Wright, C. (2021). First principles in the life sciences: the free-energy principle, organicism, and mechanism. Synthese, 198, 3463-3488.
- Millidge, B., Tschantz, A. & Buckley, C. L. (2022). Predictive coding approximates backprop along arbitrary computation graphs. Neural Computation, 34(6), 1329-1368.
- Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P. & Pezzulo, G. (2017). Active inference: a process theory. Neural Computation, 29(1), 1-49.

## see also

- [[predictive_coding]]
- [[precision_weighting]]
- [[predictive_coding_vs_next_token]]
- [[predictive_coding_to_training_objective]]
