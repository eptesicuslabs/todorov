# predictive coding

status: definitional. last fact-checked 2026-04-16.

**why this matters**: predictive coding provides a biologically plausible alternative to backpropagation -- Millidge et al. (2022) proved that predictive coding converges to exact backprop gradients on arbitrary computation graphs. this means every backprop-trained network (including todorov) is implicitly computing the same weight updates that a predictive coding network would produce, linking next-token prediction loss to hierarchical error minimization.

## summary

predictive coding is a theory of cortical function in which each level of a **neural hierarchy** (a stack of processing stages, analogous to layers in a deep network) maintains an internal **generative model** (a model that can produce or predict data, as opposed to a discriminative model that only classifies). each level predicts the activity at the level below. only the residual **prediction errors** -- the discrepancy between the prediction and the actual input -- propagate upward. this inverts the classical view of the cortex as a purely feedforward feature extractor: feedback connections carry predictions downward and feedforward connections carry errors upward. the foundational model was proposed by Rao & Ballard (1999), building on earlier ideas from Helmholtz (perception as unconscious inference) and Mumford (1992, cortical feedback as top-down priors). ML analog: this is equivalent to a hierarchical autoencoder where each layer predicts its input and only the residual passes to the next layer -- structurally similar to residual connections where each block computes x + f(x).

the core equations for a two-level hierarchy:

    e_l = x_l - f(r_{l+1})
    dr_{l+1}/dt = -e_{l+1} + (df/dr_{l+1})^T * e_l

where e_l is the prediction error at level l, x_l is the input to level l, f is the generative model mapping from the higher-level representation r_{l+1} to the predicted input at level l, and e_{l+1} is the prior error from a still-higher level. the representation r_{l+1} adjusts to simultaneously minimize the bottom-up prediction error (e_l) and the top-down prior error (e_{l+1}).

this framework unifies perception, learning, and attention under a single principle: all neural computation is inference on a hierarchical generative model. it is the direct predecessor to Friston's [[free_energy_principle]].

## mechanism

### the generative model

the fundamental assumption is that the brain maintains a hierarchical generative model of sensory causes. for a hierarchy with L levels:

    level 1 (sensory): receives raw sensory data x_0
    level 2: generates predictions of level 1 activity
    ...
    level L: generates predictions of level L-1 activity

at each level l, a representation r_l encodes the "best current explanation" of the input. a generative function f_l maps r_{l+1} to a prediction of r_l:

    prediction_l = f_l(r_{l+1})

the generative function f_l is parameterized by synaptic weights U_l (in the original Rao & Ballard formulation, these are the feedback connection weights).

### prediction error computation

the prediction error at each level is the difference between the actual input and the top-down prediction:

    e_l = r_l - f_l(r_{l+1})

or equivalently, at the sensory level:

    e_0 = x_0 - f_0(r_1)

this error has two components in the full model: a bottom-up error weighted by the inverse sensory noise variance, and a top-down error weighted by the inverse prior variance. the precision-weighted version (see [[precision_weighting]]):

    e_l^(bottom-up) = Sigma_l^(-1) * (r_l - f_l(r_{l+1}))
    e_l^(top-down) = Sigma_{l+1}^(-1) * (r_{l+1} - v_{l+1})

where Sigma_l is the noise covariance at level l and v_{l+1} is the prior mean at level l+1.

### representation update rule

the representation at each level updates by gradient descent on the combined prediction error:

    dr_l/dt = -e_l^(top-down) + (df_{l-1}/dr_l)^T * e_{l-1}^(bottom-up)

expanding:

    dr_l/dt = -Sigma_l^(-1) * (r_l - f_l(r_{l+1})) + (df_{l-1}/dr_l)^T * Sigma_{l-1}^(-1) * (r_{l-1} - f_{l-1}(r_l))

the first term pulls r_l toward the top-down prediction f_l(r_{l+1}). the second term pushes r_l to better explain the bottom-up input r_{l-1}. the balance is determined by the precision (inverse variance) at each level.

### synaptic weight learning

the generative model parameters U_l are learned by gradient descent on the prediction error:

    dU_l/dt = eta * e_l * (dr_l/dU_l)^T

for a linear generative model f_l(r_{l+1}) = U_l * r_{l+1}, this simplifies to:

    dU_l/dt = eta * e_l * r_{l+1}^T

this is a **Hebbian outer product rule**: the weight change is proportional to the product of the prediction error (postsynaptic) and the representation at the level above (presynaptic). ML analog: this is identical to the gradient of the mean squared error loss with respect to the weight matrix in a linear layer -- predictive coding derives backpropagation from local Hebbian learning. this is the learning rule for the FEEDBACK connections. it is local: only information available at the synapse is needed.

## neural implementation

### error units vs representation units

Rao & Ballard (1999) proposed a mapping to cortical circuitry:

**error units** correspond to superficial pyramidal neurons (layers 2/3). they compute the prediction error e_l and send it upward through feedforward connections to the next cortical area. their activity should be suppressed when predictions are accurate (small error) and elevated when predictions fail (large error). this is consistent with the observation that superficial pyramidal neurons project to higher cortical areas via feedforward connections.

**representation units** correspond to deep pyramidal neurons (layers 5/6). they encode the current best representation r_l and send predictions downward through feedback connections to the level below. their activity should be more stable and sustained than error unit activity. this is consistent with deep layer neurons providing the primary source of cortico-cortical feedback.

the anatomical predictions:
- feedforward connections (superficial to layer 4 of next area): carry prediction errors
- feedback connections (deep layers to superficial layers of lower area): carry predictions
- within-area connections: compute the prediction error (subtract prediction from input)

### experimental evidence

**repetition suppression**: when a stimulus is repeated, the cortical response decreases. predictive coding explains this as the brain successfully predicting the repeated stimulus, reducing the prediction error. this is observed across all sensory modalities and has been demonstrated with fMRI (Summerfield et al. 2008), EEG (mismatch negativity), and single-unit recordings.

**mismatch negativity (MMN)**: the auditory cortex produces a larger response to unexpected (deviant) stimuli than to expected (standard) stimuli. predictive coding interprets this as an increased prediction error when the prediction (standard tone) fails to match the input (deviant tone).

**extra-classical receptive field effects**: V1 neurons respond differently to the same local stimulus depending on the surrounding context. Rao & Ballard (1999) showed that their predictive coding model, when trained on natural images, produced error units with properties matching extra-classical receptive field effects including end-stopping, which had previously been attributed to feedforward mechanisms.

**top-down modulation latency**: feedback-mediated effects in V1 appear ~30-50 ms after feedforward responses, consistent with a top-down prediction arriving after the initial feedforward sweep. Lamme & Roelfsema (2000) showed that the initial feedforward response to a stimulus is context-independent, while later activity is strongly modulated by context -- matching the prediction error interpretation.

**cortical layer-specific responses**: Keller & Mrsic-Flogel (2018) reviewed evidence that neural activity consistent with prediction error signals is found predominantly in superficial layers, while activity consistent with prediction signals is found in deep layers, matching the proposed circuit.

## relationship to other frameworks

### vs efficient coding

[[efficient_coding]] (Barlow 1961) proposes that the goal of sensory processing is to remove redundancy from neural representations. predictive coding achieves this as a consequence: if the prediction is accurate, the prediction error is small and low-entropy (whitened). the two frameworks are complementary -- efficient coding describes WHAT the representation achieves (redundancy reduction), predictive coding describes HOW it achieves it (hierarchical prediction).

### vs sparse coding

[[sparse_coding]] (Olshausen & Field 1996) assumes the input is generated from a sparse set of basis functions. predictive coding does not assume sparsity in the representation, but it produces sparse ERRORS: when predictions are good, most error units are near zero. this is an important distinction -- in predictive coding, sparsity appears in the ERROR signal, not necessarily in the representation itself.

### vs free energy principle

[[free_energy_principle]] (Friston 2005, 2010) subsumes predictive coding as a special case of variational inference. predictive coding minimizes the sum of squared prediction errors, which is equivalent to minimizing variational free energy under Gaussian assumptions. the free energy framework generalizes this to arbitrary probability distributions and adds active inference (actions that reduce expected prediction error). predictive coding IS the free energy principle under linear-Gaussian generative models.

## relationship to todorov

todorov does not implement predictive coding. the architecture uses standard next-token prediction with cross-entropy loss computed at the output. there is no explicit prediction error computation between layers, no top-down connections carrying predictions, and no layer-wise error minimization.

however, several structural features of todorov have interesting parallels:

**residual connections as implicit predictions**: each layer computes x + f(x), which can be read as "the prediction x, updated by a correction f(x)." but the "correction" f(x) is not driven by an explicit error signal -- it is whatever the layer computes from its input. this is structurally similar to predictive coding but mechanistically different.

**ternary spikes and surprise**: if spike = {-1, 0, +1}, one could interpret 0 as "as predicted" (no spike) and +/-1 as "surprised" (spike). the 41% firing rate would then mean 41% surprise. this interpretation is appealing but problematic: the spike decision is based on magnitude (|x| > threshold), not on comparison with a prediction. there is no prediction to compare against.

**KDA state as implicit generative model**: the KDA state matrix S_t accumulates key-value associations and can be read out with any query. this is similar to a generative model that "generates" a value given a query. but S_t is not used to generate predictions of incoming tokens -- it is used for retrieval in the forward pass only.

see [[predictive_coding_to_training_objective]] for a detailed analysis of whether todorov could benefit from explicit predictive coding mechanisms.

## challenges

1. **the inference-learning tradeoff is unresolved.** the representation update (inference) and the weight update (learning) must converge at different timescales. the theory does not specify how this is coordinated in biological circuits. if learning occurs before inference converges, the weight updates will be based on incorrect representations.

2. **linearity assumption limits applicability.** the simplest predictive coding models assume linear or near-linear generative models, while cortical computation is highly nonlinear. extending predictive coding to deep nonlinear hierarchies introduces computational challenges (convergence of iterative inference, credit assignment across many levels) that are equivalent in difficulty to the problems that backpropagation solves.

3. **convergence requirements may be biologically implausible.** Millidge et al. (2021, 2022) showed that predictive coding converges to exact backpropagation gradients on arbitrary computation graphs. but this convergence requires iterating the inference dynamics to equilibrium at each layer before learning -- a requirement that may not be met in biological real-time processing where the brain must act on incomplete inference.

4. **the error unit / representation unit distinction is speculative.** the mapping of error units to superficial pyramidal cells and representation units to deep pyramidal cells is an anatomical prediction, not a demonstrated fact. layer-specific recordings are beginning to test this, but the evidence is still indirect and could be consistent with alternative circuit models.

## key references

- Rao, R. P. N. & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. Nature Neuroscience, 2(1), 79-87.
- Friston, K. (2005). A theory of cortical responses. Philosophical Transactions of the Royal Society B, 360(1456), 815-836.
- Bogacz, R. (2017). A tutorial on the free-energy framework for modelling perception and learning. Journal of Mathematical Psychology, 76, 198-211.
- Keller, G. B. & Mrsic-Flogel, T. D. (2018). Predictive processing: a canonical cortical computation. Neuron, 100(2), 424-435.
- Millidge, B., Tschantz, A. & Buckley, C. L. (2022). Predictive coding approximates backprop along arbitrary computation graphs. Neural Computation, 34(6), 1329-1368.
- Whittington, J. C. R. & Bogacz, R. (2017). An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity. Neural Computation, 29(5), 1229-1262.
- Huang, Y. & Rao, R. P. N. (2011). Predictive coding. WIREs Cognitive Science, 2(5), 580-593.
- Summerfield, C. et al. (2008). Neural repetition suppression reflects fulfilled perceptual expectations. Nature Neuroscience, 11(9), 1004-1006.
- Lamme, V. A. F. & Roelfsema, P. R. (2000). The distinct modes of vision offered by feedforward and recurrent processing. Trends in Neurosciences, 23(11), 571-579.

## see also

- [[free_energy_principle]]
- [[precision_weighting]]
- [[efficient_coding]]
- [[sparse_coding]]
- [[predictive_coding_vs_next_token]]
- [[predictive_coding_to_training_objective]]
