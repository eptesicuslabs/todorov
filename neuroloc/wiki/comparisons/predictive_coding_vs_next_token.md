# predictive coding vs next-token prediction

## the question

[[predictive_coding]] and next-token prediction are both prediction-error-driven learning frameworks. predictive coding says the brain maintains a hierarchical generative model and only propagates prediction errors. next-token prediction says a language model learns to predict the next token in a sequence by minimizing cross-entropy. superficially, both "predict and compute errors." are they doing the same thing?

this question is not academic for todorov. if next-token prediction IS predictive coding (or a sufficient approximation), then adding explicit predictive coding mechanisms would be redundant. if they are fundamentally different, then there may be computational benefits that todorov is missing.

## structured comparison

### directionality

**predictive coding**: bidirectional (hierarchical). each level sends predictions DOWN to the level below and receives prediction errors UP from the level below. information flows in both directions simultaneously. the hierarchy is spatial (across cortical areas), not temporal (across timesteps).

**next-token prediction**: unidirectional (temporal). the model predicts the next token x_{t+1} from the preceding context x_{1:t}. information flows forward through time (autoregressive) and forward through layers (feedforward). there are no top-down connections during inference (in a standard transformer).

**verdict**: fundamentally different. predictive coding is about SPATIAL hierarchy (level L predicts level L-1). next-token prediction is about TEMPORAL sequence (step t predicts step t+1). conflating the two requires reinterpreting layers as "levels" and residual connections as "predictions," which is a stretch.

### error signal

**predictive coding**: local prediction errors at each level. e_l = x_l - f(r_{l+1}). the error is computed between the PREDICTION from the level above and the ACTUAL representation at the current level. each level has its own error, and each error drives a local update. the error is precision-weighted (see [[precision_weighting]]).

**next-token prediction**: global cross-entropy loss at the output. L = -log p(x_{t+1} | x_{1:t}). the loss is computed at the FINAL output of the full model, comparing the predicted probability distribution with the one-hot target. there is no per-layer error signal -- all internal layers receive their error signal through backpropagation from the output loss.

**verdict**: fundamentally different error structure. predictive coding uses O(L) local errors (one per level). next-token prediction uses 1 global error, propagated through L layers by backpropagation. the information requirements for weight updates are entirely different.

### learning rule

**predictive coding**: local Hebbian updates. dU_l/dt = eta * epsilon_l * r_{l+1}^T. the weight change at each level is the outer product of the (precision-weighted) prediction error and the presynaptic representation. this is local: only information available at the synapse is needed. no backward pass required.

**next-token prediction**: backpropagation through time (BPTT). dW/dL is computed by applying the chain rule through the full computational graph. the gradient at layer l depends on the loss at the output AND the weights and activations at all layers between l and the output. this is maximally non-local.

**verdict**: fundamentally different. the claim that they produce the SAME weight updates (Whittington & Bogacz 2017, Millidge et al. 2022) is true only in the limit of iterating the predictive coding inference dynamics to convergence at each layer. in practice, biological predictive coding likely runs for a few iterations, producing approximate (not exact) backpropagation gradients.

### hierarchy

**predictive coding**: explicit multi-level hierarchy with distinct roles at each level. level l's generative model f_l maps r_{l+1} to a prediction of r_l. the hierarchy is asymmetric: feedback (top-down) carries predictions, feedforward (bottom-up) carries errors. different levels represent different levels of abstraction (edges at level 1, textures at level 2, objects at level 3, etc.).

**next-token prediction**: layers are stacked but not hierarchically structured. each layer transforms the representation, but there is no explicit prediction/error decomposition between adjacent layers. the residual stream carries the accumulated representation, and each layer adds a correction. the "hierarchy" is implicit in the learned representations, not explicit in the architecture.

**verdict**: different. a transformer's layers are not "levels" in the predictive coding sense. each layer sees the SAME residual stream and adds to it. in predictive coding, each level operates on a DIFFERENT representation (its own internal state r_l) and communicates with adjacent levels through predictions and errors.

### biological plausibility

**predictive coding**: high. local learning rules, anatomically plausible circuit (error units in superficial layers, representation units in deep layers, top-down feedback connections). the main biological weakness is the requirement for iterative inference (multiple settling steps per input).

**next-token prediction**: low. global backpropagation, symmetric weight transport problem, no biological equivalent of the backward pass. the architecture (transformer self-attention) has no known cortical analog.

**verdict**: predictive coding wins on biological plausibility. next-token prediction wins on engineering performance.

### representation sparsity

**predictive coding**: prediction errors are sparse when predictions are accurate. most error units have near-zero activity. representations (deep layer units) can be dense. the sparsity is in the ERROR, not in the representation.

**next-token prediction**: representations are typically dense (all dimensions active). todorov's ternary spikes introduce sparsity at ~41% firing rate, but this is in the REPRESENTATION, not in the error signal. the cross-entropy loss is a scalar, not a sparse signal.

**verdict**: different sparsity structure. predictive coding produces sparse errors with dense representations. transformers (standard) produce dense representations with a scalar error. todorov produces sparse representations with a scalar error. none of these match.

## the reframing question

### can next-token prediction be reframed as predictive coding?

the strongest reframing attempt proceeds as follows:

1. the residual stream IS a prediction. at any layer l, the current residual x_l is the model's "best current prediction" of the final output.
2. each layer computes f_l(x_l) = a correction to the prediction. so x_{l+1} = x_l + f_l(x_l) = old_prediction + correction.
3. the "correction" f_l(x_l) is analogous to a prediction error that updates the prediction.
4. the final output x_L is the refined prediction, and the cross-entropy loss evaluates how good this prediction is.

**why this reframing fails:**

the "correction" f_l(x_l) is NOT a prediction error. a prediction error is the difference between an ACTUAL input and a PREDICTED input: e = actual - predicted. but f_l(x_l) is computed from the prediction itself (x_l), not from a comparison between the prediction and something external. there is nothing at layer l that corresponds to the "actual input" that the prediction is being compared against. the only actual-vs-predicted comparison happens at the final output (cross-entropy loss).

in predictive coding, each level receives input from the level below (bottom-up) and predictions from the level above (top-down). the error is the difference between these two signals. in a transformer, each layer receives the residual stream from the layer below, but there is NO top-down signal. the architecture is purely feedforward during the forward pass.

### the Millidge et al. objection

Millidge et al. (2021, 2022) proved that predictive coding converges to exact backpropagation gradients on arbitrary computation graphs. does this mean next-token prediction IS predictive coding?

strictly: no. the result says that the WEIGHT UPDATES computed by predictive coding at convergence are the same as those computed by backpropagation. this is a statement about the LEARNING RULE, not about the FORWARD DYNAMICS or the REPRESENTATION. the forward pass of a predictive coding network (iterative, bidirectional, with explicit error units) is very different from the forward pass of a transformer (single-pass, feedforward). the two systems compute the same gradients but through entirely different dynamical processes.

analogy: gradient descent and the Nelder-Mead simplex method can both find the minimum of a convex function. they produce the same answer but work completely differently. saying "predictive coding approximates backprop" does not mean "transformers implement predictive coding."

### what about ternary spikes as surprise signals?

if spike = {-1, 0, +1}, one could interpret:
- 0 = "as predicted" (no spike, no surprise)
- +1 or -1 = "surprised" (spike, prediction violated)

this is appealing but has a critical flaw: the spike decision is based on MAGNITUDE (|x| > threshold), not on PREDICTION ERROR (|x - prediction| > threshold). there is no prediction to compare against. the threshold alpha * mean(|x|) is a function of the current input statistics, not of a discrepancy between input and expectation.

for ternary spikes to genuinely implement surprise signaling, one would need:

    spike_t = quantize(x_t - prediction_t, threshold_t)

where prediction_t is an explicit top-down prediction. this is not what todorov's spike mechanism computes.

## verdict

next-token prediction cannot be meaningfully reframed as predictive coding. the two frameworks share the word "prediction" but differ in every substantive aspect: directionality, error structure, learning rule, hierarchy, and sparsity.

the honest summary:

| dimension | predictive coding | next-token prediction |
|---|---|---|
| direction | hierarchical (spatial, bidirectional) | sequential (temporal, unidirectional) |
| error signal | local per-level prediction errors | global cross-entropy at output |
| error propagation | feedforward (errors up) | backpropagation (errors backward through layers) |
| learning rule | local Hebbian | global backpropagation |
| hierarchy | explicit (predictions down, errors up) | implicit (residual stream accumulates) |
| representation at each level | distinct r_l per level | shared residual stream |
| sparsity | sparse errors, dense representations | dense errors, dense representations* |
| biological plausibility | high | low |
| engineering performance | unproven at scale | state of the art |

*todorov uses ternary spikes for sparse representations, but the error signal (cross-entropy) is still a dense scalar.

**the strongest connection**: Millidge et al.'s proof that predictive coding approximates backprop means that todorov's backpropagation-trained weights APPROXIMATE the weights that a predictive coding network would learn. the representations discovered by gradient descent may therefore be similar to those discovered by hierarchical prediction error minimization. but the PROCESS by which they are discovered (global backprop vs local Hebbian updates) is fundamentally different.

**the strongest disconnection**: next-token prediction has no per-layer error signal, no top-down predictions, and no precision weighting. adding these would change the architecture into something resembling a predictive coding network. whether that change would improve performance is an empirical question that has not been answered at the scale of modern language models.

## dissenting argument

one could argue that the comparison above is too strict. the relevant question is not "are the mechanisms identical?" but "do they achieve the same computational goal?"

both frameworks learn to predict their inputs: predictive coding predicts each level's input from the level above, next-token prediction predicts the next token from the context. both drive learning by prediction errors. both develop hierarchical representations (empirically, transformers develop low-level features in early layers and high-level features in late layers, similar to the cortical hierarchy).

from this perspective, the mechanisms are different but the FUNCTION is converging. backpropagation may be nature's way of solving the same optimization problem that predictive coding solves (or vice versa). the fact that they produce the same weight updates (Millidge et al.) supports this view.

the counter-counter-argument: functional convergence does not imply mechanistic equivalence. birds and airplanes both fly but use fundamentally different mechanisms. if the mechanisms are different, the failure modes are different, the strengths are different, and the possible improvements are different. understanding the mechanistic differences is precisely what enables architectural innovation.

## key references

- Rao, R. P. N. & Ballard, D. H. (1999). Predictive coding in the visual cortex. Nature Neuroscience, 2(1), 79-87.
- Millidge, B., Tschantz, A. & Buckley, C. L. (2022). Predictive coding approximates backprop along arbitrary computation graphs. Neural Computation, 34(6), 1329-1368.
- Whittington, J. C. R. & Bogacz, R. (2017). An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity. Neural Computation, 29(5), 1229-1262.
- Goldstein, A. et al. (2022). Shared computational principles for language processing in humans and deep language models. Nature Neuroscience, 25(3), 369-380.
- Caucheteux, C. & King, J. R. (2022). Brains and algorithms partially converge in natural language processing. Communications Biology, 5, 134.
- Schrimpf, M. et al. (2021). The neural architecture of language: integrative modeling converges on predictive processing. PNAS, 118(45).

## see also

- [[predictive_coding]]
- [[free_energy_principle]]
- [[precision_weighting]]
- [[predictive_coding_to_training_objective]]
- [[plasticity_local_vs_global]]
