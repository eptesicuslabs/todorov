# learning rules research

curated peer-reviewed research on non-backpropagation learning rules and their viability at scale. the central question for the neural machine: can a biologically plausible learning rule replace backpropagation without sacrificing performance? the current answer is no -- not at scale -- but the gap is narrowing, and several methods show promise in restricted domains.

## target propagation

### scaled difference target propagation on imagenet

ernoult, m., guo, x., kording, k., & bengio, y. (2022). towards scaling difference target propagation by learning backprop targets. *proceedings of the 39th international conference on machine learning (ICML)*.

key finding: difference target propagation (dtp) replaces backpropagated error gradients with locally generated target activations that each layer attempts to reach. ernoult et al. scaled dtp to imagenet 32x32 with a resnet-18, achieving top-1 accuracy within 1% of backpropagation. the key insight was learning the backward mapping (used to generate targets) alongside the forward mapping, stabilizing training at depth. this is the closest any target propagation method has come to backprop parity on a non-trivial image benchmark.

relevance to neural machine: dtp is among the most promising backprop alternatives because it uses only local information at each layer -- no global error signal needs to propagate through the full depth. however, the 32x32 resolution and the requirement for learned backward mappings (which themselves need gradients to train) weaken the biological plausibility claim. the method has not been demonstrated on language tasks or at scales above ~25M parameters.

confidence: high for the specific benchmark result. the gap to backprop grows with resolution and depth. caveat: the learned backward mapping introduces a second network that must be trained, partially undermining the locality argument.

## predictive coding

### predictive coding at depth

salvatori, t., song, y., lukasiewicz, t., bogacz, r., & xu, z. (2025). predictive coding beyond backpropagation. *international conference on learning representations (ICLR)*.

key finding: predictive coding networks (pcns) use local prediction error minimization at each layer, iterating to equilibrium before updating weights. salvatori et al. showed that pcns match backpropagation on shallow networks (3-5 layers) across multiple benchmarks but fail systematically at depth 9+. the failure mode is not gradient vanishing (pcns do not propagate gradients) but inference instability: the iterative equilibrium computation becomes increasingly fragile as depth increases, requiring more iterations and smaller step sizes that eventually make training impractical.

relevance to neural machine: todorov's architecture is 12-24 layers deep, well beyond the depth where predictive coding maintains parity. the inference instability problem suggests that any equilibrium-based learning rule faces fundamental scaling challenges at transformer-class depths. however, predictive coding's local error signals are conceptually aligned with the [[predictive_coding]] framework, and hybrid approaches (predictive coding for local updates, a lightweight global signal for long-range credit assignment) remain unexplored.

confidence: high. systematic depth sweep with controlled comparisons. caveat: the specific failure depth may depend on architecture details; recurrent architectures might handle depth differently than feedforward.

## forward-forward methods

### scff on stl-10

hinton, g. e. (original forward-forward, 2022) extended by park, s. et al. (2025). supervised contrastive forward-forward algorithm. *nature communications*.

key finding: the supervised contrastive forward-forward (scff) algorithm replaces backpropagation with a layer-local contrastive objective: each layer learns to increase the goodness (sum of squared activities) for positive examples and decrease it for negative examples. on stl-10 (96x96 natural images, 10 classes), scff matched backpropagation accuracy. the key advance over hinton's original forward-forward was the supervised contrastive framing, which provides stronger gradients than the original binary goodness criterion. training is fully parallelizable across layers since no inter-layer gradient flow is needed.

relevance to neural machine: scff's layer-parallel training is attractive for the neural machine because it eliminates the sequential backward pass. however, scff requires explicit negative examples, which have no clear analog in autoregressive language modeling. the method has been validated only on classification tasks with relatively small networks. the contrastive objective also discards information about relative example similarity, which may limit its applicability to sequence modeling where next-token distributions matter.

confidence: medium-high. single benchmark (stl-10) with backprop parity. caveat: stl-10 is a relatively easy benchmark; the method has not been tested on imagenet-scale or language tasks.

## evolution strategies

### evolution strategies for llm fine-tuning

qiu, h. et al. (2025). evolution strategies for llm fine-tuning at 7b scale.

key finding: evolution strategies (es) -- gradient-free optimization using population-based perturbation and fitness evaluation -- were applied to fine-tune a 7B parameter llm on arithmetic reasoning tasks. es outperformed both ppo and grpo (gradient-based rl methods) on arithmetic accuracy while using comparable compute. the key insight is that es scales to high dimensions when the fitness landscape is smooth and the effective dimensionality is low (as in fine-tuning, where only a small subspace of parameters matters).

relevance to neural machine: es eliminates both backpropagation and gradient computation entirely -- it is the most biologically plausible optimization method tested at llm scale. however, the result is specific to fine-tuning (not pretraining) and to a narrow task domain (arithmetic). es for pretraining from scratch at 7B would require orders of magnitude more compute. the effective-dimensionality argument is important: if the neural machine's parameter space has low intrinsic dimensionality (which ternary weights would encourage), es-like methods might become viable.

confidence: medium. single task domain, fine-tuning only. the compute comparison with gradient methods is approximate. caveat: es for pretraining remains intractable at scale.

## spiking network learning

### e-prop on neuromorphic hardware

bellec, g. et al. (2020). a solution to the learning dilemma for recurrent networks of spiking neurons. *nature communications*, 11, 3625. deployed on spinnaker2 hardware.

key finding: e-prop (eligibility propagation) maintains per-synapse eligibility traces that track the causal influence of each synapse on future outputs. when a global reward or error signal arrives, the eligibility trace converts it to a local weight update. on google speech commands (12-class keyword recognition), e-prop achieved 91% accuracy using only 25K trainable weights on spinnaker2 neuromorphic hardware. this is the most complete demonstration of a biologically plausible learning rule running on dedicated neuromorphic silicon.

relevance to neural machine: e-prop is directly relevant to todorov's spiking neurons. the eligibility trace mechanism could complement the current ste-based gradient flow through ternary spikes: instead of straight-through estimation (which is a mathematical convenience), e-prop provides a principled mechanism for credit assignment through discontinuous activations. the 25K-weight scale limitation is the blocker -- speech commands is a tiny task. scaling e-prop to millions of parameters remains undemonstrated.

confidence: high for the specific result. the spinnaker2 deployment confirms hardware feasibility. caveat: 91% on a 12-class task with 25K weights is far below the performance frontier.

## three-factor hebbian learning

### three-factor rules at toy scale

gerstner, w. et al. (2018 review, updated in patterns review 2025). eligibility traces and three-factor learning.

key finding: three-factor hebbian rules (delta_w = eta * f(pre, post) * M(t), where M is a neuromodulatory signal) are the dominant theoretical framework for biologically plausible learning. the pre-post correlation provides spatial credit assignment (which synapse), and the modulator provides temporal credit assignment (when to update). despite extensive theoretical development, no three-factor rule has been demonstrated above toy scale (mnist-level classification or simple rl tasks). the fundamental bottleneck is temporal credit assignment: the eligibility trace decays exponentially, limiting the temporal horizon over which credit can be assigned.

relevance to neural machine: todorov's [[three_factor_learning]] wiki article maps the three-factor framework to the architecture. the eligibility trace decay constant sets a hard limit on sequence-level credit assignment -- at typical biological time constants (10-100ms), the trace vanishes before meaningful language dependencies can be captured. any practical implementation would need either very long traces (biologically implausible) or a hierarchical credit assignment mechanism where traces at different timescales are maintained at different levels of the network.

confidence: high for the theoretical framework. the lack of scaled results is well-documented across multiple reviews. caveat: the absence of evidence at scale is not evidence of impossibility -- the methods may simply lack engineering investment.

## the scale gap

### summary of non-backprop performance

no single source; synthesis across the papers above and surveys including lillicrap et al. (2020, *nature reviews neuroscience*) and bartunov et al. (2018).

key finding: across all non-backpropagation learning rules tested at any scale, the performance gap to backprop ranges from ~1% (dtp on imagenet 32x32) to ~15% (three-factor hebbian on non-trivial tasks). the gap generally widens with (a) task complexity, (b) network depth, and (c) parameter count. no non-backprop method has been validated above ~25M parameters on standard benchmarks. the most promising methods (dtp, scff) achieve parity only on restricted benchmarks. evolution strategies sidestep backprop entirely but only for fine-tuning, not pretraining.

relevance to neural machine: todorov currently uses backpropagation with ste through ternary spikes. replacing backprop entirely is not viable at the current 300M parameter scale based on published evidence. however, the architecture's ternary quantization and recurrent structure may make it more amenable to local learning rules than standard transformers: (a) ternary weights have only 3 possible values per parameter, reducing the optimization search space, (b) the recurrent state provides a natural substrate for eligibility traces, and (c) the 3:1 hybrid architecture could use different learning rules for different layer types.

confidence: high for the gap estimate. caveat: the field moves quickly and a breakthrough at scale could change the picture.

## see also

- [[three_factor_learning]]
- [[hebbian_learning]]
- [[stdp]]
- [[predictive_coding]]
- [[dopamine_system]]
- [[ternary_spikes]]
- [[homeostatic_plasticity]]

## relevance to the neural machine

### validated connections
- ste through ternary spikes is the current credit assignment mechanism -- it works but is not biologically plausible
- the recurrent state (kda matrix-valued state, mamba3 complex state) provides a natural substrate for eligibility traces
- ternary weight quantization reduces the effective search space, potentially favoring gradient-free methods

### challenged assumptions
- no non-backprop method has been validated at the neural machine's 300M parameter scale on language tasks
- predictive coding fails at the depth (12-24 layers) required by todorov
- three-factor hebbian rules are limited to toy-scale demonstrations
- evolution strategies work only for fine-tuning, not pretraining from scratch

### open questions
- could a hybrid approach use backprop for pretraining and a local rule for continued learning or adaptation?
- does the 3:1 layer architecture enable different learning rules for kda (local delta-rule updates) vs mla (global attention gradients)?
- can eligibility traces be maintained in the recurrent state rather than as separate per-synapse variables?
- what is the minimum viable scale at which non-backprop methods must be competitive to be useful?
