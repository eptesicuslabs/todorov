# sparsity from biology to ternary spikes

## the biological principle

sparsity in the brain is not a design preference. it is a metabolic mandate.

the [[brain_energy_budget]] establishes the constraint: the human cortex consumes ~3-5 W for signaling, each action potential costs ~19-120 pJ (attwell and laughlin 2001, lennie 2003), and each synaptic event costs ~10 fJ (harris et al. 2012). lennie (2003) calculated that fewer than 1% of cortical neurons can be substantially active at any time given this budget. the revised howarth et al. (2012) budget shifts the dominant cost from action potentials (~21%) to synaptic transmission (~50%), but the conclusion is unchanged: the brain cannot afford dense activation.

the [[energy_efficient_coding]] framework formalizes why. levy and baxter (1996) showed that maximizing bits per joule -- not bits per neuron -- yields an optimal firing probability of p ~ 1/(1 + exp(E_spike/E_rest)). for cortical neurons where a spike costs 10-100x more than resting, this gives p_optimal in the range of 1-10%. their specific estimate of ~6% matches observed cortical firing rates with striking precision. niven and laughlin (2008) extended this to full sensory systems, demonstrating that energy limitation is the dominant selective pressure shaping neural coding across species, from fly photoreceptors to mammalian cortex.

the [[efficient_coding]] hypothesis provides the information-theoretic view of the same phenomenon. barlow's (1961) redundancy reduction principle demands factorial codes -- representations where each neuron's response is statistically independent of every other neuron's. the [[sparse_coding]] framework (olshausen and field 1996) showed that imposing L1 sparsity on activations of an overcomplete dictionary is sufficient to learn V1-like gabor receptive fields from natural image statistics alone. no supervised labels, no hand-tuning -- the statistics of the input plus the sparsity constraint produce the biology.

these three perspectives -- metabolic budget, energy-information tradeoff, information-theoretic efficiency -- converge on the same operating point: 2-10% cortical population sparseness, with most neurons silent at any given moment. the [[population_coding]] literature confirms this empirically: V1 at ~5-10% (vinje and gallant 2000), IT cortex at ~5-10% (rolls and tovee 1995), hippocampal place cells at ~1-2% (wilson and mcnaughton 1993), cerebellar granule cells at ~1-5%.

the biological implementation relies on lateral inhibition and k-winners-take-all circuits. as detailed in [[lateral_inhibition_to_adaptive_threshold]], PV+ basket cells provide fast perisomatic inhibition that selects the most active neurons and suppresses the rest, potentially structured by gamma oscillations (~30-80 Hz). foldiak (1990) showed that anti-hebbian lateral connections can learn to decorrelate responses, producing sparse codes through online competitive learning with an adaptive threshold. the mechanism is intrinsically competitive: one neuron's activity directly suppresses its neighbors.

what makes biological sparsity powerful is not the low firing rate per se, but the convergence of multiple mechanisms on the same operating point. metabolic constraints demand it (levy-baxter). information theory rewards it (olshausen-field). and the circuit implementation -- lateral inhibition, divisive normalization, WTA -- actively enforces it through competitive dynamics. the 2-10% range is not a single constraint but an equilibrium where physics, information, and circuit dynamics agree.

## the todorov instantiation

todorov's ternary spikes borrow the surface structure of biological sparse coding while operating in a fundamentally different regime.

the AdaptiveTernarySpike module computes threshold = alpha * mean(|x|), where alpha is a learnable parameter initialized at 1.0. the output is sign(x) * [|x| > threshold], producing values in {-1, 0, +1}. the backward pass uses the straight-through estimator (STE): gradients pass through quantization as if it were the identity function. at alpha=1.0, approximately 41% of dimensions fire (nonzero), stable across 10 runs from 6m to 267m parameters.

as [[sparse_coding_to_ternary_spikes]] documents, this threshold is a population-level statistic -- the mean absolute activation across all dimensions. it implements a crude form of gain control: the threshold adapts to the scale of the input. [[lateral_inhibition_to_adaptive_threshold]] shows this is closer to global subtractive inhibition than to biological divisive normalization: one scalar threshold for the entire layer, no per-neuron pools, no spatial structure, no competitive dynamics between individual neurons. increasing one neuron's activation raises the threshold for all neurons through the mean, but the competition is mediated entirely by a single statistic, losing all information about which specific neurons are competing.

rewriting the threshold as a two-step process clarifies the relationship, as [[lateral_inhibition_to_adaptive_threshold]] demonstrates: (1) compute z_i = |x_i| / mean(|x|), then (2) output_i = sign(x_i) * [z_i > alpha]. step 1 is a degenerate divisive normalization with n=1, sigma=0, uniform pool weights. step 2 is a hard threshold on the normalized value. the combination is population-relative sparse selection -- a meaningful computation -- but it lacks the pool structure, power-law nonlinearity, temporal adaptation, and graded output that make biological divisive normalization computationally powerful. RMSNorm before each sublayer provides genuine divisive normalization (uniform pools, learned per-feature gain), and the sequential combination of RMSNorm + ternary spikes approximates normalization-then-WTA, but without the recurrent simultaneous dynamics of biological circuits.

the spike health metrics bridge [[population_coding]] theory to engineering practice, as mapped in [[population_coding_to_spike_health]]. mutual information (MI = 1.168 at 267m) measures how much pre-spike information survives the ternary quantization -- 74% of the channel capacity of log2(3) = 1.58 bits/dim. centered kernel alignment (CKA = 0.732 at 267m) measures whether the geometry of the representation is preserved through quantization. firing rate (40.8% at 267m) sits within the validated stable range of 20-60%. all three metrics pass their thresholds, and the architecture achieves 0.663x BPB (33.7% better than transformer baseline) at 267m scale.

but as [[population_coding_to_spike_health]] honestly notes, several population coding properties go unmeasured: noise correlations between spike dimensions, cross-dimensional redundancy (total correlation), and fisher information for stimulus discrimination precision. the MI estimate samples only 8 of d dimensions. the system may have pathologies invisible to the current metrics.

the entropy of the achieved firing distribution tells a subtler story. at 267m scale, the distribution (~20.4% positive, ~20.4% negative, ~59.2% zero) gives H ~ 1.50 bits/dim -- 95% of the maximum ternary entropy of 1.58 bits/dim. the ternary alphabet is used efficiently. but [[sparse_vs_dense_representations]] notes that this efficiency metric is orthogonal to sparsity: maximum ternary entropy occurs at P(-1) = P(0) = P(+1) = 1/3, which is 67% firing rate, even denser than the current 41%. the information-theoretic optimum for the ternary alphabet and the sparsity optimum for associative memory capacity pull in opposite directions.

## the sparsity-gradient paradox

the core tension: biology achieves 2-10% cortical sparsity under metabolic constraints. todorov fires at 41% because STE gradient flow requires dense activation. neither regime can access the other without changing the learning rule.

[[sparse_coding_to_ternary_spikes]] explains the mechanism. the STE passes gradients through quantization as identity, but the effective gradient magnitude at each dimension correlates with the pre-spike activation magnitude. at 5% firing rate, 95% of dimensions contribute near-zero gradients. for a 300m-parameter model trained with AdamW, this level of gradient sparsity produces three pathologies:

1. convergence slows by an estimated 5-10x (less gradient information per step)
2. dead neuron cascades emerge (neurons that never fire never receive gradient signal to change their behavior, so they remain silent permanently)
3. gradient variance increases (the few active dimensions carry all the optimization signal, making each step noisier)

the 20-60% firing rate range was established empirically across 10 training runs: below 20%, training becomes unstable; above 60%, the sparsity benefit (information compression, noise filtering) is lost. the 41% at alpha=1.0 is not a designed target -- it is the natural equilibrium of the threshold function given the distribution of pre-spike activations after RMSNorm.

[[sparse_vs_dense_representations]] frames this as dimension 5 of the comparison: gradient flow is the axis along which ternary spikes most fundamentally depart from biological sparse codes. biological learning rules -- hebbian, anti-hebbian, STDP -- operate on local information. the STE passes a global error signal backward through a quantization boundary. no known biological mechanism does this. the firing rate that gradient-based optimization demands (41%) is dictated by the learning rule, not by the information-theoretic properties of the representation.

the theoretical optimal sparsity for associative memory capacity is f ~ 1/sqrt(N). for d=384, this is ~5.1%; for d=1024, ~3.1%. the autoencoder-based hippocampal model (bricken et al. 2025) finds optimal coding level at f ~ 5-7.5% depending on input compressibility. todorov operates at 8-13x these theoretical optima. this is not because 41% is information-theoretically superior -- it is because the optimization algorithm requires it.

the paradox is structural. the biological brain can afford 2-10% sparsity because it uses local learning rules that do not require gradient flow through every neuron simultaneously. silicon architectures using backpropagation cannot reach biological sparsity levels without either changing the learning rule or finding gradient estimators that tolerate extreme sparsity. the STE is not such an estimator -- it degrades gracefully down to ~20% but fails below that.

the capacity-sparsity tradeoff from [[sparse_coding]] reinforces this: for associative memories, storage capacity scales inversely with the firing fraction f, maximized at f ~ 1/sqrt(N). for N=1024 neurons, optimal f ~ 3.1%. the brain operates near this optimum because hebbian learning tolerates sparse gradients. backpropagation through the STE does not. the same mathematical principle that makes biological sparsity optimal makes engineering sparsity infeasible at the same operating point.

## the energy question

the per-operation energy advantage of ternary spikes is real and substantial. [[biological_vs_silicon_energy]] documents it across process nodes: 354x at 45nm, ~250x at 7nm, ~300x at 5nm, 215-285x at 3nm. the ratio remains in the 200-350x range because the fundamental asymmetry persists: a FP32 multiplier requires O(n^2) gates for n-bit operands, while a ternary multiply is an O(1) MUX.

the system-level story is different. [[energy_efficiency_to_ternary_spikes]] provides the honest accounting for 267m params at 5nm. data movement (weight reads from HBM) costs ~2.1 mJ per token. total compute energy is ~40 nJ per token. compute is ~50,000x smaller than data movement. ternary spikes save ~99.9% of compute in spiked paths, but spiked paths cover only 15-25% of total MACs (K and V projections in 18/24 KDA layers). system-level energy savings: ~0.0005%.

[[biological_vs_silicon_energy]] identifies the deeper issue: the brain's energy advantage is no longer at the per-operation level. silicon at 5nm has surpassed biological synaptic efficiency for ternary operations (~0.001 pJ vs ~10 fJ). the brain wins at the system level through four properties: extreme sparsity (1-5% vs 41%), local wiring (most connections < 1 mm), in-memory computation (weights stored at the synapse), and massive parallelism (86 billion neurons). todorov captures one of these four advantages (ternary sparsity, partially) and none of the other three.

the deployment target is GPU-only for now. on GPU hardware, ternary spikes provide activation compression (2 bits vs 16 bits = 8x less bandwidth) and sparse acceleration (skip 59% of operations in spiked paths). neuromorphic hardware is theoretical interest. dedicated ternary accelerators (achieving 63 pJ/MAC system-level vs ~1-5 nJ/MAC for GPU FP16) would realize 10-50x system-level savings, but require non-GPU deployment.

the energy comparison between biology and todorov thus reveals an irony. the biological principle that inspired ternary spikes -- metabolic efficiency through sparsity -- is the one dimension where the analogy is weakest in practice. the brain's 2-10% sparsity saves orders of magnitude in metabolic cost because each spike is biologically expensive. todorov's 41% sparsity saves ~0.0005% in system energy because each MAC is computationally cheap relative to data movement. the constraint that shapes the biology (metabolic cost per spike) is absent in the engineering (negligible compute cost per MAC relative to memory bandwidth). the biological pressure that produced sparse coding does not exist on GPU hardware.

## paths forward

five interventions could narrow the gap between todorov's 41% and the biological 2-10%, or extract more value from the current operating point.

**ATMN per-neuron thresholds.** as described in [[sparse_coding_to_ternary_spikes]], ATMN replaces the global alpha * mean(|x|) threshold with per-neuron learnable thresholds V_th_i = exp(threshold_log_i). this allows different sparsity levels for different features: high-information features fire more often, redundant features fire less. this is closer to biological divisive normalization with local gain control. ATMN is implemented but run_011 found it ~2x slower than basic ternary spikes -- performance optimization is needed before scale validation in phase 5a.

**sparsity schedule.** start training at 41% (alpha=1.0) for gradient flow during early optimization, then anneal alpha upward to target 15-20% by end of training. [[sparse_coding_to_ternary_spikes]] notes the risk: representations calibrated to 41% sparsity may catastrophically lose information when sparsity increases mid-training. MI and CKA monitoring during the transition is mandatory.

**auxiliary sparsity loss.** add L_sparsity = beta * (firing_rate - f_target)^2 to the training loss, decoupling the sparsity level from the threshold mechanism. the optimizer can find thresholds achieving the target rate while maintaining gradient flow. risk: the auxiliary loss may conflict with the primary language modeling objective.

**non-STE gradient methods.** gumbel-softmax relaxation, REINFORCE, or evolutionary strategies could replace the STE. these methods can tolerate higher sparsity because they do not require the gradient to flow through every active dimension. gumbel-softmax replaces hard ternary selection with a differentiable approximation whose temperature parameter controls the sparsity-gradient tradeoff -- high temperature gives soft selection with good gradients, low temperature gives hard ternary selection approaching the desired discrete output. REINFORCE estimates gradients via sampling, treating the ternary selection as a stochastic policy, which has high variance but does not require gradients to flow through silent neurons. this is the most promising path to biological sparsity levels but introduces substantial training complexity (variance reduction, temperature schedules) and has not been validated in the todorov context.

**k-WTA competitive selection.** [[lateral_inhibition_to_adaptive_threshold]] proposes replacing the threshold-based spike with explicit k-winners-take-all: select the top-k activations by absolute value, suppress the rest. this is the closest analog to biological lateral inhibition via basket cell perisomatic inhibition. the firing rate becomes exactly controllable (f = k/d) and the selection is competitive (relative ranking) rather than absolute (above/below threshold). but fixed k removes the network's ability to modulate sparsity per token, and topk introduces a new non-differentiable boundary.

these five paths are not mutually exclusive. a plausible combined intervention: start with k-WTA at k/d=0.40, anneal k downward to k/d=0.15 over training, using gumbel-softmax relaxation for the selection to maintain gradient flow at low sparsity. this combines competitive dynamics (k-WTA), scheduled sparsity reduction, and a non-STE gradient method. the complexity cost is substantial, and each component introduces hyperparameters that interact in unknown ways. none of these paths has been validated at scale.

## the evidence standard

the user's evidence bar for a "genuine" analogy (not merely inspirational borrowing) requires three simultaneous conditions: mathematical identity, ablation necessity, and quantitative prediction. all three required.

**mathematical identity.** is the todorov ternary spike threshold mathematically equivalent to a biological sparse coding mechanism? [[lateral_inhibition_to_adaptive_threshold]] shows the answer is nuanced. the threshold involves a normalization-like division (each activation implicitly compared to the population mean), which is a degenerate case of divisive normalization with n=1, sigma=0, uniform pool weights. this is a mathematical restriction, not an identity. the ternary spike is to divisive normalization what batch normalization is to local gain control -- same family, vastly simplified.

**ablation necessity.** does removing ternary spikes degrade performance? the architecture achieves 0.663x BPB with spikes active. the ablation has not been performed at 267m scale (removing spikes from K/V projections and comparing BPB). at 6m scale, spike MI reached 1.311 with GP active, suggesting the spikes carry substantial information. but necessity has not been formally established: the model might achieve similar BPB with dense projections and different capacity allocation.

**quantitative prediction.** does the biological theory predict todorov's operating point? the levy-baxter optimal firing rate of ~6% does not predict todorov's 41%. the associative memory optimal sparsity of f ~ 1/sqrt(d) does not predict 41%. no biological theory predicts the 20-60% stable range -- this is an empirical finding of the STE gradient dynamics. the biological theory makes quantitative predictions that the implementation violates by an order of magnitude.

by this evidence standard, ternary spikes are not a "genuine" implementation of biological sparse coding. they are a quantization mechanism whose design was inspired by biological thresholding but whose operating point is determined by gradient flow engineering.

a weaker but more defensible claim: ternary spikes are a genuine implementation of population-relative thresholding -- the principle that a neuron's firing decision depends on the population activity, not just its own input. this claim satisfies mathematical identity (the threshold is a function of population statistics), ablation necessity (removing the population-dependent threshold and using a fixed threshold degrades spike health), and quantitative prediction (alpha=1.0 predicts ~41% firing rate given the distribution of pre-spike activations, and this prediction holds across scales). the claim is about thresholding, not about sparsity. this is a narrower but honest claim.

## challenges and counter-arguments

**the analogy inversion.** [[sparse_vs_dense_representations]] argues that 41% active dimensions is closer to a lossy compression scheme than to a sparse code. biological sparse coding derives its power from the fact that the active set is small and therefore informative: knowing which 5% of neurons fire tells you a lot; knowing which 41% fire tells you comparatively little. at 41%, the inactive set (59%) is barely smaller than the active set. the representation is compressed, not sparse in the biological sense. ternary spikes are 1.58-bit quantization, not sparse coding.

**the missing competitive dynamics.** [[lateral_inhibition_to_adaptive_threshold]] documents what todorov lacks: no lateral inhibition between neurons, no structured normalization pools, no WTA selection. the ternary spike threshold depends on the population mean, not on the relative ranking of activations. RMSNorm before each sublayer provides global divisive normalization, and the combination of RMSNorm + ternary spikes is loosely analogous to normalization followed by WTA. but the biological mechanisms are recurrent and simultaneous, while the engineering instantiation is feedforward and sequential. the dynamics that make biological sparse coding adaptive -- contrast gain control, attention modulation, cross-feature suppression -- are absent.

**the temporal dimension.** biological sparse codes operate over spike trains. a cortical neuron's activity is temporally sparse: long silent periods punctuated by brief bursts. the identity of when a neuron fires matters as much as whether it fires (theta phase precession, temporal coding as documented in [[population_coding]]). todorov's ternary spikes are instantaneous -- each input produces one ternary vector with no temporal dynamics in the basic mode. ATMN adds membrane potential integration, but the training-time reset eliminates the temporal accumulation that would make it a true spiking neuron model. the temporal dimension of biological sparsity is entirely absent.

**the STE has no biological analog.** the straight-through estimator passes a global error signal backward through a quantization boundary. biological learning rules -- hebbian, STDP, anti-hebbian -- are local: each synapse adjusts based on pre- and post-synaptic activity, with no requirement for backward gradient flow through the spiking nonlinearity. [[sparse_vs_dense_representations]] notes this as the most fundamental departure: the learning rule that determines the operating point has no biological counterpart, making the operating point (41%) itself a non-biological artifact. the architecture borrows the biology's forward pass while discarding the biology's learning rule.

**energy claims require honest framing.** [[energy_efficiency_to_ternary_spikes]] and [[biological_vs_silicon_energy]] converge on the same conclusion: the per-operation advantage (200-350x) is real but system-level savings on GPU hardware are ~0.0005%. stating "ternary spikes are 354x more energy-efficient" without the system-level qualification is misleading. the brain's energy advantage comes from system architecture (in-memory compute, local wiring, extreme sparsity), not from the per-synapse cost -- and silicon has already surpassed the per-synapse cost at 5nm. [[biological_vs_silicon_energy]] shows that ternary MAC energy at 5nm (~0.001 pJ) is 10x cheaper than a biological synapse (~10 fJ). the biological inspiration for energy efficiency is historically important but technologically obsolete at the per-operation level.

**the overcomplete dictionary is absent.** [[sparse_coding]] emphasizes that biological sparse coding relies on overcomplete representations: more basis functions than input dimensions, enabling sparser codes per stimulus because any given input can be expressed with fewer active elements from a richer dictionary. olshausen and field used dictionaries of 192+ basis functions for 144-dimensional image patches -- a 1.3x overcomplete ratio. the mushroom body in insects implements a much more dramatic overcomplete-then-sparsify architecture: ~800 projection neurons fan out to ~50,000 kenyon cells, which fire at ~5-10% per odor (turner et al. 2008). todorov's ternary spikes operate on a d-dimensional vector and produce a d-dimensional ternary vector -- same dimensionality, not overcomplete. the overcompleteness that enables biological 2-10% sparsity has no counterpart in the architecture. adding overcompleteness (projecting to a higher-dimensional space before spiking) could enable lower firing rates but would increase parameter count and compute cost.

**the noise robustness argument is weaker than it appears.** [[sparse_vs_dense_representations]] notes that ternary quantization provides some noise robustness: small perturbations to pre-spike activations do not change the ternary output unless they cross the threshold. but this robustness is per-dimension and independent, lacking the population-level combinatorial guarantees that sparse distributed representations (SDRs) provide. SDRs degrade gracefully with bit flips because the overlap metric is robust to small corruptions in high-dimensional space. ternary spikes at 41% activity lack this combinatorial protection -- the active set is too large to serve as a discriminative signature.

## the honest assessment

ternary spikes are quantization masquerading as sparse coding. the threshold alpha * mean(|x|) draws inspiration from population-dependent gain control, and the ternary output {-1, 0, +1} echoes the all-or-nothing character of action potentials. but four axes of divergence separate the implementation from the biology:

1. operating point: 41% vs 2-10% (order of magnitude mismatch)
2. learning rule: STE global backpropagation vs hebbian/STDP local plasticity
3. temporal structure: instantaneous ternary vector vs temporal spike trains with phase coding
4. competitive dynamics: global mean threshold vs lateral inhibition with per-neuron pool structure

this might be fine. the architecture achieves 0.663x BPB at 267m scale -- 33.7% better than the transformer baseline. spike MI at 1.168 and CKA at 0.732 demonstrate that the ternary quantization preserves most of the representational structure despite compressing from ~32 bits/dim to ~1.58 bits/dim. the spike health metrics, grounded in [[population_coding]] theory, show a well-functioning population code at the engineering level even if it operates far outside the biological regime.

the productive framing is not "todorov implements biological sparse coding" but rather "todorov demonstrates that ternary quantization with adaptive thresholding -- a mechanism inspired by, but not equivalent to, cortical sparse coding -- can improve language model performance at matched scale." the biological inspiration was valuable as a design heuristic that led to a non-obvious architectural choice: quantizing K and V projections to ternary values, which no standard transformer playbook would suggest. the mechanism works for engineering reasons (information compression, noise filtering, implicit regularization through reduced representational capacity) that overlap with but are not identical to the biological reasons (metabolic efficiency, capacity optimization, interference minimization through combinatorial sparsity).

the distinction between "inspired by" and "equivalent to" is not a weakness to be papered over. many successful engineering designs borrow principles from biology at a high level while operating in fundamentally different regimes: airplane wings are not bird wings, but the principle of lift over a curved surface generalizes. ternary spikes borrow the principle that population-dependent thresholding can produce useful quantized codes. the principle generalizes even when the operating point does not.

the open question is whether the biological regime (2-10% sparsity) can be reached without abandoning backpropagation. if non-STE gradient methods (gumbel-softmax, REINFORCE) or k-WTA competitive selection can maintain BPB while lowering firing rate to 10-15%, that would be evidence that sparser representations are genuinely beneficial -- not just theoretically elegant. if BPB degrades monotonically as firing rate decreases below 30%, that would be evidence that the STE gradient constraint is binding and that 41% is near-optimal for this learning rule.

either outcome is scientifically valuable. the first would validate the biological principle at the engineering level: sparse representations genuinely improve language modeling, and the 41% operating point is a limitation of STE, not an optimum. the second would sharpen the understanding that ternary spikes are a quantization technique whose optimal operating point is determined by the learning rule, not by the information-theoretic properties that make biological sparsity powerful. the boundary between these hypotheses is empirical, not theoretical. the phase 5a experiment (ATMN vs ternary at matched architecture and training budget) is the next step toward distinguishing them. if ATMN's per-neuron thresholds converge to a distribution with mean firing rate significantly below 41% without BPB regression, that is evidence for the first hypothesis. if they converge to approximately the same 41%, that is evidence for the second.

## open questions

1. does ATMN's per-neuron threshold distribution converge to heterogeneous sparsity across features, or does the optimizer push all thresholds toward the same value? the answer determines whether per-neuron adaptation adds information or just parameters.

2. what is the lowest firing rate achievable with gumbel-softmax relaxation before BPB degrades? this establishes the gradient-sparsity frontier for ternary quantization.

3. does k-WTA selection at matched firing rate (41%) outperform threshold-based selection? if so, the benefit of competitive dynamics is separable from the benefit of sparsity level.

4. would an overcomplete architecture (project d-dimensional vectors to 2d or 4d before spiking, then project back) enable lower firing rates? the compute cost doubles or quadruples, but the sparsity savings could compensate if firing rate drops below 10%.

5. what fraction of the 0.663x BPB advantage is attributable to ternary spikes specifically, vs the recurrent state, GP self-interaction, or other architectural features? the ablation has not been performed at 267m scale.

## see also

- [[sparse_coding]]
- [[efficient_coding]]
- [[energy_efficient_coding]]
- [[brain_energy_budget]]
- [[population_coding]]
- [[sparse_coding_to_ternary_spikes]]
- [[energy_efficiency_to_ternary_spikes]]
- [[population_coding_to_spike_health]]
- [[lateral_inhibition_to_adaptive_threshold]]
- [[sparse_vs_dense_representations]]
- [[biological_vs_silicon_energy]]
