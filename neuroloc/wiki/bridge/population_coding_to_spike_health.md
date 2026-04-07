# bridge: population coding to spike health metrics

## the biological mechanism

[[population_coding]] theory establishes that information in neural systems is encoded in the joint activity of populations of neurons, not in individual cells. the quality of a population code is characterized by:

- **information content**: how much stimulus information the population carries, measured by mutual information or fisher information
- **representation geometry**: how the structure of the input space is preserved in the population response space
- **firing statistics**: the distribution of firing rates across the population, including dead neurons (never fire) and saturated neurons (always fire)
- **noise correlations**: the structure of trial-to-trial variability, which determines how information scales with population size

these properties are not independent. a population with many dead neurons has low information content. a population with distorted representation geometry loses structural relationships between stimuli. a population with pathological noise correlations may have less information than expected from its size.

## the current todorov implementation

the SpikingBrainValidator (src/spikes/spiking_brain.py) implements three metrics that correspond to the population coding properties above.

### metric 1: mutual information (MI)

**biological concept**: mutual information I(S;R) between stimulus S and population response R measures how much information the neural code preserves about the input. in the [[efficient_coding]] framework, the goal is to maximize I(S;R) subject to metabolic constraints.

**implementation**: the MutualInformationEstimator computes MI between spike outputs (the ternary quantized activations) and pre-spike activations (the continuous values before quantization). it uses binning-based estimation:

- the pre-spike activations are binned into 32 bins (continuous variable discretization)
- the spike outputs are discretized into 3 values: {-1, 0, +1} (natural ternary discretization)
- a 3 x 32 joint histogram is computed
- MI is calculated as: I = sum_{s,r} P(s,r) * log2(P(s,r) / (P(s) * P(r)))
- MI is averaged over 8 randomly selected dimensions

**what it measures**: how much of the pre-spike information survives the ternary quantization. an MI of 0 means the spikes are independent of the input (the quantization destroys all information). the theoretical maximum depends on the input distribution and the ternary discretization.

**thresholds**: MI > 0.1 required for pass. achieved values: 1.168 at 267m scale, 1.311 at 6m scale with GP.

**connection to population coding theory**: this is a direct measurement of the rate-distortion tradeoff. the ternary spike quantization is a lossy channel with capacity log2(3) = 1.58 bits/dim. MI measures how close the actual information transmission is to this channel capacity. MI of 1.168 at 267m scale means the spikes are transmitting ~1.168 bits/dim out of a maximum of 1.58 bits/dim -- 74% channel utilization.

**limitations of the current implementation**:
- only 8 dimensions are sampled, not the full representation. high-dimensional MI estimation is intractable, so this is necessary, but it may miss systematic pathologies in unsampled dimensions.
- MI is computed per-dimension and averaged, which ignores multi-dimensional structure. two dimensions could each have high individual MI but carry redundant information (violating the factorial code objective of [[efficient_coding]]).
- the binning estimator has known biases: it underestimates MI for continuous variables with small sample sizes and overestimates for variables with many zero bins.

### metric 2: centered kernel alignment (CKA)

**biological concept**: CKA measures the similarity of representation geometry between two sets of neural responses. in population coding, the geometry of the population response space -- which stimuli are represented as similar and which as different -- determines the discriminability of the code.

two populations with the same CKA to a reference produce representations with the same similarity structure: stimuli that are nearby in one representation are nearby in the other. this is crucial for downstream readout: if the spike quantization distorts the geometry, a linear decoder trained on pre-spike representations will fail on post-spike representations.

**implementation**: the RepresentationAnalyzer computes linear CKA between spike outputs and pre-spike activations:

- both representations are centered (mean-subtracted)
- CKA = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
- where ||.||_F is the frobenius norm

this is equivalent to the correlation between the kernel matrices K_X = XX^T and K_Y = YY^T, measuring whether the pairwise similarity structure is preserved.

**what it measures**: whether the ternary quantization preserves the geometry of the pre-spike representation. CKA of 1.0 means perfect geometric preservation. CKA of 0.0 means the geometries are unrelated.

**thresholds**: CKA > 0.3 required for pass. achieved: 0.732 at 267m scale.

**connection to population coding theory**: CKA is related to the signal-to-noise ratio of the population code. if the quantization (noise) is small relative to the signal variation, the geometry is preserved (high CKA). if quantization dominates, the geometry is destroyed (low CKA).

high CKA (0.732) means the ternary spikes preserve most of the representational structure despite the massive information compression (from ~32 bits/dim to ~1.58 bits/dim). this is evidence that the STE gradient successfully trains the model to place important information in the dimensions and magnitudes that survive quantization.

**limitations**:
- linear CKA assumes linear similarity structure. if the relevant geometry is nonlinear (e.g., if stimuli lie on a curved manifold in representation space), linear CKA may underestimate the true geometric similarity.
- CKA is invariant to orthogonal transformations and isotropic scaling but not to non-uniform scaling. if the spike quantization selectively scales some dimensions more than others, CKA may not capture this distortion.

### metric 3: firing rate health

**biological concept**: in a healthy population code, the distribution of firing rates across the population should be well-behaved:

- no dead neurons: neurons that never fire carry no information and waste resources. in cortex, dead neurons are pruned during development (synaptic pruning) or reactivated by homeostatic plasticity.
- no saturated neurons: neurons that fire for every stimulus carry no discriminative information. they contribute a constant offset, not a signal.
- moderate mean firing rate: too low = insufficient gradient flow (in the engineering context) or too few bits transmitted (in the biological context). too high = excessive metabolic cost and reduced sparsity benefit.

**implementation**: the SpikingBrainValidator computes per-neuron firing rates (fraction of inputs for which the neuron fires nonzero) and checks:

- dead neuron fraction: neurons with firing rate < 5%. alert if > 5% of neurons are dead.
- saturated neuron fraction: neurons with firing rate > 95%. alert if > 10% of neurons are saturated.
- mean firing rate: alert if outside [20%, 60%].

**connection to population coding theory**: the firing rate distribution determines the code's information capacity. for a ternary code, the maximum entropy per dimension is achieved when P(-1) = P(0) = P(+1) = 1/3. the achieved distribution at 267m scale (40.8% nonzero, ~20.4% positive, ~20.4% negative, ~59.2% zero) gives entropy of:

    H = -0.592*log2(0.592) - 0.204*log2(0.204) - 0.204*log2(0.204) ~ 1.50 bits/dim

this is 95% of the maximum ternary entropy (1.58 bits/dim), indicating efficient use of the ternary alphabet.

the dead/saturated neuron checks correspond to the biological concept of neural resource utilization: every neuron that is dead or saturated is a wasted parameter. in a 300M-parameter model, even 5% dead neurons represent 15M wasted parameters.

## the missing metrics

several aspects of population coding theory are not currently measured:

### noise correlations

the current system does not measure correlations between spike outputs across different dimensions. in biological [[population_coding]], noise correlations can be information-limiting (moreno-bote et al., 2014). if ternary spike outputs across dimensions develop correlated noise patterns (e.g., due to the shared global threshold), this could limit the information content below what the per-dimension MI suggests.

**potential implementation**: compute the correlation matrix of spike outputs across a batch, compare its structure to the correlation matrix of pre-spike activations. large differences indicate the quantization introduces structured noise correlations.

### fisher information

MI measures total information content. fisher information measures the precision of stimulus encoding at a specific stimulus value. for a language model, this would correspond to the precision with which the spike code distinguishes between nearby tokens or contexts.

**why it's hard**: fisher information requires a defined stimulus space and differentiable tuning curves, neither of which is natural for a language model's internal representations.

### temporal stability

the current metrics are computed per-batch with no temporal tracking (beyond the running averages). biological population codes must be stable over time -- the representation of a given stimulus should be consistent across repeated presentations. tracking MI and CKA across training provides a developmental trajectory but not a stability measure.

### population-level redundancy

efficient coding demands low redundancy between neurons. the current MI measure is per-dimension, which does not capture cross-dimensional redundancy. a population with high per-dimension MI but high inter-dimension redundancy is inefficient.

**potential implementation**: compute multi-information (total correlation) across spike dimensions. high total correlation indicates redundancy that the code could eliminate.

## summary of correspondences

| population coding concept | spike health metric | threshold | achieved (267m) |
|---|---|---|---|
| information content | mutual information (MI) | > 0.1 | 1.168 |
| representation geometry | CKA | > 0.3 | 0.732 |
| population health (no dead) | dead neuron fraction | < 5% | pass |
| population health (no saturated) | saturated neuron fraction | < 10% | pass |
| firing rate | mean firing rate | 20-60% | 40.8% |
| noise correlations | not measured | -- | -- |
| inter-neuron redundancy | not measured | -- | -- |
| temporal stability | not measured | -- | -- |

## see also

- [[population_coding]]
- [[efficient_coding]]
- [[sparse_coding_to_ternary_spikes]]
- [[sparse_vs_dense_representations]]
