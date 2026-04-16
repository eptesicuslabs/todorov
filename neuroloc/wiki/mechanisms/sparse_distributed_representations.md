# sparse distributed representations

status: definitional. last fact-checked 2026-04-16.

**why this matters**: SDRs formalize the tradeoff between dense continuous embeddings (used in transformers) and sparse binary activations (used in spiking networks). the mathematical properties of SDRs -- combinatorial capacity, noise robustness, O(N) set operations, one-shot learning -- define the design space for hybrid architectures like todorov that use ternary spikes within a continuous-valued pipeline.

## what SDRs are

a **sparse distributed representation** (SDR) is a binary vector with a large dimensionality N, where only a small fixed fraction f of bits are active (set to 1) at any time. "sparse" means f << 1. "distributed" means the information is spread across the pattern of active bits, not localized in any single bit. ML analog: SDRs are the binary counterpart to continuous embedding vectors -- where transformers use dense float32 vectors in R^d, SDRs use sparse binary vectors in {0,1}^N with the same goal of distributed representation.

SDRs sit between two extremes:

- **local representations** (grandmother cells): one neuron per concept. N concepts require N neurons. zero noise robustness -- one dead neuron destroys one concept.
- **dense distributed representations**: a large fraction of neurons active per pattern. high noise robustness but limited capacity due to overlap between patterns.

SDRs are the biological sweet spot. with N = 2048 and f = 0.02 (40 active bits), the number of possible distinct SDRs is C(2048, 40) ~ 10^74 -- vastly more patterns than the system could ever store, ensuring minimal interference between stored representations.

## mathematical properties

ahmad and hawkins (2015, 2016) formalized the key properties:

### capacity

for a population of N neurons with sparsity f (fraction active), the number of possible distinct patterns is:

    C(N, f*N) = N! / ((f*N)! * ((1-f)*N)!)

this grows combinatorially. for N=2048 and 40 active bits:
- possible patterns: ~2.37 * 10^74
- far exceeds the number of atoms in the universe (~10^80)

### overlap and matching

two random SDRs with the same N and f share, on average, f^2 * N active bits. for N=2048 and f=0.02, the expected overlap is ~0.8 bits out of 40. this means random SDRs are nearly orthogonal -- they can be stored without interference.

a match between a stored pattern and a query pattern is determined by a threshold theta on the number of overlapping active bits. the probability of a false positive (a random pattern exceeding the threshold) drops exponentially with theta:

    P(false positive) ~ C(f*N, theta) * f^theta

for theta = 20 (half the active bits), this is astronomically small.

### noise robustness

SDRs tolerate substantial corruption:

- **bit flips**: flipping 10% of the bits (turning some 1s to 0s and some 0s to 1s) typically preserves the pattern identity because the overlap with the correct stored pattern still exceeds the overlap with any other pattern.
- **subsampling**: observing only a random subset of the neurons still allows identification of the pattern, because the overlap between the observed active bits and any stored pattern is preserved in expectation.
- the error rate degrades gracefully, not catastrophically, as noise increases.

### set operations

SDRs support efficient approximate set operations:

- **union**: OR of two SDR vectors produces a representation of "A or B." the union is denser than either input but still recognizable.
- **intersection**: AND of two SDR vectors produces the shared components. non-empty intersection implies the two SDRs share structure.
- **membership testing**: checking if a query SDR is a member of a union set is a simple thresholded overlap computation.

these operations are single-pass bitwise computations -- O(N) time, no iteration. this is a fundamental advantage over dense vector operations that require dot products and nonlinearities.

## contrast with dense distributed representations

modern transformer architectures use dense distributed representations: each token is embedded as a continuous vector in R^d (typically d=768 to 4096) where all dimensions carry information simultaneously.

| property | SDR (biological) | dense (transformer) |
|---|---|---|
| dimensionality | high (N ~ 2000-10000) | moderate (d ~ 768-4096) |
| active fraction | low (2-10%) | 100% (all dimensions) |
| values per element | binary {0,1} | continuous float32 |
| bits per element | 1 bit | 32 bits |
| noise robustness | inherent (graceful degradation) | requires explicit training |
| set operations | bitwise (O(N)) | learned via attention (O(N^2)) |
| similarity metric | overlap count | cosine / dot product |
| one-shot learning | yes (single Hebbian write) | no (requires gradient descent) |
| catastrophic forgetting | resistant (low interference) | severe (without replay) |
| energy per operation | ~1 pJ (biological) | ~100 nJ (GPU) |

the central tradeoff: dense representations pack more information per dimension (32 bits vs 1 bit) but require more complex operations to manipulate. SDRs sacrifice per-dimension information density for operational simplicity and noise robustness.

## biological implementation

### cortical columns as SDR generators

in the numenta framework (hawkins and ahmad, 2016), cortical minicolumns implement SDRs:

- each minicolumn contains ~80-100 neurons. a cortical area contains thousands of minicolumns.
- at any given time, one neuron per minicolumn (or none) is active, creating a sparse activation pattern across the area.
- the pattern of which minicolumns are active, and which neuron within each active minicolumn, constitutes the SDR.

### one-shot learning

SDRs enable one-shot learning via hebbian plasticity:

- to store an association between two patterns, strengthen synapses between co-active neurons. one presentation suffices.
- retrieval: present a partial cue, activate neurons via the stored synaptic weights, and threshold. the pattern completes.
- capacity: a single neuron can participate in many stored patterns without interference, because each pattern activates only a sparse subset of the neuron's synapses.

this contrasts with backpropagation-trained networks, which require hundreds or thousands of presentations to learn an association and suffer from catastrophic forgetting when learning new associations.

### pattern completion and generalization

SDRs support pattern completion naturally: a partial input (e.g., half the active bits of a stored pattern) has high overlap with the stored pattern and low overlap with other stored patterns. a simple threshold operation retrieves the full pattern.

generalization arises from shared subpatterns: if two stimuli share structural features, their SDRs share active bits. a query that partially matches both will activate a representation reflecting their common structure.

## SDRs and temporal sequences

the temporal memory component of hierarchical temporal memory (HTM) uses SDRs to represent sequences:

- each element in a sequence is an SDR.
- the transition from one SDR to the next is learned via hebbian plasticity on the distal dendrites of cortical neurons.
- prediction: given the current SDR, the system activates neurons that were predictive of the next element, biasing them to fire first in a competitive WTA process.

this enables the system to learn and predict temporal sequences with a single pass through the data, without backpropagation through time.

## challenges

- SDRs require high dimensionality (N > 1000) to achieve useful capacity, which may limit their applicability in systems with tight parameter budgets.
- the binary constraint (each bit is 0 or 1) discards magnitude information. a neuron that fires at 50 Hz and one that fires at 5 Hz are both "active" in an SDR. this is a lossy abstraction.
- biological SDRs may not be strictly binary -- graded firing rates and continuous-valued synaptic weights introduce analog components that the SDR framework does not capture.
- the comparison with transformer embeddings (see [[sparse_vs_dense_representations]]) is not straightforward: transformers operate on sequences with attention, while SDR systems operate on spatial patterns with hebbian learning. the computational paradigms are sufficiently different that direct numerical comparisons of capacity or robustness may be misleading.

## key references

- ahmad, s. & hawkins, j. (2016). how do neurons operate on sparse distributed representations? a mathematical theory of sparsity, neurons and active dendrites. arXiv:1601.00720.
- ahmad, s. & hawkins, j. (2015). properties of sparse distributed representations and their application to hierarchical temporal memory. arXiv:1503.07469.
- kanerva, p. (1988). sparse distributed memory. MIT press.
- willshaw, d. j., buneman, o. p. & longuet-higgins, h. c. (1969). non-holographic associative memory. nature, 222, 960-962.
- hawkins, j. & ahmad, s. (2016). why neurons have thousands of synapses, a theory of sequence memory in neocortex. frontiers in neural circuits, 10, 23.

## see also

- [[sparse_coding]]
- [[population_coding]]
- [[efficient_coding]]
- [[sparse_vs_dense_representations]]
