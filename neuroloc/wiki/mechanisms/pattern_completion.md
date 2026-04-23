# pattern completion

status: definitional. last fact-checked 2026-04-16.

**why this matters**: transformer attention IS the update rule of a modern Hopfield network -- understanding pattern completion reveals that every attention head performs one step of associative memory retrieval, connecting classical neuroscience directly to the mechanics of transformers.

## summary

**pattern completion** (the ability of a neural network to retrieve a complete stored pattern from a partial or noisy cue) is the defining computational property of **auto-associative networks** (networks that store patterns and retrieve them from partial cues by converging to stored attractors). the most famous example is the **Hopfield network** (Hopfield 1982). the mechanism: patterns are stored as **attractors** (stable states toward which network dynamics converge) in the network's **energy landscape** (a function over network states whose local minima correspond to stored patterns). presenting a partial cue places the network's state near the **basin of attraction** (the set of initial states that converge to a given attractor) of the stored pattern, and the network's dynamics drive the state toward the attractor, completing the missing information. pattern completion is the computational basis of content-addressable memory in the brain, primarily implemented in hippocampal area CA3 (see [[hippocampal_memory]]).

## Hopfield networks (1982)

### the model

Hopfield (1982) defined a network of N **binary neurons** (neurons with only two possible states, {-1, +1}) with states x_i in {-1, +1} and **symmetric connection weights** W_ij = W_ji (with W_ii = 0). the state of the network is a vector x = (x_1, ..., x_N).

to store P patterns {xi^1, xi^2, ..., xi^P}, the weights are set by the **Hebbian outer product rule** (setting weights proportional to the sum of outer products of stored patterns):

    W_ij = (1/N) * sum_{p=1}^{P} xi_i^p * xi_j^p

in matrix form:

    W = (1/N) * sum_{p=1}^{P} xi^p * (xi^p)^T

this is the same outer product prescription as [[hebbian_learning]] and as the k * v^T association in KDA (see [[plasticity_to_matrix_memory_delta_rule]]).

### the energy function

the network has an energy function (Lyapunov function):

    E = -0.5 * sum_{i,j} W_ij * x_i * x_j = -0.5 * x^T * W * x

this energy function is borrowed from the Ising model in statistical physics. the stored patterns correspond to local minima of E. the network dynamics (asynchronous update: x_i -> sign(sum_j W_ij * x_j)) are guaranteed to decrease or maintain E at each step. therefore, the network converges to a local minimum -- ideally, the stored pattern nearest to the initial state.

the energy landscape metaphor: each stored pattern sits at the bottom of a "valley" (basin of attraction). presenting a corrupted cue places the network's state on the "slope" of the valley. the dynamics roll the state downhill to the bottom, completing the pattern.

### storage capacity

the storage capacity is the maximum number of patterns P that can be reliably stored and retrieved.

**Hopfield's original estimate**: P ~ 0.15 * N (for N neurons). above this threshold, spurious attractors (local minima that do not correspond to any stored pattern) proliferate and retrieval accuracy degrades.

**McEliece, Posner, Rodemich and Venkatesh (1987)**: rigorous analysis showed that for exact retrieval of random binary patterns, the capacity is:

    P_max = N / (4 * log(N))

this is sublinear in N -- the fraction of capacity used (P/N) shrinks as the network grows. for approximate retrieval (tolerating a small fraction of bit errors), the capacity doubles to N / (2 * log(N)).

**the 0.138N rule**: often cited as the practical capacity for binary patterns with random independent bits. this corresponds to the number of patterns for which the probability of retrieval error is below a specified threshold. at this loading, ~1% of bits are incorrect after convergence.

### limitations of classical Hopfield networks

- **low capacity**: 0.138N patterns for N neurons is restrictive. a network of 1000 neurons stores ~138 patterns, each of 1000 bits. the total stored information is ~138,000 bits, compared to the theoretical maximum of N^2/2 = 500,000 weight bits
- **spurious attractors**: above capacity, the energy landscape develops local minima that are mixtures of stored patterns (spin glass states). the network may converge to these instead of a true stored pattern
- **no temporal dynamics**: the Hopfield network stores static patterns, not sequences. extending to sequences requires asymmetric weights or more complex architectures (e.g., Kleinfeld 1986, Sompolinsky & Kanter 1986)
- **binary neurons only**: the original model uses {-1, +1} neurons. extending to continuous values requires a different energy function

## modern continuous Hopfield networks (Ramsauer et al. 2021)

### the breakthrough: exponential capacity

Ramsauer et al. (2021), building on Demircigil et al. (2017) and Krotov and Hopfield (2016), introduced a **modern continuous Hopfield network** (a Hopfield network with continuous-valued neurons and an exponential energy function, achieving exponentially higher storage capacity) with a fundamentally different energy function:

    E = -lse(beta, X^T * xi) + 0.5 * xi^T * xi + beta^{-1} * log(N) + 0.5 * M^2

where lse is the log-sum-exponential function, X = [xi^1, ..., xi^N] is the matrix of stored patterns (columns), xi is the current state (continuous-valued), beta is an inverse temperature parameter, and M bounds the pattern norms.

the critical difference from Hopfield (1982): the log-sum-exponential replaces the quadratic energy. this produces an energy landscape with exponentially separated wells -- patterns interfere much less with each other.

### the update rule is attention

the update rule derived from this energy function is:

    xi^{new} = X * softmax(beta * X^T * xi)

this is precisely the softmax attention mechanism used in transformers. mapping the notation:

- stored patterns X = keys K and values V (when K = V, this is auto-association)
- current state xi = query Q
- the softmax(beta * K^T * Q) computes attention weights
- the output K * softmax(beta * K^T * Q) is the attention output

ML analog: **transformer attention IS the update rule of a modern Hopfield network**. each attention head performs one step of pattern completion in a continuous Hopfield network. this means every softmax attention layer is performing associative memory retrieval.

### storage capacity

the exponential energy separation gives the modern Hopfield network a storage capacity of:

    C ~ 2^{d/2}

where d is the pattern dimension. for d = 128 (typical transformer head dimension), this gives C ~ 2^64 ~ 10^19 patterns. this is astronomically larger than the classical 0.138 * N.

### three types of fixed points

the inverse temperature beta controls which fixed points exist:

1. **high beta (low temperature)**: each stored pattern has its own basin of attraction. retrieval is precise: the network converges to the single nearest stored pattern. this is the "memory retrieval" regime
2. **intermediate beta**: metastable states emerge -- fixed points that are averages over a small number of similar stored patterns. this is the "prototype extraction" regime
3. **low beta (high temperature)**: a single global fixed point that averages over ALL stored patterns. this is the "mean field" regime

Ramsauer et al. observed that transformer heads behave differently across layers: early layers tend toward global averaging (low effective beta), while later layers perform more selective pattern retrieval (high effective beta).

### convergence

the modern Hopfield network is guaranteed to converge to a local minimum of the energy function. moreover, for well-separated patterns, convergence occurs in a single update step -- the softmax attention computes the answer in one shot. this is why transformers do not need iterative settling: one attention operation is sufficient for pattern retrieval.

## pattern completion in biology

### hippocampal CA3

the hippocampal CA3 region is the brain's primary auto-associative network (see [[hippocampal_memory]]). its ~250,000 pyramidal neurons are massively recurrently connected (~12,000-25,000 recurrent collaterals per cell). memories are stored via [[hebbian_learning]] at CA3-CA3 synapses, and partial cues from the entorhinal cortex (via the perforant path) initiate pattern completion.

biological pattern completion differs from Hopfield networks in important ways:
- neurons are not binary -- they fire at continuous rates or with temporal spike codes
- the update is not synchronous -- neurons update asynchronously and continuously
- weights are not symmetric -- biological synapses are directed (W_ij != W_ji in general)
- the network has sparse activity (~2-5% of neurons active) rather than the dense activity (50%) assumed in classical Hopfield analysis

despite these differences, the core principle holds: recurrent excitation among co-active neurons creates attractors, and partial cues are completed via convergent dynamics.

### prefrontal cortex

recurrent circuits in prefrontal cortex also perform pattern completion, supporting working memory. a partially cued task context (e.g., seeing a familiar environment) activates the full task representation via recurrent excitation. this is sustained activity (persistent firing during delay periods) maintained by attractor dynamics in prefrontal networks.

## relationship to todorov

### KDA as linear associative memory

KDA's state update S_t = diag(alpha) * S_{t-1} + beta_t * k_t * v_t^T and readout o_t = q_t^T * S_t implement a linear heteroassociative memory:
- **heteroassociative** (not auto-associative): the query q retrieves a value v associated with a key k, rather than completing the query itself
- **linear**: the readout is a matrix-vector product with no nonlinearity. there are no attractor dynamics. the retrieved pattern is not iteratively refined
- **capacity**: limited by the rank of S_t (head_dim x head_dim matrix, maximum rank = head_dim = 128). this is fundamentally different from Hopfield capacity (which scales with number of neurons, not matrix rank)

KDA is closer to classical Hopfield networks (linear capacity) than to modern Hopfield networks (exponential capacity). it sacrifices the exponential capacity of softmax attention for O(1) per-step computation and O(1) memory.

### MLA as modern Hopfield network

MLA performs standard softmax attention: softmax(Q * K^T / sqrt(d)) * V. by the Ramsauer et al. result, this IS the update rule of a modern Hopfield network. MLA implements pattern completion with exponential storage capacity, exact retrieval (one-step convergence), and content-addressable access.

this creates an ironic reversal: if softmax attention IS a modern Hopfield network, then MLA (not KDA) is the hippocampal analog -- the system with fast, precise pattern retrieval. KDA, with its linear readout and limited capacity, is more like a classical associative memory with Hebbian storage and interference-limited recall. see [[matrix_memory_vs_hippocampus]] for full analysis.

## challenges

### the binding problem

pattern completion in the brain must solve the binding problem: how are the features of a memory (color, shape, location, time) bound together into a coherent representation? Hopfield networks store patterns as monolithic vectors, with binding implicit in the co-activation pattern. but biological pattern completion must bind features that are distributed across different brain areas. the hippocampal indexing theory (see [[hippocampal_memory]]) addresses this by proposing that the hippocampus stores an index that reinstates the distributed cortical pattern.

### capacity vs fidelity tradeoff

in both classical and modern Hopfield networks, there is a tradeoff between the number of stored patterns and the fidelity of retrieval. as the number of patterns approaches capacity, retrieval becomes noisier. biological systems may manage this tradeoff dynamically: the hippocampus encodes new patterns continuously, relies on DG pattern separation to minimize interference, and consolidates important patterns to cortex before capacity limits bite.

### the role of inhibition

Hopfield's original model did not include inhibition. biological pattern completion requires inhibition to prevent runaway excitation and to sharpen the attractor dynamics (see [[excitatory_inhibitory_balance]]). the interplay between recurrent excitation (pattern completion) and inhibitory feedback (pattern sharpening) determines the dynamics of biological memory retrieval.

## key references

- Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. PNAS, 79(8), 2554-2558.
- McEliece, R. J., Posner, E. C., Rodemich, E. R. & Venkatesh, S. S. (1987). The capacity of the Hopfield associative memory. IEEE Transactions on Information Theory, 33(4), 461-482.
- Ramsauer, H. et al. (2021). Hopfield networks is all you need. ICLR 2021.
- Demircigil, M. et al. (2017). On a model of associative memory with huge storage capacity. Journal of Statistical Physics, 168(2), 288-299.
- Krotov, D. & Hopfield, J. J. (2016). Dense associative memory for pattern recognition. NeurIPS 2016.
- Amit, D. J. (1989). Modeling Brain Function: The World of Attractor Neural Networks. Cambridge University Press.
- Rolls, E. T. (2013). The mechanisms for pattern completion and pattern separation in the hippocampus. Frontiers in Systems Neuroscience, 7, 74.

## see also

- [[hippocampal_memory]]
- [[hebbian_learning]]
- [[complementary_learning_systems]]
- [[memory_consolidation]]
- [[plasticity_to_matrix_memory_delta_rule]]
- [[matrix_memory_vs_hippocampus]]
