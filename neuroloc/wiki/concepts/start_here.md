# start here

status: definitional. last fact-checked 2026-04-16.

## what is neuroloc

neuroloc is a computational neuroscience wiki built to support the design of brain-inspired neural architectures. it contains 61 mechanism articles, 19 bridge documents connecting biology to code, 13 comparison analyses, and 33 entity notes. every article is written to be useful to engineers, not just neuroscientists.

## who this is for

you are an ML engineer. you know transformers, attention, gradient descent, and linear algebra. you know nothing about neuroscience. you want to understand which biological principles can make neural architectures better -- and which are irrelevant distractions.

## what you will learn

- how neurons compute: spikes, thresholds, membrane potentials, and why they matter for architecture design
- how synapses learn: Hebbian learning, spike-timing-dependent plasticity, and their relationship to the delta rule and linear attention
- how the brain encodes information: sparse coding, population coding, and the tradeoffs between sparsity and capacity
- how the brain predicts: predictive coding, precision weighting, and how they differ from next-token prediction
- how the brain organizes computation: cortical columns, excitatory-inhibitory balance, and the canonical microcircuit
- where biological analogies help and where they mislead: 15 adversarial analyses that separate genuine insight from superficial resemblance

## how to read this wiki

start with the introductory material:

1. this page (you are here)
2. [[the_brain_in_one_page]] -- the 80/20 overview of neuroscience for ML engineers
3. [[neuroscience_for_ml_engineers]] -- deeper treatment of the concepts most relevant to architecture design

then read the bridge documents. these connect biological mechanisms to todorov code. each one asks "is this biological analogy real?" and gives an honest answer:

4. [[plasticity_to_kda_delta_rule]] -- is the KDA delta rule Hebbian learning?
5. [[sparse_coding_to_ternary_spikes]] -- are ternary spikes sparse coding?
6. [[energy_efficiency_to_ternary_spikes]] -- do ternary spikes actually save energy?
7. [[dendritic_computation_to_swiglu]] -- is SwiGLU analogous to dendritic gating?
8. the remaining 12 bridge documents in the [[index]]

then explore individual mechanism articles as needed. use the [[glossary]] when you encounter unfamiliar terms.

## the key insight

this wiki began as a search for biological justification of todorov's architecture. it found something more interesting: most biological analogies in todorov are superficial.

the 15 adversarial analyses in the comparison and bridge sections showed that:

- ternary spikes are not sparse coding. cortical neurons fire at 1-5% rates. todorov spikes fire at 41%. the mechanism is useful but the analogy is weak.
- the KDA delta rule is not STDP. it uses an outer product (Hebbian), but there is no timing dependence, no asymmetric learning window, no causal inference.
- LayerNorm is not divisive normalization. the mathematical form is similar but the computational role is different.
- the residual stream is not a global workspace. it is a shared bus, not a selective broadcasting mechanism.
- Mamba3 complex rotation is not oscillatory dynamics. it serves positional encoding, not temporal coordination.
- cortical layers are not transformer layers. the connectivity, cell types, and computational roles are fundamentally different.

the architecture works not because it faithfully reproduces biology, but because the biological constraints it borrows -- ternary quantization, recurrent state accumulation, adaptive thresholds, outer-product associative memory -- are independently useful engineering choices.

the one genuine biological correspondence: the outer-product associative memory in KDA (S_t = alpha * S_{t-1} + beta * k * v^T) genuinely mirrors Hebbian learning. the outer product k * v^T is the same operation the brain uses to bind key-value pairs in memory. this is not an analogy. it is the same math.

## links

- [[the_brain_in_one_page]] -- compressed neuroscience overview
- [[neuroscience_for_ml_engineers]] -- deeper treatment for architecture designers
- [[mathematical_foundations]] -- the mathematical structures shared by neuroscience and ML
- [[todorov_biology_map]] -- which todorov components map to which biological mechanisms
- [[glossary]] -- every technical term in this wiki, defined in plain language
