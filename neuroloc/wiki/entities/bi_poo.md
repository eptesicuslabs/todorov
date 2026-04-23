# Guo-qiang Bi and Mu-ming Poo

status: definitional. last fact-checked 2026-04-16.

## biographical summary

Guo-qiang Bi and Mu-ming Poo are neuroscientists who, in their landmark 1998 paper, systematically mapped the spike-timing-dependent plasticity ([[stdp]]) window in cultured hippocampal neurons. their work established the precise temporal relationship between pre- and postsynaptic spike timing and the direction and magnitude of synaptic modification.

Mu-ming Poo was at the University of California, San Diego (later UC Berkeley) and is known for foundational work on synaptic plasticity, axon guidance, and neural circuit development. Guo-qiang Bi conducted the experiments as a graduate student in Poo's laboratory and later became a professor at the University of Pittsburgh.

## the 1998 experiment

Bi and Poo recorded from pairs of connected glutamatergic neurons in dissociated rat hippocampal cultures. they induced action potentials in the presynaptic and postsynaptic neurons at controlled time intervals and measured the resulting change in synaptic strength.

key findings:
- postsynaptic spiking within ~20 ms AFTER presynaptic activation produced LTP
- postsynaptic spiking within ~20 ms BEFORE presynaptic activation produced LTD
- the magnitude of change decayed exponentially with the timing difference
- LTP occurred preferentially at synapses with low initial strength (weight dependence)
- LTD magnitude did not depend on initial synaptic strength
- 60 repetitions at 1 Hz were sufficient for lasting modification

this data produced the canonical asymmetric STDP learning window: an exponential LTP lobe for positive Delta_t (pre-before-post) and an exponential LTD lobe for negative Delta_t (post-before-pre).

## significance

the Bi and Poo experiment was not the first demonstration of timing-dependent plasticity (Levy and Steward 1983, Markram et al. 1997 preceded it), but it was the first to systematically and quantitatively map the full timing window with millisecond precision. their data provided the canonical reference curve that is used in virtually all computational models of [[stdp]].

## relevance to todorov

the Bi and Poo learning window demonstrates that biological learning is fundamentally temporal: the sign of plasticity depends on the ORDER of events, not just their co-occurrence. todorov's KDA delta rule does not implement this timing dependence (k_t and v_t are from the same timestep), which means it captures the Hebbian correlation component of plasticity but not the temporal asymmetry. see [[plasticity_to_matrix_memory_delta_rule]].

## key reference

- Bi, G.-q. & Poo, M.-m. (1998). Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type. Journal of Neuroscience, 18(24), 10464-10472.
