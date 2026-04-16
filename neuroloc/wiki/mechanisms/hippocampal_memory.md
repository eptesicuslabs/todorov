# hippocampal memory

status: definitional. last fact-checked 2026-04-16.

**why this matters**: the hippocampus is a biological content-addressable memory that encodes in one shot and retrieves via pattern completion -- the exact capabilities that recurrent state and key-value associative memories attempt to replicate in ML architectures.

## summary

the **hippocampal formation** (a medial temporal lobe structure critical for episodic memory and spatial navigation) is the brain's fast learning system, capable of **one-shot encoding** (storing a new memory from a single experience) of **episodic memories** (memories of specific events, bound to time and place). it consists of four interconnected subregions -- **dentate gyrus** (**DG**), **CA3**, **CA1**, and **subiculum** -- each performing a distinct computational operation. together they implement a **content-addressable memory** (a memory system where retrieval is driven by the content of the query, not by an address or index) that stores pointers to distributed cortical representations, not the representations themselves.

ML analog: the hippocampal circuit is analogous to a key-value memory with hash-based lookup -- DG hashes inputs to sparse keys, CA3 stores and completes associations, and CA1 compares retrieved values against current input. the hippocampal indexing theory (Teyler & DiScenna 1986) holds that when an experience activates a pattern of cortical neurons, the hippocampus forms an index -- a compressed pointer -- that can later reinstate the full cortical pattern from a partial cue.

## the hippocampal circuit

### entorhinal cortex (input)

the entorhinal cortex (EC) is the primary interface between neocortex and hippocampus. layer II of EC projects to dentate gyrus and CA3 via the perforant path. layer III projects directly to CA1. the EC carries multimodal cortical information: spatial (grid cells), object identity, temporal context, and sensory features. this convergence makes the EC the "funnel" through which cortical experience reaches the hippocampus.

### dentate gyrus (pattern separation)

the dentate gyrus transforms overlapping **entorhinal inputs** (signals from the entorhinal cortex, the primary cortical interface to the hippocampus) into sparse, orthogonal representations. this is **pattern separation** (the transformation of similar inputs into dissimilar internal representations): two similar inputs produce two dissimilar DG outputs.

ML analog: pattern separation is analogous to locality-sensitive hashing or random projection into a high-dimensional sparse space -- converting dense, overlapping inputs into sparse, non-interfering codes.

the mechanism is extreme sparsity. the DG has approximately 1 million granule cells in rats (10 million in humans), but only ~2-5% are active at any given time (Chawla et al. 2005, Leutgeb et al. 2007). this means two experiences that share 90% of their entorhinal input will activate almost completely non-overlapping populations of granule cells. the combinatorial space is enormous: choosing 50,000 active cells from 1,000,000 gives C(10^6, 5*10^4) possible patterns.

the DG achieves this sparsity through:
- high convergence ratio: ~10,000 EC neurons project to each granule cell, each granule cell receives from ~4,000 EC neurons
- strong inhibitory feedback: basket cells and hilar interneurons enforce a winner-take-all (k-WTA) competition
- high firing thresholds: granule cells have unusually negative resting potentials (~-75 mV vs ~-65 mV for pyramidal cells)

each granule cell connects to ~15 CA3 pyramidal cells via the powerful mossy fiber synapses (large boutons, multiple release sites, strong depolarization). these act as "detonator synapses" that force their target CA3 cells to fire.

### CA3 (auto-associative network, pattern completion)

CA3 is the hippocampus's auto-associative memory: a recurrent network that can retrieve a complete stored pattern from a partial or degraded cue. this is pattern completion.

CA3 pyramidal cells are massively recurrently connected: each cell sends ~12,000-25,000 **recurrent collaterals** (axon branches that loop back to contact other neurons in the same region) to other CA3 cells (Amaral & Witter 1989). these recurrent connections implement a **Hopfield-like auto-associative network** (see [[pattern_completion]]). the weight matrix is shaped by [[hebbian_learning]]: synapses between co-active cells strengthen via **LTP** (long-term potentiation) at CA3-CA3 synapses.

ML analog: CA3 auto-association is analogous to a linear attention layer with outer-product state updates -- the same mechanism as KDA's S_t = S_{t-1} + k * v^T.

storage works in two phases:
1. during encoding, the sparse DG output via mossy fibers forces a specific pattern of CA3 activity. the recurrent CA3-CA3 synapses potentiate between co-active cells, storing the pattern as an attractor in the energy landscape
2. during retrieval, a partial cue from EC (via the direct perforant path to CA3) activates a subset of the stored pattern. the recurrent dynamics complete the pattern: the network settles into the nearest stored attractor

the storage capacity of CA3 as a Hopfield-like network is limited by interference between stored patterns. for N CA3 neurons with random binary patterns, the theoretical limit is ~0.138 * N patterns (McEliece et al. 1987). with ~250,000 CA3 neurons in rat hippocampus, this gives ~34,000 patterns. but the sparse DG preprocessing dramatically reduces interference, effectively increasing usable capacity.

### CA1 (comparator, novelty detection)

CA1 receives two inputs: the completed pattern from CA3 via Schaffer collaterals, and the current sensory input from EC layer III. CA1 acts as a comparator: it computes the match between the retrieved memory (CA3 output) and the current experience (EC input).

when the match is high: the experience is familiar, and CA1 passes the retrieved memory to the subiculum for output. when there is a mismatch: the experience is novel, and CA1 generates a novelty signal that can trigger re-encoding. this mismatch detection drives the distinction between remembering a familiar scene and noticing something has changed.

Hasselmo and colleagues (Hasselmo & Schnell 1994) proposed that acetylcholine modulates the balance between encoding and retrieval in CA1: high ACh suppresses CA3->CA1 transmission (favoring new encoding from EC), low ACh enhances CA3->CA1 transmission (favoring retrieval).

### subiculum (output)

the subiculum is the primary output structure of the hippocampal formation. it projects to prefrontal cortex, retrosplenial cortex, nucleus accumbens, hypothalamus, and the mammillary bodies. it also projects back to the entorhinal cortex, completing the hippocampal loop.

the subiculum performs spatial coding (boundary vector cells, axis-tuned cells) and may integrate hippocampal output with broader cortical context before broadcasting.

## the hippocampal indexing theory

Teyler and DiScenna (1986) proposed that the hippocampus does not store the content of memories. instead, it stores an index -- a set of pointers to the cortical neurons that were active during the original experience.

the theory rests on the anatomy: the hippocampus receives convergent input from all association cortices via the entorhinal cortex, and it projects back to those same cortices. when an experience occurs:

1. a pattern of cortical neurons is activated (visual cortex for sights, auditory cortex for sounds, somatosensory cortex for textures, etc.)
2. this distributed cortical pattern converges on the hippocampus via the entorhinal cortex
3. the hippocampus rapidly forms an index: a compressed representation (via DG pattern separation and CA3 autoassociation) that is linked to the cortical pattern via bidirectional connections
4. during retrieval, a partial cue activates part of the hippocampal index. CA3 pattern completion retrieves the full index. the hippocampal output (via subiculum and EC) reactivates the original pattern of cortical neurons, producing the subjective experience of remembering

the indexing theory explains several phenomena:
- hippocampal lesions cause retrograde amnesia for recent memories (the index is destroyed, so the cortical pattern cannot be reinstated) but spare remote memories (which have been consolidated to cortex and no longer need the hippocampal index; see [[memory_consolidation]])
- hippocampal damage impairs new learning (the indexing mechanism is lost) but spares existing skills and general knowledge (stored in cortex)
- the hippocampus is small relative to the cortex (~0.1% of cortical volume), consistent with storing compressed pointers rather than full representations

## one-shot learning

the hippocampus must learn from single experiences. unlike the neocortex, which requires many repetitions to extract statistical regularities (see [[complementary_learning_systems]]), the hippocampus forms a memory from a single exposure. this demands:

- fast synaptic modification: CA3 recurrent synapses exhibit rapid LTP (potentiation within seconds of co-activation)
- sparse representations: the 2-5% DG firing rate minimizes interference between memories, allowing new patterns to be stored without overwriting existing ones
- strong detonator synapses: mossy fiber synapses reliably drive CA3 cells, ensuring that the DG-specified pattern is imposed on CA3 even from a single presentation

the cost of one-shot learning is limited capacity and vulnerability to interference. the hippocampus cannot store unlimited numbers of patterns -- as capacity fills, new patterns begin to interfere with old ones. this motivates the need for [[memory_consolidation]]: transferring memories from the capacity-limited hippocampus to the high-capacity neocortex.

## relationship to todorov

KDA's matrix-valued state S_t functions as an associative memory with content-addressable retrieval (q^T * S_t), analogous to CA3 pattern completion. however, the correspondence is limited: CA3 is a nonlinear auto-associative network with attractor dynamics, while KDA is a linear heteroassociative memory with exponential decay. see [[memory_kda_vs_hippocampus]] for a detailed comparison.

the DG's extreme sparsity (~2-5% firing rate) contrasts sharply with todorov's ternary spike firing rate (~41%). both serve the purpose of reducing interference between stored patterns, but they operate in fundamentally different regimes. see [[sparse_coding_to_ternary_spikes]].

## challenges

the hippocampal memory system faces several unresolved questions:

- how does pattern separation in DG interact with pattern completion in CA3 to determine whether a new experience is encoded as a new memory or assimilated to an existing one? the threshold between "similar enough to retrieve" and "different enough to store separately" is not well understood
- how are hippocampal indices selectively consolidated to neocortex? not all memories are consolidated -- most are forgotten. the selection mechanism (possibly involving emotional tagging via amygdala, or replay frequency during sleep) remains debated
- the indexing theory assumes that cortical representations are stable enough to be re-indexed days or weeks later. but cortical representations drift over time (Schoonover et al. 2021). how does the index remain valid as its targets change?

## key references

- Amaral, D. G. & Witter, M. P. (1989). The three-dimensional organization of the hippocampal formation: a review of anatomical data. Neuroscience, 31(3), 571-591.
- Chawla, M. K. et al. (2005). Sparse, environmentally selective expression of Arc RNA in the upper blade of the rodent fascia dentata by brief spatial experience. Hippocampus, 15(5), 579-586.
- Hasselmo, M. E. & Schnell, E. (1994). Laminar selectivity of the cholinergic suppression of synaptic transmission in rat hippocampal region CA1. Journal of Neuroscience, 14(6), 3898-3914.
- Leutgeb, J. K. et al. (2007). Pattern separation in the dentate gyrus and CA3 of the hippocampus. Science, 315(5814), 961-966.
- McEliece, R. J., Posner, E. C., Rodemich, E. R. & Venkatesh, S. S. (1987). The capacity of the Hopfield associative memory. IEEE Transactions on Information Theory, 33(4), 461-482.
- Teyler, T. J. & DiScenna, P. (1986). The hippocampal memory indexing theory. Behavioral Neuroscience, 100(2), 147-154.
- Rolls, E. T. (2013). The mechanisms for pattern completion and pattern separation in the hippocampus. Frontiers in Systems Neuroscience, 7, 74.

## see also

- [[pattern_completion]]
- [[complementary_learning_systems]]
- [[memory_consolidation]]
- [[sparse_coding]]
- [[hebbian_learning]]
- [[memory_kda_vs_hippocampus]]
