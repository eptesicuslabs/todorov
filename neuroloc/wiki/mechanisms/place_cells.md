# place cells

status: definitional. last fact-checked 2026-04-16.

**why this matters**: place cells demonstrate that individual neurons can encode discrete, abstract states (locations) within a distributed map, which is the biological prototype for sparse, localized representations in memory-augmented neural networks and state-space models.

## summary

**place cells** are **hippocampal** (located in the hippocampus, a medial temporal lobe structure critical for memory and navigation) pyramidal neurons (primarily in **CA1** and **CA3** -- two subregions of the hippocampus) that fire when an animal occupies a specific location in an environment. discovered by John O'Keefe and John Dostrovsky in 1971, they are the foundational cell type of the brain's "cognitive map" -- the internal representation of spatial environments proposed by O'Keefe and Nadel (1978). a single place cell has a well-defined **place field**: a contiguous region of the environment (typically ~30 cm diameter in rat, scaling with environment size) where the cell fires at high rates (~10-20 Hz) and is virtually silent elsewhere (<0.5 Hz). the population of place cells in the hippocampus collectively tiles the entire environment, forming a distributed spatial map.

ML analog: place cells are analogous to localized basis functions (radial basis functions) in ML. each unit responds to a specific region of input space and is silent elsewhere. the population forms a distributed, sparse representation -- similar to how a mixture-of-experts router activates different experts for different inputs.

## discovery

O'Keefe and Dostrovsky (1971) recorded from individual neurons in the hippocampus of freely moving rats and found cells whose firing was strongly correlated with the animal's position. these were not responding to any single sensory cue -- they fired in the same location regardless of the animal's orientation, behavior (eating, grooming, running), or the specific sensory stimuli available. O'Keefe termed these "place units."

the key insight was that these cells encoded location per se, not any particular sensory input. when salient visual cues were rotated, place fields rotated with them, maintaining their relationship to the cue configuration. but when individual cues were removed, place fields often persisted, suggesting that the spatial representation was controlled by the constellation of cues rather than any single landmark.

O'Keefe and Nadel formalized this in their 1978 monograph "The Hippocampus as a Cognitive Map," proposing that the hippocampus implements an allocentric (world-centered) spatial representation in which place cells are the fundamental units. this was a radical departure from the prevailing view of the hippocampus as a general-purpose memory structure, and was controversial for decades until the discovery of grid cells (see [[grid_cells]]) and the 2014 Nobel Prize vindicated the spatial navigation framework.

## place field properties

### size and shape

place fields in CA1 are roughly circular or elliptical, with diameters of ~30 cm in small environments (1 m^2 boxes in rat). they expand systematically with environment size: in a 18 m linear track, place fields can span several meters. the relationship is sublinear -- fields grow, but not proportionally to the environment.

along the dorso-ventral axis of the hippocampus, place field size increases: dorsal hippocampal cells have small, precise fields (~20-30 cm), while ventral cells have large, diffuse fields (several meters). this mirrors the dorso-ventral gradient of grid cell spacing in the medial entorhinal cortex (see [[grid_cells]]).

### firing rate

within the place field, firing rate peaks at the center and falls off roughly as a Gaussian. peak rates are typically 10-40 Hz, with mean in-field rates of 5-15 Hz. outside the field, firing drops to near zero (<0.5 Hz). the in-field to out-of-field ratio can exceed 100:1.

### stability

in a familiar, stable environment, place fields are remarkably stable over days to weeks (Thompson & Best 1990). the same cell fires in the same location across repeated visits, with consistent peak rate and field shape. this stability is the basis for the claim that place cells form a persistent spatial map.

### directionality

in open arenas, most CA1 place cells fire regardless of the animal's heading direction. in constrained environments (linear tracks, mazes), many place cells develop directional selectivity, firing preferentially on one direction of travel. this directional modulation is thought to arise from interactions with head direction cells (see [[path_integration]]) and the sequential nature of the experience.

## remapping

the same neuron can represent completely different locations in different environments. this is remapping: the reorganization of place cell firing patterns when the animal's context changes.

### global remapping

when an animal is moved to a sufficiently different environment (different room, different shape, different color), the entire population of place cells reorganizes. cells that had place fields may become silent, and cells that were silent may develop new fields in new locations. the place fields in the new environment bear no systematic relationship to the fields in the old environment.

ML analog: global remapping is analogous to how transformer representations change entirely between different input domains -- the same hidden units activate for completely different features when the context distribution shifts.

global remapping is controlled by changes in the constellation of sensory cues, the geometric properties of the environment (shape, boundaries), and possibly the animal's internal state. the mechanism is thought to involve pattern separation in the dentate gyrus (see [[hippocampal_memory]]), which orthogonalizes the hippocampal representations of different environments.

### rate remapping

when the environment remains the same but a non-spatial variable changes (different colors on the walls, different odors, different task demands), place fields maintain their locations but change their firing rates. this allows the same spatial map to encode additional information about the context.

Leutgeb et al. (2005) demonstrated that place field locations remained stable when only wall color changed, but firing rates shifted systematically. this dissociates spatial coding (field location) from contextual coding (field rate) within the same population.

### partial remapping

intermediate between global and rate remapping: some cells remap (gain or lose fields, or shift locations) while others remain stable. the proportion of remapping cells increases with the magnitude of the environmental change.

## phase precession

as an animal traverses a place field, the spikes of the place cell shift systematically in their timing relative to the ongoing theta oscillation (4-8 Hz; see [[theta_oscillations]]). when the animal enters the field, spikes occur at late phases of the theta cycle (~330 degrees). as the animal moves through the field, spikes advance to progressively earlier phases, reaching ~90 degrees by the time the animal exits the field. this phenomenon is called phase precession (O'Keefe & Recce 1993).

**phase precession** means that the **theta phase** (the position within the ~125 ms theta oscillation cycle at which a spike occurs) of a spike encodes the animal's position within the place field with higher precision than the firing rate alone. a single theta cycle contains spikes from place cells whose fields overlap with the animal's current trajectory. these spikes are ordered by position: cells representing recently visited locations fire at late theta phases, cells representing the current location fire at intermediate phases, and cells representing upcoming locations fire at early phases.

this creates a "theta sequence": a compressed representation of the animal's past, present, and near-future trajectory within each ~125 ms theta cycle. the compression ratio is approximately 8-10x (a ~1-second behavioral sequence compressed into ~125 ms), placing the information within the time window for spike-timing-dependent plasticity (see [[stdp]]).

implications for learning: theta sequences bring temporally adjacent experiences into the STDP time window (~20-40 ms), enabling the asymmetric strengthening of connections between sequentially active place cells. this could underlie the formation of forward-pointing associations in spatial memory.

implications for coding: a dual coding scheme operates simultaneously -- firing rate encodes location within the place field (coarse), while theta phase encodes sub-field position (fine). O'Keefe & Recce (1993) showed that phase carries ~2-3 bits of spatial information per spike, doubling the information conveyed by rate alone.

## inputs and computation

place cells receive their spatial information primarily from the entorhinal cortex (EC):

- **medial entorhinal cortex (MEC)**: provides spatial information via grid cells, head direction cells, border cells, and speed cells (see [[grid_cells]], [[path_integration]]). grid cell inputs may be combined to generate place fields through the superposition of periodic inputs at different scales and orientations (Solstad et al. 2006)
- **lateral entorhinal cortex (LEC)**: provides non-spatial information -- object identity, contextual cues, temporal context. LEC lesions impair object-place associations but spare basic spatial navigation

the generation of place fields from grid cell inputs is computationally elegant: the sum of several cosine functions with different spatial frequencies (grid spacings) and orientations produces a function that peaks at one location and is near-zero elsewhere. this is the Fourier synthesis model: place fields emerge as the constructive interference of grid cells at multiple scales.

alternative models propose that place fields arise from:
- boundary vector cells: cells in the subiculum/MEC that fire at a fixed distance and direction from environmental boundaries (Hartley et al. 2000)
- direct sensory input: visual landmark cells in LEC, combined with path integration
- attractor dynamics: CA3 recurrent connections form attractor states that correspond to specific locations (see [[hippocampal_memory]])

## relationship to other spatial cell types

place cells are one node in a network of specialized spatial cell types:

- [[grid_cells]]: provide the metric for the spatial map (periodic coordinate system)
- head direction cells: provide the compass (allocentric heading; see [[path_integration]])
- speed cells: provide the speedometer (running speed; see [[path_integration]])
- border cells: fire at boundaries of the environment (edges, walls)
- object vector cells: fire at a fixed distance and direction from specific objects

the hippocampal-entorhinal system combines these signals to construct a unified spatial representation. the direction of information flow is debated: grid cells may generate place cells (top-down), or place cells may help stabilize grid cells (bottom-up), or both may arise from common inputs.

## relationship to todorov

todorov does not have a direct analog to place cells. place cells are location-specific neurons that form a sparse, distributed spatial map -- the opposite of what a language model does. however, several parallels can be drawn at the computational level:

- the "cognitive map" metaphor has expanded beyond spatial navigation to describe how the hippocampal-entorhinal system organizes abstract relational knowledge (see [[cognitive_maps]]). place cells participate in representing non-spatial conceptual structures when spatial metaphors are used
- place cell remapping (the same neuron representing different things in different contexts) is loosely analogous to how embedding dimensions change their function across different input distributions, though the analogy is weak
- phase precession's dual coding (rate + phase) is an example of multiplexed information in spike timing -- a coding strategy that todorov's ternary spikes ({-1, 0, +1}) cannot express, since they carry no timing information

the GP self-interaction (see [[spatial_computation_to_pga]]) does not implement place cell-like spatial selectivity. see [[pga_vs_grid_cells]] for an adversarial comparison.

## challenges

the place cell framework has several unresolved issues. first, the mechanism by which place fields are generated from upstream inputs remains debated. the Fourier synthesis model (grid cells -> place cells via superposition) is elegant but may not be the primary mechanism -- place cells persist after grid cell disruption (via MEC lesions), suggesting alternative pathways through sensory or boundary-based inputs.

second, the generalization from rodent to human hippocampus is uncertain. human hippocampal recordings (from epilepsy patients) show cells that respond to specific concepts (the "Jennifer Aniston cell"), not spatial locations. whether these concept cells are the same as place cells operating in abstract concept space, or a fundamentally different cell type, is unresolved.

third, place cell representations are not purely spatial. firing rates encode contextual information (rate remapping), temporal information (time cells), and reward information (overrepresentation of reward locations). the degree to which place cells are "spatial" versus "contextual" or "episodic" varies with task demands, making the clean "cognitive map" interpretation an oversimplification.

## key references

- O'Keefe, J. & Dostrovsky, J. (1971). The hippocampus as a spatial map. Preliminary evidence from unit activity in the freely-moving rat. Brain Research, 34(1), 171-175.
- O'Keefe, J. & Nadel, L. (1978). The Hippocampus as a Cognitive Map. Oxford University Press.
- O'Keefe, J. & Recce, M. L. (1993). Phase relationship between hippocampal place units and the EEG theta rhythm. Hippocampus, 3(3), 317-330.
- Leutgeb, S. et al. (2005). Independent codes for spatial and episodic memory in hippocampal neuronal ensembles. Science, 309(5734), 619-623.
- Moser, E. I., Kropff, E. & Moser, M.-B. (2008). Place cells, grid cells, and the brain's spatial representation system. Annual Review of Neuroscience, 31, 69-89.
- Thompson, L. T. & Best, P. J. (1990). Long-term stability of the place-field activity of single units recorded from the dorsal hippocampus of freely behaving rats. Brain Research, 509(2), 299-308.
- Solstad, T., Moser, E. I. & Einevoll, G. T. (2006). From grid cells to place cells: a mathematical model. Hippocampus, 16(12), 1026-1031.

## see also

- [[grid_cells]]
- [[path_integration]]
- [[cognitive_maps]]
- [[hippocampal_memory]]
- [[theta_oscillations]]
- [[pattern_completion]]
- [[pga_vs_grid_cells]]
- [[spatial_computation_to_pga]]
