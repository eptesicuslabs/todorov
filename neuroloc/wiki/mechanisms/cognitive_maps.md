# cognitive maps

status: definitional. last fact-checked 2026-04-16.

**why this matters**: the cognitive map hypothesis has expanded from spatial navigation to general relational reasoning, suggesting that the brain uses a single graph-structured architecture for both physical navigation and abstract inference -- a principle that informs the design of graph neural networks and relational transformers.

## summary

the **cognitive map** is the brain's internal representation of the relational structure of an environment -- originally proposed as a spatial map (Tolman 1948, O'Keefe & Nadel 1978), now understood as a general mechanism for encoding abstract **relational knowledge** (structured representations of how entities relate to each other). the hippocampal-entorhinal system does not just map physical space: it constructs graph-structured representations of relationships between entities, states, and concepts. evidence from fMRI, single-unit recording, and computational modeling shows that the same neural machinery (place cells, grid cells, head direction cells) encodes non-spatial conceptual spaces. these include the topology of social hierarchies, the structure of task states, and the dimensions of abstract feature spaces.

this generalization transforms the spatial navigation system from a specialized GPS into a general-purpose relational computer. the Tolman-Eichenbaum Machine (Whittington et al. 2020) formalizes this view, using transformer-like attention mechanisms to navigate abstract relational structures and reproducing the firing properties of grid cells, place cells, and other hippocampal cell types as emergent properties of relational learning.

## historical development

### Tolman's cognitive maps (1948)

Edward Tolman challenged the dominant behaviorist view that rats learn spatial tasks through stimulus-response associations. in a series of maze experiments, he showed that rats formed internal representations of the maze layout that allowed flexible behavior: taking novel shortcuts, detour routes, and responding adaptively to blocked paths. Tolman (1948) called these representations "cognitive maps" -- internal models of the environment that support inference and planning, not just memorized routes.

Tolman's evidence was behavioral, not neural. the neural substrate was unknown until O'Keefe and Nadel proposed the hippocampus.

### O'Keefe and Nadel's hippocampal cognitive map (1978)

O'Keefe and Nadel (1978) proposed that the hippocampus implements Tolman's cognitive map, with place cells (see [[place_cells]]) as its fundamental units. their key claims:
- the hippocampus constructs an allocentric (world-centered) spatial representation
- place cells form a stable, self-consistent map of the environment
- this map supports flexible navigation: novel shortcuts, detours, inference of unvisited locations
- the map is built automatically during exploration, without reinforcement

this proposal was spatial: the cognitive map was a map of physical space. the extension to abstract domains came later.

### Eichenbaum's relational memory (1999-2004)

Howard Eichenbaum challenged the exclusively spatial interpretation. he argued that the hippocampus encodes relational memories -- the relationships between items, events, and contexts -- with spatial relationships as a special case. evidence:
- hippocampal neurons respond to non-spatial relationships: temporal order of events (MacDonald et al. 2011), odor-odor associations (Wood et al. 1999), transitive inference chains (Dusek & Eichenbaum 1997)
- hippocampal lesions impair relational learning broadly, not just spatial learning
- the hippocampus binds arbitrary associations between items and their contexts (what-where-when)

Eichenbaum proposed that the hippocampus creates a "memory space" -- a relational structure that connects items through their shared contexts, with spatial memory as one instance of this general relational computation.

### the convergence: cognitive maps are relational graphs

by the 2010s, the spatial and relational views converged. the hippocampal-entorhinal system builds graph-structured representations where:
- nodes correspond to states (places, items, concepts)
- edges correspond to transitions (movements, associations, transformations)
- the metric is determined by transition structure, not Euclidean distance

a place cell is a node. a grid cell encodes the structure of the graph (its periodicity, its metric). path integration is traversal of edges. pattern completion (see [[pattern_completion]]) is inference on the graph: given a partial cue, retrieve the full node.

## the Tolman-Eichenbaum Machine (TEM)

Whittington, Muller, Mark, Barry, Burgess, and Behrens (2020) proposed the Tolman-Eichenbaum Machine, a computational model that unifies spatial and relational memory through a single learning mechanism.

### architecture

TEM has two components:
1. **structural knowledge** (medial entorhinal cortex analog): learned representations of the transition structure of the environment. these form a basis set that describes how states relate to each other -- the "graph" of the world
2. **sensory knowledge** (hippocampal analog): bindings between the structural representations and specific sensory experiences. these link the abstract graph to particular observations at each node

the key insight: structural knowledge generalizes across environments. the transition structure of a 2D grid is the same whether the agent is in a room, a forest, or an abstract conceptual space. TEM learns this shared structure and transfers it to new environments.

### relationship to transformers

the TEM attention mechanism computes attention over a memory of past experiences using the structural representation as a kind of positional encoding. this is analogous to transformer self-attention with positional embeddings:
- the structural code plays the role of the positional embedding (encoding "where" in the relational graph)
- the sensory binding plays the role of the content embedding (encoding "what" is at each position)
- the attention operation retrieves content based on structural similarity (which graph position is being queried)

this parallel is suggestive but not exact: TEM uses a learned structural code (not fixed sinusoidal embeddings), and the attention operates over a memory of experiences (not a fixed context window).

### emergent cell types

after learning, TEM's internal representations reproduce the firing properties of known spatial cell types:
- **grid cell-like units**: periodic spatial firing patterns in the structural knowledge module, with discrete spacings and orientations
- **place cell-like units**: localized spatial firing in the sensory binding module, with context-dependent remapping
- **band cells, border cells, object vector cells**: arise as different patterns in the structural basis set
- **landmark cells**: sensory binding units that respond to specific objects at specific locations

critically, TEM also produces these patterns in non-spatial tasks: when navigating social hierarchies or transitive inference chains, the same units show periodic structure analogous to grid cells, demonstrating that the spatial firing patterns are a special case of general relational structure encoding.

## evidence for non-spatial cognitive maps

### conceptual spaces (Constantinescu et al. 2016)

Constantinescu, O'Reilly, and Behrens (2016) had human participants learn a 2D conceptual space defined by the dimensions of a bird stimulus (leg length and neck length). using fMRI, they found hexagonal (6-fold symmetric) modulation of BOLD signal in the entorhinal cortex as participants mentally navigated the conceptual space -- the same signal signature as spatial grid cells.

this result was replicated and extended to other abstract dimensions: reward spaces (Bongioanni et al. 2021), social hierarchies (Tavares et al. 2015), and auditory frequency spaces (Aronov et al. 2017 in rodents).

### social cognitive maps (Tavares et al. 2015)

Tavares et al. (2015) showed that the human hippocampus encodes social relationships in a 2D space defined by power and affiliation. hippocampal activity tracked the "position" of the participant in this social space, and updating social relationships activated the hippocampus in a manner analogous to spatial navigation. the encoding was geometric: social distance (Euclidean distance in the power-affiliation space) predicted hippocampal response magnitude.

### task state spaces (Schuck et al. 2016)

during a sequential decision task, hippocampal activity reflected the animal's position in an abstract task state space, not physical space. transitions between task states activated grid-like periodic signals in entorhinal cortex. this supports the view that grid cells encode transition structure in any organized space, not just physical environments.

### successor representations (Stachenfeld et al. 2017)

Stachenfeld, Botvinick, and Gershman (2017) proposed that hippocampal place cells encode the **successor representation** (SR -- a predictive map where each state is represented by the expected discounted frequency of visiting all future states from that state): M(s, s') = E[sum_{t=0}^{inf} gamma^t * I(s_t = s' | s_0 = s)].

ML analog: the successor representation is closely related to the value function in reinforcement learning and to the transition matrix learned by world models. both encode predictive structure -- not what IS, but what WILL BE.

under the SR interpretation:
- place cells encode the SR: each place cell fires proportionally to the expected future occupancy of its place field
- grid cells emerge as the eigenvectors of the SR (or equivalently, of the graph Laplacian), which are periodic functions on the graph -- explaining hexagonal firing on a 2D plane
- the cognitive map IS the SR: a predictive model of transition structure

the SR naturally extends to non-spatial domains: any domain with transition structure has an SR, and grid-like eigenvectors emerge from any smooth, isotropic transition matrix.

## graph-theoretic formalization

a cognitive map can be formalized as a weighted graph G = (V, E, W):
- V: set of nodes (states, places, items)
- E: set of edges (transitions, associations)
- W: edge weights (transition probabilities, associative strengths)

the spectral properties of this graph determine the structure of the map:
- the graph Laplacian L = D - W (where D is the degree matrix) has eigenvectors that form a natural basis for representing positions on the graph
- the low-frequency eigenvectors of L are smooth, periodic functions on the graph -- they are the "grid cells" of the abstract space
- the number of modules (discrete spatial scales) corresponds to the number of distinct eigenvalue clusters

this connects cognitive maps to graph signal processing and spectral graph theory. the grid cell coding scheme is not arbitrary: it is the mathematically optimal basis for representing positions on a graph, in the sense of minimizing mean-squared error for a given number of basis functions.

## relationship to todorov

the cognitive map framework is relevant to todorov at two levels:

### architectural level

todorov's hybrid architecture (KDA + MLA) can be seen through the cognitive map lens:
- **KDA** maintains a recurrent state that accumulates information over time, analogous to path integration in the cognitive map (tracking position through transitions). but KDA does not build a graph -- it builds a matrix of associations that decays exponentially
- **MLA** performs content-based retrieval over a context window, analogous to landmark-based localization (querying the map by content). MLA IS a modern Hopfield network (see [[pattern_completion]]), which is a form of content-addressable memory, not a graph navigation system

neither KDA nor MLA implements a cognitive map in the full sense: there is no graph structure, no transition model, no successor representation, and no spectral basis. the closest analog would be a model that learns the transition structure of token sequences and represents positions in the resulting graph -- which is closer to what a standard transformer does implicitly through its learned positional embeddings.

### the PGA module

the G(3,0,1) PGA self-interaction (see [[spatial_computation_to_pga]]) is the most plausible site for cognitive map-like computation in todorov, if such computation exists. PGA can represent the symmetry group of 3D space (rotations, translations, reflections via the sandwich product), and cognitive maps are fundamentally about symmetry: the transition structure of a regular space is a group (the translation group for a 2D plane, for example).

however, in todorov's implementation:
- the GP is a self-interaction (both inputs from the same x), not a state update or transition operator
- there is no explicit graph structure or transition model
- the 16-component multivector is projected back to d_model via a linear layer, which can discard any geometric structure that gradient descent does not find useful

the honest assessment: cognitive maps are graph-structured predictive models. todorov is a sequence model. these are different computational paradigms, even if they share some mathematical primitives (matrix operations, recurrence, content-based retrieval). see [[pga_vs_grid_cells]] for the adversarial comparison of PGA and grid cell computation.

## challenges

the cognitive map framework faces several substantive criticisms. first, the generalization from spatial to non-spatial domains may be overstated. the fMRI evidence for "grid-like" signals in abstract spaces relies on 6-fold symmetric modulation of BOLD signal, which could arise from non-grid mechanisms (vascular artifacts, eye movement patterns, or any computation with hexagonal symmetry). single-unit evidence for grid cells in abstract non-spatial tasks is still limited.

second, the graph-theoretic formalization is powerful but underspecified. any relational structure can be described as a graph, so saying "the hippocampus builds graphs" risks being unfalsifiable. the theory needs to predict WHICH graph the hippocampus builds (topology, edge weights, node features) for a given experience, and current models do not make sufficiently specific predictions.

third, the relationship between the TEM model's attention mechanism and biological attention is loose. the TEM uses transformer-like attention as a computational convenience, but the hippocampus does not implement dot-product attention. the mapping from TEM's structural codes to actual grid cell firing patterns is a post-hoc fit to known properties, not a prediction of new ones.

## key references

- Tolman, E. C. (1948). Cognitive maps in rats and men. Psychological Review, 55(4), 189-208.
- O'Keefe, J. & Nadel, L. (1978). The Hippocampus as a Cognitive Map. Oxford University Press.
- Eichenbaum, H. (2004). Hippocampus: cognitive processes and neural representations that underlie declarative memory. Neuron, 44(1), 109-120.
- Whittington, J. C. R. et al. (2020). The Tolman-Eichenbaum Machine: unifying space and relational memory through generalization in the hippocampal formation. Cell, 183(5), 1249-1263.
- Constantinescu, A. O., O'Reilly, J. X. & Behrens, T. E. J. (2016). Organizing conceptual knowledge in humans with a gridlike code. Science, 352(6292), 1464-1468.
- Stachenfeld, K. L., Botvinick, M. M. & Gershman, S. J. (2017). The hippocampus as a predictive map. Nature Neuroscience, 20(11), 1643-1653.
- Tavares, R. M. et al. (2015). A map for social navigation in the human brain. Neuron, 87(1), 231-243.
- Aronov, D., Nevers, R. & Tank, D. W. (2017). Mapping of a non-spatial dimension by the hippocampal-entorhinal circuit. Nature, 543(7647), 719-722.
- Behrens, T. E. J. et al. (2018). What is a cognitive map? Organizing knowledge for flexible behavior. Neuron, 100(2), 490-509.

## see also

- [[place_cells]]
- [[grid_cells]]
- [[path_integration]]
- [[hippocampal_memory]]
- [[pattern_completion]]
- [[complementary_learning_systems]]
- [[pga_vs_grid_cells]]
- [[spatial_computation_to_pga]]
