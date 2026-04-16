# developmental self-organization

status: definitional. last fact-checked 2026-04-16.

**why this matters**: the brain builds ordered internal representations (topographic maps, orientation columns) from simple local learning rules without any global loss function, demonstrating that useful structure can emerge from competitive Hebbian learning -- a principle relevant to unsupervised pretraining and self-supervised representation learning.

## summary

neural circuits develop ordered spatial structure -- **topographic maps** (ordered spatial arrangements where neighboring neurons represent neighboring features or locations), **orientation columns** (cortical columns where neurons share a preferred orientation), **ocular dominance columns** (alternating stripes of cortex dominated by left or right eye input) -- through activity-dependent processes that require no explicit supervision or template. the mechanism is fundamentally **Hebbian** (connections strengthen when pre- and post-synaptic neurons are co-active): neurons that fire together wire together. neurons that fire together tend to be those that respond to similar features or receive input from nearby locations. combined with **lateral competition** ([[lateral_inhibition]], [[winner_take_all]]), Hebbian learning produces self-organized maps where nearby neurons represent similar features. the result is smooth topographic organization from initially unstructured or broadly tuned connectivity.

ML analog: self-organized feature maps are the biological precursor to self-supervised representation learning (contrastive learning, MAE). both build structured internal representations from unlabeled data using local similarity-based objectives rather than global supervised loss.

the developmental sequence is: (1) molecular gradients establish coarse spatial organization (protomap), (2) spontaneous activity patterns (retinal waves, thalamic bursts) refine connectivity before sensory experience, (3) sensory-driven activity completes the refinement during [[critical_periods]]. at each stage, the principle is the same: correlated activity stabilizes connections ([[synaptic_pruning|selective stabilization]]), and competition ensures that different neurons come to represent different features. the result is ordered structure from simple local rules, without any global blueprint specifying which neuron should respond to which stimulus.

## mechanisms

### molecular protomap

before activity-dependent refinement begins, transcription factor gradients in the cortical ventricular zone (Emx2, Pax6, COUP-TFI, SP8) establish a rough positional code that specifies cortical areas (Sur and Rubenstein 2005). these gradients define proto-areas: V1 vs S1 vs M1 identity is specified genetically, but the internal organization within each area is not. the molecular protomap provides the scaffold; activity-dependent self-organization fills in the detail.

in the visual pathway, ephrin/Eph receptor gradients establish the initial topographic mapping from retina to lateral geniculate nucleus (LGN) and from LGN to V1. retinal ganglion cells with high EphA expression project to regions of the LGN with low ephrin-A expression, creating a rough retinotopic map. but this molecular map is coarse: adjacent ganglion cells may project to non-adjacent LGN targets. the refinement from coarse molecular to precise functional topography requires neural activity.

### spontaneous retinal waves

before the eyes open and before any visual experience, the retina generates spontaneous waves of activity that propagate across the ganglion cell layer. these retinal waves are the primary source of patterned activity that drives the refinement of visual topographic maps in the LGN and V1.

retinal waves have three developmental stages:

1. **stage I (embryonic):** mediated by gap junctions between retinal ganglion cells. spatially limited, low correlation.
2. **stage II (early postnatal, before eye opening):** mediated by cholinergic (nicotinic) amacrine cells. large, slow propagating waves covering hundreds of microns, lasting several seconds, recurring every 1-2 minutes. these waves are responsible for eye-specific segregation and gross retinotopic refinement.
3. **stage III (around eye opening):** mediated by glutamatergic transmission. faster, smaller waves. responsible for ON/OFF segregation and fine-scale retinotopic refinement.

the spatial and temporal correlations in retinal waves provide the activity patterns needed for Hebbian refinement: neurons that are retinal neighbors tend to participate in the same wave (and thus fire together), while distant neurons rarely co-activate. this correlation structure, combined with Hebbian learning rules at retinogeniculate and geniculocortical synapses, refines the coarse molecular topography into a precise retinotopic map.

disrupting retinal waves (genetically, e.g., beta-2 nicotinic receptor knockout; or pharmacologically, e.g., epibatidine) prevents normal topographic refinement. the result is a coarse but imprecise retinotopic map, confirming that waves are necessary for fine-scale organization (Cang et al. 2005, McLaughlin et al. 2003).

a remarkable recent finding: retinal waves in mice are not random. they have directional biases that correspond to future **optic flow** (the pattern of apparent motion of visual features caused by self-movement) patterns. this effectively "pre-trains" the visual system for motion detection before the eyes open (Ge et al. 2021). spontaneous activity is not merely correlated noise -- it contains structured information about the statistics of the environment that the visual system will encounter.

ML analog: retinal waves are the biological equivalent of pretraining on synthetic data. the developing brain generates its own training distribution (structured spontaneous activity) that approximates the statistics of the real-world data it will encounter after eye opening.

### ocular dominance columns

in many species (cat, primate, but not rodent), neurons in layer IV of V1 are organized into alternating columns dominated by the left or right eye. this ocular dominance pattern emerges through competitive Hebbian dynamics between left-eye and right-eye inputs.

the mechanism:

1. initially, both eyes provide synaptic input to the same cortical neurons. the inputs are intermingled and binocular.
2. activity from the two eyes is poorly correlated (each eye generates independent retinal waves and, after eye opening, sees slightly different views due to binocular disparity).
3. [[hebbian_learning|Hebbian plasticity]] ([[stdp]]) strengthens inputs that are correlated with the postsynaptic neuron's response and weakens inputs that are uncorrelated.
4. lateral interactions ensure that nearby cortical neurons tend to receive the same winning eye's input (local correlation from shared thalamic inputs and lateral excitation).
5. the result is spatial segregation: patches of cortex become dominated by one eye, with the other eye's inputs weakened and eventually retracted.

this is self-organized competition: the two eyes compete for cortical territory, and the outcome (the specific pattern of columns) is determined by the initial conditions and activity patterns, not by a genetic template. different animals of the same species have different column patterns, confirming that the pattern is activity-dependent, not genetically specified.

the mathematical description uses a Hebbian learning rule with constraints (weight normalization or BCM-like threshold, see [[bcm_theory]]) and correlation-based competition between inputs. the columns emerge as eigenvectors of the correlation structure of the inputs, filtered through the lateral interaction kernel.

### orientation maps

in species with orientation columns (cat, primate, tree shrew), V1 neurons are organized into a pinwheel structure where preferred orientation varies smoothly across the cortical surface, with singularities (pinwheel centers) where all orientations converge. this map also emerges through self-organization.

the proposed mechanism:

1. ON and OFF retinal ganglion cells project to nearby but not identical locations in the cortex, creating a spatial offset in the representation.
2. Hebbian learning combined with lateral competition produces neurons selective to oriented edges (edges are the simplest stimulus that creates differential ON/OFF activation).
3. the smooth variation of preferred orientation across cortex is a consequence of the lateral interaction structure: nearby neurons tend to learn similar orientations due to shared local input and lateral excitation.
4. the pinwheel structure (singularities and saddle points) is a topological necessity of mapping a 1D periodic variable (orientation, 0-180 degrees) onto a 2D surface.

computational models (e.g., elastic net model, Kohonen SOM, cortical learning algorithms) can reproduce orientation map formation with realistic statistics from simple Hebbian rules plus competition. the key requirement is that the input contains correlations at multiple spatial scales and that the cortical sheet has local lateral interactions.

### self-organizing maps (Kohonen)

Teuvo Kohonen's **self-organizing map** (SOM -- an unsupervised algorithm that projects high-dimensional inputs onto a low-dimensional lattice while preserving topology) algorithm (1982) is the computational abstraction of cortical topographic map formation. the SOM consists of:

1. a lattice of units (neurons), each with a weight vector in input space
2. competitive learning: for each input, the unit with the closest weight vector (the "winner") is identified
3. cooperative update: the winner and its lattice neighbors update their weight vectors toward the input, with the update magnitude decaying with lattice distance from the winner
4. the neighborhood function shrinks over time, producing increasingly fine-grained maps

the SOM produces topology-preserving mappings: nearby inputs in the input space are represented by nearby units in the lattice. this is the defining property of cortical topographic maps.

the biological correspondence is approximate:
- the competitive selection corresponds to WTA dynamics via lateral inhibition ([[winner_take_all]], [[lateral_inhibition]])
- the cooperative update corresponds to Hebbian learning with lateral excitation ([[hebbian_learning]])
- the shrinking neighborhood corresponds to the maturation of inhibitory circuits during [[critical_periods]] (as inhibition matures, competition becomes more local, producing finer maps)
- the weight normalization constraint in the SOM corresponds to [[homeostatic_plasticity]] or [[bcm_theory]]

the SOM is not a mechanistic model of cortical development -- it lacks spiking dynamics, realistic inhibitory circuits, and temporal structure. but it captures the essential computational principle: competitive Hebbian learning with spatial structure produces ordered maps from unstructured input.

## the role of spontaneous activity

a recurring theme is that structured activity patterns exist before sensory experience:

- **retinal waves** drive retinotopic refinement and eye-specific segregation before eye opening
- **thalamic spindle oscillations** provide correlated activity to developing cortical circuits before sensory input arrives
- **cortical slow waves** organize activity across developing cortical networks

this means that self-organization does not wait for the environment. the brain generates its own training data -- spontaneous activity with the right correlation structure -- to begin circuit refinement before birth or eye opening. sensory experience then continues and completes the process during the [[critical_periods|critical period]].

the teleological implication is that the brain is not a blank slate awaiting instruction. it arrives with structured circuits pre-organized by structured spontaneous activity, which creates an initial scaffold that experience refines. the developmental sequence is: genes -> molecular gradients -> spontaneous activity -> experience -> mature circuit.

## relationship to todorov

todorov's training does not involve any form of self-organization in the developmental sense. all layer connectivity is specified at initialization (random initialization from a known distribution), and gradient descent optimizes all parameters simultaneously from step 1.

the closest analog to developmental self-organization is the emergence of structure in the learned weights during training. gradient descent on the language modeling objective produces internally organized representations (attention heads specialize, MLP neurons develop interpretable features, spike patterns become informative -- MI reaching 1.168 at 267m scale). but this is optimization-driven, not self-organized in the developmental sense:

- **no competition between inputs:** in biology, left-eye and right-eye inputs compete for cortical territory. in todorov, different input tokens do not compete for representational resources within a layer -- each token is processed independently (within-sequence, modulo attention).
- **no spatial structure in the weight space:** in biology, nearby neurons represent similar features because of local lateral interactions. in todorov, there is no spatial relationship between adjacent dimensions of the weight matrix.
- **no spontaneous pre-training:** in biology, retinal waves pre-organize circuits before experience. todorov has no analog of structured initialization that encodes prior knowledge about input statistics. (pretraining on a different dataset would be a crude analog, but this is not part of the current training recipe.)

the Kohonen SOM principle -- competitive learning with spatial structure -- could in principle be applied to todorov's weight initialization or early training. for example, initializing spike neurons with weight vectors sampled from a structured distribution (rather than random Gaussian) could create an initial "protomap" that training then refines. but there is no evidence that this would improve language modeling performance. the spatial structure of cortical maps is adapted to the spatial structure of sensory input; language lacks the topographic structure that makes SOMs useful for vision and somatosensation.

## challenges

the primary challenge is identifying which aspects of developmental self-organization are essential for the computational function and which are accidents of biological implementation. cortical topographic maps may exist because (a) they minimize wiring length between neurons that need to interact frequently (Chklovskii and Koulakov 2004), (b) they enable efficient lateral inhibition by keeping competitors spatially nearby, or (c) they are a developmental byproduct of the Hebbian + lateral inhibition mechanism with no intrinsic computational advantage. if (c), then importing topographic organization into artificial networks would add complexity without benefit.

a second challenge is the relationship between self-organization and gradient-based optimization. both are iterative refinement processes that improve a local objective (Hebbian co-activation vs loss function gradient), and both produce organized internal representations from unstructured initialization. the fundamental difference is that self-organization is unsupervised and local (each synapse adjusts based on local activity), while gradient descent is supervised and global (each parameter adjusts based on the global loss function). whether unsupervised local self-organization could complement global gradient-based optimization during training is an open question, but the computational overhead of maintaining spatial structure in the weight space is nontrivial.

## key references

- Kohonen, T. (1982). Self-organized formation of topologically correct feature maps. Biological Cybernetics, 43(1), 59-69.
- Sur, M. & Rubenstein, J. L. R. (2005). Patterning and plasticity of the cerebral cortex. Science, 310(5749), 805-810.
- Cang, J. et al. (2005). Development of precise maps in visual cortex requires patterned spontaneous activity in the retina. Neuron, 48(5), 797-809.
- Ge, X. et al. (2021). Retinal waves prime visual motion detection by simulating future optic flow. Science, 373(6553), 397-402.
- Katz, L. C. & Shatz, C. J. (1996). Synaptic activity and the construction of cortical circuits. Science, 274(5290), 1133-1138.
- Goodhill, G. J. (2007). Contributions of theoretical modeling to the understanding of neural map development. Neuron, 56(2), 301-311.
- Chklovskii, D. B. & Koulakov, A. A. (2004). Maps in the brain: what can we learn from them? Annual Review of Neuroscience, 27, 369-392.
- Hubel, D. H. & Wiesel, T. N. (1962). Receptive fields, binocular interaction and functional architecture in the cat's visual cortex. Journal of Physiology, 160(1), 106-154.
- Miller, K. D., Keller, J. B. & Stryker, M. P. (1989). Ocular dominance column development: analysis and simulation. Science, 245(4918), 605-615.
