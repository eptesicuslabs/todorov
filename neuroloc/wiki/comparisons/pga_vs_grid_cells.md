# PGA vs grid cells

status: current (as of 2026-04-16).

## framing

todorov's architecture includes a G(3,0,1) projective geometric algebra (PGA) module that computes a geometric product self-interaction in the SwiGLU feedforward layers. grid cells in the medial entorhinal cortex compute periodic spatial representations for path integration. this comparison asks: is there a meaningful mathematical connection between these two systems, or are they superficially linked by the word "geometry"?

this question matters because if PGA and grid cells share deep mathematical structure, it could motivate biologically informed modifications to the PGA module. if they do not, it constrains claims about the biological inspiration of the spatial module.

## dimension-by-dimension comparison

### mathematical structure

**grid cells**: the firing pattern of a grid cell can be described as the sum of three cosines at 120-degree separations: f(x) = sum_{k=1}^{3} cos(2*pi * u_k^T * x / s + phi_k), where u_k are unit vectors at 0, 60, and 120 degrees, s is the spacing, and phi_k are phase offsets. the resulting function has 6-fold (hexagonal) rotational symmetry. grid cell modules form a discrete set of representations at geometrically scaled spacings (~1.4x ratio between successive modules). the mathematical framework is Lie group theory: the recurrent transformation implements a matrix Lie group, and the isotropic scaling conditions on the Lie algebra produce hexagonal symmetry (Gao et al. 2021).

**PGA**: G(3,0,1) is a 16-dimensional Clifford algebra over 3D projective space. the metric is e1^2 = e2^2 = e3^2 = 1, e0^2 = 0 (degenerate basis vector for projective coordinate). the geometric product combines inner and outer products across all grades (0-4). the algebra can represent: rotations (even-grade rotors), translations (via projective point operations using e0), reflections (grade-1 elements), and general rigid body transformations (sandwich product R*x*R_reverse). the full multiplication table has 192 non-zero entries in the Cayley table.

**comparison**: both involve algebraic structure on spaces. grid cells use Abelian group representations on R^2 (translations form an Abelian group). PGA encodes the full Euclidean group E(3) of rigid body transformations in 3D. PGA is strictly more general: the 2D translation group relevant to grid cells is a subgroup of E(3). but generality is not the same as relevance -- PGA's extra structure (3D rotations, reflections, the projective coordinate e0) has no obvious counterpart in the grid cell system.

**verdict**: WEAK connection. they share group-theoretic roots, but grid cells use a simple Abelian group (2D translations), while PGA encodes a much larger non-Abelian group (3D Euclidean group). the shared mathematics is at the level of "both involve groups," which is too generic to be meaningful.

### periodicity

**grid cells**: fundamentally periodic. the hexagonal lattice is a 2D periodic structure with period s along each of the three grid axes. the periodicity is the defining feature -- it is what makes grid cells "grid" cells. the modular organization (discrete spacings at ~1.4x ratios) is a hierarchy of periodicities.

**PGA**: not periodic. the geometric product is a bilinear map that produces a single output for given inputs. there is no inherent periodicity in the algebra. rotors can express periodic rotations (R(theta) has period 2*pi in angle), but the GP self-interaction in todorov does not compute a rotation -- it computes a general bilinear product between two learned projections of x. nothing cycles.

**verdict**: NO connection. periodicity is the central property of grid cells and is entirely absent from PGA operations. this is a fundamental mismatch.

### dimensionality

**grid cells**: operate in 2D physical space (the plane of the animal's movement). head direction adds a 1D angular variable. the total state space for path integration is SE(2): the group of rigid body motions in 2D (2D position + 1D heading).

**PGA**: G(3,0,1) operates in 3D projective space (3D position + 1D projective coordinate). the algebra has 16 components spanning grades 0-4. in todorov, the inputs are d_model-dimensional token embeddings projected to 16-dimensional multivectors, with no spatial semantics attached to any dimension.

**verdict**: MISMATCH. grid cells are 2D; PGA is 3D+projective. the dimensionality is wrong for a direct correspondence. the relevant algebra for 2D grid cell path integration would be G(2,0,1) (6 components) or the complex numbers (2 components), not G(3,0,1) (16 components). the extra dimensions in PGA are wasted from a grid cell perspective.

### learned vs hardwired structure

**grid cells**: the hexagonal pattern is NOT learned from data in the standard sense. it emerges from the network structure (attractor dynamics) or the biophysical properties (oscillator interference) of the MEC. the spacing gradient along the dorso-ventral axis is anatomically determined. grid cells appear in the first few minutes of exploration in naive animals and in animals raised without visual experience (Langston et al. 2010, Wills et al. 2010). they are not trained on a spatial task -- they are a structural prior on the representation of space.

**PGA**: in todorov, the W_left (d_model -> 16) and W_right (d_model -> 16) projections are learned parameters. the GP computes a fixed algebraic operation (the Cayley table), but the inputs to the GP are entirely determined by gradient descent on the language modeling objective. the network decides what "e1" and "e01" mean in terms of the input. there is no hardwired spatial structure.

**verdict**: FUNDAMENTAL DIFFERENCE. grid cells are a structural prior that emerges from network topology. PGA in todorov is a structured nonlinear self-interaction whose inputs are learned from data. the algebra is fixed but the mapping from input space to the algebra is unconstrained.

### computational role

**grid cells**: encode position through path integration. the computation is temporal: integrate velocity over time to update position. the output is a continuously updated estimate of the animal's location. the computation is recurrent: grid cell activity at time t depends on grid cell activity at time t-1 plus the velocity input at time t.

**PGA in todorov**: computes a self-interaction: GP(W_left(x), W_right(x)). the computation is instantaneous: it transforms a single token embedding x into a residual that is added to the SwiGLU output. there is no temporal integration, no recurrence, no velocity input. the GP does not accumulate information over time.

**verdict**: FUNDAMENTAL DIFFERENCE. grid cells compute a temporal integral (path integration). PGA computes an instantaneous bilinear function (self-interaction). these are categorically different operations. the only way to bridge them would be to use PGA within a recurrent loop (e.g., as part of KDA's state update), which is not what todorov does.

## the strongest argument FOR a connection

the best case for a mathematical connection runs through group theory and the Lie group perspective (Gao et al. 2021):

1. grid cells implement a representation of the 2D translation group. the hexagonal pattern is the optimal representation under biological constraints (Dorrell et al. 2022)
2. PGA encodes the 3D Euclidean group, which includes 2D translations as a subgroup
3. both systems compute group operations: grid cells compose translations (path integration), PGA composes rotations, translations, and reflections (geometric product)
4. the GP self-interaction could in principle learn to compute translation-like operations if gradient descent finds them useful for the task
5. Whittington et al. (2020) showed that grid-like representations emerge when a model needs to track relational structure, even in non-spatial domains. todorov processes sequences of tokens that have relational structure (syntax, semantics). the PGA module could discover grid-like representations for these abstract relational structures

this argument is coherent but speculative. it has not been tested. no one has examined the 16-component multivector outputs of the trained PGA module to see if they exhibit periodicity, hexagonal structure, or group-theoretic regularity. the MI improvement (1.311 with GP vs baseline) could be from any sufficiently expressive nonlinear self-interaction.

## the strongest argument AGAINST a connection

1. **periodicity is essential, not optional**: grid cells work BECAUSE of their periodicity. the periodic structure enables exponential coding capacity across modules, error correction through inter-module consistency, and metric computation through phase differences. PGA has no periodicity. removing periodicity from grid cells would destroy their function. this is not a surface difference -- it is the core computational mechanism
2. **dimensionality mismatch**: grid cells solve a 2D problem. PGA is built for 3D+projective. using G(3,0,1) for a 2D spatial problem is like using a sledgehammer to swat a fly -- it works, but the extra structure is irrelevant. and todorov does not even pose a spatial problem
3. **self-interaction vs integration**: the GP in todorov takes x and produces f(x). path integration takes x_t, v_t and produces x_{t+1}. one is a function, the other is a recurrence. these are categorically different computations. you cannot perform path integration with an instantaneous self-interaction
4. **language is not space**: todorov processes text, not spatial environments. the token embedding space has no Euclidean metric, no continuous paths, no velocity, no heading direction. the "positions" in a language model are discrete token indices, not coordinates in a metric space. the group-theoretic structure of spatial navigation (translations, rotations) has no clear counterpart in language processing
5. **the null hypothesis is sufficient**: the MI improvement with GP active (1.311 vs baseline) could be explained by the GP being a sufficiently expressive bilinear self-interaction that injects higher-order feature interactions into the residual stream. ANY structured bilinear map (not just the geometric product) might produce similar improvements. without a control comparing GP to a random bilinear operation with the same parameter count, the claim that the GEOMETRIC structure matters is unsupported

## the critical missing experiment

to resolve this debate, one experiment is needed:

**replace the G(3,0,1) geometric product with a random bilinear operation of the same dimensionality (16 x 16 -> 16) and measure MI, CKA, and BPB**.

if the random bilinear operation produces the same MI improvement as GP, then the geometric structure of PGA is irrelevant -- the benefit comes from generic bilinear interaction, and the connection to grid cells is vacuous.

if GP outperforms the random bilinear operation, then the specific algebraic structure of PGA contributes something beyond generic bilinear mixing, and the connection to spatial computation deserves further investigation.

this experiment has not been run. until it is, the connection between PGA and grid cells remains an untested hypothesis.

## verdict

**the mathematical connection between G(3,0,1) PGA and grid cell computation is WEAK to NONEXISTENT at the implementation level, and SPECULATIVE at the theoretical level.**

the implementations share no computational features: PGA is an instantaneous, 3D projective, aperiodic self-interaction. grid cells are a temporal, 2D periodic, velocity-integrating recurrence. the only shared structure is that both involve groups -- a connection so generic that it links almost any algebraic system to almost any other.

the theoretical connection through Lie group representations and the Dorrell et al. optimality argument is more interesting but entirely speculative. it would require: (a) that language processing benefits from group-theoretic structure, (b) that the specific structure of G(3,0,1) is the right one for this benefit, and (c) that the GP self-interaction discovers this structure through gradient descent. none of these have been demonstrated.

the honest position: PGA in todorov is a geometrically structured nonlinear self-interaction that produces a measurable spike MI improvement at zero compute cost. it may work because of the geometric structure, or it may work because of the generic bilinear interaction. the connection to grid cells is a suggestive narrative, not a scientific claim.

**dissenting argument**: the above verdict demands too much specificity. biology uses group-theoretic representations of symmetry (grid cells for spatial symmetry, head direction cells for rotational symmetry) because they are computationally efficient for encoding transformations. PGA encodes a rich group (the Euclidean group) that subsumes spatial symmetries. even if todorov does not do path integration, the algebraic structure of PGA provides a mathematical scaffold for learning relational transformations -- the same abstract function that cognitive maps serve (Whittington et al. 2020). the GP self-interaction could act as a structural prior that makes certain relational computations easier to learn, analogous to how weight sharing in CNNs provides a translation equivariance prior without the network being "about" translation. this is consistent with the MI improvement and the zero-overhead result: the GP provides a useful inductive bias from its algebraic structure, regardless of whether that structure matches grid cells specifically. the relevant comparison is not "PGA vs grid cells" but "structured bilinear vs unstructured bilinear," and this comparison has not been made.

## key references

- Hafting, T. et al. (2005). Microstructure of a spatial map in the entorhinal cortex. Nature, 436(7052), 801-806.
- Gao, P. et al. (2021). On path integration of grid cells: group representation and isotropic scaling. NeurIPS.
- Dorrell, W. et al. (2022). Actionable neural representations: grid cells from minimal constraints. ICLR.
- Whittington, J. C. R. et al. (2020). The Tolman-Eichenbaum Machine. Cell, 183(5), 1249-1263.
- Krausse, S. (2025). A grid cell-inspired structured vector algebra for cognitive maps. arXiv:2503.08608.

## see also

- [[grid_cells]]
- [[place_cells]]
- [[path_integration]]
- [[cognitive_maps]]
- [[spatial_computation_to_pga]]
