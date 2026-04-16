# bridge: spatial computation to G(3,0,1) PGA

status: current (as of 2026-04-16).

## the biological mechanism

the brain's spatial computation system comprises four interlocking cell types in the hippocampal-entorhinal circuit:

- **place cells** (hippocampus CA1/CA3): fire at specific locations, forming a distributed spatial map. see [[place_cells]]
- **grid cells** (medial entorhinal cortex): fire in hexagonal lattice patterns, providing a metric coordinate system. see [[grid_cells]]
- **head direction cells** (anterodorsal thalamus, postsubiculum): fire when the animal faces a specific direction, providing an angular reference. see [[path_integration]]
- **speed cells** (medial entorhinal cortex): fire linearly with running speed, providing the magnitude of the velocity vector. see [[path_integration]]

these cell types collectively implement path integration: x_{t+1} = x_t + v_t * dt. the system operates in SE(2), the group of rigid body motions in 2D (2D position + 1D heading). grid cells encode position using periodic representations that are mathematically optimal under biological constraints (Dorrell et al. 2022). the system extends beyond spatial navigation to abstract relational computation: cognitive maps encode the relational structure of any domain that can be represented as a graph of states and transitions (see [[cognitive_maps]]).

the mathematical framework connecting these computations is group theory: grid cells implement representations of the 2D translation group (Gao et al. 2021), head direction cells implement representations of SO(2) (the rotation group), and the combined system operates in SE(2). the Tolman-Eichenbaum Machine (Whittington et al. 2020) shows that these representations emerge from learning relational structure, with spatial firing patterns as a special case of general relational coding.

## the current todorov implementation

PGA in SwiGLU spatial mode (src/layers/swiglu.py + src/algebra/geometric_product.py):

the computation:

    left = Multivector(W_left(x))         # d_model -> 16 components
    right = Multivector(W_right(x))       # d_model -> 16 components
    product = geometric_product(left, right)  # 16 x 16 -> 16 via Cayley table
    residual = W_gp(product.values)       # 16 -> d_model
    output = swiglu_output + residual     # additive after down projection

properties of this implementation:

1. **self-interaction**: both inputs (left, right) come from the same token embedding x. this means the GP computes a function of x, not an interaction between different inputs (e.g., x and a velocity signal, or x and a context signal)

2. **full algebra**: the geometric product computes ALL grades (0-4) of the G(3,0,1) algebra via the sparse Cayley table (192 non-zero entries). the 16 output components are: 1 scalar (grade 0), 4 vectors (grade 1: e0, e1, e2, e3), 6 bivectors (grade 2: e01, e02, e03, e12, e13, e23), 4 trivectors (grade 3: e012, e013, e023, e123), 1 pseudoscalar (grade 4: e0123)

3. **learned projections**: W_left and W_right are unconstrained linear projections. gradient descent determines what the 16 components "mean" in terms of the input -- there is no hardwired spatial semantics. the network could use the algebra as a structured bilinear mixing layer without any geometric interpretation

4. **additive residual**: the GP output is added to the SwiGLU output (after down projection), not multiplied or concatenated. this means it acts as a correction or enrichment of the feedforward representation

5. **every spatial layer**: applied at every SwiGLU layer where spatial_mode=True. the GP self-interaction is computed at every position in the sequence independently (no cross-position interaction)

what PGA can express:
- **rotations**: even-grade multivectors (scalar + bivector + pseudoscalar) form the rotor group. the sandwich product R*x*R_reverse applies a rotation. todorov does not use the sandwich product in the SwiGLU pathway -- it uses the raw geometric product
- **translations**: in PGA, translations are expressed via e0 (the degenerate basis vector): T = 1 + (1/2)(d*e0), where d is a displacement bivector. the sandwich product T*x*T_reverse translates projective points
- **reflections**: grade-1 multivectors (vectors) represent reflections. the sandwich product v*x*v_reverse reflects x through the hyperplane normal to v
- **combined rigid body transformations**: composing rotors and translators via the geometric product. this is the core power of PGA: rigid body motions compose algebraically

but in todorov:
- the GP is between two LEARNED LINEAR PROJECTIONS of x
- there is no explicit spatial meaning to the 16 components
- the W_gp projection (16 -> d_model) can discard any geometric structure that is not useful
- the network can use the algebraic structure however gradient descent finds useful
- validated: MI all-time high 1.311 with GP active, zero compute overhead
- never validated: whether the geometric structure specifically (vs generic bilinear interaction) drives the improvement

## what spatial computation suggests about PGA

### the group theory connection

spatial computation operates in SE(2) or SE(3). PGA encodes E(3) (the full Euclidean group including reflections). these share a mathematical language: both are about composing transformations of space.

grid cells implement representations of the translation subgroup. head direction cells implement representations of SO(2). PGA can represent all of these and more. the algebraic structure of PGA is a superset of the algebraic structure of the spatial navigation system.

the Dorrell et al. (2022) result shows that optimal representations of space under biological constraints are periodic (hexagonal). PGA does not produce periodic representations. this means PGA is not computing grid-cell-like representations, even though it can express the underlying transformations.

the Krausse (2025) GC-VSA work explicitly connects grid cell modules to algebraic operations (circular convolution as binding, element-wise addition as bundling), but uses block codes (3D tensors), not Clifford algebra. this is a different algebraic framework that is closer to grid cells but farther from PGA.

### what the GP self-interaction actually computes

the GP between two multivectors a and b produces: ab = a dot b + a wedge b (inner + outer products, mixed across all grades by the Cayley table). for the self-interaction case (both from the same x):

- **grade 0 (scalar)**: contains inner products between same-grade components of left and right. since both come from x, this computes a quadratic form of x
- **grade 2 (bivectors)**: contains outer products between grade-1 components. since the 4 grade-1 components are linear projections of x, the bivectors compute 6 pairwise antisymmetric interactions: e_i*x * e_j*x - e_j*x * e_i*x. these are area-like quantities
- **grade 1 and grade 3**: contain mixed products involving the degenerate e0. because e0^2 = 0, products involving e0 annihilate certain terms, creating a sparse interaction pattern
- **grade 4 (pseudoscalar)**: a single value proportional to the determinant-like quantity of the grade-1 components

the self-interaction is effectively a structured quadratic/bilinear function of x, where the structure is dictated by the Cayley table of G(3,0,1). it computes 16 specific bilinear combinations of the 16 input components, with the signs and targets determined by the algebra.

this is more structured than a random bilinear operation (which would have 16^3 = 4096 free parameters for the trilinear tensor) but less structured than a rotation or translation (which would constrain the 16 components to lie in a specific subspace). the GP self-interaction occupies a middle ground: it is a structured but unconstrained bilinear mixing.

## the proposed change

### option 1: do nothing (recommended)

the PGA module works (MI 1.311, zero overhead). the connection to grid cells is speculative and lacks experimental support. the critical missing experiment (GP vs random bilinear, see [[pga_vs_grid_cells]]) should be run before any modification is considered.

### option 2: grade-constrained GP (exploratory, phase 6+)

constrain the GP output to specific grades that correspond to spatial operations:
- **even grades only (0, 2, 4)**: restricts to the rotor subgroup (rotations + translations in PGA). this would bias the GP toward rigid body transformations, which are the operations underlying path integration
- **grade 1 only**: restricts to reflections. unlikely to be useful for language modeling
- **grade 2 only (bivectors)**: the "area" elements. in PGA, bivectors represent lines (the join of two points). this is the grade that encodes the structure of 3D space most directly

implementation: add a grade mask to the GP output before the W_gp projection. grade_mask = tensor of 0s and 1s over the 16 components. trivial to implement.

risk: grade restriction removes degrees of freedom that gradient descent may need. the improvement from GP could be from the full-grade interaction, not from any specific geometric subspace. restricting grades could reduce performance.

expected impact: uncertain. could help if the network is currently "wasting" capacity on grades that are not useful. could hurt if the network uses all grades for non-geometric purposes.

### option 3: recurrent GP (exploratory, phase 6+)

use the GP not as a self-interaction but as a state update operator within KDA:

    S_{t+1} = GP(S_t_projected, v_t_projected) + alpha * S_t

where S_t is projected to a multivector, v_t is a "velocity" signal derived from the input, and the GP computes the transformation of the state. this would make the GP operate more like path integration: accumulating transformations over time.

implementation: requires changing the KDA state update formula and adding multivector projections. non-trivial. would need to handle the interaction between the 16-component multivector and the head_dim x head_dim state matrix.

risk: HIGH. this changes the core KDA computation and could destabilize training. the state matrix is not a geometric object -- it stores associative bindings, not positions. forcing it into a multivector representation may be counterproductive.

expected impact: uncertain. if language processing benefits from group-theoretic state transformations (analogous to cognitive maps navigating abstract relational spaces), this could be powerful. if the analogy does not hold, it will degrade performance.

### option 4: periodic GP output (exploratory, phase 6+)

add a periodic nonlinearity to the GP output before the W_gp projection:

    gp_output = cos(2*pi * GP(left, right) / s)

where s is a learned scale parameter. this would inject the periodicity that grid cells have and PGA lacks.

implementation: trivial (one line).

risk: LOW (additive residual, can learn to ignore). but the motivation is weak: why should language representations be periodic? grid cells are periodic because space is periodic under the hexagonal lattice. token space is not periodic.

expected impact: likely neutral or slightly negative. periodicity is a strong structural prior that may not match the statistics of language.

## recommendation

**do nothing** (option 1). the PGA module works. the connection to grid cells is not strong enough to motivate modifications. the critical experiment (GP vs random bilinear) should be run before any biological inspiration is applied. all other options are exploratory and deferred to phase 6+.

if the GP vs random bilinear experiment shows that the geometric structure of PGA matters (GP >> random bilinear at matched parameters), then option 2 (grade-constrained GP) becomes the first investigation: which grades drive the improvement? the answer would reveal whether the network uses PGA as a geometric operator or as a generic bilinear mixer.

if the GP vs random bilinear experiment shows that the geometric structure does NOT matter (GP ≈ random bilinear), then the connection to grid cells is vacuous, and the PGA module should be understood as a structured bilinear self-interaction with no spatial significance.

## risk assessment

| modification | implementation | training risk | expected BPB impact | justification strength |
|---|---|---|---|---|
| do nothing | none | none | 0 | strong (working system) |
| grade-constrained GP | trivial | low | +/- 0-2% | weak (untested hypothesis) |
| recurrent GP in KDA | hard | high | unknown | speculative (cognitive map analogy) |
| periodic GP output | trivial | low | likely 0 | very weak (periodicity mismatch) |

## key references

- Hafting, T. et al. (2005). Microstructure of a spatial map in the entorhinal cortex. Nature, 436(7052), 801-806.
- Gao, P. et al. (2021). On path integration of grid cells: group representation and isotropic scaling. NeurIPS.
- Dorrell, W. et al. (2022). Actionable neural representations: grid cells from minimal constraints. ICLR.
- Whittington, J. C. R. et al. (2020). The Tolman-Eichenbaum Machine. Cell, 183(5), 1249-1263.
- Krausse, S. (2025). A grid cell-inspired structured vector algebra for cognitive maps. arXiv:2503.08608.
- Banino, A. et al. (2018). Vector-based navigation using grid-like representations in artificial agents. Nature, 557(7705), 429-433.

## see also

- [[grid_cells]]
- [[place_cells]]
- [[path_integration]]
- [[cognitive_maps]]
- [[pga_vs_grid_cells]]
- [[memory_systems_to_kda_mla]]
