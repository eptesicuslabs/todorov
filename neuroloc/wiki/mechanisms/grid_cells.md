# grid cells

status: definitional. last fact-checked 2026-04-16.

**why this matters**: grid cells are the brain's coordinate system, encoding position through periodic hexagonal patterns that are mathematically optimal for path integration -- a finding that connects directly to positional encodings in transformers and recurrent state updates in SSMs.

## summary

**grid cells** are neurons in the **medial entorhinal cortex** (MEC -- a cortical region adjacent to the hippocampus that provides spatial input to hippocampal circuits) that fire in a remarkably regular hexagonal lattice pattern as an animal moves through space. discovered by Hafting, Fyhn, Molden, Moser, and Moser (2005), each grid cell has multiple firing fields arranged at the vertices of an equilateral triangular grid tiling the entire available environment. a grid cell is characterized by three properties: **spacing** (distance between adjacent firing fields, ranging from ~30 cm to ~300 cm), **orientation** (the angle of the grid axes relative to the environment, with approximately 3 discrete orientations per module), and **phase** (the spatial offset of the grid relative to a reference point, unique to each cell).

ML analog: grid cell modules are analogous to sinusoidal positional encodings in transformers. both use periodic functions at multiple frequencies to encode position, and both achieve exponential capacity from a linear number of components. the key difference is that grid cells use 2D hexagonal periodicity while transformer positional encodings use 1D sinusoidal periodicity. grid cells are believed to provide the brain's metric for space -- a coordinate system for path integration and spatial memory.

the discovery of grid cells earned May-Britt Moser and Edvard Moser the 2014 Nobel Prize in Physiology or Medicine, shared with John O'Keefe (who discovered place cells; see [[place_cells]]).

## discovery

the Mosers and their students were recording from the MEC while rats foraged in open arenas. earlier work (Fyhn et al. 2004) had shown that MEC neurons have spatial selectivity, but the recordings were in small boxes that obscured the full pattern. when they expanded the arena to a large open field (1-2 m^2), the hexagonal structure became unmistakable: each cell fired at multiple discrete locations, and these locations formed a strikingly regular triangular lattice (Hafting et al. 2005).

the regularity of the pattern was extraordinary. biological firing patterns are typically noisy and irregular, but grid cells produced firing fields with a periodicity and geometric precision comparable to a crystal lattice. the hexagonal symmetry (6-fold rotational symmetry) was consistent across all grid cells recorded, though different cells had different spacings, orientations, and phases.

## grid cell properties

### spacing

the distance between adjacent firing fields of a single grid cell. spacing ranges from approximately 30 cm (dorsal MEC) to approximately 300 cm or more (ventral MEC). this gradient parallels the dorso-ventral gradient of place field size in the hippocampus: dorsal cells provide fine-grained spatial resolution, ventral cells provide coarse coverage.

### orientation

the angle of the grid axes relative to some reference direction (typically the walls of the environment). within a module, all grid cells share the same orientation and spacing. across modules, orientations cluster around approximately 3 discrete values, separated by ~7.5 degrees (not uniformly distributed).

### phase

the spatial offset of the grid pattern. each grid cell within a module has a unique phase (different spatial offset), so the population of cells within a module tiles all positions. phase is the only property that distinguishes individual cells within a module.

## modular organization

grid cells are organized into discrete modules -- groups of cells sharing the same spacing and orientation but differing in phase. key properties of the modular organization:

### discrete spacing ratios

the spacings of successive modules increase in a geometric progression, with a ratio between adjacent modules of approximately 1.4-1.7 (Barry, Hayman, Burgess & Jeffery 2007; Stensola et al. 2012). the exact ratio is debated, but most studies converge on values near 1.4-1.5, close to sqrt(2).

this geometric scaling has information-theoretic consequences. if the brain reads out position by combining grid cell activity across modules, the resolution of the combined code scales exponentially with the number of modules (like digits in a number system). with n modules, the theoretical capacity is approximately (L/s_min)^n positions, where L is the environment size and s_min is the smallest grid spacing. this is a remarkably efficient encoding: ~4-5 modules can encode millions of distinct positions.

### independence

modules are functionally independent: they remap independently (rescale, rotate, or shift independently of each other in response to environmental changes), suggesting they operate as parallel, loosely coupled systems. Stensola et al. (2012) showed that individual modules maintain their internal coherence during partial remapping, while the relationship between modules can change.

### count

estimates of the number of modules range from 4-5 (in rat MEC), based on clustering of grid spacings along the dorso-ventral axis.

## proposed function: path integration

the most widely accepted computational role of grid cells is path integration (dead reckoning): computing an estimate of the animal's position by integrating self-motion signals over time. see [[path_integration]] for a full treatment.

the argument: grid cells receive velocity information (from head direction cells, speed cells, and vestibular input), and use it to update a running estimate of position. the hexagonal firing pattern is the stable state of the neural network that performs this integration. position is encoded in the pattern of active grid cells across modules, and is updated by translating the activity pattern in response to movement.

evidence for path integration in grid cells:

- grid cell firing persists in darkness: when visual landmarks are removed, grid cells continue to fire in a regular pattern, drifting slowly from the light-on positions. this demonstrates that grid cells can maintain a spatial representation based on self-motion alone, without visual input
- the pattern degrades over time in darkness, accumulating integration error (~1-2 cm per meter traveled; Hardcastle et al. 2015). this error accumulation is characteristic of path integration
- grid cells are modulated by running speed and head direction, the two signals needed for path integration

## mathematical models

two major classes of model explain how hexagonal grid patterns arise:

### continuous attractor networks (CAN)

the neural sheet of the MEC is modeled as a **continuous attractor network** (CAN -- a recurrent network whose stable states form a continuous manifold rather than discrete fixed points) in which local excitation and surround inhibition produce a stable hexagonal pattern of activity bumps (Fuhs & Touretzky 2006, Burak & Fiete 2009, Guanella, Kiper & Verschure 2007).

ML analog: continuous attractor networks are the biological analog of state-space models (SSMs) like Mamba. both maintain a continuous internal state that evolves through recurrent dynamics, and both accumulate information over time without discrete memory slots.

the mechanism:
1. neurons in MEC are connected with center-excitation, surround-inhibition connectivity (Mexican hat or similar profile)
2. this connectivity spontaneously generates a stable periodic pattern of activity: bumps of high firing separated by troughs of low firing, arranged in a hexagonal lattice (hexagonal packing minimizes energy in 2D surround-inhibition systems)
3. velocity inputs shift the activity pattern across the neural sheet. the direction and speed of the shift match the animal's movement, implementing path integration
4. each neuron's place in the activity pattern determines its grid phase; the pattern's spatial period determines the grid spacing

the CAN model explains:
- why the hexagonal pattern is universal (it is the energy-minimizing solution)
- why all cells within a module share spacing and orientation (they are part of one coherent pattern)
- how path integration works (velocity-driven translation of the bump pattern)

the CAN model predicts toroidal topology of the population activity manifold -- the activity state lives on a torus. this was confirmed experimentally by Gardner et al. (2022) using topological data analysis.

### oscillatory interference models (OI)

proposed by Burgess, Barry, and O'Keefe (2007): grid cell firing arises from the interference of **velocity-controlled oscillators** (VCOs -- neural oscillators whose frequency shifts in proportion to the animal's velocity along a preferred direction).

the mechanism:
1. each grid cell receives input from several oscillators (typically 3, at 120 degree separations), each with a frequency that increases linearly with the component of velocity along a preferred direction
2. a baseline oscillator provides a reference frequency (the theta rhythm)
3. each VCO's phase relative to the baseline encodes displacement along its preferred direction: phase = (2*pi/s) * d, where s is the grid spacing and d is displacement
4. the grid cell fires when all VCOs are in phase (constructive interference), which occurs at the vertices of a hexagonal lattice (because the 3 preferred directions are separated by 120 degrees)

the OI model explains:
- why spacing is determined by the frequency difference between VCO and baseline (spacing = v / delta_f)
- why theta oscillation is necessary (it provides the reference frequency)
- why grid cells show theta modulation (they inherit it from the VCOs)
- the dorso-ventral gradient (different baseline frequencies produce different spacings)

the OI model has a critical weakness: phase noise in biological oscillators accumulates over time, degrading the grid pattern. the VCOs must be periodically reset by external landmarks or by the attractor dynamics of the CAN model, leading to hybrid models.

### Lie group models

Gao et al. (2021) analyzed grid cell path integration through the lens of Lie groups and Lie algebras. the recurrent transformation in the neural network implements a matrix Lie group, and the algebra of the transformation dictates the resulting spatial patterns. for the simplest case (linear transformation), the Lie group is a rotation group, and isotropic scaling conditions naturally produce hexagonal patterns. this connects grid cells to group theory and abstract algebra.

Dorrell et al. (2022) extended this with the "actionable representations" framework: optimal representations of 2D space under biological constraints (non-negative firing, bounded activity, precise coding) are necessarily hexagonal grid modules. the argument uses group and representation theory to show that grid cells are the mathematically optimal solution, not just one possible solution.

## relationship to other spatial cell types

grid cells interact with several other cell types in the MEC and hippocampus:

- **head direction cells**: fire when the animal faces a specific direction. many MEC cells are conjunctive grid x head direction cells, firing only when the animal is at a grid vertex AND facing a particular direction. these conjunctive cells are proposed to carry the velocity signal needed for path integration
- **speed cells**: Kropff et al. (2015) identified MEC neurons whose firing rate increases linearly with running speed. speed cells provide the magnitude component of the velocity signal for path integration
- **border cells**: fire along environmental boundaries (walls, edges). border cells may serve as landmarks that reset path integration error
- **place cells**: grid cells project to the hippocampus and are believed to generate place cell firing through the superposition of grid inputs at multiple scales (see [[place_cells]]). the relationship is bidirectional: hippocampal feedback may stabilize grid cell patterns

## grid-like representations beyond space

the grid cell system may be a general computational mechanism for organizing relational knowledge, not just spatial maps:

- Constantinescu, O'Reilly, and Behrens (2016) showed that humans navigating a 2D conceptual space (bird morphology) exhibited hexagonal modulation of fMRI signal in entorhinal cortex, identical in form to spatial grid cell signals
- Aronov, Neber, and Tank (2017) found that rats navigating a 1D auditory frequency space showed grid-like periodic firing in MEC
- the Tolman-Eichenbaum Machine (Whittington et al. 2020; see [[cognitive_maps]]) formalizes this: grid cells encode the structural relationships in any relational graph, with spatial navigation as a special case

this generalization is important for the connection to todorov: if grid cells compute general relational structure (not just spatial position), then the algebraic operations of PGA might be relevant to this broader computation, not just to spatial navigation. see [[pga_vs_grid_cells]] and [[spatial_computation_to_pga]].

## relationship to todorov

todorov's G(3,0,1) PGA module computes geometric products between learned linear projections of the input. the question of whether this has any mathematical connection to grid cell computation is addressed in [[pga_vs_grid_cells]].

key observations:
- grid cells use hexagonal periodicity; PGA operations are not periodic
- grid cells integrate velocity over time; PGA computes a self-interaction (no temporal integration)
- grid cells operate in 2D; PGA operates in 3D+1 projective space
- but: both involve group-theoretic structure (grid cells as Lie group representations, PGA as Clifford algebra), and both may serve as "structural priors" for relational computation

see [[pga_vs_grid_cells]] for the full adversarial comparison.

## challenges

the grid cell framework has several open questions. first, the functional necessity of the hexagonal pattern is debated. Dorrell et al. (2022) argue it is mathematically optimal under biological constraints, but other periodic patterns (square, triangular) could also support path integration. whether hexagonal grids actually improve navigation performance over alternative patterns has not been experimentally tested by selectively disrupting grid symmetry while preserving periodicity.

second, the CAN and oscillatory interference models make different predictions about the cellular mechanism, but experimental evidence has not decisively distinguished them. the Gardner et al. (2022) toroidal topology finding supports CAN models, but does not rule out oscillatory mechanisms operating at the single-cell level within a CAN framework. hybrid models may be necessary.

third, the extension of grid cells to non-spatial domains (conceptual spaces, social hierarchies) is suggestive but the evidence is mostly from fMRI with limited spatial resolution. single-unit recordings in humans navigating abstract spaces are needed to confirm that the same cells that fire in hexagonal patterns during physical navigation also fire periodically during abstract reasoning.

## key references

- Hafting, T., Fyhn, M., Molden, S., Moser, M.-B. & Moser, E. I. (2005). Microstructure of a spatial map in the entorhinal cortex. Nature, 436(7052), 801-806.
- Fyhn, M., Molden, S., Witter, M. P., Moser, E. I. & Moser, M.-B. (2004). Spatial representation in the entorhinal cortex. Science, 305(5688), 1258-1264.
- Stensola, H. et al. (2012). The entorhinal grid map is discretized. Nature, 492(7427), 72-78.
- Burak, Y. & Fiete, I. R. (2009). Accurate path integration in continuous attractor network models of grid cells. PLoS Computational Biology, 5(2), e1000291.
- Burgess, N., Barry, C. & O'Keefe, J. (2007). An oscillatory interference model of grid cell firing. Hippocampus, 17(9), 801-812.
- Kropff, E., Carmichael, J. E., Moser, M.-B. & Moser, E. I. (2015). Speed cells in the medial entorhinal cortex. Nature, 523(7561), 419-424.
- Constantinescu, A. O., O'Reilly, J. X. & Behrens, T. E. J. (2016). Organizing conceptual knowledge in humans with a gridlike code. Science, 352(6292), 1464-1468.
- Gardner, R. J. et al. (2022). Toroidal topology of population activity in grid cells. Nature, 602(7895), 123-128.
- Gao, P. et al. (2021). On path integration of grid cells: group representation and isotropic scaling. NeurIPS.
- Dorrell, W., Latham, P. E., Behrens, T. E. J. & Whittington, J. C. R. (2022). Actionable neural representations: grid cells from minimal constraints. ICLR.

## see also

- [[place_cells]]
- [[path_integration]]
- [[cognitive_maps]]
- [[hippocampal_memory]]
- [[theta_oscillations]]
- [[pga_vs_grid_cells]]
- [[spatial_computation_to_pga]]
