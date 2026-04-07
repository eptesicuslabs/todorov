# spatial simulations

## grid_cell_model.py

oscillatory interference model of grid cells demonstrating hexagonal firing patterns.

### model

a virtual agent performs a random walk in a 2.0m x 2.0m arena for 600 seconds at 0.3 m/s. grid cell firing is computed using the oscillatory interference mechanism: each cell receives input from 3 velocity-controlled oscillators (VCOs) at preferred directions 0, 120, and 240 degrees. the cell fires when the summed cosine activation across all 3 VCOs exceeds a threshold (constructive interference at hexagonal lattice vertices).

three grid modules with geometrically scaled spacings:
- module 1: s = 0.40 m (3 cells with different spatial phases)
- module 2: s = 0.56 m (3 cells, ratio 1.4x from module 1)
- module 3: s = 0.78 m (3 cells, ratio 1.4x from module 2)

the spacing ratio (~1.4x) matches the experimentally observed geometric scaling of grid cell modules in the medial entorhinal cortex (Stensola et al. 2012).

### activation function

for each cell at position (x, y) with spacing s and phase offset (px, py):

    activation = (1/3) * sum_{k=1}^{3} cos(2*pi * ((x-px)*cos(theta_k) + (y-py)*sin(theta_k)) / s)

where theta_k = {0, 2*pi/3, 4*pi/3} are the preferred directions. the cell fires probabilistically when activation exceeds threshold 0.7. this produces constructive interference at the vertices of an equilateral triangular grid.

### output

- grid_cell_firing.png: 3x3 grid showing spike locations (red dots) overlaid on trajectory (grey) for each cell (rows = modules, columns = phase offsets within module)
- grid_cell_rate_maps.png: 3x3 grid showing smoothed firing rate maps (spikes/occupancy) with hot colormap, revealing the hexagonal firing pattern
- grid_cell_autocorrelation.png: 3 panels showing spatial autocorrelation of rate maps for the first cell of each module, demonstrating hexagonal symmetry as 6-fold symmetric peaks around the central peak

### dependencies

- brian2 (imported but not used for this model -- included for consistency with simulation framework)
- numpy
- matplotlib
- scipy (gaussian_filter, correlate2d)

### run

    python grid_cell_model.py

### expected results

- clear hexagonal (triangular lattice) firing patterns visible in rate maps
- spacing increases across modules (~0.40, 0.56, 0.78 m)
- spatial autocorrelation shows 6-fold symmetric peaks at the grid spacing distance
- different phase offsets within a module tile different spatial positions
- rate maps should show the characteristic "regular dot pattern" of grid cells

### relevance to todorov

the oscillatory interference model demonstrates that hexagonal grid patterns emerge from the constructive interference of periodic signals at 120-degree separations. this periodicity is the defining property of grid cells and is entirely absent from todorov's G(3,0,1) PGA self-interaction, which computes an instantaneous bilinear product with no periodic structure. see [[pga_vs_grid_cells]] for the full adversarial comparison of PGA and grid cell computation, and [[spatial_computation_to_pga]] for the bridge analysis.
