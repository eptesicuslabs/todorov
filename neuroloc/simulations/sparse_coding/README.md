# sparse coding simulation

## what this does

sparse_coding_demo.py builds a three-layer spiking network in Brian2:

1. **input layer** (50 neurons): driven by external current matching a binary pattern. active pattern neurons receive strong drive; inactive neurons receive near-threshold noise.

2. **encoding layer** (200 neurons): receives weighted input from the input layer through random projections. sparsity is controlled by the firing threshold -- higher threshold means fewer encoding neurons fire, producing a sparser code.

3. **readout layer** (50 neurons): reconstructs the original pattern from the encoding layer activity through random decoding weights.

the simulation sweeps sparsity from 5% to 80% by varying the encoding layer's firing threshold. for each sparsity level, 8 random binary patterns are presented, and two metrics are computed:

- **reconstruction error (MSE)**: how well the sparse code preserves the input pattern when decoded
- **information preservation (approximate MI)**: how much of the input's information structure is retained in the encoding

## outputs

- `sparsity_vs_reconstruction.png`: reconstruction error as a function of sparsity level, with annotations for todorov's 41% firing rate and the cortical 1-10% range
- `sparsity_vs_information.png`: information content as a function of sparsity level, with the same annotations

## how to run

```
pip install brian2 numpy matplotlib
cd <project_root>
python neuroloc/simulations/sparse_coding/sparse_coding_demo.py
```

output images are saved to the same directory as the script.

## what to look for

the key question: is there a phase transition or sharp tradeoff at a particular sparsity level?

- at very low sparsity (5-10%), the encoding layer has too few active neurons to faithfully represent the input. reconstruction error may be high, but the code is maximally efficient (fewest spikes per pattern).
- at moderate sparsity (20-40%), reconstruction improves as more encoding neurons participate. this is the regime where todorov operates.
- at high sparsity (50-80%), the code becomes dense and reconstruction is best, but the code is metabolically expensive and loses the combinatorial capacity advantage of sparse representations.

the biological claim is that cortex operates at 1-10% because the metabolic cost of spikes dominates. todorov operates at 41% because the gradient flow cost of dead neurons dominates. both are constrained optima -- different constraints produce different operating points.

## limitations

- the network uses random (untrained) weights. biological sparse codes are learned. the reconstruction quality here reflects the random projection baseline, not what a trained sparse coder can achieve.
- the encoding layer does not implement k-WTA or lateral inhibition. sparsity is controlled by threshold alone. biological sparse codes use competitive dynamics.
- the MI estimate is approximate (correlation-based proxy). true MI estimation in spiking networks is an open research problem.
- the network is feedforward. biological sparse coding involves recurrent dynamics that refine the representation over multiple cycles.

## connection to todorov

see [[sparse_coding_to_ternary_spikes]] for the full bridge analysis. the simulation demonstrates the sparsity-reconstruction tradeoff that motivates todorov's 41% operating point: enough active neurons for gradient flow and information preservation, at the cost of operating far from cortical sparsity levels.
