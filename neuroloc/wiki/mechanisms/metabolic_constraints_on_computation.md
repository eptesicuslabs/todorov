# metabolic constraints on computation

**why this matters**: the hard physical limit of fewer than 1% simultaneously active neurons is the deepest justification for sparse activations in ML, and the brain-vs-silicon energy comparison sets the benchmark that neuromorphic hardware and efficient inference engines aim to beat.

## the lennie constraint

peter lennie (2003) asked a simple question: given the brain's energy supply and the cost of each spike, how many neurons can be active simultaneously? the answer is devastating: fewer than 1%.

ML analog: this is the biological version of the "memory wall" in ML inference. just as GPU memory bandwidth limits how many parameters can be active per forward pass, metabolic supply limits how many neurons can fire per timestep.

the calculation proceeds as follows:

1. the human cerebral cortex contains ~1.6 x 10^10 neurons
2. the cortex receives ~6 W of metabolic power (from blood glucose oxidation)
3. ~25% goes to housekeeping (protein synthesis, membrane maintenance), leaving ~4.5 W for signaling
4. a single action potential in a cortical pyramidal neuron costs ~2.4 x 10^9 ATP molecules (lennie's estimate)
5. at ~5 x 10^-20 J per ATP: one spike costs ~1.2 x 10^-10 J
6. for a neuron firing at 10 Hz: ~1.2 x 10^-9 W per active neuron
7. maximum active neurons: ~4.5 W / 1.2 x 10^-9 W = ~3.75 x 10^9 = ~23% of cortical neurons at 10 Hz

but this is for 10 Hz firing. cortical pyramidal neurons in primary sensory areas fire at 20-50 Hz when active, and the cost includes not just the action potential but all postsynaptic consequences:
- EPSC energy across ~8,000 synapses per neuron
- postsynaptic action potentials in target neurons
- neurotransmitter recycling

lennie's more careful estimate, accounting for the full cascade of downstream costs, yields ~1% as the upper bound on simultaneously active neurons. this is not a theoretical preference -- it is a hard physical constraint imposed by the metabolic supply of the cortex.

## implications for neural representations

the 1% constraint has profound implications:

### sparse representations are mandatory

if only 1% of neurons can be active, then every representation must be sparse. this is not an optimization choice that evolution happened to make -- it is the only possibility given the energy supply. a dense code where 50% of neurons fire would require ~50x more metabolic power than the cortex receives.

the implication for artificial neural networks: dense activations (typical transformer hidden states with ~100% nonzero elements) are metabolically impossible in biological tissue. the brain's hardware cannot support them. this is the deepest reason why biological neural computation is sparse.

ML analog: dense activations in transformers work because silicon has no metabolic constraint -- every MAC costs the same energy regardless of activation value. sparse activations (ReLU, top-k, ternary spikes) voluntarily impose the constraint that biology enforces physically, trading representational capacity for computational efficiency.

### flexible energy allocation

lennie noted that the brain must allocate its limited energy flexibly among cortical regions according to task demands. attention is, from this perspective, an energy routing mechanism: when you attend to a visual stimulus, the energy budget shifts from other cortical areas to visual cortex, allowing a higher fraction of V1 neurons to be active.

this explains several phenomena:
- **attentional suppression**: unattended stimuli evoke weaker neural responses (fewer active neurons, less energy allocated)
- **cognitive fatigue**: sustained attention depletes local metabolic reserves
- **capacity limits**: you cannot simultaneously maintain high neural activity in many cortical areas

### the metabolic cost of information

combining lennie (2003) with the energy-efficient coding framework (levy and baxter 1996, niven and laughlin 2008):

- the cortex can sustain ~10^8-10^9 spikes per second total
- each spike carries at most ~5-10 bits of information (considering timing, identity, and rate)
- total cortical information capacity: ~10^9-10^10 bits/s = ~1-10 Gbit/s

this is a strikingly low number. a modern GPU memory bus carries ~1 Tbit/s -- 100x more. the brain compensates through efficient coding: each spike carries high-information content because it is selected from a sparse population, and the identity of the active neuron itself carries information.

## comparison with silicon: energy per operation

the brain and silicon processors solve the same problem -- computation -- with radically different energy budgets.

### brain energy efficiency

- energy per synapse per spike: ~10 fJ (harris et al. 2012)
- energy per spike per neuron: ~19 pJ (attwell and laughlin 2001) to ~120 pJ (lennie 2003)
- total power: ~20 W for ~86 billion neurons with ~150 trillion synapses

### silicon energy efficiency (45nm, Horowitz 2014)

- FP32 multiply: ~3.7 pJ
- FP32 add: ~0.9 pJ
- **FP32 MAC** (multiply-accumulate -- a single multiply followed by an add, the fundamental operation of neural network inference): ~4.6 pJ
- INT8 multiply: ~0.2 pJ
- INT8 add: ~0.03 pJ
- 8 KB SRAM read: ~5 pJ
- 32 KB SRAM read: ~10 pJ
- 1 MB SRAM read: ~20 pJ
- DRAM read: ~640 pJ (1.3-2.6 nJ for 64 bits at 45nm)

### the comparison problem

direct comparison is misleading for several reasons:

1. **the brain's synapse is not a MAC**. a biological synapse performs vesicle release, receptor binding, ion channel gating, and gradient restoration -- a complex electrochemical process. a silicon MAC is a pure arithmetic operation. they are not equivalent units of computation.

2. **the brain's energy budget includes communication**. axonal transmission accounts for a large fraction of neural energy. silicon data movement (SRAM/DRAM access) also dominates energy, but the architectures are not comparable.

3. **scale matters**. the brain runs 86 billion neurons at 20 W. modern GPUs run ~10^12 operations per second at 300-700 W. the per-operation comparison is less meaningful than the system-level comparison for equivalent computational tasks.

### moore's law vs metabolic stasis

silicon has experienced exponential improvement in energy per operation:
- 1971 (10 um): ~10 nJ per logic operation
- 2000 (130 nm): ~100 pJ per logic operation
- 2014 (45 nm): ~1-5 pJ per MAC (Horowitz)
- 2024 (5 nm): ~0.1-0.3 pJ per FP32 MAC
- 2025 (3 nm): ~0.05-0.2 pJ per FP32 MAC

the brain's energy budget has not changed. the cost per synapse per spike has been ~10 fJ for ~300 million years of mammalian evolution. evolution optimized the constant factor but could not change the fundamental physics of electrochemical signaling.

ML analog: this asymmetry -- silicon improving exponentially while biology stays flat -- is why neuromorphic computing has become less compelling over time. the original motivation (match biological efficiency) is losing force as silicon surpasses it at the per-operation level.

silicon is now approaching and in some cases surpassing biological energy efficiency at the per-operation level. the Landauer limit (k_B * T * ln(2) = ~3 x 10^-21 J at room temperature) sets the absolute floor. both biology and silicon are still far above it, but silicon is closing the gap faster because transistor scaling reduces energy while electrochemistry cannot.

see [[biological_vs_silicon_energy]] for a detailed process-node comparison.

## challenges

the brain-vs-silicon comparison has three fundamental problems. first, the "synapse = MAC" equivalence is misleading. a biological synapse performs probabilistic vesicle release, receptor binding, ion flow, and gradient restoration -- a stochastic, energy-dissipating process. a silicon MAC is a deterministic arithmetic operation. comparing their energy costs implies they do equivalent work, but they do not.

second, the Landauer limit argument overstates how close either system is to fundamental bounds. both biology (~10 fJ/synapse) and silicon (~0.1 pJ/MAC) are 3-6 orders of magnitude above the Landauer limit (~3 x 10^-21 J). the practical constraints are thermal management and noise margins, not fundamental physics. neither system is remotely near the theoretical floor.

third, the system-level comparison (20 W brain vs 300+ W GPU) is confounded by task mismatch. the brain performs perception, motor control, memory, emotion, and homeostasis simultaneously on 20 W. a GPU performs matrix multiplication on 300 W. a fair comparison would require both systems performing the same task at the same accuracy, which no experiment has achieved.

## key references

- lennie, p. (2003). the cost of cortical computation. current biology, 13, 493-497.
- attwell, d. & laughlin, s. b. (2001). an energy budget for signaling in the grey matter of the brain. journal of cerebral blood flow and metabolism, 21, 1133-1145.
- harris, j. j., jolivet, r. & attwell, d. (2012). synaptic energy use and supply. neuron, 75(5), 762-777.
- horowitz, m. (2014). computing's energy problem (and what we can do about it). ISSCC digest of technical papers, 10-14.
- levy, w. b. & baxter, r. a. (1996). energy efficient neural codes. neural computation, 8(3), 531-543.
- niven, j. e. & laughlin, s. b. (2008). energy limitation as a selective pressure on the evolution of sensory systems. journal of experimental biology, 211, 1792-1804.

## see also

- [[brain_energy_budget]]
- [[energy_efficient_coding]]
- [[sparse_coding]]
- [[biological_vs_silicon_energy]]
- [[attwell_laughlin]]
