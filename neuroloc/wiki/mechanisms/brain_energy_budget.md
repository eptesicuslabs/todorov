# brain energy budget

status: definitional. last fact-checked 2026-04-16.

**why this matters**: the brain's energy budget is the ultimate constraint on neural network density and firing rates, and directly motivates why sparse activations outperform dense ones in biologically-inspired ML architectures.

## overview

the human brain consumes ~20 W of power -- about 20% of the body's **resting metabolic rate** (total energy expenditure at rest) -- despite representing only ~2% of body mass. this disproportionate energy demand reflects the high cost of neural signaling. attwell and laughlin (2001) quantified where this energy goes, establishing the first detailed energy budget for signaling in the **grey matter** (cortical tissue containing neuronal cell bodies and synapses, as opposed to white matter which is primarily myelinated axons).

the brain runs on glucose. **oxidative phosphorylation** (the mitochondrial process that converts glucose into usable energy) of one glucose molecule yields ~30 **ATP** (adenosine triphosphate, the cell's primary energy currency). the total ATP turnover in the brain is approximately 4.7 kg of ATP per day. most of this energy is spent on ion pumping -- restoring the ionic gradients that neural signaling dissipates.

## the attwell and laughlin (2001) energy budget

using anatomic and physiologic data from rodent cortex, attwell and laughlin estimated the energy expenditure on different components of excitatory signaling. the original breakdown:

- **action potential** (a brief electrical spike that propagates along a neuron's axon) propagation: ~47% of signaling energy
- **postsynaptic potentials** (EPSCs via **glutamate receptors** -- ion channels activated by the neurotransmitter glutamate): ~34%
- resting potential maintenance (Na+/K+-ATPase leak current): ~13%
- presynaptic Ca2+ entry and transmitter release: ~5%
- glutamate recycling: ~1%

the dominant cost is the action potential. propagating a spike through a cortical neuron with its full **axonal arbor** (the branching tree of output fibers) requires restoring the Na+ and K+ gradients across the membrane. attwell and laughlin estimated this at ~3.84 x 10^8 ATP molecules per action potential for a typical **cortical pyramidal neuron** (the primary excitatory cell type in cerebral cortex).

the second largest cost is postsynaptic. each glutamatergic synapse activation triggers ion flow through AMPA and NMDA receptors, requiring ATP to restore gradients. with ~8,000 synapses per pyramidal neuron and ~0.5 vesicles released per synapse per spike (release probability ~0.5), the total postsynaptic cost per spike is substantial.

## revised budget: howarth, gleeson, and attwell (2012)

the original attwell and laughlin (2001) estimate used a **Na+ entry ratio** (actual sodium influx divided by the theoretical minimum needed to charge the membrane) of 4, derived from squid axon data. subsequent experimental work in mammalian neurons found much lower Na+ entry ratios:

- cerebellar granule cells: 1.04 (nearly perfectly efficient)
- Purkinje cells: ~2.0
- cortical pyramidal neurons: ~1.3

the revised budget (howarth et al. 2012) shifts the balance significantly:

- postsynaptic glutamate receptors: ~50% of signaling energy
- action potentials: ~21%
- resting potentials: ~20%
- presynaptic transmitter release: ~5%
- neurotransmitter recycling: ~4%

the key revision: mammalian action potentials are much more energy-efficient than previously thought. the Na+ and K+ currents overlap less in time than in squid axons, reducing wasteful simultaneous cross-membrane ion flow. this shifts the energy burden from action potentials to **synaptic transmission** (the process of chemical signaling across the gap between neurons), making synapses the dominant energy consumer.

## energy per action potential

the cost per action potential depends on axon length, diameter, myelination, branching, and the Na+ entry ratio. estimates:

- attwell and laughlin (2001): ~3.84 x 10^8 ATP per spike per cortical neuron
- lennie (2003): ~2.4 x 10^9 ATP per spike (includes all downstream costs)
- harris et al. (2012): revised downward due to lower Na+ entry ratios

converting ATP to joules: each ATP hydrolysis yields ~5 x 10^-20 J of useful free energy (under cellular conditions, ~50 kJ/mol = ~8.3 x 10^-20 J, but the actual work extracted is lower). using the attwell and laughlin estimate:

    3.84 x 10^8 ATP * 5 x 10^-20 J/ATP = ~1.9 x 10^-11 J = ~19 pJ per spike

for cortical pyramidal neurons with typical axonal arbors.

## energy per synapse per spike

the cost of a single synaptic transmission event (one vesicle release and its postsynaptic effects):

- attwell and laughlin (2001): ~1.64 x 10^5 ATP per vesicle release (including presynaptic Ca2+ entry, vesicle recycling, and postsynaptic receptor currents)
- harris et al. (2012): revised to account for updated Na+ entry ratios and more detailed synaptic models

converting: 1.64 x 10^5 ATP * 5 x 10^-20 J/ATP = ~8.2 x 10^-15 J = ~8.2 fJ per synapse per spike

this number -- on the order of 10 fJ per synaptic event -- is the fundamental energy unit of biological neural computation. it sets the target that **neuromorphic hardware** (chips designed to mimic neural architecture) aims to match.

ML analog: the biological cost of ~10 fJ per synaptic event compares to ~0.1-5 pJ per FP32 MAC operation on modern GPUs. biological synapses are ~10-500x more energy-efficient per operation, though the operations are not directly comparable.

## total brain energy partitioning

the brain's total energy is not all spent on signaling. a significant fraction goes to non-signaling processes:

- housekeeping (protein synthesis, lipid turnover, organelle transport): ~25% (engl and attwell, 2015)
- signaling (action potentials + synaptic transmission + resting potentials): ~75%

within the signaling budget, the harris et al. (2012) revised breakdown (for grey matter) allocates:
- synaptic transmission: ~43%
- action potential propagation: ~17%
- resting potentials (neuronal + glial): ~15%

a separate analysis by levy & calvert (2021) found that communication (axonal action potentials, synaptic transmission) consumes 35x more energy than computation (**postsynaptic integration** -- the summation of inputs at the cell body) in human cortex. the cortex allocates ~71% of its ATP to communication, ~2% to computation, and ~27% to maintenance and growth.

ML analog: this 35:1 communication-to-computation ratio mirrors the "memory wall" in modern ML hardware, where data movement (DRAM reads, inter-chip communication) dominates energy over arithmetic.

## why sparsity is metabolically necessary

the energy budget constrains the number of simultaneously active neurons. lennie (2003) estimated that given the cortex's energy supply and the cost per spike, fewer than 1% of cortical neurons can be substantially active at any time.

the calculation:
- human cortex: ~1.6 x 10^10 neurons
- available power for signaling: ~3-5 W
- cost per spike: ~2.4 x 10^9 ATP = ~1.2 x 10^-10 J (lennie's estimate)
- at 1 Hz average firing: each active neuron costs ~1.2 x 10^-10 W
- maximum simultaneously active neurons: ~3 W / 1.2 x 10^-10 W = ~2.5 x 10^10

but this naive calculation allows all neurons to fire at 1 Hz. at realistic mean firing rates for active neurons (5-20 Hz for cortical pyramidal cells), the fraction that can be active drops to ~1-5%.

this is not a design choice. it is a physical constraint. the brain cannot afford to have more than a few percent of neurons active simultaneously without exceeding its metabolic supply. sparse coding is not an optimization -- it is a survival requirement.

ML analog: this is the biological justification for sparse activation functions like ReLU (which zeros ~50% of activations) and ternary spikes (which zero ~60%). dense activations are metabolically impossible in biology and computationally wasteful in silicon.

## regional variation in energy budget

different brain areas have different energy budgets per neuron, reflecting different computational demands:

- cerebral cortex: high energy per neuron, low firing rates, sparse coding
- cerebellum: lower energy per neuron (granule cells are tiny, have few synapses, Na+ entry ratio ~1.04), very sparse coding in the granule cell layer (~1-5% active)
- retina: high energy density (photoreceptors consume ~75% of retinal oxygen), continuous signaling through graded potentials rather than spikes

the energy budget varies even within cortex. primary sensory areas have higher metabolic rates than association cortex, reflecting higher average firing rates during stimulation.

## challenges

the energy budget framework faces several limitations. first, most quantitative estimates derive from rodent cortex and are extrapolated to humans. human cortical neurons differ from rodent neurons in size, myelination, and dendritic complexity, so the per-spike and per-synapse energy costs may differ substantially. direct measurement of energy consumption per spike in human neurons in vivo remains technically infeasible.

second, the budget treats energy allocation as static, but the brain dynamically redistributes blood flow and glucose delivery based on activity (neurovascular coupling). the assumption that the total energy supply is fixed ignores the brain's ability to temporarily increase local metabolic supply by ~30-50% through hemodynamic responses. this means the "hard constraint" of 1% simultaneously active neurons is softer than presented -- local activity bursts can exceed the average budget for short periods.

third, the budget focuses almost exclusively on grey matter signaling and underestimates the energy cost of white matter transmission. long-range axonal propagation through myelinated fibers consumes energy for saltatory conduction and node of Ranvier repolarization, but these costs are poorly quantified and typically excluded from the signaling budget.

## key references

- attwell, d. & laughlin, s. b. (2001). an energy budget for signaling in the grey matter of the brain. journal of cerebral blood flow and metabolism, 21, 1133-1145.
- howarth, c., gleeson, p. & attwell, d. (2012). updated energy budgets for neural computation in the neocortex and cerebellum. journal of cerebral blood flow and metabolism, 32, 1222-1232.
- harris, j. j., jolivet, r. & attwell, d. (2012). synaptic energy use and supply. neuron, 75(5), 762-777.
- lennie, p. (2003). the cost of cortical computation. current biology, 13, 493-497.
- engl, e. & attwell, d. (2015). non-signalling energy use in the brain. journal of physiology, 593(16), 3417-3429.
- levy, w. b. & calvert, v. g. (2021). communication consumes 35 times more energy than computation in the human cortex, but both costs are needed to predict synapse number. PNAS, 118(18), e2008173118.

## see also

- [[efficient_coding]]
- [[energy_efficient_coding]]
- [[metabolic_constraints_on_computation]]
- [[sparse_coding]]
- [[attwell_laughlin]]
