# energy efficient coding

status: definitional. last fact-checked 2026-04-16.

**why this matters**: the tradeoff between information throughput and energy cost explains why biological networks use sparse codes, and provides the theoretical foundation for sparse activation functions in ML architectures that achieve higher performance per FLOP.

## origin

the [[efficient_coding]] hypothesis (barlow, 1961) focuses on maximizing information transmission per neural response. but there is a harder constraint than **channel capacity** (the maximum rate at which information can be reliably transmitted): energy. niven and laughlin (2008) argued that energy limitation is THE selective pressure that shaped the evolution of sensory systems -- not merely one constraint among many, but the dominant force.

the brain's signaling apparatus -- ion channels, synaptic vesicles, neurotransmitter receptors -- is expensive to build and expensive to operate. every spike consumes ATP. every maintained synapse leaks ions. every bit of information transmitted has a metabolic price. evolution has relentlessly minimized this price while preserving the information needed for survival.

## the levy and baxter information-energy tradeoff

levy and baxter (1996) formalized the tradeoff between information transmission and energy consumption. for a neuron that can fire or remain silent, the **information per spike** (bits of data conveyed by a single action potential) and the energy per spike jointly determine the optimal firing rate.

ML analog: this is the biological version of the bits-per-FLOP efficiency metric. just as ML practitioners optimize for performance per unit compute, evolution optimizes for information per unit ATP.

the key insight: maximum information transmission (**entropy maximization** -- the state where every possible output is equally likely, yielding maximum bits) occurs when a neuron fires with probability p = 0.5 (50% of the time). but this is metabolically ruinous. each spike costs ATP to restore ion gradients, while silence costs only the resting potential maintenance. the ratio of these costs -- the spike-to-rest energy ratio -- determines the optimal operating point.

levy and baxter showed that maximizing information per unit energy (bits per ATP, or equivalently bits per joule) yields an optimal firing probability:

    p_optimal = 1 / (1 + exp(E_spike / E_rest))

where E_spike is the energy cost of a spike and E_rest is the energy cost of resting for the same duration. for cortical neurons where a spike costs ~10-100x more energy than resting, this gives p_optimal in the range of 1-10%.

their specific estimate: ~6% optimal activity for a **population code** (a representation distributed across many neurons, where meaning is encoded by the pattern of active units), when considering both the energy cost per spike and the information carried per spike. this is remarkably close to observed cortical firing rates: most cortical neurons have mean firing rates of 1-5 Hz, with **population sparseness** (the fraction of neurons active at any given moment) of 2-10%.

ML analog: the ~6% optimal firing rate maps directly to the ~40% firing rate of ternary spike activations in todorov. the biological optimum is sparser because biological spikes cost ~10-100x more energy per spike than digital activations cost per FLOP.

## the niven and laughlin framework

niven and laughlin (2008) extended the energy-information tradeoff from single neurons to entire sensory systems, arguing that energy limitation acts as a selective pressure at every level:

### the fundamental constraint

nervous systems face two conflicting pressures:
1. minimize energy consumption (metabolic constraint)
2. generate adaptive behavior under fluctuating environmental conditions (information constraint)

the tradeoff between these pressures determines the structure, size, and coding strategy of sensory systems.

### energy costs scale nonlinearly with performance

two factors make high-performance signaling disproportionately expensive:

**noise**: reducing noise in a neural signal requires energy. the **signal-to-noise ratio** (SNR -- the ratio of meaningful signal power to background noise power) of a neural signal scales with the number of ion channels, which scales with membrane area, which scales with energy consumption. doubling the SNR approximately quadruples the energy cost (because SNR scales as sqrt(N) for N channels, so N must increase as SNR^2).

ML analog: this is analogous to the precision-compute tradeoff in quantized inference. moving from INT8 to FP32 quadruples the bits per weight (reducing "noise") but increases energy per operation by ~10-20x.

**speed**: faster neural signaling requires more ion channels with faster kinetics, which leak more current at rest and consume more energy per spike. the bandwidth-energy tradeoff is steep: doubling temporal bandwidth approximately doubles the resting energy cost (due to increased leak currents).

this creates a law of diminishing returns: each additional bit of information or increment of speed costs disproportionately more energy. neurons face severe penalties for excess functional capacity.

### evidence from fly photoreceptors

the most quantitative evidence comes from blowfly and drosophila photoreceptors:

- blowfly (Calliphora) photoreceptors transmit ~5x more information than drosophila photoreceptors
- but they consume ~10x more energy per bit
- the larger photoreceptors are less energy-efficient despite higher information rates
- the unit cost of information (ATP molecules per bit) increases with photoreceptor size

this demonstrates the diminishing returns: larger, higher-bandwidth photoreceptors pay a disproportionate energy price for each additional bit of information. evolution has tuned photoreceptor size to balance information needs against metabolic budgets for each species' ecological niche.

### cross-species evidence

the energy-information tradeoff explains morphological variation across species:

- nocturnal insects have larger photoreceptors (more photon capture, but higher energy cost per bit)
- diurnal insects have smaller photoreceptors (lower energy cost, adequate SNR in bright light)
- fast-flying insects invest more energy in high-bandwidth motion detection
- slow-moving species can afford lower temporal bandwidth, saving energy

the compound eye itself is an energy optimization: the number and size of ommatidia, the photoreceptor diameter, and the neural processing in the lamina all reflect energy-constrained information tradeoffs.

## why different brain areas have different firing rates

the energy-information framework explains regional variation in neural firing rates:

- **primary sensory cortex**: higher firing rates (5-20 Hz for responsive neurons) justified by high information content of sensory inputs. the marginal bit is worth the energy cost.
- **prefrontal cortex**: lower firing rates (~1-5 Hz baseline) because the information processed is more abstract and requires less temporal bandwidth.
- **hippocampal CA1**: very sparse firing (~1-2% place cell activation) because positional information is low-dimensional and only needs a few active neurons.
- **cerebellar granule cells**: extremely sparse (~1-5% active) because the energy budget per granule cell is tiny (small soma, few synapses, short axon) and the population is enormous (~50 billion granule cells).

each region operates at a firing rate that approximately maximizes information per unit energy given its specific computational demands and metabolic supply.

## the information-energy curve

the information rate I as a function of energy expenditure E follows a concave curve with diminishing returns:

    I(E) ~ log(1 + E/E_noise)

where E_noise is the energy cost of noise (background ion channel fluctuations, synaptic noise). at low energy expenditure, each additional joule buys substantial information. at high energy expenditure, the returns diminish because the signal is already well above the noise floor.

the slope of this curve at the operating point determines the evolutionary pressure:
- steep slope: strong pressure to increase energy expenditure (more information per additional joule)
- shallow slope: strong pressure to decrease energy expenditure (each additional joule buys little)

cortical neurons generally operate where the slope is moderate -- not at the maximum information rate (too expensive) and not at the minimum (too little information). this balance point is the energy-efficient code.

## relationship to sparse coding

the energy-efficient coding framework provides the deepest justification for [[sparse_coding]]:

1. each spike costs ~10-100x more energy than resting
2. the information per spike decreases as firing rate increases (redundancy increases)
3. the optimal strategy is to fire rarely but carry maximum information per spike
4. this produces sparse codes with ~1-10% population activity

the observed sparsity levels in cortex (2-10%) are quantitatively consistent with energy-efficient coding predictions. the biological brain is not sparse by design preference -- it is sparse because physics demands it.

ML analog: ReLU achieves ~50% sparsity, top-k gating in mixture-of-experts achieves ~90%+ sparsity, and ternary spikes achieve ~60% sparsity. all of these reduce compute cost per forward pass, recapitulating the biological principle that sparse codes are cheaper.

see [[brain_energy_budget]] for the quantitative energy breakdown and [[metabolic_constraints_on_computation]] for the implications at the cortical computation level.

## challenges

the energy-efficient coding framework has several weaknesses. first, the theory assumes that evolution has had time to reach the global optimum, but neural circuits may be stuck at local optima due to phylogenetic constraints. the mammalian cortex inherits its basic architecture from early amniotes, and the energy-optimal design for a mammalian cortex starting from scratch might differ substantially from what evolution could reach through incremental modification.

second, the framework treats energy as the sole selective pressure, but other constraints (wiring volume, conduction delay, developmental robustness, evolvability) also shape neural coding. a code that is suboptimal for energy efficiency might be optimal when wiring length or fault tolerance is also considered. disentangling these pressures experimentally is difficult.

third, most quantitative predictions come from invertebrate photoreceptors (flies), where the biophysics is well characterized. extending these predictions to mammalian cortical circuits involves assumptions about ion channel density, membrane properties, and metabolic supply that are much harder to verify. the theory's quantitative accuracy in cortex remains less well tested than in peripheral sensory systems.

## key references

- levy, w. b. & baxter, r. a. (1996). energy efficient neural codes. neural computation, 8(3), 531-543.
- niven, j. e. & laughlin, s. b. (2008). energy limitation as a selective pressure on the evolution of sensory systems. journal of experimental biology, 211, 1792-1804.
- laughlin, s. b. (2001). energy as a constraint on the coding and processing of sensory information. current opinion in neurobiology, 11, 475-480.
- laughlin, s. b. & sejnowski, t. j. (2003). communication in neuronal networks. science, 301, 1870-1874.
- niven, j. e., anderson, j. c. & laughlin, s. b. (2007). fly photoreceptors demonstrate energy-information trade-offs in neural coding. PLoS biology, 5(4), e116.
- barlow, h. b. (1961). possible principles underlying the transformations of sensory messages. in sensory communication, MIT press.

## see also

- [[efficient_coding]]
- [[brain_energy_budget]]
- [[metabolic_constraints_on_computation]]
- [[sparse_coding]]
- [[niven_laughlin]]
