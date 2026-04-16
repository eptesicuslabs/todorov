# synaptic pruning

status: definitional. last fact-checked 2026-04-16.

**why this matters**: the brain starts with massive overconnectivity and removes ~50% of synapses based on activity, which is the biological precedent for neural network pruning, the lottery ticket hypothesis, and sparse training methods that achieve full performance with a fraction of the parameters.

## summary

the developing brain starts with far more synapses than it will ultimately retain. during **postnatal development** (the period after birth), approximately 50% of synapses are eliminated through a process of **activity-dependent pruning**: active, functionally useful synapses are stabilized and maintained, while inactive or redundant synapses are weakened and removed.

ML analog: this is the biological version of the lottery ticket hypothesis (Frankle & Carlin 2019). the brain initializes a massively overparameterized network and then prunes it down to a sparse subnetwork that performs the actual computation. Huttenlocher (1979) first quantified this trajectory in human frontal cortex, showing that synaptic density peaks at age 1-2 years (approximately 50% above adult levels) and declines through childhood and adolescence to reach adult values by approximately age 16. the process is heterochronous: sensory cortices prune earlier than association cortices, with prefrontal cortex completing pruning last, in late adolescence (Huttenlocher and Dabholkar 1997).

the theoretical framework was established by Changeux and Danchin (1976) as **selective stabilization**: the genome specifies connections between classes of neurons (producing exuberant, redundant initial connectivity), and activity-dependent processes then select which connections survive. this is a Darwinian process applied to synapses -- variation followed by selection. it is distinct from **Hebbian learning** (the principle that neurons that fire together strengthen their connections), which strengthens existing connections without removing them.

## mechanism

### exuberant connectivity

the brain overproduces synapses during early postnatal development. the numbers are substantial:

- human visual cortex: synaptic density peaks at ~150% of adult levels by 8 months, reaching adult levels by 4-6 years (Huttenlocher et al. 1982)
- human auditory cortex: peak at ~3 months postnatal, adult levels by ~12 years (Huttenlocher and Dabholkar 1997)
- human prefrontal cortex: peak at ~15 months postnatal, adult levels not reached until ~16 years (Huttenlocher and Dabholkar 1997)
- total: the human brain may contain ~1000 trillion (10^15) synapses at peak, declining to ~500 trillion in adulthood

this initial overproduction is not random. synapses form in anatomically appropriate locations (guided by molecular cues: ephrins, semaphorins, netrins), but the precise pattern of connectivity within each region is unspecified. the overproduction creates a "search space" of possible circuits from which activity-dependent selection can operate.

### selective stabilization (Changeux and Danchin 1976)

the selective stabilization theory proposes three states for developing synapses:

1. **labile state:** newly formed synapses are unstable and can transition to either the stable or degenerate state depending on neural activity.
2. **stable state:** synapses that receive correlated pre- and postsynaptic activity are stabilized (retained). the mechanism is Hebbian: coincident activity strengthens the synapse and protects it from elimination.
3. **degenerate state:** synapses that fail to receive correlated activity degenerate and are physically removed.

the critical insight is that the final circuit is not specified by the genome. the genome specifies the initial distribution of labile synapses (the "envelope" of possible connectivity), and activity selects from within this envelope. different activity patterns during development will produce different final circuits from the same genetic program. this explains both the robustness of development (the envelope constrains possibilities) and its sensitivity to experience (activity determines which possibilities are realized).

### molecular mechanisms of elimination

synapse elimination involves several molecular pathways:

**complement-mediated pruning:** the classical **complement cascade** (C1q, C3 -- proteins of the innate immune system that tag targets for destruction) marks weak or inactive synapses for elimination by **microglia** (the brain's resident immune cells). Stevens et al. (2007) showed that C1q is expressed by developing retinogeniculate synapses and is required for normal synapse elimination in the visual system. microglia engulf and phagocytose complement-tagged synapses. this is a "tag and eat" mechanism: the molecular tag marks synapses for removal, and microglia perform the physical elimination.

ML analog: complement tagging is analogous to magnitude-based pruning in ML, where weights below a threshold are zeroed out. the key difference is that biological pruning is irreversible -- removed synapses are physically destroyed -- while ML pruning can be reversed by retraining.

**microglia-mediated phagocytosis:** Schafer et al. (2012) demonstrated that microglia actively engulf presynaptic terminals during the period of activity-dependent synapse elimination in the developing lateral geniculate nucleus. blocking microglial engulfment (via CR3/C3 knockout) prevents normal pruning, resulting in excess, functionally immature synapses.

**MHCI signaling:** major histocompatibility complex class I molecules, traditionally associated with immune function, are expressed at synapses and required for normal synapse elimination. MHCI knockout mice retain excess synapses in the retinogeniculate pathway (Huh et al. 2000).

**autophagy:** local autophagic degradation of synaptic components contributes to synapse elimination, particularly in the later stages of pruning. this pathway may be responsible for the removal of molecular machinery after the synapse has already been functionally weakened.

### activity dependence: "use it or lose it"

the central principle is that active synapses survive and inactive synapses are eliminated. the evidence:

- monocular deprivation during the [[critical_periods|critical period]] causes loss of deprived-eye synapses and expansion of open-eye synapses in V1
- blocking neural activity with TTX prevents normal pruning in the neuromuscular junction (Thompson et al. 1979)
- increasing activity (with benzodiazepines or BDNF overexpression) can accelerate pruning
- spontaneous activity (e.g., retinal waves) is sufficient to drive pruning even in the absence of sensory experience (see [[developmental_self_organization]])

the activity dependence is competitive: it is not absolute activity that determines survival, but relative activity compared to neighboring synapses. a synapse that is moderately active may be eliminated if its neighbors are highly active, because the competition for stabilization signals is local. this is why monocular deprivation is so effective: the deprived eye's synapses are not completely inactive, but they are less active than the open eye's synapses.

### the heterochronous timeline

pruning does not occur simultaneously across the brain. the developmental schedule reflects the functional hierarchy:

| region | peak synaptic density | pruning completion |
|---|---|---|
| visual cortex (V1) | ~8 months | ~4-6 years |
| auditory cortex (A1) | ~3 months | ~12 years |
| prefrontal cortex (PFC) | ~15 months | ~16 years |

primary sensory areas peak early and prune early, establishing stable sensory processing. association areas, particularly prefrontal cortex, peak later and prune throughout adolescence, consistent with the prolonged development of executive function, working memory, and social cognition.

this heterochronous schedule means that different brain regions have different [[critical_periods|critical periods]] for experience-dependent refinement. the late maturation of PFC has implications for adolescent behavior: the circuits supporting impulse control, planning, and risk assessment are not fully refined until the late teens or early twenties.

## relationship to neurodevelopmental disorders

the "too much or too little" pruning hypothesis links abnormal synaptic elimination to psychiatric disorders:

### autism spectrum disorder: too little pruning?

postmortem studies of autism brains show elevated synaptic density in several cortical regions compared to age-matched controls, particularly in temporal and prefrontal cortex (Hutsler and Zhang 2010). this is consistent with reduced pruning: the excess synapses that would normally be eliminated during development are retained. Tang et al. (2014) found increased spine density in layer V pyramidal neurons of ASD brains and linked this to reduced autophagy (mTOR pathway overactivation).

however, the "too little pruning" narrative is an oversimplification. some ASD studies show normal or even reduced spine density in specific regions (Hutsler & Zhang 2010). the pathology may be more accurately described as disrupted pruning (wrong synapses eliminated, or elimination at wrong time) rather than uniformly reduced pruning. the heterogeneity of ASD likely reflects heterogeneous synaptic pathology.

### schizophrenia: too much pruning?

the onset of schizophrenia in late adolescence coincides with the final phase of prefrontal cortex pruning. Feinberg (1982) proposed that excessive pruning in PFC could produce the cognitive symptoms of schizophrenia by eliminating too many functional synapses. postmortem studies consistently find reduced dendritic spine density in layer III of dorsolateral PFC in schizophrenia brains (Glantz and Lewis 2000).

Sekar et al. (2016) provided a molecular mechanism: schizophrenia-associated genetic variants in the complement component 4 (C4) locus increase C4A expression, leading to enhanced complement-mediated synaptic tagging and excessive microglial pruning during adolescence. this links the strongest genetic risk factor for schizophrenia to the molecular machinery of synapse elimination.

the timing is critical: the same pruning mechanism that is essential for normal PFC maturation becomes pathological when overactivated. this is not a failure of a different system -- it is a quantitative excess of the normal developmental process.

### caveats

these disorder associations remain hypotheses, not established mechanisms. the postmortem data are correlational, the sample sizes are small, and the causal direction is uncertain (do excess synapses cause autism, or does autism cause failure to prune?). animal models of complement overexpression or mTOR dysregulation show some phenotypic overlap with the human disorders but are not full models of the conditions.

## relationship to todorov

todorov does not implement any form of pruning during training. the architecture (number of parameters, connectivity pattern, layer structure) is fixed from initialization to the end of training. no connections are removed based on activity or magnitude.

this is the standard approach in language model training. pruning in ML is typically a post-training compression technique, not a developmental process. the [[development_vs_training|ML pruning literature]] (lottery ticket hypothesis, SparseGPT) focuses on removing parameters without degrading task performance, which is a different goal from developmental pruning (which actively shapes computation by removing specific connections).

the closest analog to biological pruning in todorov's framework is the ternary spike itself: by setting activations to zero (59% of dimensions at alpha=1.0), the spike function effectively "prunes" information flow through inactive channels on a per-token, per-layer basis. but this is transient (different tokens activate different channels) and does not permanently remove any connections. every weight remains available for every input.

whether todorov would benefit from a training-time pruning phase -- removing low-magnitude weights or consistently-inactive neurons after initial training -- is an empirical question. the lottery ticket hypothesis suggests that sparse subnetworks within dense networks can match full performance, but the evidence for language models is mixed. structured pruning (removing entire attention heads or MLP neurons) has shown more consistent results than unstructured pruning for transformers, but the savings at 300m parameters may not justify the complexity.

the more interesting biological parallel is not removing parameters but DEVELOPMENTAL TIMING. the brain prunes different regions on different schedules, creating a hierarchy of circuit maturity. todorov could potentially implement this by training different layer types (KDA vs MLA) for different numbers of steps, or by progressively adding layers during training. see [[development_to_training_curriculum]] for analysis.

## challenges

the pruning framework faces several unresolved issues. first, the causal direction between pruning and function is unclear. reduced spine density in schizophrenia brains could result from excessive pruning (Feinberg's hypothesis) or from reduced spine formation, or from neurodegeneration unrelated to developmental pruning. postmortem studies cannot distinguish these possibilities, and longitudinal imaging of synapse density in living human brains is not yet feasible.

second, the "too much / too little" narrative oversimplifies the pathology. the same disorder (autism) shows increased spine density in some regions and decreased density in others. schizophrenia affects different cortical layers differently. the framework needs regional and laminar specificity to make testable predictions, which current data rarely provide.

third, the relationship between synapse count and circuit function is not straightforward. removing 50% of synapses could improve or degrade computation depending on which synapses are removed. the field lacks a quantitative theory of how pruning specificity (which synapses survive) affects computational capacity. without this theory, predicting the functional consequences of altered pruning is not possible.

## key references

- Huttenlocher, P. R. (1979). Synaptic density in human frontal cortex -- developmental changes and effects of aging. Brain Research, 163(2), 195-205.
- Huttenlocher, P. R. & Dabholkar, A. S. (1997). Regional differences in synaptogenesis in human cerebral cortex. Journal of Comparative Neurology, 387(2), 167-178.
- Changeux, J.-P. & Danchin, A. (1976). Selective stabilisation of developing synapses as a mechanism for the specification of neuronal networks. Nature, 264(5588), 705-712.
- Stevens, B. et al. (2007). The classical complement cascade mediates CNS synapse elimination. Cell, 131(6), 1164-1178.
- Schafer, D. P. et al. (2012). Microglia sculpt postnatal neural circuits in an activity and complement-dependent manner. Neuron, 74(4), 691-705.
- Sekar, A. et al. (2016). Schizophrenia risk from complex variation of complement component 4. Nature, 530(7589), 177-183.
- Tang, G. et al. (2014). Loss of mTOR-dependent macroautophagy causes autistic-like synaptic pruning deficits. Neuron, 83(5), 1131-1143.
- Feinberg, I. (1982). Schizophrenia: caused by a fault in programmed synaptic elimination during adolescence? Journal of Psychiatric Research, 17(4), 319-334.
- Changeux, J.-P. (1985). Neuronal Man: The Biology of Mind. Oxford University Press.
