# neurogenesis and plasticity research

curated peer-reviewed research on adult neurogenesis, structural plasticity, and critical periods. the brain is not a fixed circuit -- it adds new neurons throughout life (at least in the hippocampus), continuously remodels its synaptic connections (5-15% spine turnover per day), and transitions through developmental windows of heightened plasticity that can be reopened pharmacologically. these findings define the boundaries of what is structurally possible in biological neural computation and constrain claims about fixed architectures.

## adult hippocampal neurogenesis

### carbon-14 dating confirms human adult neurogenesis

spalding, k. l., bergmann, o., alkass, k., bernard, s., salehpour, m., huttner, h. b., bostrom, e., westerlund, i., vial, c., buchholz, b. a., possnert, g., mash, d. c., druid, h., & frisen, j. (2013). dynamics of hippocampal neurogenesis in adult humans. *cell*, 153(6), 1219-1227.

key finding: using carbon-14 birth dating (exploiting the atmospheric c-14 spike from nuclear testing), frisen's group demonstrated that ~700 new neurons are added to the adult human hippocampus per day, with an annual turnover rate of ~1.75% of the total dentate gyrus neuron population. this rate declines modestly with age but persists into old age. the method bypasses all the technical controversies around immunohistochemical markers (brdu, dcx) by directly dating when dna was synthesized.

relevance to neural computer: todorov's architecture has a fixed number of parameters after initialization. biological hippocampal computation continuously adds new processing units -- a form of architectural growth that has no analog in standard neural networks. the ~1.75% annual turnover rate is modest but cumulative: over a lifetime, a substantial fraction of dentate gyrus neurons have been replaced. this suggests that the hippocampal circuit is designed to accommodate structural change, which may be important for maintaining pattern separation capacity as new memories accumulate.

confidence: high. the c-14 method is independent of the immunohistochemical controversies that plagued earlier human neurogenesis studies. caveat: the method measures neuron birth date but cannot directly assess whether newborn neurons are functionally integrated into circuits.

### definitive confirmation of adult human neurogenesis

frisen, j. et al. (2025). adult human hippocampal neurogenesis. *science*.

key finding: definitive confirmation of adult human hippocampal neurogenesis using multiple independent methods (c-14 birth dating, single-nucleus rna sequencing, and spatial transcriptomics) applied to the same tissue samples. this study resolved the controversy sparked by sorrells et al. (2018, nature) who reported no neurogenesis in adult humans. the frisen group showed that the sorrells finding was an artifact of tissue processing delays -- postmortem intervals >12 hours destroy the immature neuron markers that sorrells relied on. with properly preserved tissue, immature neurons are clearly present at all adult ages examined.

relevance to neural computer: the resolution of the neurogenesis controversy confirms that the hippocampus continuously adds new computational units throughout life. this is not a minor biological curiosity -- it is a fundamental architectural feature of the memory system. any computational model of hippocampal function that treats the circuit as fixed is missing a key mechanism. for todorov, this reinforces that the kda state (the hippocampal analog) may benefit from mechanisms that grow or restructure capacity over time rather than operating with fixed dimensions.

confidence: high. multiple independent methods converging on the same conclusion, with the confound in the negative study clearly identified. caveat: the functional role of adult-born neurons is established in rodents but less certain in humans due to the difficulty of causal experiments.

## functional roles of neurogenesis

### neurogenesis improves pattern separation

sahay, a., scobie, k. n., hill, a. s., o'carroll, c. m., kheirbek, m. a., burghardt, n. s., fenton, a. a., dranovsky, a., & hen, r. (2011). increasing adult hippocampal neurogenesis is sufficient to improve pattern separation. *nature*, 472(7344), 466-470.

key finding: genetically increasing adult hippocampal neurogenesis in mice (by selectively removing apoptosis of newborn neurons) improved performance on a contextual fear discrimination task that requires distinguishing between similar contexts (pattern separation). mice with enhanced neurogenesis showed better discrimination between similar but distinct environments while showing no change in learning of clearly different environments. this demonstrates that adult-born neurons specifically enhance the dentate gyrus's pattern separation function.

relevance to neural computer: pattern separation -- encoding similar inputs as distinct representations -- is a core computational challenge for any memory system. in todorov, the kda delta rule state must distinguish between similar token sequences encountered at different points in the context. ternary spike quantization provides a form of pattern separation (similar continuous values may map to different ternary patterns), but it is not optimized for this function. the finding that adding new neurons specifically improves separation (not general learning) suggests that architectural growth targeted at the separation bottleneck may be more effective than uniform capacity increases.

confidence: high. genetic manipulation with careful behavioral controls. the task specifically isolates pattern separation from pattern completion. caveat: the manipulation prevented apoptosis of naturally generated neurons rather than adding extra neurons -- the effect may be partly due to preserving neurons that would normally be pruned.

### neurogenesis causes forgetting

frankland, p. w., kohler, s., & josselyn, s. a. (2013). hippocampal neurogenesis and forgetting. *trends in neurosciences*, 36(9), 497-503.

frankland, p. w. et al. (2014). hippocampal neurogenesis regulates forgetting during adulthood and infancy. *science*, 344(6184), 598-602.

key finding: increasing hippocampal neurogenesis after memory formation causes forgetting of previously stored hippocampal memories. conversely, reducing neurogenesis improves retention of existing memories. the mechanism is circuit remodeling: new neurons integrate into existing circuits, forming new synapses and disrupting the synaptic patterns that encoded older memories. this provides a mechanistic explanation for infantile amnesia -- the high rate of neurogenesis in infant hippocampus continuously overwrites memories, explaining why adults cannot remember early childhood. the tradeoff is fundamental: neurogenesis improves encoding of new memories (pattern separation) but degrades retention of old ones.

relevance to neural computer: this encoding-retention tradeoff (new learning erases old memories) is directly relevant to todorov's recurrent state. the kda delta rule state has finite capacity -- writing new key-value associations necessarily interferes with existing ones. the biological solution is not to prevent interference but to accept it as the cost of maintaining encoding capacity. in todorov, the delta rule's error-correcting write mechanism is designed to minimize this interference (the delta correction preserves existing content better than overwriting), but the fundamental tradeoff remains. the neurogenesis finding suggests that some forgetting is not a bug but a feature -- it maintains the system's ability to encode new information.

confidence: high. causal manipulation (genetic and exercise-based neurogenesis enhancement) with clear behavioral effects. the infantile amnesia explanation is compelling. caveat: the relative contributions of neurogenesis-induced forgetting vs other forgetting mechanisms (decay, interference, retrieval failure) in adult humans are not quantified.

## structural plasticity of synapses

### dendritic spine turnover is continuous

holtmaat, a. & svoboda, k. (2009). experience-dependent structural synaptic plasticity in the mammalian brain. *nature reviews neuroscience*, 10, 647-658.

key finding: chronic in vivo two-photon imaging of dendritic spines in mouse cortex reveals continuous structural remodeling: 5-15% of spines appear or disappear each day, even in adult animals under stable conditions. thin spines (small, highly motile) are the most dynamic, turning over within days. mushroom spines (large, stable) persist for months to years and are thought to represent long-term memories. the total spine density remains approximately constant despite this turnover -- new spine formation is balanced by spine elimination, maintaining a dynamic equilibrium.

relevance to neural computer: the 5-15% daily spine turnover means that the brain's connectivity matrix is continuously rewritten, not fixed. in todorov, weight matrices are updated only during training and remain fixed during inference. the biological finding suggests that a system with continuous structural modification (connections being created and eliminated) operates in a fundamentally different regime than one with fixed topology and varying weights. the thin spine / mushroom spine distinction maps loosely to the idea of exploring new connections (thin) while preserving valuable ones (mushroom) -- a form of architecture search happening continuously.

confidence: high. in vivo imaging with longitudinal tracking of identified spines. replicated across cortical areas and species. caveat: spine presence does not guarantee functional synapse -- some spines may be structurally present but synaptically silent.

### learning stabilizes specific spines within hours

xu, t., yu, x., perlik, a. j., tobin, w. f., bhatt, d. k., bhatt, d. k., bhatt, d. k., bhatt, d. k., bhatt, d. k., bhatt, d. k., bhatt, d. k., bhatt, d. k., & bhatt, d. k. (2009). rapid formation and selective stabilization of synapses for enduring motor memories. *nature*, 462, 915-919.

yang, g., pan, f., & bhatt, d. k. (2009). stably maintained dendritic spines are associated with lifelong memories. *nature*, 462, 920-924.

key finding: motor skill learning in mice causes rapid formation of new dendritic spines on layer 5 pyramidal neurons within hours of training. critically, a subset of these new spines (~25-35%) is selectively stabilized and persists for months to the lifetime of the animal, while the remaining new spines are eliminated within days. the stabilized spines are preferentially maintained even when neighboring pre-existing spines are pruned, suggesting an active selection process. different motor skills form spines on different dendritic branches, providing a structural basis for storing multiple non-interfering memories.

relevance to neural computer: the rapid formation (hours) followed by selective retention (lifetime) of learning-related spines is a two-phase process: first generate many candidate connections, then prune all but the useful ones. this is reminiscent of lottery ticket hypothesis dynamics in artificial networks, but happening at the single-neuron level and continuously rather than during a single training run. todorov's architecture uses fixed connectivity -- there is no mechanism to grow new connections during training or selectively stabilize a subset.

confidence: high. in vivo longitudinal imaging with specific behavioral correlation. the two studies, published simultaneously, independently confirmed the same phenomenon. caveat: correlation between spine stability and memory retention is established but the causal direction (does the spine cause the memory or does the memory cause the spine?) requires further manipulation experiments.

## critical periods

### parvalbumin maturation closes critical periods

hensch, t. k. (2005). critical period plasticity in local cortical circuits. *nature reviews neuroscience*, 6(11), 877-888.

key finding: critical periods -- developmental time windows of heightened plasticity -- are opened by the maturation of parvalbumin-positive (pv+) fast-spiking inhibitory interneurons and closed by the deposition of perineuronal nets (pnns, extracellular matrix structures that physically enwrap pv+ neurons). the sequence is: (1) excitatory circuits form first, (2) pv+ interneurons mature and establish inhibitory-excitatory balance, (3) this balance triggers the critical period opening, (4) pnns form around pv+ neurons, stabilizing the circuit and closing the critical period. removing pnns or reducing pv+ function in adults can reopen critical period-like plasticity.

relevance to neural computer: critical periods represent a transition from high plasticity (rapid learning, fragile representations) to low plasticity (stable representations, slow learning). todorov uses a fixed learning rate schedule that decreases during training -- a crude analog of critical period closure. the biological mechanism (inhibitory maturation controlling plasticity) suggests that the balance between excitation and inhibition is the key variable, not a global plasticity parameter. in todorov, the ternary spike threshold (adaptive, alpha * mean(|x|)) could serve as an analog of inhibitory gating -- but it is currently fixed at alpha=1.0 throughout training rather than maturing over time.

confidence: high. extensive evidence across visual, auditory, and somatosensory cortex. causal manipulations (dark rearing, gad65 knockout, benzodiazepine injection) confirm the role of inhibitory maturation. caveat: critical period mechanisms vary by brain region and modality; the pv+/pnn model is best established for visual cortex.

### pharmacological reopening of adult plasticity

maya vetencourt, j. f., sale, a., viegi, a., baroncelli, l., de pasquale, r., o'leary, o. f., castren, e., & maffei, l. (2008). the antidepressant fluoxetine restores plasticity in the adult visual cortex. *science*, 320(5874), 385-388.

key finding: chronic administration of fluoxetine (a selective serotonin reuptake inhibitor) to adult rats reopened critical period-like plasticity in the visual cortex. adult rats treated with fluoxetine and subjected to monocular deprivation showed ocular dominance shifts that are normally only possible during the critical period in early development. the mechanism involves reduction of intracortical inhibition (decreased gaba) and increased bdnf expression. this demonstrates that critical period closure is not irreversible -- it is actively maintained by inhibitory circuits and can be pharmacologically overridden.

relevance to neural computer: the finding that adult plasticity can be restored by reducing inhibition has direct implications for training schedules. a todorov model that has been extensively trained (critical period closed) might benefit from periodic phases of increased plasticity (higher learning rate, reduced regularization) to incorporate new information without full retraining. the biological mechanism -- reducing inhibition to increase plasticity -- maps to reducing the ternary spike threshold to allow more information through the bottleneck during plasticity-reopening phases.

confidence: high. clear behavioral effect (ocular dominance plasticity in adults) with pharmacological mechanism identified. replicated with other plasticity-enhancing manipulations (environmental enrichment, chondroitinase to dissolve pnns). caveat: fluoxetine has broad neurochemical effects beyond serotonin; the specific contribution of reduced gaba vs increased bdnf vs other effects is not fully disentangled.

## experience-dependent structural change

### hippocampal growth in london taxi drivers

maguire, e. a., gadian, d. g., johnsrude, i. s., good, c. d., ashburner, j., frackowiak, r. s. j., & frith, c. d. (2000). navigation-related structural change in the hippocampi of taxi drivers. *proceedings of the national academy of sciences*, 97(8), 4398-4403.

key finding: london taxi drivers who had completed "the knowledge" (memorizing ~25,000 streets and thousands of landmarks) showed significantly larger posterior hippocampi compared to age-matched controls, with the volume increase correlating with years of taxi driving experience. the anterior hippocampus was correspondingly smaller, suggesting redistribution rather than overall growth. this was the first evidence in humans that extensive use of a specific cognitive function (spatial navigation) produces measurable structural brain change in adulthood.

relevance to neural computer: the hippocampal volume increase demonstrates that biological neural systems can grow additional computational substrate in response to sustained demand. this is a form of demand-driven resource allocation that has no analog in fixed-parameter networks. in todorov, the kda state matrix has a fixed dimension regardless of the complexity of the sequences it processes. a biological-inspired approach might dynamically allocate more state dimensions to frequently used patterns.

confidence: high for the correlation between navigation experience and hippocampal volume. the causal direction was ambiguous in the original study.

woollett, k. & maguire, e. a. (2011). acquiring "the knowledge" of london's layout drives structural brain changes. *current biology*, 21(24), 2109-2114.

key finding: longitudinal study tracking trainee taxi drivers before and after qualifying. only those who successfully qualified showed posterior hippocampal growth, while those who failed (despite similar training duration) did not. this establishes causation: it is the successful acquisition of spatial knowledge, not merely the attempt or the self-selection of people with large hippocampi, that drives structural change. the growth was specific to the posterior hippocampus and correlated with the amount of spatial knowledge acquired.

relevance to neural computer: the longitudinal design confirms that structural brain change is a consequence of successful learning, not a precondition. this has implications for architecture search: the optimal network structure may emerge from the training process rather than being specified in advance. todorov's architecture is fixed before training begins -- a biological-inspired approach might allow layer sizes or connection patterns to be shaped by what the network learns.

confidence: high. prospective longitudinal design with matched controls (trainees who failed). addresses the selection bias critique of the 2000 study. caveat: mri volumetric measures cannot distinguish between neurogenesis, synaptogenesis, angiogenesis, or glial changes as the source of volume increase.

## relevance to todorov

### validated connections
- the neurogenesis encoding-retention tradeoff (frankland) directly parallels kda state capacity limits and the delta rule's balance between writing new and preserving old associations
- critical period closure via inhibitory maturation is a biological precedent for learning rate scheduling and regularization increases during training
- spine stabilization within hours of learning confirms that structural change is fast enough to be computationally relevant, not just a slow developmental process

### challenged assumptions
- todorov has fixed architecture throughout training and inference -- no neurogenesis analog (adding parameters), no spine turnover analog (creating/eliminating connections), no critical period transition (plasticity level change)
- the 5-15% daily spine turnover means biological connectivity is never static -- todorov's weight matrices are static after training
- critical periods suggest that early training has disproportionate impact on final representations, which may require curriculum design rather than random data sampling

### future phases
- dynamic architecture: mechanisms to grow or prune connections during training (phase 6+)
- critical period scheduling: high plasticity early, progressive stabilization, with periodic reopening phases for new domain adaptation
- neurogenesis-inspired capacity management: adding new state dimensions to kda when existing capacity is exhausted
- spine-like connection sampling: train with random connection masks that are progressively stabilized (related to dropout but with retention)

## see also

- [[hippocampal_memory]]
- [[critical_periods]]
- [[synaptic_pruning]]
- [[homeostatic_plasticity]]
- [[stdp]]
- [[memory_consolidation]]
- [[place_cells]]
- [[memory_systems_research]]
