# imagination and mental simulation

status: current (as of 2026-04-16).

## hippocampal scene construction

hassabis, d., kumaran, d., vann, s. d. & maguire, e. a. (2007). patients with hippocampal amnesia cannot imagine new experiences. *pnas*, 104(5), 1726-1731.

key finding: five patients with bilateral hippocampal damage could not construct novel imagined scenarios from verbal cues. the deficit was specific to spatially coherent scenes -- single objects could still be imagined. the hippocampus assembles coherent internal representations from fragments. this is the causal evidence that imagination requires the same circuit as memory.

relevance: the core architecture of imagination is fragment recombination, not whole-scene generation. a neural machine needs a mechanism that retrieves stored fragments (kda state readout) and combines them into novel configurations (bilinear interaction). the outer product is already a recombination operation -- k * v^T creates a new association from existing components.

confidence: high.

## constructive episodic simulation

schacter, d. l. & addis, d. r. (2007). the cognitive neuroscience of constructive memory: remembering the past and imagining the future. *philosophical transactions of the royal society b*, 362(1481), 773-786.

key finding: remembering and imagining activate virtually identical neural circuits: hippocampus, medial prefrontal cortex, lateral temporal cortex, posterior parietal cortex (the default mode network). the hippocampus recombines stored episodic fragments into novel scenarios. memory errors (distortions, false memories) are the necessary byproduct of a system optimized for flexible future simulation.

relevance: a neural machine's memory system (kda) should be designed to RECOMBINE, not just retrieve. the delta rule state update already performs a form of recombination: new associations are written on top of decayed old ones, creating blended representations. controlled recombination = imagination.

confidence: high.

## the suppression circuit (controlled vs intrusive)

anderson, m. c. et al. (2025). brain mechanisms underlying the inhibitory control of thought. *nature reviews neuroscience*.

key finding: the right lateral prefrontal cortex suppresses hippocampal retrieval via GABAergic interneurons. when this pathway is intact, unwanted memory retrieval and imagination are inhibited. when disrupted, thoughts become intrusive and uncontrollable. the same mechanism that stops memory retrieval also stops future imagination (benoit, davies & anderson 2016 pnas).

relevance: this is the critical circuit for a neural machine. imagination must be ON DEMAND. the machine needs a gate that enables generative recombination when requested and suppresses it otherwise. without this gate, the generative system produces intrusive outputs. implementation: a binary control signal that enables/disables the recombinative path in the architecture.

confidence: high.

## creativity as dmn-executive coupling

beaty, r. e. et al. (2018). robust prediction of individual creative ability from brain functional connectivity. *pnas*, 115(5), 1087-1092.

key finding: creative ability is predicted by resting-state coupling between the default mode network (generation) and the frontoparietal executive network (evaluation/selection). neither alone produces creative output. the dmn generates candidates; the executive network filters, constrains, and selects. individual differences in this coupling are stable and measurable.

relevance: a neural machine needs two interacting systems for imagination: a generator (kda recombination, forward model prediction) and an evaluator (a scoring function that rates generated scenarios for plausibility, utility, consistency). the generator proposes; the evaluator disposes.

confidence: high.

## visual imagery as top-down prediction

pearson, j. (2019). the human imagination: the cognitive neuroscience of visual mental imagery. *nature reviews neuroscience*, 20, 624-634.

key finding: visual imagery activates primary visual cortex via top-down feedback from frontal and parietal areas. imagery is the generative pass of predictive coding running WITHOUT bottom-up sensory correction. individual differences are enormous (aphantasia to hyperphantasia). imagery vividness correlates with V1 anatomy.

relevance: imagination in a neural machine is the forward pass of the generative model without input. run the prediction pathway (top-down) without the error-correction pathway (bottom-up). the residual stream carries predictions; normally they're corrected by input; during imagination, they run uncorrected.

confidence: high.

## dreaming as offline generative training

deperrois, n., petrovici, m. a., senn, w. & jordan, j. (2022). learning cortical representations through perturbed and adversarial dreaming. *elife*, 11, e76384.

key finding: computational model where nrem = perturbed replay (robustness training with occlusions) and rem = adversarial generation (novel combinations evaluated against the learned model). the adversarial rem phase was necessary for extracting semantic category structure. three-phase training: wake (sensory), nrem (perturbed replay), rem (adversarial generation).

relevance: a neural machine could benefit from an offline training phase where the generative system runs without input, produces novel combinations, and the discriminator evaluates them. this is imagination-as-training: improving the model by imagining.

confidence: medium-high. computational model, not direct neural measurement.

## forward models as primitive imagination

wolpert, d. m. & ghahramani, z. (2000). computational principles of movement neuroscience. *nature neuroscience*, 3(suppl), 1212-1217.

key finding: the cerebellum predicts sensory consequences of actions before they happen. this is imagination reduced to its simplest form: "if I do X, what will happen?" the prediction runs ahead of reality by 100-200ms.

relevance: a neural machine's imagination should start here -- predict the consequence of the next output before emitting it. if the predicted consequence is bad, don't emit. this is the simplest form of planning and the foundation for more complex simulation.

confidence: high.

## see also

- [[hippocampal_memory]]
- [[memory_consolidation]]
- [[predictive_coding]]
- [[complementary_learning_systems]]
- [[sleep_and_dreaming_research]]
