# biological vision

## retinal processing

kuffler, s. w. (1953). discharge patterns and functional organization of mammalian retina. *journal of neurophysiology*, 16(1), 37-68.

key finding: retinal ganglion cells have center-surround receptive fields that compute local contrast, not absolute luminance. the retina discards mean illumination and encodes relative differences -- the same operation as the ternary spike threshold (alpha * mean(|x|)).

masland, r. h. (2012). the neuronal organization of the retina. *neuron*, 76(2), 266-280. doi: 10.1016/j.neuron.2012.10.002.

key finding: the retina contains ~20 parallel output channels to the brain, each encoding a different feature. the retina sends 20 feature-analyzed versions simultaneously, not a single image. architecturally analogous to independent projection heads.

gollisch, t. & meister, m. (2010). eye smarter than scientists believed: neural computations in circuits of the retina. *neuron*, 65(2), 150-164. doi: 10.1016/j.neuron.2009.12.009.

key finding: the retina performs motion computation, contrast gain adaptation, and predictive firing (signaling future object positions before they arrive). these computations were previously attributed exclusively to cortex.

confidence: high.

## the visual hierarchy

felleman, d. j. & van essen, d. c. (1991). distributed hierarchical processing in the primate cerebral cortex. *cerebral cortex*, 1(1), 1-47.

key finding: 32 visual areas connected by 305 pathways. more feedback connections than feedforward. organized into ~10 hierarchical levels based on laminar connection patterns. this is not a feedforward pipeline -- it is a recurrent network with hierarchical structure.

hubel, d. h. & wiesel, t. n. (1962). receptive fields, binocular interaction and functional architecture in the cat's visual cortex. *journal of physiology*, 160, 106-154.

key finding: V1 neurons are tuned to orientation, spatial frequency, and binocularity. simple cells respond to oriented edges; complex cells add position invariance. the hierarchical composition of increasingly complex features.

dicarlo, j. j. & cox, d. d. (2007). untangling invariant object recognition. *trends in cognitive sciences*, 11(8), 333-341.

key finding: the ventral stream's goal is manifold untangling -- transforming tangled, interleaved object representations into linearly separable ones through successive nonlinear stages. IT cortex achieves linear readout of object identity.

confidence: high.

## predictive coding in vision

rao, r. p. n. & ballard, d. h. (1999). predictive coding in the visual cortex. *nature neuroscience*, 2(1), 79-87. doi: 10.1038/nn0199_79.

key finding: top-down connections carry predictions, bottom-up connections carry prediction errors. the model generates gabor-like V1 receptive fields from first principles. when predictions are accurate, very little signal propagates. the brain generates vision, not passively receives it.

carandini, m. & heeger, d. j. (2012). normalization as a canonical neural computation. *nature reviews neuroscience*, 13(1), 51-62. doi: 10.1038/nrn3136.

key finding: divisive normalization appears in retina, LGN, V1, MT, V4, IT, olfactory, auditory, parietal, and prefrontal cortex. a single operation (divide by pooled neighbor activity) explains contrast gain control, attention, multisensory integration, and value encoding.

confidence: high.

## figure-ground segregation

lamme, v. a. f. (1995). the neurophysiology of figure-ground segregation in primary visual cortex. *journal of neuroscience*, 15(2), 1605-1615.

key finding: V1 neurons show delayed (100-200ms) response enhancement signaling figure vs ground, even when local stimulus is identical. this requires global scene processing fed back to V1. feedforward-only models cannot produce this signal.

lamme, v. a. f. & roelfsema, p. r. (2000). the distinct modes of vision offered by feedforward and recurrent processing. *trends in neurosciences*, 23(11), 571-579.

key finding: feedforward sweep (50-80ms) handles rapid categorization. recurrent processing (100-200ms) handles figure-ground, attention, and conscious access. anesthesia eliminates the recurrent signal but preserves feedforward response.

confidence: high.

## scene understanding

oliva, a. & torralba, a. (2006). building the gist of a scene. *progress in brain research*, 155, 23-36.

key finding: humans categorize scenes in 150ms using global spatial statistics before individual objects are recognized. the brain runs a fast global scene analyzer in parallel with the slower object system. sparse, coarse-scale statistics are sufficient.

hochstein, s. & ahissar, m. (2002). view from the top: hierarchies and reverse hierarchies in the visual system. *neuron*, 36(5), 791-804.

key finding: perception proceeds globally-to-locally (reverse hierarchy). pop-out is fast because it happens at top levels. serial search requires top-down routing back to lower levels. the reverse hierarchy motivates top-down feedback paths.

confidence: high.

## visual attention

itti, l. & koch, c. (2001). computational modelling of visual attention. *nature reviews neuroscience*, 2(3), 194-203. doi: 10.1038/35058500.

key finding: a saliency map combining center-surround contrast across color, intensity, and orientation predicts fixation locations. inhibition of return prevents repeated attention to the same location.

treisman, a. m. & gelade, g. (1980). a feature-integration theory of attention. *cognitive psychology*, 12(1), 97-136.

key finding: single features are detected in parallel. conjunctions require serial search with focused attention as the binding operator. features are registered independently before objects are bound.

confidence: high.

## motion and optical flow

born, r. t. & bradley, d. c. (2005). structure and function of visual area MT. *annual review of neuroscience*, 28, 157-189.

key finding: MT has center-surround organization for motion: center signals local direction, surround suppresses when motion matches (global motion suppression). the temporal analog of spatial center-surround.

goodale, m. a. & milner, a. d. (1992). separate visual pathways for perception and action. *trends in neurosciences*, 15(1), 20-25.

key finding: ventral stream encodes identity (what), dorsal stream encodes spatial relations and visuomotor transformations (how). patient D.F. could grasp objects she couldn't identify. two streams should be maintained as separate output pathways.

confidence: high.

## color vision

land, e. h. (1977). the retinex theory of color vision. *scientific american*, 237(6), 108-128.

key finding: color constancy is achieved by computing reflectance ratios across spatial regions. the visual system encodes relative differences, not absolute values -- the same principle as center-surround and ternary thresholding.

confidence: high.

## development

hubel, d. h. & wiesel, t. n. (1970). the period of susceptibility to the physiological effects of unilateral eye closure in kittens. *journal of physiology*, 206(2), 419-436.

key finding: 3-4 days of monocular deprivation during the critical period (weeks 3-12 in cats) irreversibly shifts V1 responses. the same deprivation outside the window has no effect. the critical period is a biological learning rate schedule.

hensch, t. k. (2005). critical period plasticity in local cortical circuits. *nature reviews neuroscience*, 6(11), 877-888.

key finding: the critical period is triggered by PV+ interneuron maturation (GABAergic inhibition), not by activity alone. pharmacological manipulation can open, close, or reopen the window.

confidence: high.

## relevance to the neural machine

the visual system validates these architectural principles:
- **center-surround contrast** (not absolute values) at every stage from retina to MT -- maps to ternary spike thresholding
- **parallel channels** (20 retinal outputs) -- maps to multi-head projections
- **predictive coding** (top-down predictions, bottom-up errors) -- maps to the delta-rule recurrent state
- **recurrence is required** for figure-ground, attention, and conscious access -- validates within-layer iteration
- **normalization is canonical** -- maps to adaptive threshold alpha * mean(|x|)
- **two streams** (identity vs spatial) -- suggests maintaining separate output pathways
- **reverse hierarchy** (global before local) -- suggests coarse-to-fine processing order

## see also

- [[predictive_coding]]
- [[divisive_normalization]]
- [[selective_attention]]
- [[lateral_inhibition]]
- [[critical_periods]]
- [[dendritic_computation]]
