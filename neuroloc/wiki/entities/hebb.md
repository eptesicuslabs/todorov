# Donald Hebb

status: definitional. last fact-checked 2026-04-16.

## biographical summary

Donald Olding Hebb (1904-1985) was a Canadian neuropsychologist whose 1949 book *The Organization of Behavior: A Neuropsychological Theory* is one of the most influential works in neuroscience. Hebb proposed that synaptic connections between neurons are strengthened when the neurons are co-activated, formulating what is now known as Hebb's postulate or the Hebbian learning rule.

## the neurophysiological postulate (1949)

"when an axon of cell A is near enough to excite cell B and repeatedly or persistently takes part in firing it, some growth process or metabolic change takes place in one or both cells such that A's efficiency, as one of the cells firing B, is increased."

this single sentence launched the modern study of synaptic plasticity. its mathematical formalization -- Delta_w = eta * x_pre * x_post -- is the foundation of [[hebbian_learning]] and, by extension, [[stdp]], [[bcm_theory]], and all correlation-based learning rules.

## the cell assembly

Hebb's second major contribution was the concept of the cell assembly: a group of neurons that become strongly interconnected through repeated co-activation, such that activation of a subset can reactivate the entire group. cell assemblies were proposed as the neural substrate of concepts, percepts, and memories. this idea anticipated modern attractor network models and is structurally related to Hopfield networks.

## legacy

- the Hebbian learning rule is the basis for unsupervised feature extraction (PCA, ICA, sparse coding)
- the cell assembly concept anticipated Hopfield networks (1982), attractor dynamics, and modern theories of distributed memory
- the phrase "neurons that fire together, wire together" (coined by Carla Shatz, not Hebb) is the most widely known summary of the postulate
- the discovery of LTP (Bliss and Lomo, 1973) and [[stdp]] (Markram et al. 1997, Bi and Poo 1998) provided experimental confirmation of the core idea
- over 31,000 citations as of 2020, making it one of the most cited works in psychology and neuroscience

## relevance to todorov

the KDA delta rule's outer product k_t * v_t^T is a direct instantiation of the Hebbian association rule. see [[plasticity_to_kda_delta_rule]].

## key reference

- Hebb, D. O. (1949). *The Organization of Behavior: A Neuropsychological Theory*. Wiley.
