# Jean-Pierre Changeux

## biographical summary

Jean-Pierre Changeux (born 1936) is a French neuroscientist at the Institut Pasteur and College de France. he is one of the founders of molecular neuroscience and a pioneer of the theory that neural circuit development is governed by Darwinian selection among synapses rather than genetic instruction. his selective stabilization theory (Changeux, Courrege, and Danchin 1973; Changeux and Danchin 1976) proposed that the genome specifies the DISTRIBUTION of possible synaptic connections, and neural activity selects which connections survive -- a radical departure from the prevailing instructionist view of brain development.

## key contributions

### selective stabilization of developing synapses (1976)

Changeux and Danchin published "Selective stabilisation of developing synapses as a mechanism for the specification of neuronal networks" in Nature (1976). the theory proposes:

1. the genome specifies connections between CLASSES of neurons, producing an initial state of exuberant, redundant connectivity ("genetic envelope")
2. developing synapses exist in a labile state from which they transition to either stable (retained) or degenerate (eliminated) states
3. the transition is activity-dependent: synapses that participate in correlated pre-postsynaptic activity are stabilized; those that do not are eliminated
4. the final circuit is thus determined by the interaction between the genetic envelope and neural activity during development

this is a Darwinian process: variation (exuberant connectivity) followed by selection (activity-dependent stabilization). the genome does not specify the wiring diagram; it specifies the search space from which activity-dependent selection operates. different activity histories produce different final circuits from the same genetic program.

the theory explained how experience could shape neural circuits without invoking Lamarckian inheritance of acquired characteristics: the genome provides the substrate, and experience selects from it. this framework preceded and influenced Gerald Edelman's theory of neural Darwinism (1987), which applied similar selectionist logic at the level of neuronal groups.

see [[synaptic_pruning]] for the full mechanism and modern evidence.

### the allosteric model of receptor function

before his work on neural development, Changeux made foundational contributions to molecular biology. his work with Monod and Wyman on the allosteric model of protein function (Monod, Wyman, and Changeux 1965) explained how proteins can switch between active and inactive conformations in response to regulatory molecules. he then applied the allosteric framework to the nicotinic acetylcholine receptor at the neuromuscular junction, solving its structure and functional transitions. this work established the molecular basis for synaptic transmission and laid the groundwork for his later developmental theories.

### neuronal man (1985)

Changeux's book "Neuronal Man: The Biology of Mind" (L'Homme Neuronal, 1983; English translation 1985) presented a comprehensive materialist account of mind and brain, arguing that all mental phenomena are ultimately reducible to neural activity. the book synthesized his selective stabilization theory with contemporary neuroscience to argue that consciousness, thought, and culture emerge from the same Darwinian selectionist processes that shape synaptic development.

### epigenesis of neuronal networks

Changeux, Courrege, and Danchin (1973) published the mathematical formalization of selective stabilization: "A Theory of the Epigenesis of Neuronal Networks by Selective Stabilization of Synapses." this paper defined the formal framework (genetic envelope, labile/stable/degenerate states, activity-dependent transition probabilities) that the 1976 Nature paper presented in accessible form. the term "epigenesis" was used in its original developmental biology sense (the emergence of structure from initially undifferentiated material), not in the modern epigenetic sense (heritable changes in gene expression).

## significance

Changeux's selective stabilization theory resolved a fundamental problem in developmental neuroscience: how can a genome with ~20,000 genes specify the connectivity of ~10^14 synapses? the answer: it does not. it specifies the statistical properties of connectivity (which classes of neurons connect to which), and activity-dependent selection determines the specific pattern. this insight has been confirmed by decades of work on [[synaptic_pruning]], [[critical_periods]], and [[developmental_self_organization]], including the identification of molecular mechanisms (complement system, microglia) that implement the elimination process.

the selectionist framework also provides a bridge between neural development and learning: both are refinement processes that select from a space of possibilities based on functional criteria. the difference is timescale (developmental pruning operates over months-years; Hebbian learning operates over seconds-minutes) and mechanism (pruning permanently removes connections; plasticity strengthens or weakens existing ones).

## relevance to todorov

todorov's ternary spike mechanism can be interpreted through the selective stabilization lens: on each forward pass, the spike function "selects" which dimensions are active ({-1, +1}) and which are eliminated ({0}). but this selection is transient (different inputs select different dimensions) and does not permanently remove any connections. the architectural analog of selective stabilization would be training-time pruning, which todorov does not implement. see [[synaptic_pruning]] and [[development_to_training_curriculum]] for analysis.

## key references

- Changeux, J.-P., Courrege, P. & Danchin, A. (1973). A theory of the epigenesis of neuronal networks by selective stabilization of synapses. Proceedings of the National Academy of Sciences, 70(10), 2974-2978.
- Changeux, J.-P. & Danchin, A. (1976). Selective stabilisation of developing synapses as a mechanism for the specification of neuronal networks. Nature, 264(5588), 705-712.
- Monod, J., Wyman, J. & Changeux, J.-P. (1965). On the nature of allosteric transitions: a plausible model. Journal of Molecular Biology, 12(1), 88-118.
- Changeux, J.-P. (1985). Neuronal Man: The Biology of Mind. Oxford University Press.
- Edelman, G. M. (1987). Neural Darwinism: The Theory of Neuronal Group Selection. Basic Books.
