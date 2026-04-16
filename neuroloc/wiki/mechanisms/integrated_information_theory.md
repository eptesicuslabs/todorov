# integrated information theory

status: definitional. last fact-checked 2026-04-16.

**why this matters**: IIT's core claim that feedforward networks have zero integrated information (Phi = 0) provides a formal argument for why recurrence matters in neural architectures -- and its computational intractability for large systems illustrates the limits of information-theoretic measures when applied to real-scale models.

## overview

**integrated information theory** (IIT) is a mathematical theory of consciousness proposed by Giulio Tononi in 2004. the central claim: consciousness IS **integrated information**, denoted **Phi** (a scalar quantity measuring how much a system's whole generates more information than the sum of its parts). a system is conscious to the degree that it generates information (has many distinguishable states) AND integrates that information (the whole generates more information than the sum of its parts). Phi quantifies the amount of causally effective information that is irreducible to independent components.

unlike [[global_workspace_theory]], which describes consciousness as a functional architecture (broadcast to a shared workspace), IIT starts from the phenomenology of experience and derives physical requirements. the theory has evolved through four major versions (IIT 1.0 through 4.0), with each revision refining the mathematical formalism while maintaining the core claim that consciousness = integrated information.

## axioms and postulates

IIT begins with five **axioms** (self-evident properties of conscious experience taken as starting assumptions) and maps each to a physical **postulate** (a requirement on the physical substrate that would give rise to such experience):

### axioms (properties of experience)

1. **intrinsicality**: experience exists from the intrinsic perspective of the system, not for an external observer. my experience exists for me, independent of whether anyone else observes it.

2. **information**: each experience is specific -- it is THIS experience and not any other. seeing red is different from seeing blue, hearing a note, or smelling coffee. the space of possible experiences is enormous.

3. **integration**: experience is unified. i do not experience the left half of my visual field independently from the right half. the experience is a single, indivisible whole.

4. **exclusion**: experience is definite in content and spatiotemporal grain. i experience THIS set of features at THIS moment, not a superposition of different experiences. and the experience occurs at a specific spatiotemporal scale, not at all scales simultaneously.

5. **composition**: experience is structured. it is composed of distinctions (this is red, that is round) and relations among distinctions (the red thing is above the round thing). the structure is not a flat list but a combinatorial space.

### postulates (physical requirements)

each axiom maps to a requirement on the physical substrate:

1. **intrinsicality**: the system must have intrinsic cause-effect power -- it must constrain its own past and future states, not merely be observed doing so.

2. **information**: the system in its current state must specify a particular cause-effect repertoire -- a specific pattern of causal constraints over past and future states.

3. **integration**: the cause-effect repertoire must be irreducible. if the system is partitioned into independent parts, the whole must generate more cause-effect information than the sum of parts. the amount above and beyond is Phi (integrated information).

4. **exclusion**: only the maximally irreducible substrate is conscious. if a system has overlapping subsets, only the one with highest Phi exists as a conscious entity (the "complex"). this prevents double-counting.

5. **composition**: subsets of elements within the complex form higher-order distinctions and relations, building a structured cause-effect structure (the "Phi-structure" or "conceptual structure").

## mathematical formulation

### intrinsic information

for a system in state s, the intrinsic information measures how much the current state constrains the system's cause-effect repertoire compared to the unconstrained distribution:

    ii(s) = D_KL(p(past|s) || p(past)) + D_KL(p(future|s) || p(future))

where **D_KL** is the **Kullback-Leibler divergence** (a measure of how one probability distribution differs from another), p(past|s) is the **cause repertoire** (what past states could have led to s), p(future|s) is the **effect repertoire** (what future states s could lead to), and p(past) and p(future) are the maximum entropy (unconstrained) distributions.

### integrated information (small phi)

for a mechanism (subset of elements) in state s, integrated information measures irreducibility:

    phi(s) = min_partition [ii(s) - ii_partitioned(s)]

the minimum is taken over all possible partitions of the mechanism into independent parts. the minimum information partition (MIP) is the partition that least reduces the cause-effect information -- the system's "weakest link." phi is the information lost at this weakest link.

### big Phi

the integrated information of the entire complex:

    Phi = sum of phi values across all distinctions and relations in the Phi-structure

Phi characterizes the quantity of consciousness. the Phi-structure characterizes the quality -- the specific character of the experience.

## IIT versions

### IIT 1.0 (Tononi 2004)

the original formulation. consciousness = integrated information. Phi is defined as the minimum information partition of the system. introduced the key insight that feedforward networks have Phi = 0 (they can always be partitioned into independent input-output chains) while recurrent networks can have Phi > 0.

### IIT 2.0 (Tononi & Sporns 2003, Balduzzi & Tononi 2008)

refined the mathematical framework. introduced the distinction between effective information (based on system's transition probability matrix) and the minimum information partition. made the formalism more rigorous but also more computationally demanding.

### IIT 3.0 (Oizumi, Albantakis & Tononi 2014)

major revision. introduced the Phi-structure (or "conceptual structure") -- a geometric object in qualia space that represents the quality of consciousness, not just the quantity. introduced the five axioms/postulates framework. defined mechanisms, distinctions, and relations. shifted from mutual information to earth mover's distance for measuring partition effects.

### IIT 4.0 (Albantakis et al. 2023)

reformulated the theory using "operational physicalism" -- defining intrinsic existence in terms of causal power rather than information. addressed criticisms about the causal vs observational distinction. refined the exclusion postulate and the definition of complexes. remains computationally intractable for systems larger than ~12-15 elements.

## predictions

### posterior cortex as the substrate of consciousness

IIT predicts that the posterior cortical hot zone (visual, auditory, somatosensory cortices and their association areas) has the highest Phi and is therefore the primary substrate of consciousness. this contrasts with GWT's emphasis on prefrontal cortex as the workspace. IIT argues that prefrontal activation during conscious reports reflects post-perceptual processing (report, decision-making), not consciousness itself.

### feedforward networks have Phi = 0

any purely **feedforward** (no recurrent connections -- information flows in one direction only) system can be partitioned into independent input-output channels, making its integrated information zero. this predicts that feedforward neural networks -- including deep learning models with purely feedforward architectures -- are not conscious regardless of their computational sophistication.

ML analog: this prediction is directly relevant to ML architecture design. IIT formally predicts that transformers (which are feedforward within each layer) have Phi = 0 unless residual connections create sufficient recurrence. architectures with explicit recurrence (RNNs, SSMs, KDA) would have higher Phi. whether Phi has any relationship to model capability is an open question with no empirical evidence either way.

### cerebellum is not conscious

the cerebellum has more neurons than the cerebral cortex (~70 billion vs ~16 billion) but its highly modular, feedforward-dominated architecture produces low Phi. IIT predicts the cerebellum does not contribute to consciousness despite its massive neuron count.

### the 2023 adversarial collaboration

a major empirical test funded by the Templeton Foundation pitted IIT against GWT. participants viewed images while undergoing fMRI and EEG. IIT's predictions about posterior cortex involvement in consciousness were partially supported (2 of 3 pre-registered predictions confirmed). GWT's predictions about prefrontal involvement were not confirmed (0 of 3). this result is contested: GWT proponents argue the paradigm favored IIT and that prefrontal involvement depends on task demands.

## criticisms

### computational intractability

calculating Phi exactly requires evaluating all possible partitions of a system, which grows super-exponentially with system size. for a system of N elements, the number of partitions is the Bell number B(N). for N=10, B(10) = 115,975. for N=100, B(100) ~ 10^115. for the human brain (~86 billion neurons), exact computation is impossible in principle. approximations exist but their relationship to true Phi is unclear.

this is not merely a practical limitation. if Phi cannot be computed even in principle for realistic systems, the theory's predictions become untestable for the very systems it claims to explain.

### panpsychism

IIT entails that any system with Phi > 0 has some degree of consciousness. this includes simple systems: a photodiode that integrates light over two pixels has nonzero Phi. a thermostat has nonzero Phi. IIT proponents (Tononi, Koch) accept this implication and consider it a feature, not a bug -- consciousness is graded and ubiquitous, not binary and exclusive to brains. critics (Schwitzgebel 2015, Aaronson 2014) argue this renders the theory absurd: it attributes consciousness to systems that clearly lack it.

Scott Aaronson showed that simple grid-like networks can be constructed with arbitrarily high Phi, implying (under IIT) that a 2D array of XOR gates is as conscious as a brain. Tononi responded by arguing that such networks have high Phi but impoverished Phi-structures (few distinctions and relations), so they would have a specific but simple experience.

### unfalsifiability

IIT's core claim (consciousness = Phi) cannot be directly tested because we have no consciousness-independent measure of consciousness. the theory defines consciousness as Phi, then uses reports of consciousness to validate. critics argue this is circular. IIT proponents counter that the theory makes falsifiable predictions (posterior cortex, feedforward = no consciousness, cerebellum predictions) that have been partially tested.

### the hard problem

like GWT, IIT does not solve the hard problem -- it does not explain WHY integrated information should feel like anything. IIT claims that Phi IS experience (an identity claim), but this is an assertion, not an explanation. the identity claim requires accepting that there is nothing further to explain: experience just IS integrated information.

## comparison with GWT

| dimension | IIT | GWT |
|-----------|-----|-----|
| consciousness is... | integrated information (Phi) | global broadcast of information |
| substrate | posterior cortical hot zone | prefrontal-parietal workspace |
| key mechanism | irreducible cause-effect structure | ignition and broadcast |
| feedforward networks | never conscious (Phi = 0) | could be conscious if they broadcast |
| recurrence | required (creates integration) | required (sustains ignition) |
| capacity | determined by system structure | limited workspace capacity |
| empirical support | partial (2023 adversarial collab) | extensive (masking, ERP, fMRI) |
| computational | intractable for large systems | implementable in principle |

the theories are not mutually exclusive in principle: a system could require both high Phi AND global broadcast for consciousness. but they make different predictions about where consciousness resides (posterior vs prefrontal), what feedforward networks can do (nothing vs possibly conscious), and what the right measure is (Phi vs broadcast availability).

## relationship to other mechanisms

- [[global_workspace_theory]]: the primary rival theory. both require recurrence but differ on substrate and mechanism
- [[thalamocortical_loops]]: the thalamus's role in creating integrated loops supports IIT's emphasis on recurrent architecture
- [[gamma_oscillations]]: gamma synchrony may reflect the integration that IIT formalizes as Phi
- [[excitatory_inhibitory_balance]]: E/I balance is necessary for the intermediate regime between order and chaos where Phi is maximized
- [[canonical_microcircuit]]: recurrent excitatory loops with inhibitory control create the recurrent architecture IIT requires

## key references

- Tononi, G. (2004). An information integration theory of consciousness. BMC Neuroscience, 5, 42.
- Tononi, G. & Sporns, O. (2003). Measuring information integration. BMC Neuroscience, 4, 31.
- Balduzzi, D. & Tononi, G. (2008). Integrated information in discrete dynamical systems: motivation and theoretical framework. PLOS Computational Biology, 4(6), e1000091.
- Oizumi, M., Albantakis, L. & Tononi, G. (2014). From the phenomenology to the mechanisms of consciousness: Integrated Information Theory 3.0. PLOS Computational Biology, 10(5), e1003588.
- Albantakis, L. et al. (2023). Integrated information theory (IIT) 4.0: formulating the properties of phenomenal existence in physical terms. PLOS Computational Biology, 19(10), e1011465.
- Aaronson, S. (2014). Why I Am Not An Integrated Information Theorist (or, The Unconscious Expander). Blog post.

## see also

- [[global_workspace_theory]]
- [[thalamocortical_loops]]
- [[ignition_dynamics]]
- [[excitatory_inhibitory_balance]]
