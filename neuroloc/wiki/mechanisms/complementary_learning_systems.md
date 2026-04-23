# complementary learning systems

status: definitional. last fact-checked 2026-04-16.

**why this matters**: CLS theory explains why catastrophic forgetting occurs in neural networks and predicts that the solution requires two memory systems with different learning rates -- a principle that directly inspired experience replay in DQN and continual learning methods like EWC.

## summary

the **complementary learning systems** (**CLS**) theory (McClelland, McNaughton & O'Reilly 1995) explains why the brain requires two separate memory systems. a single neural network that learns quickly will **catastrophically overwrite** (completely destroy previously stored representations when learning new ones) previously stored information. a network that learns slowly will preserve existing knowledge but cannot rapidly encode new episodes. the brain's solution: the hippocampus learns fast (one-shot episodic memory), the neocortex learns slow (gradual extraction of statistical structure), and sleep replay transfers information from one to the other. the theory transformed what had been a liability of connectionist models -- catastrophic interference -- into an explanation for the functional organization of mammalian memory.

## the problem: catastrophic interference

### the connectionist dilemma

in the 1980s, the **PDP** (**parallel distributed processing**) framework (Rumelhart & McClelland 1986) showed that neural networks with **distributed representations** (representations where each concept is encoded across many neurons, and each neuron participates in many concepts) could learn complex tasks via gradient-based training. but McCloskey and Cohen (1989) discovered a devastating problem: when a trained network learns new information, it catastrophically forgets what it previously knew.

the mechanism is straightforward. in a network with overlapping distributed representations, learning pattern B modifies the same weights that encode pattern A. if the learning rate is high enough to learn B in a few presentations, the weight changes will be large enough to destroy A. this is not ordinary forgetting (gradual decay) -- it is catastrophic interference (complete erasure).

### interleaved vs sequential learning

McClelland et al. (1995) demonstrated that the solution depends on how training examples are presented:

- **sequential learning**: present all examples of category A, then all examples of category B. result: B overwrites A. the network exhibits catastrophic interference
- **interleaved learning**: present examples of A and B in random order, many times. result: the network gradually learns both A and B without interference. but interleaved learning requires many repetitions and access to all past examples

the problem: biological organisms experience events sequentially, not in interleaved order. the world does not replay past experiences. and yet organisms learn without catastrophic forgetting. something must provide the interleaving.

## the solution: two systems

### system 1: hippocampus (fast learner)

the hippocampus (see [[hippocampal_memory]]) has the properties needed for rapid one-shot learning:

- **sparse representations**: the dentate gyrus maintains ~2-5% population sparsity, minimizing overlap between stored patterns and therefore minimizing interference
- **pattern separation**: similar inputs are mapped to dissimilar internal representations, further reducing interference
- **fast synaptic modification**: CA3 recurrent synapses undergo rapid LTP, enabling storage from a single experience
- **auto-associative retrieval**: CA3 pattern completion (see [[pattern_completion]]) can retrieve a full memory from a partial cue

the cost: limited capacity. the hippocampus can store thousands of patterns, but not millions. and fast learning with overlapping representations would cause the same catastrophic interference that plagues artificial networks. the hippocampus avoids this by using extremely sparse representations, but this limits how much information each memory can carry.

### system 2: neocortex (slow learner)

the neocortex has complementary properties:

- **distributed, overlapping representations**: each concept is encoded across millions of neurons, and each neuron participates in many concepts. this allows the extraction of shared statistical structure (e.g., "all birds have feathers")
- **slow learning rate**: synaptic modifications are small per experience, preventing any single event from disrupting existing knowledge
- **massive capacity**: the neocortex has ~16 billion neurons (human) with ~10^14 synapses, providing enormous storage
- **generalization**: overlapping representations enable automatic generalization -- if birds A and B share many features, they will activate overlapping populations, and knowledge about A will partially transfer to B

the cost: slow learning. the neocortex cannot learn from a single experience. it requires many repetitions to gradually adjust weights. learning the capital of a new country from a single presentation is a hippocampal function. learning that capitals tend to be large cities near rivers is a neocortical function that emerges from hundreds of examples.

### the bridge: replay

the CLS theory proposes that the hippocampus solves the interleaving problem by replaying stored memories to the neocortex during offline periods (especially sleep; see [[memory_consolidation]]). this replay provides the interleaved training that the neocortex needs:

1. during waking experience, the hippocampus rapidly encodes each episode
2. during sleep (especially slow-wave sleep), the hippocampus replays stored episodes, reactivating the corresponding cortical patterns
3. the neocortex treats each replay as a new training example, making small weight adjustments
4. over many nights of replay, the neocortical weights gradually incorporate the statistical structure of the replayed episodes
5. as the neocortical representation strengthens, the hippocampal index becomes redundant. the memory is "consolidated" and can be retrieved from cortex alone

this process is systems consolidation: the gradual transfer of memory from hippocampus to neocortex over days, weeks, or months.

## the stability-plasticity dilemma

CLS theory is a specific solution to the more general **stability-plasticity dilemma** (Grossberg 1980): any learning system must balance **plasticity** (the ability to learn new information) against **stability** (the ability to retain old information).

ML analog: the stability-plasticity dilemma is the central problem in continual learning. biological CLS solves it with two systems and replay; ML methods like EWC, progressive networks, and experience replay are engineering approximations of the same principle.

the mathematical core: consider a network with weight matrix W that has been trained on data distribution P_old. a new experience x_new requires weight change Delta_W such that:

    loss(W + Delta_W, x_new) < loss(W, x_new)    [plasticity]
    loss(W + Delta_W, P_old) ≈ loss(W, P_old)      [stability]

for overlapping distributed representations, these two requirements conflict when Delta_W is large (fast learning). CLS resolves the conflict by assigning each requirement to a different system:

- hippocampus satisfies plasticity via sparse, non-overlapping representations (Delta_W is large but affects different weights for different memories)
- neocortex satisfies stability via small learning rate (Delta_W is small, so each update causes minimal disruption)
- replay transfers information between systems, converting the hippocampus's fast sequential learning into the neocortex's slow interleaved learning

## the 2016 update

Kumaran, Hassabis and McClelland (2016) updated CLS theory in three significant ways:

### replay is more than interleaving

the original theory treated replay as simple reactivation. the update recognized that replay is selective and goal-dependent. not all hippocampal memories are replayed equally -- emotionally significant, rewarding, or surprising events are replayed more frequently. this introduces a prioritized experience replay mechanism (cf. Schaul et al. 2016 in reinforcement learning).

### hippocampal generalization

the original theory assigned generalization exclusively to the neocortex. the update acknowledged that the hippocampus can support some forms of rapid generalization through recurrent reactivation of related memories. when a new experience triggers retrieval of similar stored episodes, the comparison enables inference beyond direct experience.

### schema-consistent rapid cortical learning

Tse et al. (2007) showed that rats can learn new place-flavor associations in neocortex rapidly (within 1-2 sessions) if the associations are consistent with an existing schema (prior knowledge structure). this challenged the strict "neocortex always learns slowly" claim. the update incorporated this finding: neocortical learning can be fast when new information is consistent with existing weight structure (the gradient of the new example aligns with the existing weight configuration rather than conflicting with it).

## computational models

### the original model (McClelland et al. 1995)

the original CLS model used a three-layer feedforward network trained on the properties of living things (e.g., "robin can fly", "salmon can swim"). sequential learning of new categories caused catastrophic interference. interleaved learning did not. the model showed that a hippocampal "fast encoder" that replayed stored patterns to the slow neocortical network could learn new categories without catastrophic forgetting.

### the O'Reilly-Norman model (2002)

O'Reilly and Norman (2002) developed a more biologically detailed CLS model with explicit hippocampal (sparse, conjunctive) and neocortical (distributed, overlapping) representations. they applied it to recognition memory and fear conditioning, showing that CLS principles could explain double dissociations between hippocampal and cortical memory in amnesic patients.

### modern relevance to machine learning

CLS has influenced several lines of machine learning research:

- **experience replay** in reinforcement learning (Mnih et al. 2015, DQN): store past transitions in a buffer, sample uniformly during training. directly inspired by hippocampal replay. ML analog: experience replay IS the ML implementation of CLS's replay mechanism
- **continual learning**: the field of avoiding catastrophic forgetting in neural networks (EWC, Kirkpatrick et al. 2017; progressive neural networks; PackNet) can be seen as engineering solutions to the problem CLS identifies
- **memory-augmented neural networks**: DNC (Graves et al. 2016), NTM, and other architectures that add an external memory to a neural network implement a computational analog of CLS: the external memory for fast storage, the network weights for slow learning

## relationship to todorov

todorov's KDA + MLA architecture presents a superficial structural parallel to CLS: KDA (dominant, recurrent, fast-writing) and MLA (minority, cache-based, exact retrieval). however, the analogy fails at the mechanistic level. CLS requires consolidation -- the transfer of information from the fast system to the slow system via replay. todorov has no consolidation mechanism. see [[matrix_memory_vs_hippocampus]] and [[memory_systems_to_matrix_memory_and_compressed_attention]] for detailed analysis.

## challenges

### schema-consistent learning complicates the dichotomy

the finding that neocortex can learn rapidly when new information is schema-consistent blurs the clean fast/slow division. the rate of neocortical learning depends not just on the learning rate but on the alignment between new experience and existing representations. this makes the hippocampal-neocortical distinction more about representation structure (sparse vs distributed) than about learning speed per se.

### replay timing and selection

the CLS theory requires that hippocampal replay occurs at the right time and replays the right memories. but the mechanisms for selecting which memories to replay, and when, remain poorly understood. sharp wave ripples (see [[memory_consolidation]]) provide a physiological substrate, but the selection criteria (emotional significance? prediction error? recency?) are still debated.

### interference within the hippocampus

CLS treats the hippocampus as immune to interference thanks to sparse representations. but hippocampal memories do interfere with each other, especially when similar experiences are encoded in close temporal proximity. the DG's pattern separation reduces but does not eliminate this interference. the theory underspecifies how the hippocampus manages its own limited capacity.

## key references

- McClelland, J. L., McNaughton, B. L. & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex: insights from the successes and failures of connectionist models of learning and memory. Psychological Review, 102(3), 419-457.
- Kumaran, D., Hassabis, D. & McClelland, J. L. (2016). What learning systems do intelligent agents need? Complementary learning systems theory updated. Trends in Cognitive Sciences, 20(7), 512-534.
- O'Reilly, R. C. & Norman, K. A. (2002). Hippocampal and neocortical contributions to memory: advances in the complementary learning systems framework. Trends in Cognitive Sciences, 6(12), 505-510.
- McCloskey, M. & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: the sequential learning problem. Psychology of Learning and Motivation, 24, 109-165.
- Grossberg, S. (1980). How does a brain build a cognitive code? Psychological Review, 87(1), 1-51.
- Tse, D. et al. (2007). Schemas and memory consolidation. Science, 316(5821), 76-82.
- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.

## see also

- [[hippocampal_memory]]
- [[memory_consolidation]]
- [[pattern_completion]]
- [[hebbian_learning]]
- [[homeostatic_plasticity]]
- [[matrix_memory_vs_hippocampus]]
- [[memory_systems_to_matrix_memory_and_compressed_attention]]
