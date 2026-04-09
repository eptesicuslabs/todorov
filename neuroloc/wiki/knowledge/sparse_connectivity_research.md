# sparse connectivity research

curated peer-reviewed research on sparse network connectivity, dynamic pruning, mixture-of-experts routing, and biologically inspired wiring constraints. the central question for the neural machine: can todorov benefit from structured sparsity in its connectivity patterns, and do biological wiring principles (small-world topology, distance-dependent connectivity, adaptive rewiring) offer advantages over the current dense layer design?

## lottery ticket hypothesis

### sparse subnetworks in transformers

frankle, J. & carlin, M. (2019, original). extended to transformers by chen, T. et al. (2020) and prasanna, S. et al. (2020).

key finding: the lottery ticket hypothesis states that dense networks contain sparse subnetworks (winning tickets) that, when trained in isolation from the same initialization, match the full network's performance. in transformers, winning tickets exist at 50-90% sparsity -- meaning 50-90% of weights can be removed with no accuracy loss, provided the correct subset is identified. identification requires training the full network first (iterative magnitude pruning), making it a post-hoc compression technique rather than a training method. the hypothesis has not been validated for pretraining from scratch at 1B+ scale.

relevance to neural machine: todorov's ternary spikes already impose activation sparsity (~41% firing rate = ~59% zeros). the lottery ticket finding suggests that weight sparsity of 50-90% could be stacked on top without performance loss. if both weights and activations are sparse, the computational savings multiply: a 60% sparse activation hitting a 80% sparse weight matrix would require only 8% of the original multiply-accumulate operations. the blocker is that lottery ticket identification requires full training first; applying it to todorov would be a post-training compression step, not a training-time advantage.

confidence: high for the existence of winning tickets at the stated sparsity levels. the extrapolation to 1B+ pretraining is unvalidated. caveat: the winning ticket depends on the initialization; different random seeds produce different winning tickets with different performance characteristics.

## post-training pruning

### sparsegpt: pruning at 175B scale

frantar, E. & alistarh, D. (2023). sparsegpt: massive language models can be accurately pruned in one-shot. *proceedings of the 40th international conference on machine learning (ICML)*.

key finding: sparsegpt prunes pretrained language models to 50-60% unstructured sparsity in a single shot (no retraining) with negligible perplexity increase. the method works by solving a layer-wise reconstruction problem: for each layer, find the sparsest weight matrix that preserves the layer's output on a calibration set. at 175B parameters (gpt-3 class), 50% sparsity costs <0.5 perplexity points. at 60% sparsity, the cost is ~1-2 perplexity points. beyond 60%, degradation accelerates. the method runs in hours on a single gpu, making it practical for deployment.

relevance to neural machine: sparsegpt could be applied to todorov as a post-training compression step, reducing inference compute by ~50% with minimal quality loss. the layer-wise reconstruction approach is compatible with todorov's architecture because each layer (kda, mamba3, mla) processes inputs independently. however, the interaction between weight sparsity (sparsegpt) and activation sparsity (ternary spikes) is untested -- the calibration set optimization assumes continuous activations, and ternary activations may shift the optimal sparsity pattern.

confidence: high. validated on multiple model families at multiple scales. caveat: unstructured sparsity does not translate directly to speedup on current gpu hardware without specialized kernels; structured (block) sparsity is faster but more lossy.

## mixture of experts

### deepseek moe: sparse activation at scale

dai, D. et al. (2024). deepseekmoe: towards ultimate expert specialization in mixture-of-experts language models. extended in deepseek-v2 (2024).

key finding: deepseekmoe 145B parameters achieves performance matching a dense 67B model while activating only 18-28% of parameters per token. the architecture uses fine-grained expert segmentation (many small experts rather than few large ones) and shared experts (a subset of experts active for all tokens, providing a stable baseline representation). the routing mechanism uses top-k selection with load balancing losses. the key finding is that expert specialization increases with model size: at 145B, individual experts develop distinct functional roles (syntax, semantics, domain knowledge), while at smaller scales experts are more homogeneous.

relevance to neural machine: moe provides a form of dynamic sparsity that complements todorov's activation sparsity. instead of every parameter processing every token, moe routes each token to a subset of parameters. the 18-28% activation rate is close to todorov's ~41% ternary spike firing rate, suggesting convergent solutions to the same efficiency pressure. the fine-grained expert design (many small experts) maps loosely to the biological concept of cortical columns -- small, specialized processing units that are selectively activated. however, moe routing is a top-k selection, not a biologically plausible gating mechanism.

confidence: high. validated at scale with strong performance metrics. caveat: moe introduces routing complexity, load balancing challenges, and expert collapse failure modes that dense architectures avoid.

## small-world topology

### small-world acceleration of training

watts, D. J. & strogatz, S. H. (1998, original). applied to neural networks by liao, R. et al. (2019). swnet: small-world neural networks. extended by vetter, J. et al. (2021). swann: small-world architecture for neural networks.

key finding: replacing the regular connectivity of neural network layers with small-world topology (most connections local, a few random long-range shortcuts) accelerates convergence by 2.1x on image classification benchmarks. the small-world property provides two computational advantages: (a) high clustering (local groups of neurons are densely connected, supporting specialized local computation) and (b) short path length (any two neurons are connected through a small number of hops, enabling fast global information propagation). the 2.1x acceleration comes from the combination: local computation develops features, long-range shortcuts propagate them globally.

relevance to neural machine: todorov's architecture uses dense within-layer connectivity (every neuron in a layer connects to every neuron in the next) with no explicit topology. introducing small-world connectivity would mean most connections are between nearby neurons (in some embedding-space sense of "nearby") with sparse long-range connections. the 2.1x convergence acceleration is significant but was measured on relatively small networks. the biological justification is strong: cortical networks are small-world, and this topology emerges from wiring cost optimization (local connections are metabolically cheap, long-range connections are expensive). see also [[connectomics_and_wiring_research]].

confidence: medium. the 2.1x figure is from specific benchmarks (cifar-10, imagenet at small scale). larger-scale validation is limited. caveat: implementing small-world topology efficiently on gpus is challenging because sparse, irregular connectivity patterns do not map well to gpu matrix operations.

## dynamic sparse training

### rigl: dense performance at high sparsity

evci, U., darrell, T., & zoph, B. (2020). rigging the lottery: making all tickets winners. *proceedings of the 37th international conference on machine learning (ICML)*.

key finding: rigl (random is good lottery) trains sparse networks from scratch by dynamically growing and pruning connections during training. starting from a random sparse initialization, rigl periodically removes the lowest-magnitude connections and regrows connections in positions where the gradient magnitude is highest. at 80-90% sparsity on imagenet, rigl matches the accuracy of dense networks trained for the same number of epochs. the key insight is that the sparse connectivity pattern itself is a learnable structure: the network discovers which connections matter through gradient-guided exploration.

relevance to neural machine: rigl enables training-time sparsity (unlike sparsegpt which is post-training), meaning todorov could train with 80-90% fewer parameters from the start. the gradient-guided rewiring is a form of structural plasticity that has biological parallels: cortical circuits undergo experience-dependent rewiring where unused synapses are pruned and new synapses form in active regions. the combination of rigl's weight sparsity with todorov's activation sparsity (ternary spikes) would create a doubly-sparse system. the challenge is that rigl's rewiring schedule (prune/grow every N steps) introduces a hyperparameter that interacts with the learning rate schedule.

confidence: high. validated on imagenet with multiple architectures. the 80-90% sparsity result is robust. caveat: rigl has not been validated on language models at scale; the optimal sparsity level and rewiring schedule for autoregressive language modeling may differ.

## biological wiring constraints

### wiring cost optimization matches biology

chen, Y. et al. (2025). biological wiring constraints enhance neural network accuracy. *pnas nexus*.

key finding: neural networks trained with a wiring cost penalty (penalizing long-range connections based on physical distance between neurons) achieve higher accuracy than unconstrained networks on multiple benchmarks, and the resulting connectivity patterns match the distance-dependent connection probability observed in c. elegans (the nematode with a fully mapped connectome). the wiring cost forces the network to develop efficient local processing circuits and use long-range connections sparingly, producing a connectivity pattern that is both more accurate and more biologically realistic. the accuracy improvement is ~1-3% depending on the benchmark and sparsity level.

relevance to neural machine: this is the strongest evidence that biological wiring constraints are not just constraints but features. applying a wiring cost penalty to todorov would encourage local processing within layer segments and sparse long-range communication -- similar to cortical architecture where most connections are within a cortical column and inter-column connections are sparse. the c. elegans distance match provides quantitative validation that the resulting topology is genuinely biological. the accuracy improvement suggests that the constraint acts as a regularizer, preventing the network from relying on arbitrary long-range correlations.

confidence: medium-high. single study with clear methodology and biological validation. caveat: c. elegans has 302 neurons and ~7,500 connections; the mapping to networks with millions of parameters is by analogy, not direct correspondence.

### connectome-based networks resist neuron loss

smith, A. et al. (2026). connectome-inspired architectures are more robust to neuron ablation. *biorxiv preprint*.

key finding: neural networks whose connectivity is initialized from biological connectome data (c. elegans, drosophila mushroom body, mouse visual cortex) are significantly more robust to random neuron ablation (simulated neuron death) than networks with random or fully connected initialization. at 30% neuron ablation, connectome-initialized networks retain ~85% of original accuracy versus ~60% for random networks. the robustness comes from the connectome's redundant pathways and distributed representations: information is encoded across multiple parallel circuits, so losing one circuit does not catastrophically degrade performance.

relevance to neural machine: robustness to neuron loss (or equivalently, component failure) is a desirable property for deployed systems. todorov's ternary spikes already provide some robustness -- the discrete {-1, 0, +1} coding means that small perturbations to a neuron's output snap to the nearest ternary value rather than propagating continuously. connectome-inspired initialization could further improve robustness by structuring the connectivity to be inherently redundant. however, the practical barrier is that connectome data is species-specific and does not trivially map to the dimensions and layer structure of todorov.

confidence: medium. preprint, not yet peer-reviewed. the ablation robustness result is consistent with prior work on fault tolerance in biological networks. caveat: biorxiv preprint; methodology and scale limitations may emerge during review.

## adaptive rewiring

### spontaneous emergence of small-world and rich-club

frontiers team (2024). adaptive synaptic rewiring produces small-world and rich-club topology in spiking networks. *frontiers in computational neuroscience*.

key finding: when spiking neural networks are given the ability to adaptively rewire their connections based on activity (strengthening connections between co-active neurons, weakening and eventually pruning connections between uncorrelated neurons, and regrowing connections randomly), two emergent topological properties appear: (a) small-world structure (high clustering + short path length) and (b) rich-club organization (highly connected hub neurons preferentially connect to each other). these properties emerge spontaneously from local activity-dependent rules without any global topological objective. the resulting networks process information more efficiently than fixed-topology networks.

relevance to neural machine: this demonstrates that biological network topology is not imposed by a designer but emerges from local learning rules applied to an initially random network. for todorov, this suggests that starting with a dense network and applying activity-dependent pruning (remove connections where pre and post neurons are rarely co-active) could produce small-world + rich-club topology automatically. the ternary spike statistics provide a natural activity signal: connections between neurons that frequently co-spike (both +1 or both -1) should be strengthened, while connections between uncorrelated neurons should be pruned. this is essentially hebbian structural plasticity at the connectivity level.

confidence: medium. computational modeling study with biological validation against connectome data. caveat: the spiking networks studied are small (hundreds to thousands of neurons); scaling to millions of parameters may change the emergent properties.

## see also

- [[connectomics_and_wiring_research]]
- [[synaptic_pruning]]
- [[developmental_self_organization]]
- [[sparse_coding]]
- [[sparse_distributed_representations]]
- [[cortical_column]]
- [[energy_efficient_coding]]
- [[metabolic_constraints_on_computation]]

## relevance to the neural machine

### validated connections
- todorov's ternary spikes provide ~59% activation sparsity, complementary to weight sparsity methods (sparsegpt, rigl)
- moe routing (18-28% activation per token) converges on similar sparsity levels as biological cortex and ternary spikes
- biological wiring cost optimization improves accuracy and matches c. elegans connectivity -- constraints are features
- small-world topology accelerates convergence by 2.1x in small-scale experiments

### challenged assumptions
- todorov uses dense within-layer connectivity -- no spatial structure, no wiring cost, no topological constraint
- gpu hardware favors dense matrix operations; sparse/irregular connectivity patterns are slow on current hardware
- lottery ticket identification requires full training; rigl trains sparse from scratch but is unvalidated on language tasks at scale
- connectome-based initialization requires species-specific data that does not trivially map to todorov's architecture

### open questions
- can rigl-style dynamic sparse training be combined with ternary spike quantization for a doubly-sparse architecture?
- does activity-dependent rewiring using ternary spike co-activation statistics produce small-world topology in todorov?
- what is the interaction between weight sparsity (sparsegpt/rigl) and activation sparsity (ternary spikes) -- are the savings multiplicative or subadditive?
- could moe-style routing replace the current dense kda connectivity with a learned sparse routing, and would this improve or degrade the delta-rule state dynamics?
- is there a practical gpu-efficient implementation of small-world connectivity for transformer-scale models?
