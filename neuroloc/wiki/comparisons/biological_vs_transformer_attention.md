# biological attention vs transformer attention

status: current (as of 2026-04-16).

## the question

biological attention and transformer self-attention share the word "attention" but evolved under fundamentally different pressures and compute fundamentally different things. biological attention selects WHAT to process under severe resource constraints. transformer attention computes WHERE to look in memory to construct contextualized representations. this comparison examines whether the shared terminology reflects shared computation or mere linguistic coincidence.

## structured comparison

### dimension 1: selectivity mechanism

**biological attention**: competitive selection. multiple stimuli compete for representation through mutual suppression (see [[selective_attention]]). top-down bias from prefrontal cortex and parietal cortex resolves the competition in favor of task-relevant stimuli. bottom-up salience (sudden onset, luminance contrast, motion) also biases the competition. the result is a winner-take-all-like outcome: 1-4 items are fully represented, the rest are suppressed. the mechanism is [[divisive_normalization]] with an attention field (see [[normalization_model_of_attention]]).

**transformer self-attention**: softmax-weighted aggregation. every token computes dot-product similarity with every other token (Q @ K^T / sqrt(d)). the softmax converts these scores into weights that sum to 1 over the key positions. each token then aggregates a weighted combination of all value vectors. there is no true competition: softmax assigns nonzero weight to every position, and there is no suppression mechanism that silences losing tokens. the distribution is often sharply peaked (effective sparsity), but this is a property of the learned representations, not of a competitive circuit.

**verdict**: biological selectivity is achieved through active suppression (losers are silenced). transformer selectivity is achieved through passive weighting (losers get small but nonzero weight). the mechanisms are fundamentally different. biological attention is subtractive/divisive; transformer attention is multiplicative/additive.

### dimension 2: capacity

**biological attention**: severely capacity-limited. cowan (2001) established ~4 items as the capacity of visual attention and working memory. this limit is not a design choice but a consequence of the competitive dynamics: with [[winner_take_all]] normalization, only a few representations can survive simultaneous competition. the theta-gamma coupling model (see [[theta_oscillations]]) provides a mechanistic account: ~7 gamma cycles nest within one theta cycle, and each gamma cycle maintains one item. attention is the brain's strategy for coping with this bottleneck.

**transformer self-attention**: unlimited capacity per head. every token attends to every other token with no hard limit on how many tokens can be "attended." the constraint is computational (O(T^2) cost), not representational. within a single attention head, all T tokens are processed simultaneously. multi-head attention further increases capacity by allowing different heads to attend to different aspects of the input.

**verdict**: capacity limitation is a defining property of biological attention. the absence of capacity limitation is a defining property of transformer attention. they solve different problems: biological attention solves a resource allocation problem (scarce processing capacity must be directed to the most important stimuli); transformer attention solves an information integration problem (each representation must be enriched with relevant context from the entire sequence).

### dimension 3: temporal dynamics

**biological attention**: sequential and dynamic. attention shifts from one location or object to another over time, with each shift taking ~100-200 ms. the brain samples the visual scene through a sequence of fixations (saccades) at ~3-4 Hz, with attention preceding each saccade to the target location. within a fixation, covert attention can shift at ~7-8 Hz (theta-rhythmic attentional sampling, Landau & Fries 2012). attention has temporal dynamics: it engages, sustains, and disengages on a timescale of hundreds of milliseconds.

**transformer self-attention**: instantaneous and parallel. all pairwise similarities are computed in a single matrix multiplication. there is no temporal sequence of attention shifts. each layer's attention computation occurs simultaneously across all positions. the temporal structure of language is encoded in positional embeddings, not in the temporal dynamics of the attention mechanism itself.

**verdict**: biological attention is a process unfolding in time. transformer attention is a computation completed in one step. the sequential nature of biological attention is not a limitation to be overcome but a feature: it enables the brain to prioritize processing over time and to adaptively redirect resources when the environment changes.

### dimension 4: information flow direction

**biological attention**: bidirectional. top-down signals from prefrontal cortex bias competition in sensory cortex (feedback). sensory-driven salience signals propagate bottom-up and capture attention involuntarily (feedforward). the interaction between top-down and bottom-up signals determines the attentional state. this bidirectionality is critical: without top-down control, attention is stimulus-driven and reflexive; without bottom-up salience, attention cannot respond to unexpected events.

**transformer self-attention**: unidirectional within a layer. Q, K, V are all computed from the same input representation (self-attention). there is no separate "top-down" or "bottom-up" pathway. cross-attention (as in encoder-decoder architectures) introduces a form of directionality, but this is between two representations, not between a control signal and a stimulus. in causal language models, the attention mask enforces temporal directionality (can only attend to past tokens), but this is an information flow constraint, not an attentional control mechanism.

**verdict**: the absence of top-down control is the deepest difference. biological attention is a control mechanism that modulates processing based on goals and expectations. transformer attention is a content-based retrieval mechanism that operates on whatever representations are presented to it. the transformer has no equivalent of "deciding to pay attention to X because my current task requires it."

### dimension 5: computation type

**biological attention**: modulation of ongoing processing. attention does not compute new representations from scratch. it modulates the gain, synchrony, and noise properties of representations that are already being computed by sensory cortex. the modulation is multiplicative (firing rate gain), temporal (gamma synchronization), and statistical (noise correlation reduction). the attended stimulus is processed by the same circuits that would process it without attention, but more effectively.

**transformer self-attention**: construction of new representations. each attention head computes entirely new representations by aggregating information from all positions. the output of the attention layer is a new embedding that did not exist before the computation. the pre-attention and post-attention representations can be very different (unlike biological attention, where the pre-attention and post-attention representations are the same population of neurons firing at different rates).

**verdict**: biological attention is a modulatory operation (gain control on an existing computation). transformer attention is a constructive operation (creating new representations from scratch). this is perhaps the most fundamental difference and the strongest argument that the shared term "attention" is misleading.

### dimension 6: energy cost

**biological attention**: near-zero marginal cost. attention modulates neural responses through top-down feedback connections and neuromodulatory systems that already exist. the metabolic cost of deploying attention is a small fraction of the total cortical metabolic budget. the entire brain consumes ~20 W, and attention-related metabolic changes (measured by fMRI BOLD signal) represent a few percent of baseline activity.

**transformer self-attention**: O(T^2 * d) cost per layer. self-attention is one of the most expensive operations in a transformer, scaling quadratically with sequence length and linearly with model dimension. at T=2048, d=1024, a single attention head computes ~4 million pairwise dot products. this cost has driven extensive research into efficient attention mechanisms (linear attention, sparse attention, low-rank attention).

**verdict**: biological attention evolved under extreme energy constraints (~20 W total power budget). transformer attention evolved under computational constraints (GPU memory, FLOPs). the solutions reflect these different pressures: biology uses selective gating (cheap modulation of existing circuits); transformers use exhaustive comparison (expensive but parallelizable).

### dimension 7: normalization

**biological attention**: [[divisive_normalization]]. the attention field acts as a multiplicative gain on the input to a divisive normalization circuit. the denominator pools activity across a broad spatial and feature range. the normalization is adaptive: the pool changes with stimulus conditions, and the semi-saturation constant sigma is modulated by adaptation history.

**transformer self-attention**: softmax normalization. the attention weights are normalized by softmax: w_ij = exp(q_i @ k_j / sqrt(d)) / sum_j exp(q_i @ k_j / sqrt(d)). this ensures weights sum to 1. softmax is a form of normalization, but it operates over key positions (competitors for relevance), not over a spatial/feature pool (competitors for representation). the normalization pool in biological attention includes stimuli that are not being attended to; the softmax pool in transformers includes all tokens, including the one being queried.

**verdict**: both use normalization, but the pools and the computational roles differ. biological normalization implements resource competition (neurons with stronger inputs suppress neighbors). softmax normalization implements probabilistic weighting (tokens contribute proportionally to their relevance). the connection is mathematical (both are ratio operations) but not computational (they solve different problems).

## the strongest criticism

biological and transformer attention share the word "attention" but compute fundamentally different things.

**biological attention selects WHAT to process.** it is a control mechanism that allocates scarce processing resources to task-relevant stimuli. it evolved to solve the bottleneck problem: the brain receives far more information than it can process, and attention is the gatekeeper. the output of attention is not a new representation but an enhanced version of existing representations.

**transformer attention computes WHERE to look in memory.** it is a retrieval mechanism that constructs contextualized representations by aggregating information from all positions. it was designed to solve the long-range dependency problem: each token needs information from distant tokens to build its representation. the output of attention is a new representation composed from multiple sources.

these are different operations solving different problems. the terminological overlap has caused significant confusion in both neuroscience and machine learning. lonnqvist and bhatt (2023) argued that self-attention in vision transformers performs perceptual grouping (feature-similarity-based organization), not attention (goal-directed selective processing). the 2025 bioRxiv preprint "deficient executive control in transformer attention" showed that transformers lack the executive control component of biological attention: they fail at Stroop-like conflict resolution tasks that require suppressing a dominant response in favor of a task-relevant but weaker one.

## is KDA or MLA closer to biological attention?

### MLA as attention

MLA performs softmax attention over compressed per-token representations (see src/layers/mla.py). it computes explicit attention weights, performs content-based retrieval, and constructs new representations from aggregated information. this is closer to transformer attention than to biological attention. MLA is a retrieval mechanism, not a selective gating mechanism.

however, MLA has one property that biological attention shares: it is the minority system (3/24 layers = 12.5%). the schedule is (KDA,KDA,KDA,Mamba3,KDA,KDA,KDA,MLA) x 3 = 18 KDA + 3 Mamba3 + 3 MLA. the brain uses selective attention sparingly relative to total processing -- most cortical computation proceeds without strong attentional modulation. in this narrow sense, MLA's minority allocation echoes biological attention's sparsity.

but this analogy is shallow. within each MLA layer, attention is applied to ALL tokens (full softmax). there is no capacity limit, no competition, no suppression of losing tokens. biological attention is sparse at the ITEM level (few items attended). MLA is sparse at the LAYER level (few layers use attention) but dense at the token level (all tokens attended within each layer). these are different kinds of sparsity.

### KDA as implicit attention

KDA performs content-addressable retrieval from a matrix-valued state: o_t = q_t^T * S_t (see src/layers/kda.py). there are no explicit attention weights. the retrieval is implicit: the query selects information from the state by matrix-vector multiplication.

two KDA mechanisms have attention-like properties:

**the beta gate**: beta_t = sigmoid(beta_proj(x_t)) controls how much each token writes to the recurrent state. high beta = "this token is important, write it strongly." low beta = "this token is unimportant, write it weakly." this is a form of selective gating: not all tokens are treated equally. beta is data-dependent and learned -- it implements a form of bottom-up salience computation that determines what gets stored. this is closer to biological attention's selectivity than softmax attention weights are: beta actually GATES information (some tokens barely write to memory), while softmax merely WEIGHTS it (all tokens contribute something).

**the alpha decay**: alpha controls how quickly old state is erased. channel-wise alpha means different feature dimensions decay at different rates. this is a form of temporal attention: the effective memory horizon is shorter for some features than others. but alpha is fixed after training (not data-dependent), which makes it more like a structural property than an attentional mechanism.

### verdict

neither KDA nor MLA is a good analog for biological attention. MLA is a retrieval mechanism (WHERE to look in memory), not a selection mechanism (WHAT to process). KDA's beta gate is the closest analog to biological attention's selectivity: it implements data-dependent gating of information into memory, which is a form of selective processing. but beta operates on individual tokens, not on competition between tokens, and it has no top-down control signal.

the strongest biological analog for attention in todorov is arguably not in any specific layer type but in the LOSS function: cross-entropy loss drives the model to allocate representational capacity to tokens that are hard to predict, which is functionally similar to precision weighting (see [[precision_weighting]]). but this is a training-time phenomenon, not an inference-time mechanism.

## the 25% allocation question

is MLA's 12.5% layer allocation analogous to biological attention's sparsity?

**the analogy**: biological attention IS sparse -- you attend to ~4 items out of thousands. the brain uses strong attentional modulation rarely relative to total processing. most cortical neurons most of the time are not strongly attention-modulated. MLA is 12.5% of layers. both are minority systems.

**the disanalogy**: within each MLA layer, attention is NOT sparse. every query attends to every key (full softmax). biological attention's sparsity is at the item level (few items win the competition). MLA's sparsity is at the architectural level (few layers use this computation type). these are different dimensions of sparsity.

**the ML origin**: the 25% allocation was derived from engineering benchmarks (Kimi, Qwen3, OLMo systematic analysis), not from biological principles. the 3:1 ratio optimizes language modeling loss, not biological plausibility. the convergence of multiple ML labs on this ratio suggests it reflects an optimal compute-accuracy tradeoff for language, not a biological constraint.

**the deeper problem**: biological attention is NOT a separate module. it is a MODULATION of ongoing processing that can be applied to any cortical area, at any time, as needed. there is no "attention layer" in the brain. MLA IS a separate module: 6 specific layers in a 24-layer stack, operating at fixed positions in the processing hierarchy. the architectural metaphor of "some layers do attention" does not map onto the biological reality of "any circuit can be attention-modulated."

## dissenting argument

the comparison above may be too strict. the relevant question is not "does transformer attention replicate biological attention in mechanistic detail?" but "does transformer attention solve the same computational problem?"

one could argue that both biological and transformer attention solve an information selection problem: given limited downstream capacity (in the brain, limited working memory and motor bandwidth; in transformers, limited embedding dimensionality per token), which information sources should most strongly influence the current representation? both use similarity-based matching to answer this question (biological attention matches top-down templates to sensory input; transformers match queries to keys). both produce weighted aggregation of information sources.

the Ellwood (2024) result strengthens this view: short-term Hebbian synaptic potentiation can implement transformer-like attention in a biologically plausible network, where keys and queries are represented as spike trains and comparison occurs at individual synapses. this suggests that the mathematical operation of attention (dot-product similarity followed by weighted aggregation) may be a convergent solution to information selection, implementable by biological circuits even if the brain does not use it as the primary mechanism for selective visual attention.

the rebuttal: the computational problem is different. biological attention solves a RESOURCE ALLOCATION problem (which stimuli get access to limited processing). transformer attention solves an INFORMATION INTEGRATION problem (which context gets incorporated into each representation). resource allocation requires active suppression; information integration requires weighted averaging. these are mathematically distinct operations even if both involve weighting.

## key references

- desimone, r. & duncan, j. (1995). neural mechanisms of selective visual attention. annual review of neuroscience, 18(1), 193-222.
- reynolds, j. h. & heeger, d. j. (2009). the normalization model of attention. neuron, 61(2), 168-185.
- vaswani, a. et al. (2017). attention is all you need. advances in neural information processing systems, 30.
- lonnqvist, b. & bhatt, u. (2023). self-attention in vision transformers performs perceptual grouping, not attention. frontiers in computer science, 5, 1178450.
- doerig, a. et al. (2025). deficient executive control in transformer attention. bioRxiv, 2025.01.22.634394.
- ellwood, i. t. (2024). short-term Hebbian learning can implement transformer-like attention. PLOS computational biology, 20(1), e1011843.
- graziano, m. s. a. et al. (2024). from cognition to computation: a comparative review of human attention and transformer architectures. arXiv:2407.01548.
- feldman, h. & friston, k. (2010). attention, uncertainty, and free-energy. frontiers in human neuroscience, 4, 215.
- cowan, n. (2001). the magical number 4 in short-term memory: a reconsideration of mental storage capacity. behavioral and brain sciences, 24(1), 87-114.

## see also

- [[selective_attention]]
- [[normalization_model_of_attention]]
- [[feature_vs_spatial_attention]]
- [[divisive_normalization]]
- [[precision_weighting]]
- [[winner_take_all]]
- [[neural_synchrony]]
- [[matrix_memory_vs_hippocampus]]
