# bridge: biological attention to MLA

## the biological mechanism

selective attention is the brain's solution to a resource allocation problem: sensory input vastly exceeds processing capacity, and the organism must direct its limited resources toward the most behaviorally relevant stimuli.

the core mechanism is biased competition (Desimone & Duncan 1995, see [[selective_attention]]): multiple stimuli compete for neural representation through mutual suppression, and top-down signals from prefrontal cortex bias the competition in favor of task-relevant items. the winning stimulus receives enhanced firing rate (+20-50%), reduced noise correlations, increased gamma-band synchrony, and effective receptive field shrinkage around it. the losing stimuli are actively suppressed.

the mathematical framework is the normalization model (Reynolds & Heeger 2009, see [[normalization_model_of_attention]]): attention acts as a multiplicative gain A on the stimulus drive E before divisive normalization:

    R = (A * E)^n / (sigma^n + sum(w * (A * E)^n))

this produces contrast gain or response gain depending on the relative sizes of the attention field and the normalization pool.

biological attention has five defining properties:
1. it is SELECTIVE: ~4 items attended out of thousands ([[feature_vs_spatial_attention]])
2. it is CAPACITY-LIMITED: Cowan's (2001) ~4-item limit, enforced by competitive dynamics
3. it uses COMPETITION: biased by top-down signals, resolved by [[winner_take_all]] normalization
4. it uses NORMALIZATION: [[divisive_normalization]] implements the competitive interaction
5. it uses OSCILLATIONS: gamma synchrony gates communication, theta rhythmically samples attention (see [[neural_synchrony]], [[gamma_oscillations]])

critically, biological attention is a MODULATION of ongoing processing, not a separate computation. there is no "attention layer" in the brain. any cortical circuit can be attention-modulated by top-down feedback and neuromodulatory input. attention changes HOW existing circuits process information, not WHAT computation they perform.

## the current todorov implementation

### MLA (src/layers/mla.py) -- 6/24 layers (25%)

MLA performs softmax attention over compressed per-token representations:

    c_kv = kv_down_proj(x)                    [d_model -> d_c = 128]
    k = k_up_proj(c_kv)                       [d_c -> d_model]
    v = v_up_proj(c_kv)                       [d_c -> d_model]
    k_rope = apply_rope(k_rope_proj(c_kv))     [d_c -> d_R = 32]
    q_rope = apply_rope(q_rope_proj(x))        [d_model -> d_R = 32]
    scores = (Q @ K^T + Q_rope @ K_rope^T) / sqrt(d + d_R)
    weights = softmax(scores)
    output = weights @ V

properties:
- compressed KV cache: d_model -> d_c (1024 -> 128), ~8x compression per token
- content-based retrieval: every token attends to every past token via dot-product similarity
- softmax normalization: weights sum to 1 over all key positions
- causal masking: can only attend to past tokens (autoregressive constraint)
- no spatial structure: every token can attend to every other token regardless of distance
- no top-down modulation: attention is purely bottom-up from Q/K similarity
- no capacity limit: all T tokens processed simultaneously, no competition or suppression

### KDA (src/layers/kda.py) -- 18/24 layers (75%)

KDA performs content-addressable retrieval from a matrix-valued recurrent state:

    S_t = diag(alpha) * S_{t-1} + beta_t * k_t * v_t^T
    o_t = q_t^T * S_t

properties:
- no explicit attention weights computed
- content-addressable retrieval: the query q_t retrieves associated values from the state
- data-dependent write gating: beta_t = sigmoid(beta_proj(x_t)) controls write strength
- exponential decay: alpha controls forgetting rate per channel
- rank-limited capacity: head_dim x head_dim = 64 x 64 per head

### what MLA computes vs what biological attention computes

MLA answers the question: "given the current token's query, which past tokens have the most relevant content?" this is INFORMATION RETRIEVAL -- finding relevant context in a memory store.

biological attention answers the question: "given limited processing capacity and current task goals, which stimuli should receive enhanced processing?" this is RESOURCE ALLOCATION -- directing processing power to behaviorally relevant inputs.

these are different computational problems:
- information retrieval has no capacity constraint (the answer can reference any number of past tokens)
- resource allocation is defined by its capacity constraint (the whole point is that not everything can be processed)
- information retrieval is content-driven (Q/K similarity determines relevance)
- resource allocation is goal-driven (prefrontal task representations determine what is relevant)
- information retrieval constructs new representations (weighted sum of values)
- resource allocation modulates existing representations (gain control on sensory processing)

## the adversarial analysis: is MLA's 25% allocation biologically analogous?

### the surface analogy

biological attention is sparse: ~4 items are attended out of thousands of available stimuli. the brain uses strong attentional modulation infrequently relative to its total processing -- most neurons most of the time are not strongly attention-modulated.

MLA is 25% of layers (6/24). this is a minority of the total computation. the 3:1 ratio (KDA:MLA) means most processing is non-attentional (KDA).

surface reading: MLA is rare like attention is rare. both are minority systems.

### the deep disanalogy

this analogy fails on three axes.

**axis 1: level of sparsity.** biological attention is sparse at the ITEM level: out of all stimuli in the visual field, ~4 are attended. MLA is sparse at the LAYER level: out of 24 layers, 6 use softmax attention. but WITHIN each MLA layer, attention is NOT sparse: every query attends to every key position with nonzero weight (full softmax). biological attention produces near-zero processing of unattended items. MLA produces nonzero influence from every past token. these are different kinds of sparsity operating at different levels.

**axis 2: modularity vs modulation.** MLA is a MODULE: 6 specific layers at fixed positions in the processing hierarchy. you can point to exactly which layers "do attention." biological attention is a MODULATION: any cortical area can be attention-modulated at any time. there is no fixed layer that "does attention" -- V1, V4, IT, PFC can all be attention-modulated depending on task demands. the architectural concept of "some layers are attention layers" has no biological counterpart.

**axis 3: origin of the ratio.** the 3:1 (KDA:MLA) ratio was derived from ML engineering benchmarks. Kimi, Qwen3, and OLMo independently converged on approximately 75% recurrent / 25% attention as the optimal ratio for language modeling performance. this ratio was NOT derived from biological principles. the convergence across ML labs suggests it reflects a compute-accuracy tradeoff for autoregressive language modeling, not a biological constraint. the similarity to biological attention's sparsity is coincidental, not mechanistic.

### what IS biologically analogous in todorov?

**KDA's beta gate is the closest analog to biological selective gating.** beta_t = sigmoid(beta_proj(x_t)) is a data-dependent scalar that controls how strongly each token writes to the recurrent state. high beta: this token is salient, store it. low beta: this token is uninformative, ignore it. this is functionally similar to biological attention's enhancement of salient stimuli and suppression of irrelevant stimuli. the critical difference: biological attention operates through competition (biased winner-take-all), while beta operates through independent gating (each token's beta is computed independently, without reference to other tokens).

**precision weighting is the deepest conceptual parallel.** Feldman & Friston (2010) argue that attention IS precision optimization: the brain adjusts the gain (precision) of prediction errors to upweight reliable information and downweight noisy information (see [[precision_weighting]]). in todorov, the cross-entropy loss function implicitly does something similar: tokens that are hard to predict contribute more to the gradient, driving the model to allocate representational capacity to surprising tokens. but this is a training-time phenomenon (gradient-based learning), not an inference-time mechanism (real-time gain modulation). there is no per-token precision weighting during inference.

## proposed changes

### option 1: competitive attention in MLA (high complexity, moderate risk)

replace softmax with a competitive normalization mechanism in MLA layers:

    scores = Q @ K^T / sqrt(d)
    competition = divisive_norm(scores, sigma=sigma, n=2)
    weights = competition / sum(competition)

where divisive_norm applies the Reynolds-Heeger normalization equation along the key dimension. this would introduce winner-take-all dynamics into MLA, making it more biologically analogous: a few tokens would receive most of the weight, and losing tokens would be actively suppressed (not just passively downweighted by softmax).

risk: divisive normalization has different gradient properties than softmax. the suppression mechanism may destabilize training. the effective sparsity may reduce MLA's ability to aggregate information from many sources, which is its primary computational role. this change could make MLA worse at its engineering function (information retrieval) while making it more biologically faithful.

assessment: do NOT implement. the engineering function of MLA (full retrieval from compressed cache) is well-validated at both 6M and 267M scale. making it more biologically plausible would change its computational role, and the 3:1 architecture assumes MLA provides something KDA cannot (exact long-range retrieval). introducing competition into MLA would make it more like a sparse retrieval mechanism, which KDA already provides. the result would be two KDA-like systems, not complementary systems.

### option 2: attention-modulated beta gate in KDA (moderate complexity, low risk)

add a global "attention signal" that modulates beta across KDA layers:

    attention_signal = f(global_state)     [scalar or per-head]
    beta_t = sigmoid(beta_proj(x_t)) * attention_signal

where f is a small projection from a global state vector (e.g., running average of loss, spike density, state norm). this makes the write gate sensitive to global context, not just local token features. when the global state signals high uncertainty, beta is amplified (attend more strongly to incoming tokens). when the global state is stable, beta is dampened.

this is closer to biological attention in two ways: (1) the modulation is global and top-down (global state modulating local processing), and (2) the modulation is adaptive (changes with processing state, not fixed after training).

risk: the global state computation adds a dependency across layers that may complicate gradient flow. the signal may collapse to a constant during training if the projection is not properly regularized.

assessment: interesting but premature. defer to phase 6+. the beta gate already works well without global modulation, and the engineering benefit is unclear. this is a research idea, not a validated improvement.

### option 3: precision-weighted loss (moderate complexity, moderate risk)

add per-token precision weighting to the training loss:

    precision_t = 1 / (variance_estimate_t + epsilon)
    loss = sum(precision_t * cross_entropy_t) / sum(precision_t)

where variance_estimate_t is computed from the model's internal uncertainty about token t (e.g., entropy of the output distribution, or disagreement between heads). this makes the loss function attention-like: uncertain tokens receive higher weight (higher precision), driving the model to allocate capacity to hard tokens.

this is the closest analog to Feldman & Friston's precision weighting: the training signal itself is modulated by estimated reliability.

risk: the variance estimate introduces a second optimization problem (estimating precision) that may conflict with the primary optimization (predicting tokens). the interaction between precision estimation and representation learning is poorly understood.

assessment: interesting for phase 6+. requires careful design of the variance estimator and its gradient interaction with the primary loss.

## risk assessment

| option | complexity | risk | biological fidelity | engineering benefit | recommendation |
|---|---|---|---|---|---|
| competitive MLA | high | high | high | negative (degrades retrieval) | do NOT implement |
| modulated beta | moderate | low | moderate | unclear | defer to phase 6+ |
| precision-weighted loss | moderate | moderate | high | unknown | defer to phase 6+ |

the fundamental tension: making todorov's attention mechanisms more biologically faithful risks degrading their engineering function. biological attention evolved to solve a resource allocation problem that transformers do not have (unlimited parallel processing). importing biological attention's selectivity and capacity limits into a system that benefits from exhaustive parallel processing is solving a problem that does not exist.

the most honest assessment: the word "attention" in MLA and the word "attention" in neuroscience refer to different computations. MLA should be evaluated on its engineering merits (does it improve language modeling?), not on its biological fidelity. the bridge between biological attention and todorov is not through MLA but through KDA's gating mechanisms (beta, alpha) and through the training objective (cross-entropy as implicit precision weighting).

## key references

- desimone, r. & duncan, j. (1995). neural mechanisms of selective visual attention. annual review of neuroscience, 18(1), 193-222.
- reynolds, j. h. & heeger, d. j. (2009). the normalization model of attention. neuron, 61(2), 168-185.
- feldman, h. & friston, k. (2010). attention, uncertainty, and free-energy. frontiers in human neuroscience, 4, 215.

## see also

- [[selective_attention]]
- [[normalization_model_of_attention]]
- [[feature_vs_spatial_attention]]
- [[biological_vs_transformer_attention]]
- [[precision_weighting]]
- [[divisive_normalization]]
- [[memory_systems_to_kda_mla]]
