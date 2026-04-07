# bridge: memory systems to KDA and MLA

## the biological mechanism

mammalian memory depends on two complementary systems operating across different timescales:

the hippocampus (see [[hippocampal_memory]]) provides fast episodic binding: the dentate gyrus pattern-separates incoming experiences into sparse orthogonal codes (~2-5% firing), CA3 stores these as auto-associative attractors via rapid LTP, and CA1 compares retrieved memories against current input. the hippocampal indexing theory (Teyler & DiScenna 1986) holds that the hippocampus stores compressed pointers to distributed cortical patterns, not the patterns themselves. storage capacity is limited by interference (~0.138N patterns for N neurons in a Hopfield-like auto-associative network).

the neocortex provides slow statistical learning: distributed, overlapping representations encode the shared structure across many experiences. synaptic modification is small per experience, preventing catastrophic interference (see [[complementary_learning_systems]]). capacity is effectively unlimited for practical purposes. retrieval depends on the accumulated weight structure, not on specific episode indices.

the bridge between them is consolidation (see [[memory_consolidation]]): during sleep, hippocampal sharp wave ripples (100-250 Hz) replay recently encoded sequences in compressed time (~5-20x speedup). this replay provides the interleaved training that the neocortex needs to incorporate new information without catastrophic interference. over days to months, the cortical representation strengthens and the hippocampal index becomes redundant. the memory is "consolidated."

modern Hopfield networks (Ramsauer et al. 2021, see [[pattern_completion]]) established that softmax attention IS the update rule of a continuous Hopfield network with exponential storage capacity C ~ 2^{d/2}. this means standard transformer attention implements pattern completion in a modern associative memory.

linear attention (the class to which KDA belongs) implements the update rule of a classical Hopfield network: Hebbian outer-product storage with linear readout. capacity scales linearly with dimension (rank-limited), not exponentially.

## the current todorov implementation

### KDA (src/layers/kda.py) as working/episodic memory

the state update:

    S_t = diag(alpha) * S_{t-1} + beta_t * k_t * v_t^T

the readout:

    o_t = q_t^T * S_t

properties:
- one-step writing via outer product (fast, like hippocampal LTP). each token produces a rank-1 update to the state matrix
- exponential forgetting via alpha (alpha ~ sigmoid(-2) ~ 0.12 initially, meaning ~88% decay per step). alpha is per-channel per-head, so different feature dimensions can decay at different learned rates
- content-addressable retrieval via q^T * S (like pattern completion, but linear -- no attractor dynamics, no error correction)
- capacity limited by rank of S: head_dim x head_dim = 64 x 64. effective rank is further limited by alpha decay -- after ~8 steps, early writes have decayed below 12% of original magnitude
- 18/24 layers = dominant memory system (75%)
- data-dependent write gating via beta_t = sigmoid(beta_proj(x_t)). this determines how strongly each token writes to memory
- channel-wise alpha: different feature dimensions can decay at different rates, learned during training

what KDA is NOT:
- it is not auto-associative (retrieves associated values, not completions of queries)
- it has no attractor dynamics (linear readout, no convergent iteration)
- it does not modify weights during inference (the state is a computation, not learning)
- it has no interference-based forgetting (all memories decay equally regardless of similarity)

### MLA (src/layers/mla.py) as retrieval/reference memory

MLA performs softmax attention over compressed per-token representations:

    c_kv = kv_down_proj(x)              [d_model -> d_c = 128]
    k = k_up_proj(c_kv)                 [d_c -> d_model]
    v = v_up_proj(c_kv)                 [d_c -> d_model]
    attn = softmax(Q * K^T / sqrt(d)) * V

properties:
- stores exact per-token compressed latent c_kv in a cache. every token is preserved
- softmax attention: O(T^2) retrieval over ALL stored tokens. this IS a modern Hopfield network update (Ramsauer et al. 2021)
- compression: d_model -> d_c (e.g., 1024 -> 128) = ~8x KV compression. per-token cache cost is d_c + d_R = 128 + 32 = 160 floats
- 6/24 layers = minority retrieval system (25%)
- no forgetting: stores everything until context limit. no decay, no interference at the storage level
- RoPE applied to a shared low-rank rope projection (d_R = 32), reducing positional encoding cost

what MLA is NOT:
- it does not learn during inference (no weight modification, only cache growth)
- it has no consolidation mechanism (no transfer from MLA to anything)
- it does not interact with KDA state (each layer's memory is private)

### the split

KDA forgets old associations to make room for new ones (fast, lossy). MLA retains everything but with compression (slow growth, exact retrieval). this produces functional complementarity:

- for recent tokens (within KDA's effective memory horizon of ~8-20 tokens), KDA provides O(1)-cost retrieval from the recurrent state. this is fast and efficient
- for distant tokens (beyond KDA's decay horizon), MLA provides O(T^2)-cost exact retrieval from the cache. this is expensive but precise
- the 75/25 split means most layers use the cheap, approximate memory (KDA), with a minority using expensive, exact memory (MLA)

## the proposed change

### a consolidation mechanism for todorov

if todorov were to implement CLS, it would need a consolidation mechanism: a process that transfers salient information from KDA's transient state to a more permanent store. three approaches:

**approach 1: periodic state distillation**

every N tokens, distill the current KDA state into a "consolidated memory bank" stored alongside MLA's cache. during MLA attention, attend over both the token cache and the consolidated state summaries. this adds a slow-growing episodic summary alongside MLA's exact token cache.

    if t % N == 0:
        summary_t = compress(S_t)    [head_dim x head_dim -> d_c]
        memory_bank.append(summary_t)

risk: high. defining "compress" is non-trivial. the KDA state is a head_dim x head_dim matrix; reducing it to a d_c vector requires choosing what information to preserve. the wrong compression throws away exactly the information that consolidation should preserve.

**approach 2: experience replay during training**

during training, periodically replay stored KDA states from earlier in the sequence, using them as additional training signal. this is directly analogous to hippocampal replay: past states are reactivated and used to train the model.

risk: moderate. requires storing intermediate KDA states (memory cost) and defining a replay objective (what loss function? reconstruction of the original input? prediction of future tokens?). the replay objective is not obvious.

**approach 3: KDA state prediction loss**

add an auxiliary loss that trains MLA to predict what KDA's state would contain for a given context. this forces MLA to learn a compressed version of KDA's memory, implementing a form of knowledge distillation from fast to slow system.

    L_consolidation = ||MLA_output - detach(KDA_output)||^2

risk: moderate. the MLA output and KDA output are not directly comparable (different layer positions, different representation spaces). would need careful alignment of the targets.

### assessment

all three approaches add complexity with uncertain benefit. the fundamental question: does todorov NEED consolidation? CLS in biology exists because the organism must handle experiences that span seconds to years. a language model processes a finite context window in a single forward pass. there is no "overnight sleep" between tokens. the timescale that motivates CLS (fast encoding now, slow consolidation later) does not exist in autoregressive inference.

the strongest argument for consolidation: at very long context lengths (100K+ tokens), KDA's decay means early tokens are completely forgotten. a consolidation mechanism could preserve a compressed summary of the early context, extending effective memory beyond the decay horizon without the O(T^2) cost of full attention. but this is an engineering optimization, not a biological principle.

recommendation: do not implement consolidation for phase 5. the complexity-benefit tradeoff is unfavorable at current context lengths (2048-8192 tokens). revisit if/when todorov scales to 100K+ context, where KDA's decay becomes a genuine limitation rather than a feature.

## expected impact

### what the CLS analysis reveals

the most valuable insight from the CLS analysis is not the analogy itself (which fails at the mechanism level) but the identification of what todorov's memory architecture is actually doing:

1. **KDA is a linear associative memory** -- a classical Hopfield network with Hebbian writing and exponential decay. its capacity is rank-limited, its retrieval is linear, and it has no attractor dynamics. this is working memory, not episodic memory
2. **MLA is a modern Hopfield network** (via the Ramsauer et al. equivalence) with exponential capacity and one-step pattern completion. this is the mathematically powerful retrieval system
3. **the 75/25 split is an efficiency decision**, not a biological principle. KDA is O(1) per token, MLA is O(T^2). the split allocates most layers to the cheap system (KDA) and a minority to the expensive system (MLA), maximizing throughput
4. **the absence of consolidation is not a gap** at current scale. it would become one at very long context lengths where KDA's decay horizon becomes limiting

### what the analysis does NOT reveal

the CLS analogy does not predict whether todorov's memory architecture is optimal. it identifies structural correspondences and differences but cannot prescribe design changes with predictable outcomes. the only reliable way to determine whether consolidation, interference-based forgetting, or attractor dynamics would improve todorov is empirical testing via the autoresearch loop.

## risk assessment

| modification | expected benefit | risk | recommend |
|-------------|------------------|------|-----------|
| periodic state distillation | extended memory horizon | high (compression design problem) | no, defer |
| experience replay during training | potential training stability | moderate (replay objective unclear) | no, defer |
| KDA state prediction loss | MLA learns KDA abstractions | moderate (alignment problem) | no, defer |
| interference-based forgetting (delta rule) | better memory utilization | moderate (see [[plasticity_to_kda_delta_rule]]) | yes, phase 5b+ |
| attractor dynamics in KDA | error correction in retrieval | high (changes recurrence structure) | no, defer to phase 6+ |

## key references

- McClelland, J. L., McNaughton, B. L. & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex. Psychological Review, 102(3), 419-457.
- Ramsauer, H. et al. (2021). Hopfield networks is all you need. ICLR 2021.
- Teyler, T. J. & DiScenna, P. (1986). The hippocampal memory indexing theory. Behavioral Neuroscience, 100(2), 147-154.
- Kumaran, D., Hassabis, D. & McClelland, J. L. (2016). What learning systems do intelligent agents need? Trends in Cognitive Sciences, 20(7), 512-534.
- Klinzing, J. G., Niethard, N. & Born, J. (2019). Mechanisms of systems memory consolidation during sleep. Nature Neuroscience, 22(10), 1598-1610.

## see also

- [[hippocampal_memory]]
- [[complementary_learning_systems]]
- [[memory_consolidation]]
- [[pattern_completion]]
- [[plasticity_to_kda_delta_rule]]
- [[memory_kda_vs_hippocampus]]
