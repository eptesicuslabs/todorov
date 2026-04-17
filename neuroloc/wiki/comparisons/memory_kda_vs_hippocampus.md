# memory: KDA vs hippocampus

status: current (as of 2026-04-17).

**naming note (2026-04-17)**: "KDA" is an external paper name (Moonshot AI, 2025). the project's own component with the same mathematical form is `matrix memory`. this comparison remains valid AS a comparison between an outer-product decayed memory (whatever it is called in a given paper) and hippocampal memory. the comparison IS current; the name "KDA" in the title is legacy and should be read as "the outer-product decayed memory form" in the curriculum era. curriculum chapter ch. 19 (what the hippocampus does) covers the biological side; chapter ch. 29 (fast-weight memory) covers the mathematical side under the project's naming.

## the question

todorov's architecture has two memory systems: KDA (kernel delta attention, 18/24 layers) and MLA (multi-head latent attention, 3/24 layers). the hippocampus is the brain's fast episodic memory system. is KDA analogous to hippocampal memory? is the KDA+MLA split analogous to the complementary learning systems (hippocampus + neocortex)?

## dimension 1: learning speed

**hippocampus**: one-shot learning. a single experience modifies CA3 recurrent weights via rapid LTP. the animal can recall the event immediately after a single exposure. the dentate gyrus pattern-separates similar experiences to minimize interference.

**KDA**: one-step writing. each token writes a rank-1 update k_t * v_t^T to the state matrix S_t, gated by beta_t. this is immediate -- no iterative optimization required. in this narrow sense, KDA "learns" as fast as the hippocampus: one input, one memory trace.

**MLA**: no learning at all during inference. MLA caches the compressed latent c_kv for each token and retrieves via softmax attention. the weights are fixed after training. MLA's "memory" is a cache, not a learned association.

**verdict**: KDA's writing speed (one outer product per token) is structurally analogous to hippocampal fast binding. but the analogy is shallow: hippocampal learning modifies persistent synaptic weights via LTP, while KDA writes to a transient state that is recomputed from scratch each forward pass. KDA's "learning" is computation, not plasticity (see [[plasticity_to_kda_delta_rule]]).

## dimension 2: forgetting mechanism

**hippocampus**: interference-based forgetting. new memories stored in CA3 interfere with old ones when they share similar key features. this is proactive and retroactive interference. the dentate gyrus reduces but does not eliminate this interference. there is no biological "exponential decay" of hippocampal memories -- memories do not fade at a fixed rate. they are either consolidated to neocortex (and persist) or are overwritten by new, interfering memories.

**KDA**: exponential decay via alpha. S_t carries alpha^k fraction of the association written at timestep t-k, where alpha ~ 0.12 initially. this is indiscriminate -- all channels decay at the same rate, regardless of whether the memory is interfered with or not. there is no interference-based forgetting: a new key-value pair does not selectively erase an old pair with a similar key (unless the delta rule modification proposed in [[plasticity_to_kda_delta_rule]] is implemented).

**MLA**: no forgetting. every token's compressed latent is stored in the cache until the context limit is reached. there is no decay, no interference, no selective erasure.

**verdict**: the forgetting mechanisms are fundamentally different. hippocampal forgetting is interference-driven and content-dependent (similar memories compete). KDA forgetting is time-driven and content-independent (all memories decay at the same rate). the biological phenomenon KDA's alpha most resembles is short-term synaptic depression, not hippocampal memory loss.

## dimension 3: capacity

**hippocampus**: limited by interference, scaling as ~0.138 * N patterns for an auto-associative network of N neurons (McEliece et al. 1987). for CA3 with ~250,000 neurons, this gives ~34,000 patterns. but the sparse DG preprocessing increases effective capacity by reducing inter-pattern overlap.

**KDA**: limited by the rank of the state matrix S_t. each head maintains a head_dim x head_dim = 64 x 64 matrix. the maximum rank is 64 (after 64 linearly independent writes). subsequent writes either add to existing rank (if linearly independent of prior writes) or are absorbed into the existing subspace. with 16 heads, total rank capacity is 16 * 64 = 1024 entries. but exponential decay means that effective capacity is much less: after ~1/alpha ~ 8 timesteps, early writes have decayed by ~88%.

**MLA**: limited only by context length. stores one compressed latent (d_c = 128 dimensions) per token. at context length 2048, stores 2048 exact compressed representations. the retrieval is O(T^2) softmax attention over all stored tokens.

**verdict**: KDA's capacity is rank-limited and time-decayed, giving an effective memory horizon of O(1/log(1/alpha)) tokens. hippocampal capacity is interference-limited but persistent. MLA's capacity scales linearly with context length. none of these match exactly. the closest structural analog: KDA is like a small-capacity buffer with exponential forgetting, while MLA is like a large-capacity archive with exact retrieval.

## dimension 4: retrieval mechanism

**hippocampus**: nonlinear auto-associative retrieval via attractor dynamics. a partial cue activates a subset of CA3 neurons, recurrent excitation completes the pattern by settling into the nearest attractor. this is iterative (multiple recurrent cycles), nonlinear (threshold and saturation), and auto-associative (the network retrieves its own stored pattern).

**KDA**: linear heteroassociative retrieval. o_t = q_t^T * S_t is a single matrix-vector product. this is one-step (no iteration), linear (no attractor dynamics), and heteroassociative (the query q retrieves an associated value, not a completion of itself).

**MLA**: softmax attention retrieval. by Ramsauer et al. (2021), this IS the update rule of a modern Hopfield network with exponential storage capacity. MLA retrieval is one-step (for well-separated patterns), nonlinear (softmax), and heteroassociative (Q retrieves from K-V associations).

**verdict**: KDA's retrieval is the weakest of the three. it has no attractor dynamics, no error correction, and no ability to complete a noisy cue. hippocampal retrieval and MLA retrieval are both forms of pattern completion, but MLA (softmax attention = modern Hopfield update) is the closer mathematical analog. this inverts the naive mapping: MLA, not KDA, is the better hippocampal analog for retrieval.

## dimension 5: consolidation

**hippocampus**: active consolidation via sharp wave ripple replay during sleep and quiet rest (see [[memory_consolidation]]). the hippocampus replays stored sequences to the neocortex, gradually training cortical circuits to represent the memory independently. this is the defining feature of complementary learning systems.

**KDA+MLA**: no consolidation. there is no mechanism by which information in KDA's state is transferred to MLA's cache, or vice versa. the two systems do not communicate at the memory level. they share a residual stream, but each layer's memory (KDA state or MLA cache) is private to that layer.

**verdict**: this is the fatal disanalogy. complementary learning systems REQUIRE consolidation -- the directed transfer of information from the fast system to the slow system. without consolidation, KDA and MLA are independent memory systems operating in parallel, not complementary systems cooperating across timescales. the CLS analogy fails.

## the inversion problem

the Ramsauer et al. result creates a paradox for the naive KDA=hippocampus mapping:

**if softmax attention is a modern Hopfield network**, then MLA (softmax attention) is the system with:
- exponential storage capacity (~2^{d/2})
- one-step pattern completion
- content-addressable retrieval with attractor dynamics
- exact retrieval of stored patterns

these are hippocampal properties (fast retrieval, content-addressable, pattern completion). by the Hopfield analogy, MLA is the hippocampal system.

**KDA** (linear outer-product memory) is the system with:
- linear (rank-limited) capacity
- no attractor dynamics
- exponential forgetting
- simple Hebbian writing

these are closer to a classical associative memory with limited capacity -- more like a working memory buffer than like the hippocampus.

**but the ratio is inverted**: KDA is 75% of layers (dominant), MLA is 12.5% (minority). in biology, the hippocampus is a tiny fraction of the neocortex. if MLA is the hippocampal analog, then todorov has a tiny hippocampus (MLA, 12.5%) and a large neocortex (KDA, 75%). this ratio at least matches the biological architecture -- but only if we accept the inverted mapping (MLA=hippocampus, KDA=neocortex).

the inverted mapping also fails, because:
- the neocortex learns slowly by adjusting weights. KDA does not learn -- it computes
- MLA does not do one-shot learning -- it caches. there is no synaptic modification in MLA during inference
- both systems operate within the SAME forward pass, not across DIFFERENT timescales

## the honest verdict

the KDA+MLA split is NOT a complementary learning system. the analogy fails on three fundamental axes:

1. **no consolidation**: CLS requires replay-mediated transfer from fast to slow system. todorov has no inter-system communication at the memory level
2. **no timescale separation**: both KDA and MLA operate within the same forward pass (milliseconds on GPU). CLS operates across minutes (hippocampal encoding) to months (cortical consolidation)
3. **no learning asymmetry**: neither KDA nor MLA modifies its weights during inference. CLS requires the fast system to genuinely learn (modify synaptic weights via LTP) and the slow system to genuinely learn (modify synaptic weights via slow gradient descent or Hebbian learning)

what KDA+MLA IS: two memory ACCESS mechanisms with different capacity/precision tradeoffs, operating on a shared representation stream. KDA provides recurrent memory with O(1) cost per token but limited capacity. MLA provides exact retrieval with O(T^2) cost per token and unlimited (within context) capacity. this is a useful architectural design, but it is not CLS.

## the dissenting argument

the strongest case for the CLS analogy:

1. **functional complementarity is real**: KDA and MLA DO complement each other. KDA handles recent context efficiently (constant-cost recurrence), MLA handles the full history precisely (exact attention). this functional split -- "fast approximate" vs "slow exact" -- resembles the hippocampus (fast binding) vs cortex (slow integration) split, even if the mechanisms differ
2. **the absence of consolidation may be a feature**: biological CLS evolved because the organism has limited time for a single computation (a forward pass). a GPU has effectively unlimited computation time per token. the "need" for consolidation arises from the biological constraint of finite processing time per experience, which does not apply to autoregressive models that process one token at a time
3. **training IS consolidation**: during training (not inference), gradient updates modify both KDA and MLA weights based on all training data. this is the "slow neocortical learning" -- many repetitions gradually shaping weight structure. the distinction between training (slow, many examples) and inference (fast, one-shot computation) parallels the CLS distinction
4. **the ratio argument works under inversion**: if MLA (12.5%) is the hippocampal fast-retrieval system and KDA (75%) is the neocortical slow-state system, the ratio matches biology

the rebuttal: (1) functional complementarity without consolidation is just "two modules with different tradeoffs," which describes every hybrid architecture. (2) the absence of consolidation means there is no mechanism for KDA's transient memories to become permanent, which is the CORE function of CLS. (3) training-as-consolidation applies to ALL neural networks, not just those with KDA+MLA, making the analogy trivially true and therefore uninformative. (4) the ratio argument requires accepting the inverted mapping, which creates its own problems (KDA has no slow learning, MLA has no fast learning).

## summary table

| dimension | hippocampus | KDA | MLA |
|-----------|-------------|-----|-----|
| learning speed | one-shot (LTP) | one-step (outer product) | none (cache) |
| forgetting | interference-based | exponential decay (alpha) | none |
| capacity | ~0.138N patterns | rank(S) <= head_dim | T tokens |
| retrieval | nonlinear attractor | linear readout | softmax attention |
| consolidation | SWR replay -> cortex | none | none |
| representation | sparse (~2-5%) | ternary (~41%) | dense (continuous) |
| timescale | seconds to months | within forward pass | within forward pass |

## key references

- Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. PNAS, 79(8), 2554-2558.
- Ramsauer, H. et al. (2021). Hopfield networks is all you need. ICLR 2021.
- McClelland, J. L., McNaughton, B. L. & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex. Psychological Review, 102(3), 419-457.
- Millidge, B. (2024). Linear attention as iterated Hopfield networks. Blog post.

## see also

- [[hippocampal_memory]]
- [[complementary_learning_systems]]
- [[pattern_completion]]
- [[memory_consolidation]]
- [[plasticity_to_kda_delta_rule]]
- [[memory_systems_to_kda_mla]]
