# Context Extension Techniques

status: current (as of 2026-04-16).

This document covers five major approaches to extending transformer context
windows beyond their original training length.


## 1. CoPE -- Contextual Position Encoding

Paper: "Contextual Position Encoding: Learning to Count What's Important"
ArXiv ID: 2405.18719
Authors: Olga Golovneva, Tianlu Wang, Jason Weston, Sainbayar Sukhbaatar (Meta)

Mechanism:
- Standard position encodings use raw token indices, which cannot generalize
  to higher-level abstractions (e.g., "attend to the 3rd sentence")
- CoPE computes context-dependent gate values g_i for each token
- Positions are assigned via cumulative sum: p_i = sum_{j<=i} g_j
- The gate g_i is conditioned on token content, so positions represent
  counts of semantically relevant units (words, nouns, sentences, etc.)

Technical details:
- Gate computation: g_i = sigmoid(W_g * h_i) for each token embedding h_i
- Position: p_i = sum_{j=1}^{i} g_j (cumulative sum of gates)
- Position embedding is interpolated from a learned table using fractional p_i
- Attention: score(q_i, k_j) = q_i^T * k_j + q_i^T * pos_embed(p_i - p_j)

Results:
- Solves selective copy, counting, and Flip-Flop tasks where standard PE fails
- Improves perplexity on language and code modeling (20M-100M parameter models)
- Better generalization to longer contexts than training lengths

Limitations:
- Tested primarily on small models (up to ~100M parameters)
- Requires cumulative sum computation (sequential dependency)
- Not yet validated at large scale in production models


## 2. YaRN -- Yet Another RoPE extensioN

Paper: "YaRN: Efficient Context Window Extension of Large Language Models"
ArXiv ID: 2309.00071
Authors: Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole
Published: ICLR 2024

Mechanism:
- Extends RoPE-based models to longer contexts with minimal fine-tuning
- Three key ideas combined:
  1. NTK-by-parts interpolation: non-uniform scaling across RoPE dimensions
  2. High-frequency preservation: avoids compressing local distance information
  3. Attention temperature scaling: adjusts softmax entropy for longer sequences

Technical details:
- RoPE dimensions are partitioned by frequency:
  - High-frequency dims (short-range): no interpolation (preserved as-is)
  - Low-frequency dims (long-range): full interpolation (scaled by ratio s)
  - Medium-frequency dims: smooth ramp between the two extremes
- Temperature factor: sqrt(1/s) applied to attention logits to correct
  the entropy reduction caused by interpolation
- Interpolation ratio s = target_length / original_training_length

Proven context lengths:
- LLaMA-2 7B: extended to 32K, 64K, and 128K from 4K training length
- Fine-tuning cost: ~0.1% of original pretraining data (400 steps typical)
- 10x fewer tokens and 2.5x fewer training steps than prior methods

Limitations:
- Requires fine-tuning (though minimal)
- Diminishing returns beyond ~32x extension factor
- Does not fundamentally change the O(n^2) attention complexity


## 3. LongRoPE

Paper: "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens"
ArXiv ID: 2402.13753
Authors: Microsoft Research
Published: ICML 2024

Mechanism:
- Exploits two forms of non-uniformity in positional interpolation:
  1. Varying RoPE dimensions (different scaling per dimension)
  2. Varying token positions (non-uniform scaling along sequence)
- Uses an efficient evolutionary search to find optimal scaling factors

Technical details (three-stage approach):
1. Search: evolutionary algorithm finds optimal per-dimension rescaling
   factors, providing 8x extension without any fine-tuning
2. Progressive extension: fine-tune at 256K context, then apply a second
   round of positional interpolation to reach 2048K
3. Short-context recovery: readjust scaling factors at 8K length to
   restore short-context performance that may degrade during extension

Proven context lengths:
- Extended to 2,048K (2 million) tokens
- Fine-tuning: only ~1K steps at 256K length
- Maintains original architecture with minor positional embedding changes
- Passkey retrieval accuracy validated up to 2M context

Limitations:
- Search process required for finding optimal scaling factors
- Progressive extension adds training complexity
- Short-context readjustment step needed to avoid regression


## 4. Infini-Attention

Paper: "Leave No Context Behind: Efficient Infinite Context Transformers
        with Infini-attention"
ArXiv ID: 2404.07143
Authors: Tsendsuren Munkhdalai, Manaal Faruqui, Siddharth Gopal (Google)

Mechanism:
- Augments local causal attention with a compressive memory module
- Each attention head maintains both:
  1. Local masked attention (standard, within a segment)
  2. Long-term linear attention memory (accumulates across segments)
- The two outputs are combined with a learned gating parameter

Technical details:
- Memory update (linear attention style):
  M_s = M_{s-1} + sigma(K_s)^T * V_s       (associative memory update)
  z_s = z_{s-1} + sum(sigma(K_s))           (normalization term)
- Memory retrieval:
  A_mem = sigma(Q_s) * M_{s-1} / (sigma(Q_s) * z_{s-1})
- Combined output:
  O = gate * A_mem + (1 - gate) * A_local
  where gate is a learned sigmoid parameter
- sigma is a nonlinearity (e.g., ELU + 1) for the linear attention kernel

Proven context lengths:
- 1M token passkey retrieval (trained on 32K or even 5K sequences)
- 500K length book summarization
- Tested with 1B and 8B parameter LLMs
- Fixed memory footprint regardless of context length

Limitations:
- Compressive memory is lossy (cannot do exact retrieval of arbitrary past tokens)
- Performance on fine-grained retrieval tasks may lag full attention
- Requires careful tuning of segment size and memory update rules


## 5. StreamingLLM

Paper: "Efficient Streaming Language Models with Attention Sinks"
ArXiv ID: 2309.17453
Authors: Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis
Published: ICLR 2024

Mechanism:
- Discovers the "attention sink" phenomenon: initial tokens receive
  disproportionately high attention scores regardless of semantic content
- Proposes keeping a small number of initial tokens (attention sinks) plus
  a sliding window of recent tokens in the KV cache

Technical details:
- KV cache structure: [sink tokens (4)] + [recent window (last W tokens)]
- Total cache size: 4 + W tokens (W is configurable, e.g., 2048)
- No recomputation needed (unlike sliding window with recomputation)
- The first 4 tokens act as numerical anchors for softmax stability
- Optional: train with a dedicated [SINK] placeholder token for cleaner
  attention patterns

Proven context lengths:
- Stable language modeling up to 4 million tokens and beyond
- Tested on LLaMA-2, MPT, Falcon, and Pythia model families
- 22.2x speedup over sliding window recomputation baseline

Limitations:
- Cannot recall information that has left the sliding window
- Not suitable for tasks requiring exact long-range retrieval
- Information beyond the window is permanently lost (no compression)
- Best suited for streaming/dialogue applications, not long-document QA


## Comparison Table

    +----------------+-------------+----------+----------+----------------+
    | Method         | Max Context | Finetune | Exact    | Complexity     |
    |                |             | Needed   | Recall   | Change         |
    +----------------+-------------+----------+----------+----------------+
    | CoPE           | ~10x train  | Yes      | Yes      | O(n^2)         |
    | YaRN           | 128K        | Minimal  | Yes      | O(n^2)         |
    | LongRoPE       | 2,048K      | Minimal  | Yes      | O(n^2)         |
    | Infini-Attn    | Infinite*   | Optional | Approx   | O(n) memory    |
    | StreamingLLM   | Infinite*   | No       | Window   | O(W) memory    |
    +----------------+-------------+----------+----------+----------------+

    * "Infinite" means the method can process arbitrarily long sequences
      with bounded memory, but with information loss for distant tokens.


## References

- CoPE: arxiv 2405.18719
- YaRN: arxiv 2309.00071
- LongRoPE: arxiv 2402.13753
- Infini-Attention: arxiv 2404.07143
- StreamingLLM: arxiv 2309.17453
- LongRoPE2: arxiv 2502.20082
- Understanding RoPE Extensions: arxiv 2406.13282
- EleutherAI YaRN blog: https://blog.eleuther.ai/yarn/


---

## 6. Has Anyone Beaten Magic LTM-2-mini's 100M Context? (March 2026 Assessment)

As of March 2026, Magic's LTM-2-mini still holds the record for the
largest claimed context window at 100 million tokens. No publicly
documented model has surpassed this figure.

Source: https://magic.dev/blog/100m-token-context-windows


### 6.1 Current Landscape of Long-Context Models

    +----------------------+-------------+------------------+----------------+
    | Model                | Max Context | Practical Limit  | Public Access  |
    +----------------------+-------------+------------------+----------------+
    | Magic LTM-2-mini     | 100M        | Unknown (private)| No             |
    | Llama 4 Scout        | 10M         | ~1.4M            | Yes            |
    | Gemini 3 Pro         | 2M          | ~1M              | Yes            |
    | Grok 4               | 2M          | Unknown          | Yes            |
    | Claude               | 1M          | ~500K effective  | Yes            |
    +----------------------+-------------+------------------+----------------+

Key distinction: advertised vs practical context. The gap between
claimed maximum and effective usage is substantial for all models.

Sources:
- https://www.morphllm.com/largest-context-window-llm
- https://www.elvex.com/blog/context-length-comparison-ai-models-2026


### 6.2 LTM-2-mini Efficiency Advantage

For each decoded token, LTM-2-mini's sequence-dimension algorithm is
roughly 1000x cheaper than the attention mechanism in Llama 3.1 405B
for a 100M token context window.

Hardware comparison at 100M context:
- Llama 3.1 405B: requires 638 H100s per user just to store the KV cache
- LTM-2-mini: requires a small fraction of a single H100's HBM per user

This efficiency comes from their proprietary "Long Term Memory" mechanism,
which is not traditional attention. Details of the mechanism remain
undisclosed.

Source: https://magic.dev/blog/100m-token-context-windows


### 6.3 Quality Retention at Extreme Context: The "Context Rot" Problem

"Context rot" is the measurable performance degradation LLMs experience
as input length increases, even within the advertised context window.

Source: https://research.trychroma.com/context-rot

Key findings from 2025-2026 research:

Attention distribution patterns:
- Below 50% context fill: U-shaped recall (beginning + end favored)
- Above 50% context fill: recency bias dominates (end > middle > beginning)
- This means the "lost in the middle" problem is fill-dependent

Source: https://research.trychroma.com/context-rot

NVIDIA RULER benchmark results:
- Effective context is typically 50-65% of advertised maximum
- Recommended: stay below 50% of advertised window for high accuracy
- A 1M window model works best up to ~500K tokens
- A 200K model works best up to ~100K tokens

Source: https://www.morphllm.com/context-rot

Specific model degradation:
- Gemini 3.1 Pro (2M window): drops to 26% retrieval accuracy at 1M
- Gemini 3 Pro at 128K average: 77.0% recall
- Llama 4 Scout (10M window): 15.6% on Fiction.LiveBench long-context
  reasoning, requires 8xH100 for inference, practical limit ~1.4M

Sources:
- https://thinkpeak.ai/gemini-3-context-window-size-1-2m-tokens/
- https://www.morphllm.com/largest-context-window-llm


### 6.4 Techniques for Quality Retention at Extreme Context

No single technique has solved context rot, but several approaches show
promise as of March 2026:

1. Hybrid linear-attention architectures:
   - HypeNet (arxiv 2601.22156): trained to 128K, extrapolates to
     512K with 99.8% NIAH accuracy. Uses HyPE position encoding.
   - MiniCPM-SALA (arxiv 2602.11761): trained to 520K, extrapolates
     to 2048K with 81.6 RULER score (vs 89.37 at 128K). Uses
     sparse+linear attention hybrid.
   - These maintain quality better than pure Transformers at long
     contexts because linear attention layers do not suffer from
     the "lost in the middle" pattern.

2. Compaction (context summarization):
   - Distill context window contents at limit, reinitialize with summary
   - Serves as the primary lever for long-term coherence in agents
   - Trade-off: lossy compression, but prevents catastrophic degradation
   Source: https://blog.logrocket.com/llm-context-problem/

3. Semantic chunking + retrieval:
   - Split documents by sections/headings, not fixed token counts
   - High-precision retrieval filters before context injection
   - Reduces effective context load to relevant portions only
   Source: https://blog.logrocket.com/llm-context-problem/

4. Information discipline / context engineering:
   - Treat context as a first-class engineering concern
   - Deliberately filter, rank, prune, summarize, and isolate information
   - Dynamic tool loadout: limit tool descriptions to < 30 to prevent
     interference (above 30, tool selection accuracy degrades)
   Source: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents


### 6.5 Assessment for Todorov Project

For the Todorov project's purposes:
- No model has beaten 100M context as of March 2026
- The practical frontier for quality-preserving context is ~500K-2M tokens
- Hybrid architectures (HypeNet, MiniCPM-SALA) offer the best path to
  quality retention at extreme lengths
- The key insight is that linear attention layers avoid the U-shaped
  recall degradation of softmax attention, making hybrids naturally
  more robust to context extension
- HyPE position encoding (RoPE on linear layers, NoPE on attention)
  is emerging as a standard for length generalization


## Updated References

- Magic LTM-2-mini: https://magic.dev/blog/100m-token-context-windows
- Context rot study: https://research.trychroma.com/context-rot
- HypeNet / HALO: arxiv 2601.22156
- MiniCPM-SALA: arxiv 2602.11761
- Context length comparison 2026: https://www.elvex.com/blog/context-length-comparison-ai-models-2026
- Anthropic context engineering: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents


---

## 7. Passkey Retrieval Test: Implementation and Evaluation Protocol

Research date: 2026-03-22


### 7.1 Origin and Definition

The passkey retrieval task was proposed by Mohtashami & Jaggi (2023) in
the Landmark Attention paper (arxiv 2305.16300, NeurIPS 2023). It tests
whether a language model can recover a short piece of information (the
"passkey") hidden inside a long document filled with irrelevant text.

Source: arxiv 2305.16300


### 7.2 Standard Prompt Format

The test uses a simple, deterministic template:

    Prompt prefix:
        "There is a pass key hidden inside a lot of irrelevant text.
         Find it and memorize them. I will quiz you about what is
         the pass key later on."

    Filler text (repeated to desired context length):
        "The grass is green. The sky is blue. The sun is yellow.
         Here we go. There and back again."

    Needle (inserted at a controlled depth):
        "The pass key is {PASSKEY}. Remember it. {PASSKEY} is the
         pass key."

    Retrieval question (appended after filler):
        "What is the pass key? The pass key is"

    Expected output: the passkey digits verbatim.

Passkey format:
- Original (Mohtashami & Jaggi 2023): random 5-digit number
- Variant (nanotron dataset): 2-4 digit numbers
- Variant (InfiniteBench): 5-digit number embedded in more
  naturalistic filler text drawn from long documents

Sources:
- https://huggingface.co/datasets/nanotron/llama3-1024-passkey-retrieval-eval
- arxiv 2306.15595 (Position Interpolation, Chen et al. 2023)
- https://github.com/OpenBMB/InfiniteBench (arxiv 2402.13718)


### 7.3 Evaluation Parameters

The test varies two independent axes:

1. Context length: the total number of tokens in the prompt.
   Standard test points: 4K, 8K, 16K, 32K, 64K, 128K.

2. Depth percent: where the passkey is placed in the filler text.
   0% = beginning, 50% = middle, 100% = end.
   Typically evaluated at: 0%, 10%, 20%, ..., 90%, 100%.

This creates a 2D evaluation grid (context length x depth), which
is visualized as a heatmap showing retrieval accuracy at each cell.

Accuracy metric: exact match of the passkey digits. Binary pass/fail.
A model "passes" a context length if accuracy >= 80-95% (threshold
varies by paper; 95% is strict, 80% is lenient).

Sources:
- https://github.com/gkamradt/LLMTest_NeedleInAHaystack
- arxiv 2306.15595


### 7.4 RULER: Extended Retrieval Benchmark

RULER (arxiv 2404.06654, NVIDIA) extends passkey retrieval with more
demanding tasks. RULER evaluates at 4K, 8K, 16K, 32K, 64K, 128K.

RULER task categories:

    +-------------------------------+----------------------------------------+
    | Category                      | Tasks                                  |
    +-------------------------------+----------------------------------------+
    | Retrieval (NIAH variants)     | Single-key, Multi-key (MK-NIAH),       |
    |                               | Multi-value (MV-NIAH), Multi-query      |
    +-------------------------------+----------------------------------------+
    | Multi-hop tracing             | Variable tracking across chained        |
    |                               | logical connections                     |
    +-------------------------------+----------------------------------------+
    | Aggregation                   | Common words extraction, frequency-     |
    |                               | based word identification               |
    +-------------------------------+----------------------------------------+
    | Question answering            | SQuAD and HotpotQA embedded in          |
    |                               | long contexts                           |
    +-------------------------------+----------------------------------------+

Key/value formats: words, 7-digit numbers, or 32-digit UUIDs.
Haystack types: repeated noise, Paul Graham essays, distracted needles.

Effective context length: defined as the length at which a model's
RULER score drops below Llama-2-7B's 4K baseline of 85.6%.

Key finding: all models tested show performance degradation as
input length increases, even those achieving 100% on basic NIAH.

Sources:
- arxiv 2404.06654
- https://github.com/NVIDIA/RULER


### 7.5 Implementing Passkey Retrieval for Todorov Phase 2

For the Todorov project's context scaling tests (4K to 128K):

Recommended evaluation protocol:
1. Generate passkey prompts at lengths: 4K, 8K, 16K, 32K, 64K, 128K
2. At each length, test depths: 0%, 25%, 50%, 75%, 100%
3. Use 5-digit random passkeys (10 samples per cell for statistics)
4. Report: accuracy heatmap, mean accuracy per context length
5. Passing threshold: >= 95% accuracy at a given context length

Implementation approach:
- Use the standard filler text (repeated "The grass is green...")
- Tokenize filler text and passkey needle separately
- Insert needle at the target depth (measured in tokens)
- Pad/truncate filler to reach exact target context length
- Generate with greedy decoding, extract first 5 digits from output

For more rigorous evaluation, also run RULER at 4K, 16K, 32K, 64K, 128K
to test multi-key retrieval and aggregation capabilities.

The key diagnostic value of passkey retrieval for hybrid architectures:
- KDA layers rely on fixed-state memory; passkey tests whether the
  state can retain arbitrary information across the full context
- MLA layers have exact recall via cached KV; passkey tests whether
  MLA layers (placed every 4th layer) can compensate for any KDA
  information loss at longer contexts

Sources:
- arxiv 2305.16300, arxiv 2404.06654
- https://github.com/gkamradt/LLMTest_NeedleInAHaystack


### 7.6 Passkey Retrieval References

- Landmark Attention (passkey origin): arxiv 2305.16300
- Position Interpolation (passkey at scale): arxiv 2306.15595
- InfiniteBench (100K+ passkey): arxiv 2402.13718
- RULER (extended NIAH): arxiv 2404.06654
- RULER GitHub: https://github.com/NVIDIA/RULER
- NIAH GitHub: https://github.com/gkamradt/LLMTest_NeedleInAHaystack
- nanotron passkey dataset: https://huggingface.co/datasets/nanotron/llama3-1024-passkey-retrieval-eval


---

## 8. Passkey Retrieval at Small Scale: What Is Realistic for a 6M Parameter Model?

Research date: 2026-03-22


### 8.1 At What Parameter Count Does Passkey Retrieval Emerge?

No published work demonstrates passkey retrieval succeeding below ~130M
parameters. The smallest models tested on passkey retrieval in the
literature are:

    +------------------+--------+-----------+----------------------------------+
    | Model            | Params | State Size| Passkey Result                   |
    +------------------+--------+-----------+----------------------------------+
    | Mamba-2 (130M)   | 130M   | 16-128    | Near-zero at most lengths;       |
    |                  |        |           | worst among Mamba-2 checkpoints  |
    | Mamba-2 (370M)   | 370M   | 16-128    | Near-perfect within 8K; zero     |
    |                  |        |           | above 16K without continued      |
    |                  |        |           | pretraining; 256K with CPT       |
    | Infini-Attn (1B) | 1B     | --        | Perfect at 1M context            |
    | SmolLM2 (360M)   | 360M   | --        | Tested for RAG utilization, not  |
    |                  |        |           | raw passkey                      |
    +------------------+--------+-----------+----------------------------------+

The Mamba-2 130M checkpoint is explicitly called out as performing worse
than all other checkpoints: "Mamba-2 (except for the smaller 130M
checkpoint) has near-perfect retrieval accuracy within 8K tokens."

Source: arxiv 2410.07145 (Stuffed Mamba, COLM 2025)

The Zoology paper (Arora et al., ICLR 2024) uses 70M parameter models
as the smallest scale for associative recall benchmarks (MQAR). Even at
70M, linear attention and gated-convolution models struggle compared to
attention-based models on recall tasks.

Source: arxiv 2312.04927 (Zoology, ICLR 2024)

The Based paper (Arora et al., ICLR 2024) trains language models from
360M to 1.3B for recall-throughput tradeoff experiments. No results are
reported below 360M for language-scale passkey or NIAH tasks.

Source: arxiv 2402.18668 (Based, ICLR 2024)

Conclusion: 0% passkey retrieval at 6M parameters is expected, not a
bug. The task has never been demonstrated to work below ~130M parameters,
and even 130M Mamba-2 largely fails. The capability appears to require
at minimum hundreds of millions of parameters for standard softmax
attention, and even more for linear/recurrent architectures due to their
limited state capacity.


### 8.2 Why Passkey Fails at 6M Parameters: Fundamental Capacity Limits

Three independent lines of evidence explain the failure:

1. State capacity scales linearly with parameter count.

   A transformer with a single layer of self-attention followed by an
   MLP can obtain 100% accuracy on a synthetic factual recall task
   whenever either the total number of self-attention parameters or MLP
   parameters scales (up to log factors) linearly with the number of
   facts to recall.

   Source: arxiv 2412.06538 (Understanding Factual Recall in
   Transformers via Associative Memories)

   For a 6M parameter model, the total parameter budget limits how many
   KV associations can be stored. With ~6M parameters split across
   layers, each layer has far fewer parameters than a single passkey
   retrieval task demands when embedded in thousands of distractor tokens.

2. For recurrent/linear attention models, passkey recall length is
   exponential in state size, but state size is limited by model
   dimension.

   The Stuffed Mamba paper derives the formula:
       T_recall = 4.756 * (1.365^N_S - 1)^(-0.742)
   where N_S is the state size and T_recall is the maximum context length
   with near-perfect passkey retrieval (R^2 > 0.999).

   Source: arxiv 2410.07145 (Stuffed Mamba, COLM 2025)

   A 6M parameter model with d_model=256 and state_size=16 would yield
   T_recall in the low hundreds of tokens at best -- far below the
   typical passkey test range of 4K-128K.

3. Associative recall dominates the quality gap between attention and
   sub-quadratic models, and this gap widens at smaller scales.

   A 70M attention model outperforms a 1.4B gated-convolution model on
   associative recall. Poor AR performance accounts for >82% of the
   perplexity gap between attention-free models and transformers.

   Source: arxiv 2312.04927 (Zoology, ICLR 2024)

   At 6M parameters, a model has roughly 100x fewer parameters than
   the smallest models where AR begins to work. The recall bottleneck
   is correspondingly worse.


### 8.3 Better Retrieval Tests for Models at 6M Parameters

Since passkey retrieval is unrealistic at this scale, the following
synthetic tasks are appropriate diagnostics, ordered from easiest to
hardest:

1. Selective copy (simplest retrieval primitive).
   Task: given a sequence with marked positions, copy tokens from those
   positions to the output. Tests whether the model can attend to
   specific locations.
   Expected: solvable at very small scales with attention.
   Source: arxiv 2312.00752 (Mamba); arxiv 2405.18719 (CoPE)

2. Induction head / pattern completion.
   Task: given [A][B] ... [A] -> predict [B]. Tests second-order
   associative recall (match on A, copy successor B).
   Expected: solvable with 2+ layers and induction-capable attention.
   Mamba solves this and can extrapolate to >1M tokens.
   The time until induction head formation scales quadratically with
   context length.
   Source: arxiv 2209.11895 (In-context Learning and Induction Heads,
   Olsson et al. 2022); arxiv 2511.16893 (Predicting Formation of
   Induction Heads)

3. Single-query associative recall (SQAR).
   Task: store KV pairs, retrieve the value for a single queried key.
   Simpler than MQAR. Solvable at small scale if the number of KV pairs
   is kept small (e.g., 4-16 pairs in sequences of 64-256 tokens).
   Source: arxiv 2312.04927 (Zoology)

4. Multi-query associative recall (MQAR) -- reduced difficulty.
   Task: store KV pairs, retrieve values for multiple queried keys.
   At 6M parameters, limit to 2-4 queries with 8-16 KV pairs in
   sequences of 64-512 tokens. This is the standard benchmark from
   Zoology/Based but scaled down.
   Source: arxiv 2312.04927 (Zoology); arxiv 2402.18668 (Based)

5. Short-range passkey (custom, non-standard).
   Task: standard passkey format but with total context length of
   256-1024 tokens instead of 4K-128K. Tests the same retrieval
   mechanism at a scale compatible with 6M parameter capacity.

Recommended protocol for the Todorov 6M model:
- Test selective copy at lengths 64, 128, 256, 512, 1024
- Test induction heads at lengths 64, 128, 256, 512, 1024, 2048
- Test SQAR with 4, 8, 16 KV pairs at lengths 64, 128, 256, 512
- Test MQAR with 2-4 queries and 8 KV pairs at lengths 64, 128, 256
- Test short-range passkey at lengths 256, 512, 1024
- Only if these pass at high accuracy, attempt standard passkey at 2K+


### 8.4 How Linear Attention and Delta Rule Models Handle Retrieval

The delta rule provides the strongest retrieval among linear attention
variants, but still has fundamental limitations.

DeltaNet:
- Performs perfectly on the hardest MQAR setting and outperforms Mamba
  in low-dimension configurations.
- The delta rule minimizes MSE at each step, making it well-suited for
  AR where reducing large errors is critical.
- Limitation: poor state size scalability. Maximum supported state
  dimension was 256 in the original implementation. This constrains
  total memory capacity.
- At the 1.3B scale trained on 100B tokens, DeltaNet outperforms Mamba
  and GLA on perplexity and downstream tasks.

Source: arxiv 2406.06484 (DeltaNet, NeurIPS 2024)

Gated DeltaNet (ICLR 2025):
- Consistently surpasses Mamba2 and DeltaNet on language modeling,
  common-sense reasoning, in-context retrieval, length extrapolation.
- S-NIAH benchmark (synthetic passkey variant):
    DeltaNet:       S-NIAH-1: 97.4-99.0, S-NIAH-2/3: 14.4-22.4
    Gated DeltaNet: S-NIAH-1: 88.4-98.4, S-NIAH-2/3: 29.6-99.8 (at 8K)
- Hybrid variants (Gated DeltaNet-H1, H2) with sliding-window attention
  score 39.0-40.1 average on real-world retrieval, outperforming pure
  Transformer++ (37.0).
- Log-linear extensions preserve high accuracy as context length grows.

Source: arxiv 2412.06464 (Gated Delta Networks, ICLR 2025)

Key insight for Todorov: The delta rule's strong AR performance is
promising, but all published DeltaNet/Gated DeltaNet results use models
of 360M-1.3B+ parameters. At 6M parameters, the state matrix is too
small to store enough KV associations for standard retrieval benchmarks.
The model should first demonstrate competence on reduced-scale MQAR
(Section 8.3) before attempting passkey.


### 8.5 SSM/Mamba Models and Passkey at Small Scale

The Stuffed Mamba paper (COLM 2025) provides the most detailed analysis
of how SSM state size determines retrieval capacity:

Key findings:
- Maximum passkey retrieval context length scales EXPONENTIALLY with
  state size: T_recall = 4.756 * (1.365^N_S - 1)^(-0.742)
- State capacity (number of storable items) scales LINEARLY with state
  size.
- State collapse occurs when models are trained on contexts too short
  for their state size, enabling the model to succeed without learning
  to forget. The minimum training length to avoid collapse scales
  linearly with state size.

Mamba-2 passkey results by model size:
- 130M: near-zero accuracy at most lengths (worst checkpoint)
- 370M: near-perfect within 8K; zero above 16K without continued
  pretraining; near-perfect at 256K with continued pretraining
- 370M with continued pretraining significantly outperforms transformers
  of the same size on both retrieval accuracy and length generalization.

Source: arxiv 2410.07145 (Stuffed Mamba, COLM 2025)

Samba 1.7B (hybrid Mamba + attention):
- Fine-tuned on passkey retrieval with 4K training length
- Maintains perfect retrieval across all positions up to 256K context
- Demonstrates that hybrid SSM+attention architectures generalize length
  better than pure SSMs.

Source: SAMBA, ICLR 2025

Extrapolation to 6M parameters: With d_model likely around 128-256 and
state_size around 8-16, the Stuffed Mamba formula predicts a T_recall of
well under 1K tokens. This confirms that 0% passkey at standard lengths
(4K+) is the expected result, not a bug.


### 8.6 Byte-Level Models and Context Considerations

Byte-level models face a unique challenge: each character consumes one
position in the sequence, making effective context lengths 3-5x shorter
than token-level models for the same sequence length budget.

ByT5 (Google, 2022):
- Operates on sequences of 1024 bytes (~250 subword tokens equivalent).
- Outperforms mT5 at Small and Base configurations on English NLU.
- No passkey or long-context retrieval results published.
Source: ByT5 (Xue et al., 2022)

MegaByte (Meta, 2023):
- Multi-scale architecture: local submodel within patches, global model
  between patches. Can model sequences of >1M bytes.
- Effectively uses at least 8K bytes of context on PG19.
- Tested at 350M+ parameter scales. No small-model results.
Source: arxiv 2305.07185 (MegaByte)

Byte Latent Transformer (BLT, Meta, 2024):
- Dynamic patching based on next-byte entropy. Matches token-level
  performance at scale with improved robustness.
- Entropy model sizes tested: 1M to 100M parameters, with context
  windows from 64 to 512 bytes. Scaling performance positively
  correlates with both model size and context window length, with
  diminishing returns above 50M parameters for the entropy model.
- First FLOP-controlled scaling study of byte-level models up to
  8B parameters and 4T training bytes.
Source: arxiv 2412.09871 (BLT, Meta)

Implications for a 6M byte-level model:
- A sequence of 4096 bytes represents ~1000 tokens of English text.
  This means a "4K context" passkey test actually tests retrieval over
  the equivalent of ~1K tokens -- and even this is likely beyond the
  model's recall capacity at 6M parameters.
- No byte-level model below 100M parameters has been tested on passkey
  retrieval in the literature.
- The BLT finding that entropy models show diminishing returns above
  50M parameters suggests that 6M is below the useful floor for
  byte-level context modeling.
- A 6M byte-level model should be tested on retrieval tasks at
  byte-sequence lengths of 256-2048 (equivalent to ~60-500 tokens).


### 8.7 Progressive Training for Context Extension

The literature converges on a clear recipe for training models to handle
long contexts, applicable across model scales:

Step 1: Train on short sequences first.
- Variable Sequence Length (VSL) training: train primarily on short
  sequences (e.g., 2K), then adapt with the target long length (e.g.,
  8K). Saves 29% FLOPS vs training on long sequences throughout.
  Source: Cerebras VSL blog; NeurIPS 2024 paper (Dataset Decomposition,
  arxiv 2405.13226)

Step 2: Progressively increase context length.
- ProLong recipe (Princeton, ACL 2025): scale training to 64K over 20B
  tokens, then to 512K over another 20B tokens. Reset learning rate
  schedule and increase RoPE frequency base at each stage.
  Source: arxiv 2410.02660 (How to Train Long-Context LMs Effectively)

- LongRecipe: extend context using only 30% of target window size
  during training via impactful token analysis + position index
  transformation. Reduces compute by 85%.
  Source: arxiv 2409.00509 (LongRecipe, ACL 2025)

Step 3: Mix short and long data.
- Training ONLY on long data hurts long-context performance. It is
  critical to combine long data sources with high-quality short data.
- Training with sequence lengths beyond the evaluation target boosts
  long-context performance.
  Source: arxiv 2410.02660

Step 4: For SSMs/linear attention, ensure training length exceeds the
state collapse threshold.
- Minimum training length to learn proper forgetting scales linearly
  with state size. Training on sequences shorter than this threshold
  causes state collapse -- the model never learns to overwrite old
  information, leading to catastrophic failure on longer sequences.
  Source: arxiv 2410.07145 (Stuffed Mamba)

Recommended progressive schedule for the Todorov 6M model:
- Phase A: Train at 512 bytes (warmup, 70% of pretraining compute)
- Phase B: Extend to 2048 bytes (25% of pretraining compute)
- Phase C: Extend to 8192 bytes (5% of pretraining compute)
- At each phase, mix 80% short + 20% target-length sequences
- Monitor induction head formation and SQAR accuracy at each transition
- Only attempt passkey-style tests after Phase C, and only at reduced
  scale (256-1024 byte context)


### 8.8 Summary: What Is Realistic at 6M Parameters

    +---------------------------+------+------+------+------+-------+
    | Task                      | 6M   | 70M  | 130M | 370M | 1B+   |
    +---------------------------+------+------+------+------+-------+
    | Selective copy (short)    | YES  | YES  | YES  | YES  | YES   |
    | Induction heads (short)   | YES  | YES  | YES  | YES  | YES   |
    | SQAR (4-16 KV, <512 tok)  | MAYBE| YES  | YES  | YES  | YES   |
    | MQAR (2-4 Q, <256 tok)    | MAYBE| YES  | YES  | YES  | YES   |
    | Short passkey (<1K tok)   | NO*  | MAYBE| MAYBE| YES  | YES   |
    | Standard passkey (4K+)    | NO   | NO   | NO*  | YES* | YES   |
    | NIAH (4K+)                | NO   | NO   | NO   | YES* | YES   |
    +---------------------------+------+------+------+------+-------+

    YES   = demonstrated or expected to work
    MAYBE = may work with favorable architecture/training
    NO    = not expected to work at this scale
    *     = conditional on architecture/training choices

Key takeaways:
- 0% passkey at 6M parameters is the expected result, not a bug.
- The minimum scale for reliable passkey retrieval appears to be ~370M
  parameters for SSMs (with continued pretraining) and likely similar
  or higher for transformers.
- At 6M parameters, focus on selective copy, induction heads, and
  reduced-scale associative recall as retrieval diagnostics.
- For byte-level models, all context lengths should be interpreted as
  3-5x shorter in effective information content compared to token-level
  models.
- Progressive training (short-to-long) is well-supported by the
  literature and should be used regardless of model scale.
- State collapse is a real risk for recurrent/linear attention models
  trained on sequences shorter than the state-size-dependent threshold.


### 8.9 References for Section 8

- Stuffed Mamba (state capacity, passkey scaling): arxiv 2410.07145
- Zoology (MQAR, recall gap): arxiv 2312.04927
- Based (recall-throughput tradeoff): arxiv 2402.18668
- DeltaNet: arxiv 2406.06484
- Gated DeltaNet: arxiv 2412.06464
- Factual recall in transformers: arxiv 2412.06538
- In-context learning and induction heads: arxiv 2209.11895
- Predicting induction head formation: arxiv 2511.16893
- MegaByte: arxiv 2305.07185
- Byte Latent Transformer (BLT): arxiv 2412.09871
- ProLong (training recipe): arxiv 2410.02660
- LongRecipe: arxiv 2409.00509
- Dataset Decomposition (VSL): arxiv 2405.13226
- Mamba: arxiv 2312.00752
- Samba: ICLR 2025
- CoPE: arxiv 2405.18719
- Revisiting associative recall: arxiv 2508.19029
- Zoology GitHub: https://github.com/HazyResearch/zoology
- Based GitHub: https://github.com/HazyResearch/based
- Gated DeltaNet GitHub: https://github.com/NVlabs/GatedDeltaNet
- Cerebras VSL: https://www.cerebras.ai/blog/variable-sequence-length-training-for-long-context-large-language-models
- LongRecipe GitHub: https://github.com/zhiyuanhubj/LongRecipe


---

## 9. run_008 Research Grounding: Perplexity Scaling and Progressive Training Forgetting

Research date: 2026-03-23


### 9.1 Perplexity Scaling +4% from 256 to 4096 Tokens: Assessment

run_008 eval BPB at different context lengths (model trained progressively to s2048):
    s256:  3.96 BPB
    s512:  3.98 BPB
    s1024: 4.04 BPB
    s2048: 4.07 BPB
    s4096: 4.12 BPB

Total degradation from s256 to s4096: +4.0% (3.96 to 4.12).

For well-trained large-scale models, perplexity IMPROVES as context length grows
(more context = better predictions). Mamba's pretraining perplexity improves as
context increases up to 1M tokens. This is the expected behavior for models
evaluated within or near their training context range.

Source: arxiv 2312.00752 (Mamba)
Source: arxiv 2603.15569 (Mamba-3)

However, perplexity DEGRADING with longer context is a known failure mode with
multiple possible causes:

1. Evaluation beyond training context. The model was trained progressively up to
   s2048 (200 steps per stage). Evaluation at s4096 is 2x extrapolation. Mamba-2
   models trained on 2K tokens show "perplexity rapidly increases to more than 40"
   beyond 10K tokens. For SSMs and linear attention, extrapolation degradation is
   expected but typically modest within 2x of training length.
   Source: arxiv 2509.19633 (Mamba Modulation)

2. Insufficient training compute. 200 steps per stage at 6M parameters is extremely
   limited. Published progressive training results use orders of magnitude more
   compute (ProLong: 20B tokens per stage; LongRecipe: 30% of target window).
   Source: arxiv 2410.02660 (ProLong, ACL 2025)

3. State capacity limits at small scale. At 6M parameters with d_model=256 and
   state_size likely 16-64, the fixed-size recurrent state cannot capture enough
   information to benefit from longer context. Longer sequences just add noise
   that the state cannot distinguish from signal.
   Source: arxiv 2410.07145 (Stuffed Mamba, COLM 2025)

Assessment: +4% degradation from 256 to 4096 is MILD and not alarming for a
6M-parameter model at this training scale. Specific comparisons:

- LongRoPE (large-scale): short-context perplexity degrades by ~10-20% after
  extension to long contexts before the short-context recovery step. After
  recovery, degradation is ~5-6% at 4K context for LLaMA-2-7B extended to 2048K.
  Source: arxiv 2402.13753 (LongRoPE, ICML 2024)

- ProLong (large-scale): "more long data consistently degrades short-context
  performance." MMLU drops from 66.5% to 64.7% (~3%) and GSM8K drops from
  44.7% to 40.1% (~10%) after long-context extension of Llama-3-8B.
  Source: arxiv 2410.02660 (ProLong, ACL 2025)

- Naive continual pretraining to 32K: Llama-3-8B short-text average accuracy
  drops from 55.16% to 51.00% (7.5% relative degradation).
  Source: https://www.emergentmind.com/topics/context-degradation-in-llms

For a 6M model with only 200 steps per stage, +4% perplexity degradation over a
16x context range is a relatively stable result. The concern is not the degradation
itself but rather that perplexity never improves with context -- this indicates the
model is not learning to USE the longer context, only tolerating it.

The real diagnostic: at well-trained scale, longer context should HELP. If BPB is
flat or worsening, the model has not learned effective long-range dependencies.
This is expected at 6M params (state too small) and 200 steps (too few).


### 9.2 Progressive Training: Short-Context Degradation

run_008 observed: training BPB was 3.31 at s256 stage, but eval BPB at s256 after
completing all four stages (through s2048) was 3.96. This is a 19.6% degradation
at the original short context length.

This is a well-documented phenomenon in the literature:

1. The ProLong paper (ACL 2025) establishes that training on longer contexts
   consistently degrades short-context performance. The recommended mitigation:
   60% long data + 40% short data ("ShortMix"). Training with 100% long data
   "significantly hurts downstream long-context performance" and degrades
   short-context metrics.
   Source: arxiv 2410.02660

2. LongRoPE explicitly includes a Stage 3 "short-context recovery" step to
   mitigate this. Before recovery, LLaMA-2-7B extended to 2048K showed
   perplexity of 4.51 at 4K context (vs original ~3.5), a ~29% degradation.
   After recovery with re-searched RoPE scaling factors, this improved to
   3.85 (~10% degradation). The recovery step is essential.
   Source: arxiv 2402.13753 (LongRoPE)

3. For SSMs and linear attention models specifically, the Stuffed Mamba paper
   identifies "state collapse" when training sequences are too short relative
   to state size, and the inverse problem: catastrophic forgetting of short-
   context patterns when training shifts entirely to long contexts. The minimum
   training length for proper forgetting behavior scales linearly with state size.
   Source: arxiv 2410.07145 (Stuffed Mamba)

4. Progressive schedules (GrowLength, SkyLadder) that start short and grow to
   full length achieve 1.5x faster convergence and 2-3% lower perplexity, but
   ONLY when earlier-stage data continues to be mixed in during later stages.
   Without mixing, short-context capabilities degrade.
   Source: https://www.emergentmind.com/topics/pretraining-context-length

Assessment: the 19.6% short-context degradation in run_008 is EXPECTED and
consistent with published results. The run_008 progressive schedule trained
each stage exclusively on the target context length (no mixing of shorter
sequences), which is known to cause maximal short-context forgetting.

Mitigations for future runs (in priority order):

1. Data mixing during progressive stages: at each stage, use 60-80% target-
   length sequences and 20-40% shorter sequences from earlier stages. This is
   the single most impactful fix.
   Source: arxiv 2410.02660 (ProLong)

2. Short-context recovery step: after the final long-context stage, run a brief
   fine-tuning pass (~50-100 steps) on short-context data to recover degraded
   short-context quality. LongRoPE demonstrates this reduces degradation from
   ~29% to ~10%.
   Source: arxiv 2402.13753 (LongRoPE)

3. Learning rate reset at each stage: the ProLong recipe resets the learning
   rate schedule at each stage transition. This prevents the optimizer from
   being stuck in a low-LR regime that cannot adapt to the new context length
   while also allowing re-learning of short-context patterns.
   Source: arxiv 2410.02660 (ProLong)

4. Regularization toward previous checkpoint: add a small KL-divergence or
   L2 penalty anchoring the model toward its previous-stage weights. This is
   the continual-learning approach and limits catastrophic forgetting.
   Source: https://www.emergentmind.com/topics/context-degradation-in-llms

5. For SSM/linear attention specifically: ensure training length at each stage
   exceeds the state collapse threshold (proportional to state size). A model
   with state_size=64 needs training sequences of at least several hundred
   tokens to learn proper forgetting.
   Source: arxiv 2410.07145 (Stuffed Mamba)


### 9.3 References for Section 9

- Mamba: arxiv 2312.00752
- Mamba-3: arxiv 2603.15569
- Mamba Modulation (length generalization): arxiv 2509.19633
- ProLong (long-context training recipe): arxiv 2410.02660
- LongRoPE: arxiv 2402.13753
- Stuffed Mamba (state capacity): arxiv 2410.07145
- Context degradation: https://www.emergentmind.com/topics/context-degradation-in-llms
- Pretraining context length: https://www.emergentmind.com/topics/pretraining-context-length
- Perplexity for long-context LM: arxiv 2410.23771
