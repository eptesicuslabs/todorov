# Hybrid Architectures: Evidence for 3:1 Linear-to-Attention Ratio

status: current (as of 2026-04-16).

This document collects evidence from multiple independent research groups
converging on a 3:1 (or 75/25) ratio of linear attention layers to full
attention layers as the optimal hybrid architecture design point.


## 1. Kimi Linear (Moonshot AI)

Paper: "Kimi Linear: An Expressive, Efficient Attention Architecture"
ArXiv ID: 2510.26692
Released: October 2025

Architecture:
- 3:1 ratio of KDA (Kimi Delta Attention) to global MLA layers
- 48B total parameters, 3B activated (MoE)
- 1M token context length

Key results:
- First hybrid linear attention model to outperform full attention under
  fair comparisons across short-context, long-context, and RL scaling regimes
- KV cache reduction: up to 75% compared to full attention
- Decoding throughput: up to 6x faster at long contexts
- Pretrained on 5.7 trillion tokens

Attention variant:
- Linear layers use KDA (channel-wise gated DeltaNet, see kda_channel_gating.md)
- Full attention layers use MLA (see mla_compression.md)
- KDA is a refinement of Gated DeltaNet with channel-wise alpha gate


## 2. Qwen3-Next (Alibaba / Qwen Team)

Blog: https://blog.vllm.ai/2025/09/11/qwen3-next.html
Model card: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
Qwen3 Technical Report: arxiv 2505.09388
Released: September 2025

Architecture:
- 3:1 ratio of Gated DeltaNet to full attention layers
- 48 total layers; every 4th layer uses GQA attention, rest use linear
- 80B total parameters, 3B activated (MoE)
- 512 routed experts + 1 shared expert, 10 experts activated per token
- 65K+ context length

Key results:
- Uses scalar gate (one value per head) for Gated DeltaNet
- Multi-Token Prediction (MTP) for pretraining boost and inference speedup
- Zero-centered and weight-decayed layernorm for stability
- vLLM integration with Triton kernels from flash-linear-attention


## 3. Qwen3.5 (Alibaba / Qwen Team)

Blog: https://huggingface.co/blog/mlabonne/qwen35
Released: February 2026

Architecture:
- Continues the 3:1 hybrid attention layout from Qwen3-Next
- Three out of every four transformer blocks use linear attention (Gated DeltaNet)
- Every fourth block uses standard full attention
- Production-scale validation of the Qwen3-Next preview design

Key results:
- Confirms viability of hybrid architecture at production scale
- Combines Gated DeltaNet with sparse MoE
- Linear attention compresses input to fixed-size state per layer,
  enabling efficient long-context processing without O(n^2) memory


## 4. OLMo Hybrid (Allen Institute for AI)

Paper: "Olmo Hybrid: From Theory to Practice"
URL: https://allenai.org/papers/olmo-hybrid
Blog: https://allenai.org/blog/olmohybrid
Authors: William Merrill, Yanhong Li, et al.
Released: 2025

Architecture:
- 3:1 ratio of GDN (Gated DeltaNet) to full attention layers
- Based on OLMo 3 7B architecture
- Replaces the 75% sliding-window-attention layers with GDN layers
- Remaining 25% keep full multi-head attention

Key results:
- Matches OLMo 3 7B on MMLU with 49% fewer training tokens
- Trained on up to 6T tokens
- Represents ~2x training efficiency gain over OLMo 3 dense

Scaling experiment ranking (best to worst):
    1. Hybrid GDN 3:1 ratio
    2. Pure GDN (all RNN layers)
    3. Standard Transformer (all attention)
    4. Hybrid Mamba2
    5. Pure Mamba2

GDN block design details:
- q, k, v: linear projection -> short convolution -> SiLU
- L2 normalization on q, k for training stability
- alpha, beta: linear projection only
- Output: normalization -> gating -> output projection

Theoretical contribution:
- Proves hybrid models (attention + recurrence) are strictly more powerful
  than either transformers or linear RNNs alone
- Demonstrates problems related to code evaluation that neither
  transformers nor GDN can express individually


## 5. Systematic Analysis of Hybrid Linear Attention

Paper: "A Systematic Analysis of Hybrid Linear Attention"
ArXiv ID: 2507.06457
Released: July 2025

This paper provides the most rigorous ablation study of hybrid ratios.

Key findings:

Ratio ablation:
- Language-model accuracy varies by less than 1% across ratios
- Recall accuracy rises steadily as full-attention layers are added
- Recall saturates around 3:1 (linear:full) ratio
- Recommended range: 3:1 to 6:1 for Transformer-level recall

Best linear attention variants for hybridization:
- GatedDeltaNet: controlled forgetting via outer-product delta rule
- HGRN-2: controlled forgetting via gated diagonal decay
- Both prevent unbounded accumulation that plagues additive updates
- Both surpass Transformer baseline by 2-5 percentage points at optimal ratio

The paper confirms that the mechanism of controlled forgetting (whether via
delta rule or diagonal decay) is the critical ingredient, not the specific
architecture.


## 6. Convergence Summary

    +-------------------+-------+------------------+-------------------+
    | Model             | Ratio | Linear Attention  | Full Attention    |
    +-------------------+-------+------------------+-------------------+
    | Kimi Linear       | 3:1   | KDA (chan-gated)  | MLA               |
    | Qwen3-Next        | 3:1   | GDN (scalar-gat)  | GQA               |
    | Qwen3.5           | 3:1   | GDN (scalar-gat)  | Full attention    |
    | OLMo Hybrid       | 3:1   | GDN              | MHA               |
    | Systematic (rec)  | 3:1-6:1| GDN or HGRN-2    | Full attention    |
    +-------------------+-------+------------------+-------------------+

All four independent groups converge on the same ratio.

The 3:1 ratio appears to be a robust sweet spot because:
- 75% linear layers provide O(n) scaling for most of the computation
- 25% full attention layers provide sufficient exact-recall capability
- Adding more full attention beyond 25% yields diminishing returns
- The linear layers handle state tracking and information accumulation
- The full attention layers act as "checkpoints" for precise retrieval


## References

- Kimi Linear: arxiv 2510.26692
- Qwen3 Technical Report: arxiv 2505.09388
- Qwen3.5 blog: https://huggingface.co/blog/mlabonne/qwen35
- OLMo Hybrid: https://allenai.org/papers/olmo-hybrid
- OLMo Hybrid blog: https://allenai.org/blog/olmohybrid
- Systematic Analysis: arxiv 2507.06457
- Gated Delta Networks: arxiv 2412.06464
- Interconnects blog: https://www.interconnects.ai/p/olmo-hybrid-and-future-llm-architectures
- vLLM Qwen3-Next blog: https://blog.vllm.ai/2025/09/11/qwen3-next.html
- NVIDIA Qwen3-Next blog: https://developer.nvidia.com/blog/new-open-source-qwen3-next-models-preview-hybrid-moe-architecture-delivering-improved-accuracy-and-accelerated-parallel-processing-across-nvidia-platform/


---

## 7. HypeNet + HALO: Distillation-Based Hybrid Construction (January 2026)

Paper: "Hybrid Linear Attention Done Right: Efficient Distillation and
        Effective Architectures for Extremely Long Contexts"
ArXiv ID: 2601.22156
Submitted: January 29, 2026
Source: https://arxiv.org/abs/2601.22156


### 7.1 HALO Distillation Pipeline

HALO (Hybrid Attention via Layer Optimization) converts pre-trained
Transformer models into RNN-attention hybrids using only 2.3B tokens --
less than 0.01% of typical pre-training data. This is dramatically cheaper
than prior approaches (20-400B tokens).

Three-stage pipeline:
1. Initialization: transfer attention weights to initialize RNN layers
2. Hidden state alignment (1B tokens): MSE loss between RNN and
   attention outputs
3. Knowledge distillation (1B tokens): KL divergence between teacher
   (original Transformer) and student (hybrid) outputs
4. Long-context finetuning (additional tokens at extended lengths)


### 7.2 Layer Selection Method

HALO selects which attention layers to keep (25% retained, 75% converted
to RNN). The selection uses an importance scoring mechanism:

    importance(layer_i) = recall_drop(layer_i) / CSR_drop(layer_i)

Layers with the highest importance scores (large recall impact, small
general capability impact) are kept as full attention. This principled
selection outperforms uniform interleaving (every-4th-layer pattern).

The final ratio is 75% RNN layers / 25% attention layers, consistent
with the 3:1 consensus.


### 7.3 HyPE (Hybrid Position Encoding)

A novel position encoding scheme for hybrid architectures:

    - RNN layers: use RoPE for local positional cues
    - Attention layers: use NoPE (no positional encoding)

Rationale: NoPE in attention layers provides superior length
generalization because RoPE's frequency-based encoding degrades
at positions far beyond training length. RNN layers still benefit
from RoPE because they process sequences locally/recurrently.

Dynamic attention scaling using position-dependent factors is also
applied.


### 7.4 Performance Results

Compared to original Qwen3 models of the same size:
- CSR (general capability): 55.9 vs 58.5 (comparable)
- NIAH at 128K context: 99.8% vs 14.8% (dramatically better)
- Outperforms Jet-Nemotron and KL-LS on NIAH benchmarks
  (KL-LS achieved 68.4% at 128K; HypeNet reaches 99.8%)

Speedup at long contexts:
- 2.4x decoding speedup at 1M context length
- 3.0x decoding speedup at 512K context length
- 3.4x prefilling speedup at 512K context length

Parameter overhead: ~10% increase due to GQA-to-MHA expansion and
output gates, but offset by reduced KV cache requirements.

Source: arxiv 2601.22156


---

## 8. MiniCPM-SALA: Sparse + Linear Attention Hybrid (February 2026)

Paper: "MiniCPM-SALA: Hybridizing Sparse and Linear Attention for
        Efficient Long-Context Modeling"
ArXiv ID: 2602.11761
Submitted: February 11, 2026
Source: https://arxiv.org/abs/2602.11761
Model: https://huggingface.co/openbmb/MiniCPM-SALA


### 8.1 Architecture

MiniCPM-SALA uses a different hybrid combination than GDN+attention:
- 75% Lightning Attention (linear attention)
- 25% InfLLM-V2 (sparse attention, not full attention)

This is notable because it replaces the 25% full softmax attention layers
with sparse attention, further reducing compute while maintaining recall.
The 3:1 ratio (linear:non-linear) is preserved.

Layer placement uses the selection algorithm from Chen et al. (HypeNet/HALO)
rather than uniform interleaving, and achieves superior downstream
performance.


### 8.2 HyPE Position Encoding (shared with HypeNet)

Same scheme as HypeNet:
- RoPE applied to linear attention layers for position sensitivity
- RoPE removed from sparse attention layers to prevent long-distance decay

This confirms HyPE as an emerging standard for hybrid position encoding.


### 8.3 Training Efficiency

Continual training from MiniCPM-4.0 pre-trained weights. Five training
stages consuming ~2T tokens total. Total training budget is approximately
25% relative to training a comparable model from scratch, because the
architectural transformation preserves pre-trained knowledge.


### 8.4 Performance

9B parameter model:
- RULER benchmark at 128K tokens: 89.37 score
- Trained to 520K tokens, extrapolates to 2048K without significant
  degradation (81.6 score at 2048K)
- General capabilities (knowledge, math, coding) comparable to Qwen3-8B

Speed improvements on NVIDIA A6000D at 256K sequence length:
- Time-To-First-Token: 51.6s vs 180.8s for Qwen3-8B (3.5x speedup)

Edge deployment on NVIDIA RTX 5090 (32GB VRAM):
- Qwen3-8B: OOM failure at 128K tokens
- MiniCPM-SALA: scales to 1024K tokens without memory failure

Source: arxiv 2602.11761


---

## 9. Has the 3:1 Ratio Been Challenged? (March 2026 Assessment)

As of March 2026, the 3:1 (linear:full-attention) ratio has NOT been
overturned. Instead, it has been further reinforced by new evidence:

Evidence reinforcing the 3:1 ratio:
1. HypeNet (Jan 2026): uses 75% RNN / 25% attention, validates with
   principled layer selection that confirms ~25% attention is optimal
2. MiniCPM-SALA (Feb 2026): uses 75% linear / 25% sparse attention,
   same ratio but with sparse attention replacing full attention
3. Mamba-3 (Mar 2026): paper explicitly predicts "linear layers will be
   predominantly used in conjunction with global self-attention layers"

New development: the 25% non-linear component can be SPARSE attention
(not necessarily full softmax attention), as demonstrated by MiniCPM-SALA.
This opens a path to even more efficient hybrids where both components
scale sub-quadratically.

The Systematic Analysis paper (arxiv 2507.06457) continues to be the
definitive reference, with its finding that "recall saturates around
3:1 ratio" holding across all subsequent work.

No published work as of March 2026 has demonstrated a ratio other than
3:1 to 6:1 achieving better results at comparable compute.

Source: arxiv 2601.22156, arxiv 2602.11761, arxiv 2603.15569


## Updated References

- HypeNet / HALO: arxiv 2601.22156
- MiniCPM-SALA: arxiv 2602.11761
- Mamba-3: arxiv 2603.15569
- SoLA-Vision (fine-grained layer-wise hybrid): arxiv 2601.11164


---

## 10. Hybrid Architecture Context Extension: 4K to 128K

Research date: 2026-03-22


### 10.1 How Hybrid Models Scale Context

Hybrid models (linear attention + softmax attention in 3:1 ratio)
have a natural advantage for context extension because:

1. Linear layers (75%) maintain a fixed-size state regardless of
   context length. No KV cache growth, O(n) compute.
2. Softmax attention layers (25%) have growing KV caches but provide
   exact recall capability. O(n^2) compute, but only on 25% of layers.
3. The effective memory cost scales as: O(n) for linear layers +
   O(n * L_attn) for attention layers, where L_attn is the number
   of attention layers (25% of total).

This means a 3:1 hybrid with 24 layers has 6 attention layers. At
128K context, only 6 layers have full KV caches, reducing memory
by ~75% compared to a full-attention model of the same depth.

Sources:
- arxiv 2510.26692 (Kimi Linear)
- arxiv 2507.06457 (Systematic Analysis)


### 10.2 Perplexity Stability Across Context Lengths

Key findings from hybrid model evaluation at varying context lengths:

Samba (Mamba + SWA hybrid, arxiv 2406.07522):
- Trained on 4K sequences, extrapolates to 256K with stable perplexity
- Perplexity continues to improve up to 1M context length
- Finding: placing a full attention layer at the beginning of the model
  causes perplexity explosion at 16K; interleaved placement avoids this

HypeNet (RNN + attention hybrid, arxiv 2601.22156):
- 99.8% NIAH accuracy at 128K context
- 2.4x decoding speedup at 1M context
- Uses HALO distillation from pre-trained Transformer

MiniCPM-SALA (linear + sparse attention hybrid, arxiv 2602.11761):
- RULER 89.37 at 128K tokens
- Trained to 520K, extrapolates to 2048K with 81.6 RULER score
- On RTX 5090 (32 GB): scales to 1024K without OOM

General finding: "long-context quality rises steeply once a few full-
attention blocks are present, after which perplexity plateaus. The
linear:full ratio primarily controls recall, whereas language modeling
loss is comparatively insensitive."

Source: arxiv 2507.12442 (Characterizing SSM Long Context)


### 10.3 Position Encoding for Length Generalization

HyPE (Hybrid Position Encoding) is emerging as the standard approach
for hybrid architectures (used by both HypeNet and MiniCPM-SALA):

    +---------------------+--------------------+
    | Layer Type          | Position Encoding  |
    +---------------------+--------------------+
    | Linear/RNN layers   | RoPE               |
    | Attention layers    | NoPE (none)        |
    +---------------------+--------------------+

Rationale:
- RNN layers process sequences locally/recurrently and benefit from
  positional cues that RoPE provides for short-range dependencies
- Attention layers with NoPE (no position encoding) provide superior
  length generalization because RoPE's frequency-based encoding
  degrades at positions far beyond training length
- When context length exceeds the RNN's receptive field, RNN layers
  are agnostic to context length anyway

Attention logits scaling for long context:
    s_t = log_a(t + a)
where a is a hyperparameter determined post-training. This mitigates
entropy increases in attention scores at long context.

Alternative: RoPE everywhere + YaRN interpolation for extending
softmax attention layers. This is simpler but requires fine-tuning.

Sources:
- arxiv 2601.22156 (HypeNet / HyPE)
- arxiv 2602.11761 (MiniCPM-SALA)


### 10.4 Gated Attention for Context Extension

Gated Attention (NeurIPS 2025 Best Paper):
- Adds a sigmoid gate per attention head after Scaled Dot-Product
  Attention. Simple architectural change.
- In extrapolation tests from 32K to 128K with YaRN, models with
  Gated Attention show less performance degradation than baselines.
- Compatible with hybrid architectures: can be applied to the 25%
  softmax attention layers to improve their length generalization.

Source: https://www.askaibrain.com/en/posts/end-of-transformers-hybrids-attention-state-space-2025


### 10.5 Training Recipes for Context Extension

Three proven approaches for extending hybrid models from 4K to 128K:

Approach 1: HALO Distillation (HypeNet)
- Three stages, 2.3B total tokens:
  1. Hidden state alignment (1B tokens): MSE between RNN and
     attention layer outputs
  2. Knowledge distillation (1B tokens): KL divergence between
     teacher (Transformer) and student (hybrid)
  3. Long-context finetuning (additional tokens at extended length)
- Converts existing Transformer to hybrid with minimal training

Approach 2: Progressive Context Extension (from full-attention work)
- Continual pre-training at 32K actual max sequence length
- Short SFT on high-quality instruction data
- Long SFT on long-context instruction data
- Interleaved training: mix short and long context data in 3:1 ratio

Approach 3: Direct Training with ABF
- Adjusted Base Frequency (ABF): modify the RoPE base frequency
  to extend without interpolation
- Combined with progressive training: 4K -> 16K -> 64K -> 128K
- Each stage uses a fraction of the total training budget

For Todorov Phase 2, Approach 3 is most practical:
- Start from Phase 1 checkpoint (trained at 4K)
- Progressive stages: 4K -> 16K -> 32K -> 64K -> 128K
- At each stage, verify perplexity stability and passkey accuracy
- Adjust RoPE base frequency using ABF or NTK-aware scaling

Sources:
- arxiv 2601.22156 (HALO distillation)
- arxiv 2309.16039 (effective long-context scaling)
- arxiv 2309.00071 (YaRN)


### 10.6 Special Considerations for the 3:1 Ratio

Layer placement matters:
- HALO's importance-based layer selection outperforms uniform
  interleaving (every-4th-layer pattern). The method scores each
  layer by: importance(layer) = recall_drop / CSR_drop, and keeps
  the 25% with highest importance as full attention.

- For a 24-layer model with 6 attention layers, optimal placement
  is NOT necessarily every 4th layer. The attention layers that
  matter most are those that serve as "retrieval checkpoints" for
  information accumulated by the surrounding linear layers.

- Practical heuristic: place attention layers with increasing density
  toward later layers, where the model needs to resolve ambiguities
  and perform final retrieval. Example for 24 layers: attention at
  layers 4, 8, 14, 18, 21, 24.

KDA + Mamba-3 + MLA interaction at long context:
- KDA layers accumulate information in fixed-size state
- Mamba-3 layers provide SSM-based state (halved to 64 dims vs
  Mamba-2's 128, freeing memory)
- MLA layers provide exact recall via compressed KV cache
- At 128K, the MLA cache is the dominant memory consumer among the
  three layer types. Monitor its growth carefully.

Source: arxiv 2601.22156
Source: arxiv 2603.15569 (Mamba-3 halved state)


### 10.7 Expected Performance Targets for Todorov Phase 2

Based on published results from comparable hybrid models:

    +-------------------+-----------+------------+-------------------+
    | Context Length    | Perplexity| Passkey    | Notes              |
    |                   | (target)  | Accuracy   |                    |
    +-------------------+-----------+------------+-------------------+
    | 4K (baseline)     | X         | 100%       | Phase 1 baseline   |
    | 16K               | X + 0.1   | >= 99%     | Minimal degradation|
    | 32K               | X + 0.2   | >= 98%     | Expect slight rise |
    | 64K               | X + 0.3   | >= 95%     | KDA state stressed |
    | 128K              | X + 0.5   | >= 90%     | Full stress test   |
    +-------------------+-----------+------------+-------------------+

    X = perplexity at 4K (model-dependent).

A well-functioning hybrid should show perplexity increasing by no
more than 0.5 points from 4K to 128K. If perplexity increases by
more than 1.0, investigate state collapse in KDA/Mamba-3 layers.

Passkey retrieval at 128K >= 90% would be competitive with HypeNet
(99.8%) given the smaller model scale. Below 80% indicates a
fundamental context extension problem.

Sources:
- arxiv 2601.22156 (HypeNet NIAH results)
- arxiv 2602.11761 (MiniCPM-SALA RULER results)
- arxiv 2510.26692 (Kimi Linear RULER results)


### 10.8 Updated References

- Samba (SSM + attention hybrid): arxiv 2406.07522
- Characterizing SSM long context: arxiv 2507.12442
- Effective long-context scaling: arxiv 2309.16039
- Gated Attention (NeurIPS 2025): https://www.askaibrain.com/en/posts/end-of-transformers-hybrids-attention-state-space-2025
- Rope to Nope: arxiv 2501.18795
- Cerebras 99% less training tokens: https://www.cerebras.ai/blog/extending-llm-context-with-99-less-training-tokens
- LoLA (sparse caching for linear attention): arxiv 2505.23666
