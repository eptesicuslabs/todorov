# papers quick index

status: current (as of 2026-04-22).

this file is the quick paper-entry surface for the knowledge compartment. it stays category-first on purpose. use [[papers_library]] for the exhaustive annotated library and actionability notes.

do not treat this page as the canonical source for current project conclusions. those belong in the synthesis articles and in [[PROJECT_PLAN]].


## Core Architecture Papers

| Paper | ArXiv ID | One-line Summary |
|-------|----------|------------------|
| Kimi Linear: An Expressive, Efficient Attention Architecture | 2510.26692 | Channel-wise gated DeltaNet (KDA) in 3:1 hybrid with MLA; first linear attention to beat full attention fairly |
| DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model | 2405.04434 | Introduces Multi-Head Latent Attention (MLA) with low-rank KV compression (93.3% cache reduction) |
| DeepSeek-V3 Technical Report | 2412.19437 | Scales MLA to production with weight absorption trick and 671B MoE |
| Mamba-3: Improved Sequence Modeling using State Space Principles | 2603.15569 | Exp-trapezoidal discretization, complex-valued states, MIMO for 2x smaller state; ICLR 2026 |
| Mamba: Linear-Time Sequence Modeling with Selective State Spaces | 2312.00752 | Original selective state space model with data-dependent gating |
| Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality | 2405.21060 | Mamba-2; establishes state space duality (SSD) between linear attention and SSMs |


## Delta Rule and Linear Attention Papers

| Paper | ArXiv ID | One-line Summary |
|-------|----------|------------------|
| Parallelizing Linear Transformers with the Delta Rule over Sequence Length | 2406.06484 | DeltaNet: delta rule as online regression with WY-representation parallel training |
| DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products | 2502.10297 | Multiple Householder steps per token for more expressive state transitions |
| Gated Delta Networks: Improving Mamba2 with Delta Rule | 2412.06464 | Combines Mamba-style gating with delta rule; ICLR 2025 |
| Error-Free Linear Attention is a Free Lunch | 2512.12602 | Exact solution from continuous-time dynamics for linear attention |


## Hybrid Architecture Evidence

| Paper | ArXiv ID | One-line Summary |
|-------|----------|------------------|
| A Systematic Analysis of Hybrid Linear Attention | 2507.06457 | Rigorous ablation showing 3:1 to 6:1 ratio saturates recall; GDN and HGRN-2 best |
| Qwen3 Technical Report | 2505.09388 | Qwen3 model family including Qwen3-Next hybrid design |


## Context Extension Papers

| Paper | ArXiv ID | One-line Summary |
|-------|----------|------------------|
| Contextual Position Encoding: Learning to Count What's Important (CoPE) | 2405.18719 | Context-dependent position via gated cumulative sum; counts semantic units |
| YaRN: Efficient Context Window Extension of Large Language Models | 2309.00071 | NTK-by-parts interpolation + temperature scaling; extends to 128K; ICLR 2024 |
| LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens | 2402.13753 | Progressive non-uniform interpolation search; extends to 2048K; ICML 2024 |
| Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention | 2404.07143 | Compressive linear-attention memory + local attention; infinite context with bounded memory |
| Efficient Streaming Language Models with Attention Sinks (StreamingLLM) | 2309.17453 | Attention sink discovery; keep initial tokens + sliding window for infinite streaming; ICLR 2024 |
| LongRoPE2: Near-Lossless LLM Context Window Scaling | 2502.20082 | Follow-up to LongRoPE with improved quality preservation |
| Understanding the RoPE Extensions of Long-Context LLMs | 2406.13282 | Theoretical analysis of why RoPE extension methods work |


## Multimodal and Vision Papers

| Paper | ArXiv ID | One-line Summary |
|-------|----------|------------------|
| An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT) | 2010.11929 | Original Vision Transformer; patch embedding + position embedding for images |
| AST: Audio Spectrogram Transformer | 2104.01778 | Applies ViT to audio spectrograms with overlapping patches; convolution-free audio classification |
| Meta-Transformer: A Unified Framework for Multimodal Learning | 2307.10802 | Single frozen encoder processes 12 modalities via modality-specific tokenizers |
| 4M: Massively Multimodal Masked Modeling | 2312.06647 | Discrete tokenization of all modalities for unified masked prediction; NeurIPS 2023 |
| MSPE: Multi-Scale Patch Embedding Prompts Vision Transformers to Any Resolution | 2405.18240 | Multiple patch sizes for resolution-agnostic ViT processing |


## Geometric Algebra Papers

| Paper | ArXiv ID | One-line Summary |
|-------|----------|------------------|
| Geometric Algebra Transformer (GATr) | 2305.18415 | E(3)-equivariant transformer using G(3,0,1) multivectors; 9-param equivariant linear; NeurIPS 2023 |


## Attention Mechanism Papers

| Paper | ArXiv ID | One-line Summary |
|-------|----------|------------------|
| Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness | 2205.14135 | Tiling-based exact attention with O(n) memory; foundational GPU kernel |
| Flash Attention 2 | 2307.08691 | Improved parallelism and work partitioning over Flash Attention |
| Tiled Flash Linear Attention | 2503.14376 | Two-level tiling for unlimited chunk sizes in linear attention kernels |


## MLA Follow-up Papers

| Paper | ArXiv ID | One-line Summary |
|-------|----------|------------------|
| TransMLA: Multi-Head Latent Attention Is All You Need | 2502.07864 | Explores MLA variants and generalizations |
| Towards Economical Inference: Enabling DeepSeek's MLA in Any Transformer-based LLMs | 2502.14837 | Adapting MLA to existing pretrained models |
| MHA2MLA-VLM: Enabling DeepSeek's Economical MLA across Vision-Language Models | 2601.11464 | MLA adaptation for vision-language models |


## Non-ArXiv References

| Resource | URL | One-line Summary |
|----------|-----|------------------|
| OLMo Hybrid: From Theory to Practice | https://allenai.org/papers/olmo-hybrid | 3:1 GDN hybrid matches OLMo 3 with 49% fewer tokens; Allen AI |
| OLMo Hybrid blog | https://allenai.org/blog/olmohybrid | Introduction and overview of OLMo Hybrid architecture |
| flash-linear-attention (fla) | https://github.com/fla-org/flash-linear-attention | Pure PyTorch+Triton library for linear attention kernels |
| TFLA kernels | https://github.com/NX-AI/mlstm_kernels | Tiled Flash Linear Attention implementation for mLSTM |
| Kimi Linear code | https://github.com/MoonshotAI/Kimi-Linear | Open-source KDA kernel and model implementation |
| Gated DeltaNet code | https://github.com/NVlabs/GatedDeltaNet | Official ICLR 2025 implementation |
| GATr code | https://github.com/Qualcomm-AI-research/geometric-algebra-transformer | Geometric Algebra Transformer implementation |
| vLLM Qwen3-Next blog | https://blog.vllm.ai/2025/09/11/qwen3-next.html | vLLM support for hybrid attention architecture |
| Qwen3.5 blog | https://huggingface.co/blog/mlabonne/qwen35 | Analysis of Qwen3.5 hybrid attention choices |
| DeltaNet blog | https://sustcsonglin.github.io/blog/2024/deltanet-1/ | Detailed DeltaNet explanation by Songlin Yang |
| MLA explainer | https://planetbanatt.net/articles/mla.html | Clear technical walkthrough of MLA mechanism |
| Interconnects: OLMo Hybrid | https://www.interconnects.ai/p/olmo-hybrid-and-future-llm-architectures | Analysis of hybrid LLM architecture trends |
| EleutherAI YaRN blog | https://blog.eleuther.ai/yarn/ | Extending the RoPE explanation |
| Together AI Mamba-3 blog | https://www.together.ai/blog/mamba-3 | Mamba-3 announcement and overview |


## Byte-Level Modeling Papers

| Paper | ArXiv ID | One-line Summary |
|-------|----------|------------------|
| Beyond Language Models: Byte Models are Digital World Simulators (bGPT) | 2402.19155 | 110M byte-level GPT for cross-modal binary data; 1.06 BPB on AG News |
| Byte Latent Transformer: Patches Scale Better Than Tokens (BLT) | 2412.09871 | Dynamic entropy-based byte patching; matches Llama 3 at scale with 50% fewer inference FLOPs; ACL 2025 |
| Character-Level Language Modeling with Deeper Self-Attention | 1808.04444 | 64-layer char-level transformer; 1.06 BPC on enwik8, 1.13 on text8 |
| MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers | 2305.07185 | Multi-scale byte-level architecture; competitive with subword models on long sequences |
| Mamba Modulation: On the Length Generalization of Mamba | 2509.19633 | Analysis of Mamba length generalization; perplexity degradation beyond training context |


## Project-Internal References

| Source | Project | Summary |
|--------|---------|---------|
| Ternary spike findings | eptesicuslabs/gerhard | Adaptive threshold spiking with STE gradients and health monitoring |
| Geometric algebra findings | eptesicuslabs/echoloc | G(3,0,1) multivector encoding for 3D geometric data |

## see also

- [[papers_library]] -- canonical exhaustive paper library
- [[INDEX]] -- flat catalog of all wiki compartments
- [[PROJECT_PLAN]] -- authoritative current project state
