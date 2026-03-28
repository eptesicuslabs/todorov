# Todorov Paper Library — Eptesicus Laboratories

**Architecture reference:** ~350M parameter hybrid — [KDA, KDA, KDA, Mamba3, KDA, KDA, KDA, MLA] × 3 — targeting 4GB edge devices, INT8 quantization, 128K+ context. Combines channel-wise delta-rule linear attention (KDA), Mamba-3 SISO SSM, Multi-Latent Attention (MLA), ternary spiking neurons (ATMN), and an optional geometric product (GP) self-interaction module.

**Notation:** ⚡ = Direct action item for Todorov. Papers are ordered by relevance within each category.

---

## Category 1: Linear attention and state space models

**1.1 Kimi Linear — An Expressive, Efficient Attention Architecture**
Yu Zhang, Zongyu Lin, Xingcheng Yao et al. · Moonshot AI · arXiv:2510.26692 · October 2025
Introduces Kimi Delta Attention (KDA), extending Gated DeltaNet with **channel-wise (per-feature-dimension) gating** instead of head-wise gating, allowing each feature dimension to evolve independently. Uses a constrained DPLR (Diagonal-Plus-Low-Rank) formulation for ~2× faster kernels. The hybrid architecture interleaves KDA and MLA at a 3:1 ratio, with MLA layers using NoPE for easy conversion to MQA at inference. The 48B total / 3B activated MoE model reduces KV cache by 75% and achieves 6× decoding throughput at 1M context. ⚡ **CRITICAL — This is Todorov's primary linear attention mechanism. Study channel-wise gating implementation, DPLR kernel optimizations, and the KDA-MLA interface design.**

**1.2 Kimi K2 — Open Agentic Intelligence**
Kimi Team · Moonshot AI · arXiv:2507.20534 · July 2025
A 1.04T-parameter MoE model with 32B activated parameters, using MLA (DeepSeek V3-style) with 384 experts. Introduces MuonClip optimizer for training stability and a large-scale agentic data synthesis pipeline with joint RL. Achieves SOTA among open-source non-thinking models on agentic benchmarks (66.1 Tau2-Bench, 65.8 SWE-bench). K2's MLA architecture provides the full-attention component that Kimi Linear later hybridized with KDA. ⚡ **Study MuonClip optimizer for Todorov training stability.**

**1.3 Gated DeltaNet — Improving Mamba2 with Delta Rule**
Songlin Yang, Jan Kautz, Ali Hatamizadeh · NVIDIA · arXiv:2412.06464 · December 2024
Combines Mamba2's gated decay with the delta rule for error-correcting memory updates. The delta rule erases content aligned with the current key before writing, acting as online gradient descent on a reconstruction objective. Surpasses Mamba2 and DeltaNet across language modeling, commonsense reasoning, in-context retrieval, and length extrapolation. **The foundational linear attention adopted by Qwen3-Next, Kimi Linear, and OLMo Hybrid — the most influential linear attention paper of 2024–2025.** ⚡ **KDA is a channel-wise refinement of this mechanism. Understand the base algorithm deeply.**

**1.4 DeltaNet — Linear Transformers Are Secretly Fast Weight Programmers**
Schlag, Irie, Schmidhuber (2021, foundation); Yang, Wang et al. (arXiv:2406.06484, 2024 modernization)
Formulates state updates as online gradient descent on a reconstruction objective — erase old content associated with the current key, then write new value. Provides the principled controlled-forgetting mechanism missing from additive linear attention. Extended by Gated DeltaNet and KDA.

**1.5 Mamba — Linear-Time Sequence Modeling with Selective State Spaces**
Albert Gu, Tri Dao · CMU / Princeton · arXiv:2312.00752 · December 2023
Introduced selective SSMs making A, B, C parameters input-dependent (content-aware). Hardware-aware parallel scan algorithm. Mamba-3B outperforms same-size Transformers and matches 2× larger ones with 5× higher inference throughput. The foundational SSM architecture.

**1.6 Mamba-2 — Transformers Are SSMs: Structured State Space Duality**
Tri Dao, Albert Gu · Princeton / CMU · arXiv:2405.21060 · ICML 2024
Introduced Structured State Space Duality (SSD) connecting SSMs to linear attention. Dramatically faster training via matmul-friendly computation. Demonstrated the theoretical bridge between SSMs and attention that enables hybrid architectures.

**1.7 Mamba-3 — Improved Sequence Modeling Using State Space Principles**
Aakash Lahoti, Kevin Y. Li, Berlin Chen et al. · CMU / Princeton · arXiv:2603.15569 · ICLR 2026, March 2026
Introduces three improvements: (1) exponential-trapezoidal discretization (upgrading from Euler's method), (2) complex-valued state updates equivalent to data-dependent RoPE, and (3) a multi-input, multi-output (MIMO) formulation increasing arithmetic intensity during decoding without increasing latency. At 1.5B, Mamba-3 MIMO improves average downstream accuracy by **+1.8 points over Gated DeltaNet** and achieves comparable perplexity to Mamba-2 at half the state size. ⚡ **CRITICAL — Todorov uses Mamba-3 SISO. Study the MIMO vs SISO tradeoff for the 350M scale and the complex-valued state update mechanism for implicit positional encoding.**

**1.8 RWKV-7 "Goose" — Expressive Dynamic State Evolution**
Bo Peng, Ruichong Zhang et al. · RWKV Foundation · arXiv:2503.14456 · March 2025
Introduces a generalized delta rule formulation with vector-valued gating and in-context learning rates. Can perform state tracking and recognize all regular languages. The 2.9B model achieves new 3B SoTA on multilingual tasks. Kernels scale linearly; up to 3× faster than RWKV-6 and faster than FlashAttention v3 at 16K+. Predecessors: RWKV-4 (arXiv:2305.13048, EMNLP 2023), Eagle/Finch RWKV-5/6 (arXiv:2404.05892, 2024).

**1.9 RetNet — Retentive Network**
Yutao Sun, Li Dong et al. · Microsoft Research / Tsinghua · arXiv:2307.08621 · July 2023
Proposes retention supporting parallel, recurrent, and chunkwise computation. At 7B, RetNet decodes 8.4× faster and saves 70% memory versus Transformers. Foundational "generation 2" linear attention that influenced GLA, HGRN, and subsequent work.

**1.10 GLA — Gated Linear Attention with Hardware-Efficient Training**
Songlin Yang, Bailin Wang et al. · arXiv:2312.06635 · ICML 2024
Data-dependent gating allowing adaptive decay rates. Key contribution: hardware-efficient chunkwise parallel training competitive with FlashAttention-2 in throughput. Provides the computational foundation (chunkwise kernels) that HGRN2 and later models build upon.

**1.11 HGRN2 — Gated Linear RNNs with State Expansion**
Zhen Qin, Songlin Yang et al. · arXiv:2404.07904 · April 2024
Extends HGRN with outer-product state expansion, dramatically increasing recurrent state size without additional parameters. The 3B HGRN2 slightly outperforms Mamba and LLaMA Transformers in controlled experiments. Recommended alongside GatedDeltaNet as optimal linear components for hybrids by the Systematic Analysis paper.

**1.12 A Systematic Analysis of Hybrid Linear Attention**
Dustin Wang, Rui-Jie Zhu, Steven Abreu et al. · arXiv:2507.06457 · July 2025
**The definitive empirical guide for hybrid architecture design.** Trained 72 models (36 at 340M, 36 at 1.3B) covering 6 linear attention variants across 5 hybridization ratios. Key findings: (1) superior standalone linear models do NOT necessarily excel in hybrids; (2) quality is stable across ratios but recall significantly improves with more full attention layers, especially below 3:1; (3) selective gating, hierarchical recurrence, and controlled forgetting are the three critical features. Recommends HGRN-2 or GatedDeltaNet at **3:1 to 6:1 ratio**. ⚡ **HIGH PRIORITY — The 340M scale experiments directly match Todorov's scale. Validate Todorov's ratio choices against these findings. The paper's open-sourced models at 340M are directly comparable baselines.**

**Survey:** "From S4 to Mamba: A Comprehensive Survey on Structured State Space Models" · arXiv:2503.18970 · March 2025. Reviews S4, Mamba, S5, Jamba architectures.

---

## Category 2: Hybrid architectures (linear + attention)

**2.1 Qwen3-Next — Hybrid Gated DeltaNet Architecture**
Qwen Team · Alibaba Cloud · September 2025 (blog/model release)
Pioneered the 3:1 hybrid ratio of Gated DeltaNet (75% linear) to Gated Attention (25% full softmax with sigmoid output gating) in a production LLM. The 80B-A3B ultra-sparse MoE natively supports 262K context. 10× faster than Qwen3-32B at 128K context. **First major production deployment of DeltaNet-style attention in a frontier model.** ⚡ **CRITICAL — Validates Todorov's hybrid philosophy. Study gating mechanism and training stability optimizations.**

**2.2 Qwen3.5**
Qwen Team · Alibaba Cloud · February 2026
Inherits and refines Qwen3-Next's hybrid architecture (397B-A17B). Adds native multimodality (text+images+video), 201 languages, RL across million-agent environments. Uses 75% linear (Gated DeltaNet) + 25% full attention. Production validation that linear attention replaces quadratic attention for most layers without quality loss.

**2.3 OLMo Hybrid — From Theory to Practice**
William Merrill, Yanhong Li et al. · Allen Institute for AI · March 2026
7B-parameter model replacing 75% of attention with Gated DeltaNet (3:1 ratio). **Most rigorous controlled comparison to date.** Key findings: (1) matches Olmo 3 on MMLU with 49% fewer tokens (~2× data efficiency); (2) hybrid models provably more expressive than pure transformers or pure linear RNNs alone — formal problems exist that neither can express alone but hybrids can; (3) scaling hierarchy: hybrid GDN > pure GDN > transformer > hybrid Mamba2 > pure Mamba2. ⚡ **HIGH PRIORITY — Provides the theoretical and empirical foundation for Todorov's hybrid design. The formal proof of expressiveness superiority directly justifies the hybrid approach.**

**2.4 Jamba — Hybrid Transformer-Mamba Language Model**
Opher Lieber, Barak Lenz et al. · AI21 Labs · arXiv:2403.19887 · ICLR 2025
First production-grade hybrid Transformer–Mamba–MoE model. Interleaves Mamba and attention at 1:7 ratio with MoE on some MLPs. Supports 256K context, 52B total / 12B active. Notable finding: Mamba-1 + Attention works better than Mamba-2 + Attention in hybrid settings. Jamba 1.5 scaled to 398B/94B active with SOTA RULER scores.

**2.5 Zamba / Zamba2**
Paolo Glorioso, Quentin Anthony, Beren Millidge et al. · Zyphra · Zamba: arXiv:2405.16712 (May 2024); Zamba2: arXiv:2411.15242 (November 2024)
Pioneers shared attention in SSM hybrids — all attention layers share weights, minimizing parameter cost while retaining recall benefits. Zamba2 uses two alternating shared attention blocks with LoRA for depth-specialization. 2.7B achieves SOTA for <3B models. ⚡ **Evaluate weight-sharing of MLA layers across depth for Todorov — could dramatically reduce parameter overhead of the attention component for edge deployment.**

**2.6 TransMLA — Multi-Head Latent Attention Is All You Need**
Fanxu Meng, Pingzhi Tang et al. · Peking University / Xiaomi · arXiv:2502.07864 · NeurIPS 2025 Spotlight
Proves MLA strictly subsumes GQA — any GQA layer can be rewritten as MLA, but not vice versa. TransMLA converts GQA models to MLA, achieving **~10× speedup on LLaMA-2-7B** by compressing 93% of KV cache. ⚡ **Provides theoretical basis for MLA as optimal full-attention mechanism in Todorov's design.**

**2.7 Falcon-H1 — Hybrid-Head Language Models**
Jingwei Zuo, Maksim Velikanov et al. · TII, Abu Dhabi · arXiv:2507.22448 · July 2025
Parallel hybrid combining Transformer attention + Mamba SSMs running concurrently within each block, outputs concatenated. 0.5B to 34B parameters. Uses customized μP for scaling. Falcon-H1-34B-Instruct competitive with Qwen3-32B despite being half the size. Apache 2.0 license.

**2.8 NVIDIA Hymba — Hybrid-Head Architecture for Small Language Models**
Xin Dong, Yonggan Fu et al. · NVIDIA · arXiv:2411.13676 · ICLR 2025
Integrates attention and Mamba heads in parallel within each block (not sequential stacking). Introduces learnable "meta tokens" storing critical information. Cross-layer KV cache sharing + partial sliding window. Hymba-1.5B surpasses all sub-2B models including Llama-3.2-3B, with **10× less cache memory**. ⚡ **Study meta tokens and cross-layer KV sharing for Todorov's edge deployment. The parallel-head design is an alternative to Todorov's sequential layer pattern.**

**2.9 NVIDIA Nemotron-Flash — Latency-Optimal Hybrid SLMs**
Yonggan Fu et al. · NVIDIA · arXiv:2511.18890 · November 2025
Uses evolutionary search to discover latency-optimal combinations of Attention, Mamba-2, DeltaNet, and FFN operators. Searched architecture interleaves DeltaNet-FFN-Mamba2-FFN and Attention-FFN-Mamba2-FFN blocks. Achieves +5.5% accuracy, 1.3×/1.9× lower latency, 18.7×/45.6× higher throughput vs. Qwen3 1.7B/0.6B. ⚡ **HIGH PRIORITY — The evolutionary NAS approach for mixing attention types is directly applicable to Todorov. Consider running similar search at 350M scale.**

**2.10 Falcon Mamba — First Competitive Attention-Free 7B Model**
Jingwei Zuo, Maksim Velikanov et al. · TII · arXiv:2410.05355 · October 2024
Pure Mamba-based 7.27B model, trained on 5.8T tokens. Surpasses Mistral 7B, Llama3.1-8B, and Falcon2-11B. Demonstrates pure SSMs can compete with Transformers. Useful as a pure-SSM baseline for comparing hybrid approaches.

---

## Category 3: Chinese lab architectures

**3.1 DeepSeek-V2 — MLA and DeepSeekMoE**
DeepSeek-AI · arXiv:2405.04434 · May 2024
Introduces Multi-head Latent Attention (MLA) compressing KV cache into low-rank latent vectors — **93.3% KV cache reduction** versus MHA while achieving better performance through joint key-value compression with decoupled RoPE. 236B total / 21B activated, 128K context. ⚡ **CRITICAL — MLA is a core Todorov component. Study latent compression dimensions, decoupled RoPE handling, and INT8 quantization compatibility.**

**3.2 DeepSeek-V3 Technical Report**
DeepSeek-AI · arXiv:2412.19437 · December 2024
Scales to 671B/37B activated. Retains MLA + DeepSeekMoE, adds auxiliary-loss-free load balancing and Multi-Token Prediction (MTP) objective. FP8 mixed-precision training. ⚡ **MTP training objective may benefit Todorov; FP8 training is relevant to INT8 targets.**

**3.3 DeepSeek-V3.2 — Pushing the Frontier**
DeepSeek-AI · arXiv:2512.02556 · December 2025
Introduces DeepSeek Sparse Attention (DSA) instantiated on MLA — reduces complexity for long-context by operating in MQA mode where latent vectors are shared across query heads. Three branches: compression, selection, sliding window. ⚡ **HIGH PRIORITY — DSA's sparse selection atop MLA is directly relevant for Todorov's 128K+ context in MLA layers.**

**3.4 DeepSeek-R1 — Incentivizing Reasoning via RL**
DeepSeek-AI · arXiv:2501.12948 · January 2025 (also Nature, September 2025)
Reasoning emerges through pure RL (GRPO) without SFT. Built on MLA+MoE. Open-sourced distilled models 1.5B–70B. ⚡ **The distillation methodology (R1 → small dense models) is applicable for training Todorov's 350M model.**

**3.5 DeepSeek Engram — Conditional Memory via Scalable Lookup**
Xin Cheng, Wangding Zeng et al. · Peking University / DeepSeek-AI · arXiv:2601.07372 · January 2026
O(1) lookup conditional memory using modernized N-gram embeddings with multi-head hashing. U-shaped scaling law: ~20–25% sparse params to Engram is optimal. Frees attention for global context (Multi-Query NIAH: 84.2→97.0). ⚡ **EXPLORE — A small Engram table in system RAM could offload factual retrieval from Todorov's neural budget.**

**3.6 MiniMax-01 — Scaling Foundation Models with Lightning Attention**
MiniMax collective · arXiv:2501.08313 · January 2025
Hybrid architecture: Lightning Attention (I/O-aware linear attention) with softmax attention interleaved every 7 layers. 456B/45.9B activated, 1M training context, 4M inference. 20–32× longer context than peers. ⚡ **The 7:1 hybrid ratio closely parallels Todorov's pattern (7 KDA/Mamba : 1 MLA per block).**

**3.7 MiniMax-M1 — Test-Time Compute with Lightning Attention**
MiniMax · arXiv:2506.13585 · June 2025
Reasoning model on MiniMax-Text-01 backbone. 25% FLOPs vs DeepSeek-R1 at 100K generation. Validates hybrid linear/softmax architectures for RL-based reasoning training.

**3.8 Qwen3 Technical Report**
Qwen Team · Alibaba · arXiv:2505.09388 · May 2025
Series from 0.6B to 235B. GQA, SwiGLU, RoPE, QK-Norm. Trained on 36T tokens, 119 languages. Edge models (0.6B, 1.7B) via Strong-to-Weak Distillation. ⚡ **Study Strong-to-Weak Distillation for training Todorov's 350M.**

**3.9 ChatGLM — GLM-130B to GLM-4 All Tools**
Team GLM · Zhipu AI / Tsinghua · arXiv:2406.12793 · June 2024
Evolving family. GLM-4-9B supports 128K context with tool calling. Later versions (GLM-4.5–4.7) scale to 355B MoE with 200K context. Standard GQA architecture — less architecturally novel than DeepSeek but useful 128K reference point.

---

## Category 4: Quantization and efficient inference

**4.1 Quamba — Post-Training Quantization for Selective State Space Models**
Chiang et al. · arXiv:2410.13229 · 2024
First dedicated PTQ method for Mamba SSMs. SSM activations exhibit distinct outlier patterns vs. Transformers (in output tensor y). Uses Hadamard transforms and percentile clipping. INT8 with 0.9% accuracy drop on Jamba hybrid. ⚡ **CRITICAL — Only paper specifically addressing SSM quantization. Directly applicable to Todorov's Mamba-3 layers.**

**4.2 SpinQuant — LLM Quantization with Learned Rotations**
Liu et al. · Meta · arXiv:2405.16406 · ICLR 2025
Learned rotation matrices via spin parametrization on the orthogonal group remove activation outliers before quantization. W4A4KV4 narrows gap to 2.9 points on LLaMA-2 7B. **Used in Meta's Llama 3.2 on-device deployment.** ⚡ **HIGH PRIORITY — Rotation-based approach may improve INT8 quantization of KDA/Mamba layers where outlier patterns differ from standard attention.**

**4.3 QuaRot — Outlier-Free 4-Bit Inference in Rotated LLMs**
Ashkboos et al. · arXiv:2404.00456 · 2024
Hadamard-based rotations smooth weight-activation landscape, enabling lossless 6/8-bit and competitive 4-bit without calibration data.

**4.4 SmoothQuant — Accurate W8A8 Post-Training Quantization**
Xiao et al. · MIT · arXiv:2211.10438 · ICML 2023
Shifts quantization difficulty from activations to weights via per-channel scaling, enabling W8A8. ⚡ **Baseline INT8 technique for Todorov. Test whether activation patterns in KDA layers require different smoothing factors than standard attention.**

**4.5 AWQ — Activation-Aware Weight Quantization**
Lin et al. · arXiv:2306.00978 · MLSys 2024
Protects 0.1–1% crucial weights identified via activation magnitudes. TinyChat framework achieves 3× speedup on mobile GPUs. ⚡ **TinyChat's mobile deployment pipeline is relevant for Todorov's 4GB edge target.**

**4.6 GPTQ — Accurate Post-Training Quantization**
Frantar et al. · arXiv:2210.17323 · ICLR 2023
Approximate second-order (OBS-based) per-channel weight-only quantization to 4-bit with minimal accuracy loss. Foundation paper for practical LLM quantization.

**4.7 SqueezeLLM — Dense-and-Sparse Quantization**
Kim et al. · arXiv:2306.07629 · 2024
Separates outlier weights into sparse format + non-uniform quantization for remaining weights. Enables better accuracy at ultra-low bit-widths.

**4.8 QuIP# — Quantization with Hadamard Incoherence and Lattice Codebooks**
Tseng et al. · arXiv:2402.04396 · 2024
Incoherence processing with orthogonal matrices and lattice codebooks for 2-bit quantization.

**4.9 AQLM — Extreme Compression via Additive Quantization**
Egiazarian et al. · arXiv:2401.06118 · ICML 2024
Multi-codebook quantization. First scheme Pareto-optimal at <3 bits/parameter. LLaMA-2 7B at 2-bit: 6.93 perplexity.

**4.10 KVQuant — Towards 10M Context with KV Cache Quantization**
Hooper et al. · arXiv:2401.18079 · 2024
Targets KV cache compression for long-context scenarios. Complements weight quantization. ⚡ **Relevant for MLA's compressed latent KV cache — study whether additional quantization on top of MLA's latent compression is feasible.**

**4.11 Bi-Mamba — Towards Accurate 1-Bit SSMs**
arXiv:2411.11843 · 2024
Binary Mamba achieves performance similar to GPTQ-8bit. Demonstrates SSMs are more amenable to extreme quantization than Transformers.

**4.12 LightMamba — Efficient Mamba on FPGA**
Wei et al. · arXiv:2502.15260 · 2025
Co-designs quantization and FPGA accelerator. Rotation-assisted quantization + power-of-two SSM quantization. **4.65× energy efficiency over RTX 4090.** ⚡ **Study for potential FPGA edge deployment of Todorov's Mamba-3 layers.**

**4.13 Q-SNNs — Quantized Spiking Neural Networks**
Wei et al. · arXiv:2406.13672 · 2024
Quantizes both synaptic weights and membrane potentials. Weight-Spike Dual Regulation via information entropy theory. ⚡ **Directly applicable to Todorov's ternary spiking neurons — study membrane potential quantization approach.**

**4.14 TurboQuant — Extreme KV Cache Compression**
Amir Zandieh, Vahab Mirrokni, Praneeth Kacham, Majid Hadian, Insu Han, Majid Daliri, Lars Gottesburen, Rajesh Jayaram · Google · arXiv:2504.19874 · ICLR 2026
Two-stage KV cache compression: (1) PolarQuant — random rotation converts vectors to polar coordinates (radius + angle), eliminating expensive normalization and simplifying geometry for per-component scalar quantization; (2) QJL (Quantized Johnson-Lindenstrauss) — uses just 1 additional bit as a residual error-checker that mathematically eliminates quantization bias in attention score computation. 3-bit KV cache with **zero accuracy loss** on Gemma, Mistral, Llama-3.1-8B across LongBench, NIAH, ZeroSCROLLS, RULER, L-Eval. 6x memory reduction. 4-bit achieves **8x speedup** on H100 computing attention logits vs FP32. Builds on QJL and PolarQuant (venue details unverified; both are components of TurboQuant). ⚡ **HIGH PRIORITY — The PolarQuant rotation stage is conceptually related to SpinQuant/QuaRot but uses polar decomposition instead of Hadamard/Cayley rotations. The QJL bias elimination is critical for long-context accuracy (128K+). For Todorov: (a) apply to MLA latent vectors — study whether 3-bit quantization on top of MLA's d_c=128 latent compression is feasible (double compression); (b) the rotation-based approach may handle MLA's joint key-value latent space better than methods designed for separate K/V heads; (c) at 300M with 128K context, MLA cache is the dominant memory cost and 6x reduction directly enables the 4GB edge target.**

**Surveys:** "A Survey of Low-bit Large Language Models" (arXiv:2409.16694, 2024). "A Comprehensive Study on Quantization Techniques for LLMs" (arXiv:2411.02530, 2024).

---

## Category 5: Spiking neural networks for language

**5.1 MAR — Module-Aware Architecture Refinement**
Junhong Cai, Guiqin Wang et al. · arXiv:2601.21503 · January 2026
Combines Mamba-2 SSMs for linear-time sequence modeling with spiking activation sparsification for FFN efficiency. Introduces the **Adaptive Ternary Multi-step Neuron (ATMN)** increasing information capacity with ternary {-1, 0, +1} spikes, and the **Spike-aware Bidirectional Distillation Strategy (SBDS)** using reverse-KL compensation with Pre-Norm alignment for SSM-SNN integration. ⚡ **CRITICAL — Todorov's ternary spiking neurons are based on ATMN from this paper. SBDS directly addresses the temporal mismatch between SSMs and SNNs. Ensure SBDS training procedure is correctly implemented.**

**5.2 SpikingBrain — Brain-Inspired Models for Long-Context**
BICLab · arXiv:2509.05276 · September 2025
SpikingBrain-7B (linear) and SpikingBrain-76B (hybrid-linear MoE). Uses linear/hybrid-linear attention with adaptive spiking neurons. >100× speedup in TTFT for 4M-token sequences. **69.15% sparsity** enabling low-power operation. ⚡ **HIGH PRIORITY — Largest spiking language model to date. Study their conversion-based training pipeline and how they integrate spiking neurons with linear attention at scale.**

**5.3 SpikeLLM — Spiking Large Language Model at Scale**
Xingrun Xing et al. · arXiv:2407.04752 · July 2024
First spiking LLM at 7B–70B scale. Redesigns LLaMA-2 with Generalized Integrate-and-Fire neurons (compressing spike length) and Optimal Brain Spiking (OBS) for per-channel spiking. Reduces WikiText2 perplexity by 25.51% on 4A4W. ⚡ **The GIF neuron's spike compression technique could complement ATMN's ternary approach.**

**5.4 SpikeGPT — Generative Spiking Language Model**
Rui-Jie Zhu, Qihang Zhao, Jason Eshraghian · UC Santa Cruz · arXiv:2302.13939 · ICLR 2024
First generative SNN language model. Inspired by RWKV, integrates recurrence for SNN compatibility. Binary spiking with SEW-ResNet residual connections. 46M and 216M parameter models achieve competitive results with **22–33× fewer synaptic operations**. Uses surrogate gradient training.

**5.5 SpikingBERT**
Malyaban Bal, Abhronil Sengupta · Penn State · arXiv:2308.10873 · AAAI 2024
Encoder-only spiking LM using implicit differentiation at equilibrium (no surrogate gradients). ANN-SNN knowledge distillation from pretrained BERT.

**5.6 SpikeBERT**
Changze Lv et al. · Fudan University · arXiv:2308.15122 · ICLR 2024
Two-stage KD from BERT: (1) pretraining distillation on unlabeled text, (2) task-specific fine-tuning KD. Addresses "self-accumulating dynamics" failure mode in large-scale SNNs. ⚡ **Two-stage KD pipeline is directly relevant for Todorov's spiking component training.**

**5.7 Surrogate Gradient Learning in SNNs**
Neftci, Mostafa, Zenke · IEEE Signal Processing Magazine · 2019 (foundational)
Comprehensive treatment of surrogate gradients replacing non-differentiable spike derivatives. Common surrogates: fast sigmoid, arctangent, rectangular (STE). Gygax & Zenke (2025, Neural Computation) proved surrogate gradients equivalent to derivatives of smoothed probabilistic models. Key variants: Dspike (NeurIPS 2021), SuperSpike (Zenke & Ganguli, 2018).

---

## Category 6: Geometric algebra neural networks

**6.1 GATr — Geometric Algebra Transformer**
Johann Brehmer, Pim de Haan et al. · Qualcomm AI Research · arXiv:2305.18415 · NeurIPS 2023
First general-purpose architecture combining geometric algebra with Transformers. Operates in projective geometric algebra G(3,0,1) (16-dimensional multivectors). All operations E(3)-equivariant. Scales to thousands of tokens. More sample-efficient than SEGNN and SE(3)-Transformer. ⚡ **Foundation paper for Todorov's GP module. Study the geometric product implementation and how it can be adapted as a self-interaction layer.**

**6.2 Versor — A Geometric Sequence Architecture**
Truong Minh Huy, Edward Hirst · arXiv:2602.10195 · February 2026
Complete sequence architecture in Conformal Geometric Algebra (CGA, Cl(4,1)). Recursive Rotor Accumulator (RRA) for O(L) temporal complexity; Geometric Product Attention (GPA) with proximity (scalar) and orientational torque (bivector) components. **200× fewer parameters** than Transformers for comparable accuracy. Zero-shot scale generalization: 0.993 vs 0.070 MCC for ViT on topological reasoning. Custom Clifford kernels achieve >100× speedup. ⚡ **HIGH PRIORITY — The geometric product decomposition into scalar + bivector components may inform how Todorov's GP self-interaction should be structured. The parameter efficiency is especially relevant for edge deployment.**

**6.3 Geometric Memory in Sequence Models**
Shahriar Noroozizadeh, Vaishnavh Nagarajan et al. · arXiv:2510.26745 · October 2025
Identifies "geometric memory" as fundamentally different from associative memory in deep sequence models. Models synthesize embeddings encoding global relationships between all entities (including non-co-occurring ones), transforming hard ℓ-fold composition reasoning into easy 1-step navigation. ⚡ **Directly relevant to understanding how Todorov's spike MI/CKA metrics should be interpreted and what the GP module might be learning.**

**6.4 CGENN — Clifford Group Equivariant Neural Networks**
David Ruhe, Johannes Brandstetter, Patrick Forré · arXiv:2305.11141 · NeurIPS 2023
Constructs O(n)- and E(n)-equivariant networks using the Clifford group. Every polynomial in multivectors constitutes an equivariant map. Operates directly in vector basis (no spherical harmonics). Single implementation generalizes to any dimension.

**6.5 CliffordLayers — Clifford Neural Layers for PDE Modeling**
Brandstetter, van den Berg, Welling, Gupta · Microsoft Research · ICLR 2023
Multivector representations with Clifford convolutions and Fourier transforms. Treats correlated fields as multivector fields. Library: github.com/microsoft/cliffordlayers.

**6.6 Geometric Clifford Algebra Networks (GCANs)**
Ruhe, Gupta, de Keninck et al. · ICML 2023
Group action layers combining object transformations via Pin(p,q,r) group actions. Layers serve as "adjustable geometric templates."

**6.7 Geometric Algebra Attention Networks**
Matthew Spellings · arXiv:2110.02393 · NeurIPS 2021
Rotation- and permutation-equivariant architectures using products of geometric algebra terms reduced via attention. Earlier work combining GA with attention.

---

## Category 7: Deep equilibrium models

**7.1 DEQ — Deep Equilibrium Models**
Shaojie Bai, J. Zico Kolter, Vladlen Koltun · NeurIPS 2019 · arXiv:1909.01377
Foundational paper: finds the fixed point z* = f_θ(z*; x) of a single layer, modeling "infinite depth" with constant memory. Backprop via implicit differentiation. Proved universality: stacking multiple DEQs adds no representational power. Competitive on WikiText-103. Code: github.com/locuslab/deq. ⚡ **If Todorov explores weight-tying across the 3 block repetitions, DEQ theory provides the theoretical foundation.**

**7.2 RevDEQ — Reversible Deep Equilibrium Models**
Sam McCallum, Kamran Arora, James Foster · University of Bath · arXiv:2509.12917 · September 2025
Computes exact gradients using algebraically reversible fixed-point solver while retaining O(1) memory. 3.75× fewer function evaluations than standard DEQs, no regularization needed. Outperforms DEQ approaches and Transformer-XL on WikiText-103. ⚡ **If DEQ-like weight tying is pursued for Todorov, RevDEQ is the preferred implementation — exact gradients with O(1) memory is ideal for edge training/fine-tuning.**

**7.3 Stabilizing DEQs by Jacobian Regularization**
Bai, Koltun, Kolter · ICML 2021 · arXiv:2106.14342
Regularizes Jacobian Frobenius norm with stochastic estimation. Reduces NFEs by 50%. First implicit-depth model running at ResNet-101 speed with constant memory.

**7.4 DEQuify — Converting Pretrained Models to DEQ**
Andreas Burger et al. · University of Toronto · arXiv:2509.08734 · September 2025
Demonstrates recasting pretrained EquiformerV2 as DEQ to exploit temporal continuity. 10–20% improvement in accuracy and speed by recycling intermediate features. ⚡ **Study the conversion methodology — could Todorov's repeated 3-block pattern be DEQuified post-training?**

**7.5 Lyapunov-Stable DEQs (LyaDEQ)**
arXiv:2304.12707
Ensures Lyapunov stability of DEQ fixed points for adversarial robustness. Orthogonalizes layers after stability module.

**7.6 Monotone Operator DEQs**
Winston & Kolter · CMU
Provably convergent layers based on monotone operator theory. Guarantees convergence by construction.

---

## Category 8: Context extension and long-range modeling

**8.1 iRoPE — Interleaved RoPE (Llama 4)**
Meta · April 2025
Interleaves 3 RoPE layers (chunked local attention) + 1 NoPE layer (full causal mask) in a repeating pattern. Inference-time temperature scaling enhances length generalization. Enables Llama 4 Scout's **10M token context**. Pre-trained at 256K, generalizes to 10M at inference. ⚡ **HIGH PRIORITY — The interleaved RoPE/NoPE pattern is directly analogous to Todorov's KDA/MLA interleaving. Study how NoPE layers in MLA complement RoPE in KDA layers for context extension.**

**8.2 Infini-Attention — Leave No Context Behind**
Munkhdalai, Faruqui, Gopal · Google · arXiv:2404.07143 · April 2024
Combines masked local attention + long-term linear attention in a single block with compressive memory. Bounded parameters even at millions of tokens. Learnable gating β balances local vs. memory. **114× memory compression.** Passkey retrieval from 1M tokens. ⚡ **The compressive memory concept could augment Todorov's KDA layers — evaluate whether KDA's delta-rule state already functions as compressive memory.**

**8.3 CoPE — Contextual Position Encoding**
Golovneva, Wang et al. · Meta · arXiv:2405.18719 · May 2024
Positions conditioned on context: learned gate increments position only on certain tokens. Enables addressing by semantic units rather than token counts. Lightweight (cumulative sum). ⚡ **Could replace or augment RoPE in Todorov's KDA layers for more semantic positional awareness.**

**8.4 YaRN — Yet another RoPE extensioN**
Peng et al. · 2023
"NTK-by-parts" interpolation + temperature factor for attention distribution. Part of the PI → Dynamic NTK-RoPE → YaRN evolution.

**8.5 StreamingLLM — Efficient Streaming with Attention Sinks**
Xiao et al. · MIT HAN Lab · arXiv:2309.17453 · ICLR 2024
Discovered attention sink phenomenon. Recipe: sink tokens + sliding window KV. Stable generation up to 4M+ tokens without fine-tuning. 22.2× speedup. Important caveat: does NOT extend context window or enhance long-term memory.

**8.6 Ring Attention — Sequence Parallelism**
Liu, Zaharia, Abbeel · arXiv:2310.01889 · NeurIPS 2023
Distributes long sequences across devices using blockwise computation. Overlaps communication with computation (ring topology). Enables sequences up to device_count × single_device_length without approximation.

**8.7 Lost in the Middle**
Liu, Lin, Hewitt et al. · arXiv:2307.03172 · TACL 2024
Seminal analysis: U-shaped performance curve — models use information best at beginning/end, degrade for middle context. Even 100K+ context models exhibit this. Mid-context performance can drop below closed-book baseline. ⚡ **Evaluate whether Todorov's hybrid KDA+MLA architecture mitigates this pattern.**

**8.8 Found in the Middle (Ms-PoE)**
Zhang et al. · arXiv:2403.04797
Multi-scale Position Encoding with different scaling ratios per attention head. Mitigates lost-in-the-middle without fine-tuning.

**8.9 Llama 4 Scout** — Meta · April 2025
109B/17B active (16 experts). 10M context via iRoPE. NoPE layers use full causal; RoPE use chunked local. QK-norm in RoPE layers.

**8.10 Magic LTM-2-mini** — Magic.dev · August 2024
First 100M token context. Proprietary LTM architecture (~1000× cheaper than attention at 100M). Limited external validation. Architecture details proprietary.

**8.11 SPLA — Block Sparse Plus Linear Attention**
Bailin Wang, Dan Friedman et al. · arXiv:2601.22379 · January 2026
Uses second-order Taylor expansions for block selection + Residual Linear Attention for unselected blocks. Closes gap with dense attention at 256K tokens while maintaining sparse efficiency. ⚡ **Evaluate as alternative context extension strategy for Todorov's MLA layers.**

**8.12 SSM Long-Context Analysis**
Multiple papers establish SSM limitations: Jelassi et al. (2024, "Repeat After Me") — copying requires state to grow linearly with length. Waleffe et al. (2024) — SSM long-context lags Transformers. LongSSM (arXiv:2406.02080) — slow decay causes overflow. ⚡ **These limitations motivate Todorov's hybrid approach — the MLA layers provide the recall capability SSMs lack.**

**8.13 Overcoming Long-Context Limitations of SSMs via CDSA**
Zhan et al. · arXiv:2507.00449 · July 2025
Proves SSMs cannot solve multi-query joint recall in sub-quadratic time. Proposes HAX (locality-sensitive Hashing Attention with sparse Key Selection). ⚡ **Theoretical proof that hybrid design is necessary — pure SSMs provably cannot handle certain long-context tasks.**

**8.14 Scaling Linear Attention with Sparse State Expansion (SSE)**
Pan et al. · ByteDance · arXiv:2507.16577 · July 2025
Row-sparse updates and sparse state expansion decouple parameter size from state capacity. 2B SSE-H model: **SOTA reasoning (64.5 AIME24, 50.2 AIME25)**, surpassing similarly sized Transformers.

---

## Category 9: Mechanistic interpretability

**9.1 Toy Models of Superposition**
Elhage et al. · Anthropic · 2022
Foundational work explaining how networks compress more features than dimensions through polysemanticity/superposition. Referenced in virtually all SAE work. Essential theoretical framework.

**9.2 Scaling Monosemanticity — Features from Claude 3 Sonnet**
Templeton, Conerly et al. · Anthropic · May 2024
Applied SAEs to production Claude 3 Sonnet with 1M, 4M, 34M features. Found abstract, multilingual, multimodal features. Scaling laws guide SAE training. Safety-relevant features identified.

**9.3 Towards Monosemanticity — Dictionary Learning**
Anthropic · October 2023
Precursor showing SAEs recover monosemantic features from 1-layer transformers.

**9.4 ROME — Locating and Editing Factual Associations in GPT**
Kevin Meng, David Bau et al. · NeurIPS 2022 · arXiv:2202.05262
Causal tracing identifies mid-layer FFN modules mediating factual recall at subject tokens. Rank-One Model Editing for changing specific facts.

**9.5 MEMIT — Mass Editing Memory in a Transformer**
Meng, Sharma et al. · ICLR 2023 · arXiv:2210.07229
Scales ROME to thousands of simultaneous edits across multiple MLP layers.

**9.6 CKA — Similarity of Neural Network Representations Revisited**
Kornblith, Norouzi, Lee, Hinton · ICML 2019 · arXiv:1905.00414
Introduced CKA as a representation similarity metric based on HSIC. Reliably identifies correspondences between differently-initialized networks. ⚡ **Directly relevant for Todorov's spike MI/CKA metrics. Use for comparing representations across spiking vs. dense layers.**

**9.7 Reliability of CKA**
Davari et al. · arXiv:2210.16156 · 2022
CKA can be manipulated without changing functional behavior. Caution needed when using activation alignment metrics. ⚡ **Important caveat for interpreting Todorov's CKA measurements.**

**9.8 Representation Engineering**
Zou et al. · 2023
Identifies semantic vector directions for high-level concepts in hidden representations. Enables detection and steering.

**9.9 The Illusion of State in State-Space Models**
Merrill, Petty, Sabharwal · arXiv:2404.08819 · 2024
Theoretical analysis of SSM state representations. Key for understanding what Mamba layers actually encode.

**Survey:** "Towards Uncovering How LLMs Work: An Explainability Perspective" · arXiv:2402.10688.

---

## Category 10: Training dynamics and recipes

**10.1 LLM-JEPA — Joint Embedding Predictive Architecture for LLMs**
Hai Huang, Yann LeCun, Randall Balestriero · arXiv:2509.14252 · September 2025
First JEPA-based training objective for LLMs. Combines next-token prediction with embedding-space JEPA using natural "view" pairs (text/code, NL/SQL). Robust to overfitting (baseline degrades while JEPA continues improving). Pre-trained models improve downstream transfer. ⚡ **HIGH PRIORITY — Evaluate JEPA objective for Todorov training. Requires natural multi-view pairs; develop data augmentation for arbitrary text.**

**10.2 Tensor Programs V / μP — Zero-Shot Hyperparameter Transfer**
Greg Yang, Edward Hu et al. · Microsoft Research · arXiv:2203.03466 · NeurIPS 2021
In Maximal Update Parametrization (μP), optimal hyperparameters remain stable across model scales. Transferring from 13M outperforms published BERT-large (350M). Tuning cost only 7% of pretraining cost. ⚡ **CRITICAL — Use μP for Todorov. Tune hyperparameters at ~35M scale and transfer to 350M. The 10× proxy ratio is validated.**

**10.3 DoReMi — Optimizing Data Mixtures**
Xie et al. · NeurIPS 2023 · arXiv:2305.10429
Uses 280M proxy model with Group DRO to find domain weights for 8B model. Improves avg accuracy by 6.5 points. Reaches baseline 2.6× faster.

**10.4 Data Mixing Laws**
ICLR 2025
Nested scaling laws (steps × sizes × mixing) predict performance of unseen mixtures. Optimized mixture achieves comparable performance with 48% fewer steps.

**10.5 Topic Over Source for Data Mixing**
arXiv:2502.16802 · 2025
Topic-based partitioning consistently outperforms source-based mixing across multiple methods.

**10.6 Chinchilla Scaling Laws and Revisions**
Hoffmann et al. · DeepMind · arXiv:2203.15556 · 2022. Optimal: ~20 tokens/parameter. Replication (arXiv:2404.10102) revised to ~25.6 tokens/param. Beyond Chinchilla (arXiv:2401.00448) shows training smaller models longer (up to 10,000 tokens/param) reduces total cost when inference demand is high. ⚡ **Todorov at 350M should train on 3.5–35T tokens depending on compute budget and inference volume.**

**10.7 MiniLLM — Knowledge Distillation with Reverse KL**
Gu et al. · ICLR 2024
Reverse KL prevents student from overestimating low-probability regions. Policy gradient optimization. Tested GPT-2 125M from 1.5B teacher.

**10.8 Rethinking KL Divergence in KD for LLMs**
arXiv:2404.02657
Forward and reverse KL behaviors don't hold straightforwardly for LLM KD. Both converge to same objective given sufficient epochs. Proposes Adaptive KL (AKL) combining both. ⚡ **Use AKL for Todorov's distillation from larger teacher model.**

**10.9 SNN-Specific KD**
"Distilling Spikes" (Kushawaha et al., arXiv:2005.00288, 2021) — first KD for SNNs, teacher-assistant multi-stage. SAKD (Neural Networks 178, 2024) — bilevel ANN→SNN transfer. "A Closer Look at KD in SNN Training" (arXiv:2511.06902, 2025) — Saliency-scaled Activation Map Distillation addresses ANN-SNN distribution mismatch. ⚡ **Study SAMD for training Todorov's spiking layers — the ANN-SNN distribution mismatch is the core challenge.**

**10.10 PhoneLM — Principled Pre-training for Small Models**
arXiv:2411.05046 · 2024
0.5B and 1.5B models. Key principle: search for runtime-efficient architecture on target hardware BEFORE pretraining. ⚡ **The hardware-first architecture search principle should guide Todorov's final architecture decisions.**

**Survey:** "A Survey on Knowledge Distillation of LLMs" · Xu et al. · arXiv:2402.13116 · February 2024. Comprehensive coverage: black-box, white-box, self-improvement KD.

---

## Category 11: Attention mechanism innovations

**11.1 XSA — Exclusive Self-Attention**
Shuangfei Zhai · Apple · arXiv:2603.09078 · March 2026
Constrains attention to capture only information orthogonal to the token's own value vector. 2-line code change. Consistently outperforms standard SA up to 2.7B with minimal overhead. Gains increase with sequence length. ⚡ **CRITICAL — Already analyzed as composable with MLA. Implement as z_i = y_i - (y_i^T v_i)v_i/‖v_i‖² in Todorov's MLA layers. Free quality gain increasing with context length.**

**11.2 Differential Transformer**
Tianzhu Ye, Li Dong et al. · Microsoft Research / Tsinghua · arXiv:2410.05258 · ICLR 2025
Attention as difference between two softmax maps (differential amplifier analogy). Promotes sparse attention naturally. 7.5% average accuracy gain in math reasoning. **Reduces activation outliers**, improving quantization. ⚡ **EVALUATE — Reduced outliers directly improve INT8 quantization of MLA layers. Consider XSA + Differential Attention composition.**

**11.3 Native Sparse Attention (NSA)**
Jingyang Yuan, Huazuo Gao et al. · Peking University / DeepSeek-AI · arXiv:2502.11089 · ACL 2025
Three parallel branches: compressed coarse-grained, selectively retained fine-grained, sliding window local. End-to-end trainable during pretraining. Matches full attention on benchmarks with substantial speedups at 64K+. Evolved into DSA in V3.2. ⚡ **HIGH PRIORITY — Adapt three-branch design for Todorov's MLA layers at 128K+ context.**

**11.4 MQA → GQA → MLA Evolution**
MQA: Shazeer (Google, arXiv:1911.02150, 2019) — shared single KV head, 87.5% cache reduction. GQA: Ainslie et al. (Google, arXiv:2305.13245, EMNLP 2023) — G groups sharing KV, 5% uptraining cost. MLA: DeepSeek-V2 (2024) — joint low-rank compression, 93% cache reduction without quality loss. TransMLA proved GQA ⊂ MLA expressiveness. MHA2MLA (arXiv:2502.14837) — data-efficient MHA→MLA conversion.

**11.5 Hardware-Centric MLA Analysis**
Geens et al. · arXiv:2506.02523 · 2025
Identifies MLA execution schemes and hardware co-design opportunities. ⚡ **Study for optimizing MLA on edge accelerators.**

**11.6 Lightning Attention**
Qin et al. · MiniMax / Tsinghua · 2022–2025 series
I/O-aware linear attention implementation for million-token contexts. Used in MiniMax's 7:1 hybrid architecture.

---

## Category 12: Edge deployment and on-device AI

**12.1 MobileLLM — On-Device Sub-Billion Parameter LMs**
Liu et al. · Meta · arXiv:2402.14905 · ICML 2024
First LLM explicitly for on-device. **Architecture matters more than data/params at sub-billion scale.** Deep-and-thin design, embedding sharing, GQA. 125M at 50 tok/s on iPhone. Extensions: MobileLLM-Pro (1B, arXiv:2511.06719), MobileLLM-R1 (reasoning, arXiv:2509.24945). ⚡ **CRITICAL — Deep-and-thin finding directly relevant to Todorov's 350M design. Evaluate embedding sharing for parameter reduction.**

**12.2 SmolLM2 — Data-Centric Small Language Models**
Ben Allal et al. · HuggingFace · arXiv:2502.02737 · 2025
Models at 135M, 360M, 1.7B. Overtrained on ~11T tokens. Outperforms Qwen2.5-1.5B and Llama3.2-1B. Fully open-source including datasets (FineMath, Stack-Edu, SmolTalk). ⚡ **The 360M model is the closest size competitor to Todorov. Use as primary benchmark. Study their data curation and multi-stage training.**

**12.3 TinyLlama**
Zhang et al. · arXiv:2401.02385 · 2024
1.1B on Llama architecture, 1T tokens for 3 epochs. 24K tok/s per A100-40G throughput.

**12.4 Gemma 3**
Google DeepMind · arXiv:2503.19786 · 2025
1B to 27B multimodal. 128K context, 140+ languages. Increases ratio of local-to-global attention layers for KV-cache reduction. Includes **270M model** for edge. ⚡ **Gemma3-270M is a direct competitor to Todorov at similar scale.**

**12.5 Phi-4 / Phi-4-Mini**
Microsoft · arXiv:2412.08905 (Phi-4), arXiv:2503.01743 (Phi-4-Mini)
14B focused on data quality over scale. Surpasses teacher GPT-4 on STEM. Phi-4-Mini (3.8B) with Mixture-of-LoRAs for multimodal.

**12.6 Phi-3 — A Highly Capable Model Locally on Your Phone**
Microsoft · arXiv:2404.14219 · 2024
Predecessor to Phi-4, explicitly designed for phone deployment.

**12.7 MNN-LLM — Mobile Inference Engine**
arXiv:2506.10443 · 2025
DRAM-Flash Hybrid Storage + Hardware-Driven Data Reordering. **25.3× faster prefill and 7.1× faster decode than llama.cpp.** ⚡ **Evaluate as inference backend for Todorov on mobile.**

**12.8 XAMBA — Mamba on Resource-Constrained NPUs**
Das et al. · arXiv:2502.06924 · 2025
Deploys quantized Mamba2-2.7B (INT8) on NPU (Orange Pi): <3% accuracy degradation, **2.87GB peak memory**, 5.4 tok/s. ⚡ **CRITICAL — Demonstrates INT8 Mamba-2 fitting in <3GB on NPU. Todorov's Mamba-3 layers can target similar efficiency.**

**12.9 llama.cpp** · Gerganov · 2023-ongoing
Pure C/C++ inference, 1.5–8 bit quantization, GGUF format. Optimized for ARM NEON, Metal, AVX. ~150 tok/s on M2 Ultra.

**12.10 MLC-LLM** · 2024
TVM-based compilation. OpenCL, CUDA, Metal, WebGPU. ~190 tok/s on M2 Ultra. Paged KV caching.

**Surveys:** "On-Device Language Models: A Comprehensive Review" (arXiv:2409.00088, 2024). "Small Language Models Can Still Pack a Punch" (arXiv:2501.05465, 2025). "ELIB: Edge LLM Inference Benchmarking" (arXiv:2508.11269, 2025).

---

## Consolidated action items for Todorov

### Immediate implementation priorities
1. **XSA in MLA layers** — 2-line modification, free quality gain scaling with context length (arXiv:2603.09078)
2. **μP for hyperparameter tuning** — Tune at ~35M proxy, transfer to 350M (arXiv:2203.03466)
3. **Rotation-based quantization** — SpinQuant/QuaRot for INT8 of hybrid layers (arXiv:2405.16406, 2404.00456)
4. **Quamba techniques** — Apply Hadamard transforms to Mamba-3 layer quantization (arXiv:2410.13229)
5. **TurboQuant for MLA KV cache** — 3-bit KV cache, zero accuracy loss, 6x memory reduction. Apply PolarQuant+QJL to MLA latent vectors for double compression (MLA + TurboQuant). Critical for 4GB edge target at 128K context (arXiv:2504.19874)

### Architecture decisions to revisit
5. **Mamba-3 MIMO vs SISO** — MIMO yields +1.8 points at 1.5B; evaluate tradeoff at 350M (arXiv:2603.15569)
6. **Weight-sharing MLA layers** (Zamba2 pattern) — Could reduce MLA parameter overhead for edge (arXiv:2411.15242)
7. **Hymba meta tokens** — Learnable tokens storing critical information, 10× less cache (arXiv:2411.13676)
8. **DSA sparse branches in MLA** — Three-branch sparse selection for 128K+ efficiency (arXiv:2512.02556)
9. **Nemotron-Flash evolutionary search** — NAS for optimal operator mixing at 350M (arXiv:2511.18890)

### Training recipe priorities
10. **SBDS from MAR** — Validate bidirectional distillation for spiking component (arXiv:2601.21503)
11. **Adaptive KL distillation** — Combine forward + reverse KL for Todorov KD (arXiv:2404.02657)
12. **DoReMi data mixing** — Use 35M proxy for mixture optimization (arXiv:2305.10429)
13. **LLM-JEPA objective** — Evaluate JEPA for multi-view training (arXiv:2509.14252)

### Ablations to run
14. **Hybrid ratio sweep** — Test 3:1 through 7:1 KDA:MLA ratios per Systematic Analysis findings (arXiv:2507.06457)
15. **Differential Attention in MLA** — Evaluate impact on INT8 quantization quality (arXiv:2410.05258)
16. **CoPE vs RoPE** — Test contextual positioning in KDA layers (arXiv:2405.18719)
17. **Engram module** — Small lookup table in system RAM for factual offloading (arXiv:2601.07372)

### Benchmarking
18. **SmolLM2-360M** as primary size-class competitor (arXiv:2502.02737)
19. **Gemma3-270M** as secondary baseline (arXiv:2503.19786)
20. **Lost-in-the-middle evaluation** — Test whether hybrid KDA+MLA mitigates U-shaped degradation (arXiv:2307.03172)
21. **SSMs vs Transformers long-context profiling** — Validate 4× SSM speedup at 57K+ tokens on target edge hardware (arXiv:2507.12442)