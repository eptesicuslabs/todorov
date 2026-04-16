# compression novelty analysis

status: current (as of 2026-04-16).

## the method

hierarchical ternary compression: a two-stage activation compression pipeline.

stage 1: k-WTA selection. select the top k% of dimensions by absolute magnitude. all other dimensions are set to zero.

stage 2: ternary quantization. the surviving dimensions are quantized to {-1, 0, +1} based on their sign. the magnitude information is discarded.

result: a sparse ternary vector where only k% of dimensions are nonzero, each carrying 1 bit of sign information.

## measured compression ratios (from pilot, d=256, gaussian inputs)

| k-WTA fraction | bits/dim | MI | CKA | compression vs FP32 |
|---|---|---|---|---|
| 5% | 0.32 | 0.28 | 0.63 | 100x |
| 10% | 0.56 | 0.49 | 0.71 | 57x |
| 20% | 0.92 | 0.82 | 0.81 | 35x |
| 41% (hierarchical k=0.41) | 1.39 | 1.25 | 0.89 | 23x |

note: the 41% row uses hierarchical k-WTA at k=0.41, not standard threshold ternary (alpha * mean(|x|)). standard ternary at alpha=1.0 produces MI~1.31 and CKA~0.89. the two methods produce similar but not identical results at matched firing rate.

## novelty assessment (verified by researcher subagent)

**the specific combination is novel.** no published paper applies k-WTA + ternary as a two-stage activation compression pipeline with CKA quality validation.

**closest prior work:**
- ComPEFT (yadav et al., EMNLP 2024, arXiv:2311.13171): analogous pipeline (magnitude-based sparsification + low-bit quantization) applied to PEFT weight deltas, not runtime activations. achieves 0.34 bits/element at 5% density via golomb coding. the exact sparsification and quantization steps may differ from top-k + sign -- the original paper should be consulted for precise algorithm comparison. no CKA analysis.
- Q-Sparse (arXiv:2407.10969): top-k on activations for BitNet models, but surviving activations remain full precision (not ternary quantized).
- BitNet a4.8 (arXiv:2411.04965): sparsify-then-quantize on activations, but quantization is 8-bit, not ternary.
- sparse ternary codes (ferdowsi & voloshynovskiy, IEEE ISIT 2017): sparse ternary representations for similarity search (signal processing, not neural activations).

**what makes it novel:**
1. applied to runtime neural activations (not weights, not weight deltas)
2. measured compression-quality tradeoff with CKA as quality metric
3. architecture-agnostic (works on any activation tensor, not just transformers)
4. the specific insight that k-WTA selects the most informative dimensions, so ternary quantization of the survivors preserves structural similarity (CKA > 0.63 even at 5%). caveat: CKA measures representational geometry, not task-relevant information. davari et al. (2022) showed CKA can be manipulated without changing functional behavior. downstream task evaluation (e.g., perplexity degradation) is needed to validate the compression quality claim beyond CKA.

## architecture independence

the pipeline operates on activation tensors of shape (batch, dim). it does not depend on:
- attention mechanisms
- convolutional structure
- recurrent state shape
- the model's loss function
- the training method (works with backprop, local rules, or any other optimizer)

this means it applies to the neural machine regardless of what the final architecture looks like. it also applies to any existing architecture as a drop-in activation compressor. however, the QUALITY of the compression depends on the activation distribution, which IS architecture-dependent. the method is architecture-agnostic in applicability but architecture-dependent in effectiveness.

## limitations

- tested only on synthetic gaussian data, not trained model activations
- the quality-compression tradeoff may be different for heavy-tailed activation distributions (which trained networks produce)
- the k-WTA selection adds a sorting operation (O(d log d) per token per layer)
- needs validation at scale (267M+ parameters) and measurement of downstream task impact
- the 1.39 bits/dim reference for standard ternary at 41% firing rate is the correct value (not log2(3) = 1.58 which is the uniform maximum)

## see also

- [[sparse_coding_to_ternary_spikes]]
- [[compression_architecture]]
- [[ternary_compression_research]]
- [[hierarchical_ternary]] (simulation script)
