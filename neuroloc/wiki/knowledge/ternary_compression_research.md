# ternary compression research

curated peer-reviewed research on ternary and extreme quantization for neural networks. the central question for the neural machine: how much information can be transmitted per parameter or activation when restricted to {-1, 0, +1}, and what are the scaling laws, hardware advantages, and training techniques that make ternary viable at scale?

## ternary weight networks

### bitnet b1.58: ternary weights match fp16

ma, s. et al. (2024). the era of 1-bit llms: all large language models are in 1.58 bits. *arxiv preprint*.

key finding: bitnet b1.58 constrains all linear layer weights to {-1, 0, +1} (1.58 bits per weight) using absmean quantization. at 3B+ parameters, bitnet b1.58 matches fp16 transformer perplexity while replacing all matrix multiplications with integer additions. the key insight is that ternary weight quantization is viable at scale because the loss surface at 3B+ is smooth enough that the quantization noise is absorbed by the large number of parameters. below 1B, a small but consistent gap remains.

relevance to neural machine: todorov uses ternary quantization for activations (spikes), not weights. bitnet b1.58 demonstrates that ternary is sufficient for weights too, at scale. the complementary finding -- ternary weights + ternary activations -- would eliminate matrix multiplications entirely, reducing inference to pure addition. the threshold at which ternary matches fp16 (~3B for weights) may differ for activations, where todorov's spike health metrics (mi > 0.1, cka > 0.3) provide a direct measure of information preservation.

confidence: high. replicated by multiple groups. the 3B threshold is approximate and task-dependent. caveat: bitnet b1.58 uses fp16 activations; the combination of ternary weights + ternary activations has not been validated at comparable scale.

### fully binary w1a1: the gap persists

FBI-LLM team (2024). fully binarized weight and activation language models.

key finding: fully binarizing both weights and activations (w1a1, {-1, +1} for both) at 7B parameters produces a ~13 percentage point accuracy gap compared to fp16 baselines across standard language benchmarks. the gap is larger than ternary-weight-only quantization, confirming that activation quantization is harder than weight quantization. the binary activation constraint removes the zero state that ternary preserves, eliminating the network's ability to gate information flow through silence.

relevance to neural machine: the 13pp gap at w1a1 versus near-zero gap at ternary weights with fp16 activations isolates the cost of activation quantization. todorov's ternary spikes occupy the middle ground: {-1, 0, +1} activations preserve the zero/silence state. the fbi-llm result suggests that the zero in ternary is doing significant computational work -- it provides a gating mechanism that binary lacks. this aligns with the biological interpretation: cortical silence (~60% of neurons at any moment) is not absence of computation but active information suppression.

confidence: medium-high. single benchmark suite. caveat: the binary training procedure may not be optimally tuned; better binary methods could narrow the gap.

## ternary scaling laws

### paretoq: ternary on the pareto frontier

bondarenko, y. et al. (2025). paretoq: scaling laws in extremely low-bit quantization. *neurips*.

key finding: paretoq establishes quantization-aware scaling laws for extreme low-bit regimes. ternary (1.58-bit) quantization lies on the pareto frontier of the compute-accuracy tradeoff -- it achieves optimal performance for its bit budget. binary (1-bit) falls off the frontier, meaning that the accuracy loss from dropping from ternary to binary is not compensated by the compute savings. the pareto-optimal number of bits depends on model size: smaller models benefit more from higher precision, while very large models can absorb quantization noise.

relevance to neural machine: this provides theoretical backing for todorov's choice of ternary over binary spikes. the pareto frontier result means ternary is not an arbitrary biological constraint but an information-theoretically optimal operating point. the scaling law also predicts that as todorov scales to 1B+ parameters, the information loss from ternary quantization will decrease relative to model capacity.

confidence: high. systematic scaling study across multiple model sizes and bit widths. caveat: the scaling laws are fit to transformers; recurrent architectures may have different pareto frontiers due to state accumulation.

### data scaling dominates parameter scaling

spectra team (2024). spectra 1.1: scaling laws for ternary llms.

key finding: ternary llms follow modified scaling laws where the data exponent (0.81) is dramatically larger than the parameter exponent (0.32). in fp16 models, these exponents are closer (approximately 0.7 and 0.5 respectively). the implication is that ternary models benefit more from additional training data than from additional parameters. at fixed compute budget, ternary models should be trained on significantly more data relative to their parameter count compared to fp16 models.

relevance to neural machine: this scaling law has direct implications for todorov's training regimen. at 300M ternary-spike parameters, the data requirement may be substantially higher than for an equivalent fp16 model. the current training data budget (determined by phase 5 compute allocation) should be evaluated against the spectra scaling law to ensure the model is not data-starved. it also suggests that todorov's data efficiency at 6M-267M scale may improve less from parameter scaling and more from data scaling.

confidence: medium-high. scaling laws are empirical fits with limited extrapolation guarantees. caveat: spectra measures ternary weights, not ternary activations; the data exponent for activation-quantized models may differ.

## matmul-free architectures

### matmul-free language model

zhu, r. et al. (2024). scalable matmul-free language modeling. *neurips*.

key finding: by combining ternary weights with custom hardware-aware kernels, the matmul-free lm eliminates all matrix multiplications and replaces them with ternary accumulation (additions and subtractions). at 2.7B parameters, the model is within 0.8 perplexity points of a matched transformer baseline. the architecture uses gated recurrent units with ternary weights rather than attention, making it both matmul-free and attention-free. memory consumption scales linearly with sequence length.

relevance to neural machine: todorov already uses ternary activations and could adopt ternary weights to achieve a fully matmul-free architecture. the 0.8pp gap at 2.7B is consistent with bitnet b1.58's finding that ternary becomes competitive at multi-billion scale. the matmul-free lm's use of gated recurrence rather than attention parallels todorov's kda layers, which also use recurrence for memory. the key difference is that todorov's kda uses a delta-rule outer product for state updates, which is inherently a matmul -- eliminating it would require rethinking the state update mechanism.

confidence: high. published at neurips with detailed ablations. caveat: the 0.8pp gap may widen on harder benchmarks or longer sequences.

## hardware efficiency

### bitnet.cpp: ternary on cpu

wang, j. et al. (2024). bitnet.cpp: efficient inference for ternary llms on cpus.

key finding: a custom inference framework for ternary-weight llms achieves 2-8x speedup over fp16 baselines on commodity cpus with 55-82% energy reduction. the speedup comes from replacing multiply-accumulate operations with additions and lookup tables. the energy reduction comes from both fewer operations and lower per-operation energy (addition uses ~30x less energy than multiplication in silicon). the speedup scales with model size: larger models benefit more because memory bandwidth (not compute) becomes the bottleneck, and ternary weights compress 10x.

relevance to neural machine: todorov's ternary spikes already provide the activation-side energy savings (no multiplication needed when the activation is -1, 0, or +1). bitnet.cpp quantifies the weight-side savings. combined ternary weights and activations would reduce inference energy by potentially 90%+ compared to fp16. the 10x memory compression from ternary weights also enables running larger models on smaller hardware, which aligns with todorov's goal of biologically-inspired efficiency.

confidence: high. benchmarked on real hardware with standard models. caveat: speedups are cpu-specific; gpu inference with ternary has different characteristics due to different memory hierarchies.

### fpga ternary accelerators

liang, x. et al. (2025). tereffic: ternary efficient inference on fpga.

key finding: fpga implementations of ternary neural network inference achieve 8-19x power efficiency improvement over gpu baselines. the efficiency comes from custom logic that exploits the three-valued nature of ternary operations: each multiply-accumulate reduces to a conditional add/subtract/skip, which maps directly to simple fpga logic primitives. the 19x figure is for fully ternary (weights and activations); the 8x figure is for ternary weights with higher-precision activations.

relevance to neural machine: the 19x figure for full ternary represents the hardware ceiling for todorov if both weights and activations are ternary. fpga deployment is a realistic path for inference in resource-constrained environments. the conditional add/subtract/skip mapping mirrors the biological operation: a ternary spike either excites (+1), inhibits (-1), or has no effect (0), and the postsynaptic response is a simple accumulation.

confidence: medium-high. fpga benchmarks with specific architectures. caveat: fpga efficiency depends heavily on the specific design and quantization scheme; the 19x figure may not generalize to all model architectures.

## training techniques for ternary

### ste improvements

multiple groups (2024-2025). tequila, hestia, and related ste improvements.

key finding: the straight-through estimator (ste) used to train ternary networks has a well-known deadzone problem: gradients are zero for weights or activations that are far from the quantization thresholds. tequila introduces a temperature-scaled ste that smooths the gradient landscape, and hestia uses a hybrid ste with separate forward and backward quantization schedules. both methods improve ternary network accuracy by 4-5 percentage points over baseline ste on imagenet, primarily by reducing the fraction of "stuck" parameters that receive zero gradient.

relevance to neural machine: todorov uses ste for gradient flow through ternary spikes. the adaptive threshold (alpha * mean(|x|)) partially addresses the deadzone by adjusting the quantization boundary, but the ste itself is standard. tequila or hestia-style modifications could improve gradient flow through the spike quantization, potentially allowing higher firing rates without losing information (currently the firing rate is ~41%, and the ste deadzone limits how much this can be adjusted).

confidence: medium. improvements are consistent across multiple papers but specific to image classification. caveat: ste modifications may interact differently with todorov's adaptive threshold mechanism than with standard quantization.

## see also

- [[ternary_spikes]]
- [[energy_efficient_coding]]
- [[sparse_coding]]
- [[brain_energy_budget]]
- [[metabolic_constraints_on_computation]]
- [[sparse_coding_to_ternary_spikes]]
- [[energy_efficiency_to_ternary_spikes]]

## relevance to the neural machine

### validated connections
- ternary activations (spikes) are the core quantization in todorov, with measured firing rate ~41% and mi > 1.0 at scale
- bitnet b1.58 and matmul-free lm confirm that ternary is viable at multi-billion parameter scale for weights
- paretoq confirms ternary as pareto-optimal for its bit budget -- the biological constraint is also the information-theoretic optimum
- hardware measurements (2-8x cpu speedup, 55-82% energy reduction, 8-19x fpga efficiency) quantify the deployment advantage

### challenged assumptions
- todorov's ternary spikes quantize activations, not weights -- the combination of ternary weights + ternary activations has not been validated at llm scale
- the spectra scaling law (data exponent >> parameter exponent) suggests ternary models need more data than currently budgeted
- ste deadzone affects gradient flow through spikes; tequila/hestia-style fixes are not yet integrated

### open questions
- can todorov adopt ternary weights in addition to ternary activations to achieve fully matmul-free inference?
- does the spectra data scaling law apply to activation-quantized models, and if so, is the 300M model data-starved?
- would ste improvements (tequila, hestia) change the optimal firing rate from the current 41%?
- at what parameter scale does the ternary activation gap (separate from weight gap) close to negligible?
