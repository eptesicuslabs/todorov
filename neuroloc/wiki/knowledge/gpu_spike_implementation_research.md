# gpu spike implementation research

curated research on implementing spiking neural networks efficiently on gpu hardware. the central challenge: spiking operations are inherently sequential (each spike depends on membrane potential history) and sparse (most neurons are silent at any given time), both properties that conflict with gpu architecture designed for dense, parallel computation. the findings here inform how todorov's ternary spike implementation should evolve.

## kernel fusion

### spikingjelly cupy kernel fusion

fang, w. et al. (2023). spikingjelly: an open-source machine learning infrastructure platform for spike-based intelligence. *science advances*, 9(40), eadi1480.

key finding: spikingjelly achieves 11x training speedup over naive pytorch by fusing the entire spiking neuron forward and backward pass into a single cupy kernel. the standard approach -- separate pytorch operations for membrane update, threshold comparison, reset, and surrogate gradient -- incurs massive kernel launch overhead because each operation is trivially small but requires a full gpu kernel launch. by fusing all operations into one kernel that processes the entire neuron state update (membrane integration, spike generation, reset) in a single pass, the overhead drops from dozens of kernel launches per neuron per timestep to one.

relevance to neural machine: todorov's ternary spike currently uses separate pytorch operations for threshold computation, sign extraction, and ste backward pass. fusing these into a single triton kernel (not cupy, since todorov targets triton for portability) would eliminate kernel launch overhead. at 40% firing rate with dense computation, the bottleneck is not sparsity exploitation but launch overhead -- fusion addresses this directly.

confidence: high. published benchmarks with reproducible speedups across multiple architectures.

### temporal fusion across timesteps

wang, y. et al. (2024). temporal fusion for spiking neural networks. *international conference on artificial neural networks (ICANN)*.

key finding: rather than processing one timestep at a time across all neurons, temporal fusion processes all T timesteps for each neuron in a single kernel launch. this converts the sequential timestep loop into a single fused operation, achieving 5-40x speedup depending on the number of timesteps T. the key insight is that within a single neuron, the timestep loop is inherently sequential (each step depends on the previous membrane potential), but the loop body is small enough to run entirely in registers without touching global memory between steps.

relevance to neural machine: todorov does not currently iterate over timesteps within the spike operation (ternary spikes are stateless per-token). however, if [[neuron_models_to_atmn|atmn]] membrane dynamics become production-ready, temporal fusion would be the correct implementation strategy: process all T timesteps for each neuron's membrane potential in registers, emitting spikes along the way.

confidence: medium-high. speedups are architecture-dependent and the fused kernel loses flexibility for complex neuron models.

## sparse computation

### sparseprop binary heap

engelken, r. et al. (2023). sparseprop: efficient event-based simulation and training of sparse recurrent spiking neural networks. *advances in neural information processing systems (NeurIPS) 36*.

key finding: sparseprop achieves O(log N) cost per spike event (vs O(N) for dense matrix-vector multiply) by maintaining a binary heap of next-spike times. when a neuron spikes, only its postsynaptic targets need updating, and the heap efficiently identifies the next neuron to spike. on sparse networks (1-10% connectivity), this produces order-of-magnitude speedups over dense simulation. the method applies to both forward simulation and training via surrogate gradients.

relevance to neural machine: at todorov's current 40% firing rate, sparseprop offers no advantage -- dense computation beats sparse gpu operations when more than ~20-30% of neurons are active. the crossover point depends on network size and connectivity density. if future work reduces firing rate toward biological levels (1-5%), sparseprop-style event-driven computation becomes relevant.

confidence: high for sparse networks. not applicable at todorov's current firing rate regime.

### dense vs sparse crossover

at 40% firing rate, dense matrix operations on gpu are faster than any sparse alternative. the reasons: (1) gpu warp execution requires all 32 threads to execute the same instruction, so sparse indexing wastes lanes on inactive neurons, (2) sparse formats (csr, csc, coo) add index overhead that exceeds the savings from skipping zeros at moderate sparsity, (3) cusparse kernels are optimized for very sparse matrices (< 1% nonzero) not moderately sparse ones. the crossover where sparse beats dense on modern gpus (a100, h100) is approximately 5-15% density, depending on matrix dimensions.

relevance to neural machine: this confirms that todorov's current approach (dense computation through ternary spikes at 40% firing rate) is the correct engineering choice. sparse acceleration only becomes relevant if firing rate drops below ~15%.

## parallel scan for spiking neurons

### the lif reset problem

the fundamental obstacle to parallelizing spiking neuron simulation is the hard reset: when a neuron's membrane potential exceeds threshold, it resets to a fixed value, creating a discontinuity that breaks the associative scan required for parallel prefix computation. the membrane potential at time t depends on whether a spike occurred at t-1, which depends on the potential at t-1, creating a sequential dependency chain.

four solutions have been proposed:

### psn: removing the reset entirely

fang, w. et al. (2023). parallel spiking neurons. *advances in neural information processing systems (NeurIPS) 36*.

key finding: parallel spiking neurons (psn) simply remove the reset mechanism. without reset, the membrane dynamics become a linear recurrence (h_t = alpha * h_{t-1} + x_t) that admits standard parallel scan in O(T log T). spikes are still generated by thresholding, but the membrane potential is not modified after spiking. on sequential mnist, psn matches conventional lif accuracy while training 5-10x faster via parallel scan.

relevance to neural machine: psn validates the approach of trading biological fidelity (reset) for computational efficiency (parallel scan). todorov's ternary spikes already lack a reset mechanism (they are stateless), making them trivially parallelizable. if atmn adds temporal dynamics, the psn strategy of removing reset would preserve parallelizability.

### prf: complex domain parallel scan

cheng, x. et al. (2024). parallel receptance field (prf). O(L log L) parallel scan for spiking neurons by lifting the membrane dynamics into the complex plane, where the reset can be absorbed into a multiplicative factor.

relevance to neural machine: mathematically elegant but adds complexity. relevant if todorov needs parallel spiking dynamics with reset.

### spikingssms: surrogate dynamic network

bal, a. et al. (2025). spikingssms: learning long sequences with sparse and parallel spiking state spaces. *proceedings of the aaai conference on artificial intelligence (AAAI)*.

key finding: spikingssms replace the hard reset with a learned soft gating function that approximates reset dynamics while maintaining differentiability and parallelizability. the surrogate dynamic network learns to emulate reset behavior without introducing the sequential dependency.

relevance to neural machine: the soft gating approach is conceptually similar to todorov's beta gate in [[kda_channel_gating|kda]]. if spiking dynamics are added, learned soft reset could replace hard reset.

### bullet trains: newton-raphson spike finding

renner, a. et al. (2026). bullet trains: spike-timing-dependent parallel spiking neuron simulation. *arxiv*.

key finding: bullet trains uses newton-raphson root finding to locate spike times in parallel across all timesteps, then applies reset corrections. this preserves exact spike timing while achieving near-linear parallel speedup.

relevance to neural machine: the most biologically faithful parallel approach, but the most complex to implement. relevant only if exact spike timing becomes important for todorov (currently it is not -- ternary spikes are rate-coded).

## exact spike-timing gradients

### eventprop

wunderlich, t. & pehle, c. (2021). event-based backpropagation can compute exact gradients for spiking neural networks. *scientific reports*, 11, 12829.

key finding: eventprop computes exact gradients with respect to spike times by treating spikes as events in continuous time. rather than using surrogate gradients (smooth approximations to the non-differentiable threshold), eventprop derives the exact gradient of a loss function with respect to the timing of each spike, using the implicit function theorem at the threshold crossing. this produces gradients that are mathematically correct, not approximate.

relevance to neural machine: todorov uses ste (straight-through estimator) for spike gradients, which is a crude approximation. eventprop shows that exact gradients are computable in principle. however, eventprop requires spike times to be meaningful (which they are in temporal coding but not in todorov's rate-coded ternary scheme) and scales poorly to large networks because each spike event requires solving an implicit equation.

confidence: high for the theoretical result. practical scalability remains limited.

## time-to-first-spike coding

### ttfs efficiency

guo, w. et al. (2024). neural coding with time-to-first-spike. *nature communications*, 15, 2345.

key finding: time-to-first-spike (ttfs) coding achieves accuracy matching rate-coded anns with only ~0.3 spikes per neuron on average. each neuron fires at most once, encoding information in the latency from stimulus onset to spike time. this is dramatically more energy-efficient than rate coding (which requires many spikes per neuron to achieve reliable rate estimates) and ternary coding (which fires at ~40% rate).

relevance to neural machine: ttfs represents the extreme opposite of todorov's approach. todorov uses ternary spikes at 40% firing rate (effectively rate coding with 3 levels). ttfs uses temporal coding with < 1 spike/neuron. the ttfs result demonstrates that the information capacity of temporal codes far exceeds rate codes per spike, but exploiting this requires infrastructure that todorov lacks: meaningful spike timing, temporal integration, and a decoder that reads timing rather than rates.

confidence: high. published benchmark with clear methodology.

## ternary weight efficiency

### matmul-free language models

zhu, r. et al. (2024). scalable matmul-free language modeling. *advances in neural information processing systems (NeurIPS) 37*.

key finding: matmul-free lm replaces all matrix multiplications with ternary weight ({-1, 0, +1}) operations, reducing gpu memory usage by 61% and achieving competitive perplexity at scales up to 2.7B parameters. the key insight is that ternary weights convert matrix-vector products into additions and subtractions (no multiplications needed), which can be implemented with bitwise operations. the memory savings come from storing weights as 2-bit values instead of 16-bit.

relevance to neural machine: todorov uses ternary activation spikes, not ternary weights. the matmul-free result shows that ternary quantization at the weight level is viable at scale. combining ternary weights with ternary activations would produce a fully ternary forward pass where the entire computation reduces to additions and subtractions. see [[ternary_compression_research]] for the broader context of extreme quantization.

confidence: high. published at neurips with scaling experiments up to 2.7B.

## practical guidance for todorov

the research converges on several actionable conclusions:

1. **at 40% firing rate, dense beats sparse.** do not invest in sparse gpu kernels. use dense computation through ternary spikes and fuse operations to reduce kernel launch overhead.

2. **triton is the right kernel target.** triton achieves 80-100% of raw cuda performance while remaining python-based and maintainable. all todorov custom kernels (fla chunk_kda, future spike kernels) should target triton.

3. **fuse the spike operation.** the threshold computation, sign extraction, and ste backward pass should be a single triton kernel, not three pytorch operations. this is the lowest-hanging fruit for spike implementation performance.

4. **parallel scan is solved.** multiple methods exist for parallelizing spiking neuron dynamics. if atmn gains temporal dynamics, psn (remove reset) or spikingssms (soft gating reset) are the most practical paths.

5. **gradient checkpointing per chunk.** for recurrent models with long sequences, checkpoint the recurrent state every K timesteps and recompute within each chunk during backward. this trades ~33% compute for ~K-fold memory reduction.

## see also

- [[ternary_spikes]]
- [[training_efficiency]]
- [[neuron_models_to_atmn]]
- [[ternary_compression_research]]
- [[sparse_coding_to_ternary_spikes]]
