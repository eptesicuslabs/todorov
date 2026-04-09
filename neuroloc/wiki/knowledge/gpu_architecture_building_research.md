# gpu architecture building research

curated practical guidance on building custom neural architecture components on gpu hardware. this article covers the engineering stack -- kernel development, gradient verification, memory management, profiling -- that translates architecture ideas into trainable code. the focus is on what todorov specifically needs: custom recurrent layers, spike quantization, outer-product accumulation, and hybrid attention/recurrence.

## kernel development

### triton over raw cuda

triton (openai, 2021-present) provides a python-based gpu kernel programming model that achieves 80-100% of hand-tuned cuda performance for most workloads. the key abstraction is pointer arithmetic over blocked memory regions: the programmer specifies the computation in terms of blocks of data loaded from and stored to global memory, and the triton compiler handles thread mapping, shared memory allocation, and register allocation.

for todorov's use cases (outer-product accumulation, delta-rule state update, ternary spike forward/backward), triton is the correct choice over raw cuda for three reasons: (1) python-based development integrates with the existing pytorch codebase, (2) the compiler handles the most error-prone aspects of gpu programming (bank conflicts, thread divergence, memory coalescing), (3) fla (flash-linear-attention) already provides production-quality triton kernels for delta-rule layers that can be extended.

the 20% performance gap vs hand-tuned cuda is acceptable because kernel correctness and development speed matter more than peak throughput at research scale. at deployment scale, performance-critical kernels can be rewritten in cuda if needed.

### torch.autograd.Function and ste for spike backprop

pytorch's torch.autograd.Function provides the mechanism for custom forward/backward passes. for ternary spikes, the forward pass computes sign(x) * [|x| > threshold] (non-differentiable) and the backward pass uses the straight-through estimator (ste): treat the quantization as identity, passing gradients through unchanged where the neuron was active and zeroing them where it was inactive.

the implementation pattern:

    class TernarySpike(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, threshold):
            spikes = torch.sign(x) * (x.abs() > threshold).float()
            ctx.save_for_backward(x, threshold)
            return spikes

        @staticmethod
        def backward(ctx, grad_output):
            x, threshold = ctx.saved_tensors
            mask = (x.abs() > threshold).float()
            return grad_output * mask, None

the ste is a crude approximation (the true gradient of a step function is zero almost everywhere, infinite at the threshold). improved alternatives exist (see [[ternary_compression_research]]) but ste remains the standard because it works reliably at scale and introduces no additional hyperparameters.

## memory patterns

### flash attention tiling for outer-product accumulation

the flash attention algorithm (dao et al., 2022) introduced a tiling pattern that generalizes beyond softmax attention: load blocks of Q, K, V into shared memory (sram), compute partial results within the tile, accumulate across tiles, and write the final result to global memory (hbm). the key insight is that sram is ~10x faster than hbm, so restructuring computation to maximize sram reuse dramatically improves throughput.

this tiling pattern applies directly to outer-product accumulation in kda's delta-rule state update. the state update S_t = diag(alpha) * S_{t-1} + beta_t * k_t * v_t^T involves an outer product (k_t * v_t^T) that can be tiled: load blocks of k and v into sram, compute the outer product block, accumulate into the corresponding block of S. the fla library implements this pattern for delta-rule layers via chunkwise triton kernels.

### parallel scan via fla chunkwise triton

the fla (flash-linear-attention) library provides production-quality triton kernels for linear attention variants including delta-rule layers. the implementation uses chunkwise parallel scan: the sequence is divided into chunks of size C (typically 64-256), within each chunk the recurrence is computed sequentially, and across chunks a parallel scan propagates the state. this achieves O(T) total work with O(T/C) parallel steps.

for todorov, fla's chunk_kda kernel is the production implementation for kda layers. the kernel handles: (1) chunkwise state accumulation with exponential decay (alpha), (2) data-dependent write gating (beta), (3) outer-product state update, (4) query-state readout. autograd support is included -- the backward pass is a separate triton kernel that computes gradients with respect to q, k, v, alpha, and beta.

known issue: chunk_kda produces nan at d_model=1024 (works at d=384). the likely cause is numerical overflow in the outer-product accumulation at larger head dimensions. the fix applied in run_010 was l2 normalization of k and v before the outer product.

### gradient checkpointing per timestep chunk

for recurrent models processing long sequences, gradient checkpointing trades compute for memory: save the recurrent state only at chunk boundaries (every K timesteps), and during the backward pass, recompute the intermediate states within each chunk from the saved boundary state. this reduces memory from O(T) activations to O(T/K) activations at the cost of recomputing each chunk once during backward.

the standard pytorch implementation uses torch.utils.checkpoint.checkpoint. for recurrent layers, the checkpointing boundary should align with the chunkwise parallel scan boundaries in fla, so that the recomputation during backward uses the same efficient kernel as the forward pass.

## numerical precision

### bf16 default, fp8 on h100 with transformer engine

bf16 (bfloat16) is the default training precision for todorov. bf16 provides the same dynamic range as fp32 (8 exponent bits) with reduced precision (8 mantissa bits vs 23), which is sufficient for most neural network operations. the main advantage over fp16 is that bf16 never needs loss scaling -- its exponent range covers the full fp32 range.

fp8 (8-bit floating point) is available on h100 gpus via nvidia's transformer engine library. fp8 provides ~2x throughput improvement over bf16 for matrix multiplications but requires careful management of per-tensor scaling factors. for todorov, fp8 is relevant only for deployment optimization, not for research training: the numerical sensitivity of outer-product accumulation and ternary spike thresholds may not tolerate fp8 precision without extensive tuning.

t4 gpus lack bf16 tensor cores. on t4, use fp16 with loss scaling for all fla kernels.

## verification

### gradcheck and reference implementations

every custom kernel must be verified against a reference pytorch implementation before deployment. the verification protocol:

1. write a pure-pytorch reference implementation that is obviously correct (no performance optimization, no fusion, no tricks).
2. run torch.autograd.gradcheck comparing the custom kernel's backward pass against numerical finite differences. use double precision (fp64) for gradcheck.
3. compare forward pass outputs between the custom kernel and reference implementation at bf16 precision. maximum absolute error should be < 1e-3; maximum relative error should be < 1e-2.
4. run on random inputs of multiple sizes (small, medium, large) to catch size-dependent bugs.

the run_003-007 matmul path state_approx bug was caught because gradcheck was not run against a reference implementation at full sequence length. the reference implementation would have revealed that the last-timestep-only approximation produced different gradients than full accumulation.

### nsight compute roofline profiling

nvidia nsight compute provides roofline analysis: it measures the arithmetic intensity (flops per byte of memory traffic) and throughput of each kernel, and compares against the theoretical peak for the gpu. a kernel below the roofline is memory-bound; a kernel at the roofline is compute-bound; a kernel above the roofline indicates a measurement error.

for todorov's kernels:
- kda state update (outer product + accumulation): likely memory-bound due to large state matrix reads/writes.
- ternary spike forward: trivially compute-bound (comparison + sign extraction).
- fla chunk_kda: designed to be at the roofline via flash-attention-style tiling.

roofline analysis should be run before optimizing any kernel, to determine whether the bottleneck is memory bandwidth or compute throughput.

## fla library integration

the fla (flash-linear-attention) library is the production kernel backend for todorov's delta-rule layers. key integration points:

- chunk_kda (not chunk_gated_delta_rule) is the correct kernel for kda layers. it supports channel-wise decay (alpha) and data-dependent write gating (beta).
- the hybrid threshold determines when to use fla kernels vs naive matmul: fla for seq >= 512, matmul for seq < 512. below 512, the kernel launch overhead exceeds the computational savings.
- fla requires triton 3.2.0 on t4 gpus (triton 3.3+ dropped sm_75 support). pin torch==2.6.0 + triton==3.2.0 for t4 compatibility.
- fla kernels use fp16, not bf16, on t4 (no bf16 tensor cores).
- autograd support is built in: fla kernels register custom backward functions that compute gradients via a separate triton kernel.

## see also

- [[training_efficiency]]
- [[delta_rule_theory]]
- [[kda_channel_gating]]
- [[gpu_spike_implementation_research]]
- [[ternary_compression_research]]
