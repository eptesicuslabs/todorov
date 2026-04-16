# Mamba-3 Architecture

status: current (as of 2026-04-16).

Source paper: "Mamba-3: Improved Sequence Modeling using State Space Principles"
ArXiv ID: 2603.15569
Authors: Albert Gu (CMU), Tri Dao (Princeton), et al.
Published: ICLR 2026


## 1. Overview

Mamba-3 introduces three core innovations over Mamba-2:
1. Exponential-trapezoidal discretization (replaces exponential-Euler)
2. Complex-valued state space (enables data-dependent RoPE)
3. Multi-Input Multi-Output (MIMO) structure (improves hardware utilization)


## 2. Exponential-Trapezoidal Discretization

Mamba-2 used exponential-Euler (first-order) discretization:

    h_t = exp(Delta_t * A_t) * h_{t-1} + Delta_t * B_t * x_t

Mamba-3 uses exponential-trapezoidal (second-order) discretization:

    h_t = exp(Delta_t * A_t) * h_{t-1}
          + (1 - lambda_t) * Delta_t * exp(Delta_t * A_t) * B_{t-1} * x_{t-1}
          + lambda_t * Delta_t * B_t * x_t

This is a three-term recurrence (vs two-term in Mamba-2).

Key properties:
- Second-order accurate approximation of the state-input integral
- The recurrence can be expanded to reveal an implicit convolution on the SSM input
- Combined with explicit B, C bias terms, replaces the short causal convolution
  that was previously considered essential for recurrent models
- lambda_t controls the trapezoidal interpolation weight


## 3. Complex-Valued State Space

Mamba-3 uses complex-valued SSM parameters (A, B, C can be complex).

A complex SSM of state dimension N/2 is equivalent to a real SSM with:
- State dimension N (doubled)
- Transition matrix as a scalar-decayed block-diagonal of 2x2 rotation matrices

The rotation matrices are data-dependent (vary with input).

This equivalence means complex states achieve the same expressivity with
half the state dimension, resulting in 2x smaller states for equivalent
performance.


## 4. Data-Dependent RoPE (The "RoPE Trick")

Through the state space duality (SSD) framework:
- C corresponds to the query (Q) in attention
- B corresponds to the key (K) in attention

A discretized complex SSM is mathematically equivalent to a real-valued SSM
equipped with data-dependent Rotary Positional Embeddings (RoPE) applied to
the Q (from C) and K (from B) projections.

Standard RoPE uses fixed rotation frequencies. Mamba-3's data-dependent RoPE
has rotation angles that depend on the input, providing more expressive
position-dependent interactions.

This equivalence is established through Propositions 2-4 in the paper.


## 5. SISO vs MIMO

Mamba-1 and Mamba-2 use SISO (Single-Input Single-Output) recurrence:
- Each head processes one input channel and produces one output channel
- During autoregressive decoding, the recurrence is memory-bound
  (small matrix-vector operations that underutilize GPU compute)

Mamba-3 uses MIMO (Multi-Input Multi-Output):
- B and C parameters are tied (shared) across all heads
- Each head still has unique SSM input, output, and gate projections
- The projection size R can be increased with minimal overhead:
  state cost increases from DN to DNR per head
- With head-tying, the original SISO projection is kept and each dimension
  is element-wise scaled to size R with a learnable data-independent vector
- Result: DP + PR parameters per head (vs DP for SISO)

MIMO benefits:
- Better hardware utilization during decoding (larger matrix operations)
- 2x smaller state size for equivalent performance
- Negligible parameter increase due to tying B, C across heads


## 6. Differences from Mamba-2

    +----------------------------+-------------------+-------------------+
    | Feature                    | Mamba-2           | Mamba-3           |
    +----------------------------+-------------------+-------------------+
    | Discretization             | Exp-Euler (1st)   | Exp-Trap (2nd)    |
    | State values               | Real              | Complex           |
    | Recurrence terms           | 2                 | 3                 |
    | IO structure               | SISO              | MIMO              |
    | Causal conv needed         | Yes               | No (replaced)     |
    | Position encoding          | None explicit     | Data-dep RoPE     |
    | B, C sharing               | Per-head          | Tied across heads |
    | State size (equiv perf)    | N                 | N/2 (complex)     |
    +----------------------------+-------------------+-------------------+


## 7. State Space Duality (SSD)

The SSD framework (introduced in Mamba-2, arxiv 2405.21060) establishes that:
- Linear attention and state space models are dual views of the same computation
- The recurrent form is efficient for autoregressive decoding (O(1) per step)
- The parallel form (as attention) is efficient for training (O(L) parallelism)

Mamba-3 extends SSD by:
- Adding the complex-valued / RoPE equivalence
- Showing that MIMO can be expressed in both recurrent and parallel forms
- Demonstrating that exponential-trapezoidal preserves the duality


## 8. Practical Results

- Language modeling: outperforms Transformer++ baseline at equivalent FLOP
- Reasoning: reported 7x more efficient than Transformer at inference
- 2x smaller state than Mamba-2 for equivalent quality (due to complex states)
- The removal of causal convolution simplifies the architecture


## References

- Mamba-3: arxiv 2603.15569
- Mamba-2 (SSD / Transformers are SSMs): arxiv 2405.21060
- Mamba-1 (original): arxiv 2312.00752
- Together AI blog post: https://www.together.ai/blog/mamba-3
- Goomba Lab deep dive: https://goombalab.github.io/blog/2026/mamba3-part2/
- Cartesia blog: https://blog.cartesia.ai/p/mamba-3
