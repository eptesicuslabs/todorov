# Delta Rule Theory: Online Learning in Linear Attention

Source papers:
- "Parallelizing Linear Transformers with the Delta Rule over Sequence Length"
  ArXiv ID: 2406.06484
- "DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products"
  ArXiv ID: 2502.10297
- "Gated Delta Networks: Improving Mamba2 with Delta Rule"
  ArXiv ID: 2412.06464


## 1. The Delta Rule as Online Regression

Standard linear attention uses an additive update:

    S_t = S_{t-1} + v_t * k_t^T

This is equivalent to a Hessian-like fast weight update. The hidden state S_t
acts as an associative memory that maps keys to values.

The delta rule replaces additive updates with error-correcting updates:

    S_t = S_{t-1} + beta_t * (v_t - S_{t-1} * k_t) * k_t^T

This is mathematically identical to a single step of stochastic gradient descent
on the online least-squares objective:

    L_t = || v_t - S * k_t ||^2

The update can be rewritten as:

    S_t = S_{t-1} + beta_t * v_t * k_t^T - beta_t * S_{t-1} * k_t * k_t^T
    S_t = (I - beta_t * k_t * k_t^T) * S_{t-1} + beta_t * v_t * k_t^T

where beta_t is the learning rate (step size) for the SGD update.


## 2. Error Correction Mechanism

The key difference from additive updates:

Additive (standard linear attention):

    S_t = S_{t-1} + v_t * k_t^T
    - Writes accumulate without bound
    - Old associations are never erased
    - State capacity saturates quickly

Delta rule:

    S_t = (I - beta_t * k_t * k_t^T) * S_{t-1} + beta_t * v_t * k_t^T
    - The term (I - beta_t * k_t * k_t^T) * S_{t-1} erases the old value
      associated with k_t before writing the new association
    - Acts as a Householder-like reflection
    - When beta_t = 1 and ||k_t|| = 1: exact replacement of old association
    - When beta_t < 1: partial correction (interpolation)

This error correction is the fundamental advantage. The state does not merely
accumulate associations but actively updates them, behaving like a continually
trained associative memory.


## 3. Capacity Analysis

For a state matrix S of size d_k x d_v:

Additive linear attention:
- Can store approximately d_k independent key-value associations
- Performance degrades as more keys are added beyond capacity
- Retrieval error grows linearly with the number of stored associations

Delta rule:
- Same theoretical maximum capacity (d_k associations)
- But achieves much better practical capacity due to error correction
- When keys are reused or similar, the delta rule updates rather than
  accumulates, preventing destructive interference
- DeltaNet (1.3B params, 100B tokens) outperforms Mamba and GLA on
  downstream tasks (arxiv 2406.06484)


## 4. Connection to Fast Weight Programming

Linear attention's hidden state S_t can be viewed as "fast weights" -- a
weight matrix that is rapidly updated during inference (as opposed to the
"slow weights" learned during training).

Historical lineage:
- Fast weight memories (Schmidhuber, 1992)
- Hessian update / outer product rule -> standard linear attention
- Widrow-Hoff / delta rule -> DeltaNet
- Multiple SGD steps per token -> DeltaProduct


## 5. DeltaProduct: Composition of Updates

DeltaProduct (arxiv 2502.10297) extends DeltaNet by taking n_h gradient
descent steps per token instead of one.

Single step (DeltaNet, n_h = 1):

    S_t = (I - beta_t * k_t * k_t^T) * S_{t-1} + beta_t * v_t * k_t^T
    State transition: M_t = I - beta_t * k_t * k_t^T  (rank-1 correction)

Multiple steps (DeltaProduct, n_h steps):

    M_t = product_{i=1}^{n_h} (I - beta_t^{(i)} * k_t^{(i)} * k_t^{(i)T})
    This is a product of n_h generalized Householder transformations.
    The result is identity plus a matrix of rank at most n_h.

Theoretical power (Cartan-Dieudonne theorem):
- Any d x d orthogonal matrix can be expressed as a product of at most d
  Householder reflections
- DeltaProduct with n_h >= d can represent any orthogonal state transition
- DeltaProduct with >= 4 layers (3 if n_h >= 2) can solve any group word problem
- Gated DeltaProduct with finite layers can recognize any regular language

Practical results:
- n_h = 1 -> DeltaNet (baseline)
- n_h = 2 -> sharp improvement in length extrapolation
- n_h = 3 -> near-zero performance degradation across sequence lengths
- n_h = 4 -> diminishing returns in most settings
- Outperforms DeltaNet on state-tracking and language modeling benchmarks


## 6. Hardware-Efficient Training

The main challenge: parallelizing the delta rule across sequence length.

DeltaNet algorithm (arxiv 2406.06484):
- Uses the WY representation to compute products of Householder matrices
- Enables chunkwise parallel training
- Within each chunk: parallel computation of state transitions
- Across chunks: sequential propagation of the d_k x d_v state
- Memory-efficient: avoids materializing the full d_k x d_k transition matrices

DeltaProduct algorithm:
- Extends the WY representation for products of multiple Householder matrices
- Integrated into the flash-linear-attention (fla) library
- Chunk size C controls the parallelism-memory tradeoff

Gated Delta Networks (arxiv 2412.06464):
- Adds Mamba-style gating: S_t = alpha * [delta-rule update]
- The gate alpha enables rapid memory erasure (alpha -> 0) while the
  delta rule enables targeted updates (alpha -> 1)
- Extended the WY-based parallel algorithm to incorporate gating terms
- Published at ICLR 2025


## 7. Key Equations Summary

Standard linear attention:
    S_t = S_{t-1} + v_t * k_t^T

DeltaNet:
    S_t = (I - beta_t * k_t * k_t^T) * S_{t-1} + beta_t * v_t * k_t^T

Gated DeltaNet:
    S_t = alpha_t * [(I - beta_t * k_t * k_t^T) * S_{t-1} + beta_t * v_t * k_t^T]

KDA (Kimi Delta Attention):
    S_t = (I - beta_t * k_t * k_t^T) * Diag(alpha_t) * S_{t-1} + beta_t * k_t * v_t^T

DeltaProduct (n_h steps):
    S_t = [prod_{i=1}^{n_h} H_i] * S_{t-1} + write term
    where H_i = I - beta_i * k_i * k_i^T


## References

- DeltaNet: arxiv 2406.06484
- DeltaProduct: arxiv 2502.10297
- Gated Delta Networks: arxiv 2412.06464
- Kimi Linear (KDA): arxiv 2510.26692
- Error-Free Linear Attention: arxiv 2512.12602
- DeltaNet blog: https://sustcsonglin.github.io/blog/2024/deltanet-1/
