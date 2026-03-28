# Todorov: Unified Mathematical Formulation

Eptesicus Laboratories — Deyan Todorov
2026-03-27

---

## 1. The Single Object

A Todorov layer is a **Compressed Rotational Bilinear Recurrence** (CRBR). Every layer in the architecture — KDA, Mamba-3, MLA, SwiGLU, spikes, GP — is a specific instantiation of:

    z_t = Q(R(B(C(x_t), C(h_{t-1}))))

where:

- **C** is a compression operator (information bottleneck)
- **B** is a bilinear interaction (the computation)
- **R** is a rotational structure (geometric primitive)
- **Q** is a discretization operator (output quantization)

The architecture is a depth-wise composition of CRBRs with different parameterizations of C, B, R, Q at each layer, selected from the same family.

---

## 2. The Compression Family C

Every path through Todorov encounters a compression stage. All compressions are instances of a single family: **linear projection to a lower-dimensional sufficient statistic**, with the dimensionality determining the information-fidelity tradeoff.

### 2.1 Compression instances

**C_spike: Ternary quantization.** The most aggressive member.

    C_spike(x) = sign(x) * [|x| > theta]

where theta = alpha * mean(|x|). Maps R^d -> {-1, 0, +1}^d.
Information: 1.58 bits per dimension. Energy: 0.03 pJ or 0 pJ per INT8 add/skip (vs 4.6 pJ for FP32 MAC at 45nm CMOS, Horowitz 2014 ISSCC).
Applied to: KDA K/V paths. Optionally all 132 linear projection inputs.

**C_spike^ATMN: Membrane-potential ternary quantization.** Stateful extension.

    h_t = x_t + (1/tau) * u_{t-1}                  (membrane integration)
    C_spike^ATMN(x_t) = sign(h_t) * [|h_t| > V_th] (fire)
    u_t = h_t - C_spike^ATMN(x_t) * V_th            (reset)

where V_th = exp(a), a is learnable per-neuron. Adds temporal memory to the compression — the membrane potential accumulates sub-threshold inputs across timesteps. Same output space {-1, 0, +1}^d, but the firing decision integrates history.

**C_latent: Low-rank projection.** The MLA compression.

    C_latent(x) = W_down x,    W_down in R^{d_c x d_model}

Maps R^{d_model} -> R^{d_c} (d_c = 64 at 6M scale, 128 at 280M scale). Lossless up to rank d_c. Invertible via learned decompression W_k_up, W_v_up. Applied to: MLA KV cache. Each token is individually compressed and individually recoverable.

**C_state: Implicit recurrent compression.** The KDA and Mamba-3 state.

KDA compresses the entire history into S in R^{d_k x d_v}: information capacity O(d^2) regardless of sequence length. The delta rule manages this capacity actively — erase content aligned with the current key before writing.

Mamba-3 compresses history into h in C^{d_inner x d_state}: information capacity O(d * n) regardless of sequence length. Exponential decay manages temporal bandwidth.

**C_gate: Soft gating.** The SwiGLU dead zone.

    C_gate(x) = silu(W_gate x) * W_up x

Where silu approaches zero, the gate kills features. This is a learned continuous-valued sparsity — a soft version of C_spike.

**C_quant: Post-training quantization.** INT8 discretization of weights.

    C_quant(W) = round(W / scale) * scale

Applied globally at deployment. The rotation-based variants (SpinQuant, QuaRot) pre-multiply by a rotation matrix R that redistributes outliers before quantization:

    C_quant^rot(W, x) = round(RW / scale) * scale,  applied to R^{-1}x

This rotation R is itself a member of the rotational family (section 4).

### 2.2 Composition

Compressions compose. In a KDA layer with spikes and INT8 deployment:

    effective input = C_quant(W_k) @ C_spike(x)

The key projection weight is INT8-quantized, the input activation is ternary-quantized, and the matmul between them reduces to integer addition (ternary * INT8 = INT8 add/sub/skip). Every compression stage reduces the cost of every downstream computation.

In MLA with potential future KV cache quantization (KVQuant, rotation-based methods):

    cache = C_kv_quant(C_latent(x))

Double compression: first low-rank to d_c = 128, then 3-4 bit quantization of the latent vector. The two compressions are complementary — C_latent removes redundancy across dimensions, C_kv_quant reduces precision of the remaining dimensions.

### 2.3 The SBDS training interface

The Spike-aware Bidirectional Distillation Strategy (SBDS) is a training procedure that optimizes the compression-computation interface. Given a dense teacher T (no spikes) and a spiking student S:

    L_SBDS = alpha * KL(S || T) + beta * KL(T || S) + gamma * ||f_pre(S) - f_pre(T)||^2

The forward KL (alpha = 0.2) matches the student to the teacher's mode. The reverse KL (beta = 0.7) prevents the student from spreading probability mass where the teacher doesn't. The pre-norm feature alignment (gamma) matches intermediate representations before the compression stage, not after — this is critical because post-compression representations are discrete and gradient-uninformative.

SBDS is the training-time calibration of C_spike within the CRBR framework. It does not change the architecture. It optimizes how well the compression preserves what the bilinear interaction needs.

---

## 3. The Bilinear Family B

Every layer's core computation is a bilinear map — two inputs, one output, linear in each argument. All bilinear interactions in Todorov are instances of one family.

### 3.1 Bilinear instances

**B_outer: Outer product.** The KDA write operation.

    B_outer(k, v) = k v^T in R^{d_k x d_v}

Writes an association between key and value into the state matrix. Rank-1 update. This is the atomic unit of associative memory.

**B_dot: Inner product.** The attention score.

    B_dot(q, k) = q^T k in R

Measures similarity between query and key. Scalar output. Both MLA attention logits and KDA retrieval use this.

**B_gate: Element-wise product.** SwiGLU and Mamba-3 gating.

    B_gate(a, b) = a * b in R^d  (Hadamard product)

Selective amplification/suppression. The gate controls which features pass through.

**B_GP: Geometric product.** The G(3,0,1) PGA bilinear map.

    B_GP(a, b) = sum_{i,j} a_i b_j sigma(i,j) e_{cayley(i,j)}

where sigma and cayley are the sign and index tables of G(3,0,1). 192 non-zero entries in the 16x16 Cayley table. This is the most general bilinear interaction in the architecture — it simultaneously computes scalar products (grade 0), vector products (grade 1), area products (grade 2), volume products (grade 3), and pseudoscalar products (grade 4) between two multivectors.

### 3.2 Relationships

B_dot is a restriction of B_GP: for two pure 1-vectors a, b in G(3,0,1), the scalar part (grade 0) of the geometric product equals the inner product: grade_0(B_GP(a, b)) = a . b.

B_gate (Hadamard product) is a separate bilinear map in the same family, not a restriction of B_GP. The element-wise product preserves index position, while the geometric product mixes grades -- these are algebraically distinct operations.

B_outer (rank-1 tensor product) maps to a different output space than B_GP (R^{d x d} vs R^16) and no algebraic embedding makes it a special case. Both are bilinear maps over their respective spaces.

The architecture has one family of bilinear primitives at different specialization levels. B_GP is the most general member. Not everything should use B_GP, but understanding the family reveals which compositions are mathematically natural.

### 3.3 XSA as bilinear output correction

Exclusive Self Attention (arXiv 2603.09078) modifies the output of B_dot in the attention layers. Standard attention computes:

    y_i = sum_j softmax(B_dot(q_i, k_j)) v_j

XSA subtracts the self-value component:

    z_i = y_i - (y_i^T v_i / ||v_i||^2) v_i

This is a **projection operator** in the bilinear output space. It forces the attention to capture only information orthogonal to what the token already carries. In the CRBR framework, XSA is a post-bilinear correction P_perp applied to MLA's output:

    MLA_XSA(x) = P_perp(B_dot(Q, K) V)

where P_perp(y, v) = y - (y^T v / ||v||^2) v.

Composability with MLA: XSA operates after the bilinear interaction (attention weighted sum) and before the output projection. MLA's latent compression operates before the bilinear interaction (compressing K, V). They occupy non-overlapping stages of the CRBR pipeline, which is why they compose without interference.

Open question: whether the self-value bias manifests identically when v_i comes from MLA's W_v_up(c_kv) decompression rather than a direct W_v projection.

### 3.4 Differential Attention as bilinear structure

Differential Attention computes attention as a difference of two softmax maps:

    A_diff = softmax(Q_1 K_1^T) - lambda * softmax(Q_2 K_2^T)

This is a **signed bilinear interaction** — the differential amplifier cancels common-mode noise. In the CRBR framework, it replaces the single B_dot with a pair B_dot^+ - lambda * B_dot^-, producing naturally sparse attention patterns. The reduced outliers directly improve C_quant compatibility.

XSA + Differential Attention compose: Differential Attention modifies the bilinear structure (how attention scores are computed), XSA modifies the bilinear output (what the attention extracts). Both can apply simultaneously.

---

## 4. The Rotational Family R

Every component uses rotational geometry, drawn from a hierarchy of rotation groups. The hierarchy is algebraically nested: each is a sub-algebra of the next.

### 4.1 The hierarchy

**R_pos: SO(2) position rotations.** RoPE in KDA and MLA.

    R_pos(x, t) = [x_{2i} cos(t*f_i) - x_{2i+1} sin(t*f_i),
                   x_{2i} sin(t*f_i) + x_{2i+1} cos(t*f_i)]

Fixed frequencies f_i = 1/base^{2i/d}. Position-dependent, data-independent. Encodes relative position into dot-product similarity. Lives in SO(2)^{d/2} — d/2 independent 2D rotations.

**R_dyn: SO(2) dynamic rotations.** Mamba-3 complex state evolution.

    R_dyn(h, t) = [Re(h) cos(theta_t) - Im(h) sin(theta_t),
                   Re(h) sin(theta_t) + Im(h) cos(theta_t)]

Data-dependent angle theta_t = f(x_t). Same SO(2) rotation as RoPE, but the frequency is input-determined. Encodes temporal dynamics into state phase.

**R_spatial: Pin(3,0,1) rotors.** GP self-interaction in SwiGLU.

    R_spatial(x) = B_GP(W_left(x), W_right(x))

The geometric product of two multivectors produces a result that encodes rotations, reflections, and translations of 3D projective space. The rotor group Pin(3,0,1) is the double cover of the Euclidean group E(3). This is the maximal rotation group in the architecture.

### 4.2 Nesting

Spin(2) ≅ SO(2) ≅ U(1) ⊂ Spin(3) ⊂ Pin(3,0,1)

- Complex numbers = even sub-algebra of Cl(2,0,0) = 2D rotors = Spin(2)
- Spin(2) and SO(2) are isomorphic as Lie groups (not nested -- the same group)
- RoPE pairs = d/2 copies of Spin(2) rotors
- Mamba-3 complex state = same Spin(2) rotors with data-dependent angle
- PGA multivectors = full Cl(3,0,1) containing all lower algebras

Each layer uses the smallest rotation group that serves its function — a design choice, not a mathematical necessity. Lifting Mamba-3 to an explicit Cl(2,0,0) rotor parameterization is a notational change, not a computational one — the same two floats, the same rotation, but formally positioned within the algebraic hierarchy.

### 4.3 CoPE as learned R_pos

Contextual Position Encoding (CoPE, arXiv 2405.18719) replaces fixed-frequency RoPE with context-dependent positional gates:

    R_pos^CoPE(x, t) = R_pos(x, g(x_t))

where g is a learned gating function. This makes R_pos data-dependent, closing the gap between R_pos and R_dyn. In the CRBR framework, CoPE is a smooth interpolation between the positional and dynamic rotation regimes.

### 4.4 Rotation-based quantization as R in C_quant

SpinQuant and QuaRot use rotation matrices to redistribute activation magnitudes before quantization. The rotation R is learned (SpinQuant, via optimization over the rotation group using spin parametrization) or fixed (QuaRot, Hadamard matrix).

    C_quant^rot(x) = Q(R x),    R in O(d)

This is a member of the rotational family R applied inside the compression family C. The rotation does not encode position or dynamics — it encodes quantization geometry. It rotates the activation space so that the quantization grid aligns with the data distribution. In the Quamba variant for SSMs, this rotation must account for the distinct outlier patterns in Mamba-3's output tensor (different from Transformer patterns).

---

## 5. The Recurrence Family

The time evolution of state is the backbone. Two regimes, one family.

### 5.1 The semi-separable recurrence

KDA and Mamba-3 both implement:

    H_t = Lambda_t H_{t-1} + Gamma_t X_t

**KDA parameterization:**

    H_t = (I - beta_t k_t k_t^T) diag(alpha) H_{t-1} + beta_t k_t v_t^T

- H_t in R^{num_heads x d_k x d_v}: matrix-valued state
- Lambda_t = diag(sigmoid(alpha_log)): channel-wise decay, data-independent (learned, not input-dependent)
- Gamma_t = sigmoid(beta_proj(x_t)): data-dependent write gate
- X_t = k_t v_t^T: rank-1 outer product (bilinear B_outer)
- Erasure: (I - beta_t k_t k_t^T) erases content aligned with the current key before writing
- Output: o_t = H_t q_t (retrieval via B_dot)

The delta rule: the update performs online gradient descent minimizing ||H k_t - v_t||^2 with learning rate beta_t. The erasure term is what distinguishes KDA from plain gated linear attention -- it actively manages state capacity by removing stale associations before writing new ones.

**Mamba-3 parameterization:**

    h_t = A_bar_t h_{t-1} + B_bar_t (B_t * x_t)

- h_t in C^{d_inner x d_state}: vector-valued state (complex via R_dyn)
- Lambda_t = A_bar_t: discretized state transition (ZOH via mamba_ssm fused kernel; trapezoidal in manual fallback: (1 + dtA/2)/(1 - dtA/2))
- Gamma_t = B_bar_t: discretized input coupling
- X_t = B_t * x_t: gated input (bilinear B_gate)
- Output: y_t = (h_t * C_t).sum() (readout via B_dot)

**Equivalence via SSD:** When unrolled, both produce lower-triangular semi-separable output matrices Y = M X where M has off-diagonal blocks of bounded rank. KDA's M has rank d_k per head. Mamba-3's M has rank d_state per channel. The chunkwise parallel algorithms for both exploit this structure identically — process chunks of L tokens with intra-chunk matmuls and inter-chunk state propagation.

### 5.2 MLA: the non-recurrent complement

MLA does not fit the semi-separable recurrence. It stores per-token compressed representations and computes full pairwise interactions:

    output = softmax(Q K^T / sqrt(d)) V

This is O(t^2) in context length, O(t * d_c) in memory. The compression C_latent reduces the memory constant but not the scaling class.

The architectural principle: recurrent layers (KDA, Mamba-3) accumulate and compress the history into fixed-size states. The attention layer (MLA) performs exact attention over lossy-compressed per-token representations — the softmax-weighted sum is exact, but the K/V vectors it operates on are rank-d_c approximations of the full representations. The 3:1 ratio allocates 75% of depth to accumulation and 25% to compressed-exact retrieval.

### 5.3 DSA as sparse MLA

DeepSeek Sparse Attention (DSA) modifies the MLA bilinear interaction with three parallel branches:

    A_DSA = A_compressed + A_selected + A_sliding

- Compressed: coarse-grained attention over block summaries
- Selected: fine-grained attention over dynamically chosen tokens
- Sliding: local window attention

This reduces MLA's O(t^2) to approximately O(t * sqrt(t)) while preserving exact retrieval for important tokens. In the CRBR framework, DSA adds a structured sparsity pattern to MLA's bilinear stage.

### 5.4 Zamba2 weight sharing

Zamba2's insight: if the attention layers serve as exact-retrieval anchors, their weights can be shared across depth. All MLA layers use the same W_q, W_kv_down, W_k_up, W_v_up, with per-layer LoRA adapters for specialization:

    W_mla^(i) = W_shared + Delta_i,   Delta_i = A_i B_i (low-rank)

This dramatically reduces the parameter cost of the attention component. In a 24-layer Todorov with 3 MLA layers, sharing saves ~2/3 of MLA parameters. The LoRA adapters preserve per-layer specialization at minimal cost.

### 5.5 Hymba meta tokens

Hymba introduces learnable tokens m in R^{n_meta x d_model} prepended to the sequence, visible to all layers. These tokens function as persistent global memory — information written to meta tokens by early layers is readable by all later layers without relying on the recurrent state's finite capacity.

In the CRBR framework, meta tokens are an additional persistent state that bypasses the compression bottleneck. The recurrent state H_t is lossy and fixed-size. The MLA cache C_t is lossless but grows with context. Meta tokens are a third option: fixed-size, lossless for the n_meta most critical pieces of information.

---

## 6. The MLP as Bilinear Self-Interaction

SwiGLU is the bilinear computation inside every block:

    MLP(x) = W_down(silu(W_gate x) * W_up x)

The gating (B_gate) is a bilinear interaction between two projections of the same input. The GP self-interaction adds a second bilinear channel:

    MLP_GP(x) = W_down(silu(W_gate x) * W_up x) + W_gp(B_GP(W_left x, W_right x))

Both terms are bilinear self-interactions on x. The first (SwiGLU) operates in R^{hidden_dim} with element-wise product. The second (GP) operates in G(3,0,1) = R^16 with the geometric product. They are additive — the GP term is a residual correction that injects geometric structure without disrupting the learned SwiGLU representations.

The GP residual adds 98K parameters per layer (1.7% overhead) and zero measurable compute overhead (2.9 s/step with and without, empirically validated in Phase 3). The 192-entry sparse Cayley table makes B_GP cheaper than a dense 16x16 bilinear map.

---

## 7. The Training Framework

### 7.1 muP as scale-invariant parameterization

Maximal Update Parameterization (muP, arXiv 2203.03466) ensures that the CRBR framework's optimal hyperparameters transfer across scales. The key property: if the learning rate, initialization variance, and layer-wise scaling are set correctly at width d, they remain optimal at width kd for any k.

For Todorov: tune at ~35M proxy (d_model=256, 8 layers), transfer to 350M (d_model=1024, 24 layers). The proxy ratio of 10x is within the validated transfer range.

muP affects every CRBR component:

- C_spike: threshold alpha scales with activation magnitude, which muP stabilizes across widths
- B_outer, B_dot: the outer product and dot product scale differently with d; muP prescribes the correct normalization
- R_pos: RoPE frequencies are independent of width (no muP interaction)
- W_down, W_up, W_gate: muP prescribes fan-in-dependent initialization and per-layer LR

### 7.2 DoReMi as data-distribution optimization

DoReMi (arXiv 2305.10429) optimizes the training data mixture using group distributionally robust optimization (group DRO). A small proxy model (same 35M used for muP) learns domain weights that minimize worst-case downstream loss. These weights transfer to the full-scale model.

DoReMi does not directly interact with the CRBR compression stages. It optimizes the input distribution, which indirectly determines what representations the compression and bilinear families must learn to preserve. Different data domains stress different components — code exercises long-range state (KDA recurrence), math exercises attention precision (MLA bilinear), spatial exercises geometric structure (GP bilinear) — but this interaction is indirect, mediated through the training loss.

### 7.3 LLM-JEPA as multi-view objective

The Joint Embedding Predictive Architecture (arXiv 2509.14252) adds an embedding-space prediction objective alongside next-token prediction:

    L = L_NTP + lambda * L_JEPA

where L_JEPA uses natural multi-view pairs (e.g., text and code describing the same function). This is a training-time enrichment of the bilinear representations — it forces B_outer and B_GP to learn embeddings that are predictive across views, not just within a single modality.

---

## 8. The Quantization Pipeline

At deployment, the full CRBR framework undergoes systematic compression:

### 8.1 Weight quantization

    W -> C_quant^rot(W) = round(R W / scale) * scale

Using SpinQuant/QuaRot rotation-based approach. The rotation R (learned or Hadamard) is a member of the rotational family R applied for quantization geometry rather than positional or dynamic encoding.

### 8.2 Activation quantization

KDA/MLA activations: SmoothQuant (per-channel scaling) or rotation-based (SpinQuant).
Mamba-3 activations: Quamba (Hadamard transforms + percentile clipping for SSM-specific outlier patterns in the output tensor y).

### 8.3 KV cache quantization

MLA latent cache: rotation-based quantization (e.g., PolarQuant polar coordinate decomposition) + bias correction (e.g., QJL quantized Johnson-Lindenstrauss). Note: specific venue claims for these techniques require verification.

    cache_quantized = C_rotation(C_latent(x))

This is triple compression: d_model -> d_c (MLA latent), then FP32 -> 3-4 bit (rotation-based quantization), then bias-corrected. At d_c=128 with 3-bit quantization: 48 bytes per token per layer vs 8192 bytes for standard FP32 MHA. ~170x compression.

### 8.4 Spike-quantization interaction

Ternary spikes (C_spike) applied before INT8 weight projections (C_quant):

    output = C_quant(W) @ C_spike(x)

This is ternary input x INT8 weight = INT8 result via addition/subtraction/skip. At 42% firing rate, 58% of operations are skips (0 pJ). The remaining 42% are INT8 add/sub (~0.03 pJ). The expected operation cost is ~0.013 pJ, a ~350x reduction from FP32 MAC (4.6 pJ).

---

## 9. The Complete Layer

A Todorov block instantiates the CRBR framework twice (attention + MLP) with a residual:

    x' = x + Attn(RMSNorm(x))
    x'' = x' + MLP(RMSNorm(x'))

### 9.1 KDA block (18 of 24 layers)

Attention path:

    q = R_pos(W_q x)                      # project, rotate (Q spikes not yet implemented)
    k = R_pos(C_spike(W_k x))            # project, spike, rotate
    v = C_spike(W_v x)                   # project, spike
    S_t = (I - beta_t k_t k_t^T) diag(alpha) S_{t-1} + beta_t B_outer(k_t, v_t)  # delta rule: erase then write
    o_t = B_dot(q_t, S_t)                # retrieval with B_dot
    output = C_spike?(W_o o_t)           # project, optionally spike

MLP path:

    gate_in = C_spike?(x)
    up_in = C_spike?(x)
    hidden = B_gate(silu(W_gate gate_in), W_up up_in)     # bilinear gating
    base = W_down hidden
    gp = W_gp(B_GP(W_left x, W_right x))                  # GP bilinear (if spatial_mode)
    output = base + gp

### 9.2 Mamba-3 block (3 of 24 layers)

    xz = W_in x                          # project to 2*d_inner
    x_inner, z = split(xz)
    z = silu(z)                          # gate
    h_t = Lambda_t h_{t-1} + Gamma_t B_gate(B_t, x_inner_t)  # recurrence with B_gate
    h_t = R_dyn(h_t, t)                  # complex rotation
    y_t = B_dot(h_t, C_t)               # readout
    output = W_out(norm(B_gate(y, z)))   # gated output

### 9.3 MLA block (3 of 24 layers)

    c_kv = C_latent(x)                   # compress: d_model -> d_c
    k = W_k_up(c_kv)                     # decompress K
    v = W_v_up(c_kv)                     # decompress V
    k_rope = R_pos(W_k_rope(c_kv))       # shared RoPE for position
    q = W_q(x)
    q_rope = R_pos(W_q_rope(x))
    scores = (B_dot(q, k) + B_dot(q_rope, k_rope)) / sqrt(d + d_R)  # bilinear
    y = softmax(scores) @ v
    z = y - (B_dot(y, v_self) / ||v_self||^2) v_self      # XSA correction (planned)
    output = W_o(z)

---

## 10. The Architecture as One Equation

Todorov at depth L with input sequence X = (x_1, ..., x_T):

    Y = DecodeHead(CRBR_L(...(CRBR_2(CRBR_1(Embed(X))))))

where each CRBR_i is parameterized by:

    (C_i, B_i, R_i, Q_i, state_shape_i) in {KDA, Mamba3, MLA} x {with/without spikes} x {with/without GP}

selected by the layer schedule:

    schedule = (KDA, KDA, KDA, Mamba3, KDA, KDA, KDA, MLA) x 3

The schedule is the only discrete design choice. Everything else — the compression, the bilinear interaction, the rotational structure, the quantization — is a continuous parameterization within the CRBR family.

### 10.1 Nemotron-Flash NAS as schedule search

Nemotron-Flash's evolutionary search over operator types at each layer position is a search over CRBR schedules. The search space is:

    schedule[i] in {KDA, Mamba3, MLA} for each i in {0, ..., L-1}

with the constraint that the ratio of recurrent (KDA + Mamba3) to attention (MLA) layers is between 2:1 and 7:1 (validated range from literature). At 350M scale, this search is tractable on a single GPU.

---

## 11. Summary: What Is Unified

| Component | CRBR role | Family member |
|-----------|-----------|---------------|
| Ternary spike | Compression | C_spike |
| ATMN neuron | Compression | C_spike^ATMN |
| MLA latent | Compression | C_latent |
| SwiGLU gate | Compression | C_gate |
| INT8 weights | Compression | C_quant |
| SpinQuant rotation | Compression x Rotation | C_quant^rot |
| KV cache quant | Compression | C_rotation(C_latent) |
| KDA outer product | Bilinear | B_outer |
| Attention score | Bilinear | B_dot |
| SwiGLU gating | Bilinear | B_gate |
| GP self-interaction | Bilinear | B_GP |
| XSA correction | Bilinear output | P_perp(B_dot output) |
| Differential Attention | Bilinear structure | B_dot^+ - lambda B_dot^- |
| RoPE | Rotation | R_pos in SO(2)^{d/2} |
| CoPE | Rotation | R_pos^learned |
| Mamba-3 complex | Rotation | R_dyn in SO(2)^{d_state/2} |
| GP rotors | Rotation | R_spatial in Pin(3,0,1) |
| Quant rotations | Rotation | R in O(d) |
| KDA state | Recurrence | H in R^{d x d}, semi-separable |
| Mamba-3 state | Recurrence | h in C^{d x n}, semi-separable |
| MLA cache | Storage | C in R^{t x d_c}, per-token |
| Meta tokens | Storage | m in R^{n_meta x d}, persistent |
| muP | Scale invariance | Parameterization of all C, B, R |
| SBDS | Training | Calibration of C_spike |
| DoReMi | Training | Optimization of input distribution |
| LLM-JEPA | Training | Enrichment of B representations |
| DSA sparse | Efficiency | Sparsity pattern on MLA's B_dot |
| Zamba2 sharing | Efficiency | Parameter tying across MLA instances |
| Nemotron NAS | Architecture | Search over CRBR schedules |

Everything is C, B, R, or Q. Everything composes. The architecture is one object.
