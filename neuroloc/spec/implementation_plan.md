# neural machine implementation plan

## target

one self-contained python file (~1500-2000 lines) that trains on 2x H200.
byte-level language model on FineWeb-Edu at 350M parameters.
must generate coherent text at inference time.

## architecture from blueprint

core operation: z = Q(R(B(C(x), C(h))))

### C: compression (k-WTA rate-coded selection)

replaces ternary spikes. for each activation vector x:
1. compute absolute magnitudes |x|
2. select top-k by magnitude (k = fraction * dim, e.g., 10-20%)
3. output: survivors keep their original value, rest are zero
4. this is rate-coded (continuous values) + sparse (most dimensions zero)
5. gradient: STE on the selection mask (gradient flows through selected dims)

parameters: k_fraction (learnable or fixed per layer)

### B: bilinear mixing (outer product + dot product)

two modes controlled by a learned mode signal:

**memory write (remember mode):**
    S_t = alpha_eff * S_{t-1} - beta * k * (k^T S_{t-1}) + beta * k * v^T
    (delta rule: erase old association, write new one)

where:
- alpha_eff = sigmoid(alpha_log + gamma * log(||S_t||))  (BCM-like)
- beta = sigmoid(beta_proj(x))  (data-dependent write gate)
- k = C(W_k @ x)  (compressed key)
- v = C(W_v @ x)  (compressed value)

**memory read (retrieve mode):**
    o = q^T S_t
    (content-addressable retrieval)

where q = C(W_q @ x)

### R: rotation (phase coding)

RoPE applied to q, k before storage/retrieval.
frequencies: theta_i = base^(-2i/d) with base=10000
this IS the phase coding -- fast dimensions = gamma, slow = theta

### Q: output shaping

RMSNorm + optional output projection

### layer structure

each layer:
1. RMSNorm(x)
2. C(x) -> compressed input
3. B(C(x), C(h)) -> mix with memory state
4. R(result) -> apply rotation
5. Q(result) -> shape output
6. x = x + output  (residual)
7. RMSNorm(x)
8. FFN: multi-compartment SwiGLU (K=4 sub-gates)
9. x = x + ffn_output  (residual)

### multi-compartment SwiGLU

K=4 independent sub-gates with block-diagonal projections:
    gate_k = silu(W_gate_k @ x_k)  for k=1..K
    up_k = W_up_k @ x_k
    out_k = gate_k * up_k
    out = W_down @ concat(out_1, ..., out_K)

where x_k = x[k*chunk : (k+1)*chunk] (block-diagonal input routing)

### model configuration (350M)

d_model: 1024
n_layers: 24
n_heads: 16
head_dim: 64 (d_model / n_heads)
ffn_hidden: 2816 (d_model * 2.75)
n_compartments: 4
k_fraction: 0.20 (20% activation, 80% zero)
alpha_log_init: -0.5 (alpha ~ 0.38, slower decay than todorov's -2.0)
gamma_bcm: 0.3 (validated by pilot)
vocab_size: 256 (byte-level)
max_seq_len: 2048
batch_size: 16 per GPU (32 total with 2x H200)

estimated params: ~350M
(24 layers * (attention: 4*d^2 + ffn: ~3*d*h + overhead))

### training

optimizer: AdamW (lr=3e-4, beta1=0.9, beta2=0.95, wd=0.1)
schedule: linear warmup 2000 steps, cosine decay to 1e-5
tokens: 7B (chinchilla-optimal for 350M)
precision: bf16 (H200 has bf16 tensor cores)
gradient checkpointing: per-layer
data: FineWeb-Edu, byte-level tokenization

### what is new vs todorov

1. rate-coded k-WTA replaces ternary spikes
2. delta rule erasure (targeted overwrite) in state update
3. BCM-like alpha (activity-dependent forgetting)
4. multi-compartment SwiGLU (K=4 block-diagonal sub-gates)
5. slower default alpha (alpha~0.38 vs alpha~0.12)
6. all 24 layers use the same recurrent write/read operation
   (no separate MLA or Mamba3 -- one unified operation)

### what stays the same

1. the CRBR equation structure
2. RoPE for position
3. RMSNorm
4. residual connections
5. byte-level next-token prediction
6. STE for gradient through k-WTA selection
7. the evaluation metrics (BPB, MI, CKA, firing rate)

## file structure

one file: `neuroloc/model/neural_machine.py`

sections:
1. configuration dataclass
2. k-WTA selection module (C)
3. recurrent memory module (B) with delta rule + BCM alpha
4. rotation module (R) -- RoPE
5. multi-compartment SwiGLU (FFN with K sub-gates)
6. single layer (C + B + R + Q + FFN)
7. full model (embedding + layers + output head)
8. training loop (data loading, optimizer, logging)
9. inference (greedy generation)
10. main()

## verification before H200

1. smoke test on CPU: 1 layer, d=64, seq=32, 10 steps
2. shape check: all tensor shapes correct through full forward pass
3. gradient check: gradients flow through k-WTA STE
4. memory check: 350M fits in bf16 on 2x H200 (350M * 2 bytes = 700MB model, 
   plus optimizer states ~2.8GB, plus activations with checkpointing ~20GB)
5. generation check: model produces valid byte sequences (even if garbage before training)

## risks

1. delta rule erasure adds one extra matmul per step (k * (k^T S)) -- may slow training
2. k-WTA selection at 20% means 80% of activations are zero -- STE gradient may be
   too sparse for stable training. fallback: start at 50% and anneal to 20%.
3. multi-compartment SwiGLU with K=4 block-diagonal changes the parameter count
   calculation -- verify 350M target is still met
4. alpha~0.38 means ~62% decay per step (vs 88% at alpha~0.12). state accumulates
   more, may saturate faster. BCM gamma=0.3 should prevent this but needs verification.
5. unified recurrence (no MLA) means no O(T^2) exact attention fallback. long-range
   retrieval depends entirely on the recurrent state. may need to add MLA back if
   quality degrades on long-range tasks.
