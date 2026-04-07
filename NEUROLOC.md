# neuroloc

neuroloc is a neuroscience research wiki for eptesicus laboratories. it maps
biological neural computation mechanisms to todorov's compressed rotational
bilinear recurrence (CRBR) framework. the goal is not to summarize neuroscience
-- it is to understand biological computation deeply enough to design artificial
architectures that think in the same principled way the brain does.

todorov development is temporarily paused. todorov resumes when this research
phase produces sufficient biological grounding for every architectural decision.

## what todorov is

todorov is a proof that biological neural computation principles -- ternary
spikes, adaptive thresholds, recurrent state accumulation -- can outperform
standard transformers at matched scale. 0.663x BPB at 267M params (33.7%
better than transformer baseline). spike mutual information 1.168, CKA 0.732,
firing rate 40.8%. all gates passing across 10 runs.

the architecture is built on one mathematical object: the CRBR.

    z_t = Q(R(B(C(x_t), C(h_{t-1}))))

where C is compression (ternary spikes, latent projection, gating), B is a
bilinear interaction (outer product, dot product, geometric product), R is
rotational structure (RoPE, complex dynamics, PGA rotors), and Q is output
quantization. three layer types are different parameterizations:

- delta-rule layers (KDA): C=spike, B=outer product, R=RoPE, recurrence via
  error-correcting writes to a matrix-valued state
- continuous-dynamics layers (Mamba3): C=gate, B=element-wise product,
  R=complex rotation, recurrence via discretized differential equations
- latent-attention layers (MLA): C=low-rank projection, B=dot product,
  R=RoPE, exact retrieval over compressed per-token representations

layer schedule: (KDA, KDA, KDA, Mamba3, KDA, KDA, KDA, MLA) x 3
ratio: 75% KDA, 12.5% Mamba3, 12.5% MLA

## component registry

### KDA (18/24 layers)
file: src/layers/kda.py
delta-rule recurrence with channel-wise forgetting:
    S_t = diag(alpha) * S_{t-1} + beta_t * k_t * v_t^T
    o_t = q_t^T * S_t
- alpha_log: per-head per-channel forgetting rate (nn.Parameter)
- beta_proj: data-dependent write gate (sigmoid)
- ternary spikes on K, V projections (optional Q, O with spike_all)
- RoPE on Q, K
- three forward paths: recurrent (sequential), parallel (matmul), fla (chunk_kda)

### MLA (3/24 layers)
file: src/layers/mla.py
compressed exact attention:
    c_kv = W_down(x)  [d_model -> d_c]
    k = W_k_up(c_kv), v = W_v_up(c_kv)
    scores = (Q @ K^T + Q_rope @ K_rope^T) / sqrt(d + d_R)
    output = softmax(scores) @ V
- d_c=128: latent compression bottleneck
- d_R=32: decoupled RoPE dimension
- cache stores c_kv + k_rope per token (~170x compression)
- optional spikes on Q and KV inputs with spike_all

### Mamba3 (3/24 layers)
file: src/layers/mamba3.py
state space model with complex-valued state evolution:
    h_t = A_bar_t * h_{t-1} + B_bar_t * B_t * x_t
    h_t = rotate(h_t, theta_t)  [data-dependent RoPE]
    y_t = sum(h_t * C_t)
- d_state=32: state dimension
- expand=2: inner expansion factor
- trapezoidal discretization (biologically motivated)
- data-dependent rotation frequency via rope_freq parameter
- currently sequential scan only (parallel scan needed for scale)

### SwiGLU + geometric product
file: src/layers/swiglu.py
gated feedforward with optional spatial self-interaction:
    gate = silu(W_gate(x))
    up = W_up(x)
    out = W_down(gate * up)
    if spatial_mode:
        out = out + W_gp(geometric_product(W_left(x), W_right(x)))
- ratio=2.75 expansion
- G(3,0,1) projective geometric algebra: 16-component multivectors
- GP adds after down projection (not before -- run_009 bug)
- 98K params per layer overhead (1.7%), zero compute overhead

### ternary spikes
file: src/spikes/ternary_spike.py
adaptive threshold quantization:
    threshold = alpha * mean(|x|)
    spikes = sign(x) * [|x| > threshold]
    output: {-1, 0, +1}^d
- alpha_init=1.0, learnable
- STE (straight-through estimator) gradient
- running spike density tracking
- ~41% firing rate at alpha=1.0

### ATMN (adaptive threshold membrane neuron)
file: src/spikes/atmn_spike.py
membrane potential dynamics:
    h_t = x_t + (1/tau) * u_{t-1}
    spikes = sign(h_t) * [|h_t| > V_th]
    u_t = h_t - spikes * V_th
- V_th = exp(threshold_log): per-neuron learnable threshold
- tau=2.0: membrane time constant
- reset by subtraction (not hard reset)
- membrane potential reset to zero each batch during training

### geometric product
file: src/algebra/geometric_product.py
G(3,0,1) projective geometric algebra:
- 16-component multivectors (grades 0-4)
- sparse cayley table: 192 non-zero entries out of 256
- grade 0 (1 comp): scalar
- grade 1 (4 comp): vectors in R^3 + R^1
- grade 2 (6 comp): bivectors (oriented planes)
- grade 3 (4 comp): trivectors (volumes)
- grade 4 (1 comp): pseudoscalar
- three products: geometric, outer, inner (grade-selective)
- sandwich_product(rotor, x) for rigid body transformations

### spike health metrics
file: src/spikes/spiking_brain.py
- mutual information (MI): must be > 0.1 (measures spike-to-input dependence)
- centered kernel alignment (CKA): must be > 0.3 (structural similarity)
- firing rate: target 30-60%
- validated at: MI 1.168, CKA 0.732, FR 40.8% at 267M scale

## rules for wiki content

zero comments in code. no inline, no block, no docstrings, no TODOs.
zero emojis. anywhere. code, docs, commits, prints, logs, filenames.
zero AI attribution. no Co-authored-by. sole author: Deyan Todorov.
lowercase in all docs, commits, and readme. no tables in readme.
obsidian wikilinks [[page-name]] for all internal cross-references.

## the adversarial thinking rule

before committing any factual claim, mechanistic explanation, or
computational implication to a wiki page:

1. state the claim.
2. construct the strongest possible counter-argument.
3. evaluate both. if the counter-argument is stronger or equally plausible,
   note the controversy explicitly.
4. write the final version with the surviving position AND a challenges
   subsection documenting what you considered.

calibrated language: "strongly supported by" (multiple replications),
"suggested by" (single study), "speculated" (no direct evidence),
"contested" (active disagreement).

## brian2 simulation constraints

- must run on kaggle T4 CPU in < 10 minutes
- pip install brian2 (no exotic dependencies)
- network size: up to ~10,000 neurons
- simulation time: 1-10 seconds of biological time
- output: matplotlib figures saved as PNG, metrics as JSON
- zero comments in python code
