# Todorov Biology Map

Master reference mapping every Todorov component to its biological analog. For each component: what it does in Todorov, what it maps to in biology, what the mapping gets wrong, and how deep the correspondence runs.

## KDA (src/layers/kda.py)

KDA is a recurrent matrix-memory layer that maintains a matrix-valued associative memory. 18 of 24 layers (75%) are KDA. historical project language often called it a "delta-rule" layer, but the implemented update lacked the targeted erasure term that stronger label would imply.

Biological analogs:

- **State update: S_t = diag(alpha) * S_{t-1} + beta_t * k_t * v_t^T**
  - Hebbian learning: the outer product k_t * v_t^T is the classical Hebbian association rule. This is the most direct biological correspondence in the entire architecture.
  - Short-term plasticity: alpha decay resembles synaptic depression (activity-dependent decay of synaptic efficacy).
  - NOT STDP: no timing dependence. k and v come from the SAME timestep. STDP requires comparing activity at DIFFERENT times.

- **alpha (channel-wise forgetting rate): alpha = sigmoid(alpha_log)**
  - Homeostatic plasticity: prevents state saturation, maintains bounded activity. Different channels decay at different rates, allowing a hierarchy of memory timescales.
  - NOT neuromodulation: alpha is per-channel and FIXED after training. Biological neuromodulation is a small number of global, dynamic signals that change moment-to-moment.

- **beta (data-dependent write gate): beta_t = sigmoid(beta_proj(x_t))**
  - Acetylcholine system: input-dependent gating of memory encoding. High beta = encoding mode (write to state), low beta = preservation mode (read from state). The closest biological analog to any parameter in Todorov.
  - Mapping gap: beta sees only the current token x_t. Biological ACh is driven by global signals (novelty, uncertainty, task demands). beta has no access to global state.

- **Readout: o_t = q_t^T * S_t**
  - Pattern completion: content-addressable retrieval from associative memory, identical in structure to Hopfield network recall.
  - Mapping gap: KDA readout is LINEAR (one matrix-vector multiply). Hopfield recall uses iterative convergence with attractor dynamics. KDA has no error correction.

## MLA (src/layers/mla.py)

MLA performs softmax attention over compressed per-token representations. 3 of 24 layers (12.5%).

Biological analogs:

- **Softmax attention: softmax(Q @ K^T / sqrt(d)) @ V**
  - Pattern completion: Ramsauer et al. (2021) proved softmax attention is the update rule of a modern Hopfield network with exponential capacity 2^{d/2}.
  - NOT selective attention: MLA computes information retrieval (finding relevant past tokens). Biological attention computes resource allocation (directing processing to relevant stimuli). Different problems.

- **KV compression: c_kv = kv_down_proj(x), d_model -> d_c = 128**
  - Hippocampal memory: hippocampal indexing theory (Teyler and DiScenna 1986) proposes that the hippocampus stores compressed pointers to cortical patterns, not full patterns. MLA's compressed latent c_kv is a compressed index to the full token representation.
  - Mapping gap: hippocampal indices are stored through rapid synaptic modification (LTP). MLA stores c_kv in a passive cache with no learning.

- **Full cache (no forgetting)**
  - NOT memory consolidation: MLA retains every token until the context limit. No decay, no consolidation, no transfer to a slower system. The absence of forgetting is an engineering advantage, not a biological principle.

- **12.5% allocation (3/24 layers)**
  - NOT biologically derived. The 3:1 ratio comes from ML benchmarks (Kimi, Qwen3, OLMo independently converged on ~75/25). The resemblance to biological attention's sparsity is coincidental.

## Mamba3 (src/layers/mamba3.py)

Mamba3 is a state-space model with complex rotation and data-dependent discretization. 3 of 24 layers (at positions 4, 12, 20).

Biological analogs:

- **State evolution: h_t = A_bar_t * h_{t-1} + B_bar_t * B_t * x_t**
  - Leaky integrate-and-fire: discretized linear ODE with decay (A_bar) and input drive (B_bar * x). The state decays exponentially toward zero while accumulating input. Structurally identical to a population of LIF neurons without spikes.
  - Mapping gap: Mamba3 uses negative real eigenvalues (stable exponential decay). LIF neurons spike and reset. Mamba3 has no threshold, no spike, no reset.

- **Complex rotation: angle = rope_freq * t**
  - Gamma oscillations: SO(2) rotation at learned frequencies, producing oscillatory state dynamics. 16 dimension pairs, each oscillating at its own frequency.
  - Mapping gap: biological gamma arises from network dynamics (E-I interaction). Mamba3's rope_freq is a static parameter, not an emergent network property. Biological gamma frequency modulates with attention (+3 Hz). rope_freq does not change during inference.
  - The rotation likely serves as positional encoding (RoPE for the recurrent state), not as an oscillatory mechanism.

- **Data-dependent discretization: dt = softplus(dt_proj(x) + dt_bias)**
  - Partial analog to norepinephrine adaptive gain: dt modulates how strongly the input drives the state. High dt = strong input drive, fast state update. Low dt = weak input drive, slow state update.
  - Mapping gap: dt affects the DECAY rate (A_bar) but NOT the rotation frequency (rope_freq). Biological neuromodulation affects both.

## SwiGLU (src/layers/swiglu.py)

SwiGLU is the feedforward layer. Applied at every layer (all 24).

Biological analogs:

- **Multiplicative gating: silu(W_gate(x)) * W_up(x)**
  - Dendritic computation: multiplicative interaction between two independent transformations of the input. Analogous to NMDA receptor coincidence detection (requires both presynaptic glutamate AND postsynaptic depolarization).
  - Mapping gap (HIGH): SwiGLU uses ONE gate for the entire hidden dimension. Pyramidal neurons have ~30-50 independent dendritic branches, each performing independent nonlinear gating. SwiGLU is a single-branch neuron; biology uses many branches.
  - Mapping gap (HIGH): both paths receive the SAME input x. Biological dendritic gating involves DIFFERENT sources (feedforward via basal dendrites, feedback via apical tuft). SwiGLU is self-gating, not cross-source gating.
  - Mapping gap (MODERATE): silu is smooth and monotonic. Dendritic spikes are regenerative, threshold-crossing events.

- **Hidden dimension expansion: d_model -> ~2.75 * d_model -> d_model**
  - Two-layer neuron model: Poirazi et al. (2003) showed a pyramidal neuron is computationally equivalent to a two-layer network with ~30-50 hidden units. SwiGLU's expand-process-compress pattern is the same architecture.
  - Mapping gap: the expansion ratios differ by orders of magnitude (2.75x vs ~1000x). Biology creates INDEPENDENT subunits; SwiGLU creates one shared space.

- **GP self-interaction: out + gp_proj(geometric_product(W_left(x), W_right(x)))**
  - Grid cells: PGA can express rotations, translations, reflections (the operations underlying spatial computation). G(3,0,1) is a superset of SE(2), the group that grid cells and head direction cells operate in.
  - Mapping gap: connection is WEAK to NONEXISTENT at implementation level. PGA is a self-interaction (both inputs from same x), not path integration (accumulating velocity over time). No temporal integration, no velocity signal, no periodicity.
  - The GP likely functions as a structured bilinear mixer, not a spatial computation module. The GP-vs-random-bilinear control was later run and did not support a strong spatial-mechanism claim at random initialization.

## Ternary Spikes (src/spikes/ternary_spike.py)

Ternary quantization of activations to {-1, 0, +1}. Applied to KDA K and V projections (18 layers).

Biological analogs:

- **Quantization: sign(x) * [|x| > threshold]**
  - Sparse coding: biological neurons produce discrete spikes from continuous input. Ternary spikes implement the same transform: continuous -> discrete. The zero state means "this neuron did not fire."
  - Mapping gap: biological spikes are binary (fire or not), not ternary. The +1/-1 distinction maps loosely to excitation/inhibition but individual neurons are either excitatory or inhibitory (Dale's law), never both.

- **Adaptive threshold: threshold = alpha * mean(|x|)**
  - Divisive normalization: the threshold adapts to the population mean activation, a crude form of gain control. Strong inputs in a quiet layer fire; the same inputs in a highly active layer may not.
  - Mapping gap (HIGH): one scalar threshold for ALL neurons. Biological divisive normalization computes per-neuron or per-pool normalization. The global threshold cannot distinguish informative features from noise.

- **~41% firing rate**
  - NOT energy-efficient coding: cortical neurons fire at 1-5%. The optimal rate for bits-per-joule is ~6%. Todorov's 41% is 4-40x denser than biology. The gap exists because the STE gradient requires active neurons for gradient flow.

- **Straight-through estimator (STE) backward pass**
  - No biological analog. STE is an engineering workaround: treat the non-differentiable quantization as identity during backpropagation. Biology uses local plasticity rules (Hebbian, STDP), not backpropagated gradients.

## ATMN (src/spikes/atmn_spike.py)

ATMN adds temporal dynamics to ternary spikes via membrane potential integration. Implemented, not yet validated at scale.

Biological analogs:

- **Membrane potential: h_t = x_t + (1/tau) * u_{t-1}**
  - Leaky integrate-and-fire: temporal integration of input over time. The membrane potential accumulates input and carries history forward.
  - Mapping gap (HIGH): no leak term. The LIF's -g_L * (V - V_rest) drives voltage back toward rest. ATMN's (1/tau) scales the carried-over potential but does not decay it toward a resting value.

- **Per-neuron threshold: V_th = exp(threshold_log)**
  - Loosely analogous to the adaptive exponential integrate-and-fire model: each neuron has its own firing threshold, allowing feature-specific sparsity.
  - Mapping gap (LOW): biological thresholds are modulated dynamically by Na+ channel inactivation and neuromodulation. ATMN's threshold is static after training.

- **Reset by subtraction: u_t = h_t - spikes * V_th**
  - Leaky integrate-and-fire: after firing, the membrane potential is reduced by the threshold value. Residual potential carries over. Biologically plausible.
  - Mapping gap: none significant. This is a faithful implementation of soft reset.

- **Training reset: membrane_potential = zeros at each forward pass**
  - NO biological analog. This is the most significant departure. Biological neurons never "reset between batches." The batch reset eliminates temporal dynamics during training, making ATMN a per-token threshold function rather than a temporal integrator.

## Geometric Product (src/algebra/geometric_product.py)

G(3,0,1) projective geometric algebra self-interaction. Additive after SwiGLU down projection.

Biological analogs:

- **PGA algebraic structure**
  - Grid cells and path integration: the spatial navigation system operates in SE(2) (translations + rotations). PGA encodes E(3), a superset. Grid cells implement representations of the translation group (Gao et al. 2021). The mathematical language is shared.
  - Mapping gap (FUNDAMENTAL): PGA is a self-interaction (f(x, x)), not path integration (accumulating velocity signals over time). The periodicity that defines grid cells (hexagonal firing patterns) is absent from PGA.

- **Structured bilinear mixing (16 components via Cayley table)**
  - No specific biological analog. The GP computes 16 specific bilinear combinations of its inputs with signs and targets dictated by the algebra. It occupies a middle ground between a random bilinear layer (4096 free parameters) and a rigid geometric transformation.
  - The network can use the algebraic structure however gradient descent finds useful. Validated: spike MI all-time high 1.311 with GP active. NOT validated: whether geometric structure specifically (vs generic bilinear) drives the improvement.

## RMSNorm (src/model/decode_head.py)

Root mean square normalization before every sublayer.

Biological analogs:

- **y_i = gamma_i * x_i / sqrt(mean(x^2) + eps)**
  - Divisive normalization: division by pooled population activity. RMSNorm is a special case: uniform pool weights (w_j = 1/d for all j), no power law (linear numerator), no semi-saturation constant (eps << 1).
  - Mapping gap: RMSNorm pools ALL dimensions equally. Biological divisive normalization uses structured pools (nearby neurons contribute more). RMSNorm is global and uniform; biology is local and weighted.

## Residual Stream

The sole inter-layer communication channel. x_{l+1} = x_l + f_l(RMSNorm(x_l)).

Biological analogs:

- **Shared medium accessible by all layers**
  - Global workspace theory: the residual stream is functionally a shared workspace. It is the only communication channel. Information from any layer is available to all subsequent layers.
  - Mapping gap (CRITICAL): no ignition threshold. GWT's defining feature is a nonlinear, all-or-none broadcast event. The residual stream always broadcasts everything.
  - Mapping gap (HIGH): no capacity limit. GWT holds ~4 items. The residual stream carries the full d_model vector.
  - Mapping gap (HIGH): no write selectivity. Every layer writes unconditionally.
  - Verdict: the residual stream is a SHARED BUS, not a GLOBAL WORKSPACE.

## RoPE (Rotary Position Embedding)

Rotation applied to Q and K in KDA and MLA layers.

Biological analogs:

- **Rotation by position-dependent angle**
  - Theta oscillations: RoPE produces periodic modulation of query-key similarity as a function of relative position. This creates a pattern where nearby tokens interact differently than distant tokens, loosely analogous to theta phase precession (earlier items in a sequence occupy earlier phases).
  - Mapping gap: RoPE is a fixed, deterministic function of position. Theta oscillations are dynamic, modulated by behavior, and emerge from network dynamics. The periods in RoPE are geometric (2*pi*i/d); theta is a single ~7 Hz rhythm.

## Layer Schedule (KDA, KDA, KDA, Mamba3, KDA, KDA, KDA, MLA) x 3

The architecture's 3:1 ratio (75% KDA, 25% other).

Biological analogs:

- **Mixed architecture with dominant recurrent processing**
  - Cortical column: cortex has a canonical microcircuit repeated across areas, with different neuron types at fixed proportions. The 3:1 ratio echoes this structural regularity.
  - Mapping gap: the ratio comes from ML engineering benchmarks, not neuroscience. Cortical layers process in PARALLEL (all layers simultaneously active). Todorov layers process in SERIAL (one after another). Cortical layers have HETEROGENEOUS connectivity. Todorov layers all see the same residual stream.

## Spike Health Metrics (src/spikes/spiking_brain.py)

MI, CKA, and firing rate monitoring.

Biological analogs:

- **Mutual information (MI > 0.1)**
  - Efficient coding: MI measures how much input information survives quantization. Equivalent to measuring the rate-distortion tradeoff of the spike code. MI of 1.168 at 267M scale = 74% channel utilization.

- **CKA (> 0.3)**
  - Population coding: CKA measures whether the geometry (similarity structure) of representations is preserved through quantization. High CKA = a linear decoder trained on pre-spike data also works on post-spike data.

- **Firing rate (20-60%)**
  - Energy-efficient coding: the fraction of active neurons. Biology targets 1-5% for energy efficiency. Todorov targets 20-60% for gradient health. The 41% operating point maximizes MI while maintaining stable STE gradients.

- **Missing metrics**
  - Noise correlations (not measured): could limit information capacity below per-dimension MI estimate.
  - Inter-neuron redundancy (not measured): high per-dimension MI with redundant neurons = inefficient code.

## The CRBR Framework

The unified theory: every layer instantiates z_t = Q(R(B(C(x_t), C(h_{t-1})))).

Biological analogs:

- **C (compression)**: ternary spikes, latent projection, gating -- sparse coding, efficient coding. The brain compresses information through sparse activation.
- **B (bilinear interaction)**: outer product, dot product, geometric product -- Hebbian learning, pattern completion. The brain stores and retrieves via bilinear operations.
- **R (rotational structure)**: RoPE, complex dynamics, PGA rotors -- grid cells, theta oscillations. Periodic structure encodes position and temporal relationships.
- **Q (output quantization)**: ternary spike output -- sparse coding. Discrete output enforces information bottleneck.

The framework unifies the components but no single biological system implements all four stages in this order. The composition C-B-R-Q is an engineering abstraction, not a biological circuit.
