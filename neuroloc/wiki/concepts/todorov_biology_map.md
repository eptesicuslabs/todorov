# todorov biology map

master reference mapping every todorov component to its biological analog. for each component: what it does in todorov, what it maps to in biology, what the mapping gets wrong, and where to find the full analysis.

## KDA (src/layers/kda.py)

KDA is a delta-rule recurrent layer that maintains a matrix-valued associative memory. 18 of 24 layers (75%) are KDA.

biological analogs:

- **state update: S_t = diag(alpha) * S_{t-1} + beta_t * k_t * v_t^T**
  -> [[hebbian_learning]]: the outer product k_t * v_t^T is the classical Hebbian association rule. this is the most direct biological correspondence in the entire architecture
  -> [[short_term_plasticity]]: alpha decay resembles synaptic depression (activity-dependent decay of synaptic efficacy)
  -> NOT [[stdp]]: no timing dependence. k and v come from the SAME timestep. STDP requires comparing activity at DIFFERENT times
  -> see [[plasticity_to_kda_delta_rule]] for the full adversarial analysis

- **alpha (channel-wise forgetting rate): alpha = sigmoid(alpha_log)**
  -> [[homeostatic_plasticity]]: prevents state saturation, maintains bounded activity. different channels decay at different rates, allowing a hierarchy of memory timescales
  -> NOT [[neuromodulatory_framework]]: alpha is per-channel and FIXED after training. biological neuromodulation is a small number of global, dynamic signals that change moment-to-moment
  -> see [[neuromodulation_to_learning_and_gating]] for the full analysis

- **beta (data-dependent write gate): beta_t = sigmoid(beta_proj(x_t))**
  -> [[acetylcholine_system]]: input-dependent gating of memory encoding. high beta = encoding mode (write to state), low beta = preservation mode (read from state). the closest biological analog to any parameter in todorov
  -> mapping gap: beta sees only the current token x_t. biological ACh is driven by global signals (novelty, uncertainty, task demands). beta has no access to global state
  -> see [[neuromodulation_to_learning_and_gating]] and [[biological_attention_to_mla]]

- **readout: o_t = q_t^T * S_t**
  -> [[pattern_completion]]: content-addressable retrieval from associative memory, identical in structure to Hopfield network recall
  -> mapping gap: KDA readout is LINEAR (one matrix-vector multiply). Hopfield recall uses iterative convergence with attractor dynamics. KDA has no error correction
  -> see [[memory_systems_to_kda_mla]]

## MLA (src/layers/mla.py)

MLA performs softmax attention over compressed per-token representations. 6 of 24 layers (25%).

biological analogs:

- **softmax attention: softmax(Q @ K^T / sqrt(d)) @ V**
  -> [[pattern_completion]]: Ramsauer et al. (2021) proved softmax attention is the update rule of a modern Hopfield network with exponential capacity 2^{d/2}
  -> NOT [[selective_attention]]: MLA computes information retrieval (finding relevant past tokens). biological attention computes resource allocation (directing processing to relevant stimuli). different problems
  -> see [[biological_attention_to_mla]]

- **KV compression: c_kv = kv_down_proj(x), d_model -> d_c = 128**
  -> [[hippocampal_memory]]: hippocampal indexing theory (Teyler and DiScenna 1986) proposes that the hippocampus stores compressed pointers to cortical patterns, not full patterns. MLA's compressed latent c_kv is a compressed index to the full token representation
  -> mapping gap: hippocampal indices are stored through rapid synaptic modification (LTP). MLA stores c_kv in a passive cache with no learning
  -> see [[memory_systems_to_kda_mla]]

- **full cache (no forgetting)**
  -> NOT [[memory_consolidation]]: MLA retains every token until the context limit. no decay, no consolidation, no transfer to a slower system. the absence of forgetting is an engineering advantage, not a biological principle

- **25% allocation (6/24 layers)**
  -> NOT biologically derived. the 3:1 ratio comes from ML benchmarks (Kimi, Qwen3, OLMo independently converged on ~75/25). the resemblance to biological attention's sparsity is coincidental
  -> see [[biological_attention_to_mla]] and [[cortical_microcircuit_to_layer_schedule]]

## Mamba3 (src/layers/mamba3.py)

Mamba3 is a state-space model with complex rotation and data-dependent discretization. 3 of 24 layers (at positions 4, 12, 20).

biological analogs:

- **state evolution: h_t = A_bar_t * h_{t-1} + B_bar_t * B_t * x_t**
  -> [[leaky_integrate_and_fire]]: discretized linear ODE with decay (A_bar) and input drive (B_bar * x). the state decays exponentially toward zero while accumulating input. structurally identical to a population of LIF neurons without spikes
  -> mapping gap: Mamba3 uses negative real eigenvalues (stable exponential decay). LIF neurons spike and reset. Mamba3 has no threshold, no spike, no reset
  -> see [[oscillations_to_mamba3_rotation]]

- **complex rotation: angle = rope_freq * t**
  -> [[gamma_oscillations]]: SO(2) rotation at learned frequencies, producing oscillatory state dynamics. 16 dimension pairs, each oscillating at its own frequency
  -> mapping gap: biological gamma arises from network dynamics (E-I interaction). Mamba3's rope_freq is a static parameter, not an emergent network property. biological gamma frequency modulates with attention (+3 Hz). rope_freq does not change during inference
  -> the rotation likely serves as positional encoding (RoPE for the recurrent state), not as an oscillatory mechanism
  -> see [[oscillations_to_mamba3_rotation]]

- **data-dependent discretization: dt = softplus(dt_proj(x) + dt_bias)**
  -> partial analog to [[norepinephrine_system]] adaptive gain: dt modulates how strongly the input drives the state. high dt = strong input drive, fast state update. low dt = weak input drive, slow state update
  -> mapping gap: dt affects the DECAY rate (A_bar) but NOT the rotation frequency (rope_freq). biological neuromodulation affects both
  -> see [[oscillations_to_mamba3_rotation]]

## SwiGLU (src/layers/swiglu.py)

SwiGLU is the feedforward layer. applied at every layer (all 24).

biological analogs:

- **multiplicative gating: silu(W_gate(x)) * W_up(x)**
  -> [[dendritic_computation]]: multiplicative interaction between two independent transformations of the input. analogous to NMDA receptor coincidence detection (requires both presynaptic glutamate AND postsynaptic depolarization)
  -> mapping gap (HIGH): SwiGLU uses ONE gate for the entire hidden dimension. pyramidal neurons have ~30-50 independent dendritic branches, each performing independent nonlinear gating. SwiGLU is a single-branch neuron; biology uses many branches
  -> mapping gap (HIGH): both paths receive the SAME input x. biological dendritic gating involves DIFFERENT sources (feedforward via basal dendrites, feedback via apical tuft). SwiGLU is self-gating, not cross-source gating
  -> mapping gap (MODERATE): silu is smooth and monotonic. dendritic spikes are regenerative, threshold-crossing events
  -> see [[dendritic_computation_to_swiglu]]

- **hidden dimension expansion: d_model -> ~2.75 * d_model -> d_model**
  -> [[two_layer_neuron]]: Poirazi et al. (2003) showed a pyramidal neuron is computationally equivalent to a two-layer network with ~30-50 hidden units. SwiGLU's expand-process-compress pattern is the same architecture
  -> mapping gap: the expansion ratios differ by orders of magnitude (2.75x vs ~1000x). biology creates INDEPENDENT subunits; SwiGLU creates one shared space

- **GP self-interaction: out + gp_proj(geometric_product(W_left(x), W_right(x)))**
  -> [[grid_cells]]: PGA can express rotations, translations, reflections (the operations underlying spatial computation). G(3,0,1) is a superset of SE(2), the group that grid cells and head direction cells operate in
  -> mapping gap: connection is WEAK to NONEXISTENT at implementation level. PGA is a self-interaction (both inputs from same x), not path integration (accumulating velocity over time). no temporal integration, no velocity signal, no periodicity
  -> the GP likely functions as a structured bilinear mixer, not a spatial computation module. the critical experiment (GP vs random bilinear control) has not been run
  -> see [[spatial_computation_to_pga]]

## ternary spikes (src/spikes/ternary_spike.py)

ternary quantization of activations to {-1, 0, +1}. applied to KDA K and V projections (18 layers).

biological analogs:

- **quantization: sign(x) * [|x| > threshold]**
  -> [[sparse_coding]]: biological neurons produce discrete spikes from continuous input. ternary spikes implement the same transform: continuous -> discrete. the zero state means "this neuron did not fire"
  -> mapping gap: biological spikes are binary (fire or not), not ternary. the +1/-1 distinction maps loosely to excitation/inhibition but individual neurons are either excitatory or inhibitory (Dale's law), never both
  -> see [[sparse_coding_to_ternary_spikes]]

- **adaptive threshold: threshold = alpha * mean(|x|)**
  -> [[divisive_normalization]]: the threshold adapts to the population mean activation, a crude form of gain control. strong inputs in a quiet layer fire; the same inputs in a highly active layer may not
  -> mapping gap (HIGH): one scalar threshold for ALL neurons. biological divisive normalization computes per-neuron or per-pool normalization. the global threshold cannot distinguish informative features from noise
  -> see [[lateral_inhibition_to_adaptive_threshold]]

- **~41% firing rate**
  -> NOT [[energy_efficient_coding]]: cortical neurons fire at 1-5%. the optimal rate for bits-per-joule is ~6%. todorov's 41% is 4-40x denser than biology. the gap exists because the STE gradient requires active neurons for gradient flow
  -> see [[sparse_coding_to_ternary_spikes]] and [[energy_efficiency_to_ternary_spikes]]

- **straight-through estimator (STE) backward pass**
  -> no biological analog. STE is an engineering workaround: treat the non-differentiable quantization as identity during backpropagation. biology uses local plasticity rules (Hebbian, STDP), not backpropagated gradients

## ATMN (src/spikes/atmn_spike.py)

ATMN adds temporal dynamics to ternary spikes via membrane potential integration. implemented, not yet validated at scale.

biological analogs:

- **membrane potential: h_t = x_t + (1/tau) * u_{t-1}**
  -> [[leaky_integrate_and_fire]]: temporal integration of input over time. the membrane potential accumulates input and carries history forward
  -> mapping gap (HIGH): no leak term. the LIF's -g_L * (V - V_rest) drives voltage back toward rest. ATMN's (1/tau) scales the carried-over potential but does not decay it toward a resting value
  -> see [[neuron_models_to_atmn]]

- **per-neuron threshold: V_th = exp(threshold_log)**
  -> loosely [[adaptive_exponential]]: each neuron has its own firing threshold, allowing feature-specific sparsity
  -> mapping gap (LOW): biological thresholds are modulated dynamically by Na+ channel inactivation and neuromodulation. ATMN's threshold is static after training

- **reset by subtraction: u_t = h_t - spikes * V_th**
  -> [[leaky_integrate_and_fire]]: after firing, the membrane potential is reduced by the threshold value. residual potential carries over. biologically plausible
  -> mapping gap: none significant. this is a faithful implementation of soft reset

- **training reset: membrane_potential = zeros at each forward pass**
  -> NO biological analog. this is the most significant departure. biological neurons never "reset between batches." the batch reset eliminates temporal dynamics during training, making ATMN a per-token threshold function rather than a temporal integrator
  -> see [[neuron_models_to_atmn]]

## geometric product (src/algebra/geometric_product.py)

G(3,0,1) projective geometric algebra self-interaction. additive after SwiGLU down projection.

biological analogs:

- **PGA algebraic structure**
  -> [[grid_cells]] and [[path_integration]]: the spatial navigation system operates in SE(2) (translations + rotations). PGA encodes E(3), a superset. grid cells implement representations of the translation group (Gao et al. 2021). the mathematical language is shared
  -> mapping gap (FUNDAMENTAL): PGA is a self-interaction (f(x, x)), not path integration (accumulating velocity signals over time). the periodicity that defines grid cells (hexagonal firing patterns) is absent from PGA
  -> see [[spatial_computation_to_pga]] and [[pga_vs_grid_cells]]

- **structured bilinear mixing (16 components via Cayley table)**
  -> no specific biological analog. the GP computes 16 specific bilinear combinations of its inputs with signs and targets dictated by the algebra. it occupies a middle ground between a random bilinear layer (4096 free parameters) and a rigid geometric transformation
  -> the network can use the algebraic structure however gradient descent finds useful. validated: spike MI all-time high 1.311 with GP active. NOT validated: whether geometric structure specifically (vs generic bilinear) drives the improvement

## RMSNorm (src/model/decode_head.py)

root mean square normalization before every sublayer.

biological analogs:

- **y_i = gamma_i * x_i / sqrt(mean(x^2) + eps)**
  -> [[divisive_normalization]]: division by pooled population activity. RMSNorm is a special case: uniform pool weights (w_j = 1/d for all j), no power law (linear numerator), no semi-saturation constant (eps << 1)
  -> mapping gap: RMSNorm pools ALL dimensions equally. biological divisive normalization uses structured pools (nearby neurons contribute more). RMSNorm is global and uniform; biology is local and weighted
  -> see [[normalization_layernorm_vs_divisive]]

## residual stream

the sole inter-layer communication channel. x_{l+1} = x_l + f_l(RMSNorm(x_l)).

biological analogs:

- **shared medium accessible by all layers**
  -> [[global_workspace_theory]]: the residual stream is functionally a shared workspace. it is the only communication channel. information from any layer is available to all subsequent layers
  -> mapping gap (CRITICAL): no ignition threshold. GWT's defining feature is a nonlinear, all-or-none broadcast event. the residual stream always broadcasts everything
  -> mapping gap (HIGH): no capacity limit. GWT holds ~4 items. the residual stream carries the full d_model vector
  -> mapping gap (HIGH): no write selectivity. every layer writes unconditionally
  -> verdict: the residual stream is a SHARED BUS, not a GLOBAL WORKSPACE
  -> see [[global_workspace_to_residual_stream]] and [[gwt_vs_transformer]]

## RoPE (rotary position embedding)

rotation applied to Q and K in KDA and MLA layers.

biological analogs:

- **rotation by position-dependent angle**
  -> [[theta_oscillations]]: RoPE produces periodic modulation of query-key similarity as a function of relative position. this creates a pattern where nearby tokens interact differently than distant tokens, loosely analogous to theta phase precession (earlier items in a sequence occupy earlier phases)
  -> mapping gap: RoPE is a fixed, deterministic function of position. theta oscillations are dynamic, modulated by behavior, and emerge from network dynamics. the periods in RoPE are geometric (2*pi*i/d); theta is a single ~7 Hz rhythm
  -> see [[oscillations_to_mamba3_rotation]] for the related analysis of Mamba3 rotation

## layer schedule (KDA, KDA, KDA, Mamba3, KDA, KDA, KDA, MLA) x 3

the architecture's 3:1 ratio (75% KDA, 25% other).

biological analogs:

- **mixed architecture with dominant recurrent processing**
  -> [[cortical_column]]: cortex has a canonical microcircuit repeated across areas, with different neuron types at fixed proportions. the 3:1 ratio echoes this structural regularity
  -> mapping gap: the ratio comes from ML engineering benchmarks, not neuroscience. cortical layers process in PARALLEL (all layers simultaneously active). todorov layers process in SERIAL (one after another). cortical layers have HETEROGENEOUS connectivity. todorov layers all see the same residual stream
  -> see [[cortical_microcircuit_to_layer_schedule]] and [[cortical_layers_vs_todorov_layers]]

## spike health metrics (src/spikes/spiking_brain.py)

MI, CKA, and firing rate monitoring.

biological analogs:

- **mutual information (MI > 0.1)**
  -> [[efficient_coding]]: MI measures how much input information survives quantization. equivalent to measuring the rate-distortion tradeoff of the spike code. MI of 1.168 at 267M scale = 74% channel utilization
  -> see [[population_coding_to_spike_health]]

- **CKA (> 0.3)**
  -> [[population_coding]]: CKA measures whether the geometry (similarity structure) of representations is preserved through quantization. high CKA = a linear decoder trained on pre-spike data also works on post-spike data
  -> see [[population_coding_to_spike_health]]

- **firing rate (20-60%)**
  -> [[energy_efficient_coding]]: the fraction of active neurons. biology targets 1-5% for energy efficiency. todorov targets 20-60% for gradient health. the 41% operating point maximizes MI while maintaining stable STE gradients
  -> see [[sparse_coding_to_ternary_spikes]] and [[population_coding_to_spike_health]]

- **missing metrics**
  -> noise correlations (not measured): could limit information capacity below per-dimension MI estimate
  -> inter-neuron redundancy (not measured): high per-dimension MI with redundant neurons = inefficient code
  -> see [[population_coding_to_spike_health]] for the full gap analysis

## the CRBR framework

the unified theory: every layer instantiates z_t = Q(R(B(C(x_t), C(h_{t-1})))).

biological analogs:

- **C (compression)**: ternary spikes, latent projection, gating -> [[sparse_coding]], [[efficient_coding]]. the brain compresses information through sparse activation
- **B (bilinear interaction)**: outer product, dot product, geometric product -> [[hebbian_learning]], [[pattern_completion]]. the brain stores and retrieves via bilinear operations
- **R (rotational structure)**: RoPE, complex dynamics, PGA rotors -> [[grid_cells]], [[theta_oscillations]]. periodic structure encodes position and temporal relationships
- **Q (output quantization)**: ternary spike output -> [[sparse_coding]]. discrete output enforces information bottleneck

the framework unifies the components but no single biological system implements all four stages in this order. the composition C-B-R-Q is an engineering abstraction, not a biological circuit.
