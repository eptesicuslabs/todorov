# bridge: oscillatory dynamics to Mamba3 complex rotation

status: current (as of 2026-04-16).

## the biological mechanism

neural oscillations ([[gamma_oscillations]], [[theta_oscillations]], [[neural_synchrony]]) provide temporal coordination across cortical networks. the computationally critical properties are:

1. **network-emergent frequencies:** gamma (30-100 Hz) arises from E-I interaction (PING/ING), not from individual neuron parameters. the frequency depends on GABAA kinetics, connection strength, excitatory drive, and neuromodulation. this means the frequency is a FUNCTION of network state, not a fixed parameter
2. **cross-frequency coupling:** theta phase modulates gamma amplitude (PAC), creating nested temporal frames. ~5-8 gamma cycles per theta cycle encode sequential items
3. **communication gating:** coherence between populations determines effective connectivity (CTC). same physical synapses transmit or block depending on phase alignment
4. **dynamic frequency modulation:** gamma frequency increases ~3 Hz with attention; theta frequency tracks running speed. frequencies adapt in real time to task demands
5. **phase coding:** information encoded in spike timing relative to oscillatory phase, providing ~54% additional information beyond firing rate

see [[oscillations_vs_recurrence]] for comparison with todorov's recurrent dynamics.

## the current todorov implementation

Mamba3 (src/layers/mamba3.py):

state evolution:

    h_t = A_bar_t * h_{t-1} + B_bar_t * B_t * x_t

where A_bar_t and B_bar_t are computed by trapezoidal (Tustin) discretization:

    A_bar = (1 + dt*A/2) / (1 - dt*A/2)
    B_bar = dt / (1 - dt*A/2)

with A = -exp(A_log), A_log initialized in [-4, -1] (negative real eigenvalues, stable decay). dt is data-dependent: dt = softplus(dt_proj(x) + dt_bias).

after state update, data-dependent rotation:

    angle = rope_freq * t
    rotated_real = real * cos(angle) - imag * sin(angle)
    rotated_imag = real * sin(angle) + imag * cos(angle)

where rope_freq is an nn.Parameter of shape (d_state//2,), initialized to randn * 0.01 (small random initial frequencies). the rotation is applied to the state at every timestep after the state update.

the state h_t has shape (batch, d_inner, d_state) where d_inner = d_model * expand (e.g., 768 for d_model=384, expand=2) and d_state = 32.

readout:

    y_t = sum(h_t * C_t, dim=-1)

where C_t = C_proj(x_t) is a data-dependent readout projection.

### what this rotation IS

mathematically, the rotation is a SO(2) transformation applied to pairs of state dimensions, parameterized by angle = rope_freq * t. this is identical to:

- a 2D harmonic oscillator with angular frequency omega = rope_freq
- a complex multiplication by exp(i * rope_freq * t)
- RoPE (rotary position embedding) applied to the recurrent state

each of the d_state//2 = 16 dimension pairs oscillates at its own learned frequency. the frequencies are STATIC after training -- they do not change with input.

### analogies to biological oscillations

the analogy holds on a narrow axis:

    property                    biology                      Mamba3
    periodic dynamics           yes (E-I oscillation)        yes (complex rotation)
    per-unit frequency          yes (local circuit)          yes (per-dimension rope_freq)
    phase-space trajectory      yes (limit cycle)            yes (rotation in 2D plane)
    continuous evolution        yes (ongoing oscillation)    yes (rotation at every timestep)

### where the analogy breaks

**1. fixed vs dynamic frequency.**

biological gamma frequency modulates with attention (+3 Hz), excitatory drive, neuromodulatory state, and stimulus properties. Mamba3's rope_freq is a static nn.Parameter -- once learned, it does not change during inference.

the dt_proj provides data-dependent modulation of the discretization timestep, which affects the DECAY rate (A_bar) but NOT the rotation frequency (rope_freq). the rotation angle is always rope_freq * t, regardless of input. this is a hard limitation: the rotation cannot accelerate or slow based on what the model is processing.

proposed modification: make rope_freq data-dependent.

    freq_t = rope_freq_base + freq_proj(x_t)

where freq_proj is a linear projection from d_model to d_state//2. this allows the rotation frequency to modulate with input content, analogous to attention-dependent gamma frequency shifts. cost: one additional linear projection per Mamba3 layer (d_model * d_state//2 parameters).

risk: HIGH. data-dependent frequencies could destabilize the rotation dynamics. if freq_t fluctuates wildly across tokens, the state trajectory becomes chaotic rather than oscillatory. biological oscillations are stabilized by the E-I circuit dynamics -- PING/ING have attractors that prevent frequency from diverging. a learned linear projection has no such stability guarantee.

mitigation: constrain freq_t to a narrow range around rope_freq_base (e.g., freq_t = rope_freq_base * (1 + 0.1 * tanh(freq_proj(x_t)))). this limits modulation to +/-10% of the base frequency, preventing instability while allowing task-dependent frequency shifts.

**2. no cross-frequency coupling.**

biological theta-gamma coupling creates nested temporal frames: each theta cycle contains ~5-8 gamma cycles, with different items represented in different gamma subcycles. the theta phase modulates gamma amplitude, implementing a temporal multiplexing scheme.

Mamba3 has 16 independently rotating dimension pairs. there is no coupling between them. no dimension's oscillation modulates another dimension's amplitude. the rotation is applied element-wise: each pair rotates at its own frequency with no interaction.

proposed modification: add phase-amplitude coupling between Mamba3 state dimensions.

    slow_phase = rope_freq_slow * t
    gamma_amplitude = sigmoid(coupling_weight * cos(slow_phase))
    state_fast = state_fast * gamma_amplitude

where rope_freq_slow selects a subset of state dimensions as "theta-like" (slow oscillators) and coupling_weight determines how strongly the slow oscillation modulates the fast. this would implement PAC within the Mamba3 state.

risk: HIGH. this adds complexity to the state dynamics without clear evidence that PAC improves language modeling. biological PAC evolved for multi-item working memory and spatial navigation -- tasks with inherent sequential structure. language does have sequential structure, but the theta-gamma capacity argument (7 items) maps poorly to language (sentences are much longer than 7 items).

recommendation: do NOT implement. the biological motivation is weak for language modeling, and the implementation introduces multiple hyperparameters (which dimensions are slow vs fast, coupling strength, coupling function) with no principled way to set them.

**3. only 3/24 layers have rotation.**

if oscillatory dynamics were computationally essential for temporal coordination, all layers should have them. the architecture assigns rotation to only 12.5% of layers (Mamba3 at positions 4, 12, 20). the majority of temporal processing is handled by KDA, which has no oscillatory dynamics whatsoever.

this asymmetry suggests that Mamba3 rotation serves a COMPLEMENTARY function (likely positional encoding for the recurrent state) rather than a FUNDAMENTAL organizing principle. if rotation were analogous to gamma oscillations -- which are ubiquitous in cortex and present in every cortical area -- it should appear in every layer.

counter-argument: different cortical areas have different oscillatory profiles (V1 has strong gamma, hippocampus has strong theta, prefrontal has both). the heterogeneous deployment of rotation across layers could reflect functional specialization. but this analogy predicts that KDA layers should have their OWN oscillatory mechanism (perhaps theta-like), which they do not.

**4. no inter-layer oscillatory coordination.**

biological oscillations coordinate between brain areas through coherence. V1 and V4 synchronize their gamma oscillations to enable feedforward communication. hippocampus and prefrontal cortex synchronize at theta during memory retrieval.

todorov's three Mamba3 layers (positions 4, 12, 20) each rotate their state independently. there is no mechanism for layer 4's rotation to become coherent with layer 12's rotation. no coherence-based gating occurs between layers. the residual stream provides always-on communication between all layers.

proposed modification: shared or coupled rotation frequencies across Mamba3 layers.

    freq_global = mean(rope_freq_layer4, rope_freq_layer12, rope_freq_layer20)
    freq_layer_i = freq_global + delta_freq_layer_i

where delta_freq represents per-layer frequency offsets. this creates a "global oscillation" that all Mamba3 layers share, with per-layer detuning.

risk: LOW (just parameter tying, minimal implementation complexity). but expected benefit is also LOW: without a gating mechanism that uses the phase relationship between layers, shared frequencies provide no computational advantage. the layers still operate serially and cannot exploit phase alignment.

recommendation: do NOT implement. the benefit requires a gating mechanism that does not exist and cannot be easily added to the serial architecture.

**5. the rotation likely serves positional encoding, not oscillatory dynamics.**

the strongest alternative interpretation of Mamba3 rotation: it is RoPE for the recurrent state. RoPE (Su et al. 2021) uses exactly the same mathematical operation (rotation by angle proportional to position) to encode relative position in attention mechanisms. KDA and MLA already use RoPE for their attention computations. Mamba3's rope_freq may simply be the SSM counterpart: encoding "how many timesteps ago was this information stored" in the state's phase.

evidence for this interpretation:
- rope_freq is initialized to small random values (randn * 0.01), consistent with RoPE initialization
- the rotation is applied at every timestep with angle = freq * t, exactly the RoPE formula
- the rotation enables position-dependent state retrieval (C_t's interaction with rotated state depends on relative position)

evidence against this interpretation:
- RoPE in attention operates on query and key vectors, not on state. the semantics are different
- the rotation is applied AFTER the state update, modifying the stored representation, not just the retrieval mechanism
- if it were purely positional, there would be no reason for the frequencies to be learned -- fixed geometric frequencies (as in standard RoPE) would suffice

the truth is probably both: the rotation serves as positional encoding for the recurrent state AND produces oscillatory dynamics as a mathematical consequence. but the oscillatory dynamics are incidental to the positional function, not the primary purpose.

## the key adversarial question

**would adding periodic modulation to spike thresholds create gamma-like temporal coding in the spike train?**

idea: modulate the ternary spike threshold with a periodic function:

    V_th(t) = V_th_base * (1 + epsilon * sin(omega * t + phi))

where omega is a learned frequency, epsilon is a modulation depth, and phi is a phase offset. this would create periodic windows of high excitability (low threshold, more spikes) and low excitability (high threshold, fewer spikes), analogous to the gamma cycle's alternation between excitatory and inhibitory windows.

### strongest argument FOR

1. periodic threshold modulation would impose temporal structure on the spike train. currently, ternary spikes are applied independently per token with no temporal dynamics (especially during training, where ATMN resets each batch). modulated thresholds would create temporal patterns: tokens at certain positions would spike more easily, creating a periodic "attentional" window
2. the modulation frequency is learnable: the network could discover task-relevant rhythms. if language has periodic structure (e.g., phrase boundaries, syntactic cycles), the learned frequency could capture it
3. minimal parameter cost: one frequency (omega), one depth (epsilon), one phase (phi) per spike module. total: ~3 * num_spike_modules parameters
4. connects to SHARE-SSM (oscillatory spiking neural networks, 2025): recent work shows that oscillatory dynamics improve spiking neural network performance on sequence tasks (Chen et al. 2025)

### strongest argument AGAINST

1. **language does not have periodic temporal structure.** biological gamma oscillations process sensory input that arrives continuously in time. language tokens arrive in a sequence with no intrinsic periodicity. sentence length varies, phrase boundaries are irregular, syntactic structure is hierarchical not periodic. a periodic modulation of spike threshold would impose regularity on an irregular signal. the modulation would either be averaged out (if frequency does not match any structure) or would create artifactual periodic biases (if it resonates with sequence length, creating a spurious preference for certain positions)

2. **the training reset makes periodic modulation meaningless.** during training, ATMN resets membrane potential at each forward pass. if spikes are computed in a single parallel step (all tokens simultaneously), there is no "t" to modulate -- all tokens are processed at the same time, not sequentially. periodic modulation requires sequential processing (option A from [[neuron_models_to_atmn]]), which is O(T) serial and has not been validated

3. **spike threshold modulation is not gamma oscillation.** biological gamma arises from NETWORK dynamics (E-I interaction). a sinusoidal threshold modulation is a PARAMETER applied to individual neurons. it has no network origin, no E-I interaction, no emergent frequency, no self-stabilization. it is a scheduled threshold, not an oscillation. the computational power of gamma comes from the network dynamics (competitive selection, binding by synchrony), not from the periodic modulation per se

4. **what computational benefit would it provide for language modeling?** gamma oscillations in biology serve binding, attention, and WTA selection. todorov already has mechanisms for all of these:
   - binding: KDA state matrix stores key-value associations
   - attention: MLA performs explicit dot-product attention
   - WTA: ternary spikes select the top activations (via threshold)
   adding periodic modulation does not enhance any of these functions. it adds temporal structure to the spike pattern, but there is no downstream mechanism that READS this temporal structure. no layer in todorov processes spike timing -- all layers process spike VALUES ({-1, 0, +1})

5. **the epsilon*sin(omega*t) term could destabilize training.** if epsilon is too large, the threshold oscillates wildly, creating regions of near-zero threshold (everything spikes) and near-double threshold (nothing spikes). gradient flow through the STE would be disrupted at positions where the threshold is very high or very low. biological oscillations are stabilized by homeostatic mechanisms that do not exist in todorov

### verdict on oscillatory spike threshold

**do NOT implement.** the idea is intellectually appealing (connect ternary spikes to gamma oscillations through threshold modulation) but fails the adversarial stress test on every practical dimension:
- language lacks periodic structure to capture
- training architecture prevents sequential processing
- no downstream mechanism reads spike timing
- the computational functions of gamma (binding, WTA, gating) are already served by other mechanisms
- destabilization risk with no clear benefit

the threshold modulation idea would be more promising in a domain with genuine temporal structure (audio, video, time series) and with sequential spike processing (option A from the ATMN bridge).

## summary of proposed changes

    change                          expected impact    risk    recommendation
    data-dependent rope_freq        LOW-MODERATE       HIGH    defer (needs stability analysis)
    cross-freq coupling in state    LOW                HIGH    do NOT implement
    shared freq across Mamba3       LOW                LOW     do NOT implement (no gating)
    oscillatory spike threshold     VERY LOW           HIGH    do NOT implement
    inter-layer coherence gating    MODERATE           HIGH    do NOT implement (arch change)

## what might actually help

if any oscillation-inspired modification were to be tested, the most defensible candidate is:

**data-dependent rotation frequency in Mamba3**, constrained to a narrow modulation range:

    freq_t = rope_freq * (1 + 0.1 * tanh(freq_proj(x_t)))

rationale: this is the smallest change that adds input-dependent temporal dynamics. it preserves the existing rotation structure, adds one linear projection, and limits frequency modulation to +/-10% to prevent instability. the biological analogue is attention-dependent gamma frequency modulation (~3 Hz shift on a ~40 Hz base, which is ~7.5% modulation).

but even this modification has a low probability of improving BPB on language modeling. the expected benefit is scientific (testing whether data-dependent oscillation frequency helps) rather than practical (unlikely to produce a meaningful BPB reduction).

recommended phase: phase 6+ (after phase 5 sequence is complete). this is a research direction, not a priority.

## prerequisite knowledge

- [[gamma_oscillations]] -- generation mechanisms and computational roles
- [[theta_oscillations]] -- theta-gamma coupling and the theta-gamma code
- [[neural_synchrony]] -- communication through coherence framework
- [[oscillations_vs_recurrence]] -- structured comparison
- [[neuron_models_to_atmn]] -- ATMN analysis, sequential processing proposal

## related bridge docs

- [[neuron_models_to_atmn]] -- ATMN temporal integration (required for oscillatory thresholds)
- [[plasticity_to_kda_delta_rule]] -- KDA as associative memory (alternative to binding-by-synchrony)
- [[lateral_inhibition_to_adaptive_threshold]] -- adaptive threshold analysis

## open questions

1. what frequencies do Mamba3's rope_freq parameters actually learn? after training at 267m scale, do the learned frequencies cluster at specific values? do they correspond to any interpretable timescales (e.g., word length, sentence length, paragraph length)?

2. does the rotation in Mamba3 CONTRIBUTE to performance? ablation: replace rope_freq rotation with no rotation (use_rope=False) at 267m scale and measure BPB difference. if BPB is unchanged, the rotation is not contributing, and the oscillation analogy is moot.

3. LinOSS (oscillatory state-space models, 2024) explicitly parameterizes SSM dynamics as forced harmonic oscillators and achieves competitive performance. how does LinOSS compare to Mamba3's rotation? LinOSS uses complex-valued states with explicit oscillatory structure, which is closer to biological oscillations than Mamba3's post-hoc rotation. if LinOSS outperforms Mamba3, this would suggest that more explicitly oscillatory dynamics ARE beneficial.

4. would replacing Mamba3's rotation with a damped oscillator (rotation + amplitude decay toward equilibrium) improve stability? biological oscillations have amplitude modulation via PAC and homeostasis. Mamba3's rotation preserves amplitude (pure rotation), which could lead to resonance buildup. a damped oscillator would provide natural amplitude control.
