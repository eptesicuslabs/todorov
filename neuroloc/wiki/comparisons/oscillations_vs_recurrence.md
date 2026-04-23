# oscillations vs recurrence

status: current (as of 2026-04-16).

## the question

biological brains coordinate neural activity through oscillatory dynamics: periodic rhythms at multiple frequencies (theta 4-8 Hz, gamma 30-100 Hz) that gate communication, multiplex information, and impose temporal structure. todorov uses recurrent state dynamics (KDA delta-rule recurrence, Mamba3 complex-valued state with data-dependent rotation) to maintain and transform sequential information. do these recurrent dynamics provide equivalent temporal coordination to biological oscillations? or is something computationally significant missing?

## biological oscillatory dynamics

see [[gamma_oscillations]], [[theta_oscillations]], and [[neural_synchrony]] for full details. summary of computational functions:

### temporal coordination

oscillations synchronize distributed neural populations with millisecond precision. gamma oscillations constrain firing to brief ~3-5 ms windows within each ~25 ms cycle. this temporal coordination enables:
- binding by synchrony (features of the same object fire in the same gamma cycle)
- competitive selection ([[winner_take_all]]: strongest neurons fire first in each cycle)
- precise spike timing for [[stdp]] (temporal order within gamma cycles enables directional learning)

### multiplexing across timescales

the brain simultaneously operates at multiple temporal scales through cross-frequency coupling:
- gamma (30-100 Hz): local computation, feature processing
- theta (4-8 Hz): memory encoding, sequential organization
- alpha (8-13 Hz): inhibitory gating, top-down suppression
- beta (13-30 Hz): feedback communication, motor planning

these oscillations are nested: gamma is modulated by theta, which is modulated by delta. each frequency band carries different information at different spatial scales. a single neural population participates in all of these simultaneously.

### phase-based coding

information is encoded not just in firing rates but in the phase of firing relative to ongoing oscillations:
- theta phase precession: position encoded by firing phase within theta cycle
- gamma phase: input strength encoded by spike latency within gamma cycle
- phase-of-firing: ~54% additional information beyond rate coding (Montemurro et al. 2008)

### communication gating

oscillatory coherence between brain areas selectively gates communication (Fries 2015):
- gamma coherence between V1 and V4 enables feedforward visual processing
- alpha suppression in unattended areas prevents irrelevant input
- theta coherence between hippocampus and prefrontal cortex enables memory retrieval
- the same physical connections can transmit or block information depending on phase alignment

## todorov recurrent dynamics

### KDA (delta-rule recurrence)

the dominant layer type (18/24 layers). KDA maintains a matrix-valued state:

    S_t = S_{t-1} + sigma_t * (v_t * k_t^T - beta_t * S_{t-1})

where sigma_t is an update gate, v_t * k_t^T is the Hebbian outer product write, and beta_t * S_{t-1} is the erasure/forgetting term. the state is read out by:

    y_t = S_t * q_t

computational properties:
- **associative memory:** the outer product write stores key-value associations, retrievable by query
- **error correction:** the delta rule (subtract old association, add new) prevents interference between stored items
- **continuous state:** S_t evolves smoothly, with no periodic modulation or oscillatory dynamics
- **data-dependent updates:** the gates sigma_t and beta_t are input-dependent, providing adaptive read/write control

### Mamba3 (complex-valued state with rotation)

3/24 layers. Mamba3 maintains a state h_t in R^{d_inner x d_state}:

    h_t = A_bar_t * h_{t-1} + B_bar_t * B_t * x_t

followed by data-dependent rotation:

    angle = rope_freq * t
    rotated_real = real * cos(angle) - imag * sin(angle)
    rotated_imag = real * sin(angle) + imag * cos(angle)

where rope_freq is a learned parameter of shape (d_state//2,).

computational properties:
- **state decay:** A_bar_t provides data-dependent exponential decay (learned A_log initialized in [-4, -1])
- **periodic dynamics:** the rotation applies a per-dimension oscillation at learned frequency rope_freq
- **data-dependent discretization:** the dt_proj and dt_bias allow the model to control the effective time constant per input
- **no cross-dimensional interaction:** each state dimension rotates independently (no coupling between frequencies)

### MLA (compressed exact attention)

3/24 layers. MLA performs standard dot-product attention over compressed representations. no recurrent state -- purely feedforward within a layer, attending over the full context window. no oscillatory or temporal dynamics.

## dimension 1: temporal coordination

    property                    biology                      todorov
    timing precision            ~1-2 ms (gamma cycle)        no explicit temporal precision
    synchronization mechanism   E-I network oscillations     none (layers process serially)
    temporal binding            same-cycle co-activation     co-representation in state matrix
    population coordination     oscillatory coherence        residual stream (shared, not gated)

**assessment:** todorov has no mechanism for temporal coordination in the biological sense. KDA stores associations in a matrix; Mamba3 rotates state at learned frequencies. neither creates temporal windows that constrain when information is processed. the residual stream is always available to all layers -- there is no gating of information access by temporal phase.

however: temporal coordination in biological oscillations serves to manage interference between concurrent processes in a shared neural substrate. todorov's layers do not share substrate in the same way -- each layer processes the residual stream sequentially and has its own parameters. the need for temporal coordination may be reduced by the serial architecture.

## dimension 2: multiplexing across timescales

    property                    biology                      todorov
    concurrent timescales       4+ (delta, theta, alpha,     2 (KDA has no intrinsic timescale;
                                gamma, all simultaneously)   Mamba3 has learned frequencies)
    cross-frequency coupling    theta-gamma PAC, multiple    none (KDA and Mamba3 do not
                                nesting levels               interact oscillatorily)
    information at each scale   different content per        single residual stream carries
                                frequency band               all information

**assessment:** todorov has limited multi-timescale dynamics. Mamba3's learned rope_freq parameters could, in principle, settle to different frequencies (some fast, some slow), but there is no coupling between them. KDA has no intrinsic timescale at all -- its state dynamics are driven entirely by the input sequence.

the biological multi-scale structure is computationally powerful because it separates information by temporal resolution: gamma carries fine-grained sensory details, theta carries sequential structure, alpha/beta carry top-down context. todorov represents all of this in a single undifferentiated residual stream.

## dimension 3: phase-based coding

    property                    biology                      todorov
    phase as information        yes (theta phase precession, no explicit phase coding
                                gamma latency code)
    information beyond rate     ~54% additional (phase)      ternary spikes carry sign + magnitude
                                                             threshold, not timing
    positional encoding         phase relative to ongoing    sinusoidal positional encoding
                                oscillation                  (RoPE) in attention and Mamba3

**assessment:** todorov has no phase-based coding. the ternary spikes ({-1, 0, +1}) encode sign and presence, not timing relative to an oscillation. RoPE in KDA and Mamba3 provides positional encoding that uses sinusoidal functions, but these are applied to the input/state representation, not to spike timing. the sinusoids in RoPE define a POSITION code, not a TEMPORAL code -- they encode "where in the sequence" not "when in an oscillatory cycle."

the closest analogue: Mamba3's data-dependent rotation does impose position-dependent phase shifts on the state. but these phases encode sequential position, not temporal timing within a processing cycle.

## dimension 4: communication gating

    property                    biology                      todorov
    selective routing            gamma coherence gates       no selective routing between layers
                                inter-area communication
    top-down modulation          alpha/beta suppress or      no top-down from later to earlier
                                enhance gamma in lower       layers (feedforward only)
                                areas
    dynamic connectivity         same connections transmit   same residual stream always
                                or block depending on        available to all layers
                                phase alignment
    attention mechanism          coherence-based selection   dot-product attention (MLA) and
                                                             gate-based selection (KDA sigma)

**assessment:** todorov has no dynamic communication gating analogous to CTC. all layers always have full access to the residual stream. MLA performs attention-based selection within a layer (selecting which tokens to attend to), but this is a static computation within the layer, not a dynamic gating of inter-layer communication.

the KDA sigma gate (update gate) provides DATA-dependent gating of writes to the state matrix, which is functionally analogous to the role of attention in selecting what information to store. but it does not gate communication BETWEEN processing stages.

## dimension 5: Mamba3 rotation as oscillation

the strongest analogy between todorov and biological oscillations is Mamba3's data-dependent rotation. the rotation:

    angle = rope_freq * t
    [real, imag] = rotation_matrix(angle) * [real, imag]

is mathematically identical to a harmonic oscillator with angular frequency omega = rope_freq. each of the d_state//2 dimensions oscillates independently at its own learned frequency.

**strongest argument FOR the analogy:**
- both are periodic dynamics applied to state representations
- both have per-dimension (per-neuron) frequencies
- both frequencies can adapt (biological: via neuromodulation and network dynamics; todorov: via gradient descent)
- the complex rotation IS a rotation in phase space, just as biological oscillations trace phase-space trajectories
- learned frequencies could settle to task-relevant timescales

**strongest argument AGAINST the analogy:**
1. biological oscillations emerge from NETWORK dynamics (E-I interaction in PING/ING). Mamba3 rotation is a PARAMETERIZED operation applied to a single layer's state. the network origin of biological oscillations means they are intrinsically coupled to the circuit that generates them (frequency depends on inhibitory strength, which depends on attention, which depends on coherence). Mamba3's frequencies are fixed after training
2. biological frequencies MODULATE with brain state: gamma frequency increases by ~3 Hz with attention; theta frequency varies with running speed. Mamba3's rope_freq is a static learned parameter. it cannot modulate online based on input content or task demands. the dt_proj provides some data-dependent modulation of the effective time constant, but not of the rotation frequency itself
3. Mamba3 rotation is applied INDEPENDENTLY per feature dimension. no coupling between frequencies (no theta-gamma nesting, no phase-amplitude coupling). the computational power of biological oscillations comes largely from cross-frequency coupling, which is entirely absent
4. only 3/24 layers have rotation. if oscillatory dynamics were computationally essential, why would 75% of layers (KDA) lack them entirely? the architecture treats Mamba3 as a minority complement to KDA, not as a fundamental organizing principle
5. the rotation may serve POSITIONAL ENCODING, not oscillatory computation. RoPE in attention mechanisms provides relative positional information via rotation -- the same mathematical structure. Mamba3's rope_freq may simply be learned RoPE for the recurrent state, encoding "how many timesteps ago" rather than implementing oscillatory dynamics
6. biological oscillations coordinate BETWEEN areas/layers. Mamba3 operates WITHIN one layer. there is no mechanism for oscillatory coherence between layers 4, 12, and 20 (the three Mamba3 positions). each Mamba3 layer rotates its own state independently

## verdict

**todorov's recurrent dynamics do NOT provide equivalent temporal coordination to biological oscillations.** the analogy is weakest at the most computationally important functions:

1. **communication gating** (CTC): completely absent. todorov has no dynamic inter-layer routing
2. **cross-frequency coupling** (theta-gamma nesting): completely absent. no interaction between Mamba3 frequencies or between Mamba3 rotation and KDA state
3. **temporal binding** (synchrony-based): replaced by a fundamentally different mechanism (co-storage in KDA state matrix)
4. **phase coding** (spike timing relative to oscillation): absent. ternary spikes encode magnitude, not temporal phase

the analogy is strongest for a single, narrow claim: Mamba3's complex rotation is a per-feature oscillator. this is mathematically correct but computationally weak because the oscillator lacks the two properties that make biological oscillations powerful: (a) network-emergent frequency modulation and (b) cross-frequency coupling.

**what todorov provides instead:**
- KDA delta-rule: content-addressable associative memory (different computational primitive from oscillatory binding)
- Mamba3 rotation: position-dependent state encoding (likely serves RoPE-like positional function)
- MLA attention: explicit similarity-based retrieval over full context
- residual stream: shared communication channel (always-on, not gated)

these are effective mechanisms for sequence processing. they are not biologically analogous to oscillatory dynamics. they solve similar problems (coordination, memory, routing) through different computational strategies.

## dissenting argument

the comparison above may overstate the importance of oscillations for the computational tasks todorov actually performs. consider:

1. **oscillations coordinate parallel processing; todorov processes serially.** the need for temporal coordination arises because biological cortex processes information in parallel across many areas simultaneously, creating interference and routing problems. todorov's serial layer stack processes information one layer at a time. there is no interference to manage because no two layers are active simultaneously. oscillatory gating may be a solution to a problem todorov does not have.

2. **the residual stream is a more powerful communication channel than oscillatory coherence.** oscillatory gating is binary (transmit or block) and noisy (phase jitter limits precision). the residual stream provides full-bandwidth, deterministic communication between all layers. what biology achieves crudely through coherence, todorov achieves precisely through the residual stream.

3. **KDA's associative state may subsume binding-by-synchrony.** the binding problem (grouping features of the same object) is solved in biology by synchronizing neurons within gamma cycles. KDA solves it by storing associated features as key-value pairs in the same state matrix. the KDA solution is arguably more powerful: it can store and retrieve arbitrary associations, not just within-cycle temporal groups.

4. **the absence of oscillations may be a feature, not a bug.** biological oscillations impose computational overhead (metabolic cost of maintaining rhythms, information loss from phase-based gating, capacity limits from theta-gamma ratio). a system that achieves equivalent function without these costs is more efficient, not less capable.

**against the dissent:** these arguments explain why todorov can FUNCTION without oscillations but do not establish that nothing is lost. the specific capabilities that oscillations provide -- dynamic routing, multi-timescale multiplexing, phase-based information -- may enable computational strategies that todorov cannot access. the question is not whether todorov works (it does) but whether it could work BETTER with oscillation-inspired mechanisms. see [[oscillations_to_mamba3_rotation]] for concrete proposals.

## key references

- Fries, P. (2015). rhythms for cognition: communication through coherence. neuron, 88(1), 220-235.
- Buzsaki, G. and Draguhn, A. (2004). neuronal oscillations in cortical networks. science, 304(5679), 1926-1929.
- Lisman, J. E. and Jensen, O. (2013). the theta-gamma neural code. neuron, 77(6), 1002-1016.

## see also

- [[gamma_oscillations]]
- [[theta_oscillations]]
- [[neural_synchrony]]
- [[oscillations_to_mamba3_rotation]]
- [[plasticity_to_matrix_memory_delta_rule]]
- [[cortical_layers_vs_todorov_layers]]
