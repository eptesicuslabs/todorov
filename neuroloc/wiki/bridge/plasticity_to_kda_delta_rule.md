# bridge: synaptic plasticity to KDA delta rule

status: current (as of 2026-04-16).

## the biological mechanism

synaptic plasticity encompasses multiple mechanisms operating across different timescales:

- [[hebbian_learning]]: correlation-based strengthening. Delta_w = eta * x_pre * x_post. the outer product rule is the simplest associative storage prescription. operates on seconds-to-minutes timescale. unstable without normalization.
- [[stdp]]: timing-dependent refinement of Hebbian learning. pre-before-post LTP, post-before-pre LTD. operates on 10-20 ms timing windows. the asymmetric window implements causal inference and temporal sequence learning.
- [[bcm_theory]]: sliding modification threshold theta_M prevents runaway excitation. theta_M = <c^2>/c_0 rises with activity, making LTP harder. operates on hours timescale. implements metaplasticity.
- [[homeostatic_plasticity]]: multiplicative synaptic scaling maintains target firing rate. w_i -> s * w_i for all synapses. operates on hours-to-days timescale. preserves relative weight structure.
- [[short_term_plasticity]]: facilitation (calcium accumulation, tau_f ~ 50-500 ms) and depression (vesicle depletion, tau_d ~ 200-800 ms). not memory storage but temporal filtering. the Tsodyks-Markram model captures both with two interacting state variables.

the common thread: biological synaptic modification uses LOCAL information (pre and postsynaptic activity, current weight, possibly a neuromodulatory broadcast) and operates through multiple overlapping timescales to balance learning with stability.

## the current todorov implementation

KDA delta rule (src/layers/kda.py, lines 190-209):

    S_t = diag(alpha) * S_{t-1} + beta_t * k_t * v_t^T

component-by-component analysis:

**alpha (channel-wise forgetting rate)**
- alpha = sigmoid(alpha_log), where alpha_log ~ N(-2.0, 0.5)
- initial alpha ~ sigmoid(-2) ~ 0.12, meaning ~88% of each channel's state is erased per timestep
- learned, per-channel per-head (shape: [num_heads, head_dim])
- the channel-wise structure means different "feature dimensions" of the associative memory can decay at different rates
- functionally: exponential memory decay. S_t carries ~alpha^k fraction of the association written at timestep t-k
- biological analog: closest to short-term depression (vesicle depletion causing activity-dependent decay of synaptic efficacy), but with a critical difference: alpha is FIXED per channel, not activity-dependent

**beta_t (data-dependent write gate)**
- beta_t = sigmoid(beta_proj(x_t)), where beta_proj is a learned linear projection to [num_heads]
- one scalar gate per head, computed from the current input
- determines HOW MUCH of the new key-value association to write
- functionally: input-dependent gating of learning. some inputs write strongly (beta near 1), others barely write (beta near 0)
- biological analog: closest to neuromodulatory gating (third-factor learning). the "third factor" here is the input itself, not an external reward or error signal. also resembles the utilization parameter u in the Tsodyks-Markram model, which modulates how much of the available synaptic resources are deployed

**k_t * v_t^T (outer product association)**
- standard outer product between key and value vectors, each of shape [batch, heads, head_dim]
- produces a rank-1 update to the state matrix S_t (shape [heads, head_dim, head_dim])
- functionally: stores a Hebbian association between the "query-space representation" (k) and the "content representation" (v) of the current token
- biological analog: this is the classical Hebbian outer product rule, identical in form to the Hopfield storage prescription. it is the most direct biological correspondence in the entire KDA mechanism

**readout: o_t = q_t^T * S_t**
- the current query retrieves from the accumulated state by matrix-vector multiplication
- functionally: content-addressable retrieval. the query "probes" the associative memory for the stored value most similar to the current query
- biological analog: pattern completion in Hopfield networks, or cue-dependent recall in hippocampal memory models

**what is NOT in the current implementation:**
- no timing dependence: k_t and v_t are from the SAME timestep. there is no mechanism that compares activity at different times (unlike [[stdp]])
- no sliding threshold: alpha is fixed per channel, not adjusted based on activity history (unlike [[bcm_theory]])
- no homeostatic scaling: there is no mechanism monitoring state magnitude and adjusting alpha or beta accordingly (unlike [[homeostatic_plasticity]])
- no facilitation: there is no mechanism that increases the write strength with repeated similar input (unlike STF in the Tsodyks-Markram model)
- no explicit erasure: the delta rule in its original formulation (Schlag et al. 2021) includes an erasure term that removes old key-value associations before writing new ones. KDA relies on alpha decay for forgetting, which is indiscriminate (decays everything equally) rather than targeted

## is KDA a form of STDP?

**no.** the analysis must be precise:

1. STDP requires timing dependence between PRE and POST activity at DIFFERENT times. the sign and magnitude of Delta_w depend on t_post - t_pre. KDA's k_t * v_t^T uses k and v from the SAME timestep t. there is no timing window.

2. STDP modifies WEIGHTS that persist across the lifetime of the synapse. S_t is not a weight matrix. it is a recurrent state that is recomputed from scratch every forward pass. the "associations" in S_t are transient (they decay with rate alpha), not permanent.

3. STDP is a LEARNING rule (it changes the parameters of the system). the KDA state update is a COMPUTATION rule (it computes the state as a function of the input and previous state). the parameters of KDA (alpha_log, beta_proj, q/k/v projections) are learned by backpropagation, not by the state update itself.

4. the recurrence DOES create temporal structure: S_t contains exponentially decayed contributions from all past timesteps. but this temporal structure is in the STATE (the thing being read), not in the UPDATE RULE (the thing that modifies parameters). biological STDP creates temporal structure in the WEIGHTS.

**the strongest counter-argument against the STDP analogy:** KDA is closer to a Hopfield-like associative memory with exponential forgetting than to any form of synaptic plasticity. each timestep writes a new pattern (k_t * v_t^T) into the memory (S_t), and the query retrieves the most similar stored pattern. the exponential decay (alpha) implements a recency-weighted forgetting curve. this is an ASSOCIATIVE MEMORY model, not a PLASTICITY model. the "plasticity" analogy is misleading because it confuses the mechanism of inference (KDA state update) with the mechanism of learning (backpropagation).

**what KDA IS, biologically:**
- it is most accurately described as a recurrent associative memory with Hebbian writing, exponential forgetting, and content-addressable retrieval
- the Hebbian writing (outer product) is genuine and well-motivated
- the forgetting (alpha decay) is closest to short-term depression, not STDP
- the gating (beta_t) is closest to neuromodulatory gating (third-factor modulation)
- the readout (q^T * S) is closest to Hopfield pattern completion

## the proposed change

biological plasticity suggests three modifications that KDA currently lacks:

### 1. activity-dependent forgetting (BCM-like sliding threshold)

current: alpha is fixed per channel (learned once, constant during inference).

proposed: make alpha depend on the recent history of state activation:

    alpha_effective = sigmoid(alpha_log + gamma * log(||S_t||))

when the state norm is large (high "activity"), alpha_effective increases, accelerating forgetting. when the state is near zero (low activity), alpha_effective decreases, preserving what little information exists. this implements a BCM-like sliding threshold for the forgetting rate.

rationale: prevents state saturation during long sequences without requiring manual tuning of alpha_log initialization.

### 2. targeted erasure (delta rule proper)

current: forgetting is indiscriminate (alpha * S_{t-1} decays everything equally).

proposed: add a targeted erasure term before writing:

    S_t = diag(alpha) * S_{t-1} - beta_t * (k_t * (k_t^T * S_{t-1})) + beta_t * k_t * v_t^T

the term k_t * (k_t^T * S_{t-1}) removes the old value associated with the current key before writing the new value. this is the standard delta rule (Schlag et al. 2021) and is more biologically plausible than indiscriminate decay because it implements associative interference: new information overwrites old information at the same "address" rather than degrading all stored information equally.

rationale: improves memory utilization by selectively erasing outdated associations rather than relying on global decay.

### 3. homeostatic state scaling

current: no mechanism to prevent state magnitude from growing or shrinking unboundedly over long sequences.

proposed: add a slow homeostatic term:

    S_t = S_t * (target_norm / (||S_t|| + epsilon))^(1/tau_h)

where tau_h >> 1 is a slow timescale (e.g., tau_h = 100 timesteps) and target_norm is a learned or fixed target for the state magnitude. this implements [[homeostatic_plasticity]]-like scaling of the state matrix.

rationale: maintains stable state dynamics over arbitrarily long sequences without disrupting the relative structure of stored associations.

## implementation spec

### activity-dependent alpha (BCM-like)

in KDALayer._forward_recurrent (src/layers/kda.py, lines 190-209):

    state_norm = state.norm(dim=(-2, -1), keepdim=True)
    log_state_norm = torch.log(state_norm.clamp(min=1e-6))
    alpha_effective = torch.sigmoid(
        self.alpha_log.unsqueeze(0).unsqueeze(-1)
        + self.alpha_gamma * log_state_norm
    )

new parameter: self.alpha_gamma = nn.Parameter(torch.zeros(num_heads, 1)) initialized at 0 (no effect initially, learned from data).

### targeted erasure

in the recurrent loop:

    old_value = torch.einsum("bhd,bhde->bhe", k[:, t], state)
    erasure = torch.einsum("bhd,bhe->bhde", k[:, t], old_value)
    state = alpha_unsq * state - beta_t * erasure + beta_t * torch.einsum("bhd,bhe->bhde", k[:, t], v[:, t])

this adds one extra einsum per timestep. the parallel path needs corresponding modification.

### homeostatic scaling

applied every tau_h timesteps (e.g., every 64 or 128 tokens):

    if t % tau_h == 0:
        current_norm = state.norm(dim=(-2, -1), keepdim=True)
        scale = (target_norm / (current_norm + 1e-8)) ** (1.0 / tau_h)
        state = state * scale.clamp(0.9, 1.1)

the clamp prevents drastic rescaling.

## expected impact / risk assessment

### activity-dependent alpha
- expected: better long-context performance by preventing state saturation
- risk: low. alpha_gamma initialized at 0, so the default behavior is unchanged. the modification is a smooth, differentiable function of state norm
- validation: compare perplexity at 2048 vs 512 tokens with and without alpha_gamma

### targeted erasure
- expected: improved associative recall accuracy, especially for repeated keys with updated values (the "update" problem in sequence modeling)
- risk: moderate. adds computational cost (one extra einsum per timestep). may interact poorly with the fla kernel path, which assumes a specific state update form. the erasure term creates a dependency of the update on the current state, which may complicate gradient flow
- validation: test on selective copy task (which specifically tests the ability to overwrite old associations)

### homeostatic scaling
- expected: more stable training on very long sequences (4096+ tokens)
- risk: moderate. the periodic normalization introduces a non-smooth operation. the clamping prevents catastrophic rescaling but may introduce gradient estimation issues. the tau_h hyperparameter needs tuning
- validation: monitor state norm statistics during training with and without scaling

### overall assessment
these three modifications bring KDA closer to biological plasticity principles while preserving its core strengths (fast Hebbian writing, content-addressable retrieval, efficient recurrence). the BCM-like alpha is the lowest-risk change and should be tested first. targeted erasure is the highest-impact change for memory-intensive tasks. homeostatic scaling is a safety mechanism for long-context training stability.

all three should be tested in isolation following the phase sequencing protocol (one change per run, matched step counts, statistical significance testing). they should NOT be combined in a single experiment.

## does the absence of STDP in KDA limit what todorov can learn?

**probably not for language modeling.** language is a sequence of discrete tokens, not a continuous spike train. the relevant temporal structure is captured by the recurrence (S_t carries past information) and the positional encoding (RoPE), not by the relative timing of individual "spikes." the ternary spike quantization in todorov provides sparsity benefits, but it does not create a timing-dependent plasticity window.

**possibly for temporal pattern learning.** STDP excels at learning temporal sequences and detecting causal relationships between events separated by short time delays. KDA's outer product association is instantaneous (k_t and v_t from the same timestep), so it cannot directly learn "if pattern A occurs 5 tokens before pattern B, strengthen the association." it CAN learn this indirectly through the recurrence (the decayed state carries information about pattern A when pattern B arrives), but the learning rule itself does not explicitly encode timing dependence.

**what would be gained by adding STDP-like timing dependence?** one could modify the state update to use keys and values from DIFFERENT timesteps:

    S_t = alpha * S_{t-1} + beta_t * k_{t-delta} * v_t^T    (for some delay delta)

this would create an explicit timing window: the association is between a past key and the current value. this is more biologically realistic and might improve performance on tasks requiring precise temporal pattern matching (e.g., music generation, rhythmic pattern detection). however, it would also increase computational cost (need to buffer past keys) and add a hyperparameter (delta or a distribution over delays).

the honest assessment: for autoregressive language modeling, the current KDA design is likely sufficient. the recurrence provides implicit temporal context, and explicit STDP-like timing dependence would add complexity without clear benefit. for tasks requiring fine-grained temporal pattern learning, STDP-inspired modifications are worth investigating but would need to be validated empirically.
