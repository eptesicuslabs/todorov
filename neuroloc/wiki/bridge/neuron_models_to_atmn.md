# bridge: neuron models to ATMN

status: current (as of 2026-04-16).

## the biological mechanism

four neuron models -- [[hodgkin_huxley]], [[leaky_integrate_and_fire]], [[adaptive_exponential]], [[izhikevich_model]] -- describe how biological neurons integrate input current, fire discrete spikes, and reset. the core computational primitives shared by all models are:

1. temporal integration (accumulate input over time)
2. leak (exponential decay toward resting state)
3. threshold (emit spike when membrane potential exceeds threshold)
4. reset (return to sub-threshold state after spike)
5. adaptation (reduce excitability after repeated firing)

see [[neuron_model_comparison]] for detailed comparison.

## the current todorov implementation

ATMN (src/spikes/atmn_spike.py):

membrane potential update:

    h_t = x_t + (1/tau) * u_{t-1}

where x_t is the input and u_{t-1} is the previous membrane potential. tau=2.0 by default.

fire:

    spikes = sign(h_t) * [|h_t| > V_th]

where V_th = exp(threshold_log), a per-neuron learnable parameter. output is ternary: {-1, 0, +1}.

reset:

    u_t = h_t - spikes * V_th

the membrane potential is reduced by V_th at positions that fired. residual potential carries over.

training behavior:

    if self.training:
        self.membrane_potential = torch.zeros_like(self.membrane_potential)

the membrane potential is reset to zero at the start of each forward pass during training. this means there is NO temporal state accumulation across batches during training -- each batch starts from a blank slate.

inference behavior: membrane potential persists across forward calls (no reset to zero).

learnable parameters: threshold_log (per-neuron, d_features values). tau is fixed at construction.

## adversarial analysis: is ATMN a faithful LIF?

short answer: no. ATMN omits the most important feature of the LIF (the leak) and its training behavior eliminates temporal integration.

detailed comparison:

**1. no leak term.**

LIF: tau_m * dV/dt = -(V - V_rest) + R*I(t)

the term -(V - V_rest) is the leak. it drives the membrane potential exponentially back toward rest with time constant tau_m. biologically, it represents the passive ion channels that are always open. computationally, it prevents unbounded accumulation of input and provides a natural forgetting mechanism.

ATMN: h_t = x_t + (1/tau) * u_{t-1}

there is no decay term. the (1/tau) coefficient on u_{t-1} scales the carried-over potential but does not decay it toward zero. with tau=2.0, the carried-over potential is halved -- but this is a constant scaling, not an exponential decay. after n timesteps without input, the LIF potential decays as exp(-n*dt/tau_m) -> 0; the ATMN potential decays as (1/tau)^n = 0.5^n. these have different qualitative behavior for continuous inputs.

impact assessment: MODERATE-HIGH. the leak is essential for preventing pathological accumulation of membrane potential over long sequences. without it, neurons that receive sustained positive (or negative) input will accumulate potential without bound, eventually saturating. the (1/tau) scaling partially mitigates this but is not equivalent to a proper leak. the batch reset during training masks the problem (since potential never accumulates beyond one batch), but at inference with long sequences, the lack of leak could cause drift.

proposed fix:

    h_t = (1 - dt/tau_leak) * u_{t-1} + x_t

where dt/tau_leak is a learnable leak rate in (0, 1). this is exactly the discrete-time LIF: exponential decay of previous state plus new input. cost: one multiply per neuron per timestep.

**2. batch reset during training.**

in the forward pass:

    if self.training:
        self.membrane_potential = torch.zeros_like(self.membrane_potential)

this resets ALL membrane state to zero at the start of every forward call. since each training step calls forward() once per batch, the membrane potential never accumulates across batches.

biological neurons do not reset between "batches" of stimuli. their membrane potential evolves continuously. the state at any moment reflects the entire recent history (filtered by the leak time constant).

impact assessment: HIGH. this is the most significant departure from biological neuron dynamics. it means:

- ATMN cannot learn temporal dynamics that span multiple forward passes during training.
- the membrane potential state (u) is only meaningful within a single sequence. there is no cross-sequence state.
- the tau parameter and the (1/tau)*u term are only active within a single forward pass. if the input tensor has shape (batch, seq_len, features) and is processed token-by-token, the membrane potential accumulates across seq_len tokens. but if the entire sequence is processed in one call (as in the current implementation), the membrane is reset before every sequence.

looking at the forward() code more carefully: the input is reshaped to (batch*seq_len, features) and processed as a SINGLE TIMESTEP. this means u_{t-1} is the potential from the PREVIOUS call to forward(), not from the previous token. within a batch, there is only ONE integration step, not seq_len steps.

this is critical: ATMN is NOT integrating across the sequence. it is applying a ternary threshold function with a residual carry from the previous batch (during inference) or no carry at all (during training).

proposed fix: process tokens sequentially within forward(), accumulating membrane state across sequence positions. this would make the computation sequential (O(T) serial steps) but would enable genuine temporal integration.

alternative fix: maintain the single-step processing but remove the training reset, so that membrane potential carries across training batches. this preserves parallelism but requires careful initialization and gradient handling.

**3. learnable per-neuron threshold.**

ATMN: V_th = exp(threshold_log), where threshold_log is a learnable parameter of shape (d_features,).

biological thresholds are not individually tunable. they arise from Na+ channel density (fixed by genetics and development) and are modulated dynamically by:
- Na+ channel inactivation (threshold increases after firing)
- neuromodulation (acetylcholine, norepinephrine can shift threshold)
- recent voltage history (accommodation: slow depolarization raises threshold)

ATMN's learnable threshold is more powerful than biology in one sense (each neuron independently optimizes its threshold via gradient descent) but less powerful in another (the threshold is static once learned -- it does not adapt to input dynamics).

impact assessment: LOW-MODERATE. the learnable threshold is probably fine for artificial architectures. per-neuron thresholds allow the network to learn different sparsity levels at different positions in the feature vector, which is a useful inductive bias. the exp() parameterization ensures positivity and provides a natural log-space learning dynamic.

the missing feature is dynamic threshold adaptation (threshold that increases after firing). this is closely related to the adaptation mechanism in [[adaptive_exponential]] and [[izhikevich_model]]. if adaptation proves useful, dynamic thresholds would be a lightweight way to implement it: increase threshold_log by a fixed amount at each spike, decay it back to its learned value between spikes.

**4. the tau parameter is not a leak time constant.**

ATMN: tau=2.0 means the carried-over potential is multiplied by 0.5.

LIF: tau_m = C_m/g_L is the membrane time constant in continuous time. the discrete-time equivalent for euler integration at timestep dt is:

    V_{t+1} = V_t + (dt/tau_m) * (-(V_t - V_rest) + R*I_t)
            = (1 - dt/tau_m) * V_t + (dt/tau_m) * (V_rest + R*I_t)

the decay factor is (1 - dt/tau_m), which depends on both the time constant and the integration timestep. for tau_m = 20 ms and dt = 1 ms, the decay factor is 0.95. for tau_m = 2 ms and dt = 1 ms, the decay is 0.5.

ATMN's (1/tau) = 0.5 is equivalent to a LIF with tau_m/dt = 2 -- a very short time constant (2 timesteps). this means ATMN "forgets" the previous potential rapidly. but since ATMN has no leak toward a resting potential, the 0.5 factor is just a scaling, not a proper decay.

impact assessment: LOW. fixing the leak term (point 1) automatically resolves this. the tau parameter should be reinterpreted as a leak time constant once leak is added.

## the proposed change

based on the adversarial analysis, the minimum viable improvement to ATMN has three components, in order of priority:

**change 1 (high priority): add leak term.**

replace:

    h_t = x_t + (1/tau) * u_{t-1}

with:

    h_t = (1 - alpha) * u_{t-1} + x_t

where alpha = sigmoid(leak_log) is a per-neuron learnable leak rate in (0, 1). alpha=0 means no leak (LSTM-like); alpha=1 means full leak (stateless). the sigmoid parameterization constrains alpha to valid range and provides smooth gradients.

this is equivalent to a discrete-time LIF with learnable time constant tau_eff = -1/ln(1-alpha).

**change 2 (high priority): fix temporal integration.**

option A (sequential, correct): loop over sequence positions in forward(), accumulating membrane state across tokens:

    for t in range(seq_len):
        h_t = (1 - alpha) * u_{t-1} + x[:, t, :]
        spikes_t = quantize(h_t, V_th)
        u_t = h_t - spikes_t * V_th
        output[:, t, :] = spikes_t

cost: O(T) serial steps. not parallelizable within a sequence. this is the biologically correct approach.

option B (parallel, approximate): process all tokens in parallel, use the membrane potential only as a cross-batch carry:

    h = x + (1 - alpha) * u_prev.unsqueeze(1)
    spikes = quantize(h, V_th)
    u_new = h[:, -1, :] - spikes[:, -1, :] * V_th

cost: O(1) parallel. only the last token's membrane potential carries forward. this loses within-sequence temporal dynamics but maintains cross-batch state.

recommendation: option A for sequences where temporal dynamics matter (phase 5a validation). option B for production if temporal dynamics prove unimportant.

**change 3 (medium priority): remove training reset.**

delete:

    if self.training:
        self.membrane_potential = torch.zeros_like(self.membrane_potential)

replace with: reset membrane potential only at the start of each new document/sequence, not at each forward() call. this requires the training loop to call reset_state() explicitly at document boundaries.

risk: gradient explosion if membrane potential accumulates across many training steps without leak. change 1 (leak term) mitigates this risk.

## implementation spec

file: src/spikes/atmn_spike.py

changes to __init__:

    add: self.leak_log = nn.Parameter(torch.full((d_features,), -1.0))
    (initializes alpha = sigmoid(-1.0) ~ 0.27, corresponding to tau_eff ~ 3.2 timesteps)

add property:

    @property
    def leak_rate(self) -> Tensor:
        return torch.sigmoid(self.leak_log)

changes to forward (option A, sequential):

    def forward(self, x: Tensor) -> Tensor:
        original_shape = x.shape
        B, T = x.shape[0], 1
        if x.dim() == 3:
            B, T, D = x.shape
        else:
            D = x.shape[-1]
            x = x.unsqueeze(1)

        x_flat = x.reshape(B, T, D)

        if not self._state_initialized or self.membrane_potential.shape[0] != B:
            self.reset_state(B, x.device)

        alpha = self.leak_rate
        v_th = self.threshold

        outputs = []
        u = self.membrane_potential
        for t in range(T):
            h = (1 - alpha) * u + x_flat[:, t, :]
            spikes = ATMNQuantizer.apply(h, v_th)
            u = (h - spikes * v_th).detach()
            outputs.append(spikes)

        self.membrane_potential = u

        output = torch.stack(outputs, dim=1)
        return output.view(original_shape)

changes to reset_state: no change needed (already zeros membrane potential).

remove the training reset block:

    if self.training:
        self.membrane_potential = torch.zeros_like(self.membrane_potential)

## expected impact

positive:
- leak term prevents unbounded membrane potential accumulation at inference
- sequential processing enables genuine temporal integration across tokens
- learnable leak rate allows the network to optimize its own forgetting curve
- closer alignment with biological LIF dynamics

negative:
- sequential processing makes forward pass O(T) serial instead of O(1) parallel
- removing training reset may cause gradient issues during early training
- adding one learnable parameter per neuron (leak_log) increases parameter count by 50% (from 2 to 3 per neuron, still negligible at 300M scale)

expected BPB impact: uncertain. the leak term and temporal integration change the inductive bias fundamentally. this needs experimental validation. the comparison in [[neuron_model_comparison]] suggests leak helps (prevents drift) but the dissenting argument (leak destroys long-range memory) is also plausible.

## risk assessment

1. **sequential bottleneck.** option A makes the spike computation O(T) serial. at T=2048, this could be a training bottleneck. mitigation: use option B (parallel) during training, option A during validation/inference.

2. **gradient explosion without training reset.** removing the training reset means membrane potential accumulates across the entire sequence. with T=2048 tokens and no leak, the potential could grow to O(2048 * mean_input). mitigation: the leak term provides exponential decay, but the leak rate must be large enough to prevent divergence. consider clamping membrane potential to [-100, 100] as a safety net.

3. **backward compatibility.** changing the forward() signature to expect 3D input (B, T, D) instead of arbitrary shape may break existing callers. mitigation: preserve the original reshape logic and handle both 2D and 3D inputs.

4. **training instability during transition.** switching from batch-reset to persistent membrane potential changes the loss landscape. models pre-trained with reset may not transfer well. mitigation: fine-tune with low learning rate for the leak_log parameter.

5. **the change may not help.** if ATMN's ternary spike is already optimal as a per-token activation function, adding temporal dynamics is unnecessary complexity. mitigation: this is exactly what phase 5a is designed to test. run the experiment, measure BPB, and compare.

## prerequisite knowledge

- [[leaky_integrate_and_fire]] -- the biological model that motivates the leak term
- [[neuron_model_comparison]] -- why LIF is the recommended first step
- [[adaptive_exponential]] -- the model that motivates adaptation as a potential second step

## related bridge docs

(none yet -- this is the first bridge document for single neuron models)

## open questions

1. should the leak rate be per-neuron (current proposal) or per-layer? per-neuron allows heterogeneous time constants (biological) but increases parameters. per-layer is simpler and may suffice.

2. should tau be learnable or fixed? the current proposal makes leak_log learnable. an alternative is to fix alpha = 1/tau with a per-layer tau, which reduces to the original ATMN formulation with proper leak semantics.

3. the sequential processing (option A) creates a fundamental tension with GPU parallelism. is there a parallel scan formulation of the LIF dynamics, analogous to parallel scan for linear recurrences? the LIF with fixed leak rate is a linear recurrence: u_t = (1-alpha)*u_{t-1} + f(x_t). this CAN be parallelized with a prefix sum in O(log T) parallel steps. this would resolve the sequential bottleneck entirely.

4. the training reset currently zeros the membrane potential. an alternative is to DETACH (stop gradients) but not zero it, preserving the state for the next forward pass while preventing gradient flow through the membrane potential across batches. this is a middle ground between full reset and full persistence.
