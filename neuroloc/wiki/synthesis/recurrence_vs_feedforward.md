# recurrence vs feedforward

status: current (as of 2026-04-16).

## the biological principle

cortical computation is not feedforward. the canonical microcircuit (douglas and martin 1989) established that only 5-15% of excitatory synapses on layer 4 neurons come from thalamic afferents -- the remaining 85-95% are recurrent, from other cortical neurons. the cortex does not extract features in a single pass. it amplifies a weak input through recurrent re-processing, with inhibitory feedback controlling gain. the amplification factor 1/(1-g) can reach 5-20x, producing sharp selectivity from blunt inputs.

this recurrence is computational, not merely mnemonic. the [[canonical_microcircuit]] uses recurrent excitation within layer 2/3 to perform winner-take-all competition, gain control, and stimulus selectivity. the [[predictive_coding]] framework reinterprets the same circuit as hierarchical error computation: feedback connections carry predictions, feedforward connections carry prediction errors, and local recurrence computes the difference. in both views, the within-layer recurrent loop does real work -- it transforms the representation, not just preserves it.

three biological functions depend on recurrence specifically:

1. **attractor dynamics.** recurrent circuits settle into stable fixed points that represent discrete decisions (categorical perception, working memory maintenance). the settling process IS the computation -- it maps continuous noisy input to discrete clean output.
2. **error correction.** the [[predictive_coding]] hierarchy subtracts top-down predictions from bottom-up input at each level. the prediction error drives the representation toward a better explanation of the data. this iterative refinement requires recurrence.
3. **temporal integration.** [[leaky_integrate_and_fire]] neurons accumulate input over time through the membrane equation tau_m * dV/dt = -(V - V_rest) + R*I(t). the leak provides exponential forgetting; the integration provides temporal context. this is recurrence at the single-neuron level.

## recurrence in todorov

todorov has recurrence across timesteps but not within layers.

**kda (18/24 layers):** S_t = diag(alpha) * S_{t-1} + beta_t * k_t * v_t^T. the state matrix S_t accumulates key-value associations over the sequence. this is genuine across-timestep recurrence: S_t depends on all past inputs through exponentially decayed contributions. the state carries memory. but within a single timestep, kda is a feedforward computation: project to q/k/v, update state, read out, project back. no iteration. no settling.

**mamba3 (3/24 layers):** h_t = A_bar * h_{t-1} + B_bar * B * x_t, followed by complex rotation. the state evolves as a linear recurrence with data-dependent gating and rotation. again, recurrence across timesteps (h_t depends on h_{t-1}) but feedforward within each timestep.

**mla (3/24 layers):** softmax attention over compressed representations. purely feedforward. no recurrent state at all. the 12.5% mla allocation is non-negotiable -- ssm limitation proofs establish that pure recurrent models cannot match attention on retrieval tasks.

the architecture processes each layer serially: input enters, one feedforward pass transforms it, output goes to the residual stream for the next layer. there are no feedback connections from later layers to earlier layers. no within-layer iteration. no attractor settling. no recurrent amplification in the douglas-martin sense.

## the outer product: genuine hebbian recurrence

the one component where the biological correspondence is mathematically exact is the outer product k_t * v_t^T in the kda state update. this is the classical [[hebbian_learning]] storage prescription, identical to the hopfield network's write rule Delta_W = eta * y * x^T. the identity is not metaphorical -- it is the same matrix operation on the same data structure (an associative weight/state matrix).

the full kda write contains three biologically grounded components:

- **hebbian association** (k_t * v_t^T): one-shot storage of a key-value pair. biologically equivalent to ltp at the synapse between presynaptic (key) and postsynaptic (value) neurons.
- **exponential decay** (diag(alpha) * S_{t-1}): forgetting of old associations. biologically equivalent to [[short_term_plasticity]] depression (vesicle depletion with recovery time constant tau_d). the channel-wise alpha structure creates a bank of temporal filters with different characteristic timescales, analogous to synapses with different tau_d values.
- **data-dependent gating** (beta_t): input-dependent modulation of write strength. biologically closest to neuromodulatory gating (three-factor learning), where a third signal (here, the input itself) gates the hebbian update.

the hebbian outer product is also the weight update rule in predictive coding: dU_l/dt = eta * e_l * r_{l+1}^T. in predictive coding, this learning rule produces exact backpropagation gradients at convergence (millidge et al. 2022). kda uses the same mathematical operation for in-context association storage rather than weight learning, but the structural parallel is real.

what the outer product does NOT provide: error correction, sliding thresholds, homeostatic scaling, or timing-dependent asymmetry (stdp). these are all present in biological plasticity and all absent in kda's state update.

## the delta rule question

the kda state update is called a "delta rule" but may not implement one. the critical difference:

**true delta rule (schlag et al. 2021):**
    S_t = S_{t-1} - beta_t * k_t * (k_t^T * S_{t-1}) + beta_t * k_t * v_t^T

the erasure term k_t * (k_t^T * S_{t-1}) subtracts the old value associated with the current key BEFORE writing the new value. this is targeted forgetting: when a key appears with a new value, the old association at that key is specifically removed. this makes kda an error-correcting memory -- it overwrites rather than accumulates.

**what kda actually does:**
    S_t = diag(alpha) * S_{t-1} + beta_t * k_t * v_t^T

the alpha decay is indiscriminate. it decays everything equally, regardless of whether new information is arriving at that "address." there is no targeted erasure. as [[plasticity_to_kda_delta_rule]] documents: "kda relies on alpha decay for forgetting, which is indiscriminate (decays everything equally) rather than targeted."

the question for chunk_kda (the fla kernel): does the triton implementation include the erasure term? if not, kda is not a delta rule -- it is a decaying hebbian memory. the difference matters computationally: a true delta rule can update associations in place (key A now maps to value B instead of value A). a decaying hebbian memory can only hope that the old association has decayed sufficiently before the new one dominates. for language modeling with repeated entities (pronouns referring to different antecedents, variables reassigned), the distinction is significant.

**design decision:** verify whether chunk_kda implements erasure. if it does not, kda is best described as a leaky hebbian associative memory, not an error-correcting delta rule. the biological analogy shifts from stdp-like error correction to short-term depression with hebbian potentiation.

## the atmn question

atmn is positioned as the biological spike neuron, but its current implementation eliminates the temporal integration that makes biological neurons recurrent.

the core problem identified in [[neuron_models_to_atmn]]: atmn resets membrane potential to zero at the start of every forward pass during training. the membrane potential equation h_t = x_t + (1/tau) * u_{t-1} provides temporal integration in principle, but the training reset means u_{t-1} = 0 always. within a single forward pass, the input tensor is processed as a single timestep (batch*seq_len, features), not sequentially across tokens. atmn during training is therefore a ternary threshold function with no temporal dynamics.

two questions follow:

1. **does the lack of leak matter?** the lif model's leak term -(V - V_rest) drives membrane potential exponentially toward rest. atmn has no equivalent. the (1/tau) scaling on u_{t-1} is a constant factor, not an exponential decay. without leak, sustained input causes unbounded accumulation. the batch reset masks this during training, but at inference on long sequences, drift could occur. [[neuron_models_to_atmn]] proposes a fix: h_t = (1 - alpha) * u_{t-1} + x_t with learnable alpha = sigmoid(leak_log).

2. **can atmn be parallelized with leak?** the lif equation with fixed leak is a linear recurrence: u_t = (1-alpha) * u_{t-1} + f(x_t). linear recurrences admit parallel scan in O(log T) steps. this would resolve the sequential bottleneck that currently makes sequential atmn processing unviable at scale. the parallel scan structure is identical to what mamba uses for its state update -- the machinery exists.

**design decision:** one more run with current atmn, then evaluate whether sequential processing with leak and parallel scan is worth the engineering investment.

## within-layer recurrence

the largest gap between todorov and cortex is the absence of within-layer iteration. biological cortex amplifies weak inputs through recurrent re-processing within layer 2/3 (amplification factor 5-20x). todorov applies each layer exactly once.

a k-iteration scheme would process each layer k times before passing output to the next:

    for i in range(k):
        x = x + layer(x)

at k=1, this is the current architecture. at k=2, each layer sees its own output once and refines it. the biological analog: the recurrent excitatory loop in l2/3 iterates until inhibition stabilizes the response.

potential benefits: attractor dynamics (noisy representations cleaned up by iterative settling), error correction (each iteration reduces the residual between input and the layer's "expected" output), and sharper selectivity (the iceberg effect, where weak responses are suppressed by the iterative amplification of strong responses).

potential costs: k multiplies the compute per layer. at k=2, the model is 2x slower. gradient flow through k iterations of the same weights is equivalent to a k-deep network with shared weights, which can cause vanishing or exploding gradients.

**design decision:** a quick k=2 test at 6m scale to measure whether within-layer iteration improves bpb enough to justify the compute cost. if bpb improves by less than 2%, the compute cost is not justified. if it improves by more than 5%, the biological principle of recurrent amplification has quantitative validation.

## the predictive coding connection

millidge et al. (2022) proved that predictive coding converges to exact backpropagation gradients on arbitrary computation graphs. this means every backprop-trained network -- including todorov -- implicitly computes the weight updates that a predictive coding network would produce at convergence.

the implication: todorov's next-token prediction loss, optimized by backpropagation, implements the same objective function as a hierarchical predictive coding network minimizing prediction errors across layers. the cross-entropy loss at the output is equivalent to the sum of squared prediction errors in a gaussian predictive coding model. the weight gradients are identical.

this does NOT mean todorov implements predictive coding mechanistically. there are no explicit prediction error units, no top-down prediction connections, no within-layer inference dynamics. the [[predictive_coding]] article is precise: "todorov does not implement predictive coding. the architecture uses standard next-token prediction with cross-entropy loss computed at the output." the equivalence is at the level of the optimization objective, not the computational mechanism.

the research question is whether explicit predictive coding mechanisms (layer-wise error computation, top-down predictions) would improve on standard backpropagation training. the theoretical answer is: at convergence, they are identical. the practical answer is that predictive coding requires iterating inference to equilibrium at each layer before learning, which may be computationally expensive and biologically implausible in real-time processing.

there is a subtler connection. the weight update dU_l/dt = eta * e_l * r_{l+1}^T in predictive coding is a hebbian outer product between the prediction error and the higher-level representation. kda's state update k_t * v_t^T is a hebbian outer product between the key and the value. if we interpret the key as encoding "what was surprising about this token" (a kind of prediction error) and the value as encoding "what this token means" (a representation), then kda's write operation structurally parallels predictive coding's learning rule. but this interpretation is post-hoc and untested -- the key projection is not trained to encode prediction error, and no mechanism compares the current input against a top-down prediction.

## challenges and counter-arguments

**1. serial architecture eliminates the need for temporal coordination.**

biological recurrence solves a problem todorov does not have. cortex processes information in parallel across many areas simultaneously, creating interference and routing conflicts that require oscillatory gating and recurrent settling. todorov processes layers serially -- each layer has exclusive access to the residual stream, with no concurrent interference. the [[oscillations_vs_recurrence]] dissenting argument: "the need for temporal coordination arises because biological cortex processes information in parallel across many areas simultaneously. todorov's serial layer stack processes information one layer at a time. there is no interference to manage because no two layers are active simultaneously."

counter to the counter: serial processing eliminates inter-layer interference but does not eliminate intra-layer interference. within a single kda layer, all heads update their state matrices simultaneously. if two heads learn conflicting associations for the same key, there is no settling mechanism to resolve the conflict -- both writes proceed, and downstream layers receive an ambiguous readout. within-layer recurrence could resolve such conflicts.

**2. the "recurrence" in kda and mamba3 may be better understood as memory, not computation.**

biological recurrence (within-layer iteration, attractor settling, predictive coding error minimization) transforms representations. the recurrence in kda and mamba3 preserves information across timesteps without transforming it within a timestep. the state S_t is read passively -- it does not participate in iterative refinement of the current input.

a strict definition of "recurrent computation" requires that the output of a unit feed back as input to the same unit within the current processing step. by this definition, kda is recurrent across timesteps (S_t feeds into S_{t+1}) but not within timesteps. this is closer to an external memory (like a tape) than to cortical recurrence (like an attractor network).

**3. the exponential decay in kda conflates two distinct biological mechanisms.**

the alpha parameter in kda absorbs two functions: short-term depression (activity-dependent synaptic weakening) and working memory decay (loss of maintained representations). biologically these are distinct: stp operates at individual synapses on 200-800 ms timescales, while working memory decay operates at the network level on seconds-to-minutes timescales. conflating them into a single per-channel scalar prevents the model from independently controlling synaptic dynamics and memory duration.

**4. the evidence bar for biological correspondence is not met.**

the triple criterion (math identity + ablation necessity + quantitative prediction) is satisfied only for the outer product: k_t * v_t^T = hebbian storage prescription (math identity), removing the outer product eliminates kda entirely (ablation necessity), and the resulting state matrix behaves as a hopfield-like associative memory (quantitative prediction). no other biological correspondence in the recurrence pathway meets all three criteria. the alpha decay is functionally analogous to stp depression but is not activity-dependent. the mamba3 rotation is mathematically an oscillator but likely serves as positional encoding rather than oscillatory computation. the atmn spike is a ternary threshold but lacks leak, refractory period, and temporal integration during training.

## the honest assessment

todorov is recurrent across timesteps and feedforward within timesteps. this is a legitimate architectural choice -- it enables efficient parallelism during training and avoids the convergence issues of within-layer iteration. but it is not cortical recurrence.

what todorov does well:
- the hebbian outer product in kda is the one mathematically exact biological correspondence. it stores associations in the same way that synaptic plasticity stores correlations.
- the exponential decay provides a biologically motivated forgetting curve, even if it conflates stp and memory decay.
- the hybrid architecture (75% recurrent, 25% attention) mirrors the emerging consensus in the field (kimi, qwen3, olmo) and is backed by ssm limitation proofs.

what todorov does not do:
- within-layer recurrent amplification (the defining computation of the canonical microcircuit)
- attractor dynamics (iterative settling to stable fixed points)
- error correction within layers (predictive coding style subtraction of predictions from inputs)
- dynamic communication gating between layers (oscillatory coherence, ctc)

the gap is real but may not matter for language modeling. the question is empirical: does k=2 within-layer iteration improve bpb? does targeted erasure (true delta rule) outperform indiscriminate decay? does atmn with leak and parallel scan add value over simple ternary thresholds? three experiments, three answers.

## see also

- [[canonical_microcircuit]] -- the biological case for recurrent amplification
- [[predictive_coding]] -- hierarchical error minimization through recurrence
- [[hebbian_learning]] -- the outer product rule underlying kda writes
- [[plasticity_to_kda_delta_rule]] -- detailed analysis of kda's biological correspondence
- [[neuron_models_to_atmn]] -- atmn's departures from biological neuron dynamics
- [[oscillations_vs_recurrence]] -- oscillatory dynamics vs todorov's recurrence
- [[short_term_plasticity]] -- the tsodyks-markram model and its relation to alpha decay
- [[oscillations_to_mamba3_rotation]] -- mamba3 rotation as positional encoding vs oscillation
