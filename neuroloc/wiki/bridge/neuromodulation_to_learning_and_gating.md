# bridge: neuromodulation to todorov learning and gating

status: current (as of 2026-04-16).

## the biological mechanism

four neuromodulatory systems control the meta-parameters of learning in the brain ([[neuromodulatory_framework]], Doya 2002):

- [[dopamine_system]]: reward prediction error (delta). phasic VTA/SNc dopamine encodes r + gamma * V(s') - V(s). gates [[hebbian_learning]] via three-factor rules (Delta_w = eta * pre * post * DA). D1/D2 receptor pathways implement sign-dependent plasticity in striatal direct/indirect circuits.
- [[acetylcholine_system]]: learning rate (eta). basal forebrain ACh switches cortex between encoding mode (high ACh: enhance feedforward, suppress recurrent, increase plasticity) and retrieval mode (low ACh: release recurrent, suppress plasticity). nicotinic enhancement of afferent input + muscarinic suppression of feedback.
- [[norepinephrine_system]]: exploration/exploitation (temperature). LC-NE tonic/phasic modes control neural gain. phasic NE = steep sigmoid = focused, exploitative. tonic NE = shallow sigmoid = broad, exploratory. Aston-Jones and Cohen 2005 adaptive gain theory.
- serotonin: temporal discount (gamma). raphe nuclei 5-HT modulates the time horizon of reward evaluation. high 5-HT = patient, long-horizon. low 5-HT = impulsive, short-horizon. the weakest mapping in the framework.

the common property: each neuromodulator is a GLOBAL, BROADCAST signal that modulates MANY neurons SIMULTANEOUSLY. a single LC neuron projects across multiple cortical areas. VTA dopamine neurons project to all of prefrontal cortex, striatum, and amygdala. this broadcast architecture means neuromodulation is inherently low-dimensional: a small number of scalar signals controlling the operating regime of millions of neurons.

## the current todorov implementation

todorov uses FIXED schedules and LEARNED-BUT-STATIC parameters for all quantities that biology modulates dynamically:

### learning rate: cosine decay with warmup

    lr(t) = lr_max * 0.5 * (1 + cos(pi * (t - warmup) / (total - warmup)))

determined before training begins. no online modulation. no sensitivity to loss landscape, data difficulty, or model state. this is the opposite of cholinergic modulation: biology adjusts learning rate online based on novelty and uncertainty; todorov follows a fixed schedule regardless of what the model is encountering.

### spike threshold alpha: learnable but gradient-optimized

    threshold = alpha * mean(|x|)

alpha is a learnable parameter (init 1.0) optimized by backpropagation. it adapts during TRAINING but is FIXED during inference. there is no RPE signal, no surprise signal, and no uncertainty signal modulating the threshold. the threshold adjusts to minimize cross-entropy loss, which is an offline optimization, not online neuromodulation.

### KDA forgetting rate alpha_log: per-head per-channel, learned by gradients

    alpha = sigmoid(alpha_log)
    S_t = diag(alpha) * S_{t-1} + beta_t * k_t * v_t^T

alpha_log is initialized at N(-2.0, 0.5), giving initial alpha ~ 0.12 (fast forgetting). during training, gradients adjust alpha_log toward whatever value minimizes loss. the resulting alpha is FIXED during inference.

this is the closest analog to neuromodulatory gain control in the architecture:
- each head has its own alpha per channel: [num_heads, head_dim] values
- alpha controls how much old state is retained vs discarded
- different channels can have different decay rates (some memories last longer than others)

BUT the analogy breaks down on three axes:

**(1) granularity mismatch**: biological neuromodulation is GLOBAL. a single cholinergic signal from the basal forebrain modulates entire cortical regions. todorov's alpha is PER-CHANNEL PER-HEAD: num_heads * head_dim values. at 267m scale (16 heads, 64 dim), that is 1024 independent decay rates. no biological neuromodulator achieves this specificity. the closest biological analog for per-channel control is per-synapse short-term plasticity (the Tsodyks-Markram model), which is NOT neuromodulation -- it is a local synaptic property.

**(2) temporal dynamics mismatch**: biological neuromodulation is DYNAMIC. ACh levels change moment-to-moment with arousal, attention, and task demands. NE shifts between tonic and phasic modes on a timescale of seconds. dopamine bursts last 100-500 ms. todorov's alpha is a FIXED PARAMETER after training. it does not change during inference. the decay rate for channel 7 of head 3 is the same whether the model is processing a simple repetitive pattern or a novel complex sentence. biology would MODULATE the decay rate: fast decay for novel input (discard old patterns, encode new), slow decay for familiar input (preserve stored patterns).

**(3) signal source mismatch**: biological neuromodulation is driven by GLOBAL STATE signals: reward history (DA), uncertainty (ACh), surprise (NE), temporal context (5-HT). todorov's alpha is optimized by LOSS GRADIENTS during training. there is no online signal that says "the current input is surprising, adjust alpha." the optimization finds a COMPROMISE alpha that works ON AVERAGE across the training distribution. this is like optimizing the thermostat setting for a building once and never adjusting it -- adequate on average, suboptimal at any given moment.

### KDA write gate beta: sigmoid(beta_proj(x))

    beta_t = sigmoid(beta_proj(x_t))

this IS data-dependent: beta varies per token. beta_proj is a linear projection from x to [num_heads], so each head gets its own write gate computed from the current input.

- high beta = write more to memory (encoding mode -- ACh analog?)
- low beta = preserve existing memory (retrieval mode -- low ACh?)
- input-gated: the decision to write is based on the current token's representation

this is genuinely closer to biological neuromodulation than alpha. beta implements input-dependent gating that could be interpreted as a content-based encoding/retrieval switch. the beta_proj learned weights encode "what kinds of tokens should trigger strong writes" vs "what kinds of tokens should be mostly read from state."

BUT beta is still missing the global context:
- beta sees only the CURRENT TOKEN x_t. it has no access to: running loss, attention entropy, spike density, sequence position beyond what is encoded in x_t via the residual stream
- beta is NOT modulated by a global state signal. there is no "uncertainty estimate" or "surprise signal" that scales beta across all heads simultaneously
- beta is per-head but not per-channel. biological ACh modulates different connection types differently (feedforward vs recurrent). beta modulates all channels within a head equally

### what is completely absent

- **no reward signal**: no RPE, no TD error, no three-factor learning rule. all learning is via backpropagation of cross-entropy loss
- **no surprise signal**: no mechanism to detect that the current input violates model expectations and trigger model reset (the Dayan & Yu 2006 NE interrupt)
- **no gain modulation**: no global signal that adjusts the steepness of activation functions. all sigmoid, SwiGLU, and softmax functions have fixed gain parameters
- **no exploration mechanism**: inference is deterministic given fixed temperature. no adaptive exploration during training or inference
- **no temporal discounting**: all tokens in a training sequence contribute equally to loss. no mechanism to weight recent vs distant tokens differently based on context

## the key question: is alpha a neuromodulatory gain parameter?

### the argument for

alpha controls S_t = diag(alpha) * S_{t-1} + ..., which determines how much influence past tokens have on the current output. this IS a form of gain control:
- high alpha (slow decay): past tokens have high gain, current token has relatively low gain on the state
- low alpha (fast decay): past tokens are rapidly forgotten, current token dominates the state

the per-channel structure means different "feature dimensions" of the associative memory can independently control their temporal integration window. this is richer than a single scalar gain knob.

### the argument against

**alpha is NOT gain in the Aston-Jones sense.** Aston-Jones's gain modulation changes the SLOPE of the sigmoid transfer function, not the decay rate of a recurrent state. gain modulation affects HOW STRONGLY a neuron responds to ALL inputs. alpha affects HOW LONG a specific memory trace persists. these are computationally different:

- gain modulation: y = sigmoid(g * x). changing g changes the input-output relationship for ALL current inputs
- alpha decay: S_t = alpha * S_{t-1} + new. changing alpha changes the MIXTURE of past and present, not the response to present input

the distinction matters. gain modulation is about SENSITIVITY to current input. alpha is about MEMORY PERSISTENCE. a neuromodulatory system that adjusts alpha would be controlling "how much history to remember" -- which maps better onto cholinergic encoding/retrieval switching (high ACh -> low alpha, fast forgetting, encode new; low ACh -> high alpha, slow forgetting, retrieve old) than onto noradrenergic gain modulation.

### the strongest counterargument: Adam already does this

Adam optimizer maintains per-parameter learning rates based on gradient history:

    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    theta_t = theta_{t-1} - lr * m_t / (sqrt(v_t) + epsilon)

the effective learning rate for each parameter is lr / (sqrt(v_t) + epsilon), which is a per-parameter, history-dependent, adaptive learning rate. this is MORE granular than any biological neuromodulatory system:
- Adam: one learning rate per parameter (~300M independent rates at 300M params)
- biological ACh: one learning rate per cortical region (~100 independent signals?)
- todorov's alpha: one decay rate per channel per head (~1024 independent rates at 267m)

Adam already implements a form of metalearning MORE specific than biology achieves. the question is whether ONLINE modulation (during inference, not just training) adds something that offline optimization (during training) cannot capture.

### what online modulation would add that Adam cannot

Adam adjusts parameter update rates during TRAINING. it does not affect INFERENCE. online neuromodulation would modulate alpha and beta DURING INFERENCE, changing the model's behavior based on the current input context:

1. **sequence-dependent memory**: "this part of the text introduces new entities -- increase beta (write strongly)" vs "this part references established entities -- decrease beta (read from state)." a neuromodulator network could learn to detect novelty/familiarity and adjust accordingly.

2. **error-driven reset**: if the model produces a high-loss token (equivalent to negative RPE), a dopamine-like signal could increase beta for the next few tokens ("pay more attention, something unexpected happened"). this is the Dayan & Yu (2006) interrupt.

3. **context-dependent temporal integration**: in a dialogue, recent utterances matter more than distant context. in a technical manual, definitions from 1000 tokens ago are equally relevant. a neuromodulator could adjust alpha based on detected text type or information density.

4. **adaptive gain**: modulating the steepness of the spike threshold function based on global activity statistics. high average activation -> steeper threshold (selective, sparse coding). low average activation -> shallower threshold (preserve information, denser coding).

none of these are possible with fixed alpha/beta or Adam's training-time adaptation.

## the proposed change

a small neuromodulator network that reads global state and modulates alpha and beta:

### architecture

    inputs:
        - running_loss_ema: exponential moving average of per-token loss (scalar)
        - attention_entropy: mean entropy of attention scores across heads (scalar)
        - spike_density: current firing rate from spike layers (scalar)
        - state_norm: ||S_t|| of KDA state (scalar)

    neuromod_net: Linear(4, hidden) -> SiLU -> Linear(hidden, 2)

    outputs:
        - alpha_modulation: scalar in [-1, 1] (added to alpha_log before sigmoid)
        - beta_modulation: scalar in [-1, 1] (added to beta_proj output before sigmoid)

    modulated parameters:
        alpha_effective = sigmoid(alpha_log + alpha_modulation)
        beta_effective = sigmoid(beta_proj(x) + beta_modulation)

### parameter count

with hidden=16: 4*16 + 16 + 16*2 + 2 = 130 parameters per layer. at 24 layers: 3120 total. negligible relative to 300M model parameters.

### biological mapping

- running_loss_ema -> reward prediction error analog (dopamine). high loss = negative RPE = "model is wrong"
- attention_entropy -> uncertainty analog (acetylcholine). high entropy = uncertain attention = "input is ambiguous"
- spike_density -> arousal/activation analog (norepinephrine). firing rate tracks how much information is flowing through the sparse code
- state_norm -> memory load analog. large state = lots of stored associations. informs whether to write (high norm -> maybe slow down) or read (low norm -> not much to retrieve)

### expected behavior

when running_loss_ema is high (model making errors):
- alpha_modulation should decrease (faster forgetting -- the stored state may be misleading)
- beta_modulation should increase (stronger writes -- encode the new, surprising information)

when attention_entropy is high (uncertain processing):
- alpha_modulation should decrease (reset state, don't trust old patterns)
- beta_modulation should increase (write more aggressively to capture the ambiguous input)

when spike_density is low (sparse, selective coding):
- the model is in "exploitation mode" -- few, confident features active
- alpha_modulation should increase (preserve focused state)

when state_norm is high (memory near capacity):
- alpha_modulation should decrease (accelerate forgetting to prevent saturation)
- see the BCM-like proposal in [[plasticity_to_matrix_memory_delta_rule]]

## implementation spec

### in KDALayer.forward (src/layers/kda.py):

new parameter: self.neuromod_net = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 2))

in forward():

    state_norm = state.norm(dim=(-2, -1)).mean() if state is not None else torch.zeros(1, device=x.device)
    neuromod_input = torch.stack([running_loss_ema, attention_entropy, spike_density, state_norm])
    neuromod_output = self.neuromod_net(neuromod_input)
    alpha_mod = torch.tanh(neuromod_output[0])
    beta_mod = torch.tanh(neuromod_output[1])
    alpha = torch.sigmoid(self.alpha_log + alpha_mod)
    beta = torch.sigmoid(self.beta_proj(x) + beta_mod)

### open problems

1. **running_loss_ema during inference**: at training time, the per-token loss is available. at inference time, there is no loss. options: (a) use the model's own confidence (1 - max_prob) as a proxy, (b) use the entropy of the output distribution, (c) train a learned loss predictor. option (b) is most practical but adds a forward pass through the output head.

2. **gradient flow**: the neuromod_net receives gradients from the main loss. this means it will learn to modulate alpha and beta in whatever way reduces cross-entropy loss. there is no guarantee it will learn the "biologically motivated" behavior described above. it might learn something computationally useful but biologically uninterpretable.

3. **training stability**: modulating alpha and beta with a neural network introduces a multiplicative interaction between the neuromod_net and the KDA state dynamics. this could create training instabilities. mitigation: initialize neuromod_net output weights to zero (no modulation initially), clamp modulation magnitude to [-0.5, 0.5].

4. **is this just a fancy adaptive optimizer?** the neuromod_net adjusts alpha and beta DURING THE FORWARD PASS, which means it affects inference-time behavior, not just training dynamics. Adam adjusts parameter updates DURING BACKPROPAGATION and has no effect at inference. these are computationally different. however, if the neuromod_net learns to always output approximately zero (no modulation), then fixed alpha/beta are sufficient and the biological analogy is computationally empty.

## expected impact

**probability of meaningful improvement**: 15-25%.

**rationale for skepticism**: (1) the 4 input features (loss, entropy, spike density, state norm) may not carry enough information to usefully modulate alpha/beta -- the residual stream already encodes rich contextual information that beta_proj can access. (2) the modulation is a GLOBAL scalar applied to all channels uniformly, which is appropriate for biological neuromodulation but may be too coarse for the per-channel structure of alpha. (3) the small neuromod_net (130 params) may not have enough capacity to learn meaningful modulation policies. (4) the benefits of online modulation may be captured approximately by the current learned alpha/beta, which are optimized on the training distribution.

**rationale for optimism**: (1) beta currently has no access to global state -- it only sees x_t through a linear projection. adding loss/entropy/spike information gives it genuinely new signal. (2) the zero-initialization means the default behavior is unchanged; any learned modulation must improve loss to persist. (3) the 130-parameter overhead is negligible. (4) even small modulation effects could compound over long sequences where fixed alpha leads to state saturation or information loss.

## risk assessment

**computational risk**: negligible. 130 parameters, 1 linear + 1 SiLU + 1 linear per layer per token.

**training stability risk**: moderate. mitigated by zero initialization and clamping.

**confound risk**: the neuromod_net shares the same loss and optimizer as the main model. any improvement could be attributed to increased capacity (130 more params) rather than the neuromodulatory structure. control: compare against a baseline where the 130 params are added elsewhere (e.g., slightly wider projection).

**biological plausibility risk**: the proposal uses per-token features to modulate per-layer parameters. this is more temporally granular than biological neuromodulation (which operates on ~100 ms timescales, not per-token). biological neuromodulation is also not driven by "loss gradients" but by higher-order evaluative signals. the mapping is functional, not mechanistic.

## the adversarial verdict

**is KDA's alpha analogous to neuromodulatory gain control?**

no. the analogy fails on three of four dimensions:

1. **granularity**: alpha is per-channel per-head (1024 values at 267m). neuromodulation is a small number of broadcast signals. alpha is more like per-synapse short-term plasticity than like a neuromodulator.

2. **dynamics**: alpha is fixed after training. neuromodulators vary continuously during behavior. this is the most fundamental difference.

3. **signal source**: alpha is optimized offline by loss gradients. neuromodulators are driven online by reward, surprise, and uncertainty. the information sources are different.

4. **functional role**: alpha controls memory persistence (how long associations last in S_t). neuromodulatory gain controls response sensitivity (how strongly neurons respond to input). these are related but distinct computational operations. alpha is closer to the cholinergic encoding/retrieval switch than to noradrenergic gain.

the HONEST summary: alpha is a per-channel exponential decay rate, learned by gradient descent, with no online modulation. calling it "neuromodulatory gain" is a category error. calling it "an analog of cholinergic encoding/retrieval switching" is closer but still misleading because of the fixed-parameter / dynamic-signal mismatch.

**beta** is the stronger neuromodulatory analog. it is data-dependent (changes per token), input-driven (computed from x_t), and functionally equivalent to a write gate (encoding vs preservation). the gap is that beta has no access to global state signals (loss, surprise, uncertainty). the proposed neuromod_net addresses exactly this gap.

**the strongest intervention**: add a neuromodulator network with 130 parameters per layer that provides global-state-dependent modulation of alpha and beta. this is biologically motivated, computationally cheap, and testable. it should be validated in isolation (not combined with other changes) following the phase sequencing protocol. the most informative experiment would compare: (a) baseline (fixed alpha, input-gated beta), (b) neuromod_net modulating alpha only, (c) neuromod_net modulating beta only, (d) neuromod_net modulating both. this 4-way comparison isolates whether the benefit (if any) comes from dynamic alpha, dynamic beta, or their interaction.

## key references

- Doya, K. (2002). Metalearning and neuromodulation. Neural Networks, 15(4-6), 495-506.
- Schultz, W., Dayan, P. & Montague, P. R. (1997). A neural substrate of prediction and reward. Science, 275(5306), 1593-1599.
- Hasselmo, M. E. (2006). The role of acetylcholine in learning and memory. Current Opinion in Neurobiology, 16(6), 710-715.
- Aston-Jones, G. & Cohen, J. D. (2005). An integrative theory of locus coeruleus-norepinephrine function: adaptive gain and optimal performance. Annual Review of Neuroscience, 28, 403-450.
- Dayan, P. & Yu, A. J. (2006). Phasic norepinephrine: a neural interrupt signal for unexpected events. Network, 17(4), 335-350.
- Marder, E. (2012). Neuromodulation of neuronal circuits: back to the future. Neuron, 76(1), 1-11.

## see also

- [[dopamine_system]]
- [[acetylcholine_system]]
- [[norepinephrine_system]]
- [[neuromodulatory_framework]]
- [[precision_weighting]]
- [[plasticity_to_matrix_memory_delta_rule]]
- [[predictive_coding_to_training_objective]]
