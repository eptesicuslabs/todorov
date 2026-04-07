# neuroscience for ML engineers

the 20% of neuroscience that explains 80% of todorov's design. each section maps a biological concept to something you already know (transformers, attention, gradient descent), then shows where biology does something transformers cannot.

---

## part 1: the single neuron (the "unit")

**why this matters for your architecture:** you know what a ReLU unit does: multiply inputs by weights, sum, apply nonlinearity. a biological neuron does the same thing -- but with temporal dynamics that transformers completely lack.

**what you will learn:** the neuron's membrane equation, threshold-and-fire behavior, and how ATMN implements it.

**connect to what you know:** a ReLU unit is stateless. it produces the same output for the same input, regardless of history. a biological neuron is stateful. its output depends on what happened at every previous timestep.

### the membrane potential

the **membrane potential** (V_m) is the neuron's "activation." but unlike a ReLU activation that is computed and discarded, V_m persists across time. it accumulates input. it decays toward a resting value. it is the neuron's memory.

the governing equation:

    C_m * dV/dt = -g_L * (V - V_rest) + I_ext

each term has an ML translation:

- **C_m** (membrane capacitance): how quickly the neuron responds. high capacitance smooths rapid fluctuations, like momentum.
- **g_L** (leak conductance): the exponential decay rate. without input, V_m decays back to V_rest with time constant tau_m = C_m / g_L (typically 10-30 ms in cortex).
- **V_rest** (resting potential): the baseline activation (~-65 mV). the bias the neuron decays toward with no input.
- **I_ext** (external current): the weighted sum of upstream activations.

the discrete-time form (Euler method, one step per token):

    V_{t+1} = (1 - dt/tau_m) * V_t + (dt/tau_m) * (V_rest + R * I_t)

this is a **leaky integrator**. (1 - dt/tau_m) controls how much previous state survives. with tau_m = 20 ms and dt = 1 ms, 95% persists per step. similar to an LSTM forget gate, except the decay rate comes from physics rather than a learned sigmoid.

### threshold and spiking

when V_m crosses a threshold V_th (~-50 mV), the neuron fires a **spike** -- a brief all-or-nothing pulse. V_m then resets to V_reset (near V_rest), and the neuron enters a brief **refractory period** where it cannot fire again. the neuron integrates evidence over time and, when the evidence exceeds threshold, commits to a discrete output.

### ATMN: todorov's neuron model

todorov implements this as the ATMN (adaptive threshold membrane neuron) spike. the membrane update:

    h_t = x_t + (1/tau) * u_{t-1}

where x_t is the input and u_{t-1} is the carried-over membrane potential. the output is ternary ({-1, 0, +1}):

    spikes = sign(h_t) * [|h_t| > V_th]

where V_th = exp(threshold_log) is a per-neuron learnable parameter. after firing, the membrane resets by subtraction: u_t = h_t - spikes * V_th.

this is a simplified [[leaky_integrate_and_fire]] model. the key biological insight it captures: the neuron has MEMORY between inputs. V_m persists across timesteps. a sub-threshold input at time t can combine with a sub-threshold input at time t+1 to produce a spike that neither could produce alone. this temporal integration is something a ReLU or GELU cannot do.

the key simplifications (see [[neuron_models_to_atmn]] for the full analysis): ATMN has no proper leak term (no exponential decay toward rest), and during training, the membrane potential resets to zero each batch. these are active areas of investigation.

---

## part 2: the synapse (the "weight")

**why this matters for your architecture:** in a transformer, weights are fixed at inference. in biology, weights change WHILE the network processes data. this is like having Adam running during inference, not just training.

**what you will learn:** Hebbian learning, the outer product rule, and how KDA implements biological associative storage.

**connect to what you know:** you already know that attention computes Q * K^T to measure similarity, then uses the result to weight V. the biological version of this operation has a name: [[hebbian_learning]].

### the hebbian outer product

a **synapse** connects two neurons. its **synaptic weight** w_ij determines how strongly neuron j influences neuron i. Hebb's rule (1949) is the simplest modification rule:

    Delta_w = eta * x_pre * x_post

"neurons that fire together wire together." for a population, this becomes Delta_W = eta * x_post * x_pre^T -- an **outer product** that creates a rank-1 association between pre and post activity patterns.

the KDA state update in todorov is:

    S_t = diag(alpha) * S_{t-1} + beta_t * k_t * v_t^T

the term k_t * v_t^T is exactly the Hebbian outer product. it creates an association between the key (the "address") and the value (the "content") at each timestep. this is the most direct biological correspondence in todorov's architecture.

### beyond simple hebbian: decay and gating

pure Hebbian learning is unstable. if a synapse strengthens, it makes co-activation more likely, which strengthens it further. weights grow without bound. biology solves this with multiple mechanisms operating at different timescales:

- **[[short_term_plasticity]]**: synaptic efficacy changes on the millisecond timescale. **facilitation** (calcium accumulation) temporarily strengthens a synapse with repeated use. **depression** (vesicle depletion) temporarily weakens it. these act as temporal filters on the signal.
- **[[stdp]]** (spike-timing-dependent plasticity): the sign of the weight change depends on the TIMING of pre and post spikes. if pre fires before post (causal), the synapse strengthens. if post fires before pre (anti-causal), it weakens. this implements a form of causal inference.
- **[[bcm_theory]]**: a sliding threshold determines whether activity leads to strengthening or weakening. the threshold rises with recent activity, preventing runaway growth.

KDA maps to these stabilization mechanisms:

- **alpha** (per-channel forgetting rate): each dimension of the state matrix decays independently. old associations fade. this is closest to synaptic depression -- activity-dependent decay of efficacy. alpha ~ 0.12 initially means ~88% of the state is erased each step.
- **beta_t** (data-dependent write gate): a learned function of the input determines how strongly each token writes to memory. this resembles **neuromodulatory gating** -- a "third factor" beyond pre and post activity that controls whether learning happens.

the key difference from ML: biological weights are LOCAL. each synapse only sees its own pre and post activity. backpropagation requires GLOBAL error signals traveling backward through the entire network. todorov uses backprop for training, side-stepping this problem. but its recurrent state update (the KDA rule) is local -- each state matrix depends only on its own inputs.

---

## part 3: sparsity (the "efficiency trick")

**why this matters for your architecture:** your GPU activates 100% of neurons every forward pass. the brain activates 1-5%. this is not a limitation -- it is the most important design principle in biological computation.

**what you will learn:** why the brain is sparse, why todorov is less sparse, and the tradeoff that connects them.

**connect to what you know:** dropout, mixture-of-experts, pruning -- you already use sparsity. biological sparsity is more extreme and more principled.

### the metabolic constraint

the brain runs on about 20 watts. at any given moment, only 1-5% of cortical neurons are active (Vinje and Gallant, 2000). Levy and Baxter (1996) showed the optimal coding level -- maximizing information per unit energy -- is approximately 6%.

**[[sparse_coding]]** (Olshausen and Field, 1996) formalizes this: represent each input with a small number of active neurons from a large population. three benefits:

1. **energy efficiency**: fewer spikes means less metabolic cost.
2. **noise robustness**: sparse codes are far apart in Hamming distance, so random noise is unlikely to push one code into another.
3. **memory capacity**: for an associative memory storing patterns via outer products, capacity scales inversely with the square of the firing rate. sparser codes store more patterns before interference corrupts retrieval.

### todorov's ternary spikes

todorov quantizes activations to {-1, 0, +1}. with adaptive threshold alpha * mean(|x|) at alpha = 1.0, about 41% of dimensions are nonzero. this is far denser than the biological 1-5%.

the reason: the **straight-through estimator** (STE). the STE passes gradients through quantization as if it were the identity. at 5% firing rate, 95% of dimensions contribute near-zero gradients. training becomes slow and unstable. dead neurons never receive gradient signal to change. the validated range is 20-60%.

the tradeoff is clear:

- **biological sparsity** (1-5%): optimized for energy and memory capacity. possible because the brain does not use backpropagation. learning rules are local and do not require gradient flow through the nonlinearity.
- **todorov sparsity** (41%): optimized for gradient-based training stability. provides compute reduction (ternary multiply-adds) and acts as a regularizer, but sacrifices the capacity benefits of extreme sparsity.

spike health is monitored by three metrics: **mutual information** (MI > 0.1: how much input information survives quantization), **centered kernel alignment** (CKA > 0.3: whether representation geometry is preserved), and **firing rate** (target 20-60%). at 267 million parameters, todorov achieves MI = 1.168, CKA = 0.732, firing rate = 40.8%.

see [[sparse_coding_to_ternary_spikes]] for the full analysis and [[efficient_coding]] for the information-theoretic foundations.

---

## part 4: memory and learning (the "two systems")

**why this matters for your architecture:** you have a KV cache for exact retrieval and a recurrent state for compressed memory. the brain has the same split: hippocampus for fast binding, cortex for slow learning.

**what you will learn:** complementary learning systems theory, the Hopfield network connection, and how KDA/MLA implement the two-system architecture.

**connect to what you know:** a transformer's KV cache stores every token's key and value vectors. retrieval is exact but costs O(T^2) in sequence length T. a recurrent state (like an LSTM or SSM hidden state) compresses all of history into a fixed-size vector. retrieval is O(1) but lossy. the brain faces the same tradeoff.

### complementary learning systems

McClelland, McNaughton, and O'Reilly (1995) proposed two complementary memory systems:

- the **hippocampus**: fast, episodic, pattern-separated. stores individual experiences quickly via rapid LTP. sparse codes (~2-5% firing) keep memories distinct. capacity is limited by interference.
- the **neocortex**: slow, statistical, overlapping. extracts shared structure across many experiences. small per-experience modification prevents **catastrophic interference** (new learning destroying old knowledge).

**[[memory_consolidation]]** bridges them: during sleep, the hippocampus replays recently stored experiences, providing interleaved training data to the cortex.

### the hopfield network connection

a **Hopfield network** is an associative memory that stores patterns via outer products (the Hebbian rule) and retrieves them via pattern completion. Ramsauer et al. (2021) proved that modern Hopfield networks are mathematically equivalent to softmax attention. the attention operation IS pattern completion in an associative memory.

this creates a precise mapping:

- **KDA** = classical Hopfield network. S_t = alpha * S_{t-1} + beta_t * k_t * v_t^T writes rank-1 patterns. q_t^T * S_t retrieves by linear projection. capacity limited by rank of S. alpha decay implements exponential forgetting.
- **MLA** = modern Hopfield network (softmax attention). every token stored in compressed KV cache. capacity exponential in dimension. no forgetting.

the 75/25 split (18 KDA, 6 MLA) mirrors the biological asymmetry: most processing uses the cheap approximate system (KDA, O(1)). a minority uses the expensive exact system (MLA, O(T^2)).

what todorov LACKS: consolidation. no mechanism transfers salient KDA state to permanent storage. in the brain, hippocampal replay during sleep provides this. not a limitation at current context lengths (2048-8192 tokens). would become a gap at 100K+.

see [[memory_systems_to_kda_mla]] for the full analysis and [[complementary_learning_systems]] for the biological theory.

---

## part 5: oscillations and coordination (the "clock")

**why this matters for your architecture:** transformers process all tokens in parallel. the brain uses oscillations to coordinate which neurons talk to which, and when. this is the biggest gap in current architectures.

**what you will learn:** gamma and theta oscillations, communication through coherence, and how Mamba3's rotation relates (and does not relate) to biological rhythms.

**connect to what you know:** you know positional encoding -- RoPE rotates query and key vectors by an angle proportional to position. biological oscillations are also rotations. but they serve a fundamentally different purpose: gating communication between brain regions.

### gamma oscillations

**[[gamma_oscillations]]** (30-100 Hz) arise from excitatory-inhibitory interaction in local circuits. they create periodic windows (~10-30 ms) during which neurons can fire together. neurons within the same gamma window are "bound." neurons in different windows are kept separate. this implements temporal multiplexing, like time-division multiplexing in telecommunications.

### theta oscillations and nesting

**[[theta_oscillations]]** (4-8 Hz) are slower rhythms prominent in the hippocampus during memory encoding and retrieval. each theta cycle (~125-250 ms) contains approximately 5-8 gamma cycles. this creates a nested temporal hierarchy: theta provides the "sentence" structure, gamma provides the "word" structure.

**theta-gamma coupling** allows the brain to maintain a sequence of 5-8 items in working memory, each bound to a different gamma cycle within a single theta cycle. this is the neural basis of the "magical number seven" in human working memory capacity.

### communication through coherence

Fries (2015) proposed **communication through coherence** (CTC): two brain regions communicate effectively only when their oscillations are phase-aligned. if they are in phase, spikes arrive when the receiving region is maximally excitable. if out of phase, spikes arrive during inhibition and are suppressed. the brain dynamically routes information by adjusting phase relationships -- like a software-defined network where hardware is fixed but the routing table updates in real time.

### Mamba3's complex rotation

todorov's Mamba3 layers apply a rotation to the recurrent state at every timestep:

    angle = rope_freq * t
    rotated = complex_multiply(state, exp(i * angle))

each of the 16 state dimension pairs rotates at its own learned frequency. mathematically an oscillation. functionally, most likely **positional encoding** for the recurrent state -- encoding "how many timesteps ago was this stored" via phase, like RoPE encodes position in attention. the frequencies are fixed after training, do not modulate with input, and have no cross-frequency coupling or inter-layer coherence.

the honest assessment: Mamba3 rotation captures the MATHEMATICAL form of oscillations (rotation at a learned frequency) but not their COMPUTATIONAL function (dynamic routing, temporal multiplexing, binding). phase-dependent gating between regions -- the most powerful oscillatory mechanism -- has no analog in todorov.

see [[oscillations_to_mamba3_rotation]] for the full analysis, including proposed modifications and why most of them are not recommended for language modeling.

---

## part 6: predictive coding (the "training objective")

**why this matters for your architecture:** next-token prediction works. but the brain does something more principled: it predicts everything, everywhere, all at once.

**what you will learn:** the predictive coding framework, precision weighting, and why todorov's backprop-trained weights may already approximate what predictive coding would learn.

**connect to what you know:** you train with cross-entropy loss on the next token. the brain trains each cortical layer to predict the layer below it. the error signals are local, not global.

### the prediction error hierarchy

Rao and Ballard (1999) proposed [[predictive_coding]]: each level of the cortical hierarchy maintains a generative model that predicts activity at the level below. the prediction is sent downward. only the **prediction error** -- the difference between the actual and predicted activity -- propagates upward. the mathematical formulation:

    error_l = actual_l - predicted_l
    predicted_l = f(representation_{l+1})

each level receives only the SURPRISE from the level below. expected input is suppressed. this means the brain spends most of its bandwidth communicating what it does NOT expect, rather than what it does. this is an efficient coding strategy: predictable signals carry no new information.

### precision weighting

not all prediction errors are equally reliable. **[[precision_weighting]]** modulates how much the system trusts each error signal. high-precision errors drive large updates; low-precision errors are down-weighted. Friston (2005, 2010) showed this is equivalent to Bayesian inference under the [[free_energy_principle]]. in ML terms, precision weighting is learned loss scaling per feature dimension.

### the key convergence result

Millidge, Tschantz, and Buckley (2022) proved that predictive coding at convergence computes the SAME parameter gradients as backpropagation. this means:

1. todorov's backprop-trained weights already approximate what a predictive coding network would learn.
2. the difference is in FORWARD DYNAMICS (inference), not weight learning.
3. predictive coding networks process input through iterative settling (multiple forward-backward passes per input). todorov does one forward pass.

the practical implication: no reason to switch todorov from cross-entropy to predictive coding. the weights converge to the same solution. the biological advantage (local learning rules) is irrelevant when GPUs compute global gradients efficiently.

where todorov DOES echo predictive coding: the KDA beta gate (beta_t = sigmoid(beta_proj(x_t))) modulates how strongly each token writes to memory. high beta = "surprising, worth encoding." low beta = "expected, safely ignored." this resembles precision weighting, though it is computed from the input alone rather than from prediction error.

see [[predictive_coding_to_training_objective]] for the full analysis of this correspondence and its limits.

---

## part 7: what matters most for todorov

here are the five biological insights with the highest impact on todorov's architecture, ranked by how directly they influenced the design:

**1. recurrent associative memory.** KDA's outer product k_t * v_t^T IS the Hebbian storage rule. this is not an analogy. it is the same equation that Hopfield used in 1982, that the hippocampus uses for rapid binding, and that Hebb described in 1949. the alpha decay and beta gating are biological stabilization mechanisms (short-term depression and neuromodulatory gating) mapped to learned parameters. this is the strongest biology-to-architecture correspondence in todorov.

**2. sparsity as a design principle.** ternary spikes ({-1, 0, +1}) enforce a 41% firing rate. this is denser than biology (1-5%) due to the STE gradient constraint, but it still provides quantization from 32-bit floats to 1.58 bits per dimension. sparsity acts as both compute reduction and regularization. the adaptive threshold (alpha * mean(|x|)) is a crude form of the divisive normalization found in every cortical area.

**3. adaptive thresholds.** biological neurons adjust their excitability based on recent activity (homeostatic plasticity). todorov's per-neuron learnable thresholds (V_th = exp(threshold_log) in ATMN) serve a similar function: each dimension learns its own operating point. at 267M scale, this produces stable spike health metrics (MI = 1.168, CKA = 0.732, firing rate = 40.8%) across training.

**4. complementary memory systems.** the KDA/MLA split mirrors the hippocampus/cortex split. KDA provides O(1) recurrent memory with exponential decay (working memory). MLA provides O(T^2) exact retrieval from compressed cache (reference memory). the 75/25 ratio allocates most computation to the cheap system, with a minority for exact retrieval when needed.

**5. the missing pieces.** three biological principles have no counterpart in todorov:
- **oscillatory coordination**: the brain routes information dynamically through phase synchronization. todorov's Mamba3 rotation is positional encoding, not communication gating.
- **consolidation**: the brain transfers memories from hippocampus to cortex via replay. todorov has no mechanism to transfer salient KDA state to permanent storage.
- **dendritic computation**: biological neurons compute nonlinear functions on individual dendritic branches before summing at the cell body. todorov neurons receive a single weighted sum, like a perceptron.

these gaps are not necessarily problems. they are the places where the next round of architecture improvements might come from, if the biological analogy holds at scale.

the rest of the wiki goes deep on each of these. start with the [[bridge]] documents -- they map each biological mechanism to specific todorov source code.
