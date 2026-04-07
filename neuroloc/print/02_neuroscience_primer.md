# Neuroscience for ML Engineers

The 20% of neuroscience that explains 80% of Todorov's design. Each section maps a biological concept to something you already know (transformers, attention, gradient descent), then shows where biology does something transformers cannot.

---

## Part 1: The Single Neuron (The "Unit")

**Why this matters for your architecture:** you know what a ReLU unit does: multiply inputs by weights, sum, apply nonlinearity. A biological neuron does the same thing -- but with temporal dynamics that transformers completely lack.

**What you will learn:** the neuron's membrane equation, threshold-and-fire behavior, and how ATMN implements it.

**Connect to what you know:** a ReLU unit is stateless. It produces the same output for the same input, regardless of history. A biological neuron is stateful. Its output depends on what happened at every previous timestep.

### The Membrane Potential

The **membrane potential** (V_m) is the neuron's "activation." But unlike a ReLU activation that is computed and discarded, V_m persists across time. It accumulates input. It decays toward a resting value. It is the neuron's memory.

The governing equation:

    C_m * dV/dt = -g_L * (V - V_rest) + I_ext

Each term has an ML translation:

- **C_m** (membrane capacitance): how quickly the neuron responds. High capacitance smooths rapid fluctuations, like momentum.
- **g_L** (leak conductance): the exponential decay rate. Without input, V_m decays back to V_rest with time constant tau_m = C_m / g_L (typically 10-30 ms in cortex).
- **V_rest** (resting potential): the baseline activation (~-65 mV). The bias the neuron decays toward with no input.
- **I_ext** (external current): the weighted sum of upstream activations.

The discrete-time form (Euler method, one step per token):

    V_{t+1} = (1 - dt/tau_m) * V_t + (dt/tau_m) * (V_rest + R * I_t)

This is a **leaky integrator**. (1 - dt/tau_m) controls how much previous state survives. With tau_m = 20 ms and dt = 1 ms, 95% persists per step. Similar to an LSTM forget gate, except the decay rate comes from physics rather than a learned sigmoid.

### Threshold and Spiking

When V_m crosses a threshold V_th (~-50 mV), the neuron fires a **spike** -- a brief all-or-nothing pulse. V_m then resets to V_reset (near V_rest), and the neuron enters a brief **refractory period** where it cannot fire again. The neuron integrates evidence over time and, when the evidence exceeds threshold, commits to a discrete output.

### ATMN: Todorov's Neuron Model

Todorov implements this as the ATMN (adaptive threshold membrane neuron) spike. The membrane update:

    h_t = x_t + (1/tau) * u_{t-1}

where x_t is the input and u_{t-1} is the carried-over membrane potential. The output is ternary ({-1, 0, +1}):

    spikes = sign(h_t) * [|h_t| > V_th]

where V_th = exp(threshold_log) is a per-neuron learnable parameter. After firing, the membrane resets by subtraction: u_t = h_t - spikes * V_th.

This is a simplified leaky integrate-and-fire model. The key biological insight it captures: the neuron has MEMORY between inputs. V_m persists across timesteps. A sub-threshold input at time t can combine with a sub-threshold input at time t+1 to produce a spike that neither could produce alone. This temporal integration is something a ReLU or GELU cannot do.

The key simplifications: ATMN has no proper leak term (no exponential decay toward rest), and during training, the membrane potential resets to zero each batch. These are active areas of investigation.

---

## Part 2: The Synapse (The "Weight")

**Why this matters for your architecture:** in a transformer, weights are fixed at inference. In biology, weights change WHILE the network processes data. This is like having Adam running during inference, not just training.

**What you will learn:** Hebbian learning, the outer product rule, and how KDA implements biological associative storage.

**Connect to what you know:** you already know that attention computes Q * K^T to measure similarity, then uses the result to weight V. The biological version of this operation is Hebbian learning.

### The Hebbian Outer Product

A **synapse** connects two neurons. Its **synaptic weight** w_ij determines how strongly neuron j influences neuron i. Hebb's rule (1949) is the simplest modification rule:

    Delta_w = eta * x_pre * x_post

"Neurons that fire together wire together." For a population, this becomes Delta_W = eta * x_post * x_pre^T -- an **outer product** that creates a rank-1 association between pre and post activity patterns.

The KDA state update in Todorov is:

    S_t = diag(alpha) * S_{t-1} + beta_t * k_t * v_t^T

The term k_t * v_t^T is exactly the Hebbian outer product. It creates an association between the key (the "address") and the value (the "content") at each timestep. This is the most direct biological correspondence in Todorov's architecture.

### Beyond Simple Hebbian: Decay and Gating

Pure Hebbian learning is unstable. If a synapse strengthens, it makes co-activation more likely, which strengthens it further. Weights grow without bound. Biology solves this with multiple mechanisms operating at different timescales:

- **Short-term plasticity**: synaptic efficacy changes on the millisecond timescale. **Facilitation** (calcium accumulation) temporarily strengthens a synapse with repeated use. **Depression** (vesicle depletion) temporarily weakens it. These act as temporal filters on the signal.
- **STDP** (spike-timing-dependent plasticity): the sign of the weight change depends on the TIMING of pre and post spikes. If pre fires before post (causal), the synapse strengthens. If post fires before pre (anti-causal), it weakens. This implements a form of causal inference.
- **BCM theory**: a sliding threshold determines whether activity leads to strengthening or weakening. The threshold rises with recent activity, preventing runaway growth.

KDA maps to these stabilization mechanisms:

- **alpha** (per-channel forgetting rate): each dimension of the state matrix decays independently. Old associations fade. This is closest to synaptic depression -- activity-dependent decay of efficacy. alpha ~ 0.12 initially means ~88% of the state is erased each step.
- **beta_t** (data-dependent write gate): a learned function of the input determines how strongly each token writes to memory. This resembles **neuromodulatory gating** -- a "third factor" beyond pre and post activity that controls whether learning happens.

The key difference from ML: biological weights are LOCAL. Each synapse only sees its own pre and post activity. Backpropagation requires GLOBAL error signals traveling backward through the entire network. Todorov uses backprop for training, side-stepping this problem. But its recurrent state update (the KDA rule) is local -- each state matrix depends only on its own inputs.

---

## Part 3: Sparsity (The "Efficiency Trick")

**Why this matters for your architecture:** your GPU activates 100% of neurons every forward pass. The brain activates 1-5%. This is not a limitation -- it is the most important design principle in biological computation.

**What you will learn:** why the brain is sparse, why Todorov is less sparse, and the tradeoff that connects them.

**Connect to what you know:** dropout, mixture-of-experts, pruning -- you already use sparsity. Biological sparsity is more extreme and more principled.

### The Metabolic Constraint

The brain runs on about 20 watts. At any given moment, only 1-5% of cortical neurons are active (Vinje and Gallant, 2000). Levy and Baxter (1996) showed the optimal coding level -- maximizing information per unit energy -- is approximately 6%.

**Sparse coding** (Olshausen and Field, 1996) formalizes this: represent each input with a small number of active neurons from a large population. Three benefits:

1. **Energy efficiency**: fewer spikes means less metabolic cost.
2. **Noise robustness**: sparse codes are far apart in Hamming distance, so random noise is unlikely to push one code into another.
3. **Memory capacity**: for an associative memory storing patterns via outer products, capacity scales inversely with the square of the firing rate. Sparser codes store more patterns before interference corrupts retrieval.

### Todorov's Ternary Spikes

Todorov quantizes activations to {-1, 0, +1}. With adaptive threshold alpha * mean(|x|) at alpha = 1.0, about 41% of dimensions are nonzero. This is far denser than the biological 1-5%.

The reason: the **straight-through estimator** (STE). The STE passes gradients through quantization as if it were the identity. At 5% firing rate, 95% of dimensions contribute near-zero gradients. Training becomes slow and unstable. Dead neurons never receive gradient signal to change. The validated range is 20-60%.

The tradeoff is clear:

- **Biological sparsity** (1-5%): optimized for energy and memory capacity. Possible because the brain does not use backpropagation. Learning rules are local and do not require gradient flow through the nonlinearity.
- **Todorov sparsity** (41%): optimized for gradient-based training stability. Provides compute reduction (ternary multiply-adds) and acts as a regularizer, but sacrifices the capacity benefits of extreme sparsity.

Spike health is monitored by three metrics: **mutual information** (MI > 0.1: how much input information survives quantization), **centered kernel alignment** (CKA > 0.3: whether representation geometry is preserved), and **firing rate** (target 20-60%). At 267 million parameters, Todorov achieves MI = 1.168, CKA = 0.732, firing rate = 40.8%.

---

## Part 4: Memory and Learning (The "Two Systems")

**Why this matters for your architecture:** you have a KV cache for exact retrieval and a recurrent state for compressed memory. The brain has the same split: hippocampus for fast binding, cortex for slow learning.

**What you will learn:** complementary learning systems theory, the Hopfield network connection, and how KDA/MLA implement the two-system architecture.

**Connect to what you know:** a transformer's KV cache stores every token's key and value vectors. Retrieval is exact but costs O(T^2) in sequence length T. A recurrent state (like an LSTM or SSM hidden state) compresses all of history into a fixed-size vector. Retrieval is O(1) but lossy. The brain faces the same tradeoff.

### Complementary Learning Systems

McClelland, McNaughton, and O'Reilly (1995) proposed two complementary memory systems:

- The **hippocampus**: fast, episodic, pattern-separated. Stores individual experiences quickly via rapid LTP. Sparse codes (~2-5% firing) keep memories distinct. Capacity is limited by interference.
- The **neocortex**: slow, statistical, overlapping. Extracts shared structure across many experiences. Small per-experience modification prevents **catastrophic interference** (new learning destroying old knowledge).

**Memory consolidation** bridges them: during sleep, the hippocampus replays recently stored experiences, providing interleaved training data to the cortex.

### The Hopfield Network Connection

A **Hopfield network** is an associative memory that stores patterns via outer products (the Hebbian rule) and retrieves them via pattern completion. Ramsauer et al. (2021) proved that modern Hopfield networks are mathematically equivalent to softmax attention. The attention operation IS pattern completion in an associative memory.

This creates a precise mapping:

- **KDA** = classical Hopfield network. S_t = alpha * S_{t-1} + beta_t * k_t * v_t^T writes rank-1 patterns. q_t^T * S_t retrieves by linear projection. Capacity limited by rank of S. alpha decay implements exponential forgetting.
- **MLA** = modern Hopfield network (softmax attention). Every token stored in compressed KV cache. Capacity exponential in dimension. No forgetting.

The 75/25 split (18 KDA, 6 MLA) mirrors the biological asymmetry: most processing uses the cheap approximate system (KDA, O(1)). A minority uses the expensive exact system (MLA, O(T^2)).

What Todorov LACKS: consolidation. No mechanism transfers salient KDA state to permanent storage. In the brain, hippocampal replay during sleep provides this. Not a limitation at current context lengths (2048-8192 tokens). Would become a gap at 100K+.

---

## Part 5: Oscillations and Coordination (The "Clock")

**Why this matters for your architecture:** transformers process all tokens in parallel. The brain uses oscillations to coordinate which neurons talk to which, and when. This is the biggest gap in current architectures.

**What you will learn:** gamma and theta oscillations, communication through coherence, and how Mamba3's rotation relates (and does not relate) to biological rhythms.

**Connect to what you know:** you know positional encoding -- RoPE rotates query and key vectors by an angle proportional to position. Biological oscillations are also rotations. But they serve a fundamentally different purpose: gating communication between brain regions.

### Gamma Oscillations

**Gamma oscillations** (30-100 Hz) arise from excitatory-inhibitory interaction in local circuits. They create periodic windows (~10-30 ms) during which neurons can fire together. Neurons within the same gamma window are "bound." Neurons in different windows are kept separate. This implements temporal multiplexing, like time-division multiplexing in telecommunications.

### Theta Oscillations and Nesting

**Theta oscillations** (4-8 Hz) are slower rhythms prominent in the hippocampus during memory encoding and retrieval. Each theta cycle (~125-250 ms) contains approximately 5-8 gamma cycles. This creates a nested temporal hierarchy: theta provides the "sentence" structure, gamma provides the "word" structure.

**Theta-gamma coupling** allows the brain to maintain a sequence of 5-8 items in working memory, each bound to a different gamma cycle within a single theta cycle. This is the neural basis of the "magical number seven" in human working memory capacity.

### Communication Through Coherence

Fries (2015) proposed **communication through coherence** (CTC): two brain regions communicate effectively only when their oscillations are phase-aligned. If they are in phase, spikes arrive when the receiving region is maximally excitable. If out of phase, spikes arrive during inhibition and are suppressed. The brain dynamically routes information by adjusting phase relationships -- like a software-defined network where hardware is fixed but the routing table updates in real time.

### Mamba3's Complex Rotation

Todorov's Mamba3 layers apply a rotation to the recurrent state at every timestep:

    angle = rope_freq * t
    rotated = complex_multiply(state, exp(i * angle))

Each of the 16 state dimension pairs rotates at its own learned frequency. Mathematically an oscillation. Functionally, most likely **positional encoding** for the recurrent state -- encoding "how many timesteps ago was this stored" via phase, like RoPE encodes position in attention. The frequencies are fixed after training, do not modulate with input, and have no cross-frequency coupling or inter-layer coherence.

The honest assessment: Mamba3 rotation captures the MATHEMATICAL form of oscillations (rotation at a learned frequency) but not their COMPUTATIONAL function (dynamic routing, temporal multiplexing, binding). Phase-dependent gating between regions -- the most powerful oscillatory mechanism -- has no analog in Todorov.

---

## Part 6: Predictive Coding (The "Training Objective")

**Why this matters for your architecture:** next-token prediction works. But the brain does something more principled: it predicts everything, everywhere, all at once.

**What you will learn:** the predictive coding framework, precision weighting, and why Todorov's backprop-trained weights may already approximate what predictive coding would learn.

**Connect to what you know:** you train with cross-entropy loss on the next token. The brain trains each cortical layer to predict the layer below it. The error signals are local, not global.

### The Prediction Error Hierarchy

Rao and Ballard (1999) proposed predictive coding: each level of the cortical hierarchy maintains a generative model that predicts activity at the level below. The prediction is sent downward. Only the **prediction error** -- the difference between the actual and predicted activity -- propagates upward. The mathematical formulation:

    error_l = actual_l - predicted_l
    predicted_l = f(representation_{l+1})

Each level receives only the SURPRISE from the level below. Expected input is suppressed. The brain spends most of its bandwidth communicating what it does NOT expect, rather than what it does. This is an efficient coding strategy: predictable signals carry no new information.

### Precision Weighting

Not all prediction errors are equally reliable. **Precision weighting** modulates how much the system trusts each error signal. High-precision errors drive large updates; low-precision errors are down-weighted. Friston (2005, 2010) showed this is equivalent to Bayesian inference under the free energy principle. In ML terms, precision weighting is learned loss scaling per feature dimension.

### The Key Convergence Result

Millidge, Tschantz, and Buckley (2022) proved that predictive coding at convergence computes the SAME parameter gradients as backpropagation. This means:

1. Todorov's backprop-trained weights already approximate what a predictive coding network would learn.
2. The difference is in FORWARD DYNAMICS (inference), not weight learning.
3. Predictive coding networks process input through iterative settling (multiple forward-backward passes per input). Todorov does one forward pass.

The practical implication: no reason to switch Todorov from cross-entropy to predictive coding. The weights converge to the same solution. The biological advantage (local learning rules) is irrelevant when GPUs compute global gradients efficiently.

Where Todorov DOES echo predictive coding: the KDA beta gate (beta_t = sigmoid(beta_proj(x_t))) modulates how strongly each token writes to memory. High beta = "surprising, worth encoding." Low beta = "expected, safely ignored." This resembles precision weighting, though it is computed from the input alone rather than from prediction error.

---

## Part 7: What Matters Most for Todorov

The five biological insights with the highest impact on Todorov's architecture, ranked by how directly they influenced the design:

**1. Recurrent associative memory.** KDA's outer product k_t * v_t^T IS the Hebbian storage rule. This is not an analogy. It is the same equation that Hopfield used in 1982, that the hippocampus uses for rapid binding, and that Hebb described in 1949. The alpha decay and beta gating are biological stabilization mechanisms (short-term depression and neuromodulatory gating) mapped to learned parameters. This is the strongest biology-to-architecture correspondence in Todorov.

**2. Sparsity as a design principle.** Ternary spikes ({-1, 0, +1}) enforce a 41% firing rate. This is denser than biology (1-5%) due to the STE gradient constraint, but it still provides quantization from 32-bit floats to 1.58 bits per dimension. Sparsity acts as both compute reduction and regularization. The adaptive threshold (alpha * mean(|x|)) is a crude form of the divisive normalization found in every cortical area.

**3. Adaptive thresholds.** Biological neurons adjust their excitability based on recent activity (homeostatic plasticity). Todorov's per-neuron learnable thresholds (V_th = exp(threshold_log) in ATMN) serve a similar function: each dimension learns its own operating point. At 267M scale, this produces stable spike health metrics (MI = 1.168, CKA = 0.732, firing rate = 40.8%) across training.

**4. Complementary memory systems.** The KDA/MLA split mirrors the hippocampus/cortex split. KDA provides O(1) recurrent memory with exponential decay (working memory). MLA provides O(T^2) exact retrieval from compressed cache (reference memory). The 75/25 ratio allocates most computation to the cheap system, with a minority for exact retrieval when needed.

**5. The missing pieces.** Three biological principles have no counterpart in Todorov:
- **Oscillatory coordination**: the brain routes information dynamically through phase synchronization. Todorov's Mamba3 rotation is positional encoding, not communication gating.
- **Consolidation**: the brain transfers memories from hippocampus to cortex via replay. Todorov has no mechanism to transfer salient KDA state to permanent storage.
- **Dendritic computation**: biological neurons compute nonlinear functions on individual dendritic branches before summing at the cell body. Todorov neurons receive a single weighted sum, like a perceptron.

These gaps are not necessarily problems. They are the places where the next round of architecture improvements might come from, if the biological analogy holds at scale.
