# mathematical foundations

status: definitional. last fact-checked 2026-04-16.

a math primer for ML engineers entering computational neuroscience. every equation here connects to something you already know from machine learning. the goal is not exhaustive coverage but the 20% of neuroscience math that explains 80% of todorov's mechanisms.

## differential equations for neurons

you already know discrete updates. every recurrent neural network computes:

    h_t = f(h_{t-1}, x_t)

biological neurons use continuous-time versions of the same thing. the difference: instead of stepping from t to t+1, the neuron's state evolves according to a differential equation that describes how voltage changes at each instant.

### the leaky integrate-and-fire equation

the simplest useful neuron model is the [[leaky_integrate_and_fire]] (LIF). it describes a neuron's membrane potential V over time:

    C_m * dV/dt = -g_L * (V - V_rest) + I_ext

each term maps to an ML concept:

- **C_m** (membrane capacitance, measured in picofarads): controls how fast voltage changes for a given input. smaller C_m means faster response. ML analog: like a learning rate for the membrane potential. large C_m = slow updates, small C_m = fast updates.

- **g_L** (leak conductance, measured in nanosiemens): controls how fast voltage drifts back toward rest. the neuron "forgets" its accumulated input. ML analog: the decay factor in an exponential moving average. high g_L = fast forgetting, low g_L = long memory.

- **V_rest** (resting potential, about -65 mV): the baseline voltage when no input arrives. the neuron returns to this value when left alone. ML analog: a bias term that the state decays toward.

- **I_ext** (external current, measured in picoamperes): the input signal. in a network, this is the weighted sum of presynaptic spikes. ML analog: the linear combination W*x that feeds into a neuron.

- **the spike rule**: when V exceeds a threshold V_th (about -50 mV), the neuron fires a spike and resets to V_reset (about -70 mV). ML analog: a step function followed by a state reset.

### a worked example

consider a LIF neuron with C_m = 20 pF, g_L = 10 nS, V_rest = -65 mV, receiving constant current I_ext = 300 pA.

at steady state, dV/dt = 0, so:

    0 = -g_L * (V_ss - V_rest) + I_ext
    V_ss = V_rest + I_ext / g_L
    V_ss = -65 + 300/10 = -65 + 30 = -35 mV

this exceeds any reasonable threshold (typically -50 mV), so the neuron fires repeatedly. the time constant tau_m = C_m / g_L = 20/10 = 2 ms tells you how fast V reaches steady state: after one tau_m, it covers 63% of the gap.

### the discrete approximation

for simulation (and for understanding the connection to ML), discretize using the Euler method:

    V[t+1] = V[t] + (dt/C_m) * (-g_L * (V[t] - V_rest) + I[t])

this is just: **new_state = old_state + step_size * (leak + input)**. identical in structure to a recurrent update with learnable decay.

rearrange the terms to make the ML connection explicit:

    V[t+1] = (1 - dt*g_L/C_m) * V[t] + (dt*g_L/C_m) * V_rest + (dt/C_m) * I[t]

the factor (1 - dt*g_L/C_m) = (1 - dt/tau_m) is the decay coefficient. with dt = 1 ms and tau_m = 2 ms, the decay is 0.5: the neuron retains half its previous voltage at each step.

### the ATMN equation

todorov's ATMN spike neuron (see [[neuron_models_to_atmn]]) uses a simpler update:

    h_t = x_t + (1/tau) * u_{t-1}

where x_t is the input and u_{t-1} is the previous membrane potential. the key gap: **ATMN has no leak term**. the (1/tau) factor scales the carried-over potential, but it does not pull the voltage toward a resting value. without leak, a neuron receiving sustained input accumulates potential without bound. the bridge document [[neuron_models_to_atmn]] proposes adding an explicit leak:

    h_t = (1 - alpha) * u_{t-1} + x_t

where alpha is a learnable decay rate. this restores the LIF structure.

## Hebbian learning and outer products

you know the outer product from attention: Q @ K^T produces a matrix of pairwise similarities. Hebbian learning is the same operation applied to learning.

### the Hebb rule

[[hebbian_learning]] (Hebb 1949) states: neurons that fire together wire together. mathematically:

    Delta_w_ij = eta * x_i * x_j

where w_ij is the connection strength from neuron j to neuron i, eta is a learning rate, and x_i, x_j are the activities of the two neurons. in matrix form:

    Delta_W = eta * x * x^T

this is a rank-1 outer product update. each learning step adds a new pattern to the weight matrix. store N patterns by summing their outer products:

    W = sum_{n=1}^{N} x_n * x_n^T

### a worked example

store two patterns in a 4-neuron Hopfield network. pattern 1: x = [+1, -1, +1, -1]. pattern 2: x = [+1, +1, -1, -1].

    W = [+1,-1,+1,-1]^T * [+1,-1,+1,-1] + [+1,+1,-1,-1]^T * [+1,+1,-1,-1]

    W = [[2, 0, 0,-2],
         [0, 2,-2, 0],
         [0,-2, 2, 0],
         [-2,0, 0, 2]]

now probe with a corrupted version of pattern 1: x_probe = [+1, -1, +1, 0] (last element missing). retrieval: y = sign(W * x_probe) = sign([2, -4, 4, -2]) = [+1, -1, +1, -1]. the network completes the pattern. this is pattern completion via the outer product association.

### matrix memory's state update

the matrix-memory delta rule in todorov (see [[plasticity_to_matrix_memory_delta_rule]]) performs the same operation:

    S_t = diag(alpha) * S_{t-1} + beta_t * k_t * v_t^T

translate each part:

- **diag(alpha) * S_{t-1}**: decay the old state. alpha is a per-channel forgetting rate. this prevents the state from growing unboundedly as more patterns are stored.
- **beta_t**: a data-dependent write gate. some tokens write strongly (beta near 1), others weakly (beta near 0). acts as "this input matters, store it."
- **k_t * v_t^T**: the Hebbian outer product. creates an association between the key (what to match on) and the value (what to retrieve).

readout is q_t^T * S_t: the query probes the associative memory for the stored value most similar to the current query. this is content-addressable retrieval, identical in structure to Hopfield pattern completion.

the key difference from biological [[hebbian_learning]]: matrix memory's update modifies a recurrent STATE, not a WEIGHT matrix. the associations are transient (they decay with alpha), not permanent. matrix memory is an associative memory, not a learning rule. see [[plasticity_to_matrix_memory_delta_rule]] for the full analysis.

## energy functions and attractor dynamics

you know loss functions. in neuroscience, the equivalent is an energy function that the network minimizes through its own dynamics, not through gradient descent.

### Hopfield energy

a network of N binary neurons {-1, +1} with symmetric weights W has energy:

    E = -0.5 * sum_{i,j} w_ij * x_i * x_j

in matrix form: E = -0.5 * x^T * W * x.

the network evolves by asynchronously updating each neuron: x_i = sign(sum_j w_ij * x_j). each update DECREASES or maintains E (proven by Hopfield 1982). the network slides downhill on the energy landscape until it reaches a local minimum. these minima are the stored patterns.

### a worked example

using the 4-neuron network from above (W stores patterns [+1,-1,+1,-1] and [+1,+1,-1,-1]):

energy of pattern 1: E = -0.5 * [1,-1,1,-1] * W * [1,-1,1,-1]^T = -0.5 * (2+2+2+2+2+2+2+2) = -8.

energy of a random state [+1,+1,+1,+1]: E = -0.5 * [1,1,1,1] * W * [1,1,1,1]^T = -0.5 * (2+0+0-2+0+2-2+0+0-2+2+0-2+0+0+2) = 0.

the stored pattern has lower energy. the network dynamics flow from high-energy states toward low-energy stored patterns.

### the connection to softmax attention

Ramsauer et al. (2021) proved that softmax attention implements the update rule of a modern [[pattern_completion|Hopfield network]] with exponential storage capacity:

    softmax(Q @ K^T / sqrt(d)) @ V

this is equivalent to one step of energy minimization in a continuous Hopfield network where the energy is:

    E = -log(sum_n exp(x^T * xi_n))

the exponential function gives the modern Hopfield network capacity proportional to 2^{d/2}, vastly exceeding the classical network's 0.138*N capacity. todorov's compressed-attention layers perform this computation: they are modern Hopfield networks retrieving from a compressed cache.

## information theory basics

you know cross-entropy loss: L = -sum_i y_i * log(p_i). in neuroscience, the same math measures how well neural codes represent inputs.

### mutual information

mutual information I(X;Y) quantifies how much knowing Y tells you about X:

    I(X;Y) = H(X) - H(X|Y)

where H(X) is entropy (the total uncertainty about X) and H(X|Y) is the uncertainty remaining after observing Y. if Y perfectly predicts X, then H(X|Y) = 0 and MI equals the full entropy. if Y is independent of X, MI is zero.

### todorov's spike MI

todorov measures MI between the continuous pre-spike activations (X) and the ternary spike outputs (Y). the ternary quantization maps a 32-bit float to one of three values: {-1, 0, +1}. the theoretical maximum MI is log2(3) = 1.58 bits per dimension (the entropy of a uniform ternary distribution).

the achieved MI at 267M scale is 1.168 bits/dim. this means 74% of the channel capacity is used: the spikes transmit 1.168 out of 1.58 possible bits per dimension. the threshold: MI > 0.1 means spikes carry useful information. see [[population_coding_to_spike_health]] for the full analysis.

### CKA

centered kernel alignment measures whether two representations preserve the same similarity structure. if two inputs that are similar in the pre-spike space remain similar in the post-spike space, CKA is high.

    CKA = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

CKA of 1.0 means perfect geometric preservation. CKA of 0 means the geometries are unrelated. todorov targets CKA > 0.3 and achieves 0.732 at 267M scale. this confirms that the ternary quantization preserves the representational geometry despite compressing 32 bits to ~1.58 bits per dimension.

### firing rate

the fraction of neurons active at any time. biology operates at 1-5% in cortex (see [[energy_efficient_coding]]). todorov operates at ~41%. the gap exists because gradient flow through the straight-through estimator requires enough active neurons to provide gradient signal. below ~20%, training becomes unstable. the 20-60% target range balances information compression against gradient health.

## divisive normalization

you use LayerNorm or RMSNorm in every transformer layer. the brain does something similar but with more structure.

### the biological equation

[[divisive_normalization]] (Carandini and Heeger 2012) is called the "canonical cortical computation" because it appears in nearly every brain area:

    r_i = c_i^n / (sigma^n + sum_j w_j * c_j^n)

each term:

- **c_i**: the driving input to neuron i.
- **n**: an exponent (typically 2). raises the input, creating a power-law nonlinearity.
- **sigma**: the semi-saturation constant. prevents division by zero and controls the sensitivity curve. at low inputs (c_i << sigma), the response is approximately c_i^n / sigma^n (linear in c_i^n). at high inputs (c_i >> sigma), the response saturates.
- **sum_j w_j * c_j^n**: the normalization pool. a weighted sum of neighboring neurons' activities. each neuron's output is divided by what its neighbors are doing. strong neighbors suppress weaker ones.

### a worked example

three neurons with inputs c = [10, 5, 2], n = 2, sigma = 3, uniform weights w = [1, 1, 1].

    pool = 10^2 + 5^2 + 2^2 = 100 + 25 + 4 = 129
    denom = 3^2 + 129 = 9 + 129 = 138

    r_1 = 100 / 138 = 0.72
    r_2 = 25 / 138 = 0.18
    r_3 = 4 / 138 = 0.03

the strongest input captures 72% of the total response. the weakest gets only 3%. this is competitive gain control: strong signals dominate, weak signals are suppressed.

### comparison to RMSNorm

todorov uses RMSNorm before each sublayer:

    y_i = x_i / sqrt(mean(x^2) + eps)

RMSNorm is a special case of divisive normalization with n = 2, sigma = eps, and w_j = 1/d for all j (uniform pool across all dimensions). the differences:

- **no power function on the numerator**: RMSNorm divides x_i directly, not x_i^n.
- **no semi-saturation constant**: eps is small (1e-6), unlike sigma which is a biologically meaningful parameter.
- **uniform pool weights**: RMSNorm normalizes by ALL dimensions equally. biological pools are structured: nearby neurons contribute more, distant neurons contribute less.
- **the output is graded**: RMSNorm produces a continuous scaled value. divisive normalization with large n approaches [[winner_take_all]] (only the strongest input survives).

the ternary spike threshold (threshold = alpha * mean(|x|)) adds a second normalization step: the threshold adapts to the input scale, creating a population-relative selection. RMSNorm + ternary spikes together provide normalization (gain control) followed by sparse selection (thresholding). see [[lateral_inhibition_to_adaptive_threshold]] for the full analysis of how this relates to biological competition.

## where to go next

these foundations connect to specific todorov components through the bridge documents:

- differential equations -> [[neuron_models_to_atmn]] (ATMN's missing leak term)
- outer products -> [[plasticity_to_matrix_memory_delta_rule]] (is matrix memory Hebbian? yes. is it STDP? no.)
- energy functions -> [[memory_systems_to_matrix_memory_and_compressed_attention]] (compressed attention as modern Hopfield network)
- information theory -> [[population_coding_to_spike_health]] (MI and CKA as engineering metrics)
- divisive normalization -> [[lateral_inhibition_to_adaptive_threshold]] (spike threshold vs biological competition)
- the full mapping -> [[todorov_biology_map]] (every component, its biological analog, and where the mapping fails)

## see also

- [[start_here]]
- [[neuroscience_for_ml_engineers]]
- [[glossary]]
- [[todorov_biology_map]]
- [[plasticity_to_matrix_memory_delta_rule]]
- [[memory_systems_to_matrix_memory_and_compressed_attention]]
