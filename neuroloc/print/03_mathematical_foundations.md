# Mathematical Foundations

A math primer for ML engineers entering computational neuroscience. Every equation here connects to something you already know from machine learning. The goal is not exhaustive coverage but the 20% of neuroscience math that explains 80% of Todorov's mechanisms.

## Differential Equations for Neurons

You already know discrete updates. Every recurrent neural network computes:

    h_t = f(h_{t-1}, x_t)

Biological neurons use continuous-time versions of the same thing. The difference: instead of stepping from t to t+1, the neuron's state evolves according to a differential equation that describes how voltage changes at each instant.

### The Leaky Integrate-and-Fire Equation

The simplest useful neuron model is the leaky integrate-and-fire (LIF). It describes a neuron's membrane potential V over time:

    C_m * dV/dt = -g_L * (V - V_rest) + I_ext

Each term maps to an ML concept:

- **C_m** (membrane capacitance, measured in picofarads): controls how fast voltage changes for a given input. Smaller C_m means faster response. ML analog: like a learning rate for the membrane potential. Large C_m = slow updates, small C_m = fast updates.

- **g_L** (leak conductance, measured in nanosiemens): controls how fast voltage drifts back toward rest. The neuron "forgets" its accumulated input. ML analog: the decay factor in an exponential moving average. High g_L = fast forgetting, low g_L = long memory.

- **V_rest** (resting potential, about -65 mV): the baseline voltage when no input arrives. The neuron returns to this value when left alone. ML analog: a bias term that the state decays toward.

- **I_ext** (external current, measured in picoamperes): the input signal. In a network, this is the weighted sum of presynaptic spikes. ML analog: the linear combination W*x that feeds into a neuron.

- **The spike rule**: when V exceeds a threshold V_th (about -50 mV), the neuron fires a spike and resets to V_reset (about -70 mV). ML analog: a step function followed by a state reset.

### A Worked Example

Consider a LIF neuron with C_m = 20 pF, g_L = 10 nS, V_rest = -65 mV, receiving constant current I_ext = 300 pA.

At steady state, dV/dt = 0, so:

    0 = -g_L * (V_ss - V_rest) + I_ext
    V_ss = V_rest + I_ext / g_L
    V_ss = -65 + 300/10 = -65 + 30 = -35 mV

This exceeds any reasonable threshold (typically -50 mV), so the neuron fires repeatedly. The time constant tau_m = C_m / g_L = 20/10 = 2 ms tells you how fast V reaches steady state: after one tau_m, it covers 63% of the gap.

### The Discrete Approximation

For simulation (and for understanding the connection to ML), discretize using the Euler method:

    V[t+1] = V[t] + (dt/C_m) * (-g_L * (V[t] - V_rest) + I[t])

This is just: **new_state = old_state + step_size * (leak + input)**. Identical in structure to a recurrent update with learnable decay.

Rearrange the terms to make the ML connection explicit:

    V[t+1] = (1 - dt*g_L/C_m) * V[t] + (dt*g_L/C_m) * V_rest + (dt/C_m) * I[t]

The factor (1 - dt*g_L/C_m) = (1 - dt/tau_m) is the decay coefficient. With dt = 1 ms and tau_m = 2 ms, the decay is 0.5: the neuron retains half its previous voltage at each step.

### The ATMN Equation

Todorov's ATMN spike neuron uses a simpler update:

    h_t = x_t + (1/tau) * u_{t-1}

where x_t is the input and u_{t-1} is the previous membrane potential. The key gap: **ATMN has no leak term**. The (1/tau) factor scales the carried-over potential, but it does not pull the voltage toward a resting value. Without leak, a neuron receiving sustained input accumulates potential without bound. A proposed fix adds an explicit leak:

    h_t = (1 - alpha) * u_{t-1} + x_t

where alpha is a learnable decay rate. This restores the LIF structure.

## Hebbian Learning and Outer Products

You know the outer product from attention: Q @ K^T produces a matrix of pairwise similarities. Hebbian learning is the same operation applied to learning.

### The Hebb Rule

Hebbian learning (Hebb 1949) states: neurons that fire together wire together. Mathematically:

    Delta_w_ij = eta * x_i * x_j

where w_ij is the connection strength from neuron j to neuron i, eta is a learning rate, and x_i, x_j are the activities of the two neurons. In matrix form:

    Delta_W = eta * x * x^T

This is a rank-1 outer product update. Each learning step adds a new pattern to the weight matrix. Store N patterns by summing their outer products:

    W = sum_{n=1}^{N} x_n * x_n^T

### A Worked Example

Store two patterns in a 4-neuron Hopfield network. Pattern 1: x = [+1, -1, +1, -1]. Pattern 2: x = [+1, +1, -1, -1].

    W = [+1,-1,+1,-1]^T * [+1,-1,+1,-1] + [+1,+1,-1,-1]^T * [+1,+1,-1,-1]

    W = [[2, 0, 0,-2],
         [0, 2,-2, 0],
         [0,-2, 2, 0],
         [-2,0, 0, 2]]

Now probe with a corrupted version of pattern 1: x_probe = [+1, -1, +1, 0] (last element missing). Retrieval: y = sign(W * x_probe) = sign([2, -4, 4, -2]) = [+1, -1, +1, -1]. The network completes the pattern. This is pattern completion via the outer product association.

### KDA's State Update

The KDA delta rule in Todorov performs the same operation:

    S_t = diag(alpha) * S_{t-1} + beta_t * k_t * v_t^T

Translate each part:

- **diag(alpha) * S_{t-1}**: decay the old state. alpha is a per-channel forgetting rate. This prevents the state from growing unboundedly as more patterns are stored.
- **beta_t**: a data-dependent write gate. Some tokens write strongly (beta near 1), others weakly (beta near 0). Acts as "this input matters, store it."
- **k_t * v_t^T**: the Hebbian outer product. Creates an association between the key (what to match on) and the value (what to retrieve).

Readout is q_t^T * S_t: the query probes the associative memory for the stored value most similar to the current query. This is content-addressable retrieval, identical in structure to Hopfield pattern completion.

The key difference from biological Hebbian learning: KDA's update modifies a recurrent STATE, not a WEIGHT matrix. The associations are transient (they decay with alpha), not permanent. KDA is an associative memory, not a learning rule.

## Energy Functions and Attractor Dynamics

You know loss functions. In neuroscience, the equivalent is an energy function that the network minimizes through its own dynamics, not through gradient descent.

### Hopfield Energy

A network of N binary neurons {-1, +1} with symmetric weights W has energy:

    E = -0.5 * sum_{i,j} w_ij * x_i * x_j

In matrix form: E = -0.5 * x^T * W * x.

The network evolves by asynchronously updating each neuron: x_i = sign(sum_j w_ij * x_j). Each update DECREASES or maintains E (proven by Hopfield 1982). The network slides downhill on the energy landscape until it reaches a local minimum. These minima are the stored patterns.

### A Worked Example

Using the 4-neuron network from above (W stores patterns [+1,-1,+1,-1] and [+1,+1,-1,-1]):

Energy of pattern 1: E = -0.5 * [1,-1,1,-1] * W * [1,-1,1,-1]^T = -0.5 * (2+2+2+2+2+2+2+2) = -8.

Energy of a random state [+1,+1,+1,+1]: E = -0.5 * [1,1,1,1] * W * [1,1,1,1]^T = -0.5 * (2+0+0-2+0+2-2+0+0-2+2+0-2+0+0+2) = 0.

The stored pattern has lower energy. The network dynamics flow from high-energy states toward low-energy stored patterns.

### The Connection to Softmax Attention

Ramsauer et al. (2021) proved that softmax attention implements the update rule of a modern Hopfield network with exponential storage capacity:

    softmax(Q @ K^T / sqrt(d)) @ V

This is equivalent to one step of energy minimization in a continuous Hopfield network where the energy is:

    E = -log(sum_n exp(x^T * xi_n))

The exponential function gives the modern Hopfield network capacity proportional to 2^{d/2}, vastly exceeding the classical network's 0.138*N capacity. Todorov's MLA layers perform this computation: they are modern Hopfield networks retrieving from a compressed cache.

## Information Theory Basics

You know cross-entropy loss: L = -sum_i y_i * log(p_i). In neuroscience, the same math measures how well neural codes represent inputs.

### Mutual Information

Mutual information I(X;Y) quantifies how much knowing Y tells you about X:

    I(X;Y) = H(X) - H(X|Y)

where H(X) is entropy (the total uncertainty about X) and H(X|Y) is the uncertainty remaining after observing Y. If Y perfectly predicts X, then H(X|Y) = 0 and MI equals the full entropy. If Y is independent of X, MI is zero.

### Todorov's Spike MI

Todorov measures MI between the continuous pre-spike activations (X) and the ternary spike outputs (Y). The ternary quantization maps a 32-bit float to one of three values: {-1, 0, +1}. The theoretical maximum MI is log2(3) = 1.58 bits per dimension (the entropy of a uniform ternary distribution).

The achieved MI at 267M scale is 1.168 bits/dim. This means 74% of the channel capacity is used: the spikes transmit 1.168 out of 1.58 possible bits per dimension. The threshold: MI > 0.1 means spikes carry useful information.

### CKA

Centered kernel alignment measures whether two representations preserve the same similarity structure. If two inputs that are similar in the pre-spike space remain similar in the post-spike space, CKA is high.

    CKA = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

CKA of 1.0 means perfect geometric preservation. CKA of 0 means the geometries are unrelated. Todorov targets CKA > 0.3 and achieves 0.732 at 267M scale. This confirms that the ternary quantization preserves the representational geometry despite compressing 32 bits to ~1.58 bits per dimension.

### Firing Rate

The fraction of neurons active at any time. Biology operates at 1-5% in cortex. Todorov operates at ~41%. The gap exists because gradient flow through the straight-through estimator requires enough active neurons to provide gradient signal. Below ~20%, training becomes unstable. The 20-60% target range balances information compression against gradient health.

## Divisive Normalization

You use LayerNorm or RMSNorm in every transformer layer. The brain does something similar but with more structure.

### The Biological Equation

Divisive normalization (Carandini and Heeger 2012) is called the "canonical cortical computation" because it appears in nearly every brain area:

    r_i = c_i^n / (sigma^n + sum_j w_j * c_j^n)

Each term:

- **c_i**: the driving input to neuron i.
- **n**: an exponent (typically 2). Raises the input, creating a power-law nonlinearity.
- **sigma**: the semi-saturation constant. Prevents division by zero and controls the sensitivity curve. At low inputs (c_i << sigma), the response is approximately c_i^n / sigma^n (linear in c_i^n). At high inputs (c_i >> sigma), the response saturates.
- **sum_j w_j * c_j^n**: the normalization pool. A weighted sum of neighboring neurons' activities. Each neuron's output is divided by what its neighbors are doing. Strong neighbors suppress weaker ones.

### A Worked Example

Three neurons with inputs c = [10, 5, 2], n = 2, sigma = 3, uniform weights w = [1, 1, 1].

    pool = 10^2 + 5^2 + 2^2 = 100 + 25 + 4 = 129
    denom = 3^2 + 129 = 9 + 129 = 138

    r_1 = 100 / 138 = 0.72
    r_2 = 25 / 138 = 0.18
    r_3 = 4 / 138 = 0.03

The strongest input captures 72% of the total response. The weakest gets only 3%. This is competitive gain control: strong signals dominate, weak signals are suppressed.

### Comparison to RMSNorm

Todorov uses RMSNorm before each sublayer:

    y_i = x_i / sqrt(mean(x^2) + eps)

RMSNorm is a special case of divisive normalization with n = 2, sigma = eps, and w_j = 1/d for all j (uniform pool across all dimensions). The differences:

- **No power function on the numerator**: RMSNorm divides x_i directly, not x_i^n.
- **No semi-saturation constant**: eps is small (1e-6), unlike sigma which is a biologically meaningful parameter.
- **Uniform pool weights**: RMSNorm normalizes by ALL dimensions equally. Biological pools are structured: nearby neurons contribute more, distant neurons contribute less.
- **The output is graded**: RMSNorm produces a continuous scaled value. Divisive normalization with large n approaches winner-take-all behavior (only the strongest input survives).

The ternary spike threshold (threshold = alpha * mean(|x|)) adds a second normalization step: the threshold adapts to the input scale, creating a population-relative selection. RMSNorm + ternary spikes together provide normalization (gain control) followed by sparse selection (thresholding).
