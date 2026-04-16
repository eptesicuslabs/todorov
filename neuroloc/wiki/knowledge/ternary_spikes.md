# Ternary Spikes: Findings from the Gerhard Project

status: current (as of 2026-04-16).

Source: Gerhard project (eptesicuslabs/gerhard)
No external web research. This documents empirical findings from the project.


## 1. Spike Generation via Adaptive Threshold

Spike values are ternary: {-1, 0, +1}

Threshold function:

    theta(x) = alpha * mean(|x|)

where alpha is the spike_threshold hyperparameter and x is the input tensor.

Spike generation:

    spike(x) =  +1   if x >  theta(x)
                -1   if x < -theta(x)
                 0   otherwise

The threshold is adaptive: it scales with the mean absolute activation of
the current input, making the spiking behavior invariant to the scale of
the representation across layers and training stages.


## 2. Straight-Through Estimator (STE) Gradient Flow

The spike function is non-differentiable (hard threshold). Gradients are
passed through using the Straight-Through Estimator:

    Forward:  spike(x)  (ternary quantization)
    Backward: grad_output * 1  (identity gradient, as if spike = identity)

This means the gradient of the spike function is treated as 1 everywhere,
allowing standard backpropagation to flow through the discrete spiking layer.

The STE approximation works because:
- The spike function is close to identity in expectation for well-calibrated thresholds
- The adaptive threshold keeps the fraction of spiking neurons in a healthy range
- Gradient clipping may be applied downstream if needed


## 3. Health Monitoring Metrics

Three metrics are used to assess whether spike layers are functioning properly:

Mutual Information (MI):
- Target: MI > 0.1
- Measures: information preserved between spike input and spike output
- If MI drops below 0.1, the spike layer is destroying too much information
- Computed between binned input activations and ternary output values

Centered Kernel Alignment (CKA):
- Target: CKA > 0.3
- Measures: representational similarity between layers before and after spiking
- If CKA drops below 0.3, the spiking layer is distorting representations
  beyond recovery

Firing Rate:
- Target: 30-60% of neurons should be non-zero (firing)
- Below 30%: too sparse, information bottleneck
- Above 60%: too dense, spikes are not providing useful sparsity
- Firing rate = count(|spike(x)| > 0) / total_neurons


## 4. Key Hyperparameter Finding

    spike_threshold = 0.1  (i.e., alpha = 0.1)

This value was found to produce healthy spiking behavior:
- theta(x) = 0.1 * mean(|x|)
- Results in ~40-50% firing rate for typical transformer activations
- MI remains above 0.1 threshold
- CKA remains above 0.3 threshold
- Training remains stable with standard learning rates

Higher values (alpha > 0.3) tend to produce too-sparse firing (< 30%)
and can cause gradient starvation in deeper layers.

Lower values (alpha < 0.05) produce near-dense firing (> 70%) which
defeats the purpose of the sparse ternary representation.


## 5. Monitoring Protocol

During training, monitor at regular intervals (e.g., every 100 steps):

    for each spike layer:
        1. Compute firing_rate = mean(spike(x) != 0)
        2. Compute MI between x and spike(x)
        3. Compute CKA between pre-spike and post-spike representations
        4. Log warnings if any metric falls outside healthy range

    Health check pass conditions:
        - MI > 0.1
        - CKA > 0.3
        - 0.30 < firing_rate < 0.60

If health checks fail consistently:
- Check if spike_threshold needs adjustment
- Verify that layer normalization precedes the spike layer
- Consider reducing learning rate for affected layers


## 6. Integration Notes

- Ternary spikes can replace standard activations in feed-forward blocks
- The sparse ternary representation enables efficient hardware execution
  (multiply-accumulate reduces to add/subtract)
- Combined with standard floating-point attention, the hybrid approach
  retains model quality while reducing compute in FFN layers


## Source

All findings from: eptesicuslabs/gerhard


---

## 7. MAR Framework and ATMN Neuron Model

Paper: "MAR: Efficient Large Language Models via Module-aware Architecture Refinement"
ArXiv ID: 2601.21503
Submitted: January 29, 2026
Source: https://arxiv.org/abs/2601.21503

MAR is a two-stage framework that (1) replaces softmax attention with Mamba-2
SSM blocks for linear-time sequence modeling, then (2) inserts spiking neurons
into the FFN and SSM projections to sparsify dense activations. The spiking
neuron used is the Adaptive Ternary Multi-step Neuron (ATMN).


### 7.1 ATMN Membrane Potential Dynamics

The ATMN extends the classical Leaky Integrate-and-Fire (LIF) neuron to emit
ternary spikes {-1, 0, +1}. The dynamics across timesteps t are:

    h_t = I_t * delta_{t,0} + (1 / tau) * u_{t-1}

    s_t = { +1   if h_t >= V_adaptive
           { -1   if h_t <= -V_adaptive
           {  0   otherwise

    u_t = h_t - s_t * V_adaptive

Where:
- I_t is the input current at timestep t
- delta_{t,0} is the Kronecker delta (input injected only at t=0)
- tau is the membrane time constant (decay factor)
- u_{t-1} is the residual membrane potential from the previous timestep
- h_t is the pre-spike membrane potential
- s_t is the ternary spike output
- V_adaptive is the adaptive firing threshold

The residual potential u_t carries over to the next timestep, enabling
multi-step temporal accumulation. The term (1/tau) * u_{t-1} implements
leaky integration: the membrane potential decays by factor 1/tau each step,
but residual information persists across timesteps. This is the "multi-step"
aspect of ATMN -- unlike single-timestep binary neurons, ATMN accumulates
temporal state.

Source: arxiv 2601.21503, Equations in Section 3.2


### 7.2 Adaptive Threshold Mechanism

    V_adaptive = exp(a)

where a is a learnable, per-neuron parameter. The exponential function
guarantees V_adaptive > 0 (non-negativity constraint). This makes the
threshold adaptive: each neuron learns its own firing threshold during
training, adjusting to the local activation scale.

Unlike the Gerhard project's threshold (theta = alpha * mean(|x|)), which
is input-adaptive per forward pass, MAR's threshold is parameter-adaptive
(learned during training and fixed at inference).

Source: arxiv 2601.21503, Section 3.2


### 7.3 Performance Results vs Baselines (Zero-Shot Average across 6 benchmarks)

    +------------------+--------+-----+------------+
    | Model            | Size   | SNN | Accuracy   |
    +------------------+--------+-----+------------+
    | LLaMA-3.2        | 1.3B   | No  | 61.80%     |
    | Llamba (teacher)  | 1.4B   | No  | 61.88%     |
    | MAR (full)        | 1.4B   | Yes | 57.20%     |
    | SpikeLLM          | 7B     | Yes | 52.48%     |
    +------------------+--------+-----+------------+

MAR at 1.4B parameters outperforms SpikeLLM at 7B by +4.72pp, demonstrating
that ATMN ternary spikes are far more efficient than binary spikes at the
same or smaller scale.

The gap to the non-spiking teacher (Llamba) is 4.68pp, which is the cost
of converting to a fully spiking architecture.

Source: arxiv 2601.21503, Table 2


### 7.4 Ablation: Effect of Individual Components

    +------+------------+---------------+----------+
    | ATMN | Reverse KL | PreNorm Loss  | Accuracy |
    +------+------------+---------------+----------+
    |  No  |    No      |      No       | 46.28%   |
    |  Yes |    No      |      No       | 55.20%   |
    |  Yes |    Yes     |      No       | 55.46%   |
    |  Yes |    Yes     |     Yes       | 57.20%   |
    +------+------------+---------------+----------+

Key findings:
- ATMN alone provides +8.92pp gain over binary spikes (46.28 -> 55.20)
- Adding reverse KL to distillation provides +0.26pp (55.20 -> 55.46)
- Adding pre-norm feature alignment provides +1.74pp (55.46 -> 57.20)
- Total SBDS (reverse KL + pre-norm) provides +2.00pp on top of ATMN

Source: arxiv 2601.21503, Table 3


### 7.5 Ablation: Spike Placement (Which Projections Benefit Most)

Four spike insertion positions per layer were tested:
1. Before input projection of Mamba-2 block
2. Before output projection of Mamba-2 block
3. Before input projection of FFN block
4. Before output projection of FFN block

Finding: Pre-normalization feature alignment at spike insertion points proved
most effective (57.20% vs 56.75% with post-norm alignment). This indicates
that pre-normalization features provide more stable and transferable
representations for distillation across the spike boundary.

All four positions contribute; removing spikes from any single position
degrades accuracy, but the FFN projections show the largest individual
contribution to compute savings due to the higher dimensionality of FFN
intermediate activations.

Source: arxiv 2601.21503, Section 4.3


### 7.6 Related: SpikingMamba

Paper: "SpikingMamba: Towards Energy-Efficient Large Language Models
        via Knowledge Distillation from Mamba"
ArXiv ID: 2510.04595

SpikingMamba integrates a ternary-integer spiking neuron (TI-LIF) that
preserves semantic polarity through signed multi-level spike representations,
and a training-exclusive Smoothed Gradient Compensation (SGC) path to
mitigate quantization loss while preserving spike-driven efficiency.

SpikingMamba-1.3B achieves a 4.76x energy benefit with only a 4.78%
zero-shot accuracy gap compared to the original Mamba model.

Source: arxiv 2510.04595


---

## 8. SBDS: Spike-aware Bidirectional Distillation Strategy

Source: arxiv 2601.21503 (same MAR paper)

SBDS is the distillation framework designed to recover performance lost
when converting dense activations to sparse ternary spikes. It operates
at two levels: logit-level and feature-level.


### 8.1 Logit-Level Loss (Bidirectional KL)

    L_1(p || q) = sum_k [ alpha * p(k) - beta * q(k) ] * [ log p(k) - log q(k) ]

Where:
- p is the teacher distribution
- q is the student (spiking) distribution
- alpha weights the forward KL component (teacher -> student)
- beta weights the reverse KL component (student -> teacher)

When alpha=1, beta=0: standard forward KL (mode-covering)
When alpha=0, beta=1: pure reverse KL (mode-seeking)

Source: arxiv 2601.21503, Equation 7


### 8.2 Why Reverse KL Helps Sparse Spiking Outputs

Standard (forward) KL divergence KL(p || q) forces the student to cover
all modes of the teacher distribution, spreading probability mass widely.
This is problematic for spiking networks because:
- Spike outputs are inherently sparse and bursty
- Covering all modes forces the spiking network to fire broadly, defeating
  the purpose of sparse activation

Reverse KL divergence KL(q || p) encourages the student to concentrate on
high-confidence predictions -- it is "mode-seeking." This matches the
episodic firing pattern of spiking neurons: the network can focus on the
most important output modes and stay silent elsewhere.

The optimal balance found by ablation is alpha=0.2, beta=0.7 (emphasizing
reverse KL), achieving 56.40% before adding feature-level loss.

Source: arxiv 2601.21503, Table 4


### 8.3 Alpha/Beta Coefficient Ablation

    +-------+-------+----------+
    | alpha | beta  | Accuracy |
    +-------+-------+----------+
    |  0    |  1    | 55.92%   |
    |  1    |  0    | 55.20%   |
    |  0.7  |  0.2  | 56.39%   |
    |  0.2  |  0.7  | 56.40%   |  <-- best
    +-------+-------+----------+

The best configuration (alpha=0.2, beta=0.7) emphasizes the reverse KL
term by 3.5x relative to the forward term, confirming the importance of
mode-seeking distillation for sparse spiking outputs.

Source: arxiv 2601.21503, Table 4


### 8.4 Feature-Level Loss (Pre-Norm Alignment)

    L_2(h^j, h^s) = || PreNorm(h^j) - PreNorm(h^s) ||_2

Where:
- h^j is the hidden state from the teacher (non-spiking) layer j
- h^s is the hidden state from the student (spiking) layer
- PreNorm applies layer normalization before computing the L2 distance

Pre-norm alignment is critical because:
- Raw activations before and after spike insertion have different scales
- Pre-normalization maps both to a comparable range
- This yields +1.74pp improvement over no feature-level loss

Post-norm alignment (normalizing after the full block) was tested and
yields only 56.75% vs 57.20% for pre-norm, confirming that aligning
representations before the spike boundary provides more stable gradients.

Source: arxiv 2601.21503, Section 3.3


### 8.5 Total SBDS Loss

    L_distill = (1 / (T * M)) * sum L_1 + (1 / (T * L)) * sum L_2

Where:
- T = number of timesteps
- M = sequence length
- L = number of layers
- L_1 is summed over timesteps and sequence positions
- L_2 is summed over timesteps and layers

Source: arxiv 2601.21503, Equation 8


### 8.6 Related Work: Head-Tail-Aware KL for SNN Distillation

Paper: "Head-Tail-Aware KL Divergence in Knowledge Distillation
        for Spiking Neural Networks"
ArXiv ID: 2504.20445

This paper independently validates the importance of asymmetric KL
divergence in SNN distillation, finding that standard symmetric KL
is suboptimal for the sparse output distributions of spiking networks.

Source: arxiv 2504.20445
