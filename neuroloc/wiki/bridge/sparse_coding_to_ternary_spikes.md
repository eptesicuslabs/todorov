# bridge: sparse coding to ternary spikes

## the biological mechanism

the [[efficient_coding]] hypothesis (barlow, 1961) and [[sparse_coding]] framework (olshausen and field, 1996) establish that cortical neurons encode information sparsely: only a small fraction of neurons are active at any time, and the identity of the active subset carries most of the information.

key biological features:
- population sparseness: 1-10% of neurons active in cortex (vinje and gallant, 2000)
- adaptive thresholding: firing threshold adjusts based on recent input statistics (divisive normalization; carandini and heeger, 2012)
- k-winners-take-all: lateral inhibition selects the most active neurons and suppresses the rest
- metabolic constraint: each spike costs ~10^4 ATP molecules; sparsity minimizes energy use (laughlin, 2001)
- overcomplete representation: more neurons than input dimensions, enabling sparser codes per stimulus
- temporal sparsity: neurons are silent most of the time, firing in brief bursts

see also: [[sparse_distributed_representations]] for the information-theoretic properties of sparse binary codes (capacity, noise robustness, set operations).

## the current todorov implementation

### ternary spikes (src/spikes/ternary_spike.py)

the AdaptiveTernarySpike module:
- computes threshold = alpha * mean(|x|), where alpha is a learnable parameter (init 1.0)
- clamps threshold to [min_threshold, max_threshold] (default [0.01, 10.0])
- produces output: sign(x) * [|x| > threshold] -> {-1, 0, +1}^d
- backward pass: straight-through estimator (STE) -- gradient passes through quantization as identity
- tracks running spike density with momentum 0.99
- firing rate at alpha=1.0: ~41% (59% of dimensions are zero)
- information content: 1.58 bits/dimension (log2(3)) vs 32 bits for FP32

the threshold alpha * mean(|x|) is a population-level statistic: the mean absolute activation across all dimensions. this is a crude form of gain control -- the threshold adapts to the scale of the input. but it is a single scalar applied uniformly to all dimensions, unlike biological divisive normalization which computes per-neuron or per-local-pool gain.

### ATMN spikes (src/spikes/atmn_spike.py)

the ATMNSpike module adds temporal dynamics:
- membrane potential integration: h_t = x_t + (1/tau) * u_{t-1}
- per-neuron threshold: V_th = exp(threshold_log), where threshold_log is a learnable parameter per dimension
- ternary firing: fire +1 if h_t >= V_th, fire -1 if h_t <= -V_th, else 0
- reset by subtraction: u_t = h_t - spikes * V_th (the membrane potential is reduced by V_th when a spike fires)
- during training: membrane potential is reset to zero each forward call (no temporal accumulation across training batches)
- STE backward pass identical to basic ternary spikes

ATMN adds two biologically meaningful features: membrane potential accumulation across timesteps (leaky integration) and per-neuron adaptive thresholds. but the training-time reset eliminates the temporal dynamics that would make ATMN a true spiking neuron model. ATMN is implemented but not yet validated at scale (unit tests pass; never trained beyond smoke tests).

### spike health metrics (src/spikes/spiking_brain.py)

the SpikingBrainValidator monitors spike health via:
- **firing rate**: fraction of nonzero activations. target range: 20-60%. threshold for dead neurons: <5%. threshold for saturated neurons: >95%.
- **mutual information (MI)**: binning-based MI between spike outputs and pre-spike activations. target: MI > 0.1. measures how much of the input information survives quantization. achieved: MI = 1.168 at 267m scale, MI = 1.311 at 6m scale with GP active.
- **centered kernel alignment (CKA)**: linear CKA between spike representations and pre-spike representations. target: CKA > 0.3. measures whether the geometry of the representation is preserved through quantization. achieved: CKA = 0.732 at 267m scale.

these metrics are the bridge from [[population_coding]] theory to engineering practice. see [[population_coding_to_spike_health]] for the full analysis.

## what biology suggests about optimal sparsity

### the cortical evidence

cortical sparsity levels (population sparseness):
- V1 with natural stimuli: ~5-10% (vinje and gallant, 2000)
- IT cortex for objects: ~5-10% (rolls and tovee, 1995)
- hippocampus CA1 place cells: ~1-2% per location
- auditory cortex A1: ~5-15% (hromadka et al., 2008)

these levels are far sparser than todorov's 41%.

### the theoretical optimum

for associative memory capacity, optimal sparsity is f ~ 1/sqrt(N) for N neurons. for d=384 (todorov's small scale), this gives f ~ 5.1%. for d=1024 (the 267m scale), f ~ 3.1%.

for autoencoder-based memory (hippocampal model, 2025), optimal coding level is f ~ 5-7.5% depending on input compressibility.

for maximum information entropy of a ternary code, the optimum is 1/3 active per symbol (~33% for each of {-1, 0, +1}), which gives H = log2(3) = 1.58 bits/dim. but this ignores the sparsity constraint -- it treats all three values as equally informative, which is not the case when zero means "no information."

### the gradient flow constraint

the reason todorov does not operate at 5%: gradient flow through the STE.

the STE passes the gradient through quantization as if it were the identity. but the effective gradient magnitude at each dimension correlates with the pre-spike activation magnitude. at 5% firing rate, 95% of dimensions contribute near-zero gradients. for a 300M-parameter model trained with standard optimizers (AdamW), this level of gradient sparsity would likely:
- slow convergence by 5-10x (less gradient information per step)
- create dead neuron pathologies (neurons that never fire never receive gradient signal to change their behavior)
- destabilize training through high gradient variance

empirical evidence: the 20-60% firing rate range was established through training runs at 6m and 267m scale. below 20%, training becomes unstable. above 60%, the sparsity benefit (information compression, noise filtering) is lost.

## the proposed change

the biological evidence suggests that 41% is too dense for the information-theoretic benefits of sparse coding (Olshausen & Field 2004; capacity, noise robustness) but may be necessary for gradient-based optimization. two potential interventions:

### option A: graduated sparsity schedule

start training at 40% firing rate (alpha=1.0) for stable gradient flow during early training, then anneal alpha upward to increase the threshold, targeting 15-20% firing rate by end of training. this follows the intuition that early training needs broad gradient flow to find the right loss basin, while late training benefits from sharper, sparser representations.

implementation: linear or cosine schedule on alpha from 1.0 to 2.5 over the course of training. the running_spike_density tracker already provides the feedback signal. no architectural change needed -- only a scheduler on the alpha parameter.

risk: if the model learns representations calibrated to 40% sparsity, increasing sparsity mid-training may catastrophically lose information that the downstream layers depend on. requires careful monitoring of MI and CKA during the transition.

### option B: per-dimension adaptive thresholds (ATMN-style)

replace the global alpha * mean(|x|) threshold with per-dimension thresholds, as ATMN already implements. per-dimension thresholds allow different sparsity levels for different features: high-information features can have lower thresholds (fire more often), while redundant features can have higher thresholds (fire less often).

this is closer to biological divisive normalization, where gain control is local. it also enables the system to find its own optimal sparsity per dimension rather than imposing a uniform rate.

implementation: ATMN is already implemented. the change is to validate it at scale (phase 5a) and compare MI, CKA, and BPB against the baseline ternary spikes.

risk: ATMN was found to be too slow in run_011 (~2x overhead). the per-neuron threshold adds d learnable parameters per spike layer. performance optimization needed before scale validation.

### option C: auxiliary sparsity loss

add a loss term penalizing deviation from a target firing rate f_target:

    L_sparsity = beta * (firing_rate - f_target)^2

where beta controls the strength and f_target is the desired sparsity level (e.g., 0.15). this decouples the sparsity level from the threshold mechanism, allowing the optimizer to find thresholds that achieve the target rate while maintaining gradient flow.

implementation: straightforward addition to the training loss. the firing rate is already tracked. requires tuning beta and f_target as hyperparameters.

risk: the auxiliary loss may conflict with the primary language modeling objective. if the optimizer pushes alpha to achieve the target firing rate at the expense of representation quality, MI and BPB will degrade.

## implementation spec

the recommended intervention for phase 5a is option B (ATMN validation), which is already in the phase 5 sequencing plan. no new code needed -- the implementation exists in src/spikes/atmn_spike.py.

if ATMN is too slow at scale, option A (sparsity schedule) is the fallback:

```python
class SparsityScheduler:
    def __init__(self, spike_modules, alpha_start=1.0, alpha_end=2.5, total_steps=10000):
        self.spike_modules = spike_modules
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.total_steps = total_steps

    def step(self, current_step):
        frac = min(current_step / self.total_steps, 1.0)
        alpha = self.alpha_start + frac * (self.alpha_end - self.alpha_start)
        for module in self.spike_modules:
            module.alpha.data.fill_(alpha)
```

this modifies only the training loop, not the model architecture.

## expected impact / risk assessment

| intervention | expected firing rate | expected MI impact | expected BPB impact | risk |
|---|---|---|---|---|
| baseline (alpha=1.0) | 41% | 1.168 (known) | 0.663x (known) | none (status quo) |
| ATMN per-neuron thresh | 20-40% (learned) | unknown | unknown | speed overhead, training instability |
| sparsity schedule 40->15% | 15% at end | likely decrease | likely worse early, possibly better late | catastrophic forgetting during transition |
| auxiliary sparsity loss (f=0.15) | 15% (enforced) | likely decrease | unknown | loss conflict, hyperparameter sensitivity |

the honest prediction: reducing firing rate below 30% will hurt BPB at current model scale (6m-267m) because gradient flow through the STE is the binding constraint. the biological optimum of 5-10% may only be achievable with non-STE gradient methods (e.g., REINFORCE, evolutionary strategies) or at much larger scale where each dimension's gradient contribution is less critical.

the key experiment is phase 5a: ATMN vs ternary at matched architecture and training budget. this isolates the neuron model impact and determines whether per-neuron thresholds can find a better operating point than the global threshold.

## see also

- [[efficient_coding]]
- [[sparse_coding]]
- [[sparse_vs_dense_representations]]
- [[population_coding_to_spike_health]]
