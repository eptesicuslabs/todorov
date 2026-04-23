# unified learning hypothesis

status: current (as of 2026-04-16).

## the core insight

the neural machine's forward pass contains operations that are structurally analogous to biological local learning rules. this suggests the POSSIBILITY of forward-only training -- a hypothesis that has not been tested.

**what exists in the current architecture (state updates, not parameter learning):**

- outer product (k * v^T): writes associations to per-sequence recurrent state. this IS hebbian association, but it operates on transient working memory, not on persistent weights. the state is discarded at sequence end.
- channel-wise decay (alpha): per-channel per-head forgetting rate. learned by backprop, FIXED during inference. this is NOT BCM (which requires activity-dependent runtime adaptation). it is a static decay constant.
- delta rule erasure (proposed, not yet implemented): targeted overwrite of old associations. this IS error correction but only operates on the recurrent state.
- k-WTA competitive selection: selects most active dimensions. this IS lateral inhibition but does not modify weights.

**what does NOT exist but would be needed:**

- prediction error computation between layers: no top-down predictions, no error signals. would need explicit feedback connections.
- runtime-adaptive thresholds: current alpha is fixed after training. BCM requires alpha to slide based on postsynaptic activity during processing.
- weight updates from forward pass: the outer product modifies state (in-context), not weights (cross-example). true forward-only learning requires the forward pass to update persistent parameters.

## the hypothesis (untested conjecture)

**if the architecture were redesigned to include prediction error between layers and runtime-adaptive thresholds, the forward pass COULD become the learning step.** each time the core operation z = Q(R(B(C(x), C(h)))) runs, it simultaneously:

1. computes the output (inference)
2. updates the memory state h via outer product (learning)
3. adjusts the adaptive threshold via BCM alpha (meta-learning)
4. erases outdated associations via delta rule (forgetting)
5. selects which dimensions survive via k-WTA (competition)

no separate backward pass is needed. the prediction error at each layer is the local teaching signal for that layer. the BCM threshold ensures stability. the delta rule ensures capacity. the k-WTA ensures selectivity.

## biological precedent

this is what the brain does. biological neurons do not have a "forward pass" and a separate "backward pass." the same synaptic activity that transmits information also modifies the synapse (hebbian plasticity). the same prediction error that drives perception also updates the internal model (predictive coding). the same competitive dynamics that select which neurons fire also determine which connections strengthen (lateral inhibition + STDP).

millidge et al. (2022, Neural Computation) showed that predictive coding weight updates converge to exact backprop gradients on arbitrary computation graphs, but only in the limit of iterating inference dynamics to equilibrium at each layer. in practice, finite iterations produce an approximation whose quality degrades with network depth. the question is whether this architecture -- with its prediction-error-driven processing, local outer-product updates, and competitive selection -- is one where the local rule equals the optimal update.

## the prediction

an architecture designed for local learning MIGHT not suffer the depth-scaling failure seen in predictive coding on standard networks (salvatori et al. ICLR 2025: PC matches backprop on shallow networks, fails at depth 9+). the salvatori failure is due to inference instability (iterative inference at each layer doesn't converge reliably in deep networks), not error signal attenuation. whether the neural machine's architecture avoids this specific failure mode is an untested conjecture. the todorov architecture is 24 layers deep, well into the failure zone observed by salvatori.

## what needs testing

1. implement the forward-pass-as-learning variant: the outer product write to memory state uses the prediction error (input minus predicted input) as the value signal, not the raw input
2. compare against standard backprop at matched scale on:
   - pattern completion (association quality)
   - sequence prediction (next-token accuracy)
   - capacity scaling (how many patterns can be stored)
3. measure whether the local update converges to the same solution as backprop, or to a different (possibly better) solution

## the compression connection

the hierarchical k-WTA compression (novel, unpublished) is also architecture-agnostic -- it applies to any activation tensor, not just transformers. the two-stage pipeline (top-k selection by magnitude, then ternary quantization of survivors) produces 0.32-0.56 bits/dim at 5-10% selection with CKA > 0.63. no prior work applies this to runtime activations with measured compression-quality tradeoff.

closest prior: ComPEFT (yadav et al., EMNLP 2024) applies the identical math to weight deltas (not activations). Q-Sparse (arXiv 2407.10969) applies top-k to activations but keeps them full precision. the combination of k-WTA + ternary on activations is the novel contribution.

## implications for the blueprint

if the unified learning hypothesis holds (after adding prediction error and adaptive thresholds):
- the "learning method" open question in the blueprint would be answered: the learning method emerges from the architecture
- backprop could be replaced by local updates for some or all components
- the architecture would be fundamentally different from all existing ML: it would not distinguish training from inference
- IMPORTANT: no non-backprop method has been validated above ~25M parameters on standard benchmarks (see [[learning_rules_research]]). this hypothesis contradicts the current evidence base and requires experimental validation before any commitment.

if it fails:
- use backprop for v1 (proven, scales)
- test scaled DTP and evolution strategies for v2
- the architecture still works as designed, just with conventional training

## see also

- [[learning_rules_research]]
- [[memory_capacity_research]]
- [[compression_architecture]]
- [[predictive_coding]]
- [[hebbian_learning]]
- [[bcm_theory]]
- [[plasticity_to_matrix_memory_delta_rule]]
