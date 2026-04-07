# local vs global computation

## the biological principle

cortex is dominated by local recurrence. the [[canonical_microcircuit]] establishes that only 5-15% of excitatory synapses on layer 4 neurons come from the thalamus; the remaining 85-95% come from other cortical neurons, mostly within the same column (douglas and martin 1991). the computational consequence is that cortical "processing" is overwhelmingly recurrent re-processing of a weak external signal, not feedforward transformation of a strong one. the amplification factor 1/(1-g) can reach 5-20x when the recurrent gain g approaches unity, constrained from below by [[inhibitory_interneurons]] that prevent runaway excitation.

this local recurrence is not homogeneous. three cardinal interneuron types impose distinct computational operations on different spatial compartments of pyramidal neurons: PV+ basket cells control timing at the soma (when to fire), SST+ martinotti cells gate integration at the apical dendrites (what to integrate), and VIP+ bipolar cells disinhibit SST+ neurons (whether to allow top-down influence). the VIP -> SST -> pyramidal disinhibitory pathway is the biological mechanism for contextual routing -- it selectively opens or closes the dendritic gate based on behavioral state.

within each neuron, [[dendritic_computation]] adds another layer of locality. a pyramidal neuron with ~30-50 thin terminal dendrites operates as ~30-50 independent nonlinear subunits (the [[two_layer_neuron]] model of poirazi et al. 2003). NMDA spikes on each branch are strictly local -- they do not propagate to neighboring branches -- creating compartmentalized processing that makes a single neuron a two-layer neural network with ~30-50 hidden units. [[dendritic_spikes]] of three types (Na+, Ca2+, NMDA) provide coincidence detection at three timescales (1 ms, 50 ms, 100 ms), enabling multi-timescale feature binding within a single cell.

the most dramatic expression of source-segregated local computation is [[apical_amplification]]: layer 5 pyramidal neurons integrate bottom-up input at basal dendrites and top-down feedback at the apical tuft, ~300-800 um apart. BAC firing requires coincidence of both signals -- a cellular AND gate that binds feedforward features to contextual predictions. SST+ interneurons can veto this coincidence by inhibiting the apical compartment, and VIP+ neurons can release that veto, creating a three-level control circuit (VIP -> SST -> dendrite -> BAC) for context-dependent gating.

the result is a computational architecture with three nested levels of locality: (1) local circuits within a column, where 85-95% of connections are recurrent, (2) local compartments within a neuron, where ~30-50 dendritic branches operate as independent nonlinear subunits, and (3) local gating within a compartment, where NMDA coincidence detection requires spatiotemporally clustered input on a single branch (~10-50 synapses within ~20-50 um and ~5-10 ms). global communication is the exception, not the rule -- cortical long-range connections (inter-areal, callosal) are a small fraction of total synapses, and they arrive at specific laminar targets (feedback to L1/L6, feedforward to L4), not broadcast everywhere.

the principle unifying all of this: cortex computes by recurrent amplification within local circuits that receive source-segregated inputs at distinct spatial compartments, gated by type-specific inhibition.

## the todorov instantiation

todorov replaces all of this with serial depth. 24 layers, each receiving the same residual stream, each adding its output back to that stream. within a layer, swiglu performs the closest operation to dendritic gating:

    gate = silu(W_gate(x))
    up = W_up(x)
    out = W_down(gate * up)

this is a multiplicative interaction between two independent linear projections -- formally analogous to NMDA coincidence detection (requiring both presynaptic glutamate and postsynaptic depolarization). the gp self-interaction adds a second nonlinear pathway after the down projection, operating in 16-dimensional projective geometric algebra space. together they provide two nonlinear pathways per layer.

the layer schedule -- (kda, kda, kda, mamba3, kda, kda, kda, mla) x 3 -- creates functional diversity across layers but not within them. the [[cortical_layers_vs_todorov_layers]] comparison finds that this schedule is structurally reminiscent of cortical hierarchical processing (each 8-layer block as a "cortical area"), but the analogy fails on connectivity: cortical layers have asymmetric, layer-specific wiring (L4 -> L2/3 is strong, L2/3 -> L4 is weak), while all todorov layers read from and write to the same symmetric residual stream. the 3:1 ratio (75% kda, 25% other) was derived from ml benchmarks (kimi, qwen3, olmo), not neuroscience. its convergence with cortical ratios is coincidental.

the architecture has no within-layer recurrence. no source segregation (both swiglu paths receive the same x). no multi-branch independence (one gate for the entire hidden dimension of 2816). no inhibitory populations. no parallel processing across sublayers. each layer is a single-pass feedforward transformation, and recurrence exists only across layers (kda delta-rule state, mamba3 ssm state), not within them.

## the residual stream as communication bus

the [[global_workspace_to_residual_stream]] bridge analysis concludes that the residual stream is topologically a workspace but dynamically a bus. the distinction matters. a global workspace (baars 1988, dehaene and naccache 2001) has four properties the residual stream lacks:

1. ignition threshold -- information enters the workspace only through a nonlinear threshold-crossing event. the residual stream has no threshold; every layer writes unconditionally.
2. selectivity -- processors compete for workspace access and only the winner broadcasts. every todorov layer adds its output without competition.
3. capacity limits -- the workspace holds ~4 items, forcing selection. the residual stream carries the full d_model vector (1024 dimensions) with unlimited superposition.
4. active maintenance -- workspace representations are sustained by recurrent NMDA-mediated feedback loops. the residual stream is passive: add and move on.

the residual stream is a shared bus with full bandwidth, always on, no access control. this is architecturally clean and gradient-friendly, but it is the opposite of what cortex does. cortical communication is selective, capacity-limited, and gated by inhibition at every level.

the quantitative contrast is stark. the residual stream carries 1024 floating-point values per token at every layer -- a theoretical information capacity of ~16,000 bits per layer transition (at fp16). the biological workspace, limited to ~4 items with ~40-bit precision per item (spike count codes), carries ~160 bits. the ratio is ~100x. todorov's "workspace" is 100x wider than biology's, which means it never needs to select. the selection pressure that drives biological workspace competition -- and that gwt argues is the computational purpose of consciousness -- simply does not exist in the residual stream.

## mla as workspace?

a more interesting analogy emerges from mla's compression bottleneck. mla compresses the residual stream from d_model (1024) to d_c (128) -- an 8x bottleneck -- before reconstructing queries, keys, and values. this compression is a genuine capacity-limiting gate: not all information survives the bottleneck. the compressed latent c_t = W_dkv(x_t) forces the network to select which information to preserve for cross-token retrieval.

three of the four workspace properties are partially present in mla:

- capacity limit: the 128-dimensional latent can hold far less than the 1024-dimensional stream. this is a real information bottleneck that forces selection, analogous to the ~4-item workspace limit.
- selectivity: softmax attention over compressed representations selects which past tokens to retrieve. this is competitive -- high attention weight for one token reduces weight for others.
- active maintenance: the kv cache persists compressed representations across the sequence, providing a form of sustained availability (though passively, without recurrent maintenance).

what mla still lacks: ignition. there is no threshold-crossing event that transforms local processing into global broadcast. the compression is smooth and continuous, not all-or-none. and mla operates on the inter-token dimension (selecting which past tokens matter), not the inter-layer dimension (selecting which features to broadcast). the biological workspace selects features for broadcast across processors; mla selects tokens for retrieval within one processor type.

the mla-as-workspace framing is worth exploring not because it is a faithful reproduction of gwt, but because the compression bottleneck creates a genuine information-processing constraint that could force the network toward more selective, workspace-like representations. the 8x compression ratio is a designable parameter: making it more aggressive (16x, 32x) would tighten the bottleneck, pushing mla closer to workspace-like capacity limits at the cost of retrieval fidelity.

there is a deeper structural parallel. in gwt, the workspace is not always active -- it ignites transiently in response to sufficiently strong or attended signals, then broadcasts, then decays. mla layers appear at positions 8, 16, and 24 in the 24-layer stack -- once per block. between mla layers, 7 layers of kda and mamba3 process the stream without global retrieval. this intermittent access to exact attention is structurally reminiscent of intermittent workspace ignition: most processing is "unconscious" (local kda recurrence), punctuated by periodic "broadcast" events (mla retrieval). whether this intermittency creates useful computational structure (forcing kda layers to compress their representations for the upcoming mla checkpoint) or is merely a scheduling artifact is unknown.

## the dendritic gap

the deepest structural gap between cortex and todorov is in local computation per layer. the [[dendritic_computation_to_swiglu]] bridge identifies four critical dimensions where swiglu diverges from dendritic processing:

**compartmentalization.** a pyramidal neuron has ~30-50 independent branches, each computing its own nonlinear function before somatic summation. swiglu has one gate for all 2816 hidden dimensions. the two-layer neuron model predicts that multiple independent subunits increase computational capacity beyond what a single large hidden layer provides, because the block-diagonal structure of W_1 (each branch sees only its own subset of inputs) forces factored representations that generalize better.

**source segregation.** different dendritic compartments receive inputs from different origins: basal dendrites get feedforward thalamic drive, the apical tuft gets top-down cortical feedback, oblique branches get lateral input. swiglu receives the same x on both paths. this means swiglu can learn to selectively gate features of x based on other features of x (self-gating), but it cannot detect coincidence between two independent information streams (cross-source gating). BAC firing specifically requires feedforward-feedback coincidence -- a computation swiglu cannot perform.

**regenerative nonlinearity.** dendritic spikes are threshold-crossing, regenerative events with positive feedback. silu is smooth, monotonic, and continuously differentiable for x > 0. the gidon et al. (2020) dcaap adds a non-monotonic, XOR-capable response in human cortical neurons. the qualitative dynamics differ fundamentally: silu is a graded dimmer switch, dendritic spikes are explosive switches. the ternary spike mechanism before swiglu input provides a hard threshold, but it operates at the somatic level (whole-vector quantization), not the branch level.

**branch-specific state.** NMDA plateau potentials persist for ~100-500 ms per branch, providing local short-term memory without ongoing input. swiglu is stateless -- each token is processed independently. kda and mamba3 provide layer-level persistent state, but there is no per-subunit memory within the feedforward network.

the gap is not in the mathematical operation (both use multiplicative gating) but in the computational architecture: one branch vs many, one source vs many sources, smooth vs regenerative, stateless vs persistent. the [[two_layer_neuron]] model predicts that closing the compartmentalization gap alone (multi-branch swiglu with K=8-32 independent branches at matched parameter count) should increase per-layer capacity.

## within-layer recurrence

todorov layers are single-pass feedforward transformations. cortical layers are recurrent circuits that iterate toward attractors. the canonical microcircuit settles in 2-3 recurrent cycles (~5-10 ms) before producing its output, with the amplification factor 1/(1-g) reflecting the number of effective iterations.

a minimal test of within-layer recurrence: apply the swiglu transformation k=2 times before writing to the residual stream.

    h = rmsnorm(x)
    for i in range(k):
        h = swiglu(h)
    x = x + h

at k=2, this doubles the effective depth of the feedforward network within each layer without adding parameters (weight sharing across iterations). the expected effects:

- attractor dynamics: repeated application of the same nonlinear transformation can sharpen representations by pushing them toward fixed points of the transformation. if swiglu has stable attractors, k=2 would move representations closer to those attractors, potentially improving feature selectivity (analogous to the iceberg effect from recurrent amplification).
- cost: approximately 2x the flops of a single swiglu pass per layer. at 24 layers, this is equivalent to a 48-layer single-pass network in compute, but with only 24 layers of parameters. the parameter efficiency argument is that recurrence at matched compute is more parameter-efficient than depth at matched compute, because weight sharing amortizes the parameter cost.
- risk: if swiglu has no useful attractors (the transformation is contractive toward a single fixed point), k=2 degenerates to a stronger version of a single pass with diminishing returns. if the transformation is expansive, k=2 could produce unstable dynamics.

the biological precedent is strong (recurrent amplification is the dominant mechanism of cortical processing), but the architectural context is different (biology uses recurrence because axonal propagation is slow; serial depth is effectively free in silicon). the empirical question is whether weight-shared depth outperforms unique-weight depth at matched flops.

there is a subtlety in how k=2 interacts with the residual connection. the residual stream adds the output of the k-iterated swiglu to the original x, not the intermediate iterates. this means the recurrence operates inside a skip connection, which stabilizes it (the eigenvalues of the effective jacobian are shifted toward 1). in cortex, the analogous stabilization comes from inhibitory feedback: the recurrent gain g is kept below 1 by PV+ inhibition, preventing divergence while allowing amplification. rmsnorm before the iteration serves a similar role -- it bounds the input magnitude, preventing the iterate from growing without limit.

a 6m-scale test with k in {1, 2, 3} at matched total flops (reduce number of layers proportionally to keep compute constant) would establish whether within-layer recurrence provides any advantage. the key diagnostic: if k=2 improves bpb at matched flops, measure whether the iterate h_2 = swiglu(swiglu(x)) is closer to a fixed point than h_1 = swiglu(x) by computing ||h_2 - swiglu(h_2)||. if this residual is small, the recurrence is converging to an attractor and the biological analogy holds. if not, the second pass is computing a genuinely different transformation and the benefit (if any) comes from effective depth, not attractor dynamics.

an important interaction: kda layers already have across-token recurrence (the delta-rule state S_t accumulates key-value associations). adding within-token recurrence (k=2 swiglu iterations) would create a doubly recurrent system: recurrent across tokens (kda state) and recurrent within tokens (swiglu iterations). cortex has exactly this structure: recurrent across time (persistent firing, synaptic traces) and recurrent within each processing cycle (the 2-3 iteration amplification loop). whether the two forms of recurrence are complementary or redundant in todorov is the central empirical question.

## challenges and counter-arguments

**challenge 1: serial depth may be computationally superior to local recurrence for language.**

cortex evolved local recurrence because axonal propagation is slow (~1-10 m/s) and metabolically expensive. within a column, synaptic transmission takes ~1 ms per hop, so a 3-hop recurrent loop takes ~3 ms -- fast enough for the ~100 ms processing cycle. in silicon, serial depth costs only memory bandwidth and compute, not biological time. the reason cortex is recurrent may be a constraint of wetware, not a computational advantage. transformers with 100+ unique layers outperform any recurrent architecture at matched compute (gpt-4, claude, gemini), suggesting that unique-weight depth captures something that weight-shared recurrence does not: compositional transformations where each layer applies a genuinely different function.

counter-counter: todorov already uses recurrence in kda and mamba3 layers (across tokens, not within layers). the question is specifically about within-layer, within-token recurrence. the strongest argument for it is parameter efficiency: at 300m scale, adding effective depth through weight sharing could outperform adding parameters through wider layers. universal transformers (dehghani et al. 2019) explored weight sharing across layers and found benefits on algorithmic tasks but mixed results on language modeling -- evidence that the question is task-dependent.

**challenge 2: the dendritic gap may not matter at scale.**

point neuron models (which ignore dendritic computation entirely) reproduce many network-level phenomena: oscillations, working memory, decision-making, sequence generation. the [[dendritic_computation]] article notes this explicitly. at the network level, the additional computational capacity per neuron from dendritic branching may be redundant with what is achievable through additional neurons (layers, width). a transformer with d_model=1024 and 24 layers has ~300m parameters, each independently trained. a cortical column has ~80,000 neurons with ~30 branches each, giving ~2.4m hidden units, but with far fewer independently trainable parameters (synaptic plasticity is local and slow). the artificial network may compensate for simpler per-unit computation with vastly more independently optimized parameters.

counter-counter: the two-layer neuron model predicts that multi-branch gating increases the expressiveness of the representational bottleneck (the down projection in swiglu). if this bottleneck limits performance at 300m scale, multi-branch gating could help even at scale. but this is an empirical claim, not a theoretical guarantee. notably, the biological evidence for the two-layer neuron model comes from hippocampal CA1 neurons (poirazi et al. 2003) and L5 pyramidal neurons (polsky et al. 2004) -- neurons with the most elaborate dendritic trees. neurons with simpler morphology (interneurons, L4 spiny stellate cells) are better approximated by point models. the relevant question is whether swiglu units at 300m scale are more like elaborate pyramidal neurons (bottleneck-limited) or simple stellate cells (capacity-sufficient).

**challenge 3: source segregation requires feedback connections, which break feedforward training.**

the most biologically faithful implementation of cross-source gating (dual-input swiglu where gate receives top-down feedback and value receives feedforward input) requires connections from later layers to earlier layers. this creates dependency cycles that standard backpropagation through the layer stack cannot handle without truncation, iteration, or approximation. the training cost of feedback connections is substantially higher than the inference cost, and the optimization landscape may be harder (feedback connections create additional local minima). biology solves this with local learning rules (stdp, hebbian learning); gradient-based training requires either unrolled iteration or biologically implausible credit assignment.

counter-counter: the kda state readout version of dual-input swiglu avoids feedback connections entirely: the gate receives the kda recurrent state (from the previous token) while the value receives the current residual stream. this provides genuinely different information sources without architectural feedback. however, the kda state is derived from previous x values, so the two sources are not truly independent in the way that bottom-up and top-down cortical signals are.

**challenge 4: the residual stream's lack of selectivity may be a feature, not a bug.**

the biological workspace's capacity limits and selectivity evolved under metabolic and bandwidth constraints that do not apply to silicon. the residual stream's unlimited bandwidth (full d_model at every layer) may be strictly superior for language modeling because it preserves all information for potential use by later layers. adding capacity limits (bottleneck gating, sparse broadcast) risks discarding information that turns out to be useful. the [[global_workspace_to_residual_stream]] analysis estimates only 10-20% probability that adding workspace-like constraints would improve bpb at matched parameters. the strongest empirical evidence: every successful transformer-scale model uses an unrestricted residual stream. no production model has adopted gwt-style bottlenecks. mixture-of-experts (MoE) architectures implement a form of selectivity (only k-of-n experts are activated per token), but this gates compute, not communication -- the residual stream remains unrestricted. the closest production system to a workspace bottleneck is deepseek's mla compression, but it was designed for kv cache efficiency, not information selection.

## the honest assessment

todorov instantiates none of the three core principles of cortical local computation: within-layer recurrence, source-segregated inputs, or multi-branch dendritic processing. the residual stream provides global communication but lacks every property that distinguishes a workspace from a bus. the closest biological analogy is mla's compression bottleneck, which creates a genuine capacity limit but operates on the wrong dimension (inter-token vs inter-layer).

the dendritic gap is the most concrete and testable. multi-branch swiglu (K=8 independent gating subunits at matched parameter count) addresses compartmentalization without requiring feedback connections or training procedure changes. the two-layer neuron model provides a quantitative prediction (independent subunits > one large unit). the cost of testing is low (parameter-neutral architectural change, one 6m-scale run). this is the single most actionable intervention suggested by the local computation analysis.

within-layer recurrence (k=2) is the second most actionable. it is parameter-free (weight sharing), biologically motivated (recurrent amplification), and testable at 6m scale. but the empirical evidence from ml (deeper unique-weight networks outperform shallower weight-shared ones at matched compute) creates a strong prior against it.

source segregation is the hardest gap to close. it requires either feedback connections (breaking feedforward training) or an alternative second information source (kda state readout, mean-pooled context). the kda state readout is implementable but may not provide genuinely independent information. true cross-source gating requires true architectural feedback, which is a fundamental change. the [[cortical_microcircuit_to_layer_schedule]] bridge identifies layer-type-specific residual gating (different learned masks for kda, mamba3, and mla inputs) as a lightweight approximation to source segregation -- not true cross-source gating, but at least layer-type-specific input selection. the cost is negligible (3 x d_model parameters), the expected benefit is uncertain (20-30% probability of meaningful improvement), and it is testable at 6m scale.

the overall pattern: cortex achieves computational power through local complexity (multi-branch neurons, source-segregated inputs, recurrent dynamics) and global selectivity (workspace bottlenecks, ignition thresholds, capacity limits). todorov achieves computational power through global simplicity (uniform residual stream, single-branch layers, serial depth) and local simplicity (one gate, one source, one pass). the two strategies are not obviously ordered -- cortex optimizes under metabolic and bandwidth constraints that do not apply to silicon, while todorov optimizes under gradient flow and parallelism constraints that do not apply to wetware. the question is whether any of the biological local-computation principles transfer to the silicon context in a way that improves performance at matched cost.

the ranking of interventions by tractability:

1. multi-branch swiglu (K=8): parameter-neutral, no training procedure changes, testable at 6m. highest prior probability of benefit (~15-25%).
2. within-layer recurrence (k=2): parameter-free, moderate flop cost, testable at 6m. moderate prior (~10-15%).
3. layer-type-specific gating: negligible parameter cost, testable at 6m. moderate prior (~20-30%) but lower expected magnitude.
4. mla compression sweep (d_c = 64, 128, 256): changes one hyperparameter, testable at 6m. informative for workspace theory but may not improve bpb.
5. dual-input swiglu with kda state: requires architectural change, moderate complexity. lowest prior (~5-10%).

the triple criterion for any intervention: mathematical identity (the operation must be formally specified), ablation (removing it must degrade performance), quantitative prediction (the improvement must be predicted before measurement, not rationalized after). none of these interventions have passed any gate. they are hypotheses, not recommendations.

## see also

- [[canonical_microcircuit]]
- [[dendritic_computation]]
- [[dendritic_spikes]]
- [[apical_amplification]]
- [[two_layer_neuron]]
- [[inhibitory_interneurons]]
- [[global_workspace_to_residual_stream]]
- [[dendritic_computation_to_swiglu]]
- [[cortical_microcircuit_to_layer_schedule]]
- [[cortical_layers_vs_todorov_layers]]
