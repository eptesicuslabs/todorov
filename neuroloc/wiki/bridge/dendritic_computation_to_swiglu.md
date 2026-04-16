# bridge: dendritic computation to SwiGLU gating

status: current (as of 2026-04-16).

## the biological mechanism

dendrites are active computational elements that perform local, nonlinear transformations of their inputs before the soma integrates them (see [[dendritic_computation]]). the key computational principles are:

1. **multiplicative gating.** NMDA receptors require BOTH presynaptic glutamate AND postsynaptic depolarization (Mg2+ block relief) to conduct. this is a biological AND gate -- a multiplicative interaction between two signals. dendritic spikes (Na+, Ca2+, NMDA; see [[dendritic_spikes]]) are triggered only when local input exceeds a threshold, creating a nonlinear gate.

2. **multi-branch independence.** a pyramidal neuron has ~30-50 thin terminal dendrites, each functioning as an independent nonlinear subunit. the neuron as a whole is computationally equivalent to a two-layer neural network with ~30-50 hidden units (see [[two_layer_neuron]], Poirazi et al. 2003).

3. **source-segregated inputs.** different dendritic compartments receive inputs from different sources: basal dendrites receive feedforward input (from L4/thalamus), the apical tuft receives feedback input (from higher cortical areas via L1). BAC firing (see [[apical_amplification]]) requires coincidence of bottom-up and top-down signals.

4. **regenerative nonlinearity.** dendritic spikes are threshold-crossing, regenerative events -- once initiated, positive feedback drives a rapid, large-amplitude depolarization. this is qualitatively different from smooth, monotonic activation functions. the Gidon et al. (2020) dCaAP adds a further twist: a graded, inverted-U nonlinearity enabling XOR computation.

## the current todorov implementation

SwiGLU (src/layers/swiglu.py):

    gate = silu(W_gate(x))
    up = W_up(x)
    out = W_down(gate * up)

    if spatial_mode:
        out = out + gp_proj(geometric_product(W_left(x), W_right(x)))

the multiplicative gating (silu(W_gate(x)) * W_up(x)) involves:
- two independent linear transformations of the same input x
- one path produces a gate signal after silu nonlinearity
- the other produces the value to be gated
- element-wise multiplication selectively passes information
- down projection compresses back to d_model

the hidden dimension is int(d_model * 2.75), rounded up to nearest 64. for d_model=1024 at 267m scale, this gives hidden_dim=2816. the gate and up paths each have d_model x hidden_dim parameters.

when spatial_mode=True, the GP self-interaction adds:
- W_left and W_right project x to 16-dim multivectors
- geometric_product computes the PGA product
- gp_proj projects back to d_model
- the result is ADDED to the SwiGLU output (residual, not gated)

## adversarial analysis: is SwiGLU analogous to dendritic gating?

### the argument FOR

**claim 1: SwiGLU gate = dendritic coincidence detection.**

silu(W_gate(x)) * W_up(x) is a multiplicative interaction between two independent linear projections of the input. the gate path (silu(W_gate)) determines WHICH features pass, while the up path (W_up) provides the VALUES that pass. this maps to dendritic coincidence detection: information only flows when both the gating signal and the input signal are present.

strength of this claim: MODERATE. the mathematical operation (element-wise multiplication of two independent transformations) is genuinely analogous to the fundamental operation of dendritic gating. the NMDA receptor requires both presynaptic glutamate (analogous to W_up providing value) and postsynaptic depolarization (analogous to silu(W_gate) providing gate). the analogy holds at the level of the multiplicative interaction.

**claim 2: hidden_dim expansion = dendritic fanout.**

SwiGLU expands from d_model to ~2.75x d_model. a pyramidal neuron with 30-50 dendritic branches, each with ~50-200 synapses, has a fanout ratio of ~1500-10000 synaptic inputs to ~1 somatic output. the expansion ratio in SwiGLU is much smaller (2.75x vs ~1000x+), but the architectural pattern -- expand to higher dimension, perform nonlinear processing, compress back -- is shared.

strength of this claim: WEAK. the expansion ratios differ by orders of magnitude. more importantly, the biological expansion creates INDEPENDENT subunits (branches), while SwiGLU creates a single high-dimensional space where all dimensions share the same gate and value. the analogy confuses dimensionality expansion with computational compartmentalization.

**claim 3: GP self-interaction = second dendritic nonlinearity.**

the geometric product self-interaction (geometric_product(W_left(x), W_right(x))) is a second nonlinear computation on the same input, added to the SwiGLU output. this is analogous to having a second dendritic branch with a different nonlinear integration rule (geometric product instead of element-wise product).

strength of this claim: WEAK-MODERATE. the GP self-interaction does provide a second independent nonlinear pathway operating on x, which is structurally analogous to a second dendritic branch. however, the GP operates in a 16-dimensional space (vs hidden_dim=2816 for SwiGLU), so it is a minor contributor. and the GP output is additive (not gated), which corresponds to a dendritic branch whose output always reaches the soma, regardless of somatic activity. in biology, even branch outputs are modulated by dendritic tree interactions.

**claim 4: silu ≈ dendritic nonlinearity.**

silu(x) = x * sigmoid(x) is a smooth nonlinearity that maps (approximately): strong negative input -> 0, near-zero input -> ~0, strong positive input -> ~x. this creates a soft gate that passes positive features and blocks negative features, similar to how a dendritic spike threshold passes strong input and blocks weak input.

strength of this claim: WEAK. silu is smooth and monotonic for x > 0. dendritic spikes are regenerative, threshold-crossing events with distinct on/off states and (in the case of dCaAPs) non-monotonic amplitude. the qualitative dynamics are fundamentally different: silu is a graded gate, dendritic spikes are explosive events.

### the argument AGAINST

**criticism 1: SwiGLU uses ONE gate for the entire hidden dimension.**

dendritic computation derives its power from having MANY INDEPENDENT branches (~30-50 per neuron), each performing its own nonlinear integration. this is what makes the two-layer neuron model work: J independent hidden units, not one large hidden layer.

SwiGLU has ONE gate path and ONE value path. the element-wise multiplication applies the SAME gating function to ALL hidden dimensions simultaneously. there is no branch independence, no compartmentalization, no factored computation.

to match the biological model, SwiGLU would need to be split into ~8-32 independent sub-gates, each receiving a different subset of the input and producing an independent gated output. this would be "multi-branch SwiGLU" or equivalently "multi-head FFN."

severity: HIGH. this is the single biggest structural difference. SwiGLU is a one-branch neuron; biology uses ~30-50 branches.

**criticism 2: SwiGLU has no source segregation.**

in biology, different dendritic compartments receive input from DIFFERENT sources (feedforward vs feedback, different brain regions, different cortical layers). BAC firing requires coincidence of bottom-up (soma) and top-down (apical tuft) signals from DIFFERENT origins.

in SwiGLU, both W_gate and W_up receive the SAME input x (the residual stream). the two paths are not receiving different information -- they are computing different linear projections of identical information. this is self-gating, not cross-source gating.

the computational consequence: SwiGLU can learn to selectively pass or block features of x based on other features of x, but it cannot implement the key biological function of detecting coincidence between two INDEPENDENT information streams. apical amplification requires feedforward-feedback coincidence; SwiGLU can only do feedforward-feedforward coincidence.

severity: HIGH. this is a fundamental architectural difference that no parameter change can fix within the current single-input architecture.

**criticism 3: silu is not a regenerative nonlinearity.**

biological dendritic spikes are threshold-crossing events with positive feedback: once the voltage reaches threshold, ion channel opening drives further depolarization, creating a rapid, large-amplitude event. the spike is either triggered or not -- it has (approximately) binary output. the temporal dynamics (rising phase, plateau, repolarization) carry computational information.

silu is a smooth, monotonic, continuously differentiable function with no threshold, no positive feedback, no temporal dynamics, and no binary output. it is a soft gate, not a spike.

would replacing silu with a harder gate (e.g., step function with STE, threshold nonlinearity) make SwiGLU more dendritic? possibly, but this is exactly what the ternary spike mechanism already does before SwiGLU input. the spikes provide the hard threshold; SwiGLU provides the soft gating. the two mechanisms together (spike -> SwiGLU) capture different aspects of dendritic computation: spike = threshold, SwiGLU = gating.

severity: MODERATE. the smooth/hard distinction matters for the type of computation (analog gating vs digital switching) but SwiGLU's gradient-friendly silu may be computationally preferable to a hard threshold in the gating path, even if it is less biological.

**criticism 4: dendritic computation is spatial; SwiGLU is not.**

in biology, which synapses land on which branch is determined by physical location. synapses from one presynaptic source cluster on specific branches. this spatial structure creates an inductive bias: inputs that should be jointly processed are physically co-located on a branch.

SwiGLU has no spatial structure. all dimensions of x are processed identically by W_gate and W_up. there is no constraint that certain input dimensions should be processed together or that certain gate values should apply to specific subsets of the value.

severity: MODERATE. this could be addressed by using group-wise or block-diagonal projections in W_gate and W_up, enforcing that subsets of dimensions are processed independently. this is the block-diagonal W_1 constraint from the Poirazi model.

**criticism 5: dendritic branches have persistent state; SwiGLU is stateless.**

NMDA plateau potentials last ~100-500 ms, maintaining a local depolarization per branch without ongoing input. this gives each branch a form of short-term memory. SwiGLU is a purely feedforward, stateless transformation: each token is processed independently with no carry from previous tokens.

severity: LOW-MODERATE. the recurrent state in KDA and Mamba3 provides persistent memory at the layer level. the question is whether per-branch (per-subunit within SwiGLU) persistent state would add value. at current scale, the complexity cost likely outweighs the benefit.

### verdict

SwiGLU's multiplicative gating IS analogous to dendritic coincidence detection at the most abstract level: both are multiplicative interactions between two independent transformations. but the analogy breaks on four critical dimensions:

1. **compartmentalization**: ONE gate vs MANY independent branches (HIGH impact)
2. **source segregation**: same input to both paths vs different input sources (HIGH impact)
3. **nonlinearity type**: smooth silu vs regenerative threshold-crossing (MODERATE impact)
4. **spatial structure**: unstructured vs spatially constrained input routing (MODERATE impact)

the honest summary: SwiGLU is a SINGLE-BRANCH, SELF-GATING, SMOOTH version of what biology does with MANY-BRANCH, CROSS-SOURCE, REGENERATIVE dendritic computation. the mathematical operation (multiplicative gating) is shared. the computational architecture (independent subunits, source-segregated inputs, threshold dynamics) is not.

## the proposed change

based on the adversarial analysis, three modifications are identified, ranked by expected impact:

### modification 1: multi-branch SwiGLU (addresses criticism 1)

replace the single gate-value pair with K independent branches:

    branches = []
    for k in range(K):
        gate_k = silu(W_gate_k(x[:, subset_k]))
        up_k = W_up_k(x[:, subset_k])
        branches.append(gate_k * up_k)
    out = W_down(concat(branches))

where subset_k is a subset of input dimensions assigned to branch k (block-diagonal constraint). K=8 would match KDA head count; K=32 would approach biological branch count.

parameter cost: approximately equal to standard SwiGLU if total hidden_dim is preserved (each branch has hidden_dim/K dimensions). the block-diagonal constraint actually REDUCES parameters compared to dense projections.

expected impact: MODERATE. multi-branch gating increases the rank of the representational bottleneck (SwiGLU's down projection) and enforces factored computation. however, the optimal branch count K is unknown and may interact with the attention head count.

### modification 2: dual-input SwiGLU (addresses criticism 2)

provide SwiGLU with two inputs: the residual stream x (bottom-up) and a second signal c (context/top-down):

    gate = silu(W_gate(c))
    up = W_up(x)
    out = W_down(gate * up)

where c could be:
- output from a later layer (requires feedback connections, breaks feedforward training)
- a compressed representation of the global context (mean-pooled residual stream)
- the KDA state readout (connects recurrent memory to feedforward gating)

parameter cost: one additional projection (W_context: d_model -> d_model) per layer, plus the mechanism to provide c.

expected impact: LOW-MODERATE. the feedback connection version would be a fundamental architecture change with unknown training stability. the KDA state readout version is more practical but may not provide genuinely "different" information from x (the KDA state is derived from previous x values).

### modification 3: group-wise gating (addresses criticism 4, partial fix for criticism 1)

replace the dense W_gate and W_up with group-wise projections:

    W_gate: (d_model) -> (hidden_dim), but block-diagonal with G groups
    W_up: (d_model) -> (hidden_dim), same block-diagonal structure

each group of d_model/G input dimensions is processed independently, creating G semi-independent "branches" without explicit branch loops.

parameter cost: REDUCES parameters by factor G (block-diagonal has 1/G the parameters of dense). can reinvest the saved parameters in more branches or larger hidden dimension.

expected impact: LOW. group convolution is well-studied in vision models and provides efficiency but rarely improves quality at matched parameter count. the inductive bias (independent processing of dimension groups) may or may not match the structure of language representations.

## implementation spec

for modification 1 (multi-branch SwiGLU), changes to src/layers/swiglu.py:

    class MultiBranchSwiGLU(nn.Module):
        def __init__(self, d_model, ratio=2.75, n_branches=8, ...):
            super().__init__()
            branch_input = d_model // n_branches
            branch_hidden = int(branch_input * ratio)
            branch_hidden = ((branch_hidden + 15) // 16) * 16

            self.branches = nn.ModuleList([
                nn.ModuleDict({
                    'w_gate': nn.Linear(branch_input, branch_hidden, bias=False),
                    'w_up': nn.Linear(branch_input, branch_hidden, bias=False),
                })
                for _ in range(n_branches)
            ])
            self.w_down = nn.Linear(branch_hidden * n_branches, d_model, bias=False)

        def forward(self, x):
            chunks = x.chunk(self.n_branches, dim=-1)
            branch_outs = []
            for chunk, branch in zip(chunks, self.branches):
                gate = F.silu(branch['w_gate'](chunk))
                up = branch['w_up'](chunk)
                branch_outs.append(gate * up)
            return self.w_down(torch.cat(branch_outs, dim=-1)), {}

## expected impact

positive:
- multi-branch gating increases the factored capacity of the FFN layer
- block-diagonal structure reduces parameter count, allowing larger hidden dim or more branches
- matches the biological inductive bias of compartmentalized computation
- could improve spike MI by providing more structured pathways for information to be selectively gated

negative:
- branch independence may prevent beneficial cross-dimension interactions that dense SwiGLU captures
- the optimal branch count K is unknown and adds a hyperparameter
- chunked processing may not benefit from GPU parallelism as well as dense matmuls
- interaction with the existing spike mechanism (spikes before SwiGLU input) is untested

estimated probability of meaningful BPB improvement at matched parameters: 15-25%.

## risk assessment

1. **branch count hyperparameter.** adding K as a hyperparameter creates a new axis for ablation. wrong K could degrade performance. mitigation: sweep K in {1, 2, 4, 8, 16} on a small-scale pilot.

2. **GPU efficiency.** block-diagonal matmuls are less efficient than dense matmuls on modern GPUs. mitigation: use grouped linear operations (torch.nn.functional.group_norm equivalent for linear layers) or reshape to exploit batch parallelism.

3. **interaction with spikes.** if spikes zero out most of the input before SwiGLU, the branches may receive near-zero input for many dimensions, causing dead branches. mitigation: monitor per-branch activation statistics during training.

4. **this is a phase 6+ modification.** phase 5 must complete first (see CLAUDE.md phase sequencing). do not implement during phase 5 runs.

## prerequisite knowledge

- [[dendritic_computation]] -- why dendrites are computational
- [[dendritic_spikes]] -- the nonlinearities that create branch independence
- [[two_layer_neuron]] -- the formal model of neuron as two-layer network
- [[apical_amplification]] -- the specific case of two-source input integration
- [[sparse_coding_to_ternary_spikes]] -- how spikes interact with SwiGLU

## related bridge docs

- [[neuron_models_to_atmn]] -- single neuron model analysis
- [[lateral_inhibition_to_adaptive_threshold]] -- another gating mechanism analysis

## open questions

1. does multi-branch SwiGLU at matched parameter count outperform standard SwiGLU? the two-layer neuron model predicts yes (independent subunits > one large unit), but this is not established in practice for language models.

2. should branches share the down projection (W_down is dense, combining all branches) or have independent down projections (each branch projects to its own d_model subset)? shared W_down allows cross-branch interaction at the output; independent W_down enforces full branch independence.

3. the GP self-interaction (when spatial_mode=True) already provides a second nonlinear pathway. if multi-branch SwiGLU is implemented, should the GP be integrated as one of the branches (branch-level geometric product) or kept as a separate additive residual?

4. the biological model has branch-specific plasticity (see [[stdp]], [[bcm_theory]]). could per-branch learning rate scaling (different Adam beta values per branch) improve training of multi-branch SwiGLU?
