# bridge: cortical microcircuit to todorov layer schedule

status: current (as of 2026-04-16).

## the biological mechanism

the neocortex processes information through a [[canonical_microcircuit]] replicated across cortical areas. within each [[cortical_column]], six functionally distinct layers (see [[laminar_processing]]) are connected by specific, asymmetric pathways:

- L4 receives thalamic input and projects to L2/3
- L2/3 performs recurrent associative processing and projects to L5 and to other cortical areas
- L5 generates subcortical output (motor commands, etc.)
- L6 modulates the thalamus (feedback control of input)
- L1 receives top-down modulatory signals from higher areas
- inhibitory interneurons (~20% of neurons) maintain [[excitatory_inhibitory_balance]] through fast feedback

the critical computational properties of this architecture:
1. recurrent amplification: weak feedforward input is amplified ~5-20x by L2/3 recurrent excitation
2. layer-specific routing: different information flows through different layers
3. parallel processing: all layers are active simultaneously within a column
4. hierarchical areas: the same circuit is replicated at multiple levels (V1->V2->V4->IT in vision)
5. gain control: inhibitory neurons prevent runaway excitation while preserving selectivity

the biological ratios (excitatory neurons, from Potjans and Diesmann 2014): L2/3 ~33%, L4 ~35%, L5 ~8%, L6 ~23%.

## the current todorov implementation

layer schedule: (KDA, KDA, KDA, Mamba3, KDA, KDA, KDA, MLA) x 3

this means:
- 18 KDA layers: delta-rule recurrence, associative memory
- 3 Mamba3 layers: SSM with complex rotation, temporal dynamics
- 3 MLA layers: compressed exact attention, retrieval
- all layers have identical connectivity: residual stream -> RMSNorm -> layer -> add back
- no laminar-specific routing (every layer sees the same residual stream)
- the pattern repeats 3 times (3 "blocks")

the 3:1 ratio (KDA:other) was adopted because:
- Kimi, Qwen3, OLMo validated 3:1 hybrid ratio independently
- systematic analysis papers show performance saturates at 3:1 to 6:1
- this is an ENGINEERING decision, not a biological one

see [[cortical_layers_vs_todorov_layers]] for the detailed adversarial comparison. summary: the mapping between cortical layers and todorov layers fails on connectivity (homogeneous vs heterogeneous), processing mode (serial vs parallel), and ratio provenance (ML benchmarks vs evolution).

## the proposed change

cortical laminar organization suggests three modifications to todorov's architecture, ordered by expected impact and implementation difficulty:

### modification 1: within-block feedback connections (medium impact, medium difficulty)

cortical columns have feedback from deep layers (L5/6) to superficial input layers (L1/L4). todorov blocks have no within-block feedback: information flows strictly forward from layer 1 to layer 24.

proposed change: add a weighted skip connection from the MLA layer (position 8 in each block) back to the input of the first KDA layer in the NEXT block. this approximates the cortico-cortical feedback pathway (L5 output of one area feeds into L4 of the next area).

implementation:

    for block_idx in range(n_blocks):
        x = block_layers[block_idx](x)
        if block_idx > 0:
            x = x + feedback_gate * prev_block_output
        prev_block_output = x

where feedback_gate is a learnable scalar per block. this adds 3 parameters (one per block boundary) and approximately zero compute overhead.

expected impact: LOW. the residual stream already carries all information from previous layers. an explicit feedback connection would be redundant unless combined with layer-specific routing (modification 2). the cortical feedback pathway is powerful because it targets DIFFERENT layers than the feedforward pathway. without that asymmetry, feedback is just another skip connection.

### modification 2: layer-type-specific residual gating (high impact, medium difficulty)

cortical layers receive DIFFERENT inputs because of layer-specific connectivity. todorov layers all receive the same residual stream. the simplest approximation of layer-specific routing: gate the residual stream differently for different layer types.

proposed change: before each layer, apply a learnable projection that selects a subspace of the residual stream appropriate for that layer type.

    if layer_type == 'kda':
        layer_input = rmsnorm(x) * gate_kda
    elif layer_type == 'mamba3':
        layer_input = rmsnorm(x) * gate_mamba3
    elif layer_type == 'mla':
        layer_input = rmsnorm(x) * gate_mla

where gate_X is a learnable vector of dimension d_model that weights which residual stream dimensions are visible to each layer type. this adds 3 * d_model parameters (~3K at d_model=1024) and one element-wise multiply per layer.

expected impact: MODERATE. this gives different layer types access to different information, approximating laminar-specific routing. the biological analogy: L4 receives thalamic input (external, sensory), L2/3 receives local recurrent input (internal, associative), L5 receives both. the gate would let KDA layers attend to "associative" dimensions, Mamba3 layers to "temporal" dimensions, and MLA layers to "retrieval" dimensions.

risk: the gating may learn to be uniform (all gates converge to ones), making it a no-op. to prevent this, the gates could be initialized to different random subsets of dimensions, or an auxiliary diversity loss could encourage gate differentiation.

### modification 3: explicit inhibitory normalization (low impact, low difficulty)

cortical [[excitatory_inhibitory_balance]] provides dynamic gain control: inhibition tracks excitation with ~1 ms delay, preventing runaway activity while enabling transient responses. todorov uses static RMSNorm, which normalizes the magnitude of the residual stream but does not provide the dynamic, activity-dependent gain control of biological inhibition.

proposed change: replace RMSNorm before selected layers with a dynamic normalization that depends on the recent history of the residual stream.

    running_var = ema_decay * running_var + (1 - ema_decay) * rms(x)^2
    x_normed = x / sqrt(running_var + eps)

where ema_decay is a learnable parameter controlling the timescale of the "inhibitory" response. this adds 1 parameter per layer and approximately zero compute overhead.

expected impact: LOW. RMSNorm already provides adequate normalization for gradient-based optimization. the biological advantage of dynamic E/I balance (fast transient responses, temporal precision) is unlikely to help in a serial architecture where layers do not process input in real time. this modification is biologically motivated but may not translate to a useful computational advantage.

## implementation spec

only modification 2 (layer-type-specific residual gating) is recommended for testing, as it is the only one with a plausible mechanism for improving performance beyond what the residual stream already provides.

in train.py, the layer forward pass currently looks like:

    x = x + layer(rmsnorm(x))

the modified version:

    gate = self.layer_gate  # shape: (d_model,), learnable, initialized to ones
    x = x + layer(rmsnorm(x) * gate)

where self.layer_gate is a nn.Parameter with separate instances for each layer type. initialization: ones for all gates (start as identity, learn specialization).

parameter count: 3 * d_model = 3072 at 300M scale (d_model=1024). negligible.

training protocol: run as a controlled experiment with matched compute (per phase 5 sequencing protocol). compare against the current architecture on BPB. monitor gate values during training to determine whether the gates learn meaningful specialization.

## expected impact

modification 2 has a reasonable chance of producing a measurable improvement (0.5-2% BPB reduction) if different layer types benefit from attending to different subspaces of the residual stream. the biological motivation is clear: cortical layers specialize partly because they receive different inputs. giving todorov layers the ability to select their inputs could enable similar specialization.

the probability of a meaningful improvement is estimated at 20-30%. the cost of testing is negligible (3K additional parameters, one training run). the risk is that the gates converge to uniform values, indicating that residual stream specialization is not useful for this architecture.

## risk assessment

    risk                              severity    probability    mitigation
    gates learn to be uniform         low         50%            diversity loss, different initialization
    gates create optimization         medium      15%            initialize to ones (identity), small LR for gates
     instability
    gates interact badly with         medium      10%            test at 6M scale first, then 300M
     ternary spikes
    improvement is real but < noise   low         30%            run 3 seeds, statistical test

## what this bridge does NOT recommend

1. adding biological-style inhibitory neurons. the computational cost of maintaining an explicit inhibitory population (20% additional parameters, inhibitory-to-excitatory connectivity) is not justified by the expected benefit in a gradient-trained system.

2. converting todorov to parallel layer processing. this would require fundamentally rearchitecting the forward pass and would likely reduce performance (serial processing enables the deep compositional computation that makes transformers effective).

3. attempting to replicate cortical laminar ratios (33% L2/3, 35% L4, 8% L5, 23% L6). these ratios evolved for spatial/temporal sensory processing, not language. the 3:1 ratio derived from ML benchmarks is better suited to todorov's task.

## key references

- Douglas, R. J. and Martin, K. A. C. (2004). neuronal circuits of the neocortex. annual review of neuroscience, 27, 419-451.
- Bastos, A. M. et al. (2012). canonical microcircuits for predictive coding. neuron, 76(4), 695-711.
- Harris, K. D. and Shepherd, G. M. G. (2015). the neocortical circuit: themes and variations. nature neuroscience, 18(2), 170-181.
- Potjans, T. C. and Diesmann, M. (2014). the cell-type specific cortical microcircuit. cerebral cortex, 24(3), 785-806.

## see also

- [[cortical_column]]
- [[laminar_processing]]
- [[canonical_microcircuit]]
- [[excitatory_inhibitory_balance]]
- [[cortical_layers_vs_todorov_layers]]
- [[plasticity_to_kda_delta_rule]]
