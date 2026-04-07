# cortical layers vs todorov layers

## the question

the neocortex processes information through 6 functionally distinct layers with specific, non-uniform connectivity. todorov processes information through 24 layers in a repeating pattern of (KDA, KDA, KDA, Mamba3, KDA, KDA, KDA, MLA) x 3. does todorov's layer schedule map to cortical laminar organization? does the 3:1 ratio (75% KDA, 12.5% Mamba3, 12.5% MLA) correspond to any biological ratio?

## cortical layer architecture

see [[laminar_processing]] for full details. summary:

    layer    fraction of neurons    primary function                 connectivity
    L1       <5%                    modulatory (top-down feedback)   receives feedback from higher areas
    L2/3     ~30%                   associative processing           corticocortical, recurrent within layer
    L4       ~20%                   sensory input processing         receives thalamic input, projects to L2/3
    L5       ~20%                   subcortical output               receives from L2/3, projects subcortically
    L6       ~15%                   thalamic feedback                projects to thalamus
    inhibitory ~20% (distributed)  gain control, E/I balance        local, all layers

key architectural properties:
- layers connect to each other with SPECIFIC, ASYMMETRIC connectivity (L4->L2/3 is strong; L2/3->L4 is weak)
- each layer has DISTINCT cell types (spiny stellate in L4, large pyramidal in L5, CT neurons in L6)
- information flows through a DEFINED pathway: thalamus->L4->L2/3->L5->subcortical
- feedback and feedforward streams use DIFFERENT layers (feedforward from superficial, feedback from deep)
- all layers process inputs SIMULTANEOUSLY within a column (parallel, not serial)

## todorov layer architecture

layer schedule: (KDA, KDA, KDA, Mamba3, KDA, KDA, KDA, MLA) x 3 = 24 layers

    layer type    count    fraction    primary function
    KDA           18       75%         delta-rule recurrence, associative memory
    Mamba3        3        12.5%       SSM with complex rotation, temporal dynamics
    MLA           3        12.5%       compressed exact attention, retrieval

key architectural properties:
- all layers share the SAME residual stream (x -> RMSNorm -> layer -> x + output)
- no layer-specific routing (every layer sees the same input)
- no asymmetric connectivity (the residual stream is fully symmetric)
- layers process inputs SEQUENTIALLY (serial stack, not parallel)
- the pattern repeats 3 times (3 "blocks" of 8 layers each)

## dimension 1: layer type ratios

cortical excitatory neuron ratios (approximate, from Potjans and Diesmann 2014):

    L2/3: 33.4% (20,683 / 61,843)
    L4:   35.4% (21,915 / 61,843)
    L5:    7.8% ( 4,850 / 61,843)
    L6:   23.3% (14,395 / 61,843)

todorov layer type ratios:

    KDA:   75.0% (18/24)
    Mamba3: 12.5% (3/24)
    MLA:   12.5% (3/24)

the superficial mapping attempt: KDA = L2/3 (both do associative processing with recurrent dynamics), Mamba3 = L4 (both process temporal input), MLA = L5 (both perform readout/retrieval for downstream consumption)?

this mapping fails on multiple grounds:

1. the ratios do not match. L2/3 is 33%, not 75%. L4 is 35%, not 12.5%. the cortical column devotes far more resources to input processing (L4) relative to associative processing (L2/3) than todorov does.

2. L6 (23% of excitatory neurons) has no todorov analogue. there is no layer type in todorov that sends feedback to an "input source" to modulate future input. the closest analogue would be a layer that modifies the data pipeline, not the model.

3. the inhibitory population (20% of all cortical neurons) has no todorov analogue. todorov has no dedicated "gain control" layers. gain control is distributed across all layers via RMSNorm (static) and ternary spike thresholds (adaptive but not inhibitory in the biological sense).

## dimension 2: connectivity architecture

this is the most fundamental difference.

cortical connectivity is LAYER-SPECIFIC and ASYMMETRIC:

    L4 -> L2/3 (strong feedforward)
    L2/3 -> L5 (strong descending)
    L2/3 -> L2/3 (strong recurrent)
    L5 -> subcortical (output)
    L6 -> thalamus (feedback)
    L1 <- higher areas (modulatory)
    L2/3 -> L4 (weak or absent)
    L5 -> L4 (weak)

todorov connectivity is UNIFORM:

    all layers: residual_stream -> RMSNorm -> layer -> residual_stream + output

every layer in todorov receives the same input (the residual stream) and adds its output back to the same stream. there is no equivalent of the cortical wiring asymmetry where L4 projects strongly to L2/3 but not vice versa, or where L5 receives from L2/3 but projects subcortically.

implication: cortical layers can perform different computations on different representations because they receive different inputs. todorov layers all operate on the same representation and can only modify it incrementally.

## dimension 3: parallel vs serial processing

cortical layers within a column process information SIMULTANEOUSLY. L4, L2/3, L5, and L6 are all active at the same time, with information flowing between them through direct synaptic connections. the latency from L4 activation to L5 output is ~10-20 ms (a few synaptic delays).

todorov layers process information SEQUENTIALLY. layer 1 must complete before layer 2 begins. information from layer 1 reaches layer 24 only after passing through all intermediate layers.

this difference changes what kinds of computations are possible. parallel processing within a cortical column enables:
- simultaneous feedforward and feedback processing (L4 is still processing new input while L5 is outputting the result of the previous cycle)
- rapid convergence through recurrent dynamics (L2/3 recurrent excitation settles in 2-3 cycles, ~5-10 ms)
- multiplexing of different signals in different layers at the same time

serial processing in todorov enables:
- progressive refinement (each layer refines the representation from the previous layer)
- compositional computation (later layers build on the outputs of earlier layers)
- but no within-block recurrence (information cannot flow backward within a block)

## dimension 4: the 3-block repetition

todorov repeats its 8-layer block 3 times. does this map to cortical hierarchy?

the cortical hierarchy (e.g., V1->V2->V4 in vision) consists of multiple cortical areas, each containing the full canonical microcircuit. each area processes its input through the canonical circuit and passes the result to the next area. this is a hierarchy of canonical circuits, each containing all layers.

todorov's 3-block repetition is structurally similar: each block contains all three layer types (KDA, Mamba3, MLA), and blocks are stacked hierarchically. this is the strongest structural analogy between todorov and cortical organization.

within each block, the pattern KDA-KDA-KDA-Mamba3-KDA-KDA-KDA-MLA could be interpreted as:
- KDA layers (positions 1-3): initial associative processing (analogous to L2/3 recurrent processing)
- Mamba3 (position 4): temporal state consolidation (analogous to L4 integrating new input with ongoing state)
- KDA layers (positions 5-7): further associative processing (analogous to L2/3 output computation)
- MLA (position 8): global retrieval/readout (analogous to L5 output to downstream areas)

this interpretation is suggestive but should not be taken too seriously. it is a post-hoc narrative imposed on an architecture that was designed from ML benchmarks, not neuroscience.

## dimension 5: the origin of the 3:1 ratio

todorov's 3:1 ratio (75% KDA : 25% other) was adopted because:

1. Kimi (Moonshot AI) validated a 3:1 hybrid ratio in their production architecture
2. Qwen3 (Alibaba) uses a similar ratio
3. OLMo (AI2) adopted comparable ratios
4. systematic analysis papers show performance saturates between 3:1 and 6:1 ratios of attention-like to state-space layers

this is an ENGINEERING decision derived from ML benchmark performance, not a biological finding. the fact that the ratio happens to be numerically close to some cortical ratios (L2/3 = ~33% = roughly 1:2 with other layers) is coincidental.

the strongest argument against the biological mapping: the 3:1 ratio in todorov was independently derived by multiple ML labs (Kimi, Qwen3, OLMo) who were not thinking about cortical columns. convergent evolution in engineering does not imply shared mechanism with biology.

## verdict

**the cortical laminar architecture and todorov's layer schedule are NOT meaningfully analogous.** the fundamental differences are:

1. cortical layers have HETEROGENEOUS connectivity; todorov layers have HOMOGENEOUS connectivity (residual stream)
2. cortical layers process in PARALLEL; todorov layers process in SERIAL
3. cortical layers have DISTINCT cell types; todorov layer types (KDA, Mamba3, MLA) differ in computation but share the same input/output interface
4. the 3:1 ratio was derived from ML benchmarks, not biology

**however, there is one structural similarity worth investigating:** the 3-block repetition in todorov resembles cortical hierarchical processing (V1->V2->V4), where each level applies the full local circuit to progressively refined representations. the analogy is: each 8-layer block in todorov is a "cortical area" that processes its input through a stereotyped circuit and passes the result to the next "area."

## dissenting argument

the comparison above may be too dismissive. consider:

1. the residual stream in todorov is functionally similar to the thalamo-cortical loop. both provide a shared information channel that all processing elements access. the thalamus relays information between cortical areas through layer-specific channels; the residual stream relays information between layers through a shared vector. the difference is in specificity, not in principle.

2. the lack of layer-specific routing in todorov may be a STRENGTH, not a weakness. cortical wiring asymmetries constrain what each layer can compute. the residual stream gives every layer access to all information, which may be more efficient for the statistical structure of language (which does not have the spatial/temporal structure that cortical laminar organization evolved to exploit).

3. the 3:1 ratio convergence across independent ML labs may reflect a genuine computational constraint on how many "memory layers" (KDA/attention) vs "dynamics layers" (SSM/recurrence) are needed for sequence processing. if cortex also faces this constraint, the convergence could be non-coincidental even though the mechanisms differ.

4. the cortical column processes spatial/temporal sensory input with strong local structure (edges, frequencies, somatotopic maps). language has different structure (long-range dependencies, hierarchical syntax, semantic compositionality). optimizing the same basic circuit for different input statistics should produce different parameter settings. the analogy should be at the level of computational principles, not specific ratios.

against the dissent: these arguments rescue the analogy only by making it so abstract that it loses predictive power. if the mapping is "both systems combine different computational primitives in some ratio," that describes every hybrid architecture ever built. the analogy is useful only if it generates testable predictions for todorov's design. what specific modification to todorov's architecture does the cortical analogy predict would improve performance? if none, the analogy is decorative.

## what cortical organization suggests for todorov

if we take the cortical analogy seriously despite the objections, it suggests three modifications worth investigating:

1. **layer-specific routing.** instead of a uniform residual stream, route different information to different layer types. for example, Mamba3 layers could receive a different (filtered) view of the residual stream than KDA layers. this would approximate the layer-specific connectivity of cortex.

2. **feedback connections.** add skip connections from late layers to early layers within each block, approximating cortical feedback from L5/6 to L1/L4. this would enable within-block recurrence without full autoregressive recurrence.

3. **inhibitory layers.** add lightweight "inhibitory" layers (perhaps simple gating or normalization modules) that control the gain of specific layer types based on overall activity. this would approximate the E/I balance mechanism.

all three are testable hypotheses. none are predicted by the current ML literature. none have been validated. they should be treated as research directions, not recommendations.

## key references

- Douglas, R. J. and Martin, K. A. C. (2004). neuronal circuits of the neocortex. annual review of neuroscience, 27, 419-451.
- Harris, K. D. and Shepherd, G. M. G. (2015). the neocortical circuit: themes and variations. nature neuroscience, 18(2), 170-181.
- Bastos, A. M. et al. (2012). canonical microcircuits for predictive coding. neuron, 76(4), 695-711.
- Potjans, T. C. and Diesmann, M. (2014). the cell-type specific cortical microcircuit. cerebral cortex, 24(3), 785-806.

## see also

- [[cortical_column]]
- [[laminar_processing]]
- [[canonical_microcircuit]]
- [[excitatory_inhibitory_balance]]
- [[cortical_microcircuit_to_layer_schedule]]
