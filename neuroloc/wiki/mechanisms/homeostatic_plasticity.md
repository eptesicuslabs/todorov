# homeostatic plasticity

**why this matters**: homeostatic plasticity is the biological solution to the stability problem that every deep network faces -- maintaining activations in a useful dynamic range during training. understanding synaptic scaling provides biological grounding for layer normalization, weight decay, and the spike firing rate targets used in todorov's ATMN health monitoring.

## summary

homeostatic plasticity is a set of negative feedback mechanisms that stabilize neural activity around a **target firing rate** (a neuron's set point for average activity). the most studied form is **synaptic scaling**, discovered by Gina Turrigiano and colleagues (1998): when a neuron's firing rate deviates from its set point, all of its excitatory synapses are multiplicatively scaled up or down to restore the target rate. this operates on a timescale of hours to days, far slower than Hebbian plasticity ([[hebbian_learning]], [[stdp]]). it serves as the "thermostat" that prevents runaway excitation or pathological silencing from unchecked Hebbian learning. ML analog: synaptic scaling is equivalent to a per-neuron multiplicative weight rescaling that preserves relative weight magnitudes, functionally similar to weight normalization or the rescaling step in RMSNorm.

## mechanism

### synaptic scaling

synaptic scaling adjusts the strength of all excitatory synapses on a neuron by a common multiplicative factor:

    w_i -> s * w_i    for all synapses i

where s > 1 when the neuron is less active than its target rate, and s < 1 when it is more active. the multiplicative nature of scaling is functionally critical: it preserves the relative differences between synaptic weights. if synapse A is twice as strong as synapse B before scaling, it remains twice as strong after scaling. this means that information stored by Hebbian plasticity (the relative weight structure) is not disrupted by homeostatic correction (the absolute scale).

the scaling factor s evolves according to:

    ds/dt = (1/tau_h) * (r_target - r_actual)

where:
- tau_h: homeostatic time constant (hours to days, typically 12-48 hours)
- r_target: the target firing rate (set point)
- r_actual: the neuron's current average firing rate
- the sign ensures negative feedback: too little firing increases s, too much decreases it

### molecular mechanisms

the calcium hypothesis: neurons detect their own firing rate through intracellular calcium concentration, which is proportional to average activity. the key molecular players:

- **calcium sensors**: CaMKIV and CaMKK pathways detect time-averaged calcium levels
- **BDNF**: brain-derived neurotrophic factor acts as an activity-dependent scaling signal. elevated BDNF (from high activity) promotes scaling down; reduced BDNF promotes scaling up
- **TNF-alpha**: glial-derived tumor necrosis factor alpha is critical for maintaining synapses in a plastic state and mediating scaling up during activity blockade
- **Arc/Arg3.1**: the immediate-early gene Arc promotes AMPA receptor endocytosis (removal from the synapse surface), contributing to scaling down. reduced Arc expression during inactivity allows AMPA receptor accumulation, producing scaling up
- **receptor trafficking**: the final common pathway is regulation of AMPA receptor density at the postsynaptic membrane. scaling up inserts more GluA1/GluA2 receptors; scaling down removes them

### timescale

synaptic scaling is gradual and cumulative, with measurable changes after 4-6 hours of sustained activity change and full expression after 24-48 hours. this timescale is consistent with the requirement for transcription and translation of new proteins (Arc, BDNF, TNF-alpha receptors). the slow timescale ensures that homeostatic correction does not interfere with the fast, input-specific changes driven by Hebbian learning.

### the target firing rate

the target firing rate (set point) is not a single fixed value but an emergent property of multiple opposing signaling pathways. Turrigiano (2008) proposed that the set point "represents the stable point of all the opposing signaling pathways that operate on synaptic strength." different neuron types have different set points (e.g., fast-spiking interneurons fire at much higher rates than pyramidal cells), and the set point itself may be modifiable under some conditions, though this remains controversial.

## intrinsic plasticity

alongside synaptic scaling, neurons also regulate their intrinsic excitability -- the relationship between input current and firing rate. this operates through modification of voltage-gated ion channels:

- upregulation of sodium channels or downregulation of potassium channels increases excitability (the neuron fires more for the same synaptic input)
- the reverse decreases excitability

intrinsic plasticity and synaptic scaling are complementary homeostatic mechanisms. intrinsic plasticity adjusts the input-output gain of the neuron; synaptic scaling adjusts the effective input strength. both serve to stabilize firing rates, but they act through different molecular pathways and may operate on partially different timescales.

## global vs. local mechanisms

### global scaling

global synaptic scaling affects all excitatory synapses on a neuron uniformly. it is triggered by cell-wide changes in firing rate, detected through somatic calcium concentration. this is the classical form described by Turrigiano et al. (1998).

### local homeostatic mechanisms

there is growing evidence for local (dendritic-branch-level) homeostatic mechanisms:

- local synaptic scaling can be induced by focal application of activity blockers to individual dendritic branches
- presynaptic release probability can be homeostatically adjusted at the level of individual synapses or small groups of synapses
- miniature excitatory postsynaptic currents (mEPSCs) can be regulated locally

the existence of both global and local mechanisms suggests a hierarchical control system: global scaling provides a coarse adjustment based on overall activity, while local mechanisms fine-tune individual branches.

## contrast with Hebbian plasticity

| property | hebbian/STDP | homeostatic scaling |
|---|---|---|
| timescale | seconds to minutes | hours to days |
| specificity | input-specific (homosynaptic) | global (heterosynaptic) |
| direction | strengthens correlated inputs | restores target rate |
| effect on relative weights | creates differences | preserves differences |
| stability | unstable (positive feedback) | stabilizing (negative feedback) |
| molecular basis | NMDA, CaMKII | BDNF, TNF-alpha, Arc |

the two systems are complementary: Hebbian plasticity writes information (encodes correlations), and homeostatic plasticity maintains the dynamic range within which that information remains readable.

## relationship to todorov

the KDA delta rule has no explicit homeostatic mechanism. the forgetting rate alpha (sigmoid of a learned parameter initialized near 0.12) provides exponential decay of old associations, but this is a fixed (per-channel, per-head) rate rather than an activity-dependent feedback mechanism. there is no mechanism that monitors the "firing rate" of the state S_t and adjusts alpha or beta_t accordingly.

however, two implicit stabilization mechanisms exist:
1. the sigmoid activation on alpha clamps the forgetting rate to (0, 1), preventing both unbounded growth (alpha > 1 would amplify old state) and instant erasure (alpha = 0 would discard all history)
2. the sigmoid activation on beta_t clamps the write gate to (0, 1), preventing individual associations from being written with unbounded strength

these are architectural constraints, not learned homeostatic responses. a BCM-like or scaling-like mechanism that dynamically adjusts alpha or beta_t based on state magnitude could provide more robust stabilization. see [[plasticity_to_kda_delta_rule]] for discussion of whether this would improve KDA performance.

## challenges

1. **scaling may erase Hebbian information.** the multiplicative nature of scaling preserves relative weight structure, but this is not guaranteed to preserve all information in networks with complex recurrent dynamics. a network that has learned fine-grained temporal associations through [[stdp]] could lose critical weight ratios during a prolonged scaling event.

2. **global vs local coordination is unclear.** how does a neuron coordinate cell-wide scaling with branch-specific adjustments? the molecular signaling pathways that implement this hierarchical control remain poorly understood. calcium sensing at the soma provides a global signal, but local dendritic mechanisms may operate semi-independently.

3. **relationship to BCM is debated.** synaptic scaling and the [[bcm_theory]] sliding threshold operate on similar timescales and serve similar stabilizing functions. whether they are redundant, complementary, or specialized for different circuit contexts is unresolved. both adjust the LTP/LTD balance, but through different mechanisms (multiplicative rescaling vs threshold shifting).

## key references

- Turrigiano, G. G., Leslie, K. R., Desai, N. S., Rutherford, L. C. & Nelson, S. B. (1998). Activity-dependent scaling of quantal amplitude in neocortical neurons. Nature, 391(6670), 892-896.
- Turrigiano, G. G. (2008). The self-tuning neuron: synaptic scaling of excitatory synapses. Cell, 135(3), 422-435.
- Turrigiano, G. G. (2012). Homeostatic synaptic plasticity: local and global mechanisms for stabilizing neuronal function. Cold Spring Harbor Perspectives in Biology, 4(1), a005736.
- Abbott, L. F. & Nelson, S. B. (2000). Synaptic plasticity: taming the beast. Nature Neuroscience, 3, 1178-1183.
- Desai, N. S., Rutherford, L. C. & Bhatt, D. H. (1999). BDNF regulates the intrinsic excitability of cortical neurons. Learning & Memory, 6(3), 284-291.
- Stellwagen, D. & Malenka, R. C. (2006). Synaptic scaling mediated by glial TNF-alpha. Nature, 440(7087), 1054-1059.
