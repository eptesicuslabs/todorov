# hodgkin-huxley model

**why this matters**: the HH model is the ground-truth biophysical description of how neurons generate spikes. every simplified neuron model used in ML (LIF, AdEx, Izhikevich, ATMN) is an approximation of HH dynamics, and understanding which HH features are preserved or discarded in each simplification reveals the biological constraints that matter for neural architecture design.

## status
[DRAFT]
last updated: 2026-04-06
sources: 6 papers, 2 textbooks

## biological description

the hodgkin-huxley (HH) model is the first quantitative description of the electrical activity of a neuron. it was developed in 1952 from **voltage-clamp** (a technique that holds membrane voltage at a fixed value to measure ionic currents in isolation) experiments on the squid giant axon (Loligo). the model describes how **action potentials** (brief electrical spikes that neurons use to communicate) arise from the opening and closing of **voltage-gated ion channels** (protein pores in the membrane that open or close in response to voltage changes) in the cell membrane.

the model treats the neuron membrane as an electrical circuit: a capacitor (the lipid bilayer) in parallel with voltage-dependent conductances (ion channels) and batteries (ionic concentration gradients). three types of current flow through the membrane:

1. sodium current (I_Na): carried by Na+ ions flowing inward through voltage-gated channels. these channels activate rapidly (opening within ~0.1 ms) and then inactivate (close despite maintained depolarization). the rapid activation followed by inactivation produces the rising phase and peak of the action potential.

2. potassium current (I_K): carried by K+ ions flowing outward through voltage-gated channels. these channels activate more slowly than Na+ channels (~1 ms) and do not inactivate on the timescale of an action potential. the delayed activation produces the falling phase and undershoot.

3. leak current (I_L): a small, voltage-independent current carried primarily by Cl- ions. it represents all other membrane conductances not captured by the Na+ and K+ channels.

the action potential emerges from a positive feedback loop: depolarization opens Na+ channels, which further depolarizes the membrane, which opens more Na+ channels. this explosive process terminates because Na+ channels inactivate and K+ channels open, repolarizing the membrane.

for an ML researcher: this is the "ground truth" biophysical model. every simpler neuron model (LIF, AdEx, Izhikevich) is an approximation of HH dynamics. the model has four state variables (V, m, h, n) and ~20 parameters. it is too expensive for large-scale network simulations but defines what biological accuracy means.

## mathematical formulation

the membrane equation:

    C_m dV/dt = -g_Na * m^3 * h * (V - E_Na) - g_K * n^4 * (V - E_K) - g_L * (V - E_L) + I_ext

where:
- C_m = 1.0 uF/cm^2 (membrane capacitance per unit area)
- V = membrane potential (mV)
- I_ext = externally injected current (uA/cm^2)
- g_Na = 120 mS/cm^2 (maximal sodium conductance)
- g_K = 36 mS/cm^2 (maximal potassium conductance)
- g_L = 0.3 mS/cm^2 (leak conductance)
- E_Na = +50 mV (sodium reversal potential)
- E_K = -77 mV (potassium reversal potential)
- E_L = -54.4 mV (leak reversal potential)

the gating variables m, h, n each obey first-order kinetics:

    dx/dt = alpha_x(V) * (1 - x) - beta_x(V) * x

where x is any of {m, h, n}. equivalently:

    dx/dt = (x_inf(V) - x) / tau_x(V)

with x_inf(V) = alpha_x / (alpha_x + beta_x) and tau_x(V) = 1 / (alpha_x + beta_x).

the original 1952 rate functions (V measured as depolarization from rest, in mV):

sodium activation (m):

    alpha_m(V) = 0.1 * (25 - V) / (exp((25 - V) / 10) - 1)
    beta_m(V) = 4.0 * exp(-V / 18)

sodium inactivation (h):

    alpha_h(V) = 0.07 * exp(-V / 20)
    beta_h(V) = 1.0 / (exp((30 - V) / 10) + 1)

potassium activation (n):

    alpha_n(V) = 0.01 * (10 - V) / (exp((10 - V) / 10) - 1)
    beta_n(V) = 0.125 * exp(-V / 80)

note: the original 1952 convention measures V as depolarization from rest (V_HH = V_modern - V_rest). modern implementations typically shift to absolute membrane potential, which changes the constants in the rate functions but not the dynamics.

physical interpretation of gating variables:
- m: probability that a Na+ activation gate is open. fast (tau_m ~ 0.1 ms). the m^3 term reflects three independent activation gates per channel.
- h: probability that a Na+ inactivation gate is open (h=1 means not inactivated). slow (tau_h ~ 1 ms). the m^3*h product means the channel conducts only when all three activation gates are open AND the inactivation gate has not closed.
- n: probability that a K+ activation gate is open. intermediate speed (tau_n ~ 1 ms). the n^4 term reflects four independent activation gates per channel.

total system: 4 coupled ODEs (V, m, h, n), 4 state variables, ~20 parameters.

## evidence strength

STRONG. the hodgkin-huxley model is among the most validated models in all of biology. hodgkin and huxley received the 1963 Nobel Prize in Physiology or Medicine for this work. the model quantitatively predicts:

- action potential shape, amplitude, and duration
- conduction velocity along the axon
- refractory period (absolute and relative)
- threshold behavior
- repetitive firing under sustained current
- anode break excitation
- subthreshold oscillations

subsequent single-channel recordings (Neher and Sakmann, Nobel 1991) confirmed the channel-gating interpretation at the molecular level. the framework has been extended to hundreds of channel types across species.

## challenges and counter-arguments

1. **channel independence assumption is wrong.** the model assumes Na+ and K+ channels gate independently. recent evidence (Naundorf et al. 2006) shows that Na+ and K+ gating variables are correlated, particularly during high-frequency firing. the interdependence of sodium and potassium gating variables means the m, h, n factorization is an approximation, not a physical truth.

2. **the model is fit to one preparation.** all original parameters come from the squid giant axon at 6.3 C. mammalian cortical neurons have different channel densities, different temperature dependencies, and additional channel types (Ca2+, HCN, A-type K+, etc.) not present in the original model. applying the 1952 parameters to cortical neurons is physically meaningless; each neuron type requires its own parameter fit.

3. **computational cost is prohibitive for networks.** the system requires solving 4 coupled stiff ODEs per neuron per timestep, with rate functions involving exponentials. at dt = 0.01 ms (necessary for numerical stability), a single neuron requires ~1200 FLOPS per ms of simulated time. for a 300M-parameter language model processing 2048 tokens, this is 6 orders of magnitude more expensive than a simple threshold operation.

4. **the model omits known important mechanisms.** calcium dynamics, synaptic plasticity, dendritic computation, gap junctions, neuromodulation, stochastic channel gating, and intracellular signaling cascades are all absent. the model is a model of the axon hillock, not a model of a neuron.

5. **the gating particle interpretation is phenomenological.** the m^3*h and n^4 exponents were chosen to fit voltage-clamp data, not from structural knowledge. modern structural biology shows that ion channels are tetrameric proteins with cooperative gating, not collections of independent particles. the mathematical form works but the physical interpretation is misleading.

## simulation

see [[lif_fi_curve]] for a simpler neuron simulation. a full HH simulation is not included in the current Brian2 scripts because the HH model requires specialized numerical methods (implicit or exponential euler) for stability.

key parameters to explore:
- I_ext: below rheobase (~6.2 uA/cm^2) the neuron is silent; above it, repetitive firing emerges
- temperature: the original model uses Q10 factors of ~3; increasing temperature accelerates all kinetics
- g_Na/g_K ratio: controls action potential shape and threshold

## computational implications

the HH model establishes several principles relevant to artificial architectures:

1. **threshold is emergent, not a parameter.** in HH, spiking arises from the interaction of Na+ activation and inactivation -- there is no explicit "threshold" parameter. the apparent threshold depends on the rate of depolarization and recent history. this contrasts with all simplified models (LIF, AdEx, Izhikevich, ATMN) where threshold is an explicit parameter.

2. **the gating variables are multiplicative interactions.** the product m^3*h is a bilinear-like interaction between activation and inactivation. ML analog: this maps conceptually to todorov's B (bilinear) family in the CRBR framework, and more generally to multiplicative gating in LSTMs and GRUs where two signals must be simultaneously active for information to flow. channel conductance requires the conjunction of multiple gating states.

3. **temporal dynamics span multiple timescales.** m operates at ~0.1 ms, h at ~1 ms, n at ~1 ms, and the membrane at ~1 ms. this multi-timescale structure is what produces complex firing patterns. ML analog: this is equivalent to having multiple exponential moving averages with different decay rates operating in parallel, similar to how the KDA delta rule uses per-channel alpha values to create a bank of temporal filters. simplified models collapse these into fewer timescales.

4. **the leak is essential for stability.** without the leak conductance g_L, the resting potential is unstable. every biologically realistic neuron model includes a leak term. this is relevant to ATMN, which omits it (see [[neuron_models_to_atmn]]).

## bridge to todorov

the HH model is the theoretical ancestor of all neuron models used in todorov, but it is not directly implemented. the relationship is:

- HH -> LIF (collapse gating dynamics, keep leak + threshold) -> ATMN (remove leak, add learnable threshold)
- HH's m^3*h product maps to the B (bilinear) family: it is a multiplicative gate on ion flow
- HH's voltage-dependent rate functions map to the R (rotational/dynamic) family: they are nonlinear functions of state
- HH's spike quantization (all-or-none action potential) maps to the Q (quantization) family: the output is effectively ternary (depolarized/resting/hyperpolarized)

what exists: ATMN captures the threshold-and-reset essence of HH at minimal cost.
what is missing: leak dynamics, refractory period, multi-timescale gating, history-dependent threshold.
what matters for 300M scale: the leak term and refractory period are the most computationally tractable additions. gating variable dynamics are too expensive.

## related mechanisms

- [[leaky_integrate_and_fire]] -- the standard simplification of HH
- [[adaptive_exponential]] -- adds exponential spike initiation approximating HH's Na+ activation
- [[izhikevich_model]] -- polynomial approximation reproducing HH-like dynamics
- [[neuron_model_comparison]] -- side-by-side comparison

## open questions

1. can the multi-timescale gating structure of HH be captured by a low-dimensional approximation that is computationally feasible at 300M scale? the Izhikevich model achieves this with 2 variables; can we do it with 1?

2. the Na+ channel inactivation gate (h) is responsible for spike-frequency adaptation and refractory periods. ATMN has neither. is this a critical omission for sequence modeling, where refractory-like behavior could prevent runaway activation?

3. the HH model predicts that threshold varies with the rate of depolarization (accommodation). ATMN's threshold is learnable but does not depend on input dynamics. would a rate-dependent threshold improve spike coding?

4. cooperative channel gating (beyond the independent-gate approximation) creates history-dependent behavior at the molecular level. does this matter at the population level, or is it averaged out?

## source bibliography

- Hodgkin, A.L. and Huxley, A.F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. Journal of Physiology, 117(4), 500-544.
- Hodgkin, A.L. and Huxley, A.F. (1952). Currents carried by sodium and potassium ions through the membrane of the giant axon of Loligo. Journal of Physiology, 116(4), 449-472.
- Hodgkin, A.L. and Huxley, A.F. (1952). The components of membrane conductance in the giant axon of Loligo. Journal of Physiology, 116(4), 473-496.
- Hodgkin, A.L. and Huxley, A.F. (1952). The dual effect of membrane potential on sodium conductance in the giant axon of Loligo. Journal of Physiology, 116(4), 497-506.
- Gerstner, W., Kistler, W.M., Naud, R., and Paninski, L. (2014). Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition. Cambridge University Press. Chapter 2.
- Naundorf, B., Wolf, F., and Volgushev, M. (2006). Unique features of action potential initiation in cortical neurons. Nature, 440(7087), 1060-1063.
- Bean, B.P. (2007). The action potential in mammalian central neurons. Nature Reviews Neuroscience, 8(6), 451-465.
