# short-term plasticity

**why this matters**: short-term plasticity implements temporal filtering at every synapse, turning static connections into dynamic bandpass filters. the Tsodyks-Markram model's depression-facilitation dynamics map directly to the exponential decay and gated write operations in recurrent state machines like the KDA delta rule.

## summary

**short-term synaptic plasticity** (STP, activity-dependent changes in synaptic strength lasting milliseconds to minutes) refers to transient modifications of synaptic efficacy. unlike long-term plasticity ([[hebbian_learning]], [[stdp]]), STP is not thought to underlie memory storage. instead it performs temporal filtering of neural signals: transforming the temporal pattern of presynaptic spikes into a distinct pattern of postsynaptic responses. the two main forms are **short-term facilitation** (STF, where successive spikes produce progressively larger responses) and **short-term depression** (STD, where successive spikes produce progressively smaller responses). the Tsodyks-Markram model (1997) captures both in a compact dynamical system based on **vesicle depletion** (exhaustion of neurotransmitter-filled packets at the synapse) and calcium accumulation.

## mechanism

### short-term facilitation (STF)

facilitation is an increase in synaptic strength following closely spaced presynaptic spikes. the mechanism is primarily presynaptic:

1. an action potential invades the presynaptic terminal, opening voltage-gated calcium channels
2. calcium influx triggers vesicle fusion and neurotransmitter release
3. residual calcium that has not yet been cleared accumulates in the terminal
4. the next action potential adds to this residual calcium, producing a larger calcium transient
5. higher peak calcium increases the probability of vesicle release
6. the postsynaptic response is therefore larger

facilitation has a time constant of tau_f ~ 50-500 ms, reflecting the rate of presynaptic calcium clearance by pumps and buffers. it decays back to baseline within seconds after the end of presynaptic activity.

### short-term depression (STD)

depression is a decrease in synaptic strength during sustained presynaptic activity. the primary mechanism is vesicle depletion:

1. each presynaptic spike releases neurotransmitter from a pool of readily releasable vesicles (RRP)
2. the RRP is finite (typically 5-20 vesicles per release site)
3. rapid firing depletes the RRP faster than it can be replenished
4. fewer available vesicles means lower release probability per spike
5. the postsynaptic response decreases

recovery from depression has a time constant of tau_d ~ 200-800 ms, reflecting the rate of vesicle replenishment through recycling, refilling, and docking.

additional mechanisms contributing to STD:
- presynaptic autoreceptor activation (mGluR, GABA_B) reducing release probability
- postsynaptic receptor desensitization (especially at AMPA receptors during high-frequency transmission)
- glial cell uptake of neurotransmitter becoming saturated

### augmentation and post-tetanic potentiation (PTP)

beyond facilitation and depression, two slower forms of short-term enhancement exist:

- **augmentation** (tau ~ 5-10 s): enhancement that outlasts facilitation, possibly involving calcium-dependent modulation of the release machinery
- **post-tetanic potentiation** (tau ~ 30 s to minutes): longer-lasting enhancement after high-frequency tetanic stimulation, involving residual calcium acting on distinct molecular targets from those mediating facilitation

these multiple timescales of STP create a rich temporal filtering landscape at each synapse.

## the Tsodyks-Markram model

Tsodyks and Markram (1997) proposed a phenomenological model that captures both facilitation and depression with a small number of parameters. the model tracks two state variables:

### state variables

- **x**: fraction of available (recovered) vesicles, 0 <= x <= 1
- **u**: utilization parameter (effective release probability), 0 <= u <= 1

### dynamics between spikes

    dx/dt = (1 - x) / tau_d        (vesicle recovery)
    du/dt = (U - u) / tau_f         (release probability decay toward baseline U)

### update at each presynaptic spike (at time t_sp)

    u -> u + U * (1 - u)           (calcium-driven increase in release probability)
    x -> x - u * x                 (vesicle depletion proportional to release probability)

the postsynaptic current is proportional to:

    I_syn = A * u * x * delta(t - t_sp)

where A is the absolute synaptic efficacy (maximal response).

### parameters

- **U**: baseline utilization (release probability at rest). U ~ 0.5-0.9 for depressing synapses, U ~ 0.05-0.2 for facilitating synapses
- **tau_d**: recovery time constant for vesicle replenishment (200-800 ms)
- **tau_f**: decay time constant for facilitation (50-500 ms)
- **A**: absolute synaptic efficacy

### regimes

the relative magnitudes of tau_d and tau_f determine the dominant behavior:

- **depressing synapses** (tau_d >> tau_f, large U): vesicle depletion dominates. common at neocortical excitatory-to-excitatory connections. these synapses act as low-pass filters, responding strongly to the first spike in a burst but attenuating sustained activity. ML analog: this is equivalent to the alpha decay in KDA, where old state is exponentially forgotten -- the first token in a sequence has the strongest influence on the recurrent state.
- **facilitating synapses** (tau_f >> tau_d, small U): calcium accumulation dominates. common at excitatory-to-inhibitory connections (e.g., pyramidal-to-Martinotti cell synapses). these synapses act as high-pass filters, responding weakly to isolated spikes but strongly to bursts. ML analog: this resembles a warmup mechanism where repeated activation builds up influence, similar to how attention scores accumulate evidence over multiple tokens.
- **mixed synapses** (comparable tau_d, tau_f): both effects contribute, producing complex temporal dynamics including initial facilitation followed by depression

## computational role

### temporal filtering

depressing synapses act as high-pass temporal derivative filters: they respond most strongly to the onset of activity (the first spike after silence) and attenuate sustained firing. this makes them sensitive to changes in input rather than absolute input levels.

facilitating synapses act as low-pass temporal integrators: they respond weakly to single spikes but accumulate over repeated activation. this makes them sensitive to sustained or bursting activity.

### gain control

short-term depression implements a form of automatic gain control. at high presynaptic firing rates, depression reduces effective synaptic strength, compressing the dynamic range of the postsynaptic response. this allows the postsynaptic neuron to remain sensitive across a wide range of input intensities.

### working memory

facilitating synapses have been proposed as a substrate for short-term memory. the elevated synaptic efficacy following a stimulus carries information about recent input history for hundreds of milliseconds, even in the absence of sustained neural activity. this is analogous to the recurrent state in the KDA delta rule, where past associations persist through the exponentially decaying state matrix.

### temporal coding

the frequency-dependent filtering properties of STP enable synapses to act as temporal pattern detectors. a synapse with specific tau_d and tau_f values responds maximally to input at a preferred frequency, effectively implementing a bandpass filter in the time domain.

### redundancy reduction

depressing synapses reduce the transmission of redundant (repeated) information while faithfully transmitting novel (unexpected) signals. this is equivalent to temporal decorrelation and can improve the efficiency of neural coding.

## relationship to todorov

the KDA alpha decay (S_t -> diag(alpha) * S_t) is functionally analogous to short-term depression: the recurrent state exponentially decays in the absence of new input. the channel-wise structure of alpha (different decay rates for different head channels) is analogous to having synapses with different tau_d values, creating a bank of temporal filters with different characteristic timescales.

however, the analogy is incomplete:
- KDA has no facilitation mechanism (there is no state variable analogous to u that increases with repeated input)
- KDA's "depression" (alpha decay) is fixed per channel, not activity-dependent
- the Tsodyks-Markram model's interaction between x and u produces nonlinear temporal dynamics that alpha decay alone cannot replicate

a closer biological analog to the KDA state update would be a depressing synapse with a data-dependent recovery rate (beta_t modulating how much new information is written), but without the facilitating component.

## challenges

1. **the model is phenomenological.** the Tsodyks-Markram model captures macroscopic STP behavior but does not model the underlying molecular machinery (calcium dynamics, vesicle cycling, SNARE complex assembly). more biophysically detailed models exist but require many more parameters and are harder to fit to data.

2. **diversity of STP profiles is unexplained.** how the diversity of STP profiles across different synapse types (depressing, facilitating, mixed) is specified during development remains a key open question. is it determined by pre- and postsynaptic cell type, by activity-dependent tuning, or by both? the answer determines whether STP parameters should be treated as fixed or learnable in artificial architectures.

3. **interaction with long-term plasticity is complex.** repeated activation of a depressing synapse reduces neurotransmitter release, which could impair the calcium signals needed for [[stdp]]. conversely, facilitation could artificially boost calcium and promote unintended LTP. the interplay between short-term and long-term plasticity timescales remains an active area of investigation.

## key references

- Tsodyks, M. V. & Markram, H. (1997). The neural code between neocortical pyramidal neurons depends on neurotransmitter release probability. PNAS, 94(2), 719-723.
- Zucker, R. S. & Regehr, W. G. (2002). Short-term synaptic plasticity. Annual Review of Physiology, 64, 355-405.
- Markram, H., Wang, Y. & Tsodyks, M. (1998). Differential signaling via the same axon of neocortical pyramidal neurons. PNAS, 95(9), 5323-5328.
- Abbott, L. F. & Regehr, W. G. (2004). Synaptic computation. Nature, 431(7010), 796-803.
- Dittman, J. S., Kreitzer, A. C. & Regehr, W. G. (2000). Interplay between facilitation, depression, and residual calcium at three presynaptic terminals. Journal of Neuroscience, 20(4), 1374-1385.
