# development simulations

## what these demonstrate

a spiking network where STDP plasticity is modulated over time, mimicking the maturation of PV+ inhibitory interneurons that controls [[critical_periods|critical period]] timing. the simulation shows that training during the high-plasticity phase (early development) shapes receptive fields permanently, while the same stimuli presented during the low-plasticity phase (mature circuit) produce much less weight change.

this demonstrates the core finding of Hensch (2005): the critical period is a time window of maximal plasticity, and its closure is a reduction in the capacity for experience-dependent circuit refinement. see [[critical_periods]], [[synaptic_pruning]], and [[developmental_self_organization]].

## scripts

### critical_period.py
- model: 20 input neurons (SpikeGeneratorGroup) projecting via STDP synapses to 5 LIF output neurons. input consists of two alternating patterns (A: neurons 0-9 active, B: neurons 10-19 active). plasticity (STDP learning rate) is modulated over 4 phases: 100% -> 70% -> 20% -> 5%, mimicking the developmental trajectory from immature (high plasticity) to mature (low plasticity, critical period closed).
- output: critical_period.png (3 panels)
- shows:
  1. top panel: weight evolution across the 4 developmental phases. pattern A and pattern B weights diverge during the high-plasticity phase and stabilize as plasticity decreases. the weight structure established during the critical period persists even as plasticity drops.
  2. middle panel: plasticity modulation schedule over time. the four phases correspond to immature circuit (high plasticity, no PV+ maturation), moderate plasticity, maturing circuit (PV+ cells consolidating, PNNs forming), and mature circuit (critical period closed).
  3. bottom panel: critical period effect comparison. selectivity (|pattern A weights - pattern B weights|) is measured for two conditions: (a) training during the high-plasticity phase, (b) training with the same stimuli but at the low-plasticity level (mimicking adult exposure). early training produces much greater selectivity, demonstrating the critical period effect.
- key parameters:
  - n_input = 20, n_output = 5
  - tau_m = 20 ms, tau_stdp = 20 ms
  - plasticity_schedule = [1.0, 0.7, 0.2, 0.05]
  - a_plus_base = 0.015, a_minus_base = 0.018
  - stim_rate = 40 Hz (active pattern), base_rate = 5 Hz (background)
  - duration per phase = 5 seconds
- requires: brian2, matplotlib, numpy

## how to run

    pip install brian2 matplotlib numpy
    cd neuroloc/simulations/development
    python critical_period.py

## parameters to vary

- plasticity_schedule: change the rate of plasticity decline. a sharper decline (e.g., [1.0, 0.1, 0.01, 0.001]) produces a more dramatic critical period effect. a gradual decline (e.g., [1.0, 0.9, 0.8, 0.7]) shows that some developmental refinement occurs even with moderate plasticity reduction.
- duration_per_phase: longer phases give more time for weight changes to accumulate. at very short durations, the critical period effect is less pronounced because even the high-plasticity phase does not have enough time to shape weights.
- stim_rate: higher stimulation rates produce faster weight changes during the high-plasticity phase. at very low rates, even high plasticity produces minimal selectivity.
- n_output: more output neurons create more complex competitive dynamics. with many output neurons, different neurons may become selective to different patterns, demonstrating self-organized specialization.

## biological correspondence

- the plasticity schedule represents the maturation of PV+ interneurons: as PV+ cells mature, they increase inhibition, which (paradoxically) increases the precision of Hebbian competition but reduces the overall magnitude of synaptic change
- the high-plasticity phase corresponds to the open critical period (before PNN formation, before myelination)
- the low-plasticity phase corresponds to the closed critical period (PNNs enwrap PV+ cells, myelin restricts axon sprouting)
- the selectivity comparison (early vs late training) demonstrates the central result: the same experience has qualitatively different effects depending on when it occurs during development
- the weight persistence after plasticity reduction shows that critical period learning is PERMANENT: it shapes circuits that subsequent reduced plasticity cannot easily overwrite
