# single neuron simulations

## what these demonstrate

three simulations exploring the behavior of single neuron models, from the simplest (LIF) to more complex (AdEx, Izhikevich). each produces a figure showing how the model responds to constant current injection.

the purpose is to build intuition for what these neuron models do and how their parameters control behavior. the results inform the design of ATMN spike neurons in todorov (see [[neuron_models_to_atmn]]).

## scripts

### lif_fi_curve.py
- model: leaky integrate-and-fire ([[leaky_integrate_and_fire]])
- output: fi_curve.png
- shows: frequency-current (f-I) curve -- firing rate as a function of input current. also shows membrane potential traces at different current levels.
- key parameters: tau_m=20 ms, V_th=-50 mV, V_reset=-65 mV, t_ref=2 ms
- requires: brian2

### adex_patterns.py
- model: adaptive exponential integrate-and-fire ([[adaptive_exponential]])
- output: adex_patterns.png
- shows: four firing patterns (regular spiking, adaptation, initial bursting, regular bursting) produced by varying parameters a, b, tau_w, V_r.
- key parameters: from Naud et al. 2008 classification
- requires: brian2

### izhikevich_gallery.py
- model: Izhikevich 2003 ([[izhikevich_model]])
- output: izhikevich_gallery.png
- shows: six firing patterns (RS, IB, CH, FS, LTS, RZ) from the same model with different parameter values a, b, c, d.
- key parameters: from Izhikevich 2003 original paper
- requires: numpy, matplotlib (no brian2 needed -- the model is simple enough to simulate directly)

## how to run

    pip install brian2 matplotlib numpy
    cd neuroloc/simulations/single_neuron
    python lif_fi_curve.py
    python adex_patterns.py
    python izhikevich_gallery.py

## parameters to vary

for the LIF f-I curve:
- tau_m: larger values produce shallower f-I curves (slower integration)
- t_ref: limits the maximum firing rate to 1/t_ref
- V_th - V_reset: controls the "gain" (slope) of the f-I curve

for AdEx patterns:
- b: spike-triggered adaptation. b=0 gives tonic spiking; b>0 gives adaptation
- tau_w: adaptation timescale. short tau_w = fast recovery; long tau_w = sustained adaptation
- V_r: reset voltage. shallow reset (V_r near V_T) enables bursting
- a: subthreshold adaptation. a<0 creates delayed spiking and resonance

for the Izhikevich gallery:
- c: post-spike reset voltage. the key control for regular spiking (-65) vs bursting (-50)
- d: recovery increment. larger d = stronger adaptation
- a: recovery speed. a=0.1 for fast spiking, a=0.02 for regular/adapting
- b: coupling strength. b=0.25-0.26 for resonant/LTS behavior
