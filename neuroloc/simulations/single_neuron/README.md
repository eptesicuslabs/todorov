# single neuron simulations

## what these demonstrate

these scripts cover two different goals.

- `lif_fi_curve.py` is now a bridge-validation pilot for [[neuron_models_to_atmn]], not a plot-only classroom demo. it compares matched discrete-time explicit leak, current atmn-style carry, and an integrator control on paired-pulse retention and long-sequence drift, with a passive brian2 lif trace kept as a qualitative anchor.
- `adex_patterns.py` and `izhikevich_gallery.py` remain intuition-building references for richer single-neuron firing regimes.

the current single-neuron lesson is narrower than "leak is better." the validated pilot shows a tradeoff: explicit leak preserves more subthreshold evidence across a gap, while atmn-style carry stays closer to the passive lif anchor on second-pulse peak and drifts less under long zero-mean noise.

## scripts

### lif_fi_curve.py
- bridge question: does adding explicit leak improve the biological plausibility of atmn-style state updates without creating worse drift behavior?
- outputs: `lif_leak_validation.png`, `lif_leak_validation_metrics.json`
- shows: paired-pulse retention across gaps 1, 4, 8, 16, 32, 64 ms; long-sequence drift across lengths 128, 256, 512, 1024; tau sweep over explicit leak constants 5, 10, 20, 30 ms
- key validated result: at 16 ms gap, `explicit_leak_tau20` retained 0.447 of the first pulse vs `atmn_carry_tau2` at -0.001, but `atmn_carry_tau2` had lower mean second-pulse peak error to the passive lif anchor (0.166 mV vs 0.507 mV) and lower length-1024 state standard deviation (0.576 mV vs 1.567 mV)
- requires: brian2, numpy, scipy, matplotlib

### adex_patterns.py
- model: adaptive exponential integrate-and-fire ([[adaptive_exponential]])
- output: `adex_patterns.png`
- shows: four firing patterns (regular spiking, adaptation, initial bursting, regular bursting) produced by varying parameters `a`, `b`, `tau_w`, `v_r`
- key parameters: from naud et al. 2008 classification
- requires: brian2

### izhikevich_gallery.py
- model: izhikevich 2003 ([[izhikevich_model]])
- output: `izhikevich_gallery.png`
- shows: six firing patterns (rs, ib, ch, fs, lts, rz) from the same model with different parameter values `a`, `b`, `c`, `d`
- key parameters: from izhikevich 2003 original paper
- requires: numpy, matplotlib

## how to run

    pip install brian2 matplotlib numpy scipy
    cd neuroloc/simulations/single_neuron
    python lif_fi_curve.py
    python adex_patterns.py
    python izhikevich_gallery.py

## parameters to vary

for `lif_fi_curve.py`:
- `leak_tau_ms`: larger leak time constants preserve more cross-gap state but also raise long-horizon variance relative to the carry condition in this pilot
- `atmn_tau`: sets the carry retention factor (`1 / atmn_tau` here); lower retention reduces both cross-gap memory and drift
- `pulse_amplitude_mv` and `gap_noise_sigma_mv`: define the subthreshold paired-pulse regime; if they are too large the comparison collapses into trivial spiking
- `drift_noise_sigma_mv`: controls how aggressively the long-sequence drift task stresses each recurrence
- `threshold_mv`: should stay high enough that the paired-pulse task remains mostly subthreshold, otherwise the result becomes a reset-policy comparison instead of a retention comparison

for `adex_patterns.py`:
- `b`: spike-triggered adaptation. `b=0` gives tonic spiking; `b>0` gives adaptation
- `tau_w`: adaptation timescale. short `tau_w` = fast recovery; long `tau_w` = sustained adaptation
- `v_r`: reset voltage. shallow reset (`v_r` near `v_t`) enables bursting
- `a`: subthreshold adaptation. `a<0` creates delayed spiking and resonance

for `izhikevich_gallery.py`:
- `c`: post-spike reset voltage. the key control for regular spiking (`-65`) vs bursting (`-50`)
- `d`: recovery increment. larger `d` = stronger adaptation
- `a`: recovery speed. `a=0.1` for fast spiking, `a=0.02` for regular or adapting
- `b`: coupling strength. `b=0.25-0.26` for resonant and lts behavior
