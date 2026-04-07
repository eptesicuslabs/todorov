# predictive coding simulations

## predictive_coding_2level.py

two-level predictive coding hierarchy implemented in Brian2 spiking neurons.

### architecture

- level 1 (sensory): 50 LIF neurons receiving external input patterns
- error units: 50 LIF neurons computing prediction error (bottom-up input minus top-down prediction)
- level 2 (representation): 20 LIF neurons encoding the internal model

connections:
- sensory -> error (feedforward, one-to-one, fixed)
- representation -> error (feedback, all-to-all, learned -- carries top-down predictions)
- error -> representation (feedforward, all-to-all, learned -- carries prediction errors upward)

### input

4 random binary patterns (25% active neurons each), presented in repeating sequence. each pattern is shown for 200 ms. 40 total presentations.

### learning

Hebbian updates applied every 5 presentations. the feedback weights (representation -> error) learn to carry top-down predictions. the feedforward weights (error -> representation) learn to route prediction errors upward. weight updates are the outer product of pre and post firing rates, clipped to [-2, 2].

### expected behavior

early presentations: error units fire strongly (predictions are poor, large error signal). late presentations: if the feedback weights learn the input statistics, error unit firing decreases (successful prediction suppresses the error signal). representation unit firing may increase or stabilize as the internal model develops.

this is a simplified demonstration. the full Rao & Ballard model uses rate-based neurons with continuous dynamics and gradient-based weight updates. the Brian2 implementation uses spiking neurons with Hebbian plasticity, which is biologically more realistic but less computationally precise.

### output

predictive_coding_demo.png: 4-panel figure showing sensory spikes, error unit spikes, representation unit spikes, and population firing rates across presentations. pattern identities are color-coded in the sensory panel.

### dependencies

- brian2
- numpy
- matplotlib

### usage

```
python predictive_coding_2level.py
```
