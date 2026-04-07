# neuromodulation simulations

## dopamine_rpe.py

a simple TD learning agent demonstrating dopamine-modulated synaptic plasticity.

### what it does

- 20 sensory LIF neurons project to 10 action LIF neurons through plastic synapses
- 1 dopamine neuron computes the reward prediction error (RPE): delta = reward - value_estimate
- the value estimate is updated by TD(0): V <- V + eta * delta
- synaptic weights are modulated by the three-factor rule: Delta_w = eta * delta * eligibility_trace
- eligibility traces mark recently active synapses (exponential decay, tau = 200 ms)

### protocol

- trials 1-20: reward delivered (learning phase). RPE starts positive and decays toward zero as the model learns to predict reward.
- trials 21-40: reward continues (asymptotic phase). RPE near zero (reward fully predicted). weights stable.
- trials 41-60: reward omitted (omission phase). RPE goes negative (expected reward missing). weights decrease as the model unlearns the reward association.

### output

four-panel figure (dopamine_rpe_demo.png):
1. RPE signal across trials (should decay to zero during learning, go negative during omission)
2. phasic dopamine response (tracks RPE)
3. mean synaptic weight evolution with standard deviation band
4. weight distribution comparison: initial vs peak (trial 40) vs post-omission (trial 60)

### expected results

the simulation demonstrates:
- RPE decreases to near-zero as reward becomes predicted (Schultz 1997 "no response to fully predicted reward")
- RPE goes negative when expected reward is omitted (Schultz 1997 "depression for expected-but-missing reward")
- weights increase during reward phase (dopamine-gated LTP)
- weights decrease during omission phase (dopamine-gated LTD)
- eligibility traces provide temporal credit assignment: only recently active synapses are modified

### dependencies

- brian2
- numpy
- matplotlib

### running

```
cd neuroloc/simulations/neuromodulation
python dopamine_rpe.py
```
