# attention simulations

## biased_competition.py

biased competition between two neural populations demonstrating attentional selection.

network: 200 LIF neurons split into two populations of 100 (pool A and pool B). both pools receive equal stimulus drive (3.5 nA). a top-down bias current is applied to pool A at t=200 ms. noise added to all neurons (0.5 nA sigma).

experiments:
1. temporal dynamics: population firing rates over time at three bias levels (0, 1.0, 2.0 nA). shows bias onset driving rate divergence between pools. spike raster plots show individual neuron activity pre- and post-bias.
2. bias strength sweep: rate difference (A - B) and selectivity index ((A-B)/(A+B)) as functions of bias strength from 0 to 4.0 nA across 20 levels. demonstrates monotonic increase in selectivity with bias.
3. stimulus ratio sweep: competition outcome (winner and loser firing rates) as a function of relative stimulus strength (A/B ratio from 0.5 to 2.0) with fixed moderate bias (1.5 nA). demonstrates that top-down bias can override bottom-up stimulus strength at low ratios but is dominated at high ratios.

dependencies: numpy, matplotlib, Brian2

output: biased_competition.png (6-panel figure: 3 rate traces + 3 rasters), biased_competition_analysis.png (3-panel figure: rate difference, selectivity, competition outcome)

run: `python biased_competition.py`

relevance: demonstrates the Desimone & Duncan (1995) biased competition mechanism. top-down bias (analogous to prefrontal feedback) resolves competition between equally-driven populations by enhancing one and suppressing the other. todorov has no equivalent competitive attention mechanism: MLA uses softmax (passive weighting, no suppression), and KDA's beta gate operates on individual tokens without inter-token competition. the simulation illustrates the computational capability that is absent from the todorov architecture: active suppression of competing representations through competitive dynamics.
