# memory simulations

## pattern_completion.py

ca3-like attractor baseline for pattern completion with an explicit shuffled-weight control.

network: 200 binary neurons ({-1, +1}), hebbian outer-product weights $w_{ij} = \frac{1}{n}\sum_p \xi_i^p\xi_j^p$, true in-place asynchronous updates, and a matched shuffled-weight control that preserves the weight histogram while destroying the learned structure.

experiments:
1. convergence example at 30% corruption for hebbian weights and the shuffled control
2. corruption sweep from 0% to 50% with trial-level overlap, bit error rate, and exact retrieval probability
3. load sweep at $n=200$ with exact retrieval probability under 10% corruption
4. scaling sweep across $n \in \{100, 200, 300\}$ with load-fraction comparisons around the $0.138n$ reference point
5. visual comparison of target, cue, hebbian retrieval, and shuffled-control retrieval

dependencies: numpy, scipy, matplotlib

output:
- pattern_completion.png
- pattern_completion_metrics.json

run: `python pattern_completion.py`

quantitative verdict: this script is a controlled attractor-memory baseline, not a validation of hippocampal memory as a whole. it measures how much structured recurrent weights improve recovery relative to a matched control and how retrieval degrades with corruption and load. the $0.138n$ line is included as theoretical context, but this run does not claim to prove the asymptotic hopfield limit.

relevance: this simulation isolates the part of biological memory that todorov's current kda path does not implement: nonlinear attractor cleanup through recurrent dynamics. it is useful as a baseline for future associative-memory circuit work, but it should not be read as evidence that kda already matches ca3 or that hippocampal memory has been captured without pattern separation.
