# Effective Receptive Field Analysis

No ERF measurements yet. Will be populated after Phase 1 training.

## Measurement Method

Gradient-based ERF: backpropagate from a single output position through the
trained model, measure gradient magnitude at each input position.

## Expected Behavior

- KDA layers: ERF grows with training as delta-rule state accumulates
- Mamba-3 layers: ERF depends on learned A matrix decay rates
- MLA layers: full context ERF (softmax attention)

## Tools

See src/utils/erf.py for measurement utilities.
