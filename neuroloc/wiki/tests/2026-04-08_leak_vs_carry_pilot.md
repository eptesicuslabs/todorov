# 2026-04-08 leak vs carry pilot

status: historical context only. frozen as of 2026-04-08. do not edit.

test type: bridge-validation pilot

script:
- `neuroloc/simulations/single_neuron/lif_fi_curve.py`

artifacts:
- `neuroloc/simulations/single_neuron/lif_leak_validation.png`
- `neuroloc/simulations/single_neuron/lif_leak_validation_metrics.json`

evolution:
- this replaces the old plot-only lif f-i demo with a quantitative bridge test for [[neuron_models_to_atmn]]
- it follows the shared artifact pattern established in [[tests/2026-04-07_pattern_completion_baseline|2026-04-07 pattern completion baseline]] and extended in [[tests/2026-04-07_kwta_vs_threshold_pilot|2026-04-07 k-wta vs threshold pilot]]

## what was done

- replaced the old constant-current lif visualization with a matched discrete-time comparison between explicit leak, current atmn-style carry, and an integrator control
- kept a passive brian2 lif trace as a qualitative biological anchor for the paired-pulse task instead of using the anchor as the whole verdict
- ran paired-pulse sweeps across gaps 1, 4, 8, 16, 32, 64 ms
- ran long-sequence drift sweeps across lengths 128, 256, 512, 1024 under zero-mean noise
- swept explicit leak constants 5, 10, 20, 30 ms to check whether the result depended on a single arbitrarily chosen decay rate

## configuration

- dt: 1.0 ms
- threshold: 15.0 mV
- explicit leak taus: 5, 10, 20, 30 ms
- atmn carry retention: `1 / 2 = 0.5` per step
- gap trials: 32 per condition
- gap pulse amplitude: 0.1 mV
- gap noise sigma: 0.004 mV
- drift lengths: 128, 256, 512, 1024
- drift trials: 48 per condition
- drift noise sigma: 0.5 mV
- passive lif anchor: `tau_m = 20 ms`, `r = 100 mohm`
- numpy seed: 17

## key results

- mean second-pulse peak error to the passive lif anchor was 0.507 mV for `explicit_leak_tau20` and 0.166 mV for `atmn_carry_tau2`; paired delta 0.341 mV with permutation `p = 0.001`
- at 16 ms gap, retained fraction was 0.447 for `explicit_leak_tau20` and -0.001 for `atmn_carry_tau2`; paired delta 0.448 with permutation `p = 0.001`
- at length 1024, state standard deviation was 1.567 mV for `explicit_leak_tau20` and 0.576 mV for `atmn_carry_tau2`; paired delta 0.990 mV with permutation `p = 0.001`
- the integrator control retained almost everything at 16 ms gap (`0.996`) but had the worst long-sequence drift at length 1024 (`4.549 mV` state standard deviation)
- within the leak tau sweep, shorter explicit leak improved anchor matching but did not remove the basic tradeoff: `explicit_leak_tau5` had mean peak error 0.319 mV, still above `atmn_carry_tau2` at 0.166 mV

## verdict

this pilot does not justify the simple claim that adding explicit leak is a strict improvement over current atmn-style carry.

the defensible claim is narrower: explicit leak improves cross-gap subthreshold retention in a matched discrete-time toy recurrence, but the same change worsens anchor matching on the second pulse and increases long-sequence state variance relative to the carry condition. in other words, leak buys memory and spends stability.

for the bridge note, the right update is not "explicit leak validated." it is "explicit leak introduces a retention-stability tradeoff that may still be worth testing downstream, but only with clear awareness that the toy pilot did not produce a dominant win."

## limitations

- the statistical verdict is based on matched discrete-time toy recurrences, not on trained language-model activations or downstream bpb
- the brian2 lif trace is a qualitative biological anchor for pulse integration, not a production runtime claim
- the atmn carry condition here is a minimal recurrence abstraction, not the full trained neuron model with learned per-neuron thresholds
- the paired-pulse regime is intentionally subthreshold; different conclusions could emerge in a suprathreshold or task-trained regime
- the recorded metrics json marks the run as `git_dirty: true`, so exact reproduction refers to the working-tree state captured in the artifact metadata rather than a clean commit alone