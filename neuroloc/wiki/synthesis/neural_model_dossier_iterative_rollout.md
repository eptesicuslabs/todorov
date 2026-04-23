# neural model dossier: iterative rollout

status: current (as of 2026-04-23).

## claim

reasoning for this phase means latent-state refinement and counterfactual rollout, not longer text output. extra internal compute must help hard cases more than easy cases.

## mathematical operation

the candidate loop is:

```text
z_0 = encode(observation, memory)
z_{i+1} = refine(z_i, memory, query)
answer_i = readout(z_i)
```

the number of internal iterations is controlled during evaluation: 0, 1, 3, and 5.

## evidence basis

world-model, energy-refinement, and iterative-reasoning work support extra computation as latent refinement. the project still needs a local proof that additional internal steps improve the neural model's own hard cases.

## failure mode targeted

the model may spend extra compute without changing state, may improve easy cases only, or may leak answers through longer input-output traces rather than internal state.

## required test material

use latent worlds with matched easy and hard cases. hard cases should require hidden-state disambiguation or counterfactual choice, not longer surface patterns.

## success metrics

- hard-case rollout gain
- easy-case rollout gain
- counterfactual accuracy
- state refinement accuracy
- compute-normalized improvement

## controls

- zero-iteration readout
- random refinement steps
- matched parameter budget
- no-memory
- oracle latent state
- shuffled counterfactual labels

## telemetry

- state-change norm per iteration
- confidence per iteration
- memory-read concentration per iteration
- answer flip rate
- compute cost

## kill condition

kill the mechanism if extra iterations improve easy and hard cases equally, if state does not change meaningfully, or if gains disappear under shuffled counterfactual controls.

## see also

- [[neural_model_research_test_material_plan]]
