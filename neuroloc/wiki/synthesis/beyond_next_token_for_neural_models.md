# beyond next token for neural models

status: current (as of 2026-04-23).

## thesis

the research pass reinforces the existing project pivot away from next-token framing. the strongest external architecture stack is:

- latent state learning
- action-conditioned dynamics
- planning or value heads
- bounded memory

plain autoregression can still be a useful transport surface or auxiliary task, but it is not the right outer abstraction for the neural-model backlog.

![beyond-autoregression architecture families](../assets/statistics/s4_beyond_autoregression_architecture_families.svg)

## strongest imports

### latent-target state learning

the jepa and data2vec lines show that state can be learned without forcing surface reconstruction at every step.

### world-model dynamics

dreamer, muzero, and td-mpc2 show that control-relevant latent dynamics deserve explicit treatment.

### bounded support memory

titans and related work support surprise-gated bounded memory as a support layer rather than as a substitute for world models.

### factorized state

slot and object-centric lines remain the clearest route to structured entity state when the project later moves beyond symbolic tasks.

## what this does not mean

- it does not mean copying one external architecture family wholesale
- it does not mean retrieval is enough
- it does not mean active inference is already an engineering winner rather than a conceptual source

## see also

- [[architectures_beyond_next_token_research]]
- [[visual_sources_beyond_autoregression]]
- [[world_models_imagination_and_planning]]
- [[state_action_memory_architecture_direction]]
- [[dreamer_muzero_jepa_titans]]
- [[research_implications_for_neural_model_direction]]
