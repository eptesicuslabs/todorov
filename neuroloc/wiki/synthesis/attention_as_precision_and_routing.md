# attention as precision and routing

status: current (as of 2026-04-23).

## thesis

the strongest cross-source reading is that attention is not best modeled as a separate inner agent. it is better modeled as precision control plus routing:

- increase gain where uncertainty or relevance is high
- open the right pathway at the right time
- suppress competing pathways that would waste bandwidth or create interference

## supporting lines

- predictive-processing accounts emphasize precision weighting
- workspace accounts emphasize limited-bandwidth broadcast
- systems-neuroscience accounts emphasize thalamic loops, synchrony, and modulatory state

taken together, they support a routing-and-access picture more than a “spotlight” metaphor.

## implications for the backlog

- routing is a first-class primitive alongside memory and compression
- the project should not assume one routing substrate yet
- useful candidate abstractions include dynamic access control, state-dependent gating, and controlled broadcast

## what remains open

- whether phase-like control deserves direct implementation
- whether routing should live in the memory path, the state path, or a separate control plane

## see also

- [[systems_neuroscience_research]]
- [[visual_sources_systems_neuroscience]]
- [[cognitive_architecture_research]]
- [[cross_scale_building_blocks_for_biological_computation]]
- [[synchrony_to_dynamic_routing]]
- [[state_action_memory_architecture_direction]]
