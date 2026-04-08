# neuroloc index

this is the flat reference catalog. for guided navigation, start at [[start_here]].

## mechanisms (61 articles)

### single neuron models
- mechanisms/leaky_integrate_and_fire.md -- simplest useful neuron model, RC circuit, f-I curve
- mechanisms/hodgkin_huxley.md -- gold standard biophysical model, 4 coupled ODEs, Na+/K+ channels
- mechanisms/adaptive_exponential.md -- AdEx, exponential spike + adaptation variable w
- mechanisms/izhikevich_model.md -- quadratic dynamics, 20+ firing patterns from 4 parameters

### dendritic computation
- mechanisms/dendritic_computation.md -- dendrites as active computational elements
- mechanisms/dendritic_spikes.md -- Na+, Ca2+, NMDA spike types
- mechanisms/apical_amplification.md -- Larkum 2013, BAC firing, top-down/bottom-up coincidence
- mechanisms/two_layer_neuron.md -- Poirazi 2003, single neuron = two-layer network

### synaptic plasticity
- mechanisms/hebbian_learning.md -- correlation-based learning, outer product rule
- mechanisms/stdp.md -- spike-timing-dependent plasticity, asymmetric learning window
- mechanisms/bcm_theory.md -- sliding threshold, metaplasticity
- mechanisms/homeostatic_plasticity.md -- synaptic scaling, target firing rate
- mechanisms/short_term_plasticity.md -- facilitation, depression, Tsodyks-Markram model
- mechanisms/three_factor_learning.md -- Delta_w = eta * f(pre,post) * M(t), eligibility traces, neuromodulatory gating

### development and learning
- mechanisms/critical_periods.md -- time windows of maximal plasticity, PV+ maturation
- mechanisms/synaptic_pruning.md -- exuberant connectivity pruned ~50%
- mechanisms/developmental_self_organization.md -- activity-dependent structure without supervision

### energy and metabolism
- mechanisms/brain_energy_budget.md -- Attwell & Laughlin 2001, 20W, energy per spike
- mechanisms/energy_efficient_coding.md -- energy as selective pressure, optimal ~6% firing
- mechanisms/metabolic_constraints_on_computation.md -- <1% neurons active, sparse coding as survival

### neural coding
- mechanisms/efficient_coding.md -- Barlow 1961, redundancy reduction, infomax
- mechanisms/sparse_coding.md -- Olshausen & Field 1996, overcomplete dictionaries, L1 sparsity
- mechanisms/population_coding.md -- rate vs temporal coding, Fisher information
- mechanisms/sparse_distributed_representations.md -- SDRs, capacity, noise robustness

### predictive processing
- mechanisms/predictive_coding.md -- Rao & Ballard 1999, hierarchical prediction errors
- mechanisms/free_energy_principle.md -- Friston, variational free energy, active inference
- mechanisms/precision_weighting.md -- precision = inverse variance, attention as precision

### lateral inhibition and competition
- mechanisms/lateral_inhibition.md -- center-surround receptive fields, edge enhancement
- mechanisms/divisive_normalization.md -- Carandini & Heeger 2012, canonical computation
- mechanisms/winner_take_all.md -- WTA circuits, Maass 2000, k-WTA, softmax relationship
- mechanisms/inhibitory_interneurons.md -- PV+, SST+, VIP+, E/I balance

### neuromodulation
- mechanisms/dopamine_system.md -- Schultz 1997, reward prediction error, TD learning
- mechanisms/acetylcholine_system.md -- encoding/retrieval switch, learning rate
- mechanisms/norepinephrine_system.md -- adaptive gain, explore/exploit, neural interrupt
- mechanisms/serotonin_system.md -- raphe nuclei, temporal discounting, risk sensitivity, 5-HT
- mechanisms/neuromodulatory_framework.md -- Doya 2002, metalearning

### inhibitory signaling
- mechanisms/gaba_signaling.md -- GABA_A fast ionotropic, GABA_B slow metabotropic, developmental switch
- mechanisms/nmda_receptors.md -- coincidence detection, voltage-dependent Mg2+ block, biological AND gate

### action selection
- mechanisms/basal_ganglia.md -- direct/indirect/hyperdirect pathways, selective disinhibition, action gating

### cortical microcircuits
- mechanisms/cortical_column.md -- Mountcastle 1957, repeating unit
- mechanisms/laminar_processing.md -- 6 layers, feedforward vs feedback
- mechanisms/excitatory_inhibitory_balance.md -- 80:20 ratio, balanced networks
- mechanisms/canonical_microcircuit.md -- Douglas & Martin, recurrent amplification

### memory systems
- mechanisms/hippocampal_memory.md -- DG, CA3, CA1, pattern separation/completion
- mechanisms/complementary_learning_systems.md -- McClelland 1995, fast + slow
- mechanisms/memory_consolidation.md -- synaptic/systems consolidation, sharp wave ripples
- mechanisms/pattern_completion.md -- Hopfield 1982, modern Hopfield, softmax equivalence

### spatial computation
- mechanisms/place_cells.md -- O'Keefe 1971, cognitive map, phase precession
- mechanisms/grid_cells.md -- Hafting 2005, hexagonal lattice, path integration
- mechanisms/path_integration.md -- dead reckoning, head direction cells, speed cells
- mechanisms/cognitive_maps.md -- Tolman, abstract relational maps, TEM

### attention
- mechanisms/selective_attention.md -- Desimone & Duncan 1995, biased competition
- mechanisms/normalization_model_of_attention.md -- Reynolds & Heeger 2009
- mechanisms/feature_vs_spatial_attention.md -- spatial, feature-based, object-based, binding

### oscillatory dynamics
- mechanisms/gamma_oscillations.md -- 30-100 Hz, PING/ING, temporal binding
- mechanisms/theta_oscillations.md -- 4-8 Hz, theta-gamma coupling, phase precession
- mechanisms/neural_synchrony.md -- communication through coherence, cross-frequency coupling

### consciousness and integration
- mechanisms/global_workspace_theory.md -- Baars 1988, ignition, global broadcast
- mechanisms/integrated_information_theory.md -- Tononi 2004, Phi
- mechanisms/ignition_dynamics.md -- all-or-none transition, bifurcation dynamics
- mechanisms/thalamocortical_loops.md -- thalamus as relay and regulator

## concepts (6 articles)
- [[start_here]] -- entry point for newcomers, reading order
- [[the_brain_in_one_page]] -- 80/20 neuroscience overview for ML engineers
- [[neuroscience_for_ml_engineers]] -- the big primer, 7 parts
- [[mathematical_foundations]] -- math primer with worked examples
- [[todorov_biology_map]] -- master mapping of every component to biology
- [[glossary]] -- 55 terms in plain language with ML analogs
- [[notation]] -- mathematical notation conventions

## entities (33 notes)
- entities/hebb.md, entities/bi_poo.md, entities/turrigiano.md
- entities/gerstner.md, entities/hodgkin_huxley.md, entities/izhikevich.md
- entities/barlow.md, entities/olshausen_field.md
- entities/rao_ballard.md, entities/friston.md
- entities/douglas_martin.md, entities/mountcastle.md
- entities/carandini_heeger.md, entities/hartline.md
- entities/schultz.md, entities/doya.md
- entities/buzsaki.md, entities/fries.md
- entities/mcclelland.md, entities/hopfield.md
- entities/attwell_laughlin.md, entities/niven_laughlin.md
- entities/desimone_duncan.md, entities/reynolds_heeger.md
- entities/london_hausser.md, entities/larkum.md
- entities/hensch.md, entities/changeux.md
- entities/okeefe.md, entities/moser_moser.md
- entities/baars.md, entities/dehaene.md, entities/tononi.md

## comparisons (13 articles)
- comparisons/neuron_model_comparison.md -- HH vs LIF vs AdEx vs Izhikevich
- comparisons/sparse_vs_dense_representations.md -- sparse bio vs dense transformer
- comparisons/plasticity_local_vs_global.md -- local vs global learning rules
- comparisons/predictive_coding_vs_next_token.md -- predictive coding vs autoregressive
- comparisons/cortical_layers_vs_todorov_layers.md -- laminar vs serial stack
- comparisons/normalization_layernorm_vs_divisive.md -- LayerNorm vs divisive norm
- comparisons/oscillations_vs_recurrence.md -- oscillatory vs recurrent dynamics
- comparisons/memory_kda_vs_hippocampus.md -- KDA vs hippocampal memory
- comparisons/biological_vs_silicon_energy.md -- energy per operation comparison
- comparisons/biological_vs_transformer_attention.md -- selection vs retrieval
- comparisons/development_vs_training.md -- biological development vs ML training
- comparisons/pga_vs_grid_cells.md -- G(3,0,1) PGA vs grid cell computations
- comparisons/gwt_vs_transformer.md -- global workspace vs residual stream

## synthesis (5 articles)
- synthesis/sparsity_from_biology_to_ternary_spikes.md -- metabolic mandate, energy-information tradeoff, gradient flow constraint, 41% vs cortical 2-10%
- synthesis/timescale_separation.md -- nested oscillatory clocks, cross-frequency coupling, todorov's two fixed timescales
- synthesis/local_vs_global_computation.md -- cortical local recurrence, dendritic compartments, source segregation vs residual stream
- synthesis/compression_and_bottlenecks.md -- DG pattern separation, hippocampal indexing, consolidation pipeline, capacity limits as features
- synthesis/recurrence_vs_feedforward.md -- canonical microcircuit recurrence, attractor dynamics, error correction, KDA/Mamba3 comparison

## bridge (18 articles)
- bridge/neuron_models_to_atmn.md -- ATMN: no leak, batch reset, proposed fix
- bridge/sparse_coding_to_ternary_spikes.md -- 41% vs cortical 2-10%, STE constraint
- bridge/population_coding_to_spike_health.md -- MI, CKA, firing rate as population metrics
- bridge/plasticity_to_kda_delta_rule.md -- KDA is NOT STDP, it is Hopfield-like
- bridge/predictive_coding_to_training_objective.md -- alpha/beta as precision, defer to phase 6+
- bridge/cortical_microcircuit_to_layer_schedule.md -- 3:1 ratio from ML not biology
- bridge/lateral_inhibition_to_adaptive_threshold.md -- threshold is NOT divisive normalization
- bridge/neuromodulation_to_learning_and_gating.md -- alpha is NOT neuromodulatory gain
- bridge/oscillations_to_mamba3_rotation.md -- rotation is positional encoding, not oscillation
- bridge/memory_systems_to_kda_mla.md -- NOT complementary learning systems
- bridge/energy_efficiency_to_ternary_spikes.md -- 354x per-op correct, <1% system-level
- bridge/biological_attention_to_mla.md -- different operations, beta is closest analog
- bridge/dendritic_computation_to_swiglu.md -- single-branch vs 30-50 branches
- bridge/development_to_training_curriculum.md -- NOT critical period plasticity
- bridge/spatial_computation_to_pga.md -- weak to nonexistent connection
- bridge/global_workspace_to_residual_stream.md -- shared bus, NOT global workspace
- bridge/positional_encoding_to_rope.md -- theta phase precession, grid cell phase, tonotopy to RoPE rotation
- bridge/normalization_to_rmsnorm.md -- divisive normalization, synaptic scaling, gain control to RMSNorm

## tests (2 records)
- tests/2026-04-07_pattern_completion_baseline.md -- first dated experiment record; repaired ca3-like attractor baseline with shuffled control, sweeps, and metrics json
- tests/2026-04-07_kwta_vs_threshold_pilot.md -- lateral-inhibition bridge pilot; matched-sparsity threshold vs k-wta with exact-support and active-fraction metrics

## simulations (18 scripts)
- simulations/single_neuron/ (3 scripts: lif_fi_curve, adex_patterns, izhikevich_gallery)
- simulations/plasticity/ (2 scripts: stdp_weight_evolution, homeostatic_scaling)
- simulations/sparse_coding/ (1 script: sparse_coding_demo)
- simulations/predictive_coding/ (1 script: predictive_coding_2level)
- simulations/cortical_microcircuit/ (1 script: canonical_circuit)
- simulations/neuromodulation/ (1 script: dopamine_rpe)
- simulations/lateral_inhibition/ (1 script: wta_dynamics)
- simulations/oscillations/ (1 script: gamma_ping)
- simulations/memory/ (1 script: pattern_completion)
- simulations/attention/ (1 script: biased_competition)
- simulations/dendritic/ (1 script: multicompartment_neuron)
- simulations/energy/ (1 script: energy_comparison)
- simulations/development/ (1 script: critical_period)
- simulations/spatial/ (1 script: grid_cell_model)
- simulations/consciousness/ (1 script: ignition_dynamics)

## knowledge (11 articles)
- knowledge/neuroscience_research_2026.md -- curated peer-reviewed neuroscience research library (2020-2026), 16 papers across dendritic computation, cortical microcircuits, memory systems, energy constraints, attention, oscillations
- knowledge/perception_and_consciousness_research.md -- curated research library on perception, consciousness, attention, embodiment, and time at the cognitive level, 30+ sources from helmholtz to cogitate 2025
- knowledge/decision_and_emotion_research.md -- curated research library on decision making, emotion, reward, intuition, social cognition, and rationality, 25+ sources across evidence accumulation, somatic markers, dopamine, heuristics, theory of mind, predictive brain
- knowledge/memory_systems_research.md -- curated cognitive and systems-level memory research library, 25 sources across spatial memory, episodic/semantic, working memory, reconsolidation, expertise, prospective memory, cross-species memory
- knowledge/sleep_and_dreaming_research.md -- curated research library on sleep stages as computational operations, 9 sources across SHY synaptic renormalization, SWR replay, REM emotional processing, dreaming as generative model optimization, sleep deprivation, lucid dreaming, glymphatic clearance
- knowledge/cerebellum_research.md -- curated research library on the cerebellum as computational organ, 7 sources across expansion recoding (marr-albus), forward models, cerebellar cognition, scaling laws, supervised error signal
- knowledge/neurogenesis_and_plasticity_research.md -- curated research library on adult neurogenesis, structural plasticity, and critical periods, 11 sources across hippocampal neurogenesis, pattern separation, forgetting, spine turnover, critical period closure, pharmacological reopening, experience-dependent growth
- knowledge/language_in_the_brain_research.md -- curated research library on biological language processing, 8 sources across dual-stream architecture, fedorenko language network, N400 prediction, statistical learning, motor theory, sign language, linguistic relativity
- knowledge/glial_computation_research.md -- curated research library on glial cells as active computational participants, 9 sources across glia:neuron ratio, tripartite synapse, astrocytic LTP gating, gliotransmission, activity-dependent myelination, microglial pruning, active forgetting, OPC synaptic input
- knowledge/autonomic_and_interoception_research.md -- curated research library on body-brain interface, 10 sources across insular interoception, vagus nerve, gut-brain axis, cardiac gating, allostasis, gate control theory, neuromatrix, somatic markers, polyvagal theory
- knowledge/connectomics_and_wiring_research.md -- curated research library on brain wiring architecture, 10 sources across complete connectomes (C. elegans, drosophila), small-world topology, rich-club hubs, wiring optimization, communication costs, sparse connectivity, non-random connectivity, human vs mouse cortex

## statistics
- total mechanism articles: 61
- total bridge notes: 18
- total synthesis articles: 5
- total test records: 2
- total simulations: 18
- total entity notes: 33
- total comparison articles: 13
- total concept articles: 7
- total knowledge articles: 11
- last updated: 2026-04-08
