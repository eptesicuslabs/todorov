# neuroloc index

status: current (as of 2026-04-17, curriculum pivot recorded).

this is the flat reference catalog of every article in the wiki, grouped
by topic within each top-level directory. for guided navigation, start
at [[start_here]]. for the rules that govern every article below, see
[[OPERATING_DIRECTIVE]]. for the canonical project state, see
[[PROJECT_PLAN]].

## entry points for new readers

- [[OPERATING_DIRECTIVE]] — binding rules for the wiki
- [[PROJECT_PLAN]] — canonical project state (current run, decision rules, prior runs)
- [[Home]] — top-level eptesicus laboratories landing page
- [[log]] — operation log: append-only running narrative of project activity
- [[_audit_2026-04-16]] — audit of the wiki state before the 2026-04-16 refactor
- [[concepts/start_here]] — overview for new readers
- [[concepts/the_brain_in_one_page]] — 80/20 bio overview
- [[concepts/neuroscience_for_ml_engineers]] — neuroscience mapped to ML

## current project-state analyses (synthesis/)

all 11 articles in `synthesis/` are the load-bearing project-level
reasoning documents. one is superseded, ten are current.

- [[synthesis/substrate_requires_architectural_change]] — the
  post-run-3 analysis after the cognition corpus returned 0% passkey.
  the canonical reference for why the next paid run needs an
  architectural intervention, not another training-corpus change.
  ranks five candidate interventions (A-E). supersedes the "next paid
  run needs a different corpus" stance in `training_objective_vs_architectural_goal`.
- [[synthesis/training_objective_vs_architectural_goal]] — the
  root-cause analysis after five paid runs on fineweb-edu. reasoning
  structure intact; its proposed discriminant (cognition corpus) was
  executed as run3 and returned 0%, which triggered the next article
  above. supersedes `linear_attention_retrieval_wall`.
- [[synthesis/slot_memory_design]] — slot memory substrate design
  and its paid-run empirical status (now three paid runs deep)
- [[synthesis/correction_field_memory]] — prediction-residual value storage
- [[synthesis/compression_beyond_quantization]] — six-mechanism compound compression
- [[synthesis/compression_and_bottlenecks]] — bio vs todorov compression
- [[synthesis/local_vs_global_computation]] — cortical local recurrence
- [[synthesis/recurrence_vs_feedforward]] — when recurrence helps
- [[synthesis/sparsity_from_biology_to_ternary_spikes]] — firing-rate analyses
- [[synthesis/timescale_separation]] — KDA / Mamba / MLA timescale split
- [[synthesis/linear_attention_retrieval_wall]] — SUPERSEDED. retained for evidence continuity.

## teaching curriculum (2026-04-17, active workstream)

**the project's active workstream is the teaching PDF curriculum** specified in the plan file at `~/.claude/plans/compressed-dancing-haven.md`. paid compute is paused indefinitely; the architectural-intervention track is held in the research backlog until the curriculum completes. the curriculum specifies 36 chapters across 6 phases at 20-25 pages per chapter in English LaTeX:

- phase 1 (ch. 1-8): foundations of math — numbers, change, accumulation, vectors, matrices, multi-dimensional change, probability, information
- phase 2 (ch. 9-14): foundations of biology — cells, electricity, action potentials, synapses, compartmental neurons, circuits
- phase 3 (ch. 15-20): computation in the brain — what neurons compute, population coding, plasticity, cortex, hippocampus, consolidation
- phase 4 (ch. 21-26): math for machine learning — loss, gradient descent, backpropagation, neural networks, recurrence, attention
- phase 5 (ch. 27-32): advanced architectures and compression — associative memory, modern Hopfield, fast-weight memory, sparse/quantized coding, generative memory, the six-mechanism compression thesis
- phase 6 (ch. 33-36): paper implementation — how to read a paper, modern Hopfield from Ramsauer 2020, Titans from the paper, the project's slot memory from scratch

per-chapter production protocol: outline → user approval → parallel research agents (wiki + cited papers + classical textbooks + online resources + reference implementations + recent papers) → prosecutor agents validate research → LaTeX draft → prosecutor on draft → user review → revision → finalized PDF → next chapter. naming rule: published-technique names only when quoting external sources or naming external paper's architectures; the project's own components use the glossary terms (`matrix memory`, `compressed attention`, `slot memory`, `output gate`, `surprise ratio`). full detail in the plan file and in `PROJECT_PLAN.md` section "curriculum track".

## paid-run cards (tests/, historical context only)

frozen evidence records, one per paid run:

- [[tests/god_run_results]] (2026-04-11, 283M, bundle of all features)
- [[tests/god_run_v2_results]] (2026-04-12, 283M, 31 prosecutor fixes)
- [[tests/run1_baseline_noerasure_results]] (2026-04-14, 353M, all bundle features off)
- [[tests/run2_slot_memory_first_launch_results]] (2026-04-15, 355M, broken retention)
- [[tests/run2_slot_memory_retention_fixed_results]] (2026-04-15, 355M, retention fixed, FLA active, the fifth paid run)
- [[tests/run3_cognition_phase1_results]] (2026-04-17, 355M, synthetic cognition corpus, val_bpb plateaued at alphabet prior, passkey 0/100, the sixth paid run)

## mistakes (mistakes/, historical context only, append-only)

- [[mistakes/run2_slot_memory_decay_copy_paste]] — inherited retention bug
- [[mistakes/run2_slot_memory_fla_silent_fall_through]] — FLA not installed silent slowdown

## flat catalog — articles by topic (sums to 207 content articles; add 6 navigation/meta files plus tests/index.md for 214 total wiki markdown files on disk)

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

## concepts (7 articles)
- [[start_here]] -- entry point for newcomers, reading order
- [[the_brain_in_one_page]] -- 80/20 neuroscience overview for ML engineers
- [[neuroscience_for_ml_engineers]] -- the big primer, 7 parts
- [[mathematical_foundations]] -- math primer with worked examples
- [[todorov_biology_map]] -- master mapping of every component to biology
- [[glossary]] -- 55 terms in plain language with ML analogs
- [[notation]] -- mathematical notation conventions

## entities (33 notes)
- entities/hebb.md, entities/bi_poo.md, entities/turrigiano.md
- entities/gerstner.md, entities/hodgkin_huxley_researchers.md, entities/izhikevich.md
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

## synthesis (11 articles)
- synthesis/substrate_requires_architectural_change.md -- post-run-3 analysis: six paid runs at 0% passkey across two substrates and two corpora trigger the architecture-cannot-be-trained branch; ranks five candidate interventions (output gate init, auxiliary retrieval loss, orthogonal key init, warm-start, substrate replacement)
- synthesis/training_objective_vs_architectural_goal.md -- root-cause analysis after five paid runs: LM loss on fineweb-edu does not exercise the memory substrate. the proposed discriminant (cognition corpus) ran as run3 and returned 0%, which triggered the article above
- synthesis/slot_memory_design.md -- softmax addressing over prototype keys, surprise-gated lru writes, output gate; substrate for run2_slot_memory and run3_cognition_phase1
- synthesis/correction_field_memory.md -- prediction-residual value storage; memory_capacity_delta=0 falsified by trained-prediction sim
- synthesis/compression_beyond_quantization.md -- six-mechanism compound compression thesis
- synthesis/linear_attention_retrieval_wall.md -- SUPERSEDED: the five-failure-mode diagnosis. retained for evidence continuity
- synthesis/sparsity_from_biology_to_ternary_spikes.md -- metabolic mandate, energy-information tradeoff, gradient flow constraint, 41% vs cortical 2-10%
- synthesis/timescale_separation.md -- nested oscillatory clocks, cross-frequency coupling, todorov's two fixed timescales
- synthesis/local_vs_global_computation.md -- cortical local recurrence, dendritic compartments, source segregation vs residual stream
- synthesis/compression_and_bottlenecks.md -- DG pattern separation, hippocampal indexing, consolidation pipeline, capacity limits as features
- synthesis/recurrence_vs_feedforward.md -- canonical microcircuit recurrence, attractor dynamics, error correction, KDA/Mamba3 comparison

## bridge (19 articles)
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
- bridge/memory_compression_to_tiered_architecture.md -- 5-tier memory architecture proposal for phase 6+

## tests (21 records)

### paid-run cards (6)
- tests/god_run_results.md -- first paid neural-machine run, 2026-04-11, 283M params, val_bpb 1.3950, passkey 0/20
- tests/god_run_v2_results.md -- paid re-run with 17+14 prosecutor fixes, 2026-04-12, val_bpb 1.4453, passkey 0/100
- tests/run1_baseline_noerasure_results.md -- paid run with all bundle features off, 2026-04-14, 353M, val_bpb 1.4499, passkey 0/100
- tests/run2_slot_memory_first_launch_results.md -- first slot-memory paid run, 2026-04-15, inherited retention bug, val_bpb 1.5107, passkey 0/100
- tests/run2_slot_memory_retention_fixed_results.md -- fifth paid run, 2026-04-15, retention fixed, FLA active, val_bpb 1.4777, passkey 0/100
- tests/run3_cognition_phase1_results.md -- sixth paid run, 2026-04-17, synthetic cognition corpus (50% passkey / 30% kv recall / 20% copy), 355M, val_bpb 6.3519 (plateaued at alphabet prior from step 150), passkey 0/100 at 256 and 1024. executed the training_objective_vs_architectural_goal.md discriminant and triggered substrate_requires_architectural_change.md

### pilot experiments (5)
- tests/2026-04-07_pattern_completion_baseline.md -- first dated experiment record; repaired ca3-like attractor baseline with shuffled control, sweeps, and metrics json
- tests/2026-04-07_kwta_vs_threshold_pilot.md -- lateral-inhibition bridge pilot; matched-sparsity threshold vs k-wta with exact-support and active-fraction metrics
- tests/2026-04-08_leak_vs_carry_pilot.md -- single-neuron bridge pilot; explicit leak vs atmn-style carry with paired-pulse retention, drift scaling, and passive lif anchor metrics
- tests/2026-04-09_bcm_alpha_pilot.md -- plasticity bridge pilot; bcm-like adaptive alpha with activity-dependent forgetting and state norm stabilization
- tests/2026-04-09_gp_vs_bilinear_pilot.md -- spatial bridge pilot; pga geometric product vs random bilinear at random initialization

### simulation results and analyses (10)
- tests/god_run_findings.md -- original long-form synthesis of god_run's results
- tests/decay_sweep_results.md -- asymmetric matrix memory d_head=64 decay sweep; retention knee at decay=0.90 (32 patterns), decay=0.95 (64 patterns)
- tests/head_dim_sweep_results.md -- head-dim 32-256 sweep at d_head=64; p*(d) sub-linear; retention dominates width
- tests/overwrite_sweep_results.md -- erasure at decay=0.90 hurts all 7 encodings at 32-pattern knee
- tests/encoding_simulation_round_a.md -- symmetric memory sign-only vs three-level encoding comparison
- tests/encoding_simulation_round_b.md -- asymmetric matrix memory encoding comparison; capacity ceiling below symmetric
- tests/correction_field_trained_prediction_results.md -- trained-predictor correction-field sim; memory_capacity_delta=0 at every quality
- tests/multi_resolution_head_split_results.md -- fast/medium/slow heads with surprise gates; rare-class recall improves
- tests/thinking_loop_prototype_results.md -- recurrent hidden-state refinement pilot on modular arithmetic
- tests/aesthetic_logger_prototype.md -- phase 6a logging module, implemented not yet wired (current status)

## simulations (35+ scripts across 17 script-containing directories plus 3 root-level utilities)
- simulations/single_neuron/ (3 scripts: lif_fi_curve leak-validation, adex_patterns, izhikevich_gallery)
- simulations/plasticity/ (3 scripts: stdp_weight_evolution, homeostatic_scaling, bcm_alpha_pilot)
- simulations/sparse_coding/ (2 scripts: sparse_coding_demo, hierarchical_ternary)
- simulations/predictive_coding/ (1 script: predictive_coding_2level)
- simulations/cortical_microcircuit/ (2 scripts: canonical_circuit, sparse_topology)
- simulations/neuromodulation/ (1 script: dopamine_rpe)
- simulations/lateral_inhibition/ (1 script: wta_dynamics)
- simulations/oscillations/ (1 script: gamma_ping)
- simulations/memory/ (10 scripts: pattern_completion, capacity_scaling, imagination_recombination, asymmetric_outer_product_recall, correction_field_capacity, correction_field_trained_prediction, multi_resolution_head_split, slot_buffer_capacity, slot_surprise_writes, slot_integration)
- simulations/attention/ (1 script: biased_competition)
- simulations/dendritic/ (1 script: multicompartment_neuron)
- simulations/energy/ (1 script: energy_comparison)
- simulations/development/ (1 script: critical_period)
- simulations/spatial/ (2 scripts: grid_cell_model, gp_vs_bilinear_pilot)
- simulations/consciousness/ (1 script: ignition_dynamics)
- simulations/prototypes/ (3 scripts: rate_coded_spike, linoss_dynamics, forward_learning)
- simulations/reasoning/ (1 script: thinking_loop_prototype)
- simulations/shared.py, simulations/suite_registry.py, simulations/suite_runner.py (root-level utilities)

## knowledge (40 articles)
- knowledge/unified_theory.md -- crbr formulation unifying kda, mamba-3, mla, spikes, swiglu, and gp under one mathematical object
- knowledge/delta_rule_theory.md -- delta-rule linear attention, online regression view, and fast-weight state updates
- knowledge/kda_channel_gating.md -- kimi delta attention with channel-wise forgetting and constrained dplr implementation notes
- knowledge/mamba3_architecture.md -- mamba-3 overview covering exp-trapezoidal discretization, complex-valued state, and mimo structure
- knowledge/mla_compression.md -- multi-head latent attention as low-rank kv compression with cache-saving analysis
- knowledge/hybrid_architectures.md -- evidence for the 3:1 linear-to-attention ratio across kimi, qwen3, and olmo-style hybrids
- knowledge/context_extension.md -- long-context extension methods such as cope and related position-encoding strategies
- knowledge/training_efficiency.md -- flash-linear-attention, triton kernels, and training-efficiency constraints
- knowledge/ternary_spikes.md -- gerhard empirical findings on adaptive-threshold ternary spikes and firing-rate behavior
- knowledge/geometric_algebra.md -- projective geometric algebra g(3,0,1), gatr notes, and gp implementation context
- knowledge/multimodal_encoding.md -- techniques for mapping images, audio, and 3d data into unified token sequences
- knowledge/papers.md -- compiled index of papers referenced across the todorov knowledge files
- knowledge/papers_library.md -- extended eptesicus paper library organized by architecture relevance and actionability
- knowledge/neuroscience_research_2026.md -- curated peer-reviewed neuroscience research library (2020-2026), 16 papers across dendritic computation, cortical microcircuits, memory systems, energy constraints, attention, oscillations
- knowledge/perception_and_consciousness_research.md -- curated research library on perception, consciousness, attention, embodiment, and time at the cognitive level, 30+ sources from helmholtz to cogitate 2025
- knowledge/decision_and_emotion_research.md -- curated research library on decision making, emotion, reward, intuition, social cognition, and rationality, 25+ sources across evidence accumulation, somatic markers, dopamine, heuristics, theory of mind, predictive brain
- knowledge/memory_systems_research.md -- curated cognitive and systems-level memory research library, 25 sources across spatial memory, episodic and semantic memory, working memory, reconsolidation, expertise, prospective memory, and cross-species memory
- knowledge/sleep_and_dreaming_research.md -- curated research library on sleep stages as computational operations, 9 sources across shy synaptic renormalization, swr replay, rem emotional processing, dreaming as generative model optimization, sleep deprivation, lucid dreaming, glymphatic clearance
- knowledge/cerebellum_research.md -- curated research library on the cerebellum as computational organ, 7 sources across expansion recoding, forward models, cerebellar cognition, scaling laws, and supervised error signals
- knowledge/neurogenesis_and_plasticity_research.md -- curated research library on adult neurogenesis, structural plasticity, and critical periods, 11 sources across hippocampal neurogenesis, pattern separation, forgetting, spine turnover, critical period closure, pharmacological reopening, and experience-dependent growth
- knowledge/language_in_the_brain_research.md -- curated research library on biological language processing, 8 sources across dual-stream architecture, the fedorenko language network, n400 prediction, statistical learning, motor theory, sign language, and linguistic relativity
- knowledge/glial_computation_research.md -- curated research library on glial cells as active computational participants, 9 sources across glia:neuron ratio, tripartite synapse, astrocytic ltp gating, gliotransmission, activity-dependent myelination, microglial pruning, active forgetting, and opc synaptic input
- knowledge/autonomic_and_interoception_research.md -- curated research library on body-brain interface, 10 sources across insular interoception, vagus nerve, gut-brain axis, cardiac gating, allostasis, gate control theory, neuromatrix, somatic markers, and polyvagal theory
- knowledge/connectomics_and_wiring_research.md -- curated research library on brain wiring architecture, 10 sources across complete connectomes, small-world topology, rich-club hubs, wiring optimization, communication costs, sparse connectivity, non-random connectivity, and human vs mouse cortex

- knowledge/executive_function_research.md -- prefrontal working memory maintenance, basal ganglia gating, conflict monitoring, metacognition, planning, task switching. miller & cohen 2001, o'reilly & frank 2006, botvinick 2001, fleming & dolan 2012
- knowledge/motor_and_forward_models_research.md -- cerebellum as forward model, motor cortex as dynamical system, sequence chunking, optimal feedback control, active inference. wolpert 1998, churchland 2012, graybiel 1998, todorov & jordan 2002, friston 2011
- knowledge/imagination_research.md -- hippocampal scene construction, constructive episodic simulation, suppression circuit for controlled vs intrusive thought, dmn-executive coupling for creativity, visual imagery, dreaming as offline generation, forward models as primitive imagination. hassabis 2007, schacter & addis 2007, anderson 2025, beaty 2018, pearson 2019, deperrois 2022
- knowledge/compression_architecture.md -- novel compression proposals for the neural machine: hierarchical ternary coding (0.37 bits/dim), state-predictive residual coding, consolidated state snapshots, content-addressable sparse memory, ternary weight matrices. combines hippocampal indexing, pattern separation, chunking, predictive coding, and consolidation principles

- knowledge/learning_rules_research.md -- curated research library on non-backprop learning rules, 8 sources across target propagation, predictive coding at depth, forward-forward, evolution strategies, e-prop on neuromorphic hardware, three-factor hebbian, and the 1-15pp scale gap
- knowledge/ternary_compression_research.md -- curated research library on ternary and extreme quantization, 9 sources across bitnet b1.58, w1a1 gap, paretoq pareto frontier, matmul-free lm, hardware efficiency, fpga accelerators, scaling laws, and ste improvements
- knowledge/memory_capacity_research.md -- curated research library on associative memory capacity, 9 sources across modern hopfield exponential capacity, tight upper bounds, gated deltanet mqar, ternary synaptic scaling, ssm retrieval horizon, rnn formal lower bounds, svd kv compression, memory caching, and the unmeasured joint of outer product + decay + ternary
- knowledge/imagination_computation_research.md -- curated research library on computational imagination, 8 sources across novelty as feature superposition, meta-learning for compositionality, dreamerv3 latent imagination, cmmd and vendi score quality metrics, logit arithmetic, outer-product generative interpolation, and memorization-to-generalization phase transition
- knowledge/sparse_connectivity_research.md -- curated research library on sparse network connectivity, 8 sources across lottery tickets in transformers, sparsegpt at 175b, deepseek moe, small-world acceleration, rigl dynamic sparse training, biological wiring cost optimization, connectome robustness, and adaptive rewiring emergence

- knowledge/gpu_spike_implementation_research.md -- curated research library on gpu spiking implementations, covering spikingjelly cupy fusion (11x speedup), sparseprop O(log N) binary heap, parallel scan solutions for lif reset (psn, prf, spikingssms, bullet trains), eventprop exact spike-timing gradients, ttfs 0.3 spikes/neuron, matmul-free lm ternary weights, temporal fusion 5-40x speedup, and dense-beats-sparse crossover at 40% firing rate
- knowledge/gpu_architecture_building_research.md -- curated research library on gpu architecture engineering, covering triton vs cuda tradeoffs, torch.autograd.Function + ste for spike backprop, flash attention tiling for outer-product accumulation, fla chunkwise parallel scan, gradient checkpointing, bf16/fp8 precision, gradcheck verification, and nsight compute profiling
- knowledge/phase_coding_research.md -- curated research library on phase-based neural coding, covering lisman theta-gamma model, liebe et al. 2025 falsification of phase-order-encodes-sequence-order, trained rnns developing limit cycles (pals et al. 2024), linoss forced harmonic oscillators (rusch & rus iclr 2025 oral), akorn kuramoto synchronization, rope as structural phase code (novel observation), complex-valued position embeddings (wang et al. iclr 2020), and spike timing vs rate coding

- knowledge/unified_learning_hypothesis.md -- the forward pass IS the learning step: outer product = hebbian, BCM alpha = metaplasticity, delta rule = error correction, k-WTA = competition, prediction error = teaching signal. unified learning where computation and training are the same operation. testable hypothesis with specific experiment design.
- knowledge/compression_novelty.md -- novelty analysis of hierarchical k-WTA + ternary compression. confirmed novel: no prior work on runtime activations with CKA quality validation. closest: ComPEFT (EMNLP 2024) on weight deltas. architecture-agnostic, applies to any activation tensor.
- knowledge/biological_vision_research.md -- curated research library on biological vision, retinal processing, center-surround receptive fields, ventral/dorsal streams, and visual cortex hierarchies
- knowledge/generative_memory_research.md -- curated library on memory compression via generative models: quantization ceilings, Dreamer V3, INRs, hypernetworks, Larimar, Memorizing Transformers, Titans, DeltaKV

## statistics
- total mechanism articles: 61
- total bridge notes: 19
- total synthesis articles: 11 (10 current + 1 superseded)
- total test records: 21 (6 paid-run cards + 5 pilots + 10 simulation results including aesthetic_logger_prototype + god_run_findings; excluding tests/index.md)
- total entity notes: 33
- total comparison articles: 13
- total concept articles: 7
- total knowledge articles: 40
- total mistake docs: 2
- total navigation / meta: 6 (INDEX.md, OPERATING_DIRECTIVE.md, PROJECT_PLAN.md, _audit_2026-04-16.md, Home.md, log.md)
- total simulations: 35+ scripts across 17 script-containing directories plus 3 root-level utilities (shared.py, suite_registry.py, suite_runner.py)
- last updated: 2026-04-17
