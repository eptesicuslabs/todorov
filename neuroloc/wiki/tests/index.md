# tests

status: current (as of 2026-04-16).

this folder records concrete simulations and experiments that were actually run.

each test page should include:
- date run
- status
- exact script or artifact tested
- what was done
- key quantitative outputs
- verdict
- limitations
- evolution link when the test extends an earlier baseline

## catalog

### paid-run cards

- [[tests/god_run_results]] -- first paid neural-machine run (2026-04-11, 283M, bundle of all features; val_bpb 1.3950, passkey 0/20)
- [[tests/god_run_v2_results]] -- paid re-run with 17+14 prosecutor fixes (2026-04-12, 283M; val_bpb 1.4453, passkey 0/100)
- [[tests/run1_baseline_noerasure_results]] -- paid run with all bundle features off (2026-04-14, 353M; val_bpb 1.4499, passkey 0/100)
- [[tests/run2_slot_memory_first_launch_results]] -- first slot-memory paid run (2026-04-15, 355M; inherited retention bug; val_bpb 1.5107, passkey 0/100)
- [[tests/run2_slot_memory_retention_fixed_results]] -- fifth paid run (2026-04-15, 355M; retention fixed, FLA active; val_bpb 1.4777, passkey 0/100)

### pilot experiments

- [[tests/2026-04-07_pattern_completion_baseline|2026-04-07 pattern completion baseline]] -- ca3-like attractor baseline with shuffled-weight control, corruption/load/scaling sweeps, and machine-readable metrics
- [[tests/2026-04-07_kwta_vs_threshold_pilot|2026-04-07 k-wta vs threshold pilot]] -- matched-sparsity bridge pilot showing stronger exact support recovery for k-wta at moderate noise and exact active-fraction control at higher noise
- [[tests/2026-04-08_leak_vs_carry_pilot|2026-04-08 leak vs carry pilot]] -- matched discrete-time bridge pilot showing that explicit leak improves gap retention but loses to atmn-style carry on anchor match and long-sequence drift
- [[tests/2026-04-09_bcm_alpha_pilot|2026-04-09 bcm-like adaptive alpha pilot]] -- gamma=0.3-0.5 significantly stabilizes kda state norm over long sequences (p=0.001) without degrading retrieval. bcm-like forgetting works as predicted.
- [[tests/2026-04-09_gp_vs_bilinear_pilot|2026-04-09 gp vs bilinear pilot]] -- pga provides no advantage over random bilinear or elementwise at random init. geometric structure benefit must come from trained weight interaction, not raw algebra.

### simulation results and analyses

- [[tests/head_dim_sweep_results|head dimension sweep results]] -- five-point asymmetric matrix-memory width sweep showing sub-linear $p^*(d)$ growth and identifying retention, not width alone, as the real bottleneck.
- [[tests/decay_sweep_results|decay sweep results]] -- nine-point retention sweep at `d_head=64` showing the first exact-query 32-pattern reopening at `decay=0.90` and the first 64-pattern reopening at `decay=0.95`.
- [[tests/overwrite_sweep_results|overwrite sweep results]] -- focused overwrite comparison at `decay=0.90` showing that erasure hurts every tested encoding at the first useful 32-pattern retention knee.
- [[tests/encoding_simulation_round_a|encoding simulation round a]] -- symmetric memory sign-only vs three-level encoding comparison
- [[tests/encoding_simulation_round_b|encoding simulation round b]] -- asymmetric matrix memory encoding comparison; capacity ceiling below symmetric
- [[tests/correction_field_trained_prediction_results|correction-field trained-prediction results]] -- trained-predictor correction-field sim; memory_capacity_delta=0 at every quality
- [[tests/multi_resolution_head_split_results|multi-resolution head split results]] -- fast/medium/slow heads with surprise gates; rare-class recall improves
- [[tests/thinking_loop_prototype_results|thinking-loop prototype results]] -- recurrent hidden-state refinement pilot on modular arithmetic
- [[tests/god_run_findings|god_run findings]] -- original long-form synthesis of god_run's results

### module prototypes

- [[tests/aesthetic_logger_prototype|aesthetic logger prototype]] -- phase 6a logging module; implemented, not yet wired into train_model