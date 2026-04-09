# tests

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

- [[tests/2026-04-07_pattern_completion_baseline|2026-04-07 pattern completion baseline]] -- ca3-like attractor baseline with shuffled-weight control, corruption/load/scaling sweeps, and machine-readable metrics
- [[tests/2026-04-07_kwta_vs_threshold_pilot|2026-04-07 k-wta vs threshold pilot]] -- matched-sparsity bridge pilot showing stronger exact support recovery for k-wta at moderate noise and exact active-fraction control at higher noise
- [[tests/2026-04-08_leak_vs_carry_pilot|2026-04-08 leak vs carry pilot]] -- matched discrete-time bridge pilot showing that explicit leak improves gap retention but loses to atmn-style carry on anchor match and long-sequence drift
- [[tests/2026-04-09_bcm_alpha_pilot|2026-04-09 bcm-like adaptive alpha pilot]] -- gamma=0.3-0.5 significantly stabilizes kda state norm over long sequences (p=0.001) without degrading retrieval. bcm-like forgetting works as predicted.
- [[tests/2026-04-09_gp_vs_bilinear_pilot|2026-04-09 gp vs bilinear pilot]] -- pga provides no advantage over random bilinear or elementwise at random init. geometric structure benefit must come from trained weight interaction, not raw algebra.