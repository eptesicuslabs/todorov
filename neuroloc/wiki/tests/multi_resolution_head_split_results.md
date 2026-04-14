# multi-resolution head split results

last updated: 2026-04-14

## what this is

this note records the cpu simulation `neuroloc/simulations/memory/multi_resolution_head_split.py`. the simulation tests the head-split hypothesis from `wiki/synthesis/correction_field_memory.md`: splitting the 16 heads per layer into fast, medium, and slow groups with different decays and surprise-gated writes gives higher effective capacity on rare long-range recall than a uniform 16-head configuration at matched total memory.

the design tested is identical to the article:

- **uniform**: 16 heads, all with decay=0.9, write gate=1.0 (every token writes).
- **split**: 8 fast heads (decay=0.5, write always), 6 medium heads (decay=0.95, writes only when surprise > 0.1), 2 slow heads (decay=0.999, writes only when surprise > 0.5).

total memory (16 heads * d_head * d_head parameters) is matched across configurations. the comparison is paired: both configurations see the SAME stream of random key-value associations per trial.

## run configuration

- stream lengths: {32, 64, 128, 256, 512}
- head dimensions: {32, 64, 128}
- trials per cell: 16
- queries per class: 8
- stream composition: 60% immediate-class (surprise ~ U[0.00, 0.09]), 30% recent-class (surprise ~ U[0.15, 0.40]), 10% rare-class (surprise ~ U[0.60, 0.99])
- recall windows: immediate queries taken from the last 4 tokens, recent from the last 32 tokens, rare from the last 200 tokens
- 16 trials per (stream_length, head_dim) cell, 5 * 3 * 16 = 240 paired trials total

artifacts at `neuroloc/output/simulation_runs/memory/multi_resolution_head_split/` (metrics json + png). wall-clock about 21 s on cpu.

## headline results

mean cosine recall at d_head=64, split minus uniform, by stream length:

- **immediate-class (last 4 tokens)**: -0.20 to -0.30 (split LOSES 20-30 cosine points).
- **recent-class (last 32 tokens)**: +0.11 to +0.20 (split gains 10-20 cosine points).
- **rare-class (last 200 tokens)**: +0.37 at L=32, +0.51 at L=64, +0.57 at L=128, +0.56 at L=256, +0.56 at L=512. the gap WIDENS as the stream grows because the uniform config forgets rare-class tokens at decay=0.9 while the split's slow heads retain them at decay=0.999.

paired permutation test on the rare class at d_head=64 rejects the null of equal means with p_permutation = 0.001 at every stream length. effect sizes (dz) above 2 across the board. the rare-class result is not a noise artifact; it is the cleanest effect in the sweep.

absolute recall at d_head=64 for the rare class:

| stream length | uniform mean cosine | split mean cosine |
|---|---:|---:|
| 32 | 0.536 | 0.903 |
| 64 | 0.313 | 0.826 |
| 128 | 0.155 | 0.723 |
| 256 | 0.105 | 0.661 |
| 512 | 0.089 | 0.646 |

uniform rare-class recall is near chance (0.089 at L=512) because decay=0.9 for 512 steps is 0.9^512 ~ 10^-24 attenuation. split retains 0.646 at L=512 because the 2 slow heads with decay=0.999 attenuate only by 0.999^512 ~ 0.60.

## crossover pattern count

at every tested (d_head, class) cell, split beats uniform on recent and rare classes at the smallest tested stream length (L=32) and at every larger length (5/5 lengths beat). the crossover "stream length above which split beats uniform" is therefore at or below L=32 for both recent and rare classes, and does NOT exist for the immediate class (uniform wins at every length, 0/5).

## does the split hurt immediate recall

yes. split underperforms uniform on immediate-class queries by 0.17 to 0.31 cosine at every (stream_length, d_head) cell. the reason is mechanical:

- uniform's 16 heads all decay at 0.9. for tokens within the last 4 steps, attenuation is 0.9^4 = 0.66, which is still retrievable.
- split's fast heads (8 of 16) decay at 0.5. for a 4-step-old token, fast-head attenuation is 0.5^4 = 0.0625 -- nearly gone. split's medium and slow heads skip most immediate-class writes (surprise 0.00-0.09 is below both thresholds tau=0.1 and tau=0.5), so only the fast heads carry any immediate content, and they forget it fast.
- averaging across all 16 heads dilutes the remaining fast-head signal by 16x, while uniform pools 16 retrievable heads.

this is NOT a defect of the design; it is a direct consequence of the temporal specialization. split allocates 8 of 16 heads to immediate recall with aggressive decay, while uniform allocates all 16 to a middle-ground decay. the split is less effective at immediate recall by design, and the tradeoff is what enables the massive rare-class gain.

in production this may be partially offset by (a) tuning fast decay upward toward uniform levels if immediate recall is the bottleneck, or (b) relying on the compressed-attention layers (MLA mode) for immediate recall while the matrix-memory heads specialize in longer horizons. the current architecture already has dedicated compressed-attention layers for recent-context exact retrieval.

## recommendation

adopt the split in a LATER run, not run 2.

the run-2 candidate is the trained-prediction correction-field mechanism (pending). the head split is a parameter-free reorganization of existing heads -- it adds no parameters -- but it cannot be validated until the base matrix memory is retrievable at scale. run 1 (pending) is the prerequisite: if the baseline model cannot retrieve rare-class content at long context, the split moving decay to 0.999 cannot rescue what was never written correctly in the first place.

the sequencing from `blueprint.md` and `spec/implementation_plan.md` is already:

1. run 1: baseline dense matrix memory at 350m scale (pending).
2. run 2: value-side compression via trained-prediction correction field (pending).
3. **run 3: multi-resolution head split (this sim's candidate) + correction-field residuals on slow heads.**
4. run 4: codebook compression of corrections stored in slow heads.

run 3 is the right slot. the sim gives three pieces of evidence for run 3:

1. rare-class capacity gain +0.37 to +0.65 cosine, paired permutation p=0.001 at every length.
2. recent-class gain +0.09 to +0.20, also significant but smaller.
3. immediate-class cost -0.17 to -0.31; manageable if compressed-attention layers handle immediate retrieval.

the strongest evidence is that the gap WIDENS with length (uniform rare-class recall collapses while split holds at ~0.65 out to L=512). this is the exact failure mode god_run_v1 exhibited: passkey 0/20 at 4096 tokens because the delta state was noise, not content-addressable memory. a split that retains rare content at 0.999 decay for the 2 slow heads may be the structural fix, not a stylistic tweak.

caveats for interpretation:

- the sim uses synthetic uniform-random keys and values; trained activations will have correlated structure that changes the interference pattern. the relative ordering is likely robust, but the absolute numbers are not directly transferable.
- surprise values are synthetic uniform draws within class-specific ranges, not a learned surprise estimator. a trained surprise head would produce a different distribution.
- queries are evaluated only on tokens within each class's recall window; tokens tagged with a class but outside the window are excluded from that class's recall score.
- the sim uses mean-over-heads readout; the architecture uses concatenated per-head output projected by W_o. averaging is the scalar equivalent used here because all heads share the same output dimension in this cpu test; concat+project would change absolute numbers but not the relative ranking.

## artifacts

- `neuroloc/simulations/memory/multi_resolution_head_split.py` (the sim)
- `tests/test_multi_resolution_head_split.py` (5 unit tests, all pass)
- `neuroloc/output/simulation_runs/memory/multi_resolution_head_split/multi_resolution_head_split_metrics.json`
- `neuroloc/output/simulation_runs/memory/multi_resolution_head_split/multi_resolution_head_split.png`
- registered in `neuroloc/simulations/suite_registry.py` under category "compression".

## see also

- `wiki/synthesis/correction_field_memory.md` (the head-split design, section "the multi-resolution head split")
- `wiki/tests/decay_sweep_results.md` (decay frontier for the raw matrix memory)
- `spec/blueprint.md` (run sequencing)
- `spec/implementation_plan.md` (5-run protocol)
