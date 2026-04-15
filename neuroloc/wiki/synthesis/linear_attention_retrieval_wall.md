# linear attention retrieval wall

## the three-failure diagnosis (2026-04-14)

three consecutive paid h200 runs produced zero passkey at 256 tokens despite very different feature configurations: `god_run` 0/20, `god_run_v2` 0/100, `run1_baseline_noerasure` 0/100 (the g6 prosecutor finding bumped trial count from 20 to 100 between v1 and v2; v1 was evaluated at the looser 20-trial bar). the first two ran the full five-feature bundle (rate-coded value compression, delta erasure, activity-adaptive decay, multi-compartment feedforward, compressed attention, imagination probe, predictive-coding diagnostic). the third ran with every feature off: dense k/v, no erasure, no adaptive decay, no compartments, no imagination, no pc head. all three achieved similar val_bpb (1.3950, 1.4453, 1.4499). retrieval failed at the same rate each time.

this article synthesizes why the baseline architecture cannot retrieve, based on five independent lines of evidence that converged on the same conclusion.

## evidence line 1: classical capacity ceiling

the matrix memory update is:

```
S_t = a * S_{t-1} + b_t * k_t * v_t^T
o_t = q_t^T * S_t
```

at head dimension 64, the classical outer-product associative memory has a capacity ceiling of:

- error-tolerant retrieval: `C ≈ 0.14 * d = 9 patterns` (amit, gutfreund, sompolinsky 1985)
- error-free retrieval (asymptotic): `C ≈ d / (2 * ln d) ≈ 8 patterns` at d=64 (standard natural-log formulation from the statistical-mechanics literature; hopfield 1982's original log_2 scaling gives `64 / (2 * 6) ≈ 5`, a constant-factor difference within the same asymptotic class)

ramsauer et al. 2021 (arxiv 2008.02217) proved that the transition from linear `O(d)` capacity to exponential `2^(d/2)` capacity requires replacing the linear inner-product readout with a softmax (log-sum-exp energy). this is a qualitative structural change, not a quantitative tuning. the matrix memory's linear readout `q^T S` stays in the `O(d)` regime by mathematical construction.

schlag, irie, schmidhuber 2021 (arxiv 2102.11174) showed that the delta rule (which adds an erase-before-write step) improves the practical interference floor but does not move the theoretical ceiling. the ceiling is still `d_k = 64`; the delta rule just fills the available capacity more efficiently.

## evidence line 2: learned keys are not orthogonal

cabannes, simsek, bietti icml 2024 (arxiv 2402.18724) studied gradient descent on cross-entropy over an outer-product associative memory and proved three relevant results:

- when embeddings are non-orthogonal, training leads to oscillatory transitory regimes and memory interferences
- in the underparameterized regime (`d < N`), competition between memories leads to sub-optimal memorization
- catastrophic forgetting occurs because the dynamics is first dominated by frequent tokens until rare classes come into play

the implication for this project: next-byte pretraining does not produce orthogonal keys. it produces frequency-weighted, correlated keys. the interference term between two stored keys `i` and `j` is proportional to `k_i^T k_j`, which under pretraining is statistically positive, not zero. even the classical 9-pattern bound is an optimistic upper bound that assumes orthogonality; with correlated keys the effective capacity is lower.

## evidence line 3: fixed-state copying impossibility

jelassi, brandfonbrener, kakade, malach 2024 (arxiv 2402.01032) proved that a two-layer transformer can copy strings of exponential length while generalized state-space models are fundamentally limited by their fixed-size latent state. passkey at 256 tokens is literally a copying task: copy a 5-byte numeric string from ~250 bytes of intervening filler. the result applies exactly to the matrix memory layer, which has a fixed `(B, H, D, D)` state shape.

yang et al. colm 2025 (arxiv 2410.07145) "stuffed mamba" quantified state capacity with the fitted relationship `T_recall = 4.756 * (1.365^NS - 1) - 0.742`, meaning near-perfect passkey at length `T` requires state size `NS` proportional to `log(T)`. at head dim 64, this implies near-perfect passkey at roughly 4K tokens for mamba-2. the formula was fitted to mamba-2; the delta rule has a different write rule and the constants may shift, but the functional form (exponential in state size) is general.

## evidence line 4: physical state evaporation

the `god_machine.py` initialization sets `alpha_log_mean = -0.5`. the effective retention factor is:

```
alpha_eff = exp(logsigmoid(-0.5)) = sigmoid(-0.5) = 1 / (1 + exp(0.5)) = 0.3775
```

(note: `exp(logsigmoid(x)) = sigmoid(x)` by definition of logsigmoid; the two-step evaluation is shown to illustrate the training-code path, which computes `log_alpha = logsigmoid(alpha_log_base)` and then takes `exp(log_alpha)`.)

compounding over 256 tokens:

```
0.377^256 ≈ 10^-109
```

this is below float32 machine epsilon (`~1.2e-7`). the state of a matrix memory layer with standard retention init is **literally rounded to zero** by the time a 256-token passkey needs to be read. no amount of key orthogonality or write-rule correction changes this: the state is gone.

this explains why the decay sweep simulation (`wiki/tests/decay_sweep_results.md`) found that exact-query recall of 32 patterns only reopens at `decay ≥ 0.90`. at `decay = 0.377`, the state has evaporated well before the query arrives.

## evidence line 5: delta contaminates attention

the architecture has 24 matrix-memory layers (delta) interleaved with 4 compressed-attention layers (attn), ratio 6:1. the residual stream math at each layer is unconditional:

```
residual = x
x = residual + mixer(norm(x))
```

there is no gate between `mixer_out` and the residual stream. if a delta layer outputs noise (as the structure probe confirms: `mean_structure_ratio = 0.977`, `pairwise_cos = 0.003`, statistically indistinguishable from random in god_run_v2), that noise is added unconditionally to the residual and propagates to every downstream layer.

the attn layers, which are in principle capable of retrieval via softmax-addressed cached keys, receive a residual stream that has passed through 6 delta layers before each attn layer. if the delta layers contaminate the residual with noise, the attn layer sees corrupted queries and corrupted cached keys. softmax attention over a corrupted cache cannot find the passkey even when the underlying attention mechanism is mathematically capable.

the god_run_v2 run card notes a +0.050 bpb regression from v1 to v2 attributed specifically to larger `alpha_eff` in v2 injecting more noise into the residual stream. this is direct evidence that the delta layers' output magnitude correlates with bpb degradation under this architecture.

## the fivefold stack

the architecture fails at retrieval because all five mechanisms compound:

1. **capacity ceiling** at head dim 64: `O(d)` linear retrieval caps at ~9 patterns (evidence line 1)
2. **learned correlation**: pretraining produces non-orthogonal keys, effective capacity is lower than the classical ceiling (evidence line 2)
3. **fixed-state copying impossibility**: the jelassi et al. 2024 theorem plus the stuffed-mamba exponential state-size relationship predict retrieval failure at 256 tokens with 64-dim state (evidence line 3)
4. **state evaporation**: default retention gives `0.377^256 ≈ 10^-109`, state is rounded to zero before retrieval (evidence line 4)
5. **contamination**: delta outputs unconditionally added to residual, corrupting the inputs to attn layers that could otherwise retrieve (evidence line 5)

any single fix addresses at most one or two of these. a slower static retention (`run1a`) addresses only #4. an orthogonalized write rule (larimar-style rls) addresses #2. a wider head dim addresses only #1 and only sub-linearly. softmax addressing (ramsauer 2021) addresses #1 and #2 together but requires a substrate change. none of these alone are sufficient.

## the literature has no counter-example

no peer-reviewed published architecture combines (a) pure linear-recurrence memory, (b) ~300m parameters, (c) byte-level vocabulary, (d) passkey success at 256+ tokens. every successful passkey result at scale uses at least one of:

- delta-rule error correction (deltanet/gateddeltanet) at token level, not byte level, and still fails on s-niah-2 standalone at 8k
- softmax attention layers interleaved with linear layers (samba, jamba)
- external kv cache with knn retrieval (memorizing transformers)
- pseudo-inverse slot memory (larimar)
- test-time gradient updates to an mlp memory (titans)

arora et al. icml 2024 "based" (arxiv 2402.18668) theorem 3.1 formalized a related lower bound: any recurrent model causally depending on input requires `Ω(N)` bits of state to solve multi-query associative recall (MQAR). passkey is single-query rather than multi-query, so the theorem does not formally apply, but the state-size dependency is analogous: byte-level passkey at 256 requires at minimum enough state bits to disambiguate the target token among 256 positions. nominally the matrix memory has `64 * 64 * 16 * 16 ≈ 1m` float bits per head, far above any reasonable lower bound. the theorem shows budget is not the bottleneck; in practice the state is occupied by pretraining statistics before retrieval is queried.

## what this means for the next run

continuing the current substrate is a rigorous dead end. the project moves from tuning the current substrate to replacing it.

the external-opinion synthesis (2026-04-14, pasted into the project conversation) ranked three alternatives: orthogonalized precision-matrix write (larimar-inspired recursive-least-squares), discrete slot buffer with exponential addressing, test-time fast-weight mlp (titans-inspired). the project chose the slot-buffer direction because it solves the failure modes catalogued in this article cleanly and has the lowest cpu-simulation validation cost. design details are in `wiki/synthesis/slot_memory_design.md`.

## see also

- `wiki/synthesis/correction_field_memory.md` (prior substrate proposal, reconstruction-side only; memory-capacity claim falsified)
- `wiki/synthesis/compression_beyond_quantization.md` (compound compression thesis — slot buffer natively implements mechanisms 3 and 4)
- `wiki/synthesis/slot_memory_design.md` (the proposed replacement substrate)
- `wiki/tests/decay_sweep_results.md` (empirical retention thresholds)
- `wiki/tests/head_dim_sweep_results.md` (sub-linear scaling of head-dim widening)
- `wiki/tests/overwrite_sweep_results.md` (erasure hurts at the retention knee)
- `wiki/tests/correction_field_trained_prediction_results.md` (memory substrate is invariant to stored content)
- `wiki/comparisons/memory_kda_vs_hippocampus.md` (prior analysis of the linear-readout weakness)

## references cited

- hopfield 1982; amit, gutfreund, sompolinsky 1985. classical associative memory capacity.
- ramsauer et al. 2021, "hopfield networks is all you need," iclr 2021, arxiv 2008.02217.
- schlag, irie, schmidhuber 2021, "linear transformers are secretly fast weight programmers," icml 2021, arxiv 2102.11174.
- cabannes, simsek, bietti 2024, "learning associative memories with gradient descent," icml 2024, arxiv 2402.18724.
- jelassi, brandfonbrener, kakade, malach 2024, "repeat after me: transformers are better than state space models at copying," arxiv 2402.01032.
- yang et al. 2025, "stuffed mamba: state collapse and state capacity of rnn-based long-context modeling," colm 2025, arxiv 2410.07145.
- arora et al. 2024, "simple linear attention language models balance the recall-throughput tradeoff" (based), icml 2024, arxiv 2402.18668.
- yang et al. 2024, "parallelizing linear transformers with the delta rule over sequence length" (deltanet), neurips 2024, arxiv 2406.06484.
- yang, kautz, hatamizadeh 2025, "gated delta networks: improving mamba2 with delta rule," iclr 2025, arxiv 2412.06464.
- behrouz, zhong, mirrokni 2025, "titans: learning to memorize at test time," arxiv 2501.00663.
- das et al. 2024, "larimar: large language models with episodic memory control," icml 2024, arxiv 2403.11901.
- liu et al. 2024, "samba: simple hybrid state space models," iclr 2025, arxiv 2406.07522.
