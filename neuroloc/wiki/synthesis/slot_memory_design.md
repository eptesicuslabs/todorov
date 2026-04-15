# slot memory design

## empirical status (2026-04-15, UPDATED after run 2)

the first paid test of this design (run2_slot_memory, 2026-04-15) returned val_bpb
1.5107 and passkey 0/100 at every tested length. **this run does NOT falsify the
slot-memory hypothesis**, because the preset inherited `alpha_log_mean=-0.5` from
`Config` defaults, reproducing the state-evaporation failure mode documented in
`wiki/synthesis/linear_attention_retrieval_wall.md` evidence line 4. the state
decayed to `10^-109` over 256 tokens — below float32 epsilon — so no retrieval was
architecturally possible regardless of substrate. the run measured the retention
bug, not the substrate.

the `delta_state_structure_probe` entry in run 2's eval artifact is
`{"error": "no DELTA layers had populated state", "per_layer": {}}`. the probe
only reads DELTA blocks, and the run 2 preset replaces every DELTA block with
SLOT, so the probe did not run. the `0.000` visible in the step log is the
logger's `.get(..., 0)` default, not a measurement. no structural state
evidence is available from run 2; the evaporation diagnosis rests on the init
math and the behaviour of the reused decay code, not on probe data.

fix committed as `7abb781` (alpha_log_mean=5.0, alpha_eff=sigmoid(5.0)=0.9933,
0.9933^256 ≈ 0.18). not relaunched at time of writing. the substrate's actual
retrieval capability remains untested at paid scale. see
`wiki/mistakes/run2_slot_memory_decay_copy_paste.md` — including the follow-up
audit showing all four paid runs inherited the same broken default, and the
`_assert_preset_retention_safe` structural guard added after the audit.



## the core change

replace the matrix memory `S ∈ R^{B×H×D×D}` with a **slot memory** `M ∈ R^{B×H×N×D}` where N is a fixed number of slots and D is the per-slot content dimension. reads use softmax addressing over slots. writes use a learned allocation policy that places content into the most appropriate slot.

this is a substrate change, not a parameter-count change. at N=64 and D=64, each head's memory is 4096 floats — identical to the matrix memory's 64×64 footprint. the mathematical behavior is fundamentally different.

## why this fixes the four failure modes

### failure 1: linear capacity ceiling (hopfield O(d))

**fix**: softmax addressing replaces linear inner-product readout. ramsauer et al. 2021 (iclr 2021) proved that for binary patterns, softmax (log-sum-exp energy) restores exponential capacity with the theoretical bound `2^(d/2)` — at d=64 this is roughly 10^9 patterns. this bound is for binary hopfield-style patterns; slot memory stores continuous learned projections, so the tight constant will differ and the exponential-scaling claim is qualitative, not a tight operating point. what remains true is that the transition from linear `O(d)` to exponential capacity requires softmax addressing as a structural change, not a tuning parameter. even a modest practical capacity of 30-100 slot patterns per head at d=64 would be a massive improvement over the current 9-pattern ceiling.

### failure 2: learned key correlation (cabannes)

**fix**: the softmax suppression factor `exp(k^T q)` grows exponentially with similarity, so the read strongly prefers one matching slot over many weakly-matching ones. non-orthogonal keys still interfere in raw dot product, but softmax temperature provides tunable separation. additional fix: an **allocation policy** at write time can explicitly route distinct content into distinct slots, so slot keys become orthogonal by construction rather than by hope.

### failure 3: state evaporation (decay^256 ≈ 10^-109)

**fix**: slots do not decay. they overwrite. a slot either holds content or is empty. the least-recently-used or least-relevant slot is chosen for the next write. there is no exponential decay term. a passkey written at position t=5 is still in its slot at position t=250 unless N other writes have filled every slot first.

### failure 4: delta contamination of attn residual stream

**fix**: add an **explicit output gate** on the slot-memory layer:

```
o = softmax(q @ M_keys^T) @ M_values
gated_output = sigmoid(learned_gate_init = -4.0) * o
```

the gate starts near zero (sigmoid(-4) ≈ 0.018). during training, the model learns to open the gate for heads and layers where the memory actually helps. any head that produces noise stays gated closed, contributing near-zero to the residual stream. this directly addresses the contamination problem identified in `linear_attention_retrieval_wall.md`.

## the write policy

the slot buffer's value depends entirely on which slot receives each token's write. three options, ranked:

### option A: surprise-gated lru write

1. compute surprise: `s_t = ||r_t||^2 / (||h_t||^2 + eps)` where `r_t = h_t - predict(h_{t-1})` is the prediction residual from the correction-field mechanism. the surprise-gate PRIMITIVE was validated in isolation by the correction-field capacity simulation (write fraction drops to 1.6% at 5%-predictable content with tau=0.99, see `correction_field_memory.md`). its combination with lru slot selection and softmax readout is untested end-to-end and is the subject of cpu simulation B.
2. if `s_t > tau`, the token is surprising and gets written. otherwise, skip.
3. the slot chosen is the least-recently-used slot, with recency tracked per slot as a scalar step counter incremented at each write to that slot (standard lru); a more sophisticated attention-weighted read-recency variant is an option but not the default.
4. content written: the learned projection `W_v @ h_t` (not the residual — slot stores content, the projection is the chosen abstraction).

this preserves mechanism 1 from the six-mechanism compound compression (predictive filtering) while adding native mechanism 4 (content-addressable indexing via the slot structure). mechanism 3 (schema-delta encoding) becomes natural: individual slots can be schemas, with deltas stored in other slots.

### option B: learned allocation score

replace the lru with a learned scoring network: for each incoming (k_t, v_t), compute an allocation score per slot `a_t = softmax(slot_key_embedding^T f(h_t))`. write into the slot with the highest score. this learns allocation end-to-end.

more expressive than A. more expensive. tested second if A works.

### option C: fixed hash allocation

hash the key to a slot index. simple, predictable. useful as a baseline for ablation. not expected to win on retrieval but provides a sanity check for the write pipeline.

## how it reads

query-based softmax addressing:

```
attn_scores = q_t @ M_keys^T     # shape (B, H, 1, N)
weights = softmax(attn_scores / sqrt(D))
o_t = weights @ M_values         # shape (B, H, 1, D)
```

this is structurally identical to the attn layer's softmax attention, but over N slots instead of T cached tokens. cost is O(N) per read regardless of sequence length.

## parameter count and memory footprint at 353M scale

per layer, per head:
- slot keys: N × D = 64 × 64 = 4096 floats
- slot values: N × D = 64 × 64 = 4096 floats  
- slot age counter: N = 64 floats (non-trainable, state)
- learned gate: 1 scalar per head

total per head: ~8K floats + negligible gate param.

across 24 such layers with 16 heads each: 24 × 16 × 8K = 3m trainable params for the slot banks (but these are STATE, not parameters — they're written and overwritten during forward pass, only the projection matrices are trainable).

new trainable parameters vs current matrix memory (per layer, with option A / lru write policy):
- remove: `alpha_log` (H=16 params), `beta_proj` which is `nn.Linear(d_model, num_heads)` = 1024*16 + 16 = 16,400 params
- add: output gate (H=16 params)
- add: slot projection from d_model to slot_dim if slot_dim != d_head (needs verification against the existing `k_proj`, `v_proj`; if reusable with reshape, zero new params; if not reusable, ~1024*16*64 = 1m extra per layer)
- net delta with reusable projections: -16,400 params per layer (saving)
- net delta without reusable projections: +~1m params per layer

with option B (learned allocation scoring net at ~d_model * num_slots = 1024 * 64 = 65k per layer per head-group, times 16 heads = ~1m per layer): net +~2m per layer.

the "essentially zero" claim requires option A AND reusable projections. option B explicitly adds real parameter cost. implementation must verify the projection reuse question before the parameter count can be finalized.

## integration with god_machine.py

replace `DeltaRuleMemory.forward` with a new `SlotMemory.forward` that:

1. maintains `self.slot_keys: (B, H, N, D)` and `self.slot_values: (B, H, N, D)` as state buffers (passed in, not module attributes, like the existing matrix memory's state)
2. computes q, k, v as currently (with rope on q)
3. reads: `o = softmax(q @ slot_keys.transpose(-2,-1)) @ slot_values` per head
4. applies learned output gate: `o_gated = sigmoid(gate) * o`
5. writes: if `surprise > tau`, update slot_keys[lru_slot] = k_t, slot_values[lru_slot] = v_t
6. returns `(o_gated, new_state, aux)` where aux includes slot usage stats for telemetry

the existing compressed attention layers (4 of them) stay unchanged. they already work as softmax attention with low-rank kv cache. the slot memory replaces only the delta layers.

## alignment with project vision

### memory (primary)
the slot buffer passes passkey by construction of its read mechanism (softmax addressing is the same primitive every successful passkey-passing architecture uses).

### compression (six-mechanism stack)
- mechanism 1 (predictive filtering): NOT native to the substrate. AVAILABLE via option A (surprise-gated writes). the surprise-gate primitive itself was validated in isolation by the correction-field capacity simulation; the slot-write integration is untested and is the subject of cpu simulation B.
- mechanism 2 (generative replacement): NOT native. the strict definition from `compression_beyond_quantization.md` requires storing a sparse pointer into a shared generative model, not storing content itself. slot memory stores learned projections as content, which is closer to mechanism 4. a loose analogy can be drawn ("the rest of the model reconstructs output from retrieved content"), but it is weaker than strict generative replacement.
- mechanism 3 (schema-delta encoding): native. each slot can be a schema; deltas can live in other slots.
- mechanism 4 (content-addressable indexing): native. the slot structure IS an index.
- mechanism 5 (manifold abstraction): not native; would layer as codebook compression over slot values later (tier 2 in the 5-tier proposal).
- mechanism 6 (reconsolidation): native — when a slot is overwritten, the content is lost, and future retrievals fall back on other slots or the model's prior.

slot memory natively implements mechanisms 3 and 4 of the compound compression stack. mechanisms 1 and 6 are available via policy choices (surprise gating for 1, overwrite semantics for 6). mechanisms 2 and 5 require additional layers (the correction-field design for 2, codebook compression for 5). the current matrix memory implements none natively. this is a substantive upgrade but does not subsume the full compound compression stack.

### thinking (iterative forward passes)
a thinking loop re-queries the slots with refined queries across think-steps. because the slot contents are stable across think-steps (no decay), iterative refinement converges instead of destabilizing. compatible with thinking by construction.

### imagination (generative probe)
the closed-gate structure probe we already have generalizes: probe the slots with novel queries, measure whether the outputs are structured (slot interpolation) or random (slots empty). imagination becomes slot recombination rather than matrix-state sampling.

### coherent output
orthogonal to the substrate change. the output head is unchanged.

## testable predictions

1. **capacity**: at N=64 slots, `d_head=64`, softmax temperature 1.0, cpu simulation should achieve >80% exact retrieval on 64 randomly-keyed slots under paired comparison (same tonnage as the asymmetric sim).
2. **no evaporation**: retrieval accuracy should be independent of the number of intervening non-writing tokens. at 256 intervening tokens with no write, retrieval should match retrieval at 0 intervening tokens.
3. **learned allocation beats lru**: option B should outperform option A on byte-level synthetic passkey tasks, but by less than the gap between option A and the raw matrix memory.
4. **gate learns to open**: during training, the average gate value across layers should move from ~0.018 toward a layer-dependent optimum; layers whose slots are useful should open more. heads whose slots stay uninformative should stay closed.
5. **passkey survives a 256-length filler**: in a small model (d_model=128, 6 layers, 1 attn layer, batch 8, 500 steps on synthetic corpus with inserted passkeys), passkey@256 should reach >20% after training, vs 0% for the current matrix memory. this is the minimum viable sign of life before a full-scale paid run.

## cpu validation plan

### simulation A: slot buffer capacity (pure standalone)

`neuroloc/simulations/memory/slot_buffer_capacity.py`. 

tests: N slots, N random key-value pairs written, query with exact key + noise, measure retrieval accuracy. sweeps N in {16, 32, 64, 128}, d_head in {32, 64, 128}, query noise in {0, 0.1, 0.2}, temperature in {0.1, 1.0, 10.0}. paired comparison vs matrix memory baseline.

pass criterion: at N=64, d=64, temperature=1.0, noise=0.0, exact retrieval > 0.95.

### simulation B: slot buffer with surprise-gated writes

`neuroloc/simulations/memory/slot_buffer_surprise_writes.py`.

tests: a stream of 512 mixed predictable/surprising tokens. only surprising tokens write. measure whether after the stream, querying with a surprising token's key returns that token's value even through 256+ tokens of intervening predictable tokens.

pass criterion: at 10% surprising fraction, passkey-equivalent retrieval > 0.80 at 256-token intervening distance.

### simulation C: integration smoke

`neuroloc/simulations/memory/slot_vs_matrix_integration.py`.

tiny model, 6 delta layers + 1 attn, d_model=128. train for 500 cpu steps on synthetic passkey-containing byte data. measure passkey retrieval on held-out passkeys. compare matrix-memory baseline vs slot-memory swap. expected: matrix 0%, slot > 15%.

pass criterion: slot memory meaningfully exceeds matrix memory on the task. specific threshold TBD after sim A results inform the expected upper bound.

all three sims run on cpu, total compute < 2 hours.

## what could still go wrong

1. **softmax temperature tuning**: if temperature is too low, retrievals get stuck on one slot; too high, retrievals are uniform. training should learn a good operating point but may plateau.
2. **slot thrashing**: if every token writes, the 64 slots cycle quickly and nothing persists. surprise gating must filter aggressively enough. sim B validates this before paid compute.
3. **gradient flow through discrete allocation**: the lru allocation is non-differentiable. learned allocation (option B) is differentiable via softmax. option A works in simulation because retrieval only depends on WHICH slot holds the content, not HOW it got there. but at training scale, the model may not learn to route through the slots without end-to-end gradient through the allocation.
4. **the byte-level problem**: no published pure-recurrent architecture has passed passkey on byte level. even softmax addressing might need specific byte-level tuning.
5. **the dead-weight possibility**: cpu experiment A (currently running) might show that the delta layers contribute near-zero to val_bpb in their contaminated form. if the matrix memory is functionally dead weight already, replacing it with slot memory will be a clear win. if the matrix memory is doing real work (which seems unlikely but possible), the replacement needs to preserve whatever it was doing.

## implementation sequencing

1. write cpu simulation A (standalone slot buffer capacity) — ~2 hours
2. run it, if pass criterion met: proceed. if not: reassess (maybe softmax+indexing isn't enough, need titans-style nonlinear memory)
3. write cpu simulation B (surprise-gated writes at 256-length) — ~3 hours  
4. run it, if pass: proceed. if not: simplify write policy, retry
5. write cpu simulation C (integration smoke on tiny byte-level model) — ~4 hours including training
6. if all three sims pass: implement `SlotMemory` class in `god_machine.py`, add new preset `run2_slot_memory`, smoke-test
7. if smoke passes: run benchmark on paid compute, verify throughput + memory fit
8. launch full run

no paid compute until steps 2-5 all pass. sunk cost (three failed runs) teaches us to validate the substrate before spending again.

## see also

- `wiki/synthesis/linear_attention_retrieval_wall.md` (why the current substrate fails — sets up this design)
- `wiki/synthesis/correction_field_memory.md` (surprise-gated writes — the write policy A in this design uses the same predictive filtering mechanism)
- `wiki/synthesis/compression_beyond_quantization.md` (six-mechanism compound compression — slot buffer natively implements 1-4)
- `wiki/bridge/memory_compression_to_tiered_architecture.md` (5-tier architecture — slot buffer is the tier 1-2 substrate)
- `wiki/tests/encoding_simulation_round_b.md` (confirms the matrix memory's d_head=64 ceiling that slot softmax addressing escapes)
- `wiki/tests/multi_resolution_head_split_results.md` (different decay rates across heads also help long-horizon retrieval; could combine with slot memory for fast/slow heads)
