# compression architecture for the neural machine

status: current (as of 2026-04-16).

## the problem

a neural machine that computes through recurrent state, competitive selection, and associative memory needs to be extremely memory efficient. biology achieves this through multiple compression stages. the current todorov architecture has:
- ternary spikes: 1.58 bits/dim (20x compression of activations)
- mla kv compression: d_model -> d_c = 128 (8x compression of cache)
- kda recurrent state: fixed O(1) per token (already compressed by design)

this is not enough for a neural computer. we need novel compression that goes beyond quantization.

## biological compression principles

### hippocampal indexing (teyler & rudy 2007)

the hippocampus stores POINTERS to cortical patterns, not the patterns themselves. retrieval reactivates the distributed cortical representation from the compressed index. the index is ~100x smaller than the pattern it references.

machine analog: instead of storing full d_model vectors in mla cache, store a learned hash (32-64 bits per token) that addresses into a shared codebook. retrieval reconstructs the full vector from the hash. this converts O(T * d_c) cache to O(T * hash_bits + codebook_size).

### pattern separation via extreme sparsity (dentate gyrus)

the dentate gyrus encodes at 2-5% sparsity, creating near-orthogonal representations that minimize interference. at 5% sparsity with ternary values, each vector needs only ~0.15 bits/dim (entropy coding).

machine analog: k-wta at k=5% of dimensions + ternary values = 0.15 bits/dim. vs current 1.58 bits/dim at 41% firing rate. this is 10x additional compression on top of ternary quantization. total: 200x compression of activations vs fp32.

### chunking (chase & simon 1973, cowan 2001)

experts compress information into chunks stored in long-term memory. working memory holds 4 chunks, but each chunk can contain arbitrary complexity. the compression ratio is unbounded -- it depends on how much structure has been learned.

machine analog: detect repeated subsequences in the recurrent state and replace them with single tokens that reference a learned chunk dictionary. this is lossless compression applied to the temporal dimension. the chunk dictionary grows during training.

### predictive coding (rao & ballard 1999)

the brain transmits only prediction errors between layers, not full activations. if the prediction is good, the error is sparse and compressible. the better the model, the sparser the communication.

machine analog: instead of passing the full residual stream between layers, pass x - predicted_x. if the model predicts well, this residual is small and sparse. apply ternary quantization to the RESIDUAL, not the activation. the compression ratio improves as the model improves.

### consolidation (mcclelland, mcnaughton & o'reilly 1995)

the hippocampus stores raw episodes. during sleep, these are compressed into statistical summaries and transferred to neocortex. the neocortex stores gist, not detail.

machine analog: every N tokens, compress the kda recurrent state S_t into a low-rank summary (svd truncation to rank r). store the summary in a consolidated memory bank alongside mla cache. old kda state can then be discarded. this converts O(heads * head_dim^2) state to O(heads * head_dim * r) with r << head_dim.

## novel compression proposals

### 1. hierarchical ternary coding

current: all dimensions quantized to {-1, 0, +1} uniformly.
proposed: hierarchical quantization. first pass: coarse selection (k-wta picks top 10% dimensions). second pass: those 10% get ternary values. remaining 90% are zero.

encoding: 10% selection mask (entropy coded, ~0.47 bits/dim) + ternary values for selected dims (1.58 bits * 0.1 = 0.16 bits/dim). total: 0.63 bits/dim. vs current 1.58 bits/dim. 2.5x additional compression.

at 5% selection: 0.29 bits/dim + 0.08 bits/dim = 0.37 bits/dim. 4.3x additional compression.

### 2. state-predictive residual coding

instead of transmitting full layer outputs through the residual stream, transmit the INNOVATION -- what this layer adds that couldn't be predicted from the previous layer's output.

implementation: each layer predicts the next layer's output using a lightweight projection (d_model -> d_model, tied weights). the actual residual stream carries x + layer_output - predicted_next. if the prediction is good, the innovation is sparse.

this is predictive coding applied to inter-layer communication. the compression ratio improves as the network learns to predict its own internal dynamics.

### 3. consolidated state snapshots

every 64 tokens, perform rank-r svd on each kda head's state matrix S_t. store the top-r singular vectors as a "memory snapshot." reset S_t to the rank-r approximation. append the snapshot to a consolidated memory bank that mla can attend over.

this creates a two-tier memory: fast kda (recent, full-rank, decaying) + consolidated snapshots (older, rank-r, permanent). the total memory grows as O(T/64 * r * head_dim) instead of O(T * d_c) for full mla cache. at r=4, head_dim=64, this is 256 floats per snapshot per head vs 128 floats per token for mla cache. the snapshot covers 64 tokens worth of state, so it's 64x more compact per token of coverage.

### 4. content-addressable sparse memory

replace the mla dense cache with a sparse content-addressable memory. instead of storing one latent per token, store one entry per UNIQUE concept encountered. duplicate or similar tokens share entries via learned hashing.

implementation: hash each token's kv representation to a 64-bit key. if the key already exists in memory, update the existing entry (running average). if new, allocate a new slot. retrieval via approximate nearest neighbor on the hash.

this bounds memory by the number of unique concepts, not the number of tokens. for a document with 10K tokens but 500 unique concepts, this is 20x compression.

### 5. ternary weight matrices

extend ternary quantization from activations to weights. a fully ternary network ({-1, 0, +1} weights and activations) replaces all multiplications with additions and sign flips. each weight needs ~1.58 bits instead of 16 (fp16) or 32 (fp32). compression: 10-20x on model size.

the combination: ternary weights (1.58 bits/param) + k-wta ternary activations (0.37 bits/dim) + consolidated state snapshots + content-addressable sparse memory. this is a neural machine that fits in a fraction of the memory of a standard transformer at matched parameter count.

## see also

- [[sparse_coding_to_ternary_spikes]]
- [[energy_efficiency_to_ternary_spikes]]
- [[memory_systems_to_kda_mla]]
- [[compression_and_bottlenecks]]
