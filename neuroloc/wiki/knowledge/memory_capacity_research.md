# memory capacity research

status: current (as of 2026-04-16).

curated peer-reviewed research on the storage capacity of associative memories, recurrent states, and compressed representations. the central question for the neural machine: how many associations can the kda delta-rule state store, and how do ternary spikes, decay, and outer-product updates interact to determine this capacity? no existing paper answers this joint question, but the individual components are well-characterized.

## modern hopfield networks

### exponential capacity via softmax

ramsauer, h., schafl, b., lehner, j., seidl, p., widrich, m., adler, t., gruber, L., holzleitner, m., pavlovic, m., sandve, g. k., bock, c., hochreiter, s., & klambauer, g. (2021). hopfield networks is all you need. *international conference on learning representations (ICLR)*.

key finding: modern hopfield networks replace the classical quadratic energy function with an exponential (log-sum-exp) energy, yielding storage capacity that scales as 2^(d/2) where d is the pattern dimension. this is exponentially larger than the classical O(d) capacity (hopfield 1982). the update rule of the modern hopfield network is mathematically equivalent to the attention mechanism in transformers: the softmax attention weights are the fixed-point iteration of the modern hopfield energy. this establishes a formal bridge between associative memory and attention.

relevance to neural machine: todorov's mla layers implement attention and therefore implicitly operate as modern hopfield networks with exponential capacity. the kda layers, however, use a different mechanism: delta-rule updates to a matrix-valued state. the kda state is a linear attention variant, which corresponds to the classical (not modern) hopfield model with O(d) capacity per chunk. the architectural choice to hybridize kda (linear capacity, fast) with mla (exponential capacity, slow) at a 3:1 ratio is a capacity-throughput tradeoff.

confidence: high. the mathematical equivalence between modern hopfield and softmax attention is exact. caveat: the exponential capacity is for pattern retrieval at zero noise; in practice, noise and pattern correlations reduce effective capacity.

### tight upper bounds via spherical codes

hu, j. et al. (2024). tight bounds for modern hopfield network capacity. *neurips*.

key finding: the exponential capacity 2^(d/2) is not just a lower bound but matches the tight upper bound for random patterns on the hypersphere. the proof uses connections to spherical codes and packing theory. the bound is tight in the sense that no energy function of the softmax family can store more than 2^(d/2) patterns while maintaining metastable retrieval. the result also shows that capacity depends on the separation margin between stored patterns: correlated patterns reduce effective capacity exponentially.

relevance to neural machine: the tight bound establishes that mla's capacity ceiling is fundamental, not an artifact of the specific implementation. for kda, the relevant question is different: the delta-rule state is not a hopfield energy minimum but a least-squares regression state, and its capacity is governed by the rank of the state matrix (d_k x d_v) rather than exponential pattern separation. the tight bound also highlights that correlated patterns (which are common in natural language) reduce capacity -- todorov's ternary spikes, by forcing patterns to be sparse and discrete, may improve effective capacity by increasing inter-pattern separation.

confidence: high. mathematical proof with clear assumptions. caveat: the bound assumes random patterns; natural language patterns have structure that may increase or decrease effective capacity.

## delta-rule linear attention

### gated deltanet: perfect associative recall

yang, s. et al. (2024). gated delta networks: parallelizable linear recurrence with delta rule. extended from schlag et al. (2021).

key finding: gated deltanet implements the delta rule as a linear attention mechanism: the state matrix is updated by writing key-value associations while erasing old values for the same key. this error-correcting write mechanism gives deltanet perfect performance on multi-query associative recall (mqar) at 1.3B parameters, where standard linear attention and mamba fail. the gating mechanism (channel-wise forget gates) controls the decay of old associations, preventing catastrophic interference. the delta rule converts the state from a simple accumulator (as in linear attention) to a least-squares solver.

relevance to neural machine: todorov's kda layers implement exactly this mechanism. the perfect mqar result confirms that the delta-rule state can serve as a reliable associative memory at scale. the critical parameter is the gate decay rate: too fast and old memories are lost; too slow and new associations interfere with old ones. todorov's channel-wise gating (from kimi's kda) provides per-dimension control over this tradeoff. the 1.3B scale demonstration is the strongest evidence that delta-rule recurrence is competitive with attention for memory tasks.

confidence: high. replicated across multiple variants (deltanet, gated deltanet, kda). caveat: mqar is a synthetic task; performance on natural language memory tasks (e.g., long-range fact retrieval) may differ.

## ternary synaptic capacity

### ternary synapses preserve scaling

bhattacharjee, s. & martin, i. (2025). accuracy and capacity of modern hopfield networks with synaptic noise. *physical review e*, 112. arXiv:2503.00241.

key finding: synaptic noise (including ternary quantization) preserves the N^(n-1) scaling exponent for n-th order hopfield networks. the prefactor is reduced but the exponent survives. for the classical pairwise case (n=2), the baseline capacity is 0.138N (McEliece et al. 1987 for binary {-1,+1} patterns with continuous weights). ternary synapses reduce the prefactor relative to continuous weights but do not change the scaling class. the zero state in ternary provides an additional degree of freedom compared to binary, allowing the network to suppress weak connections.

relevance to neural machine: ternary quantization does not fundamentally limit how many patterns the kda state can store -- only the fidelity per pattern. the N^(n-1) scaling is preserved under quantization noise. however, the exact prefactor for ternary synapses with ternary patterns (the joint condition in the crbr architecture) has not been measured empirically.

confidence: medium. the paper studies ising (binary) patterns with noisy synapses, not the joint ternary-ternary case. the exact prefactor for {-1,0,+1} synapses AND {-1,0,+1} patterns is extrapolated, not directly computed.

## ssm retrieval and capacity

### ssm retrieval horizon scales with state size

arora, s. et al. (2025). stuffed mamba: state capacity and retrieval in recurrent models. *conference on language modeling (COLM)*.

key finding: the retrieval horizon of state space models (ssms, including mamba) scales exponentially with state size but is fundamentally limited: beyond a critical sequence length (determined by state dimensions), retrieval accuracy drops sharply. the "stuffed mamba" experiments systematically measured how many facts can be stored in the recurrent state before retrieval degrades. the critical finding is that ssms face a hard capacity wall: once the state is full, new information overwrites old information in a lossy fashion, unlike attention which can always look back at the kv cache.

relevance to neural machine: todorov's mamba3 layers face this same capacity wall. the kda delta-rule state has a softer degradation (old associations are gradually overwritten via the decay gate) but still has finite capacity determined by the state matrix dimensions. the 3:1 architecture mitigates this by providing mla layers that can retrieve from the full kv cache without capacity limits. the stuffed mamba result quantifies the expected retrieval horizon and provides a basis for choosing state dimensions: larger state = longer horizon but more computation per step.

confidence: high. systematic empirical study with clear methodology. caveat: the specific capacity numbers depend on the ssm variant and task; extrapolation to kda (which uses a different state update rule) is approximate.

### rnns cannot solve associative recall perfectly

wen, K. et al. (2025). formal lower bounds on associative recall in recurrent models. *international conference on learning representations (ICLR)*.

key finding: for any rnn with fixed-dimensional hidden state, there exists a minimum sequence length beyond which perfect associative recall is impossible regardless of the model's parameters. the lower bound is information-theoretic: a d-dimensional state can encode at most O(d log d) bits, and each key-value association in mqar requires O(log V) bits where V is the vocabulary size. when the number of associations exceeds the state's information capacity, errors are inevitable. this is a fundamental limit, not an optimization failure.

relevance to neural machine: this result applies to todorov's kda and mamba3 layers. the kda state matrix (d_k x d_v) has capacity O(d_k * d_v * log 3) bits under ternary quantization of the state values (if quantized) or O(d_k * d_v * 16) bits in fp16. for d_k = d_v = 64, this is ~65K bits in fp16, enough for ~4000 associations at 16-bit vocabulary indices. beyond this, the mla layers must handle retrieval. the formal lower bound confirms that the hybrid architecture is not optional -- pure recurrence will always have a retrieval horizon.

confidence: high. mathematical proof. caveat: the bound is worst-case; for structured data (like natural language), the effective capacity may be higher because associations are not independent.

## state compression

### svd kv compression

chang, C. et al. (2025). palu: compressing kv cache with low-rank projection. *international conference on learning representations (ICLR)*.

key finding: the key-value cache in attention can be compressed by ~50% using svd-based low-rank projection with negligible perplexity increase (<0.1pp). the compression works because kv representations are empirically low-rank: the top singular values capture most of the variance. palu applies a learned low-rank projection to keys and values before caching, and reconstructs the full-rank representations on the fly during attention computation. the method is training-free (applied post-hoc to pretrained models).

relevance to neural machine: todorov's mla layers already use low-rank kv compression (multi-head latent attention compresses kv into a shared latent). palu's finding that 50% compression is nearly lossless validates the mla design choice. for the kda state, svd compression could reduce the state matrix dimensions without proportional capacity loss, potentially extending the retrieval horizon at fixed compute cost. however, the kda state is dynamically updated (not a static cache), so post-hoc svd would need to be applied to the accumulated state, which may interact with the delta-rule updates.

confidence: high. validated on multiple pretrained models across standard benchmarks. caveat: training-free compression; training with compression in the loop may achieve better results.

## memory augmentation

### memory caching gives rnns growing capacity

behrouz, a. et al. (2026). memory caching: rnns with growing memory. arXiv:2602.24281.

key finding: by periodically checkpointing the recurrent state and allowing the model to attend over past checkpoints, rnns can achieve growing (rather than fixed) memory capacity. the checkpointed states serve as a compressed kv cache, where each checkpoint summarizes a segment of the input sequence. retrieval from checkpoints uses a lightweight attention mechanism over the checkpoint embeddings. this converts a fixed-capacity rnn into a growing-capacity model at the cost of linear memory growth in the number of checkpoints.

relevance to neural machine: this is directly applicable to todorov's architecture. the kda state already accumulates a compressed representation of past inputs; periodically saving this state and allowing cross-attention over saved states would extend the effective context without changing the core recurrence. the mla layers could handle checkpoint retrieval, while kda layers continue to process the current segment. the key design question is checkpoint frequency: too frequent and it becomes equivalent to full kv caching; too rare and the compression is too lossy.

confidence: medium. recent result with limited replication. caveat: the checkpoint attention adds latency and memory proportional to the number of checkpoints; the tradeoff with simply using longer kv caches needs quantification.

## the unmeasured joint

### outer product + decay + ternary: no existing theory

synthesis across the papers above.

key finding: no published paper measures the associative memory capacity under the simultaneous constraints of: (a) outer-product state updates (as in kda), (b) exponential decay gating (channel-wise forget), and (c) ternary input quantization (spikes). each constraint is individually characterized -- outer-product capacity is well-studied (classical hopfield), decay is studied in gated deltanet, ternary synaptic capacity is studied by bhattacharjee & martin -- but their interaction is unknown. the decay rate and ternary quantization may be complementary (decay removes old low-confidence associations, ternary prevents low-confidence associations from forming) or antagonistic (ternary noise may accelerate state degradation under decay).

relevance to neural machine: this is the core open question for todorov's memory system. the kda state matrix is updated via delta-rule outer products between ternary spike keys and continuous value vectors, with channel-wise exponential decay. no theory predicts the resulting capacity. the empirical evidence (267m model achieves bpb 0.663x vs transformer, implying competitive memory performance) suggests the joint capacity is adequate, but the theoretical understanding is missing. a dedicated simulation sweeping pattern count, decay rate, and quantization level would be the most direct way to characterize this.

confidence: high for the gap identification. this is not a claim that the joint is problematic, only that it is unmeasured.

## see also

- [[pattern_completion]]
- [[hippocampal_memory]]
- [[complementary_learning_systems]]
- [[delta_rule_theory]]
- [[kda_channel_gating]]
- [[mla_compression]]
- [[plasticity_to_kda_delta_rule]]
- [[memory_systems_to_kda_mla]]

## relevance to the neural machine

### validated connections
- kda delta-rule state implements gated deltanet's error-correcting write mechanism, achieving perfect mqar at 1.3B
- mla layers provide exponential-capacity modern hopfield retrieval at 2^(d/2) patterns
- ternary synapses preserve polynomial capacity scaling (N^(n-1)), reduced prefactor only
- the 3:1 hybrid ratio (kda:mla) balances linear-capacity fast recurrence with exponential-capacity slow attention

### challenged assumptions
- no theory characterizes kda capacity under joint outer-product + decay + ternary constraints
- rnns face a formal lower bound on associative recall -- hybrid architecture is necessary, not optional
- ssm retrieval horizon scales exponentially with state size but has a hard wall -- state dimensions must be sized to the expected retrieval range

### open questions
- what is the empirical capacity of the kda state matrix under realistic (non-random) pattern distributions?
- does ternary input quantization increase or decrease effective capacity relative to continuous inputs? (sparse patterns may increase separation, noise may decrease fidelity)
- could state checkpointing (memory caching) extend kda's retrieval horizon at acceptable latency cost?
- what is the optimal channel-wise decay schedule to maximize capacity across multiple timescales?
