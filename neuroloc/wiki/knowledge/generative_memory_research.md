# generative memory research

status: current (as of 2026-04-16).

curated peer-reviewed research on memory compression through structural replacement of cached content with generative mechanisms. this is a different level from bit-quantization work like turboquant or kvtc: where quantization asks "how many bits per stored value," generative memory asks "do we need to store the value at all, or can we regenerate it from a learned model." the biological brain operates at the latter level: it retains an estimated 10^5 to 10^7 bytes of effective content memory from approximately 10^17 bytes of lifetime raw sensory input (both figures are cognitive-science back-of-envelope estimates, not direct measurements; the 10^7 upper bound already requires assumptions about near-zero marginal cost for schema-consistent storage that are not directly measured), implying an effective compression ratio of roughly 10^10 to 10^12. this is nine to eleven orders of magnitude beyond what any bit-quantization method achieves.

this article curates the engineering literature that approaches memory compression structurally. it is a companion to `compression_and_bottlenecks.md` (which treats compression as a biological principle) and `compression_architecture.md` (which enumerates specific compression mechanisms for todorov). the focus here is on architectures where "the model is the memory."

## quantization ceilings to beat

the published quantization frontier as of early 2026:

- turboquant (zandieh et al., google research, arxiv 2504.19874, to appear iclr 2026): online vector quantization with proved information-theoretic near-optimality. information-theoretic lower bound for any randomized quantizer at bit-width b is mse ~ 1/4^b, and turboquant's upper bound is within constant factor ~2.7 of this bound. the paper reports near-lossless longbench-e performance at 3.5 bits per value (claimed ~4.5x at zero degradation, author-reported).

- kvtc (nvidia research, arxiv 2511.01815): pca decorrelation + dynamic programming bit allocation + deflate entropy coding. the paper claims up to 20x kv cache reduction at under 1 point accuracy drop on longbench, math-500, and ruler, combining a classical rate-distortion pipeline with nvcomp gpu-accelerated entropy coding. specific numbers require direct verification against the paper's main tables.

- mla (deepseek, arxiv 2405.04434): multi-head latent attention projects kv to a low-rank latent per token. the deepseek-v3 technical report (arxiv 2412.19437) reports large kv cache reductions vs standard mha and has been production-validated at 670b scale. still stores the compressed kv; this is architecture-level compression not structural replacement.

- vqkv (arxiv 2603.16435, march 2026): vector quantization applied directly to kv cache at inference time. reports 82.8 percent compression (approximately 5.7x) on llama-3.1-8b with 98.6 percent longbench retention. training-free, verbatim indexed storage, not model-regenerative. this is a strong recent baseline for structural kv compression.

- commvq (apple ml, arxiv 2506.18879, august 2025): additive commutative quantization with a lightweight encoder; codebook designed to commute with rope. reports ~87.5 percent kv cache reduction (approximately 8x) at 2-bit quantization; 1-bit feasible with minimal accuracy loss. training-free inference-time compression.

- vqllm (kumar et al., neurips enlsp-iv 2024, arxiv 2410.15704): residual vq applied to transformer kv cache with grouped-channel quantization. reports approximately 5.5x compression vs fp16, competitive with scalar quantization baselines at simpler design. verbatim indexed storage.

- deltakv (arxiv 2602.08005, february 2026): encodes semantic residuals of each kv entry relative to similar historical tokens. reports kv cache reduced to ~29 percent of original (approximately 3.45x) with near-lossless performance on longbench, scbench, and aime. structurally this combines mechanism 1 (predictive filtering of residuals) with mechanism 4 (similarity-based content-addressable reference), making it one of the few current papers that explicitly compounds two of the biology-inspired mechanisms.

the relevant observation: all these methods store the kv cache verbatim as indices or residuals after compression. none of them replace storage with computation. the best published structural (non-bit-quantization) compression ratio for language model memory is in the range of ~5 to ~8x from vqkv/commvq/vqllm and ~3.5x from the mechanism-compounding deltakv. the shannon rate-distortion bound applies to the stored content; beating it requires changing what is stored.

## generative replacement: memory as a learned decoder

### dreamer v3

hafner, d., pasukonis, j., ba, j., lillicrap, t. (2023). mastering diverse domains through world models. nature.

dreamer v3 learns a recurrent state-space model (rssm) that predicts future latent states from actions. policy optimization occurs entirely on imagined trajectories in the latent space, with no verbatim storage of observations. the rssm maintains a 32-category discrete latent of 32 dimensions per timestep, compressing each timestep to approximately 160 bits. for an hour of environment interaction (3600 timesteps), this is 72 kilobytes total. the "memory" is the learned world model weights plus the discrete latent codes.

relevance: dreamer v3 is the strongest demonstration that memory can be replaced by a world model + tiny latent codes. it has not been applied to autoregressive language modeling at scale. the key transferable idea is that storing a trained decoder and a small latent index achieves effective compression ratios beyond any quantization scheme. the discrete latents are reconstructed by the decoder into full observations, which is structurally what the brain does at retrieval.

confidence: high. dreamer v3 is peer-reviewed in nature (2025), open-sourced, and validated on 150+ benchmark domains. caveat: the compression ratio is domain-specific and depends on how predictable the environment is. noisy environments compress less.

### neural implicit representations

mildenhall, b., srinivasan, p., tancik, m., barron, j., ramamoorthi, r., ng, r. (2020). nerf: representing scenes as neural radiance fields for view synthesis. eccv.

sitzmann, v., martel, j., bergman, a., lindell, d., wetzstein, g. (2020). implicit neural representations with periodic activation functions. neurips.

neural implicit representations (inrs) store entire signals (images, 3d scenes, audio) in the weights of small mlps. nerf compresses a 3d scene from hundreds of images (hundreds of megabytes) into an mlp of roughly 1 mb. the mlp is the storage; retrieval is a forward pass at the query coordinates. compression ratios of 10^3 to 10^5 are routinely demonstrated for natural scenes.

relevance: inrs are the cleanest example of "the model is the memory." for language memory, the analog is learning a decoder mlp whose weights encode a database of past activations, retrievable by a learned query. no published paper does this for language model kv caches, but the mechanism is directly transferable to outer-product associative memories.

confidence: high for the mechanism. medium for direct applicability to language memory. caveat: inrs assume the signal is representable by a smooth function of spatial coordinates. language state trajectories are discrete and non-smooth, so a naive inr does not work. progress requires adapting the inr idea to semantic (not spatial) coordinates.

### hypernetworks

ha, d., dai, a., le, q. v. (2016). hypernetworks. iclr 2017. arxiv 1609.09106.

a hypernetwork is a neural network whose output is the weights of another network. instead of storing the weights of network b, you store the hypernetwork that generates them, which can be much smaller. ha et al. showed parameter-count reductions of roughly an order of magnitude on small lstm language models with modest quality impact; the frequently-quoted "30x" figure is author-reported on specific ptb-scale language-model variants and should not be extrapolated to modern transformer scales without direct verification.

relevance: hypernetworks extend the "model is memory" idea one level: memory is now "a code that generates the parameters of a model that then reconstructs the content." for our case, a hypernetwork could generate per-memory decoder networks from a small code, which then reconstruct the stored content. this compounds with generative replacement.

confidence: medium. hypernetworks are well-established but have not been scaled to memory compression for transformer-based language models at the 300m+ parameter scale. caveat: training hypernetworks is unstable; most published work is on small vision models or ptb-scale lstms.

### larimar

das, p. et al. (2024). larimar: large language models with episodic memory control. icml 2024. arxiv 2403.11901.

larimar adds a distributed episodic memory matrix to a pretrained language model. memory is a learned matrix of shape (k, d) where k is the number of slots and d is the dimension. write operations use a pseudo-inverse update rule. retrieval is associative. larimar enables one-shot knowledge updates (add or remove a fact) without fine-tuning, with 4x-10x speedup over fine-tuning baselines on the zsre knowledge-editing benchmark.

relevance: larimar demonstrates that an associative memory with delta-rule-like updates can store and retrieve facts with a fixed-size memory matrix. the memory does not grow with new facts; new facts overwrite existing slots. this satisfies the "no weight growth" constraint for continuous learning. the update rule is mathematically equivalent to the kda delta-rule erasure when keys are orthogonal.

confidence: high. larimar is peer-reviewed at icml with reproducible benchmarks. caveat: larimar's memory is tested on small, structured knowledge-editing tasks (single facts), not on streaming continuous experience. adversarial robustness is not evaluated.

### memorizing transformers

wu, y., rabe, m. n., hutchins, d., szegedy, c. (2022). memorizing transformers. iclr 2022. arxiv 2203.08913.

memorizing transformers augment standard transformers with a large external knn-based memory indexed by key-value pairs from past context. retrieval uses approximate nearest neighbor search. the memory can scale to 262k tokens. the model gets "for free" access to arbitrarily long context at the cost of a database.

relevance: this is the "naive cache" approach. it demonstrates that external long memory helps, but it does not compress: it stores verbatim kv pairs and grows linearly with context length. it is the baseline that generative memory must beat in compression-per-bit-of-performance-gained. the memorizing transformer architecture does not satisfy the "no weight growth" constraint because the memory grows with tokens seen.

confidence: high for described behavior, low for relevance to our goals. flagged here because it is commonly cited as a memory-extension paper when in fact it is a storage-extension paper.

### titans

behrouz, a., zhong, p., mirrokni, v. (google research, 2025). titans: learning to memorize at test time. arxiv 2501.00663.

titans has three memory modules: persistent (fixed pretrained parameters), short-term (sliding window attention), and long-term (a neural memory whose weights are updated at test time by gradient descent on a "surprise" signal). the surprise signal is the gradient magnitude of the model's own loss on each token; high-surprise tokens get large memory updates, predictable tokens get small ones. forgetting is implemented as a gated decay.

relevance: titans is the most architecturally sophisticated published memory system that satisfies the no-weight-growth constraint. the long-term memory is a fixed-size mlp that gets updated in place. the surprise signal is exactly the "predictive filter" the brain uses: only unexpected information causes memory updates. titans has been demonstrated to 2m token context. however, the long-term memory stores parameter updates, not a generative code; it is closer to online learning than to generative replacement.

confidence: high for the described architecture. medium for the claim that surprise-gating is equivalent to predictive coding. caveat: titans was released january 2025 and has not yet been independently reproduced at scale. the benchmark claims (2m context, needle-in-haystack) require verification.

## discrete latent codebooks

### vq-vae

van den oord, a., vinyals, o., kavukcuoglu, k. (2017). neural discrete representation learning. neurips. arxiv 1711.00937.

vector-quantized variational autoencoder learns a discrete codebook of latent codes. input signals are encoded to the nearest codebook entry, reconstructed by a decoder. compression is defined by codebook size (typically 512 to 8192 entries, so 9-13 bits per latent token) and the downsampling factor of the encoder.

relevance: vq-vae is the canonical method for discrete latent compression. each "memory" becomes a sequence of codebook indices (tens of bits per token versus hundreds of bytes). the decoder is shared across all memories, so storage cost amortizes over the dataset. for language, this is a path to achieving 100x-1000x compression by storing indices instead of hidden states.

caveat: vq-vae has not been demonstrated on autoregressive language model hidden states at scale. the discrete bottleneck often loses information that affects downstream language tasks. residual vq-vae (rvq-vae) partially addresses this with multi-stage codebooks.

confidence: high for the method. medium for direct applicability to transformer kv or delta-rule states.

### soundstream / encodec

zeghidour, n., luebs, a., omran, a., skoglund, j., tagliasacchi, m. (2021). soundstream: an end-to-end neural audio codec. ieee/acm transactions on audio, speech, and language processing.

défossez, a., copet, j., synnaeve, g., adi, y. (2022). high fidelity neural audio compression. arxiv 2210.13438 (encodec).

encodec compresses audio at 1.5-24 kbps using residual vector quantization on learned features. the compression ratio is 50x-300x vs raw waveforms. the codebook is fixed after training; inference-time audio is compressed by selecting codebook entries and decoded by a learned generator.

relevance: these audio codecs demonstrate that a combination of neural encoder + discrete rvq codebook + neural decoder can achieve 100x+ compression at perceptual quality. for non-audio signals (language, vision), the same architecture applies if the encoder maps to a domain-appropriate latent space. no published paper applies neural codec principles to transformer kv memories.

confidence: high for the results in audio. low for direct language applicability, moderate as an engineering template.

## predictive filtering: surprise-based memory

### rao and ballard predictive coding

rao, r. p. n., ballard, d. h. (1999). predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. nature neuroscience 2, 79-87.

foundational biological paper. each cortical level maintains a generative model of the level below. only prediction errors propagate upward. the implication is that cortex only processes and stores unexpected information.

relevance: the direct algorithmic consequence is that a learned prediction model for the next token / next state can filter out predictable input, leaving only the residual. over an average natural-language sequence, most tokens are predictable given context, so the residual stream is small. the compression ratio achievable this way depends on the prediction quality and is bounded by the shannon entropy of the data conditioned on the model's predictions.

confidence: very high for the biological claim. medium for the engineering implementation: adding explicit prediction filtering to a transformer kv cache has been discussed (see titans above) but not benchmarked against bit quantization on matched tasks.

### free energy principle and active inference

friston, k. (2010). the free-energy principle: a unified brain theory? nature reviews neuroscience 11, 127-138.

friston's free-energy principle is a mathematical framework that unifies perception, action, and learning as minimization of variational free energy. in memory terms, the brain stores a generative model whose parameters minimize the surprise of past experience. the model itself is the compression of experience.

relevance: this is the most general statement of "memory is a model." it suggests that the correct compression target is not bits-per-stored-value but bits-per-model-parameter that minimize free energy on held-out data. the engineering equivalent is training a compact generative model that reproduces past context at retrieval time.

confidence: high for the theoretical framework. low for how to translate it into a specific algorithm. caveat: active inference has produced few scaling-law-beating engineering results despite 20 years of theory development.

## continual learning without weight growth

### generative replay

shin, h., lee, j. k., kim, j., kim, j. (2017). continual learning with deep generative replay. neurips. arxiv 1705.08690.

shin et al. propose training a generator alongside the main task model. when new tasks arrive, the generator produces samples from the distribution of old tasks, and the main model is trained on a mixture of new data and generated old data. this prevents catastrophic forgetting without storing old examples.

relevance: this is the canonical "generative replay" approach to continual learning. it satisfies the no-weight-growth constraint if both the generator and the main model are fixed-size. it demonstrates that storing a generator is sufficient to preserve old experience without storing raw data. the compression ratio depends on how well the generator captures the old distribution.

confidence: high for the method. caveat: shin et al. tested on small classification tasks (mnist, cifar). scaling to language modeling with streaming updates has not been demonstrated. the generator must be trained before the new task arrives, which is incompatible with true online continual learning.

### elastic weight consolidation and follow-ons

kirkpatrick, j. et al. (2017). overcoming catastrophic forgetting in neural networks. pnas. arxiv 1612.00796.

zenke, f., poole, b., ganguli, s. (2017). continual learning through synaptic intelligence. icml. arxiv 1703.04200.

aljundi, r. et al. (2018). memory aware synapses: learning what (not) to forget. eccv. arxiv 1711.09601.

these methods protect important weights from being overwritten by new learning. ewc adds an l2 penalty weighted by the fisher information of each parameter; synaptic intelligence uses an online estimate of parameter importance; mas uses gradient magnitude. all satisfy the no-weight-growth constraint since they only modify how weights are updated, not how many there are.

relevance: these are the established techniques for continual learning on fixed-size models. they do not address memory compression directly; they address the question "how do you safely update existing weights with new information." for our tiered memory, these methods apply at the tier boundaries when old memories are overwritten.

confidence: high for the methods. medium for the level of protection achieved: all three are known to fail catastrophically on long task sequences, typically collapsing after 5-20 tasks. adversarial robustness has not been systematically evaluated.

## biological inspiration

### complementary learning systems

mcclelland, j. l., mcnaughton, b. l., o'reilly, r. c. (1995). why there are complementary learning systems in the hippocampus and neocortex. psychological review 102, 419-457.

cls theory: the brain has two learning systems with different properties. the hippocampus learns fast, from single experiences, with high-detail storage. the neocortex learns slowly, extracts statistical regularities, and stores schema-level knowledge. memories are transferred from hippocampus to cortex during sleep via replay. this is the biological precedent for tiered memory with different compression levels.

relevance: the cls architecture is directly analogous to the proposed 5-tier memory for todorov. hippocampus corresponds to tiers 0-1 (verbatim, recent, high-detail). neocortex corresponds to tiers 2-4 (compressed, old, schema-level). the transfer mechanism (sleep replay) corresponds to an offline consolidation pass.

confidence: high for the biological model. high for the architectural analogy. caveat: the cls model has been implemented in deep learning several times (fast-weight transformers, parisotto et al. 2020; memorizing transformers; larimar) but none of these implementations achieve the brain's compression ratio.

### hippocampal indexing theory

teyler, t. j., discenna, p. (1986). the hippocampal memory indexing theory. behavioral neuroscience 100, 147-154.

indexing theory: the hippocampus does not store episodic memories; it stores pointers to cortical patterns that represent the memory. retrieval activates the pointer, which reinstates the cortical pattern. the pointer is much smaller than the pattern. estimated compression ratio is 100x to 1000x per stored memory, because the cortical pattern is ~10^6 bits and the hippocampal index is ~10^2-10^4 bits.

relevance: this is the biological precedent for content-addressable memory where storage is tiny (pointer) and retrieval reconstructs the full content (via the cortical generative model). for todorov, this is the justification for replacing the delta-rule state with a learned index plus a generative decoder.

confidence: high for the biological claim. the engineering implementation is not standard; the closest is memorizing transformers but that stores full kv pairs, not indices.

### tolman-eichenbaum machine

whittington, j. c. r., muller, t. h., mark, s., chen, g., barry, c., burgess, n., behrens, t. e. j. (2020). the tolman-eichenbaum machine: unifying space and relational memory through generalization in the hippocampal formation. cell 183, 1249-1263.

tem is a deep learning model that implements hippocampal-entorhinal memory. it uses a factorized representation where abstract spatial structure is separated from specific sensory content. this factorization enables transfer learning across environments. the memory is a learned generative model that reconstructs episodes from sparse cues.

relevance: tem is one of the few deep learning models that explicitly implements generative-replacement memory for autoregressive sequences. it shows that factored representations (structure + content) enable much better compression than dense hidden-state storage. this is a direct template for how to build tiered memory for language.

confidence: high. tem is published in cell and has been replicated. caveat: tem is tested on grid-world navigation, not on language. the structure-content factorization may not transfer cleanly.

## theoretical limits

### rate-distortion with a learned prior

ballé, j., minnen, d., singh, s., hwang, s. j., johnston, n. (2018). variational image compression with a scale hyperprior. iclr. arxiv 1802.01436.

this paper demonstrates that learned image compression with a variational prior approaches the shannon rate-distortion lower bound for natural images. the key insight is that a learned prior captures the structural regularity of natural images, which a generic quantizer cannot. compression ratios are 2x-10x better than jpeg for equivalent quality.

relevance: the theoretical ceiling for memory compression is not the raw shannon entropy of the stored values (which is what turboquant approaches), but the shannon entropy conditioned on a learned prior of what "valid" memory states look like. a learned prior over natural language hidden states could push the rate-distortion frontier far beyond turboquant's 2.7x factor.

confidence: high for the image case. high for the transferable principle. the application to memory compression is speculative but theoretically sound.

### kolmogorov complexity perspective

li, m., vitanyi, p. m. b. (2019). an introduction to kolmogorov complexity and its applications (4th edition). springer.

kolmogorov complexity of a string is the length of the shortest program that produces it. for natural language, kolmogorov complexity is upper-bounded by the shannon entropy but can be much lower if structural regularities exist. no practical method can compute kolmogorov complexity, but approximations via learned generative models (e.g., perplexity of a language model on its own output) provide useful upper bounds.

relevance: the theoretical floor for memory compression is kolmogorov complexity, not entropy. a perfect generative model would compress memory to its kolmogorov complexity. published language models (llama, gpt) achieve compression ratios of 10-20x via next-token prediction, which approximates kolmogorov compression in the large-data limit. the gap between current llm compression and kolmogorov is large (likely 10x-100x more compression is possible) and represents the theoretical headroom for generative-replacement memory.

confidence: high for the theoretical framework. unknown for the practical achievable ratio.

## summary of confirmed and unconfirmed claims

| claim | status | source |
|---|---|---|
| brain achieves approximately 10^10-10^12x effective compression of sensory stream (raw input ~10^17 bytes, retained content ~10^5-10^7 bytes) | back-of-envelope, uncertain | bartol et al. 2015 (synaptic info; measured ca1 specifically) + cognitive science estimates |
| turboquant is within constant-factor 2.7 of shannon rate-distortion lower bound | confirmed | zandieh et al. iclr 2026 |
| generative-replacement memory achieves 10^3x+ compression in rl (dreamer v3) | confirmed | hafner et al. nature 2025 |
| inr compresses 3d scenes 10^3-10^5x | confirmed | nerf and follow-on work |
| vq-vae compresses images at 100x+ via codebooks | confirmed | van den oord et al. 2017, subsequent work |
| titans surprise-gated memory scales to 2m tokens | claimed, not reproduced | behrouz et al. 2025 |
| larimar's memory update is equivalent to delta-rule erasure with orthogonal keys | consistent with math | das et al. icml 2024 |
| hypernetworks compress language models by roughly an order of magnitude (author-reported ~30x on small lstms) | confirmed for small lstms only; not demonstrated at modern transformer scale | ha et al. iclr 2017 |
| no published architecture compounds all six brain-inspired mechanisms | confirmed as a gap | review of 2024-2026 literature |
| the 100/80/30/forget tiered cascade for recurrent memory | not implemented anywhere | gap |
| kolmogorov complexity is the theoretical floor for memory compression | confirmed theoretically | li and vitanyi |
| current llm compression ratios are 10-100x away from kolmogorov | likely true, unmeasured | inference from scaling laws |

## open problems

1. **how to train a generative replacement for transformer hidden states** at the scale of current language model kv caches. dreamer v3 works for rl latent states of 32 categories; language hidden states are 1024-dimensional continuous vectors. the analogous architecture for language has not been published.

2. **how to combine predictive filtering (surprise gating) with generative replacement** in a single architecture. titans does the filtering; dreamer v3 does the replacement; no paper does both.

3. **how to train a fixed-size memory under adversarial continuous learning** without catastrophic forgetting and without weight growth. larimar handles benign one-shot updates; it has not been tested adversarially.

4. **how to evaluate memory compression at the kolmogorov level** instead of at the bit level. no published benchmark measures "shortest program reproducing the context" for language memory.

5. **what the right latent space is for a semantic memory codebook**. image latents (vq-vae, encodec) are well-understood; language latents at the feature level are not standard.

## see also

- `compression_and_bottlenecks.md` (biology-focused companion article)
- `compression_architecture.md` (todorov-specific compression mechanisms)
- `memory_systems_research.md` (biological memory systems)
- `ternary_compression_research.md` (bit-quantization methods)
- `imagination_computation_research.md` (generative recombination)
- `memory_capacity_research.md` (associative memory capacity)
