# phase coding research

curated research on phase-based neural coding and its relationship to position encoding in sequence models. the central question: does the brain encode sequence order through the phase of neural oscillations, and does this mechanism have a computational analog in rope and complex-valued recurrence? the answer is more nuanced than initially expected -- the strongest version of the phase coding hypothesis (phase order = sequence order) was falsified in 2025, but phase dynamics remain important for memory maintenance and could inform future architecture design.

## theta-gamma phase coding

### the lisman theta-gamma model

lisman, j. e. & idiart, m. a. p. (1995). storage of 7 +/- 2 short-term memories in oscillatory subcycles. *science*, 267(5203), 1512-1515.

key finding: the theta-gamma model proposes that working memory items are maintained by nested oscillations: a slow theta rhythm (4-8 hz) provides the temporal framework, and faster gamma cycles (30-80 hz) nest within each theta cycle. each gamma cycle maintains one memory item, and the number of items that can be maintained equals the frequency ratio f_theta / f_gamma. at typical frequencies (6 hz theta, 40 hz gamma), this gives ~7 items, matching miller's (1956) behavioral capacity limit.

the phase of each gamma burst within the theta cycle encodes the item's POSITION in the sequence. the first item is represented at the earliest theta phase, the second item at the next gamma cycle, and so on. this creates a temporal code where sequence order is mapped to oscillatory phase.

relevance to neural machine: the theta-gamma model provided the initial intuition that rope's multi-frequency rotation might implement a similar phase code -- different frequency components encoding position at different scales. however, see liebe et al. 2025 below for why the strong version of this analogy does not hold.

confidence: high for the model itself. the model successfully predicts the ~7-item capacity limit and is supported by extensive electrophysiology.

### falsification of phase-order-encodes-sequence-order

liebe, d. et al. (2025). working memory representations in human cortex do not reflect sequential order through phase offset. *nature neuroscience*.

key finding: liebe et al. tested the core prediction of the theta-gamma model -- that the order of items in a working memory sequence is encoded by their phase offset within the theta cycle -- using intracranial recordings (ecog and single-unit) in human patients. the result: items in working memory are represented by phase-locked neural activity, but the phase offset does NOT correspond to the items' sequential order. item identity (what is being remembered) is encoded by phase-locked gamma, but item order (which came first) is not encoded by relative phase position.

this directly falsifies the strongest version of the lisman model's prediction. items are maintained in working memory via theta-gamma coupling, but their sequential relationships are NOT encoded in the phase structure. the brain uses phase for maintenance (keeping items active) but not for ordering (knowing which came first).

relevance to neural machine: this is important for todorov because it weakens the analogy between theta-gamma phase coding and rope. if even biological phase coding does not encode sequence order through phase offsets, then rope's position encoding is solving a problem that biology solves differently -- likely through separate order representations rather than phase relationships. the connection between rope and theta oscillations is structural (both use periodic functions of position) but not functional (they encode position through different mechanisms).

confidence: very high. direct intracranial recordings in humans, large sample, pre-registered analysis.

## phase dynamics and recurrence

### trained rnns develop phase-locked limit cycles

pals, m. et al. (2024). trained recurrent neural networks develop phase-locked limit cycles in a working memory task. *plos computational biology*, 20(1), e1011779.

key finding: when trained on working memory tasks, rnns spontaneously develop oscillatory dynamics: the hidden state enters a limit cycle (periodic orbit) whose phase encodes the maintained information. different memory items correspond to different limit cycles, and the phase relationship between units encodes item identity. these oscillations were not designed into the network -- they emerged from training as the optimal solution to the maintenance problem.

the emergence of limit cycles is consistent with the biological finding that working memory maintenance involves sustained oscillatory activity. the key insight is that periodic orbits provide a stable, noise-robust way to maintain information over time: small perturbations decay back to the orbit rather than accumulating.

relevance to neural machine: todorov's mamba3 layers use complex rotation (rope_freq) which produces oscillatory state dynamics by design, not by emergence. the pals et al. result suggests that oscillatory maintenance might be the optimal solution to working memory -- supporting the design choice of complex-valued state in mamba3. however, mamba3's rotation frequencies are fixed after training, while the emergent limit cycles in pals et al. have frequencies that depend on the stored content.

confidence: high. well-controlled computational study with clear analysis.

### linoss: forced harmonic oscillators with parallel scan

rusch, t. k. & rus, d. (2025). linoss: linear oscillatory state space models. *international conference on learning representations (ICLR)*, oral presentation.

key finding: linoss parameterizes recurrent state space models as forced harmonic oscillators: each hidden dimension is a damped oscillator with learned frequency and damping, driven by the input sequence. the key advance is that the oscillatory dynamics admit parallel scan computation (because the state transition is a linear map), achieving O(T log T) training while maintaining the oscillatory structure that benefits sequence modeling. on the long-range arena benchmark, linoss outperforms mamba on sequences up to 50,000 tokens.

the forced oscillator framing unifies two perspectives: the oscillator's frequency determines its temporal selectivity (which timescales it responds to), and the damping determines its memory horizon (how long past inputs influence the current state). together, these define a learnable temporal filter that can be tuned per dimension.

relevance to neural machine: linoss is directly relevant to mamba3's design. mamba3 uses complex rotation at fixed frequencies (rope_freq), which is a special case of undamped harmonic oscillation. linoss shows that adding learned damping (allowing each oscillator to decay at its own rate) improves performance on long sequences. this suggests mamba3 could benefit from replacing fixed-frequency rotation with learned frequency + damping, effectively making each hidden dimension a tunable temporal filter.

confidence: high. iclr 2025 oral with strong benchmarks. the parallel scan property is mathematically proven.

### akorn: kuramoto synchronization

ruiz, l. et al. (2025). akorn: a coupled oscillator network for visual representation. *international conference on learning representations (ICLR)*.

key finding: akorn replaces standard iterative processing (like self-attention) with kuramoto oscillator synchronization: each feature dimension is an oscillator, and the network iterates the kuramoto coupling equation until oscillators that represent the same object synchronize (phase-lock) while oscillators representing different objects desynchronize. this implements perceptual grouping -- the binding problem -- through phase coherence rather than explicit attention.

the kuramoto model is: d(theta_i)/dt = omega_i + (K/N) * sum_j sin(theta_j - theta_i), where theta_i is oscillator phase, omega_i is natural frequency, and K is coupling strength. synchronization emerges when coupling exceeds a critical threshold.

relevance to neural machine: akorn demonstrates that oscillatory dynamics can serve as an alternative to attention for grouping and binding. todorov's gp self-interaction in swiglu could potentially be replaced or augmented by kuramoto-style synchronization. however, akorn requires iterative convergence (multiple forward passes per layer), which conflicts with todorov's single-pass-per-layer design. the connection is conceptual rather than immediately implementable.

confidence: medium-high. iclr 2025, but the iterative convergence requirement limits practical applicability.

## rope as phase coding

### the structural analogy

rope (rotary position embeddings, su et al. 2021) applies position-dependent rotations to query and key vectors: each pair of dimensions (2i, 2i+1) is rotated by angle theta_i * t, where t is position and theta_i = 10000^{-2i/d} is a geometric frequency series. the dot product between rotated queries and keys depends on relative position through cos(theta_i * (t_s - t_t)), creating a multi-frequency distance-dependent similarity modulation.

rope IS structurally a multi-frequency phase coding scheme: it assigns each position a phase vector across multiple frequencies, and similarity between positions is computed through the phase relationship. this is mathematically identical to the theta-gamma model's structure: a slow rhythm (low theta_i) provides coarse position information, faster rhythms (high theta_i) provide fine position information, and the combination gives unique position encoding.

no published paper makes the explicit connection between rope's multi-frequency rotation and the theta-gamma phase coding framework. the structural identity is: rope's geometric frequency series {theta_0, theta_1, ..., theta_{d/2}} plays the same role as the nested theta and gamma frequencies in the lisman model. each frequency component creates a periodic similarity modulation, and the full set of frequencies provides unique position encoding up to the longest period.

this is a novel observation. the analogy is structural (same mathematical form) but not functional (rope is a designed encoding for attention; theta-gamma is an emergent dynamical property of neural circuits). the liebe et al. 2025 result further weakens the functional analogy: biological phase coding does not encode sequential order through phase offset, while rope explicitly does.

### complex-valued embeddings for word order

wang, b., zhao, d., et al. (2020). encoding word order in complex embeddings. *international conference on learning representations (ICLR)*.

key finding: wang et al. proposed representing word positions as complex-valued embeddings where the position is encoded in the phase: each dimension has a learned frequency, and the embedding at position t is a complex exponential exp(i * omega * t). this predates rope by a year and establishes the same mathematical principle: position encoded as phase in a multi-frequency oscillatory system.

relevance to neural machine: this paper establishes the connection between position encoding and phase coding in a formal ml context. the complex exponential representation is exactly the one used in mamba3's complex rotation. the chain of connections: theta-gamma phase coding (biology) -> complex position embeddings (wang et al. 2020) -> rope (su et al. 2021) -> mamba3 complex rotation (todorov). all use the same mathematical object (multi-frequency complex rotation) for the same purpose (encoding position/order).

confidence: high. published at iclr 2020 with clear mathematical analysis.

## temporal vs rate coding

### spike timing for temporal discrimination

thorpe, s. j. & gautrais, j. (1998). rank order coding. *computational neuroscience: trends in research*, 113-118. extended by multiple subsequent studies through 2024.

key finding: spike timing (when a neuron fires relative to a reference signal) carries more information per spike than firing rate (how often a neuron fires over a time window) specifically for temporal sequence discrimination -- tasks where the order or timing of events matters. rate coding requires integrating over many spikes (reducing temporal resolution to ~100ms windows), while temporal coding can discriminate events separated by < 1ms.

the strongest evidence comes from auditory processing, where interaural time differences of ~10 microseconds are discriminated -- impossible with rate coding but natural with timing-based codes. in the visual system, first-spike latency codes (see [[gpu_spike_implementation_research|ttfs]]) achieve classification in a single feedforward pass with one spike per neuron.

relevance to neural machine: todorov uses ternary spikes, which are rate-coded (the value {-1, 0, +1} at each timestep conveys 1.58 bits). there is no spike timing information -- the ternary value is computed independently at each position. if temporal sequence discrimination becomes important (e.g., for phase 4 multimodal or temporal reasoning), adding timing information to the spike code could increase information capacity per spike. however, this would require the infrastructure described in [[gpu_spike_implementation_research]] (temporal integration, parallel scan for spike dynamics).

confidence: high for the theoretical result. the practical relevance to todorov is limited at current firing rates.

## synthesis

the research converges on the following picture:

1. **phase coding is real but not for sequence order.** biological theta-gamma coupling maintains working memory items via phase-locked activity, but the phase offset does not encode sequential order (liebe et al. 2025). phase codes information about WHAT is maintained, not WHERE it falls in a sequence.

2. **oscillatory dynamics emerge as optimal memory maintenance.** trained rnns spontaneously develop limit cycles for working memory (pals et al. 2024). linoss shows that parameterizing recurrence as forced oscillators outperforms non-oscillatory alternatives (rusch & rus 2025). this supports mamba3's complex rotation as a design choice.

3. **rope is structurally a phase code but functionally different.** rope uses the same mathematical form as theta-gamma coupling (multi-frequency phase assignment) but for a different purpose (encoding position for attention computation vs maintaining items in working memory). the analogy is real at the mathematical level but does not imply shared biological mechanism.

4. **spike timing beats rate coding for temporal tasks.** todorov's ternary spikes are rate-coded and sacrifice temporal resolution. this is acceptable for language modeling but may limit performance on temporal reasoning or multimodal tasks where event timing matters.

## see also

- [[theta_oscillations]]
- [[gamma_oscillations]]
- [[neural_synchrony]]
- [[oscillations_to_mamba3_rotation]]
- [[positional_encoding_to_rope]]
- [[gpu_spike_implementation_research]]
