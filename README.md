# Todorov

a neural architecture built on one mathematical object: the **compressed
rotational bilinear recurrence (CRBR)**. every layer instantiates
`z_t = Q(R(B(C(x_t), C(h_{t-1}))))` -- compression, bilinear interaction,
rotational structure, and quantization -- with different parameterizations
drawn from four composable families. the biological constraints (ternary
spikes, adaptive thresholds, recurrent state accumulation) are not
approximations of standard transformers. they are the architecture.

todorov is a proof-of-concept for a neural computer. the architecture
demonstrates that biological computation principles -- outer-product
associative memory, exponential decay, threshold quantization, rotational
position encoding -- outperform standard transformers at matched scale.
the long-term goal is a general-purpose neural substrate that computes
through these dynamics on standard hardware (GPU/CPU/NPU).

## latest results

267M parameters, byte-level, FineWeb-Edu.

BPB ratio vs matched transformer at 267M: **0.663x** (33.7% better)
spike mutual information: **1.168** (threshold 0.1)
spike CKA similarity: **0.732** (threshold 0.3)
spike firing rate: **40.8%** (target 30-60%)

the architecture advantage grows with scale: 16% better at 6M, 34%
better at 267M. ternary spikes preserve information across a 45x scale
increase with zero dead neurons. the geometric product spatial module
improves n-body dynamics prediction by 29% at zero compute overhead.

## neuroloc

neuroloc (`neuroloc/`) is the biological grounding research phase.
it maps 61 biological neural computation mechanisms to todorov's CRBR
framework through an obsidian wiki (185 articles), 28 simulations,
and a 32-page printed guide.

key finding: 14 of 15 biological analogies in the architecture are
superficial. the one genuine mathematical correspondence is the
outer-product associative memory (k * v^T = hebbian learning). the
architecture works not because it faithfully reproduces biology, but
because the biological constraints happen to be independently useful
engineering choices.

top 5 biologically-motivated interventions identified, ranked by
expected impact: ATMN leak term, activity-dependent forgetting (BCM-like),
k-WTA competitive selection, progressive spike activation, neuromodulator
network. the k-WTA pilot shows 0.925 exact support recovery vs 0.500
for threshold spiking (p=0.001).

## the CRBR framework

four families, one equation.

**compression (C).** every path through todorov encounters an information
bottleneck. ternary spike quantization maps activations to {-1, 0, +1}
at 1.58 bits per dimension. low-rank latent projection compresses key-value
representations to a learned sufficient statistic. gated activation creates
continuous-valued sparsity. these are all instances of one family: projection
to a lower-dimensional representation, with the dimensionality controlling
the fidelity tradeoff.

**bilinear interaction (B).** every layer's core computation is a bilinear
map. the outer product writes key-value associations into recurrent state.
the inner product retrieves by similarity. element-wise gating selectively
amplifies features. the G(3,0,1) geometric product simultaneously computes
scalar, vector, area, volume, and pseudoscalar products between multivectors
-- the most general bilinear map in the framework, subsuming dot product
and gating as grade restrictions.

**rotation (R).** rotational geometry at every scale. RoPE encodes position
via SO(2) rotations. complex-valued state dynamics evolve via data-dependent
SO(2) rotations. the geometric product produces Pin(3,0,1) rotors encoding
spatial transformations. these form an algebraically nested hierarchy:
SO(2) inside Spin(3) inside Pin(3,0,1). each layer uses the smallest
rotation group that serves its function.

**quantization (Q).** output discretization from ternary spikes (training
time) through INT8 weight compression (deployment time). rotation-based
quantization methods pre-multiply activations by learned rotations that
align the data distribution with the quantization grid -- the rotational
family R applied inside the compression family C.

## architecture

three CRBR parameterizations, one schedule:

**delta-rule layers** (18 of 24). matrix-valued recurrent state S updated
via hebbian outer-product writes: S_t = decay * S_{t-1} + gate * k_t v_t^T.
O(1) inference memory. ternary spikes on key/value paths. note: the current
implementation does not include targeted erasure -- forgetting is via
exponential decay only.

**continuous-dynamics layers** (3 of 24). vector-valued state evolved via
discretized differential equations with data-dependent selection. complex
rotation encodes temporal phase.

**latent-attention layers** (3 of 24). full context compressed to low-rank
per-token representations. exact softmax retrieval over compressed cache.
O(t) memory, O(t^2) compute -- the precise complement to recurrent layers.

**spatial module.** G(3,0,1) projective geometric algebra as additive
self-interaction in the gated feed-forward network. zero measurable compute
overhead. 29% improvement on spatial reasoning tasks.

schedule: (delta, delta, delta, dynamics, delta, delta, delta, attention) x 3.

## status

11 training runs completed across 5 phases. the architecture scales and
the biological constraints hold. current work: biological grounding via
neuroloc research, parallel scan kernels for the continuous-dynamics
layers, and convergence at competitive token budgets.

## funding

this research is compute-limited. the architecture is validated but training
at competitive token budgets (100B+ tokens) and scales (1B+ parameters)
requires GPU access beyond what free tiers provide. if you are interested
in funding compute for this work, please reach out to eptesicus laboratories.

eptesicus laboratories.
