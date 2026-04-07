# biological vs silicon energy

## the comparison

every claim about "brain-like efficiency" or "biologically inspired energy savings" ultimately rests on a comparison between biological neural computation and silicon digital computation. this article presents the numbers as honestly as possible, including the caveats that make direct comparison problematic.

## energy per operation: the table

all values are approximate. biological values from attwell and laughlin (2001), harris et al. (2012), lennie (2003). silicon values at 45nm from horowitz (2014). modern node values from ISSCC papers and industry estimates (2024-2025).

### biological operations

| operation | energy | source |
|---|---|---|
| single synapse activation | ~10 fJ | harris et al. 2012 |
| single synapse (attwell & laughlin estimate) | ~8.2 fJ | attwell & laughlin 2001 (~1.64 x 10^5 ATP) |
| action potential per cortical neuron | ~19-120 pJ | attwell & laughlin 2001 / lennie 2003 |
| resting potential per neuron per second | ~0.5-2 pJ | derived from attwell & laughlin 2001 |

### silicon operations at 45nm (Horowitz 2014)

| operation | energy (pJ) |
|---|---|
| 8-bit integer add | 0.03 |
| 8-bit integer multiply | 0.2 |
| 16-bit FP add | 0.4 |
| 16-bit FP multiply | 1.1 |
| 32-bit integer add | 0.1 |
| 32-bit integer multiply | 3.1 |
| 32-bit FP add | 0.9 |
| 32-bit FP multiply | 3.7 |
| 32-bit FP MAC | 4.6 |
| 8 KB SRAM read | 5 |
| 32 KB SRAM read | 10 |
| 1 MB SRAM read | 20 |
| DRAM read (64-bit) | 640 |

### silicon operations at modern nodes (estimates, 2024-2025)

process node scaling reduces dynamic energy roughly proportional to capacitance reduction, which tracks area scaling. however, leakage current increases at smaller nodes, and interconnect energy does not scale well.

| operation | 45nm | 7nm | 5nm | 3nm |
|---|---|---|---|---|
| FP32 MAC | 4.6 pJ | ~0.5 pJ | ~0.3 pJ | ~0.15-0.2 pJ |
| FP16 MAC | 1.5 pJ | ~0.15 pJ | ~0.1 pJ | ~0.05-0.08 pJ |
| INT8 MAC | 0.23 pJ | ~0.03 pJ | ~0.02 pJ | ~0.01-0.015 pJ |
| ternary MAC (MUX + conditional add) | ~0.013 pJ | ~0.002 pJ | ~0.001 pJ | ~0.0007 pJ |
| 1 MB SRAM read | 20 pJ | ~5 pJ | ~3 pJ | ~2 pJ |
| DRAM read (64-bit) | 640 pJ | ~15-20 pJ | ~10-15 pJ | ~8-12 pJ |

note: ternary MAC energy is estimated. a ternary multiply is a 2-input MUX (select between +weight, 0, -weight) followed by a conditional add. the MUX is ~2-3 transistors; the add is a standard adder gated by a nonzero check. at 45nm, this is roughly 0.013 pJ. scaling to modern nodes is approximately proportional to V^2 * C_gate scaling, yielding ~0.001 pJ at 5nm.

the DRAM values at modern nodes are for HBM/LPDDR5, which have lower per-access energy than traditional DDR4 at 45nm. the data movement problem has improved dramatically with on-die memory and high-bandwidth memory stacks.

## the ternary advantage: does it hold?

### at 45nm

- FP32 MAC: 4.6 pJ
- ternary MAC: ~0.013 pJ
- ratio: ~354x

this is the frequently cited number. a ternary multiply-accumulate uses ~354x less energy than an FP32 MAC at 45nm. the reason: FP32 multiplication requires a 24-bit mantissa multiplier (24x24 = 576 full adders), while ternary multiplication is a 2:1 MUX (2-3 transistors).

### at 7nm

- FP32 MAC: ~0.5 pJ
- ternary MAC: ~0.002 pJ
- ratio: ~250x

the ratio shrinks because both operations scale, but FP32 benefits from better transistor efficiency at smaller nodes. the multiplier's energy scales with transistor capacitance, which decreases faster than MUX energy at smaller nodes because the multiplier has more internal nodes that benefit from scaling.

### at 5nm

- FP32 MAC: ~0.3 pJ
- ternary MAC: ~0.001 pJ
- ratio: ~300x

### at 3nm

- FP32 MAC: ~0.15-0.2 pJ
- ternary MAC: ~0.0007 pJ
- ratio: ~215-285x

the ratio remains in the 200-350x range across all process nodes. the ternary advantage does NOT converge to zero at modern nodes. this is because the fundamental asymmetry persists: a multiplier has O(n^2) gates for n-bit operands, while a ternary select is O(1) gates. scaling shrinks all gates equally, preserving the ratio.

### but: the operation-level comparison is misleading

the 354x number applies ONLY to the arithmetic operation itself. it ignores:

**1. data movement dominates total energy.**

at 45nm, a DRAM read costs 640 pJ -- 139x more than an FP32 MAC and 49,000x more than a ternary MAC. the ternary operation saves 4.587 pJ per MAC, but the data movement cost is the same: you still need to read the weight from memory.

at 5nm with HBM, a DRAM read costs ~10-15 pJ -- still 30-50x more than an FP32 MAC. the compute energy is a minority of total energy. making compute cheaper via ternary quantization gives diminishing returns as data movement dominates.

**2. ternary spikes only cover part of the compute.**

in todorov (267M params), ternary spikes are applied to K and V projections in KDA layers (and optionally Q and O). this is 2-4 out of ~8 major matrix multiplications per layer (K, V, Q, O for attention; gate, up, down for FFN). the FFN -- which is larger than the attention -- remains in FP32/FP16.

estimated fraction of total MACs in ternary paths: ~15-25%.

**3. training still uses FP32.**

the STE (straight-through estimator) passes gradients through quantization as if it were the identity. the forward pass produces ternary activations, but the backward pass and weight updates use full-precision arithmetic. the energy saving is inference-only.

**4. sparsity does not multiply with the ternary advantage.**

with 41% firing rate, 59% of ternary values are zero (no MAC needed). this gives a ~2.4x speedup from sparsity. but the 354x advantage already counts only the non-zero operations. the total advantage for spiked paths is: (fraction non-zero) * (ternary MAC energy) vs (FP32 MAC energy). for 41% sparsity:

    effective energy per dimension = 0.41 * 0.013 pJ = 0.00533 pJ (ternary)
    vs 1.0 * 4.6 pJ = 4.6 pJ (dense FP32)
    ratio: ~863x

but this comparison is unfair in the other direction: it assumes the FP32 baseline has no sparsity. a fair comparison would be ternary spikes at 41% sparsity vs FP16 with no sparsity at 5nm:

    ternary at 5nm: 0.41 * 0.001 pJ = 0.00041 pJ per dimension
    FP16 at 5nm: 0.1 pJ per dimension
    ratio: ~244x

## the honest system-level assessment

for todorov at 267M params, inference on a modern GPU (5nm):

total MACs per token: ~5 x 10^8 (rough estimate for 267M params, single forward pass)
- MACs in ternary paths (~20% of total): ~1 x 10^8
- MACs in FP16 paths (~80% of total): ~4 x 10^8

compute energy per token:
- ternary paths: 1 x 10^8 * 0.00041 pJ = 41 pJ (with 41% sparsity at 5nm)
- FP16 paths: 4 x 10^8 * 0.1 pJ = 40,000 pJ = 40 nJ

total compute energy: ~40 nJ per token

data movement energy per token (assuming ~267M params read from HBM):
- at FP16: 267M * 2 bytes = 534 MB
- HBM read energy: ~3-5 pJ per byte
- total: 534 x 10^6 * 4 pJ = ~2.1 x 10^9 pJ = ~2.1 mJ

the compute energy (40 nJ) is 50,000x less than the data movement energy (2.1 mJ). ternary spikes save ~99.9% of the compute in spiked paths, but compute is only ~0.002% of total energy. the system-level energy saving from ternary spikes is negligible when weight data movement dominates.

ternary spikes help with activation data movement: a ternary activation vector can be stored in 2 bits per element instead of 16 bits (8x compression). this reduces activation memory bandwidth, which matters for long sequences. but weight reads dominate over activation reads for typical inference.

## comparison with biology

the brain's ~10 fJ per synapse per spike is:
- 460x more efficient than FP32 MAC at 45nm (4.6 pJ)
- 30x more efficient than FP32 MAC at 5nm (0.3 pJ)
- 10x more efficient than FP16 MAC at 5nm (~0.1 pJ but with lower precision)
- 10x MORE expensive than ternary MAC at 5nm (~0.001 pJ)

silicon at 5nm has surpassed biological energy efficiency for ternary operations and is approaching it for full-precision operations. the brain's advantage is no longer at the per-operation level -- it is at the system level (20 W for 86 billion neurons vs 300-700 W for ~10^10 MACs/s on a GPU).

the brain achieves system-level efficiency through:
1. extreme sparsity (1-5% active neurons) reducing total operations
2. local wiring (most connections are within <1 mm, minimizing data movement)
3. in-memory computation (synaptic weights are stored at the site of computation)
4. analog signaling within dendrites (no A/D conversion overhead)

silicon's equivalent strategies: sparse computation, near-memory computing, compute-in-memory architectures, analog/mixed-signal accelerators. these are all active research areas aiming to capture the brain's system-level advantages.

## verdict

the 354x energy claim for ternary spikes is CORRECT at the per-operation level and holds (200-350x) across all process nodes from 45nm to 3nm. but it is MISLEADING as a system-level claim because:

1. compute energy is a small fraction of total inference energy
2. ternary spikes cover only ~15-25% of todorov's total compute
3. the energy saving is inference-only (training uses FP32)
4. data movement dominates at all process nodes

the honest claim: ternary spikes reduce compute energy by ~200-350x for the spiked paths, which reduces total compute energy by ~15-20%, which reduces total inference energy by ~0.0005% when data movement dominates (compute is ~50,000x smaller than data movement at 5nm with HBM).

where ternary spikes DO provide real system-level benefits:
- activation compression (2 bits vs 16 bits = 8x less activation bandwidth)
- sparse acceleration (skip 59% of operations in spiked paths)
- dedicated ternary hardware (e.g., RISC-V ternary accelerators achieve 63 pJ/MAC system-level, vs ~1-5 nJ/MAC for GPU FP16)

the per-operation advantage is real. the system-level advantage depends entirely on hardware that exploits it.

## key references

- horowitz, m. (2014). computing's energy problem (and what we can do about it). ISSCC digest of technical papers, 10-14.
- attwell, d. & laughlin, s. b. (2001). an energy budget for signaling in the grey matter of the brain. journal of cerebral blood flow and metabolism, 21, 1133-1145.
- harris, j. j., jolivet, r. & attwell, d. (2012). synaptic energy use and supply. neuron, 75(5), 762-777.
- han, s., mao, h. & dally, w. j. (2016). deep compression: compressing deep neural networks with pruning, trained quantization and huffman coding. ICLR.
- zhu, s. et al. (2022). xTern: energy-efficient ternary neural network inference on RISC-V-based edge systems.
- sze, v. et al. (2017). efficient processing of deep neural networks: a tutorial and survey. proceedings of the IEEE, 105(12), 2295-2329.

## see also

- [[brain_energy_budget]]
- [[metabolic_constraints_on_computation]]
- [[energy_efficient_coding]]
- [[sparse_coding_to_ternary_spikes]]
- [[energy_efficiency_to_ternary_spikes]]
