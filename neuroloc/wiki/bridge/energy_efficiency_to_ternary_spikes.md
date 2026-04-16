# bridge: energy efficiency to ternary spike energy savings

status: current (as of 2026-04-16).

## the biological mechanism

the brain is severely energy-limited. the [[brain_energy_budget]] (attwell and laughlin 2001, updated by howarth et al. 2012 and harris et al. 2012) establishes that neural signaling consumes the majority of the brain's ~20 W power budget, with synaptic transmission accounting for ~43% and action potentials ~17% of signaling energy.

the cost per synaptic event is ~10 fJ (harris et al. 2012). the cost per action potential is ~19-120 pJ depending on neuron type and axonal extent (attwell and laughlin 2001, lennie 2003). these costs are dominated by ion pumping: restoring the Na+/K+ gradients that neural signaling dissipates.

the [[metabolic_constraints_on_computation]] (lennie 2003) show that fewer than 1% of cortical neurons can be substantially active at any time. the [[energy_efficient_coding]] framework (levy and baxter 1996, niven and laughlin 2008) shows that the optimal firing rate -- maximizing bits per joule -- is ~6%, remarkably close to observed cortical sparsity of 2-10%.

the biological conclusion: sparse coding is not a design choice but a metabolic necessity. energy constraints force the brain to compute with few active neurons, each carrying high information content per spike.

## the current todorov implementation

### ternary spikes (src/spikes/ternary_spike.py)

the AdaptiveTernarySpike module:
- computes threshold = alpha * mean(|x|), where alpha is a learnable parameter (init 1.0)
- produces output: sign(x) * [|x| > threshold] -> {-1, 0, +1}^d
- backward pass: straight-through estimator (STE), gradient passes through as identity
- firing rate at alpha=1.0: ~41% (59% of dimensions are zero)
- information content: 1.58 bits/dimension (log2(3)) vs 32 bits for FP32

### energy claim: the 354x reduction

the claim: a ternary spike MAC costs ~0.013 pJ at 45nm vs ~4.6 pJ for FP32 MAC (horowitz 2014), yielding a 354x energy reduction.

the mechanism: FP32 multiplication requires a 24-bit mantissa multiplier (24x24 array of full adders = ~576 gates in the critical path). ternary multiplication is a 2:1 multiplexer: select between +weight, 0, or -weight based on the 2-bit spike value. the MUX is ~2-3 transistors. the conditional add (when spike != 0) uses a standard adder gated by a nonzero flag. total: ~0.013 pJ at 45nm.

### where ternary spikes are applied

in the todorov architecture (267M params, 24 layers):
- KDA layers (18/24 layers): K and V projections are spiked (2 out of 4 projections: K, V, Q, O)
- MLA layers (3/24 layers): no ternary spikes (full-precision dot-product attention)
- SwiGLU FFN (all 24 layers): no ternary spikes (gate, up, down projections in FP16/FP32)

the STE training constraint: ternary quantization is applied in the forward pass, but gradients flow through the STE as if quantization were the identity. all weight updates and gradient computations use full precision. the energy saving from ternary spikes exists ONLY at inference time.

## adversarial analysis: does the 354x claim hold?

### per-operation: YES, across all process nodes

see [[biological_vs_silicon_energy]] for the full table. the ternary-vs-FP32 MAC energy ratio remains in the 200-350x range from 45nm to 3nm because the fundamental asymmetry (O(n^2) multiplier gates vs O(1) MUX gates) is preserved by uniform gate scaling.

### system-level: NO, the claim dramatically overstates savings

**problem 1: data movement dominates.**

at 45nm, DRAM read = 640 pJ per 64-bit access. at 5nm with HBM, DRAM read = ~10-15 pJ per access. both dwarf the compute energy. for a 267M-param model where weights must be read from memory for each token:
- weight read energy: ~2.1 mJ per token (FP16 weights from HBM at 5nm)
- compute energy: ~40 nJ per token (all paths combined)
- ratio: data movement is ~50,000x more expensive than compute

ternary spikes save compute energy but not weight read energy. the system-level saving is negligible when data movement dominates.

**problem 2: only ~15-25% of MACs are in ternary paths.**

KDA K/V projections represent ~2/4 of attention projections in 18/24 layers. the FFN (larger than attention) and MLA layers use full precision. estimated fraction of total MACs in ternary paths: 15-25%.

**problem 3: sparsity and ternary are not multiplicative in the way claimed.**

the 354x ratio counts energy per non-zero ternary MAC vs energy per FP32 MAC. with 41% firing rate, 59% of ternary operations are zero (free). the combined advantage for spiked dimensions is:

    0.41 * 0.013 pJ = 0.00533 pJ per dimension (ternary with sparsity)
    vs 4.6 pJ per dimension (dense FP32)
    = ~863x per dimension

but this compares a sparse ternary vector against a dense FP32 vector. a fairer comparison against FP16 at 5nm:

    0.41 * 0.001 pJ = 0.00041 pJ per dimension (ternary at 5nm with sparsity)
    vs 0.1 pJ per dimension (FP16 at 5nm, dense)
    = ~244x per dimension

the advantage is real but only for spiked paths.

**problem 4: the STE means training energy is unchanged.**

ternary spikes save zero training energy. the STE backward pass computes gradients as if activations were continuous. weight updates use full-precision arithmetic. the energy saving is inference-only.

### what fraction of todorov's total inference energy is saved?

rough estimate for 267M params at 5nm:

| component | MACs | energy/MAC | total | ternary savings |
|---|---|---|---|---|
| KDA K,V (spiked) | ~1 x 10^8 | 0.00041 pJ* | 41 pJ | saves ~9,959 pJ vs FP16 |
| KDA Q,O (FP16) | ~1 x 10^8 | 0.1 pJ | 10,000 pJ | 0 |
| MLA (FP16) | ~0.5 x 10^8 | 0.1 pJ | 5,000 pJ | 0 |
| FFN (FP16) | ~2.5 x 10^8 | 0.1 pJ | 25,000 pJ | 0 |
| data movement | -- | -- | ~2,100,000,000 pJ (~2.1 mJ) | 0** |

*with 41% sparsity at 5nm
**ternary activations can be compressed 8x (2 bits vs 16 bits), saving ~1-5% of activation bandwidth

total compute energy without ternary: ~50,000 pJ
total compute energy with ternary: ~40,041 pJ
compute savings: ~20%
total inference energy savings (including data movement): ~0.0005% (compute is ~50,000x smaller than data movement)

the honest answer: ternary spikes save ~20% of compute energy and ~0.0005% of total inference energy on current GPU hardware. the 354x per-operation advantage is swamped by data movement.

### where ternary spikes WOULD matter

1. **dedicated ternary accelerators**: hardware designed to exploit ternary sparsity (e.g., RISC-V ternary accelerators achieving 63 pJ/MAC system-level vs ~1-5 nJ/MAC for GPU FP16). on such hardware, ternary spikes could reduce system-level inference energy by 10-50x.

2. **edge inference**: on memory-limited devices where weights fit in SRAM, data movement costs drop by 100x, making compute energy a larger fraction of total. ternary spikes then provide meaningful system-level savings.

3. **activation compression**: with 41% sparsity and ternary values, activations can be compressed to ~1.5 bits/dimension (entropy coding) vs 16 bits for FP16. this 10x activation bandwidth reduction helps with long sequences where activation memory dominates.

4. **if spikes are expanded to all projections** (spike_all_projections=True): spiking K, V, Q, O, gate, up would increase the fraction of ternary MACs to ~60-70%, making the compute savings ~60-65% instead of ~20%.

## comparison with biological energy efficiency

| metric | biology | todorov ternary | ratio |
|---|---|---|---|
| energy per operation | ~10 fJ/synapse | ~1 fJ/ternary MAC (5nm) | biology 10x more expensive |
| sparsity | 2-10% active | 41% active | biology 4-20x sparser |
| operations per second | ~10^15 synaptic events/s | ~10^12 MACs/s (GPU) | biology 1000x more parallel |
| total power | ~20 W | ~300-700 W (GPU) | biology 15-35x more efficient |
| data movement | ~0 (in-memory compute) | ~2 mJ/token (HBM) | biology fundamentally wins |

the brain's energy advantage is NOT at the per-operation level (silicon has surpassed it). the advantage is:
1. massive parallelism (86 billion neurons operating simultaneously)
2. in-memory computation (weights stored at the synapse, no data movement)
3. extreme sparsity (1-5% vs todorov's 41%)
4. local wiring (most connections < 1 mm)

todorov captures ONE of these four advantages (ternary sparsity, partially) but not the other three. the energy efficiency analogy to biology is therefore weak at the system level.

## the proposed change

### what would make the energy claim real

to achieve meaningful system-level energy savings from ternary spikes, todorov would need:

1. **expand spike coverage**: apply ternary spikes to all projections (K, V, Q, O, gate, up) to increase ternary fraction from 15-25% to 60-70% of compute. this is spike_all_projections=True in config, already implemented but not validated.

2. **reduce firing rate**: move from 41% to 15-20% to increase sparsity savings. this requires either ATMN per-neuron thresholds or a sparsity schedule (see [[sparse_coding_to_ternary_spikes]]).

3. **target ternary-aware hardware**: the energy advantage is stranded on GPUs. deploying on dedicated ternary accelerators (FPGA with ternary compute, RISC-V with ternary ISA extensions) would realize the per-operation advantage at the system level.

4. **activation compression**: implement entropy coding of ternary activations for reduced memory bandwidth. with 41% firing rate, the entropy is ~1.5 bits/dimension, achievable with simple run-length or Huffman coding.

### risk assessment

| change | expected benefit | risk |
|---|---|---|
| spike_all_projections | 3x more ternary MACs | may hurt BPB if FFN projections lose precision |
| reduce firing to 15-20% | 2-3x more sparsity savings | gradient flow through STE may break (see [[sparse_coding_to_ternary_spikes]]) |
| ternary hardware deployment | 10-50x system-level savings | requires non-GPU hardware, limits deployment |
| activation compression | 5-10x activation bandwidth reduction | encoding/decoding overhead, latency increase |

### recommendation

the energy efficiency claim should be stated precisely:
- CORRECT: "ternary spike MACs use 200-350x less compute energy than FP32 MACs across all CMOS process nodes"
- MISLEADING: "todorov uses 354x less energy than a transformer"
- HONEST: "ternary spikes save ~20% of compute energy and ~0.0005% of total inference energy on GPU hardware. on dedicated ternary hardware, savings could reach 10-50x system-level."

the system-level energy benefit requires either (a) dedicated hardware or (b) expanding spike coverage + reducing firing rate. neither is currently validated at scale. phase 5b (spike_all_projections) is the next step toward validating (b).

## see also

- [[brain_energy_budget]]
- [[energy_efficient_coding]]
- [[metabolic_constraints_on_computation]]
- [[biological_vs_silicon_energy]]
- [[sparse_coding_to_ternary_spikes]]
- [[attwell_laughlin]]
- [[niven_laughlin]]
