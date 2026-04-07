# energy comparison simulation

## what this does

energy_comparison.py computes and plots energy per operation for four arithmetic precisions (FP32, FP16, INT8, ternary MAC) across six CMOS process nodes (45nm, 28nm, 16nm, 7nm, 5nm, 3nm), compared against the biological synapse energy (~10 fJ per synaptic event, harris et al. 2012).

this is a numerical analysis, not a hardware simulation. the values are derived from:
- horowitz 2014 (ISSCC) for 45nm baseline measurements
- published scaling trends and ISSCC papers for 7nm-3nm estimates
- attwell and laughlin 2001, harris et al. 2012 for biological synapse energy

the script also computes a system-level energy breakdown for todorov at 267M params, showing the relative contribution of compute energy vs data movement energy under four scenarios: no spikes, current spike placement (K,V only), expanded spike placement (all projections), and expanded placement with reduced firing rate.

## outputs

- `energy_comparison_operations.png`: energy per MAC across process nodes (log scale) + ternary advantage ratio bar chart
- `energy_comparison_overview.png`: horizontal bar chart comparing all operation types including biological synapse, SRAM, and DRAM
- `energy_comparison_system.png`: system-level energy breakdown for todorov 267M, compute vs data movement

## how to run

```
cd <project_root>
python neuroloc/simulations/energy/energy_comparison.py
```

requires numpy and matplotlib. output images saved to the same directory.

## key findings

the ternary MAC advantage over FP32 remains 200-350x across all process nodes (45nm to 3nm). the ratio does not converge to zero at modern nodes because the fundamental asymmetry (O(n^2) multiplier gates vs O(1) MUX gates) is preserved by uniform transistor scaling.

however, the system-level energy saving for todorov on GPU hardware is <1%, because data movement (weight reads from HBM) dominates total inference energy by ~50,000x over compute energy. the ternary advantage is real at the arithmetic level but stranded on hardware designed for dense FP16/FP32 computation.

silicon ternary MAC energy surpassed biological synapse energy at ~16nm. at 5nm, a ternary MAC (~1 fJ) is ~10x more energy-efficient than a biological synapse (~10 fJ). the brain's system-level advantage comes from massive parallelism, in-memory computation, and extreme sparsity -- not per-operation efficiency.

## connection to todorov

see [[energy_efficiency_to_ternary_spikes]] for the full bridge analysis. the simulation quantifies the adversarial finding: the 354x per-operation claim is correct but the system-level claim is misleading. real energy savings require either dedicated ternary hardware or expanded spike coverage with reduced firing rate.
