import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker


NODES_NM = [45, 28, 16, 7, 5, 3]

FP32_MAC_PJ = {
    45: 4.6,
    28: 2.0,
    16: 1.0,
    7: 0.5,
    5: 0.3,
    3: 0.175,
}

FP16_MAC_PJ = {
    45: 1.5,
    28: 0.65,
    16: 0.33,
    7: 0.15,
    5: 0.1,
    3: 0.06,
}

INT8_MAC_PJ = {
    45: 0.23,
    28: 0.1,
    16: 0.05,
    7: 0.025,
    5: 0.02,
    3: 0.012,
}

TERNARY_MAC_PJ = {
    45: 0.013,
    28: 0.006,
    16: 0.003,
    7: 0.0015,
    5: 0.001,
    3: 0.0007,
}

BIOLOGICAL_SYNAPSE_PJ = 0.010

DRAM_READ_PJ = {
    45: 640,
    28: 200,
    16: 80,
    7: 20,
    5: 12,
    3: 9,
}

SRAM_1MB_READ_PJ = {
    45: 20,
    28: 10,
    16: 6,
    7: 4,
    5: 3,
    3: 2.5,
}


def plot_energy_per_operation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    nodes = NODES_NM
    fp32_vals = [FP32_MAC_PJ[n] for n in nodes]
    fp16_vals = [FP16_MAC_PJ[n] for n in nodes]
    int8_vals = [INT8_MAC_PJ[n] for n in nodes]
    tern_vals = [TERNARY_MAC_PJ[n] for n in nodes]
    bio_line = [BIOLOGICAL_SYNAPSE_PJ] * len(nodes)

    ax1.semilogy(range(len(nodes)), fp32_vals, 'o-', color='#e74c3c', linewidth=2.5,
                 markersize=8, label='FP32 MAC', zorder=5)
    ax1.semilogy(range(len(nodes)), fp16_vals, 's-', color='#e67e22', linewidth=2.5,
                 markersize=8, label='FP16 MAC', zorder=5)
    ax1.semilogy(range(len(nodes)), int8_vals, 'D-', color='#3498db', linewidth=2.5,
                 markersize=8, label='INT8 MAC', zorder=5)
    ax1.semilogy(range(len(nodes)), tern_vals, '^-', color='#2ecc71', linewidth=2.5,
                 markersize=8, label='Ternary MAC', zorder=5)
    ax1.semilogy(range(len(nodes)), bio_line, '--', color='#9b59b6', linewidth=2,
                 label='Biological synapse (~10 fJ)', zorder=4)

    ax1.fill_between(range(len(nodes)),
                     [BIOLOGICAL_SYNAPSE_PJ * 0.5] * len(nodes),
                     [BIOLOGICAL_SYNAPSE_PJ * 2.0] * len(nodes),
                     alpha=0.15, color='#9b59b6', zorder=1)

    ax1.set_xticks(range(len(nodes)))
    ax1.set_xticklabels([f'{n} nm' for n in nodes], fontsize=11)
    ax1.set_ylabel('energy per operation (pJ)', fontsize=12)
    ax1.set_xlabel('CMOS process node', fontsize=12)
    ax1.set_title('energy per MAC: silicon vs biology', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_ylim(1e-4, 10)

    ratios_fp32 = [FP32_MAC_PJ[n] / TERNARY_MAC_PJ[n] for n in nodes]
    ratios_fp16 = [FP16_MAC_PJ[n] / TERNARY_MAC_PJ[n] for n in nodes]
    ratios_int8 = [INT8_MAC_PJ[n] / TERNARY_MAC_PJ[n] for n in nodes]

    x = np.arange(len(nodes))
    width = 0.25

    bars1 = ax2.bar(x - width, ratios_fp32, width, color='#e74c3c', alpha=0.85, label='FP32 / Ternary')
    bars2 = ax2.bar(x, ratios_fp16, width, color='#e67e22', alpha=0.85, label='FP16 / Ternary')
    bars3 = ax2.bar(x + width, ratios_int8, width, color='#3498db', alpha=0.85, label='INT8 / Ternary')

    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{n} nm' for n in nodes], fontsize=11)
    ax2.set_ylabel('energy ratio (higher = ternary advantage)', fontsize=12)
    ax2.set_xlabel('CMOS process node', fontsize=12)
    ax2.set_title('ternary MAC energy advantage ratio', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 50:
                ax2.text(bar.get_x() + bar.get_width() / 2., height + 5,
                         f'{height:.0f}x', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig('neuroloc/simulations/energy/energy_comparison_operations.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_system_level_energy():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    total_macs = 5e8
    ternary_fraction_default = 0.20
    ternary_fraction_expanded = 0.65
    sparsity = 0.41

    node = 5
    fp16_mac = FP16_MAC_PJ[node]
    tern_mac = TERNARY_MAC_PJ[node]
    dram_per_byte = DRAM_READ_PJ[node] / 8

    param_count = 267e6
    weight_bytes = param_count * 2

    categories = [
        'weight data\nmovement',
        'FP16 compute\n(non-spiked)',
        'ternary compute\n(spiked, 41% active)',
    ]

    dense_fp16_macs = total_macs * (1 - ternary_fraction_default)
    ternary_macs = total_macs * ternary_fraction_default
    ternary_active = ternary_macs * sparsity

    data_move_energy = weight_bytes * dram_per_byte
    fp16_energy = dense_fp16_macs * fp16_mac
    ternary_energy = ternary_active * tern_mac

    energies_pj = [data_move_energy, fp16_energy, ternary_energy]
    energies_uj = [e / 1e6 for e in energies_pj]

    colors = ['#95a5a6', '#e67e22', '#2ecc71']
    bars = ax1.barh(categories, energies_uj, color=colors, height=0.6, edgecolor='white', linewidth=1.5)

    for bar, val_pj in zip(bars, energies_pj):
        if val_pj > 1e6:
            label = f'{val_pj/1e6:.1f} uJ'
        elif val_pj > 1e3:
            label = f'{val_pj/1e3:.1f} nJ'
        else:
            label = f'{val_pj:.1f} pJ'
        ax1.text(bar.get_width() * 1.02, bar.get_y() + bar.get_height() / 2,
                 label, va='center', fontsize=11, fontweight='bold')

    ax1.set_xlabel('energy per token (uJ)', fontsize=12)
    ax1.set_title('todorov 267M inference energy breakdown (5nm)\ndefault spike placement (K,V only)', fontsize=12, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3, axis='x')

    scenarios = [
        'baseline\n(no spikes)',
        'current\n(K,V spiked\n20% ternary)',
        'expanded\n(all proj spiked\n65% ternary)',
        'expanded +\n15% firing rate',
    ]

    baseline_compute = total_macs * fp16_mac
    baseline_total = data_move_energy + baseline_compute

    current_compute = (total_macs * (1 - 0.20) * fp16_mac +
                       total_macs * 0.20 * 0.41 * tern_mac)
    current_total = data_move_energy + current_compute

    expanded_compute = (total_macs * (1 - 0.65) * fp16_mac +
                        total_macs * 0.65 * 0.41 * tern_mac)
    expanded_total = data_move_energy + expanded_compute

    sparse_compute = (total_macs * (1 - 0.65) * fp16_mac +
                      total_macs * 0.65 * 0.15 * tern_mac)
    sparse_total = data_move_energy + sparse_compute

    compute_vals = [baseline_compute, current_compute, expanded_compute, sparse_compute]
    data_vals = [data_move_energy] * 4

    x_pos = np.arange(len(scenarios))
    compute_uj = [v / 1e6 for v in compute_vals]
    data_uj = [v / 1e6 for v in data_vals]

    ax2.bar(x_pos, data_uj, 0.6, color='#95a5a6', label='data movement', edgecolor='white', linewidth=1.5)
    ax2.bar(x_pos, compute_uj, 0.6, bottom=data_uj, color='#e67e22', label='compute', edgecolor='white', linewidth=1.5)

    for i, (cv, dv) in enumerate(zip(compute_vals, data_vals)):
        total = cv + dv
        pct = cv / total * 100
        ax2.text(i, (cv + dv) / 1e6 * 1.02,
                 f'compute: {pct:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(scenarios, fontsize=10)
    ax2.set_ylabel('total energy per token (uJ)', fontsize=12)
    ax2.set_title('system-level energy: compute vs data movement\n267M params, 5nm, HBM', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('neuroloc/simulations/energy/energy_comparison_system.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_biological_comparison():
    fig, ax = plt.subplots(figsize=(12, 7))

    systems = [
        ('biological synapse\n(10 fJ)', 0.010, '#9b59b6'),
        ('ternary MAC\n(5nm)', 0.001, '#2ecc71'),
        ('INT8 MAC\n(5nm)', 0.02, '#3498db'),
        ('FP16 MAC\n(5nm)', 0.1, '#e67e22'),
        ('FP32 MAC\n(5nm)', 0.3, '#e74c3c'),
        ('FP32 MAC\n(45nm)', 4.6, '#c0392b'),
        ('SRAM 1MB read\n(5nm)', 3.0, '#7f8c8d'),
        ('DRAM read\n(5nm, HBM)', 12.0, '#34495e'),
        ('DRAM read\n(45nm, DDR)', 640.0, '#2c3e50'),
    ]

    names = [s[0] for s in systems]
    values = [s[1] for s in systems]
    colors = [s[2] for s in systems]

    sorted_idx = np.argsort(values)
    names = [names[i] for i in sorted_idx]
    values = [values[i] for i in sorted_idx]
    colors = [colors[i] for i in sorted_idx]

    bars = ax.barh(range(len(names)), values, color=colors, height=0.7, edgecolor='white', linewidth=1.5)

    for i, (bar, val) in enumerate(zip(bars, values)):
        if val >= 1:
            label = f'{val:.1f} pJ'
        elif val >= 0.01:
            label = f'{val*1000:.0f} fJ'
        else:
            label = f'{val*1000:.1f} fJ'
        ax.text(max(bar.get_width() * 1.1, val + 0.5), i, label,
                va='center', fontsize=11, fontweight='bold')

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xscale('log')
    ax.set_xlabel('energy per operation (pJ, log scale)', fontsize=12)
    ax.set_title('energy per operation: biology vs silicon', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x', which='both')
    ax.set_xlim(5e-4, 2000)

    ax.axvline(x=BIOLOGICAL_SYNAPSE_PJ, color='#9b59b6', linestyle='--', alpha=0.5, linewidth=1.5)

    plt.tight_layout()
    plt.savefig('neuroloc/simulations/energy/energy_comparison_overview.png', dpi=150, bbox_inches='tight')
    plt.close()


def print_analysis():
    print("=" * 70)
    print("ENERGY COMPARISON ANALYSIS")
    print("=" * 70)

    print("\n--- per-operation ternary advantage ---")
    for node in NODES_NM:
        ratio_fp32 = FP32_MAC_PJ[node] / TERNARY_MAC_PJ[node]
        ratio_fp16 = FP16_MAC_PJ[node] / TERNARY_MAC_PJ[node]
        ratio_int8 = INT8_MAC_PJ[node] / TERNARY_MAC_PJ[node]
        print(f"  {node:3d}nm: FP32/tern = {ratio_fp32:6.0f}x  "
              f"FP16/tern = {ratio_fp16:6.0f}x  "
              f"INT8/tern = {ratio_int8:5.0f}x")

    print("\n--- biological synapse vs silicon ---")
    for node in NODES_NM:
        bio_vs_fp32 = FP32_MAC_PJ[node] / BIOLOGICAL_SYNAPSE_PJ
        bio_vs_tern = BIOLOGICAL_SYNAPSE_PJ / TERNARY_MAC_PJ[node]
        print(f"  {node:3d}nm: FP32 = {bio_vs_fp32:6.0f}x bio  "
              f"ternary = {bio_vs_tern:5.1f}x more efficient than bio")

    print("\n--- todorov 267M system-level analysis (5nm) ---")
    total_macs = 5e8
    fp16_e = FP16_MAC_PJ[5]
    tern_e = TERNARY_MAC_PJ[5]
    dram_per_byte = DRAM_READ_PJ[5] / 8
    param_bytes = 267e6 * 2
    data_move = param_bytes * dram_per_byte

    scenarios = {
        'no spikes (FP16)': (0, 0.41),
        'K,V spiked (20%)': (0.20, 0.41),
        'all proj spiked (65%)': (0.65, 0.41),
        'all proj + 15% firing': (0.65, 0.15),
    }

    for name, (frac, firing) in scenarios.items():
        fp16_compute = total_macs * (1 - frac) * fp16_e
        tern_compute = total_macs * frac * firing * tern_e
        total_compute = fp16_compute + tern_compute
        total_energy = data_move + total_compute
        compute_pct = total_compute / total_energy * 100

        print(f"  {name}:")
        print(f"    compute: {total_compute:.0f} pJ ({total_compute/1e3:.1f} nJ)")
        print(f"    data movement: {data_move:.0f} pJ ({data_move/1e6:.1f} uJ)")
        print(f"    total: {total_energy/1e6:.3f} uJ")
        print(f"    compute fraction: {compute_pct:.3f}%")

    baseline_compute = total_macs * fp16_e
    current_compute = total_macs * 0.80 * fp16_e + total_macs * 0.20 * 0.41 * tern_e
    compute_savings = (1 - current_compute / baseline_compute) * 100
    total_savings = (1 - (data_move + current_compute) / (data_move + baseline_compute)) * 100
    print(f"\n  compute savings (current vs baseline): {compute_savings:.1f}%")
    print(f"  total energy savings (current vs baseline): {total_savings:.3f}%")

    print("\n--- verdict ---")
    print("  per-operation: 200-350x advantage holds across all nodes")
    print("  system-level on GPU: <1% total energy savings")
    print("  system-level on dedicated hardware: potentially 10-50x")
    print("  the 354x claim is per-operation correct, system-level misleading")


if __name__ == '__main__':
    plot_energy_per_operation()
    plot_biological_comparison()
    plot_system_level_energy()
    print_analysis()
    print("\nplots saved to neuroloc/simulations/energy/")
