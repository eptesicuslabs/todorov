from brian2 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

defaultclock.dt = 0.5*ms

arena_size = 2.0
speed = 0.3
dt_movement = 0.01
total_time = 600.0
n_steps = int(total_time / dt_movement)

np.random.seed(42)

x_pos = np.zeros(n_steps)
y_pos = np.zeros(n_steps)
x_pos[0] = arena_size / 2
y_pos[0] = arena_size / 2
heading = np.random.uniform(0, 2 * np.pi)

for i in range(1, n_steps):
    heading += np.random.normal(0, 0.3)
    dx = speed * np.cos(heading) * dt_movement
    dy = speed * np.sin(heading) * dt_movement
    new_x = x_pos[i-1] + dx
    new_y = y_pos[i-1] + dy

    if new_x < 0 or new_x > arena_size:
        heading = np.pi - heading
        new_x = np.clip(new_x, 0, arena_size)
    if new_y < 0 or new_y > arena_size:
        heading = -heading
        new_y = np.clip(new_y, 0, arena_size)

    x_pos[i] = new_x
    y_pos[i] = new_y

n_vcos = 3
preferred_dirs = np.array([0, 2*np.pi/3, 4*np.pi/3])

spacings = [0.4, 0.56, 0.78]
phase_offsets = [
    [(0.0, 0.0), (0.1, 0.15), (0.2, 0.05)],
    [(0.0, 0.0), (0.05, 0.2), (0.15, 0.1)],
    [(0.0, 0.0), (0.1, 0.1), (0.2, 0.2)],
]

n_cells = sum(len(po) for po in phase_offsets)

spike_positions_x = [[] for _ in range(n_cells)]
spike_positions_y = [[] for _ in range(n_cells)]

cell_idx = 0
for mod_idx, s in enumerate(spacings):
    for phase_x, phase_y in phase_offsets[mod_idx]:
        for i in range(n_steps):
            activation = 0.0
            for k in range(n_vcos):
                proj = (x_pos[i] - phase_x) * np.cos(preferred_dirs[k]) + \
                       (y_pos[i] - phase_y) * np.sin(preferred_dirs[k])
                activation += np.cos(2 * np.pi * proj / s)
            activation /= n_vcos
            threshold = 0.7
            if activation > threshold:
                prob = (activation - threshold) / (1.0 - threshold)
                if np.random.random() < prob * 0.3:
                    spike_positions_x[cell_idx].append(x_pos[i])
                    spike_positions_y[cell_idx].append(y_pos[i])
        cell_idx += 1

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle('oscillatory interference grid cell model', fontsize=16, y=0.98)

cell_idx = 0
for mod_idx in range(3):
    for phase_idx in range(3):
        ax = axes[mod_idx][phase_idx]
        ax.plot(x_pos[::10], y_pos[::10], color='#cccccc', linewidth=0.3, alpha=0.5)
        if len(spike_positions_x[cell_idx]) > 0:
            ax.scatter(
                spike_positions_x[cell_idx],
                spike_positions_y[cell_idx],
                s=3, color='#e74c3c', alpha=0.6
            )
        ax.set_xlim(0, arena_size)
        ax.set_ylim(0, arena_size)
        ax.set_aspect('equal')
        ax.set_title(f'module {mod_idx+1} (s={spacings[mod_idx]:.2f}m), cell {phase_idx+1}',
                     fontsize=10)
        if mod_idx == 2:
            ax.set_xlabel('x (m)')
        if phase_idx == 0:
            ax.set_ylabel('y (m)')
        cell_idx += 1

plt.tight_layout()
output_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(output_dir, 'grid_cell_firing.png'), dpi=150, bbox_inches='tight')
plt.close()

n_bins = 50
fig2, axes2 = plt.subplots(3, 3, figsize=(15, 15))
fig2.suptitle('grid cell rate maps (spikes/occupancy)', fontsize=16, y=0.98)

occupancy, xedges, yedges = np.histogram2d(x_pos, y_pos, bins=n_bins,
                                             range=[[0, arena_size], [0, arena_size]])
occupancy_time = occupancy * dt_movement
occupancy_time[occupancy_time == 0] = 1e-10

cell_idx = 0
for mod_idx in range(3):
    for phase_idx in range(3):
        ax = axes2[mod_idx][phase_idx]
        if len(spike_positions_x[cell_idx]) > 0:
            spike_hist, _, _ = np.histogram2d(
                spike_positions_x[cell_idx],
                spike_positions_y[cell_idx],
                bins=n_bins,
                range=[[0, arena_size], [0, arena_size]]
            )
            rate_map = spike_hist / occupancy_time
            from scipy.ndimage import gaussian_filter
            rate_map_smooth = gaussian_filter(rate_map.T, sigma=1.5)
            im = ax.imshow(rate_map_smooth, origin='lower', cmap='hot',
                          extent=[0, arena_size, 0, arena_size],
                          interpolation='bilinear')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Hz')
        ax.set_xlim(0, arena_size)
        ax.set_ylim(0, arena_size)
        ax.set_aspect('equal')
        ax.set_title(f'module {mod_idx+1} (s={spacings[mod_idx]:.2f}m), cell {phase_idx+1}',
                     fontsize=10)
        if mod_idx == 2:
            ax.set_xlabel('x (m)')
        if phase_idx == 0:
            ax.set_ylabel('y (m)')
        cell_idx += 1

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'grid_cell_rate_maps.png'), dpi=150, bbox_inches='tight')
plt.close()

fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
fig3.suptitle('spatial autocorrelation (hexagonal structure test)', fontsize=14, y=1.02)

cell_idx = 0
for mod_idx in range(3):
    ax = axes3[mod_idx]
    if len(spike_positions_x[cell_idx]) > 0:
        spike_hist, _, _ = np.histogram2d(
            spike_positions_x[cell_idx],
            spike_positions_y[cell_idx],
            bins=n_bins,
            range=[[0, arena_size], [0, arena_size]]
        )
        rate_map = spike_hist / occupancy_time
        from scipy.ndimage import gaussian_filter
        rate_map_smooth = gaussian_filter(rate_map.T, sigma=1.5)

        rate_map_centered = rate_map_smooth - np.mean(rate_map_smooth)
        from scipy.signal import correlate2d
        autocorr = correlate2d(rate_map_centered, rate_map_centered, mode='full')
        autocorr /= np.max(autocorr)

        cx, cy = autocorr.shape[0] // 2, autocorr.shape[1] // 2
        radius = min(cx, cy)
        extent_ac = arena_size * (2 * radius) / n_bins
        im = ax.imshow(autocorr, origin='lower', cmap='jet',
                      extent=[-extent_ac/2, extent_ac/2, -extent_ac/2, extent_ac/2],
                      interpolation='bilinear')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f'module {mod_idx+1} (s={spacings[mod_idx]:.2f}m)', fontsize=12)
        ax.set_xlabel('lag x (m)')
        if mod_idx == 0:
            ax.set_ylabel('lag y (m)')
        ax.set_aspect('equal')
    cell_idx += 1

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'grid_cell_autocorrelation.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f'trajectory: {n_steps} steps, {total_time}s, {n_steps * speed * dt_movement:.1f}m total distance')
print(f'arena: {arena_size}m x {arena_size}m')
print(f'grid modules: {len(spacings)} (spacings: {spacings})')
print(f'cells per module: {[len(po) for po in phase_offsets]}')
print(f'total cells: {n_cells}')
for i in range(n_cells):
    print(f'  cell {i}: {len(spike_positions_x[i])} spikes')
print('output: grid_cell_firing.png, grid_cell_rate_maps.png, grid_cell_autocorrelation.png')
