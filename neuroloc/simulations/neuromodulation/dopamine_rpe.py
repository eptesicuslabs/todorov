import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from brian2 import *

prefs.codegen.target = "numpy"
defaultclock.dt = 0.5 * ms

N_sensory = 20
N_action = 10
N_da = 1

sensory_eqs = """
dv/dt = (-v + I_ext) / (10*ms) : 1
I_ext : 1
"""

action_eqs = """
dv/dt = (-v + I_syn) / (20*ms) : 1
I_syn : 1
"""

da_eqs = """
dv/dt = (-v + rpe_signal) / (5*ms) : 1
rpe_signal : 1
da_level : 1
"""

sensory = NeuronGroup(N_sensory, sensory_eqs, threshold="v > 0.8", reset="v = 0", method="euler")
action = NeuronGroup(N_action, action_eqs, threshold="v > 0.7", reset="v = 0", method="euler")
da_neuron = NeuronGroup(N_da, da_eqs, threshold="v > 0.5", reset="v = 0", method="euler")

syn_sa = Synapses(sensory, action, """
w : 1
eligibility : 1
delig_trace/dt = -elig_trace / (200*ms) : 1 (clock-driven)
""", on_pre="""
I_syn_post += w
elig_trace += 1.0
""")
syn_sa.connect(p=0.5)
syn_sa.w = "0.1 + 0.05 * rand()"

mon_da = StateMonitor(da_neuron, ["v", "da_level"], record=True)
mon_action = SpikeMonitor(action)
mon_sensory = SpikeMonitor(sensory)

n_trials = 60
trial_duration = 300 * ms
iti_duration = 200 * ms

reward_trials = np.zeros(n_trials, dtype=bool)
reward_trials[:20] = True
reward_trials[20:40] = True
reward_trials[40:] = False

cs_onset = 50 * ms
cs_duration = 100 * ms
reward_time = 200 * ms

weight_history = np.zeros((n_trials, len(syn_sa.w)))
rpe_history = np.zeros(n_trials)
da_burst_history = np.zeros(n_trials)
value_estimate = 0.0
learning_rate_critic = 0.05
learning_rate_actor = 0.02

net = Network(sensory, action, da_neuron, syn_sa, mon_da, mon_action, mon_sensory)

for trial in range(n_trials):
    weight_history[trial] = np.array(syn_sa.w)

    sensory.v = 0
    action.v = 0
    da_neuron.v = 0
    da_neuron.da_level = 0
    sensory.I_ext = 0
    action.I_syn = 0
    da_neuron.rpe_signal = 0

    net.run(cs_onset)

    sensory.I_ext = np.zeros(N_sensory)
    cs_pattern = np.random.choice(N_sensory, size=10, replace=False)
    sensory.I_ext[cs_pattern] = 1.5

    net.run(cs_duration)

    sensory.I_ext = 0

    net.run(reward_time - cs_onset - cs_duration)

    reward = 1.0 if reward_trials[trial] else 0.0
    rpe = reward - value_estimate
    rpe_history[trial] = rpe

    da_neuron.rpe_signal = np.clip(rpe * 2.0, -1.0, 2.0)
    if rpe > 0:
        da_neuron.da_level = rpe
    else:
        da_neuron.da_level = rpe * 0.5

    da_burst_history[trial] = float(da_neuron.da_level[0])

    net.run(trial_duration - reward_time)

    value_estimate += learning_rate_critic * rpe

    current_elig = np.array(syn_sa.elig_trace)
    w_update = learning_rate_actor * rpe * current_elig
    new_w = np.array(syn_sa.w) + w_update
    syn_sa.w = np.clip(new_w, 0.01, 1.0)

    syn_sa.elig_trace = 0

    net.run(iti_duration)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.plot(range(n_trials), rpe_history, "k-", linewidth=1.5, label="RPE (delta)")
ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
ax.axvline(x=20, color="red", linestyle=":", linewidth=1, alpha=0.7)
ax.axvline(x=40, color="red", linestyle=":", linewidth=1, alpha=0.7)
ax.set_xlabel("trial")
ax.set_ylabel("reward prediction error")
ax.set_title("dopamine RPE signal")
ax.text(10, max(rpe_history) * 0.8, "learning\n(reward)", ha="center", fontsize=9)
ax.text(30, max(rpe_history) * 0.8, "learned\n(reward)", ha="center", fontsize=9)
ax.text(50, -0.3, "omission\n(no reward)", ha="center", fontsize=9)
ax.legend(loc="upper right", fontsize=9)

ax = axes[0, 1]
ax.plot(range(n_trials), da_burst_history, "b-", linewidth=1.5, label="DA level")
ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
ax.axvline(x=20, color="red", linestyle=":", linewidth=1, alpha=0.7)
ax.axvline(x=40, color="red", linestyle=":", linewidth=1, alpha=0.7)
ax.set_xlabel("trial")
ax.set_ylabel("dopamine level")
ax.set_title("phasic dopamine response")
ax.legend(loc="upper right", fontsize=9)

ax = axes[1, 0]
mean_w = np.mean(weight_history, axis=1)
std_w = np.std(weight_history, axis=1)
ax.plot(range(n_trials), mean_w, "g-", linewidth=1.5, label="mean weight")
ax.fill_between(range(n_trials), mean_w - std_w, mean_w + std_w, alpha=0.2, color="green")
ax.axvline(x=20, color="red", linestyle=":", linewidth=1, alpha=0.7)
ax.axvline(x=40, color="red", linestyle=":", linewidth=1, alpha=0.7)
ax.set_xlabel("trial")
ax.set_ylabel("synaptic weight")
ax.set_title("dopamine-modulated weight evolution")
ax.legend(loc="upper left", fontsize=9)

ax = axes[1, 1]
sorted_final_w = np.sort(weight_history[-1])
sorted_peak_w = np.sort(weight_history[39])
sorted_init_w = np.sort(weight_history[0])
x_pos = np.arange(len(sorted_init_w))
ax.bar(x_pos - 0.25, sorted_init_w, width=0.25, color="gray", alpha=0.7, label="initial (trial 0)")
ax.bar(x_pos, sorted_peak_w, width=0.25, color="green", alpha=0.7, label="peak (trial 40)")
ax.bar(x_pos + 0.25, sorted_final_w, width=0.25, color="red", alpha=0.7, label="final (trial 60)")
ax.set_xlabel("synapse (sorted)")
ax.set_ylabel("weight")
ax.set_title("weight distribution: initial vs peak vs post-omission")
ax.legend(loc="upper left", fontsize=9)

plt.tight_layout()
plt.savefig("dopamine_rpe_demo.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"trials: {n_trials}")
print(f"reward trials: {int(reward_trials.sum())} / {n_trials}")
print(f"initial RPE: {rpe_history[0]:.4f}")
print(f"RPE at trial 20 (learned): {rpe_history[19]:.4f}")
print(f"RPE at trial 40 (omission start): {rpe_history[40]:.4f}")
print(f"final RPE: {rpe_history[-1]:.4f}")
print(f"mean weight initial: {np.mean(weight_history[0]):.4f}")
print(f"mean weight peak: {np.mean(weight_history[39]):.4f}")
print(f"mean weight final: {np.mean(weight_history[-1]):.4f}")
print(f"value estimate final: {value_estimate:.4f}")
print("saved: dopamine_rpe_demo.png")
