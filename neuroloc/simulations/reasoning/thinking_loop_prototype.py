from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from shared import (
    apply_plot_style,
    build_rng,
    build_run_record,
    env_float,
    env_int,
    env_list,
    mean_confidence_interval,
    output_dir_for,
    require_positive,
    require_positive_list,
    require_unit_interval,
    utc_now_iso,
    write_json,
)

SCRIPT_PATH = Path(__file__).resolve()
SEED = env_int("THINK_SEED", 42)
MODULUS = env_int("THINK_MODULUS", 7)
N_OPERATIONS = env_int("THINK_N_OPERATIONS", 3)
D_MODEL = env_int("THINK_D_MODEL", 48)
DEPTH = env_int("THINK_DEPTH", 4)
DEEP_DEPTH = env_int("THINK_DEEP_DEPTH", 12)
N_TRAIN = env_int("THINK_N_TRAIN", 4096)
N_VAL = env_int("THINK_N_VAL", 1024)
N_TEST = env_int("THINK_N_TEST", 1024)
N_EPOCHS = env_int("THINK_N_EPOCHS", 15)
BATCH_SIZE = env_int("THINK_BATCH_SIZE", 128)
LEARNING_RATE = env_float("THINK_LR", 3e-3)
TRAIN_K = env_int("THINK_TRAIN_K", 4)
N_LIST = env_list("THINK_N_LIST", int, [2, 4, 8, 16, 32])
N_TRAIN_CAP = env_int("THINK_N_TRAIN_CAP", 8)
K_LIST = env_list("THINK_K_LIST", int, [1, 2, 4, 8, 16])
N_SEEDS = env_int("THINK_N_SEEDS", 3)
FIXED_POINT_TOL = env_float("THINK_FIXED_POINT_TOL", 1e-3)
CONVERGENCE_K_MAX = env_int("THINK_CONV_K_MAX", 32)

require_positive("THINK_MODULUS", MODULUS)
require_positive("THINK_N_OPERATIONS", N_OPERATIONS)
require_positive("THINK_D_MODEL", D_MODEL)
require_positive("THINK_DEPTH", DEPTH)
require_positive("THINK_DEEP_DEPTH", DEEP_DEPTH)
require_positive("THINK_N_TRAIN", N_TRAIN)
require_positive("THINK_N_VAL", N_VAL)
require_positive("THINK_N_TEST", N_TEST)
require_positive("THINK_N_EPOCHS", N_EPOCHS)
require_positive("THINK_BATCH_SIZE", BATCH_SIZE)
require_positive("THINK_TRAIN_K", TRAIN_K)
require_positive_list("THINK_N_LIST", N_LIST)
require_positive("THINK_N_TRAIN_CAP", N_TRAIN_CAP)
require_positive_list("THINK_K_LIST", K_LIST)
require_positive("THINK_N_SEEDS", N_SEEDS)
require_positive("THINK_CONV_K_MAX", CONVERGENCE_K_MAX)

if not 0.0 < LEARNING_RATE <= 1.0:
    raise ValueError("THINK_LR must be in (0, 1]")
require_unit_interval("THINK_FIXED_POINT_TOL", FIXED_POINT_TOL)


def build_vocab_size(modulus: int, n_operations: int) -> int:
    return 1 + n_operations * modulus


def encode_sequence(operations: np.ndarray, constants: np.ndarray) -> np.ndarray:
    token_ids = np.empty(operations.shape, dtype=np.int64)
    for t in range(operations.shape[1]):
        op = operations[:, t]
        c = constants[:, t]
        token_ids[:, t] = 1 + op * MODULUS + c
    return token_ids


def compute_result(start_value: np.ndarray, operations: np.ndarray, constants: np.ndarray, modulus: int) -> np.ndarray:
    value = start_value.copy().astype(np.int64)
    n = operations.shape[1]
    for t in range(n):
        op = operations[:, t]
        c = constants[:, t]
        next_value = value.copy()
        add_mask = (op == 0)
        sub_mask = (op == 1)
        mul_mask = (op == 2)
        next_value = np.where(add_mask, (value + c) % modulus, next_value)
        next_value = np.where(sub_mask, (value - c) % modulus, next_value)
        next_value = np.where(mul_mask, (value * c) % modulus, next_value)
        if not (add_mask | sub_mask | mul_mask).all():
            raise ValueError(f"operation index out of [0, 3) at step {t}")
        value = next_value
    return value


def generate_task_batch(rng: np.random.Generator, n_samples: int, n_ops: int, modulus: int, n_op_kinds: int) -> tuple[np.ndarray, np.ndarray]:
    operations = rng.integers(0, n_op_kinds, size=(n_samples, n_ops))
    constants = rng.integers(0, modulus, size=(n_samples, n_ops))
    token_ids = encode_sequence(operations, constants)
    start_values = np.zeros(n_samples, dtype=np.int64)
    targets = compute_result(start_values, operations, constants, modulus)
    return token_ids, targets


class ReasoningBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.w_ff1 = nn.Linear(d_model, d_ff)
        self.w_ff2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm1(x)
        qkv = self.w_qkv(h)
        d = x.shape[-1]
        q, k, v = torch.split(qkv, d, dim=-1)
        scale = d ** -0.5
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_logits, dim=-1)
        attended = torch.matmul(attn_weights, v)
        x = residual + self.w_out(attended)
        residual = x
        h = self.norm2(x)
        h = self.w_ff2(F.gelu(self.w_ff1(h)))
        x = residual + h
        return x


class ReasoningModel(nn.Module):
    def __init__(self, vocab_size: int, modulus: int, d_model: int, depth: int, max_seq_len: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.modulus = modulus
        self.d_model = d_model
        self.depth = depth
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len + 1, d_model)
        self.state_slot = nn.Parameter(torch.zeros(1, 1, d_model))
        d_ff = d_model * 2
        self.blocks = nn.ModuleList([ReasoningBlock(d_model, d_ff) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, modulus)

    def _embed(self, tokens: torch.Tensor, state_vec: torch.Tensor | None) -> torch.Tensor:
        batch, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch, -1)
        x = self.token_emb(tokens) + self.pos_emb(positions)
        if state_vec is not None:
            state_pos = self.pos_emb(torch.full((batch, 1), self.max_seq_len, device=tokens.device, dtype=torch.long))
            slot_bias = self.state_slot.expand(batch, -1, -1)
            state_token = state_vec.unsqueeze(1) + state_pos + slot_bias
            x = torch.cat([x, state_token], dim=1)
        return x

    def forward_once(self, tokens: torch.Tensor, state_vec: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._embed(tokens, state_vec)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        summary = x[:, -1, :]
        logits = self.head(summary)
        return logits, summary

    def forward(self, tokens: torch.Tensor, n_think: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
        state_vec: torch.Tensor | None = None
        hidden_trace: list[torch.Tensor] = []
        logits = None
        for _ in range(n_think):
            logits, summary = self.forward_once(tokens, state_vec)
            hidden_trace.append(summary)
            state_vec = summary
        if logits is None:
            raise ValueError("n_think must be >= 1")
        return logits, hidden_trace


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(
    model: ReasoningModel,
    rng: np.random.Generator,
    n_ops_train: int,
    n_train: int,
    n_val: int,
    n_epochs: int,
    batch_size: int,
    learning_rate: float,
    think_steps: int,
    device: torch.device,
) -> dict[str, list[float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_tokens, train_targets = generate_task_batch(rng, n_train, n_ops_train, MODULUS, N_OPERATIONS)
    val_tokens, val_targets = generate_task_batch(rng, n_val, n_ops_train, MODULUS, N_OPERATIONS)
    train_tokens_t = torch.from_numpy(train_tokens).to(device)
    train_targets_t = torch.from_numpy(train_targets).to(device)
    val_tokens_t = torch.from_numpy(val_tokens).to(device)
    val_targets_t = torch.from_numpy(val_targets).to(device)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}
    n_batches = (n_train + batch_size - 1) // batch_size
    for _ in range(n_epochs):
        model.train()
        perm_seed = int(rng.integers(0, 2**31 - 1))
        perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(perm_seed))
        total_loss = 0.0
        for batch_idx in range(n_batches):
            idx = perm[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            tokens_batch = train_tokens_t[idx]
            targets_batch = train_targets_t[idx]
            logits, _ = model(tokens_batch, n_think=think_steps)
            loss = F.cross_entropy(logits, targets_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().item()) * tokens_batch.shape[0]
        history["train_loss"].append(total_loss / n_train)
        model.eval()
        with torch.no_grad():
            logits, _ = model(val_tokens_t, n_think=think_steps)
            val_loss = float(F.cross_entropy(logits, val_targets_t).item())
            pred = logits.argmax(dim=-1)
            val_acc = float((pred == val_targets_t).float().mean().item())
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
    return history


def measure_accuracy(model: ReasoningModel, tokens_t: torch.Tensor, targets_t: torch.Tensor, think_steps: int) -> float:
    model.eval()
    with torch.no_grad():
        logits, _ = model(tokens_t, n_think=think_steps)
        pred = logits.argmax(dim=-1)
        return float((pred == targets_t).float().mean().item())


def measure_convergence(model: ReasoningModel, tokens_t: torch.Tensor, k_max: int, tol: float) -> tuple[float, float]:
    model.eval()
    converged = 0
    final_deltas: list[float] = []
    with torch.no_grad():
        _, trace = model(tokens_t, n_think=k_max)
    for b in range(tokens_t.shape[0]):
        diffs = []
        for t in range(1, len(trace)):
            prev = trace[t - 1][b]
            curr = trace[t][b]
            denom = torch.norm(prev).item() + 1e-8
            delta = float(torch.norm(curr - prev).item() / denom)
            diffs.append(delta)
        if diffs:
            final_deltas.append(float(diffs[-1]))
            if diffs[-1] < tol:
                converged += 1
    fraction = float(converged) / float(tokens_t.shape[0]) if tokens_t.shape[0] > 0 else 0.0
    mean_final_delta = float(np.mean(final_deltas)) if final_deltas else 0.0
    return fraction, mean_final_delta


def run_thinking_cell(
    seed: int,
    n_ops_train: int,
    think_steps_train: int,
    k_sweep_list: list[int],
    n_sweep_list: list[int],
    device: torch.device,
) -> dict:
    torch.manual_seed(seed)
    rng = build_rng(seed)
    max_seq_len = max(n_sweep_list + [n_ops_train])
    vocab_size = build_vocab_size(MODULUS, N_OPERATIONS)
    model = ReasoningModel(
        vocab_size=vocab_size,
        modulus=MODULUS,
        d_model=D_MODEL,
        depth=DEPTH,
        max_seq_len=max_seq_len,
    ).to(device)
    n_params = count_parameters(model)
    history = train_model(
        model=model,
        rng=rng,
        n_ops_train=n_ops_train,
        n_train=N_TRAIN,
        n_val=N_VAL,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        think_steps=think_steps_train,
        device=device,
    )
    results: dict = {"seed": int(seed), "n_params": int(n_params), "history": history, "accuracy": {}, "convergence": {}, "mean_final_delta": {}}
    rng_assess = build_rng(seed + 7919)
    for n_ops in n_sweep_list:
        tokens_test, targets_test = generate_task_batch(rng_assess, N_TEST, n_ops, MODULUS, N_OPERATIONS)
        tokens_t = torch.from_numpy(tokens_test).to(device)
        targets_t = torch.from_numpy(targets_test).to(device)
        per_k = {}
        for k in k_sweep_list:
            acc = measure_accuracy(model, tokens_t, targets_t, think_steps=k)
            per_k[int(k)] = acc
        results["accuracy"][int(n_ops)] = per_k
        fraction, final_delta = measure_convergence(model, tokens_t, k_max=CONVERGENCE_K_MAX, tol=FIXED_POINT_TOL)
        results["convergence"][int(n_ops)] = float(fraction)
        results["mean_final_delta"][int(n_ops)] = float(final_delta)
    return results


def run_deep_baseline_cell(
    seed: int,
    n_ops_train: int,
    n_sweep_list: list[int],
    device: torch.device,
) -> dict:
    torch.manual_seed(seed)
    rng = build_rng(seed)
    max_seq_len = max(n_sweep_list + [n_ops_train])
    vocab_size = build_vocab_size(MODULUS, N_OPERATIONS)
    model = ReasoningModel(
        vocab_size=vocab_size,
        modulus=MODULUS,
        d_model=D_MODEL,
        depth=DEEP_DEPTH,
        max_seq_len=max_seq_len,
    ).to(device)
    n_params = count_parameters(model)
    history = train_model(
        model=model,
        rng=rng,
        n_ops_train=n_ops_train,
        n_train=N_TRAIN,
        n_val=N_VAL,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        think_steps=1,
        device=device,
    )
    results: dict = {"seed": int(seed), "n_params": int(n_params), "history": history, "accuracy": {}}
    rng_assess = build_rng(seed + 7919)
    for n_ops in n_sweep_list:
        tokens_test, targets_test = generate_task_batch(rng_assess, N_TEST, n_ops, MODULUS, N_OPERATIONS)
        tokens_t = torch.from_numpy(tokens_test).to(device)
        targets_t = torch.from_numpy(targets_test).to(device)
        acc = measure_accuracy(model, tokens_t, targets_t, think_steps=1)
        results["accuracy"][int(n_ops)] = float(acc)
    return results


def aggregate_accuracy_by_k_and_n(cells: list[dict]) -> dict:
    structured: dict[str, dict[str, dict]] = {}
    k_values = sorted({int(k) for cell in cells for v in cell["accuracy"].values() for k in v.keys()})
    n_values = sorted({int(n) for cell in cells for n in cell["accuracy"].keys()})
    for k in k_values:
        structured[str(k)] = {}
        for n in n_values:
            per_seed = [float(cell["accuracy"][n][k]) for cell in cells if n in cell["accuracy"] and k in cell["accuracy"][n]]
            summary = mean_confidence_interval(per_seed, bounds=(0.0, 1.0))
            summary["K"] = int(k)
            summary["N"] = int(n)
            structured[str(k)][str(n)] = summary
    return structured


def compute_per_step_gain(cells: list[dict], k_list: list[int]) -> dict:
    result: dict[str, list[dict]] = {}
    n_values = sorted({int(n) for cell in cells for n in cell["accuracy"].keys()})
    sorted_k = sorted(k_list)
    for n in n_values:
        entries = []
        for i in range(1, len(sorted_k)):
            k_prev = sorted_k[i - 1]
            k_curr = sorted_k[i]
            gains = []
            for cell in cells:
                if n not in cell["accuracy"]:
                    continue
                if k_prev not in cell["accuracy"][n] or k_curr not in cell["accuracy"][n]:
                    continue
                gains.append(float(cell["accuracy"][n][k_curr]) - float(cell["accuracy"][n][k_prev]))
            summary = mean_confidence_interval(gains)
            summary["K_from"] = int(k_prev)
            summary["K_to"] = int(k_curr)
            entries.append(summary)
        result[str(n)] = entries
    return result


def compute_fixed_point_fraction(cells: list[dict]) -> dict:
    aggregated: dict[str, dict] = {}
    n_values = sorted({int(n) for cell in cells for n in cell["convergence"].keys()})
    for n in n_values:
        per_seed = [float(cell["convergence"][n]) for cell in cells if n in cell["convergence"]]
        summary = mean_confidence_interval(per_seed, bounds=(0.0, 1.0))
        summary["N"] = int(n)
        aggregated[str(n)] = summary
    return aggregated


def compute_generalization_gap(
    thinking_cells: list[dict],
    deep_cells: list[dict],
    k_probe: int,
    n_train_cap: int,
    n_sweep_values: list[int],
) -> dict:
    gap: dict[str, dict] = {}
    for n in n_sweep_values:
        if n <= n_train_cap:
            continue
        thinking_accs = [float(cell["accuracy"][n][k_probe]) for cell in thinking_cells if n in cell["accuracy"] and k_probe in cell["accuracy"][n]]
        deep_accs = [float(cell["accuracy"][n]) for cell in deep_cells if n in cell["accuracy"]]
        if not thinking_accs or not deep_accs:
            continue
        thinking_summary = mean_confidence_interval(thinking_accs, bounds=(0.0, 1.0))
        deep_summary = mean_confidence_interval(deep_accs, bounds=(0.0, 1.0))
        if thinking_summary["mean"] is None or deep_summary["mean"] is None:
            continue
        gap[str(n)] = {
            "N": int(n),
            "thinking_k": int(k_probe),
            "thinking_mean": thinking_summary["mean"],
            "thinking_ci95": thinking_summary["ci95"],
            "deep_mean": deep_summary["mean"],
            "deep_ci95": deep_summary["ci95"],
            "delta": float(thinking_summary["mean"]) - float(deep_summary["mean"]),
        }
    return gap


def plot_results(
    accuracy_by_k_and_n: dict,
    per_step_gain: dict,
    fixed_point_fraction: dict,
    generalization_gap: dict,
    figure_path: Path,
    k_list: list[int],
    n_list: list[int],
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for k in sorted(k_list):
        means = []
        highs = []
        lows = []
        xs = []
        for n in sorted(n_list):
            entry = accuracy_by_k_and_n.get(str(k), {}).get(str(n))
            if entry is None or entry.get("mean") is None:
                continue
            means.append(entry["mean"])
            highs.append(entry["ci95"]["high"])
            lows.append(entry["ci95"]["low"])
            xs.append(n)
        if xs:
            ax.plot(xs, means, marker="o", label=f"K={k}")
            ax.fill_between(xs, lows, highs, alpha=0.15)
    ax.axhline(1.0 / MODULUS, color="#991b1b", linestyle="--", alpha=0.5, label="chance")
    ax.set_xlabel("task depth N")
    ax.set_ylabel("accuracy")
    ax.set_title("accuracy by thinking steps and task depth")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for n_key, entries in per_step_gain.items():
        xs = [entry["K_to"] for entry in entries]
        means = [entry["mean"] for entry in entries if entry["mean"] is not None]
        if len(means) == len(xs) and xs:
            ax.plot(xs, means, marker="o", label=f"N={n_key}")
    ax.axhline(0.0, color="#111827", linewidth=1)
    ax.set_xlabel("K")
    ax.set_ylabel("per-step accuracy gain")
    ax.set_title("marginal gain per thinking step")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    xs = []
    means = []
    for n_key, entry in fixed_point_fraction.items():
        if entry.get("mean") is None:
            continue
        xs.append(int(n_key))
        means.append(entry["mean"])
    if xs:
        order = np.argsort(xs)
        xs_sorted = [xs[i] for i in order]
        means_sorted = [means[i] for i in order]
        ax.plot(xs_sorted, means_sorted, marker="o")
    ax.set_xlabel("task depth N")
    ax.set_ylabel("fraction at fixed point")
    ax.set_title(f"fixed-point convergence at K={CONVERGENCE_K_MAX}")

    ax = axes[1, 1]
    xs = []
    thinking_means = []
    deep_means = []
    for n_key, entry in generalization_gap.items():
        xs.append(int(n_key))
        thinking_means.append(entry["thinking_mean"])
        deep_means.append(entry["deep_mean"])
    if xs:
        order = np.argsort(xs)
        xs_sorted = [xs[i] for i in order]
        thinking_sorted = [thinking_means[i] for i in order]
        deep_sorted = [deep_means[i] for i in order]
        ax.plot(xs_sorted, thinking_sorted, marker="o", label="thinking")
        ax.plot(xs_sorted, deep_sorted, marker="s", label="deep feedforward")
    ax.axhline(1.0 / MODULUS, color="#991b1b", linestyle="--", alpha=0.5)
    ax.set_xlabel("task depth N (beyond train cap)")
    ax.set_ylabel("accuracy")
    ax.set_title("generalization: thinking vs deeper feedforward")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()


def main() -> None:
    apply_plot_style()
    torch.set_num_threads(1)
    device = torch.device("cpu")

    started = utc_now_iso()
    t0 = time.time()
    rng = build_rng(SEED)

    output_dir = output_dir_for(SCRIPT_PATH)
    figure_path = output_dir / "thinking_loop_prototype.png"
    metrics_path = output_dir / "thinking_loop_prototype_metrics.json"

    thinking_cells: list[dict] = []
    for _ in range(N_SEEDS):
        seed_value = int(rng.integers(0, 2**31 - 1))
        cell = run_thinking_cell(
            seed=seed_value,
            n_ops_train=N_TRAIN_CAP,
            think_steps_train=TRAIN_K,
            k_sweep_list=K_LIST,
            n_sweep_list=N_LIST,
            device=device,
        )
        thinking_cells.append(cell)

    deep_cells: list[dict] = []
    for _ in range(N_SEEDS):
        seed_value = int(rng.integers(0, 2**31 - 1))
        cell = run_deep_baseline_cell(
            seed=seed_value,
            n_ops_train=N_TRAIN_CAP,
            n_sweep_list=N_LIST,
            device=device,
        )
        deep_cells.append(cell)

    accuracy_by_k_and_n = aggregate_accuracy_by_k_and_n(thinking_cells)
    per_step_gain = compute_per_step_gain(thinking_cells, K_LIST)
    fixed_point_fraction = compute_fixed_point_fraction(thinking_cells)
    generalization_gap_at_max_k = compute_generalization_gap(
        thinking_cells,
        deep_cells,
        k_probe=max(K_LIST),
        n_train_cap=N_TRAIN_CAP,
        n_sweep_values=N_LIST,
    )

    trial_records: list[dict] = []
    for cell in thinking_cells:
        for n_ops, per_k in cell["accuracy"].items():
            for k, acc in per_k.items():
                trial_records.append({
                    "mode": "thinking",
                    "seed": int(cell["seed"]),
                    "n_params": int(cell["n_params"]),
                    "n_ops": int(n_ops),
                    "k": int(k),
                    "accuracy": float(acc),
                    "convergence_fraction": float(cell["convergence"].get(int(n_ops), 0.0)),
                    "mean_final_delta": float(cell["mean_final_delta"].get(int(n_ops), 0.0)),
                })
    for cell in deep_cells:
        for n_ops, acc in cell["accuracy"].items():
            trial_records.append({
                "mode": "deep_feedforward",
                "seed": int(cell["seed"]),
                "n_params": int(cell["n_params"]),
                "n_ops": int(n_ops),
                "k": 1,
                "accuracy": float(acc),
                "convergence_fraction": 0.0,
                "mean_final_delta": 0.0,
            })

    plot_results(
        accuracy_by_k_and_n=accuracy_by_k_and_n,
        per_step_gain=per_step_gain,
        fixed_point_fraction=fixed_point_fraction,
        generalization_gap=generalization_gap_at_max_k,
        figure_path=figure_path,
        k_list=K_LIST,
        n_list=N_LIST,
    )

    summary = {
        "accuracy_by_K_and_N": accuracy_by_k_and_n,
        "per_step_gain_by_task_difficulty": per_step_gain,
        "generalization_gap_thinking_vs_feedforward": generalization_gap_at_max_k,
        "fixed_point_convergence_fraction": fixed_point_fraction,
        "model_param_counts": {
            "thinking_and_shallow_feedforward": int(thinking_cells[0]["n_params"]) if thinking_cells else None,
            "deep_feedforward": int(deep_cells[0]["n_params"]) if deep_cells else None,
        },
        "train_conditions": {
            "modulus": MODULUS,
            "n_operations_vocab": N_OPERATIONS,
            "d_model": D_MODEL,
            "depth": DEPTH,
            "deep_depth": DEEP_DEPTH,
            "train_k": TRAIN_K,
            "n_train_cap": N_TRAIN_CAP,
        },
    }

    statistics: dict = {}

    finished = utc_now_iso()
    duration = time.time() - t0

    record = build_run_record(
        simulation_name="thinking_loop_prototype",
        script_path=SCRIPT_PATH,
        started_at_utc=started,
        finished_at_utc=finished,
        duration_sec=duration,
        parameters={
            "modulus": MODULUS,
            "n_operations_vocab": N_OPERATIONS,
            "d_model": D_MODEL,
            "depth": DEPTH,
            "deep_depth": DEEP_DEPTH,
            "n_train": N_TRAIN,
            "n_val": N_VAL,
            "n_test": N_TEST,
            "n_epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "train_k": TRAIN_K,
            "n_list": N_LIST,
            "n_train_cap": N_TRAIN_CAP,
            "k_list": K_LIST,
            "n_seeds": N_SEEDS,
            "fixed_point_tol": FIXED_POINT_TOL,
            "convergence_k_max": CONVERGENCE_K_MAX,
        },
        seed_numpy=SEED,
        n_trials=len(trial_records),
        summary=summary,
        statistics=statistics,
        trials=trial_records,
        artifacts=[
            {"path": figure_path.as_posix(), "type": "figure"},
            {"path": metrics_path.as_posix(), "type": "metrics"},
        ],
        warnings=[
            "toy modular-arithmetic task, not natural-language reasoning",
            "same-parameter thinking comparison; deep feedforward baseline has more parameters by design",
            "state is fed as an extra appended token, not via gated residual injection; fixed-point analysis reflects this design",
        ],
    )

    write_json(metrics_path, record)

    print(f"done in {duration:.1f}s")
    for k in sorted(K_LIST):
        per_n = accuracy_by_k_and_n.get(str(k), {})
        line_parts = [f"K={k}"]
        for n in sorted(N_LIST):
            entry = per_n.get(str(n))
            if entry is None or entry.get("mean") is None:
                continue
            line_parts.append(f"N={n}:{entry['mean']:.3f}")
        print("  " + "  ".join(line_parts))


if __name__ == "__main__":
    main()
