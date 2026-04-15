from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SIM_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
for p in [str(SIM_ROOT), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from shared import (
    build_rng,
    build_run_record,
    env_float,
    env_int,
    mean_confidence_interval,
    output_dir_for,
    require_positive,
    utc_now_iso,
    write_json,
)

from neuroloc.model.god_machine import Config, GodMachine

SEED = env_int("SLOT_INT_SEED", 42)
D_MODEL = env_int("SLOT_INT_D_MODEL", 128)
N_LAYERS = env_int("SLOT_INT_N_LAYERS", 7)
MAX_STEPS = env_int("SLOT_INT_MAX_STEPS", 800)
SEQ_LEN = env_int("SLOT_INT_SEQ_LEN", 256)
BATCH_SIZE = env_int("SLOT_INT_BATCH_SIZE", 8)
LR = env_float("SLOT_INT_LR", 1e-3)
PASSKEY_LENGTH = env_int("SLOT_INT_PASSKEY_LEN", 3)
PASSKEY_DISTANCE = env_int("SLOT_INT_PASSKEY_DISTANCE", 128)
PASSKEY_TRIALS = env_int("SLOT_INT_PASSKEY_TRIALS", 50)
PASSKEY_THRESHOLD = env_float("SLOT_INT_PASSKEY_THRESHOLD", 0.05)
PASSKEY_LOSS_WEIGHT = env_float("SLOT_INT_PASSKEY_LOSS_WEIGHT", 10.0)

_TRAIN_DISTANCES_STR = os.environ.get("SLOT_INT_TRAIN_DISTANCES", "16,64,128")
TRAIN_DISTANCES = tuple(int(x) for x in _TRAIN_DISTANCES_STR.split(",") if x.strip())

PASSKEY_ALPHABET_BYTES = tuple(list(range(48, 58)) + list(range(97, 103)))

MARK_START_BYTES = (255, 254, 253)
MARK_END_BYTES = (253, 254, 255)
MARK_QUERY_BYTES = (252, 251, 250)

require_positive("SLOT_INT_D_MODEL", D_MODEL)
require_positive("SLOT_INT_N_LAYERS", N_LAYERS)
require_positive("SLOT_INT_MAX_STEPS", MAX_STEPS)
require_positive("SLOT_INT_SEQ_LEN", SEQ_LEN)
require_positive("SLOT_INT_BATCH_SIZE", BATCH_SIZE)
require_positive("SLOT_INT_PASSKEY_TRIALS", PASSKEY_TRIALS)
require_positive("SLOT_INT_PASSKEY_LEN", PASSKEY_LENGTH)
if not TRAIN_DISTANCES:
    raise ValueError("SLOT_INT_TRAIN_DISTANCES must be a non-empty comma-separated list of integers")
for _d in TRAIN_DISTANCES:
    if _d <= 0:
        raise ValueError(f"SLOT_INT_TRAIN_DISTANCES contains non-positive value {_d}")
if PASSKEY_LOSS_WEIGHT < 1.0:
    raise ValueError(f"SLOT_INT_PASSKEY_LOSS_WEIGHT must be >= 1.0, got {PASSKEY_LOSS_WEIGHT}")


class ByteDataset(Dataset):
    def __init__(self, data: bytes, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(1, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx % max(1, len(self.data) - self.seq_len - 1)
        chunk = self.data[start : start + self.seq_len + 1]
        ids = torch.frombuffer(bytearray(chunk), dtype=torch.uint8).long()
        return ids[:-1], ids[1:]


def generate_synthetic_corpus(
    size: int,
    seed: int,
    distances: tuple[int, ...] | None = None,
    block_seq_len: int = 256,
) -> bytes:
    rng = np.random.default_rng(seed)
    if distances is None:
        distances = TRAIN_DISTANCES
    mark_start = list(MARK_START_BYTES)
    mark_end = list(MARK_END_BYTES)
    mark_query = list(MARK_QUERY_BYTES)
    passkey_len = PASSKEY_LENGTH
    alphabet = np.array(PASSKEY_ALPHABET_BYTES, dtype=np.int64)
    marker_overhead = 3 + passkey_len + 3 + 3 + passkey_len
    min_opening = 5
    for d in distances:
        needed = marker_overhead + d + min_opening
        if block_seq_len < needed:
            raise ValueError(
                f"block_seq_len={block_seq_len} too small for mid-filler distance {d} "
                f"plus marker overhead {marker_overhead} plus minimum opening {min_opening}"
            )
    out: list[int] = []
    while len(out) < size:
        d = int(rng.choice(distances))
        passkey_digits = alphabet[rng.integers(0, len(alphabet), size=passkey_len)].tolist()
        remaining = block_seq_len - marker_overhead - d
        opening_filler_len = int(rng.integers(min_opening, max(min_opening + 1, remaining)))
        trailing_filler_len = remaining - opening_filler_len
        block: list[int] = []
        block.extend(rng.integers(32, 127, size=opening_filler_len).tolist())
        block.extend(mark_start)
        block.extend(passkey_digits)
        block.extend(mark_end)
        block.extend(rng.integers(32, 127, size=d).tolist())
        block.extend(mark_query)
        block.extend(passkey_digits)
        if trailing_filler_len > 0:
            block.extend(rng.integers(32, 127, size=trailing_filler_len).tolist())
        out.extend(block)
    return bytes(out[:size])


def compute_passkey_target_mask(input_ids: torch.Tensor, passkey_len: int) -> torch.Tensor:
    B, T = input_ids.shape
    mask = torch.zeros_like(input_ids, dtype=torch.float)
    if T < 3 + passkey_len:
        return mask
    q0, q1, q2 = MARK_QUERY_BYTES
    query_at = (
        (input_ids[:, : T - 2] == q0)
        & (input_ids[:, 1 : T - 1] == q1)
        & (input_ids[:, 2 : T] == q2)
    )
    for k in range(passkey_len):
        start_idx = k + 2
        if start_idx >= T:
            break
        src_len = min(T - start_idx, query_at.shape[1])
        mask[:, start_idx : start_idx + src_len] = torch.maximum(
            mask[:, start_idx : start_idx + src_len],
            query_at[:, :src_len].float(),
        )
    return mask


def make_slot_config() -> Config:
    layer_pattern = ("SLOT",) * 6 + ("ATTN",)
    return Config(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        vocab_size=256,
        max_seq_len=SEQ_LEN,
        delta_num_heads=4,
        delta_head_dim=max(32, D_MODEL // 4),
        alpha_log_mean=5.0,
        alpha_log_std=0.3,
        attn_d_c=max(32, D_MODEL // 2),
        attn_d_R=max(8, D_MODEL // 8),
        attn_num_heads=4,
        mlp_ratio=2.0,
        layer_pattern=layer_pattern,
        kwta_enabled=False,
        delta_erasure_enabled=False,
        bcm_alpha_enabled=False,
        multi_compartment_enabled=False,
        imagination_enabled=False,
        pc_diagnostic_enabled=False,
        slot_num_slots=64,
        slot_log_temperature_init=-1.0,
        slot_surprise_tau=0.1,
        slot_gate_init=-4.0,
        slot_predict_rank=8,
        slot_prediction_lambda=1e-3,
        lr=LR,
        weight_decay=0.1,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        max_steps=MAX_STEPS,
        warmup_steps=50,
        grad_clip=1.0,
        grad_accum_steps=1,
        grad_checkpointing=False,
        amp=True,
        seed=SEED,
    )


def make_matrix_config() -> Config:
    layer_pattern = ("DELTA",) * 6 + ("ATTN",)
    return Config(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        vocab_size=256,
        max_seq_len=SEQ_LEN,
        delta_num_heads=4,
        delta_head_dim=max(32, D_MODEL // 4),
        alpha_log_mean=5.0,
        alpha_log_std=0.3,
        attn_d_c=max(32, D_MODEL // 2),
        attn_d_R=max(8, D_MODEL // 8),
        attn_num_heads=4,
        mlp_ratio=2.0,
        layer_pattern=layer_pattern,
        kwta_enabled=False,
        delta_erasure_enabled=False,
        use_fla_if_available=True,
        bcm_alpha_enabled=False,
        multi_compartment_enabled=False,
        imagination_enabled=False,
        pc_diagnostic_enabled=False,
        lr=LR,
        weight_decay=0.1,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        max_steps=MAX_STEPS,
        warmup_steps=50,
        grad_clip=1.0,
        grad_accum_steps=1,
        grad_checkpointing=False,
        amp=True,
        seed=SEED,
    )


def run_passkey_test(
    model: nn.Module,
    config: Config,
    device: torch.device,
    trials: int,
    distance: int,
    length: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    alphabet = list(PASSKEY_ALPHABET_BYTES)
    correct = 0
    total = 0
    cosines: list[float] = []
    model.eval()
    chunk_size = min(config.seq_len, 256)
    if chunk_size < config.seq_len:
        raise ValueError(
            f"run_passkey_test currently indexes target_start as an absolute position "
            f"but the single-shot logits returned after a chunked forward only cover the "
            f"last chunk. pick SEQ_LEN <= 256 or generalize target_start to a "
            f"chunk-relative offset. got SEQ_LEN={config.seq_len} chunk_size={chunk_size}"
        )
    marker_start = list(MARK_START_BYTES)
    marker_end = list(MARK_END_BYTES)
    marker_query = list(MARK_QUERY_BYTES)
    with torch.no_grad():
        for trial in range(trials):
            passkey_bytes = [int(alphabet[i]) for i in rng.integers(0, len(alphabet), size=length)]
            filler = rng.integers(32, 127, size=max(distance * 2, 64)).tolist()
            overhead = len(marker_start) + len(passkey_bytes) + len(marker_end) + len(marker_query)
            max_insert = max(1, distance - overhead - 20)
            insert_pos = int(rng.integers(5, max(6, min(max_insert, distance // 3))))
            before = filler[:insert_pos]
            mid_len = max(0, distance - overhead - insert_pos)
            middle = filler[insert_pos : insert_pos + mid_len]
            seq = before + marker_start + passkey_bytes + marker_end + middle + marker_query
            total_seq_len = config.seq_len
            if len(seq) < total_seq_len:
                remaining = total_seq_len - len(seq)
                seq = seq + rng.integers(32, 127, size=remaining).tolist()
            seq = seq[:total_seq_len]
            input_ids = torch.tensor([seq], dtype=torch.long, device=device)
            states: list[Any] | None = None
            logits = None
            for start in range(0, input_ids.shape[1], chunk_size):
                end = min(start + chunk_size, input_ids.shape[1])
                chunk = input_ids[:, start:end]
                logits, states, _ = model(chunk, states=states, offset=start)
            if logits is None:
                continue
            query_pos = None
            for i in range(len(seq)):
                if seq[i : i + len(marker_query)] == marker_query:
                    query_pos = i + len(marker_query)
                    break
            if query_pos is None:
                query_pos = len(seq) - 1
            target_start = query_pos - 1
            if target_start < 0:
                target_start = 0
            if target_start + length > logits.shape[1]:
                target_start = max(0, logits.shape[1] - length)
            predicted = logits[0, target_start : target_start + length, :].argmax(dim=-1).cpu().numpy().tolist()
            target = passkey_bytes[:length]
            matches = sum(1 for p, t in zip(predicted, target) if p == t)
            if matches == length:
                correct += 1
            total += 1
            tgt_tensor = torch.tensor(target, dtype=torch.long, device=device).unsqueeze(0)
            pred_probs = torch.softmax(logits[0, target_start : target_start + length, :], dim=-1)
            correct_probs = pred_probs.gather(1, tgt_tensor)
            cosines.append(float(correct_probs.mean().item()))
    model.train()
    exact_rate = correct / max(total, 1)
    mean_cosine = float(np.mean(cosines)) if cosines else 0.0
    return {
        "exact_rate": exact_rate,
        "correct": correct,
        "total": total,
        "mean_cosine": mean_cosine,
        "distance": distance,
        "passkey_length": length,
    }


def train_and_evaluate(config: Config, name: str, rng: np.random.Generator) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GodMachine(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(
        f"[{name}] d_model={config.d_model} n_layers={config.n_layers} params={param_count:,} "
        f"train_distances={TRAIN_DISTANCES} passkey_loss_weight={PASSKEY_LOSS_WEIGHT} device={device}",
        flush=True,
    )

    corpus_size = SEQ_LEN * BATCH_SIZE * MAX_STEPS * 2
    corpus = generate_synthetic_corpus(
        max(corpus_size, 100_000),
        seed=int(rng.integers(0, 2**31)),
        distances=TRAIN_DISTANCES,
    )
    val_size = max(SEQ_LEN * 20, 50_000)
    val_start = len(corpus) - val_size
    train_data = corpus[:val_start]
    val_data = corpus[val_start:]

    train_ds = ByteDataset(train_data, SEQ_LEN)
    val_ds = ByteDataset(val_data, SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=config.warmup_steps
    )
    criterion_unweighted = nn.CrossEntropyLoss()

    step_losses: list[float] = []
    val_bpbs: list[float] = []
    best_val_bpb = float("inf")

    train_iter = iter(train_loader)
    for step in range(MAX_STEPS):
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        input_ids, targets = batch[0].to(device), batch[1].to(device)
        logits, _, _ = model(input_ids)
        loss_per_token = F.cross_entropy(
            logits.reshape(-1, 256),
            targets.reshape(-1),
            reduction="none",
        ).view(input_ids.shape[0], input_ids.shape[1])
        passkey_mask = compute_passkey_target_mask(input_ids, PASSKEY_LENGTH)
        weights = 1.0 + (PASSKEY_LOSS_WEIGHT - 1.0) * passkey_mask
        weighted_loss = (loss_per_token * weights).sum() / weights.sum().clamp(min=1.0)
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        step_losses.append(float(weighted_loss.item()))
        if step < 5 or (step + 1) % max(1, MAX_STEPS // 50) == 0:
            passkey_count = int(passkey_mask.sum().item())
            print(
                f"[{name}] step {step+1}/{MAX_STEPS} weighted_loss={weighted_loss.item():.4f} "
                f"passkey_tokens_in_batch={passkey_count}",
                flush=True,
            )

        if (step + 1) % max(1, MAX_STEPS // 10) == 0:
            model.eval()
            total_loss = 0.0
            total_tokens = 0
            with torch.no_grad():
                vb_count = 0
                for vbatch in val_loader:
                    if vb_count >= 20:
                        break
                    vb_count += 1
                    v_input, v_target = vbatch[0].to(device), vbatch[1].to(device)
                    v_logits, _, _ = model(v_input)
                    v_loss = criterion_unweighted(v_logits.view(-1, 256), v_target.view(-1))
                    total_loss += float(v_loss.item()) * v_target.numel()
                    total_tokens += v_target.numel()
            val_bpb = total_loss / max(total_tokens, 1) / 0.6931
            val_bpbs.append(val_bpb)
            if val_bpb < best_val_bpb:
                best_val_bpb = val_bpb
            print(
                f"[{name}] step {step+1}/{MAX_STEPS} weighted_loss={weighted_loss.item():.4f} "
                f"val_bpb_unweighted={val_bpb:.4f} best={best_val_bpb:.4f}",
                flush=True,
            )
            model.train()

    passkey_result = run_passkey_test(
        model, config, device,
        trials=PASSKEY_TRIALS,
        distance=PASSKEY_DISTANCE,
        length=PASSKEY_LENGTH,
        seed=int(rng.integers(0, 2**31)),
    )

    return {
        "name": name,
        "param_count": param_count,
        "max_steps": MAX_STEPS,
        "final_train_loss": step_losses[-1] if step_losses else None,
        "best_val_bpb": best_val_bpb,
        "final_val_bpb": val_bpbs[-1] if val_bpbs else None,
        "passkey_exact_rate": passkey_result["exact_rate"],
        "passkey_correct": passkey_result["correct"],
        "passkey_total": passkey_result["total"],
        "passkey_mean_cosine": passkey_result["mean_cosine"],
        "passkey_distance": passkey_result["distance"],
        "passkey_length": passkey_result["passkey_length"],
        "step_losses": step_losses,
        "val_bpbs": val_bpbs,
    }


def main() -> dict[str, Any]:
    started_at = utc_now_iso()
    start_wall = time.time()
    rng = build_rng(SEED)

    slot_config = make_slot_config()
    matrix_config = make_matrix_config()

    slot_result = train_and_evaluate(slot_config, "slot_memory", rng)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    matrix_result = train_and_evaluate(matrix_config, "matrix_memory", rng)

    slot_exact = slot_result["passkey_exact_rate"]
    matrix_exact = matrix_result["passkey_exact_rate"]
    slot_cosine = slot_result["passkey_mean_cosine"]
    matrix_cosine = matrix_result["passkey_mean_cosine"]

    slot_passes_exact = slot_exact > PASSKEY_THRESHOLD
    matrix_passes_exact = matrix_exact > PASSKEY_THRESHOLD
    slot_beats_matrix = slot_exact > matrix_exact

    summary = {
        "gate_name": "tiny_integration_slot_vs_matrix",
        "pass_threshold": PASSKEY_THRESHOLD,
        "slot_memory": {
            "exact_rate": slot_exact,
            "correct": slot_result["passkey_correct"],
            "total": slot_result["passkey_total"],
            "mean_cosine": slot_cosine,
            "best_val_bpb": slot_result["best_val_bpb"],
            "param_count": slot_result["param_count"],
        },
        "matrix_memory": {
            "exact_rate": matrix_exact,
            "correct": matrix_result["passkey_correct"],
            "total": matrix_result["passkey_total"],
            "mean_cosine": matrix_cosine,
            "best_val_bpb": matrix_result["best_val_bpb"],
            "param_count": matrix_result["param_count"],
        },
        "slot_passes_threshold": slot_passes_exact,
        "matrix_passes_threshold": matrix_passes_exact,
        "slot_beats_matrix": slot_beats_matrix,
        "gate_met": slot_passes_exact and slot_beats_matrix,
    }

    finished_at = utc_now_iso()
    duration = time.time() - start_wall

    parameters = {
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "max_steps": MAX_STEPS,
        "seq_len": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "passkey_length": PASSKEY_LENGTH,
        "passkey_distance": PASSKEY_DISTANCE,
        "passkey_trials": PASSKEY_TRIALS,
        "passkey_loss_weight": PASSKEY_LOSS_WEIGHT,
        "train_distances": list(TRAIN_DISTANCES),
        "passkey_alphabet_size": len(PASSKEY_ALPHABET_BYTES),
        "alpha_log_mean": 5.0,
        "fla_enabled": True,
    }

    script_path = Path(__file__).resolve()
    output_dir = output_dir_for(script_path)
    metrics_path = output_dir / "slot_integration_metrics.json"

    record = build_run_record(
        simulation_name="slot_integration",
        script_path=script_path,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_sec=duration,
        parameters=parameters,
        seed_numpy=SEED,
        n_trials=2,
        summary=summary,
        statistics={"elapsed_seconds": duration},
        trials=[slot_result, matrix_result],
        artifacts=[],
        warnings=[
            "tiny-scale integration test. results are directional evidence, not "
            "conclusive proof. the passkey threshold is intentionally low (5%) "
            "because tiny models have limited capacity; the relevant comparison "
            "is slot vs matrix under identical training and data.",
        ],
    )
    write_json(metrics_path, record)

    print(f"\n{'='*60}")
    print(f"INTEGRATION GATE: slot_vs_matrix at d_model={D_MODEL}, L={PASSKEY_LENGTH}, "
          f"alphabet={len(PASSKEY_ALPHABET_BYTES)}, loss_weight={PASSKEY_LOSS_WEIGHT}, "
          f"train_distances={list(TRAIN_DISTANCES)}")
    print(f"{'='*60}")
    print(f"slot_memory: exact={slot_exact:.3f} cosine={slot_cosine:.4f} bpb={slot_result['best_val_bpb']:.4f}")
    print(f"matrix_memory: exact={matrix_exact:.3f} cosine={matrix_cosine:.4f} bpb={matrix_result['best_val_bpb']:.4f}")
    print(f"slot_passes_threshold({PASSKEY_THRESHOLD}): {slot_passes_exact}")
    print(f"slot_beats_matrix: {slot_beats_matrix}")
    print(f"gate_met: {summary['gate_met']}")
    print(f"{'='*60}")
    return summary


if __name__ == "__main__":
    result = main()
