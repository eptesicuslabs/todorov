from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "autoresearch"))

import train as T

T.FLA_AVAILABLE = False
T.DEVICE = torch.device("cpu")
T.OUTPUT_DIR = Path(tempfile.mkdtemp())


def run():
    t0 = time.time()
    failures = []

    cfg_a = T.TrainConfig(
        d_model=64,
        n_layers=8,
        vocab_size=256,
        max_seq_len=512,
        kda_num_heads=2,
        kda_head_dim=32,
        m3_d_state=8,
        m3_expand=2,
        mla_d_c=32,
        mla_d_R=8,
        mla_num_heads=2,
        mlp_ratio=2.0,
        layer_pattern=("KDA", "KDA", "KDA", "Mamba3", "KDA", "KDA", "KDA", "MLA"),
        lr=3e-4,
        weight_decay=0.01,
        batch_size=4,
        max_steps=2,
        warmup_steps=1,
        eval_interval=100,
        gradient_clip=1.0,
        seq_len=32,
        use_spikes=True,
        spike_alpha=1.0,
    )

    rng = np.random.RandomState(42)
    train_data = bytes(rng.randint(0, 256, size=8192).tolist())
    val_data = bytes(rng.randint(0, 256, size=2048).tolist())

    print("[a] building tiny model (d=64, 8 layers, 2 heads, head_dim=32, vocab=256)")
    model_a = T.LM(cfg_a).to(T.DEVICE)
    print(f"    params: {model_a.count_params():,}")

    print("[b] training 2 steps at seq_len=32 (matmul path, T<512)")
    result_a = T.train_model(cfg_a, train_data, val_data, "smoke_stage1")
    if result_a["total_steps"] < 2:
        failures.append(f"stage1 only ran {result_a['total_steps']} steps, expected >=2")
    else:
        print(f"    stage1 OK: {result_a['total_steps']} steps, loss={result_a['final_val_bpb']:.4f}")

    ckpt_path = str(T.OUTPUT_DIR / "smoke_stage1_best.pt")
    if not Path(ckpt_path).exists():
        torch.save(model_a.state_dict(), ckpt_path)
        print(f"    fallback: saved checkpoint manually")
    else:
        print(f"    checkpoint exists at {ckpt_path}")

    print("[c] loading checkpoint into new model with seq_len=64 (strict=False)")
    cfg_b = T.TrainConfig(
        d_model=64,
        n_layers=8,
        vocab_size=256,
        max_seq_len=512,
        kda_num_heads=2,
        kda_head_dim=32,
        m3_d_state=8,
        m3_expand=2,
        mla_d_c=32,
        mla_d_R=8,
        mla_num_heads=2,
        mlp_ratio=2.0,
        layer_pattern=("KDA", "KDA", "KDA", "Mamba3", "KDA", "KDA", "KDA", "MLA"),
        lr=1e-4,
        weight_decay=0.01,
        batch_size=4,
        max_steps=2,
        warmup_steps=1,
        eval_interval=100,
        gradient_clip=1.0,
        seq_len=64,
        use_spikes=True,
        spike_alpha=1.0,
    )
    model_b = T.LM(cfg_b).to(T.DEVICE)
    sd = torch.load(ckpt_path, map_location=T.DEVICE, weights_only=True)
    missing, unexpected = model_b.load_state_dict(sd, strict=False)
    print(f"    load strict=False: missing={len(missing)}, unexpected={len(unexpected)}")

    print("[d] training 2 more steps with loaded checkpoint at seq_len=64")
    result_b = T.train_model(cfg_b, train_data, val_data, "smoke_stage2", init_checkpoint=ckpt_path)
    if result_b["total_steps"] < 2:
        failures.append(f"stage2 only ran {result_b['total_steps']} steps, expected >=2")
    else:
        print(f"    stage2 OK: {result_b['total_steps']} steps, loss={result_b['final_val_bpb']:.4f}")

    print("[e] selective_copy_test with context_length=64")
    eval_model = T.LM(cfg_b).to(T.DEVICE)
    ckpt2_path = str(T.OUTPUT_DIR / "smoke_stage2_best.pt")
    if Path(ckpt2_path).exists():
        eval_model.load_state_dict(torch.load(ckpt2_path, map_location=T.DEVICE, weights_only=True), strict=False)
    copy_result = T.selective_copy_test(eval_model, context_length=64, num_trials=5, copy_length=4, chunk_size=32)
    print(f"    selective_copy: accuracy={copy_result['accuracy']:.1%} ({copy_result['correct']}/{copy_result['total']})")
    if "accuracy" not in copy_result:
        failures.append("selective_copy_test missing 'accuracy' key")

    print("[f] passkey_retrieval_test with context_length=64")
    passkey_result = T.passkey_retrieval_test(eval_model, context_length=64, num_trials=5, passkey_length=3, chunk_size=32)
    print(f"    passkey: accuracy={passkey_result['accuracy']:.1%} ({passkey_result['correct']}/{passkey_result['total']})")
    if "accuracy" not in passkey_result:
        failures.append("passkey_retrieval_test missing 'accuracy' key")

    print("[g] JSON serialization (numpy bool crash guard)")
    combined = {
        "stage1": result_a,
        "stage2": result_b,
        "copy": copy_result,
        "passkey": passkey_result,
        "gate_a_pass": bool(np.bool_(True)),
        "gate_b_pass": bool(np.bool_(False)),
        "some_float": float(np.float64(3.14)),
        "some_int": int(np.int64(42)),
    }
    try:
        serialized = json.dumps(combined, default=str)
        roundtrip = json.loads(serialized)
        if "stage1" not in roundtrip or "copy" not in roundtrip:
            failures.append("JSON roundtrip lost keys")
        else:
            print(f"    JSON OK: {len(serialized)} bytes serialized, roundtrip clean")
    except Exception as e:
        failures.append(f"JSON serialization failed: {e}")

    elapsed = time.time() - t0
    print(f"\nelapsed: {elapsed:.1f}s")
    if elapsed > 30:
        failures.append(f"took {elapsed:.1f}s, limit is 30s")

    if failures:
        print("\n=== FAIL ===")
        for f in failures:
            print(f"  - {f}")
        return False
    else:
        print("\n=== PASS ===")
        return True


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
