from __future__ import annotations

import numpy as np


MARK_STORE_START = (255, 254, 253)
MARK_STORE_END = (253, 254, 255)
MARK_QUERY = (252, 251, 250)
MARK_KV_KEY = (248, 247, 246)
MARK_KV_VAL = (246, 247, 248)
MARK_COPY_START = (244, 243, 242)
MARK_COPY_END = (242, 243, 244)

HEX_DIGITS = tuple(list(range(48, 58)) + list(range(97, 103)))
FILLER_LO = 32
FILLER_HI = 127

MAX_BLOCK_SEQ_LEN = 2048


def _random_filler(rng: np.random.Generator, length: int) -> list[int]:
    if length <= 0:
        return []
    return rng.integers(FILLER_LO, FILLER_HI, size=length).tolist()


def _random_hex_sequence(rng: np.random.Generator, length: int) -> list[int]:
    alphabet = np.array(HEX_DIGITS, dtype=np.int64)
    return alphabet[rng.integers(0, len(alphabet), size=length)].tolist()


def _passkey_block(
    rng: np.random.Generator,
    block_len: int,
    distance: int,
    passkey_len: int,
) -> list[int]:
    marker_overhead = 3 + passkey_len + 3 + 3 + passkey_len
    min_opening = 5
    if block_len < marker_overhead + distance + min_opening:
        raise ValueError(
            f"passkey block_len={block_len} too small for distance={distance}, "
            f"marker_overhead={marker_overhead}, min_opening={min_opening}"
        )
    passkey = _random_hex_sequence(rng, passkey_len)
    remaining = block_len - marker_overhead - distance
    opening = int(rng.integers(min_opening, max(min_opening + 1, remaining)))
    trailing = remaining - opening
    block: list[int] = []
    block.extend(_random_filler(rng, opening))
    block.extend(MARK_STORE_START)
    block.extend(passkey)
    block.extend(MARK_STORE_END)
    block.extend(_random_filler(rng, distance))
    block.extend(MARK_QUERY)
    block.extend(passkey)
    if trailing > 0:
        block.extend(_random_filler(rng, trailing))
    return block


def _kv_recall_block(
    rng: np.random.Generator,
    block_len: int,
    num_pairs: int,
    key_len: int,
    value_len: int,
) -> list[int]:
    pair_overhead = 3 + key_len + 3 + value_len
    query_overhead = 3 + key_len + value_len
    marker_budget = num_pairs * pair_overhead + query_overhead
    min_filler_between = 4
    min_total = marker_budget + (num_pairs + 1) * min_filler_between
    if block_len < min_total:
        raise ValueError(
            f"kv block_len={block_len} too small for {num_pairs} pairs "
            f"(key_len={key_len}, value_len={value_len}, min_total={min_total})"
        )
    keys = [_random_hex_sequence(rng, key_len) for _ in range(num_pairs)]
    values = [_random_hex_sequence(rng, value_len) for _ in range(num_pairs)]
    query_idx = int(rng.integers(0, num_pairs))
    extra = block_len - min_total
    filler_slots = num_pairs + 1
    filler_extra = rng.multinomial(extra, [1.0 / filler_slots] * filler_slots) if extra > 0 else [0] * filler_slots
    block: list[int] = []
    for i in range(num_pairs):
        block.extend(_random_filler(rng, min_filler_between + int(filler_extra[i])))
        block.extend(MARK_KV_KEY)
        block.extend(keys[i])
        block.extend(MARK_KV_VAL)
        block.extend(values[i])
    block.extend(_random_filler(rng, min_filler_between + int(filler_extra[-1])))
    block.extend(MARK_QUERY)
    block.extend(keys[query_idx])
    block.extend(values[query_idx])
    return block[:block_len]


def _copy_block(
    rng: np.random.Generator,
    block_len: int,
    sequence_len: int,
) -> list[int]:
    marker_overhead = 3 + sequence_len + 3 + sequence_len
    min_opening = 5
    if block_len < marker_overhead + min_opening:
        raise ValueError(
            f"copy block_len={block_len} too small for sequence_len={sequence_len}, "
            f"marker_overhead={marker_overhead}, min_opening={min_opening}"
        )
    sequence = _random_hex_sequence(rng, sequence_len)
    remaining = block_len - marker_overhead
    opening = int(rng.integers(min_opening, max(min_opening + 1, remaining)))
    trailing = remaining - opening
    block: list[int] = []
    block.extend(_random_filler(rng, opening))
    block.extend(MARK_COPY_START)
    block.extend(sequence)
    block.extend(MARK_COPY_END)
    block.extend(sequence)
    if trailing > 0:
        block.extend(_random_filler(rng, trailing))
    return block


def generate_cognition_corpus(
    size: int,
    seed: int,
    block_seq_len: int = 256,
    passkey_distances: tuple[int, ...] = (16, 64, 128, 256, 512),
    passkey_length: int = 5,
    kv_num_pairs: int = 4,
    kv_key_length: int = 5,
    kv_value_length: int = 5,
    copy_sequence_length: int = 8,
    mix_passkey: float = 0.5,
    mix_kv: float = 0.3,
    mix_copy: float = 0.2,
) -> bytes:
    if block_seq_len > MAX_BLOCK_SEQ_LEN:
        raise ValueError(f"block_seq_len={block_seq_len} exceeds MAX_BLOCK_SEQ_LEN={MAX_BLOCK_SEQ_LEN}")
    weights = np.array([mix_passkey, mix_kv, mix_copy], dtype=np.float64)
    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError(f"mix weights must sum to 1.0, got {weights.sum()}")
    if (weights < 0).any():
        raise ValueError(f"mix weights must be non-negative, got {weights.tolist()}")
    rng = np.random.default_rng(seed)
    out: list[int] = []
    while len(out) < size:
        task = int(rng.choice(3, p=weights))
        block_len = block_seq_len
        if task == 0:
            feasible_distances = [d for d in passkey_distances if block_len >= 3 + passkey_length + 3 + 3 + passkey_length + d + 5]
            if not feasible_distances:
                raise ValueError(f"no passkey_distance fits block_seq_len={block_len} with passkey_length={passkey_length}")
            distance = int(rng.choice(feasible_distances))
            block = _passkey_block(rng, block_len, distance, passkey_length)
        elif task == 1:
            num_pairs = kv_num_pairs
            pair_overhead = 3 + kv_key_length + 3 + kv_value_length
            query_overhead = 3 + kv_key_length + kv_value_length
            min_total = num_pairs * pair_overhead + query_overhead + (num_pairs + 1) * 4
            while num_pairs > 1 and block_len < min_total:
                num_pairs -= 1
                min_total = num_pairs * pair_overhead + query_overhead + (num_pairs + 1) * 4
            block = _kv_recall_block(rng, block_len, num_pairs, kv_key_length, kv_value_length)
        else:
            block = _copy_block(rng, block_len, copy_sequence_length)
        out.extend(block)
    return bytes(out[:size])


def split_train_val(corpus: bytes, val_fraction: float = 0.05) -> tuple[bytes, bytes]:
    if not 0 < val_fraction < 0.5:
        raise ValueError(f"val_fraction must be in (0, 0.5), got {val_fraction}")
    n = len(corpus)
    val_size = max(2048, int(n * val_fraction))
    return corpus[:-val_size], corpus[-val_size:]
