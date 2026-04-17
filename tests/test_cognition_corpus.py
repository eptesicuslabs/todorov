import numpy as np
import pytest

from neuroloc.data.cognition_corpus import (
    HEX_DIGITS,
    MARK_COPY_END,
    MARK_COPY_START,
    MARK_KV_KEY,
    MARK_KV_VAL,
    MARK_QUERY,
    MARK_STORE_END,
    MARK_STORE_START,
    generate_cognition_corpus,
    split_train_val,
)


def test_generate_cognition_corpus_returns_exact_size() -> None:
    corpus = generate_cognition_corpus(size=4096, seed=42, block_seq_len=256)
    assert len(corpus) == 4096


def test_generate_cognition_corpus_contains_all_three_task_markers() -> None:
    corpus = generate_cognition_corpus(size=64_000, seed=42, block_seq_len=256)
    b = bytes(corpus)
    assert b.count(bytes(MARK_STORE_START)) > 0
    assert b.count(bytes(MARK_STORE_END)) > 0
    assert b.count(bytes(MARK_QUERY)) > 0
    assert b.count(bytes(MARK_KV_KEY)) > 0
    assert b.count(bytes(MARK_KV_VAL)) > 0
    assert b.count(bytes(MARK_COPY_START)) > 0
    assert b.count(bytes(MARK_COPY_END)) > 0


def test_generate_cognition_corpus_byte_values_respect_alphabet() -> None:
    corpus = generate_cognition_corpus(size=32_000, seed=7, block_seq_len=256)
    allowed = (
        set(range(32, 127))
        | set(HEX_DIGITS)
        | set(MARK_STORE_START)
        | set(MARK_STORE_END)
        | set(MARK_QUERY)
        | set(MARK_KV_KEY)
        | set(MARK_KV_VAL)
        | set(MARK_COPY_START)
        | set(MARK_COPY_END)
    )
    unique = set(corpus)
    assert unique.issubset(allowed), f"unexpected byte values: {sorted(unique - allowed)}"


def test_generate_cognition_corpus_is_deterministic_for_same_seed() -> None:
    a = generate_cognition_corpus(size=8000, seed=123, block_seq_len=256)
    b = generate_cognition_corpus(size=8000, seed=123, block_seq_len=256)
    assert a == b


def test_generate_cognition_corpus_differs_across_seeds() -> None:
    a = generate_cognition_corpus(size=8000, seed=123, block_seq_len=256)
    b = generate_cognition_corpus(size=8000, seed=456, block_seq_len=256)
    assert a != b


def test_generate_cognition_corpus_mix_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="must sum to 1.0"):
        generate_cognition_corpus(
            size=1024,
            seed=1,
            block_seq_len=256,
            mix_passkey=0.5,
            mix_kv=0.3,
            mix_copy=0.0,
        )


def test_generate_cognition_corpus_rejects_negative_weight() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        generate_cognition_corpus(
            size=1024,
            seed=1,
            block_seq_len=256,
            mix_passkey=1.2,
            mix_kv=-0.1,
            mix_copy=-0.1,
        )


def test_generate_cognition_corpus_rejects_block_len_over_max() -> None:
    with pytest.raises(ValueError, match="MAX_BLOCK_SEQ_LEN"):
        generate_cognition_corpus(size=4096, seed=1, block_seq_len=8192)


def test_split_train_val_preserves_total_size() -> None:
    corpus = generate_cognition_corpus(size=40_000, seed=99, block_seq_len=256)
    train, val = split_train_val(corpus, val_fraction=0.05)
    assert len(train) + len(val) == len(corpus)
    assert len(val) >= 2048
    assert train + val == corpus


def test_split_train_val_rejects_invalid_fraction() -> None:
    corpus = b"\x00" * 1024
    with pytest.raises(ValueError):
        split_train_val(corpus, val_fraction=0.0)
    with pytest.raises(ValueError):
        split_train_val(corpus, val_fraction=0.6)


def test_generate_cognition_corpus_query_targets_match_stored_value() -> None:
    corpus = generate_cognition_corpus(
        size=20_000,
        seed=42,
        block_seq_len=256,
        mix_passkey=1.0,
        mix_kv=0.0,
        mix_copy=0.0,
        passkey_length=5,
    )
    b = bytes(corpus)
    i = 0
    matched = 0
    checked = 0
    while i < len(b) - 20:
        if b[i : i + 3] == bytes(MARK_STORE_START):
            passkey = b[i + 3 : i + 8]
            end_idx = i + 8
            query_idx = b.find(bytes(MARK_QUERY), end_idx)
            if query_idx < 0 or query_idx + 3 + 5 > len(b):
                break
            recalled = b[query_idx + 3 : query_idx + 8]
            if passkey == recalled:
                matched += 1
            checked += 1
            i = query_idx + 3 + 5
        else:
            i += 1
    assert checked > 10
    assert matched == checked, f"passkey/recall mismatch: {matched}/{checked}"
