from __future__ import annotations

import math
from typing import Any

import numpy as np

DEFAULT_SPEED_VALUES = (1, 2, 3)
ATTRIBUTE_NAMES = ("color", "shape", "pos")


def _bits_for_cardinality(cardinality: int) -> int:
    if cardinality <= 1:
        return 1
    return int(math.ceil(math.log2(cardinality)))


def generate_identity_bank(
    rng: np.random.Generator,
    n_identities: int,
    n_colors: int = 4,
    n_shapes: int = 4,
    speed_values: tuple[int, ...] = DEFAULT_SPEED_VALUES,
) -> dict[str, np.ndarray]:
    if n_identities <= 0:
        raise ValueError("n_identities must be > 0")
    if n_colors <= 0 or n_shapes <= 0:
        raise ValueError("n_colors and n_shapes must be > 0")
    if n_identities > n_colors * n_shapes:
        raise ValueError("n_identities must not exceed n_colors * n_shapes")
    if not speed_values:
        raise ValueError("speed_values must not be empty")
    combos = np.array([(c, s) for c in range(n_colors) for s in range(n_shapes)], dtype=np.int64)
    rng.shuffle(combos)
    selected = combos[:n_identities]
    speeds = rng.choice(np.array(speed_values, dtype=np.int64), size=n_identities, replace=True)
    return {
        "color": selected[:, 0].astype(np.int64),
        "shape": selected[:, 1].astype(np.int64),
        "speed": speeds.astype(np.int64),
    }


def candidate_ids_for_cue(identity_bank: dict[str, np.ndarray], cue_color: int, cue_shape: int) -> np.ndarray:
    mask = np.ones(identity_bank["color"].shape[0], dtype=bool)
    if cue_color >= 0:
        mask &= identity_bank["color"] == int(cue_color)
    if cue_shape >= 0:
        mask &= identity_bank["shape"] == int(cue_shape)
    return np.nonzero(mask)[0].astype(np.int64)


def time_to_boundary(position: int, velocity: int, track_length: int) -> float:
    if velocity == 0:
        return float("inf")
    if velocity > 0:
        return float(track_length - 1 - position) / float(velocity)
    return float(position) / float(-velocity)


def _advance_positions(
    positions: np.ndarray,
    velocities: np.ndarray,
    track_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    next_positions = positions + velocities
    next_velocities = velocities.copy()
    out_of_bounds = (next_positions < 0) | (next_positions > track_length - 1)
    next_velocities[out_of_bounds] *= -1
    next_positions = np.clip(positions + next_velocities, 0, track_length - 1)
    return next_positions.astype(np.int64), next_velocities.astype(np.int64)


def _choose_focus_step(
    rng: np.random.Generator,
    visible: np.ndarray,
    focus_idx: int,
) -> int:
    visible_steps = np.nonzero(visible[:, focus_idx])[0]
    if visible_steps.size == 0:
        return 0
    return int(rng.choice(visible_steps))


def _ensure_cue_features(
    rng: np.random.Generator,
    obs_color: np.ndarray,
    obs_shape: np.ndarray,
    truth_color: int,
    truth_shape: int,
    step_idx: int,
    object_idx: int,
) -> None:
    if obs_color[step_idx, object_idx] >= 0 or obs_shape[step_idx, object_idx] >= 0:
        return
    if rng.random() < 0.5:
        obs_color[step_idx, object_idx] = truth_color
    else:
        obs_shape[step_idx, object_idx] = truth_shape


def _select_imagination_pair(
    active_ids: np.ndarray,
    identity_bank: dict[str, np.ndarray],
) -> tuple[int, int, bool]:
    colors = identity_bank["color"][active_ids]
    shapes = identity_bank["shape"][active_ids]
    existing_pairs = {
        (int(color), int(shape))
        for color, shape in zip(identity_bank["color"].tolist(), identity_bank["shape"].tolist(), strict=False)
    }
    novel_pairs: list[tuple[int, int]] = []
    all_pairs: list[tuple[int, int]] = []
    for i in range(len(active_ids)):
        for j in range(len(active_ids)):
            if i == j:
                continue
            pair = (i, j)
            all_pairs.append(pair)
            if (int(colors[i]), int(shapes[j])) not in existing_pairs:
                novel_pairs.append(pair)
    if novel_pairs:
        i, j = novel_pairs[0]
        return int(i), int(j), True
    i, j = all_pairs[0]
    return int(i), int(j), False


def _select_reasoning_pair(
    active_ids: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    track_length: int,
) -> tuple[int, int, float]:
    times = np.array(
        [time_to_boundary(int(pos), int(vel), track_length) for pos, vel in zip(positions, velocities, strict=False)],
        dtype=np.float64,
    )
    best_i = 0
    best_j = 1
    best_margin = -1.0
    for i in range(len(active_ids)):
        for j in range(i + 1, len(active_ids)):
            margin = abs(float(times[i] - times[j]))
            if margin > best_margin:
                best_i = i
                best_j = j
                best_margin = margin
    return int(best_i), int(best_j), float(best_margin)


def generate_nm_world_episode(
    seed: int,
    seq_len: int = 12,
    n_identities: int = 12,
    n_active: int = 4,
    track_length: int = 21,
    n_colors: int = 4,
    n_shapes: int = 4,
    occlusion_prob: float = 0.25,
    feature_dropout_prob: float = 0.35,
    position_noise: int = 1,
    speed_values: tuple[int, ...] = DEFAULT_SPEED_VALUES,
) -> dict[str, Any]:
    if seq_len < 3:
        raise ValueError("seq_len must be >= 3")
    if n_active < 2:
        raise ValueError("n_active must be >= 2")
    if n_active > n_identities:
        raise ValueError("n_active must not exceed n_identities")
    if track_length < 4:
        raise ValueError("track_length must be >= 4")
    if not 0.0 <= occlusion_prob < 1.0:
        raise ValueError("occlusion_prob must be in [0, 1)")
    if not 0.0 <= feature_dropout_prob < 1.0:
        raise ValueError("feature_dropout_prob must be in [0, 1)")
    if position_noise < 0:
        raise ValueError("position_noise must be >= 0")

    rng = np.random.default_rng(seed)
    identity_bank = generate_identity_bank(
        rng,
        n_identities=n_identities,
        n_colors=n_colors,
        n_shapes=n_shapes,
        speed_values=speed_values,
    )
    active_ids = rng.choice(np.arange(n_identities, dtype=np.int64), size=n_active, replace=False)
    active_colors = identity_bank["color"][active_ids]
    active_shapes = identity_bank["shape"][active_ids]
    base_speeds = identity_bank["speed"][active_ids]
    active_velocities = base_speeds * rng.choice(np.array([-1, 1], dtype=np.int64), size=n_active, replace=True)
    current_positions = rng.integers(1, track_length - 2, size=n_active, dtype=np.int64)
    positions = np.zeros((seq_len, n_active), dtype=np.int64)
    velocities = np.zeros((seq_len, n_active), dtype=np.int64)
    for step_idx in range(seq_len):
        positions[step_idx] = current_positions
        velocities[step_idx] = active_velocities
        if step_idx < seq_len - 1:
            current_positions, active_velocities = _advance_positions(current_positions, active_velocities, track_length)

    visible = rng.random((seq_len, n_active)) >= occlusion_prob
    obs_color = np.where(
        visible & (rng.random((seq_len, n_active)) >= feature_dropout_prob),
        active_colors[None, :],
        -1,
    ).astype(np.int64)
    obs_shape = np.where(
        visible & (rng.random((seq_len, n_active)) >= feature_dropout_prob),
        active_shapes[None, :],
        -1,
    ).astype(np.int64)
    noise = (
        rng.integers(-position_noise, position_noise + 1, size=(seq_len, n_active), dtype=np.int64)
        if position_noise > 0
        else np.zeros((seq_len, n_active), dtype=np.int64)
    )
    obs_pos = np.where(
        visible,
        np.clip(positions + noise, 0, track_length - 1),
        -1,
    ).astype(np.int64)

    focus_idx = int(rng.integers(0, n_active))
    recognition_time = _choose_focus_step(rng, visible, focus_idx)
    visible[recognition_time, focus_idx] = True
    if obs_pos[recognition_time, focus_idx] < 0:
        obs_pos[recognition_time, focus_idx] = positions[recognition_time, focus_idx]
    _ensure_cue_features(
        rng,
        obs_color,
        obs_shape,
        int(active_colors[focus_idx]),
        int(active_shapes[focus_idx]),
        recognition_time,
        focus_idx,
    )
    recognition_candidates = candidate_ids_for_cue(
        identity_bank,
        int(obs_color[recognition_time, focus_idx]),
        int(obs_shape[recognition_time, focus_idx]),
    )

    recollection_source_time = int(rng.integers(0, seq_len - 1))
    recollection_query_time = int(rng.integers(recollection_source_time + 1, seq_len))
    recollection_attr = ATTRIBUTE_NAMES[int(rng.integers(0, len(ATTRIBUTE_NAMES)))]
    if recollection_attr == "color":
        recollection_target = int(active_colors[focus_idx])
    elif recollection_attr == "shape":
        recollection_target = int(active_shapes[focus_idx])
    else:
        recollection_target = int(positions[recollection_source_time, focus_idx])

    prediction_time = int(rng.integers(0, seq_len - 1))
    prediction_target = int(positions[prediction_time + 1, focus_idx])

    raw_color_bits = int((obs_color >= 0).sum()) * _bits_for_cardinality(n_colors + 1)
    raw_shape_bits = int((obs_shape >= 0).sum()) * _bits_for_cardinality(n_shapes + 1)
    raw_pos_bits = int((obs_pos >= 0).sum()) * _bits_for_cardinality(track_length + 1)
    raw_bits = raw_color_bits + raw_shape_bits + raw_pos_bits
    latent_bits = int(n_active) * (
        _bits_for_cardinality(n_identities)
        + _bits_for_cardinality(track_length)
        + _bits_for_cardinality(2 * max(abs(int(v)) for v in speed_values) + 1)
    )
    compression_ratio = float(raw_bits) / float(max(latent_bits, 1))

    imagination_parent_a, imagination_parent_b, imagination_is_novel = _select_imagination_pair(active_ids, identity_bank)
    imagination_child_color = int(active_colors[imagination_parent_a])
    imagination_child_shape = int(active_shapes[imagination_parent_b])
    imagination_time = int(rng.integers(0, seq_len))

    reasoning_time = int(rng.integers(0, seq_len))
    reasoning_a, reasoning_b, reasoning_margin = _select_reasoning_pair(
        active_ids,
        positions[reasoning_time],
        velocities[reasoning_time],
        track_length,
    )
    reasoning_a_time = time_to_boundary(
        int(positions[reasoning_time, reasoning_a]),
        int(velocities[reasoning_time, reasoning_a]),
        track_length,
    )
    reasoning_b_time = time_to_boundary(
        int(positions[reasoning_time, reasoning_b]),
        int(velocities[reasoning_time, reasoning_b]),
        track_length,
    )
    reasoning_winner_local = reasoning_a if reasoning_a_time < reasoning_b_time else reasoning_b

    return {
        "seed": int(seed),
        "identity_bank": identity_bank,
        "active_ids": active_ids.astype(np.int64),
        "active_colors": active_colors.astype(np.int64),
        "active_shapes": active_shapes.astype(np.int64),
        "positions": positions,
        "velocities": velocities,
        "observations": {
            "visible": visible.astype(np.int64),
            "color": obs_color,
            "shape": obs_shape,
            "pos": obs_pos,
        },
        "tasks": {
            "recognition": {
                "time": int(recognition_time),
                "focus_local_index": int(focus_idx),
                "target_identity": int(active_ids[focus_idx]),
                "cue_color": int(obs_color[recognition_time, focus_idx]),
                "cue_shape": int(obs_shape[recognition_time, focus_idx]),
                "cue_pos": int(obs_pos[recognition_time, focus_idx]),
                "candidate_ids": recognition_candidates.astype(np.int64),
                "candidate_count": int(recognition_candidates.shape[0]),
            },
            "recollection": {
                "source_time": int(recollection_source_time),
                "query_time": int(recollection_query_time),
                "focus_local_index": int(focus_idx),
                "attribute": recollection_attr,
                "target": int(recollection_target),
                "lag": int(recollection_query_time - recollection_source_time),
            },
            "prediction": {
                "time": int(prediction_time),
                "focus_local_index": int(focus_idx),
                "current_pos": int(positions[prediction_time, focus_idx]),
                "current_vel": int(velocities[prediction_time, focus_idx]),
                "target_next_pos": int(prediction_target),
                "step_distance": int(abs(prediction_target - int(positions[prediction_time, focus_idx]))),
            },
            "compression": {
                "raw_bits": int(raw_bits),
                "latent_bits": int(latent_bits),
                "ratio": float(compression_ratio),
            },
            "imagination": {
                "time": int(imagination_time),
                "parent_a_local_index": int(imagination_parent_a),
                "parent_b_local_index": int(imagination_parent_b),
                "child_color": int(imagination_child_color),
                "child_shape": int(imagination_child_shape),
                "child_pos": int(positions[imagination_time, imagination_parent_a]),
                "child_vel": int(velocities[imagination_time, imagination_parent_a]),
                "is_novel": bool(imagination_is_novel),
                "is_plausible": True,
            },
            "reasoning": {
                "time": int(reasoning_time),
                "object_a_local_index": int(reasoning_a),
                "object_b_local_index": int(reasoning_b),
                "winner_local_index": int(reasoning_winner_local),
                "winner_identity": int(active_ids[reasoning_winner_local]),
                "margin": float(reasoning_margin),
                "object_a_time_to_boundary": float(reasoning_a_time),
                "object_b_time_to_boundary": float(reasoning_b_time),
            },
        },
    }


def generate_nm_world_batch(
    n_episodes: int,
    seed: int,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    if n_episodes <= 0:
        raise ValueError("n_episodes must be > 0")
    rng = np.random.default_rng(seed)
    episodes: list[dict[str, Any]] = []
    for _ in range(n_episodes):
        episode_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        episodes.append(generate_nm_world_episode(seed=episode_seed, **kwargs))
    return episodes
