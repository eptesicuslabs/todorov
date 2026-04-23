import numpy as np

from neuroloc.data.nm_worlds import (
    candidate_ids_for_cue,
    generate_nm_world_batch,
    generate_nm_world_episode,
    time_to_boundary,
)


def test_generate_nm_world_episode_is_deterministic() -> None:
    first = generate_nm_world_episode(seed=7, seq_len=10, n_identities=8, n_active=3, track_length=17)
    second = generate_nm_world_episode(seed=7, seq_len=10, n_identities=8, n_active=3, track_length=17)
    assert np.array_equal(first["positions"], second["positions"])
    assert np.array_equal(first["velocities"], second["velocities"])
    assert first["tasks"]["recollection"] == second["tasks"]["recollection"]
    assert first["tasks"]["prediction"] == second["tasks"]["prediction"]


def test_recognition_target_belongs_to_candidate_set() -> None:
    episode = generate_nm_world_episode(seed=11, seq_len=10, n_identities=8, n_active=3, track_length=17)
    recognition = episode["tasks"]["recognition"]
    candidate_ids = recognition["candidate_ids"]
    assert recognition["target_identity"] in set(candidate_ids.tolist())
    recomputed = candidate_ids_for_cue(
        episode["identity_bank"],
        recognition["cue_color"],
        recognition["cue_shape"],
    )
    assert np.array_equal(candidate_ids, recomputed)


def test_recollection_target_matches_source_state() -> None:
    episode = generate_nm_world_episode(seed=21, seq_len=12, n_identities=10, n_active=4, track_length=19)
    recollection = episode["tasks"]["recollection"]
    focus_idx = recollection["focus_local_index"]
    source_time = recollection["source_time"]
    active_ids = episode["active_ids"]
    identity_bank = episode["identity_bank"]
    if recollection["attribute"] == "color":
        expected = int(identity_bank["color"][active_ids[focus_idx]])
    elif recollection["attribute"] == "shape":
        expected = int(identity_bank["shape"][active_ids[focus_idx]])
    else:
        expected = int(episode["positions"][source_time, focus_idx])
    assert recollection["target"] == expected
    assert recollection["lag"] == recollection["query_time"] - recollection["source_time"]


def test_prediction_target_matches_next_position() -> None:
    episode = generate_nm_world_episode(seed=33, seq_len=9, n_identities=8, n_active=3, track_length=17)
    prediction = episode["tasks"]["prediction"]
    focus_idx = prediction["focus_local_index"]
    time_idx = prediction["time"]
    expected = int(episode["positions"][time_idx + 1, focus_idx])
    assert prediction["target_next_pos"] == expected


def test_compression_ratio_exceeds_one_for_temporal_episode() -> None:
    episode = generate_nm_world_episode(seed=44, seq_len=12, n_identities=8, n_active=3, track_length=17)
    compression = episode["tasks"]["compression"]
    assert compression["raw_bits"] > compression["latent_bits"]
    assert compression["ratio"] > 1.0


def test_imagination_marks_novel_pair_when_present() -> None:
    episode = generate_nm_world_episode(seed=55, seq_len=10, n_identities=8, n_active=4, track_length=17, n_colors=4, n_shapes=4)
    imagination = episode["tasks"]["imagination"]
    existing_pairs = {
        (int(color), int(shape))
        for color, shape in zip(
            episode["identity_bank"]["color"].tolist(),
            episode["identity_bank"]["shape"].tolist(),
            strict=False,
        )
    }
    pair = (imagination["child_color"], imagination["child_shape"])
    assert imagination["is_plausible"] is True
    assert imagination["is_novel"] == (pair not in existing_pairs)


def test_reasoning_winner_matches_time_to_boundary() -> None:
    episode = generate_nm_world_episode(seed=66, seq_len=10, n_identities=8, n_active=4, track_length=17)
    reasoning = episode["tasks"]["reasoning"]
    time_idx = reasoning["time"]
    a = reasoning["object_a_local_index"]
    b = reasoning["object_b_local_index"]
    pos = episode["positions"][time_idx]
    vel = episode["velocities"][time_idx]
    a_time = time_to_boundary(int(pos[a]), int(vel[a]), 17)
    b_time = time_to_boundary(int(pos[b]), int(vel[b]), 17)
    expected_local = a if a_time < b_time else b
    assert reasoning["winner_local_index"] == expected_local
    assert reasoning["margin"] == abs(a_time - b_time)


def test_generate_nm_world_batch_respects_count_and_seed() -> None:
    first = generate_nm_world_batch(4, seed=77, seq_len=8, n_identities=8, n_active=3, track_length=17)
    second = generate_nm_world_batch(4, seed=77, seq_len=8, n_identities=8, n_active=3, track_length=17)
    assert len(first) == 4
    assert len(second) == 4
    assert [episode["seed"] for episode in first] == [episode["seed"] for episode in second]
