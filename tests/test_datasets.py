"""Pure-NumPy tests for the compact PathBridger training sampler."""

from __future__ import annotations

import numpy as np
import pytest

from utils.datasets import Dataset, PathBridgerDataset


BATCH_KEYS = {
    "observations",
    "next_observations",
    "actions",
    "trajectory",
    "endpoint_goals",
    "endpoint_targets",
    "value_goals",
    "value_offsets",
    "base_goals",
    "base_offsets",
    "transitive_subgoals",
    "transitive_offsets",
    "transitive_valids",
}


def _sampler_config(**overrides):
    config = {
        "horizon": 5,
        "discount": 0.99,
        "actor_p": (0.0, 0.0, 1.0, 0.0),
        "critic_p": (0.0, 1.0, 0.0, 0.0),
    }
    config.update(overrides)
    return config


def _two_episode_dataset() -> Dataset:
    """Episodes occupy indices 0..6 and 7..14."""

    state_ids = np.arange(15, dtype=np.float32)
    observations = np.stack((state_ids, state_ids + 100.0), axis=-1)
    actions = state_ids[:, None] / 10.0
    terminals = np.zeros(15, dtype=np.float32)
    terminals[[6, 14]] = 1.0
    return Dataset.create(
        observations=observations,
        actions=actions,
        terminals=terminals,
    )


def _state_ids(values: np.ndarray) -> np.ndarray:
    return np.asarray(values)[..., 0].astype(np.int64)


def test_batch_contract_and_close_goal_clip_pad(monkeypatch):
    """The current research sampler pads close goals and clips distant ones at K."""

    sampler = PathBridgerDataset(
        _two_episode_dataset(),
        _sampler_config(),
    )

    # sample() calls random() for endpoint interpolation, base offsets, then
    # the one valid transitive split.  These values deliberately exercise a
    # close endpoint in row 0 and an endpoint beyond K in row 1.
    random_results = iter(
        (
            np.asarray([1.0, 0.0]),       # endpoint goals: indices 2 and 14
            np.asarray([0.0, 0.999]),     # base offsets: 1 and 5
            np.asarray([0.4]),            # long-pair split: offset 3
        )
    )
    monkeypatch.setattr(np.random, "random", lambda size: next(random_results))
    monkeypatch.setattr(
        np.random,
        "geometric",
        lambda p, size: np.asarray([3, 6], dtype=np.int64),
    )

    batch = sampler.sample(batch_size=2, idxs=np.asarray([1, 8]))

    assert set(batch) == BATCH_KEYS
    assert batch["observations"].shape == (2, 2)
    assert batch["actions"].shape == (2, 1)
    assert batch["trajectory"].shape == (2, 6, 2)
    assert all(np.issubdtype(value.dtype, np.floating) for value in batch.values())

    np.testing.assert_array_equal(_state_ids(batch["observations"]), [1, 8])
    np.testing.assert_array_equal(_state_ids(batch["next_observations"]), [2, 9])
    np.testing.assert_array_equal(_state_ids(batch["endpoint_goals"]), [2, 14])
    np.testing.assert_array_equal(_state_ids(batch["endpoint_targets"]), [2, 13])

    # Close goal: advance once, then pad at that goal through K.
    np.testing.assert_array_equal(
        _state_ids(batch["trajectory"][0]),
        [1, 2, 2, 2, 2, 2],
    )
    # Distant goal: retain the full K-window and clip before the final goal.
    np.testing.assert_array_equal(
        _state_ids(batch["trajectory"][1]),
        [8, 9, 10, 11, 12, 13],
    )

    np.testing.assert_array_equal(_state_ids(batch["value_goals"]), [4, 14])
    np.testing.assert_array_equal(batch["value_offsets"], [3.0, 6.0])
    np.testing.assert_array_equal(_state_ids(batch["base_goals"]), [2, 13])
    np.testing.assert_array_equal(batch["base_offsets"], [1.0, 5.0])
    np.testing.assert_array_equal(batch["transitive_valids"], [0.0, 1.0])
    np.testing.assert_array_equal(batch["transitive_offsets"], [0.0, 3.0])
    np.testing.assert_array_equal(
        _state_ids(batch["transitive_subgoals"]),
        [1, 11],
    )


def test_valid_starts_and_explicit_windows_never_cross_episode_boundaries():
    sampler = PathBridgerDataset(
        _two_episode_dataset(),
        _sampler_config(),
    )

    np.testing.assert_array_equal(sampler.valid_starts, [0, 1, 7, 8, 9])

    with pytest.raises(ValueError, match="cannot cross an episode boundary"):
        sampler.sample(batch_size=1, idxs=np.asarray([2]))
    with pytest.raises(ValueError, match="cannot cross an episode boundary"):
        sampler.sample(batch_size=1, idxs=np.asarray([10]))


def test_every_sampled_index_stays_inside_its_source_episode():
    np.random.seed(7)
    sampler = PathBridgerDataset(
        _two_episode_dataset(),
        _sampler_config(),
    )
    batch = sampler.sample(batch_size=256)

    starts = _state_ids(batch["observations"])
    episode_starts = np.where(starts <= 6, 0, 7)
    episode_finals = np.where(starts <= 6, 6, 14)
    indexed_fields = (
        "next_observations",
        "trajectory",
        "endpoint_goals",
        "endpoint_targets",
        "value_goals",
        "base_goals",
        "transitive_subgoals",
    )
    for field in indexed_fields:
        ids = _state_ids(batch[field])
        lower = episode_starts.reshape((-1,) + (1,) * (ids.ndim - 1))
        upper = episode_finals.reshape((-1,) + (1,) * (ids.ndim - 1))
        assert np.all(ids >= lower), field
        assert np.all(ids <= upper), field

    endpoint_ids = _state_ids(batch["endpoint_goals"])
    endpoint_target_ids = _state_ids(batch["endpoint_targets"])
    assert np.all(endpoint_ids > starts)
    np.testing.assert_array_equal(
        endpoint_target_ids,
        np.minimum(starts + 5, endpoint_ids),
    )

    trajectory_ids = _state_ids(batch["trajectory"])
    expected = np.minimum(
        starts[:, None] + np.arange(6, dtype=np.int64)[None, :],
        endpoint_target_ids[:, None],
    )
    np.testing.assert_array_equal(trajectory_ids, expected)


def test_dataset_rejects_missing_final_terminal():
    observations = np.zeros((6, 2), dtype=np.float32)
    terminals = np.zeros(6, dtype=np.float32)
    terminals[3] = 1.0
    dataset = Dataset.create(
        observations=observations,
        actions=np.zeros((6, 1), dtype=np.float32),
        terminals=terminals,
    )
    with pytest.raises(ValueError, match="final compact-dataset observation"):
        PathBridgerDataset(dataset, _sampler_config())


def test_four_tuple_goal_mixes_select_geometric_and_trajectory_future(monkeypatch):
    sampler = PathBridgerDataset(
        _two_episode_dataset(),
        _sampler_config(
            actor_p=(0.0, 1.0, 0.0, 0.0),
            critic_p=(0.0, 0.0, 1.0, 0.0),
        ),
    )

    random_results = iter(
        (
            np.asarray([1.0, 0.0]),       # critic trajectory goals: 2 and 14
            np.asarray([0.0, 0.999]),     # base offsets: 1 and 5
            np.asarray([0.4]),            # long-pair split: offset 3
        )
    )
    monkeypatch.setattr(np.random, "random", lambda size: next(random_results))
    monkeypatch.setattr(
        np.random,
        "geometric",
        lambda p, size: np.asarray([2, 10], dtype=np.int64),
    )

    batch = sampler.sample(batch_size=2, idxs=np.asarray([1, 8]))

    np.testing.assert_array_equal(_state_ids(batch["endpoint_goals"]), [3, 14])
    np.testing.assert_array_equal(_state_ids(batch["endpoint_targets"]), [3, 13])
    np.testing.assert_array_equal(_state_ids(batch["value_goals"]), [2, 14])
    np.testing.assert_array_equal(batch["value_offsets"], [1.0, 6.0])
    np.testing.assert_array_equal(
        _state_ids(batch["trajectory"][0]),
        [1, 2, 3, 3, 3, 3],
    )


def test_random_actor_goal_keeps_endpoint_target_on_source_trajectory(monkeypatch):
    dataset = _two_episode_dataset()
    sampler = PathBridgerDataset(
        dataset,
        _sampler_config(actor_p=(0.0, 0.0, 0.0, 1.0)),
    )

    monkeypatch.setattr(
        dataset,
        "get_random_idxs",
        lambda size: np.asarray([14, 0], dtype=np.int64),
    )
    monkeypatch.setattr(
        np.random,
        "geometric",
        lambda p, size: np.asarray([3, 6], dtype=np.int64),
    )
    random_results = iter(
        (
            np.asarray([0.0, 0.999]),     # base offsets: 1 and 5
            np.asarray([0.4]),            # long-pair split: offset 3
        )
    )
    monkeypatch.setattr(np.random, "random", lambda size: next(random_results))

    batch = sampler.sample(batch_size=2, idxs=np.asarray([1, 8]))

    np.testing.assert_array_equal(_state_ids(batch["endpoint_goals"]), [14, 0])
    np.testing.assert_array_equal(_state_ids(batch["endpoint_targets"]), [6, 13])
    np.testing.assert_array_equal(
        _state_ids(batch["trajectory"]),
        [[1, 2, 3, 4, 5, 6], [8, 9, 10, 11, 12, 13]],
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("actor_p", (0.0, 1.0, 0.0), "4-tuple"),
        ("actor_p", (0.0, -0.1, 1.1, 0.0), "non-negative"),
        ("actor_p", (0.0, np.nan, 1.0, 0.0), "finite"),
        ("actor_p", (0.0, 0.4, 0.0, 0.5), "sum to 1"),
        (
            "actor_p",
            (0.0, 0.5, 0.5, 0.0),
            "geometric and ordinary trajectory-future",
        ),
        (
            "critic_p",
            (0.1, 0.9, 0.0, 0.0),
            "ordered future-goal component",
        ),
        (
            "critic_p",
            (0.0, 0.9, 0.0, 0.1),
            "ordered future-goal component",
        ),
    ),
)
def test_goal_mix_validation(field, value, message):
    with pytest.raises(ValueError, match=message):
        PathBridgerDataset(
            _two_episode_dataset(),
            _sampler_config(**{field: value}),
        )
