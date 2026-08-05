"""Pure NumPy tests for aligned PBF and Triangle-Q batches."""

import numpy as np
import pytest

from utils.datasets import Dataset, PathBridgerDataset


BATCH_KEYS = {
    'observations', 'next_observations', 'actions', 'trajectory',
    'endpoint_goals', 'endpoint_targets', 'value_goals', 'value_offsets',
    'action_chunk_actions', 'valids', 'trl_base_goals', 'trl_base_offsets',
    'trl_split_observations', 'trl_split_goals',
    'trl_split_action_chunk_actions', 'trl_split_offsets', 'trl_valid_mask',
}


def _config(**overrides):
    config = dict(
        horizon=5,
        sequence_horizon=5,
        action_chunk_horizon=5,
        discount=0.9,
        actor_p=(0.0, 0.0, 1.0, 0.0),
        value_geom_sample=False,
    )
    config.update(overrides)
    return config


def _dataset():
    ids = np.arange(32, dtype=np.float32)
    terminals = np.zeros(32, dtype=np.float32)
    terminals[[15, 31]] = 1.0
    return Dataset.create(
        observations=np.stack((ids, ids + 100), axis=-1),
        actions=np.stack((ids / 10, -ids / 10), axis=-1),
        terminals=terminals,
    )


def _ids(array):
    return np.asarray(array)[..., 0].astype(np.int64)


def test_batch_shapes_and_triangle_split_are_episode_safe():
    np.random.seed(7)
    sampler = PathBridgerDataset(_dataset(), _config())
    batch = sampler.sample(128)
    assert set(batch) == BATCH_KEYS
    assert batch['trajectory'].shape == (128, 6, 2)
    assert batch['action_chunk_actions'].shape == (128, 10)
    assert batch['trl_split_action_chunk_actions'].shape == (128, 10)
    assert batch['valids'].shape == (128, 5)

    starts = _ids(batch['observations'])
    goals = _ids(batch['value_goals'])
    splits = _ids(batch['trl_split_observations'])
    valid = batch['trl_valid_mask'].astype(bool)
    finals = np.where(starts <= 15, 15, 31)
    assert np.all(goals > starts)
    assert np.all(goals <= finals)
    assert np.all(splits[valid] > starts[valid])
    assert np.all(splits[valid] < goals[valid])
    assert np.all(splits[valid] + 5 <= finals[valid])
    np.testing.assert_array_equal(batch['trl_split_goals'], batch['trl_split_observations'])


def test_triangle_base_offsets_and_targets_stay_on_trajectory():
    sampler = PathBridgerDataset(_dataset(), _config(sequence_horizon=8))
    batch = sampler.sample(2, idxs=np.asarray([0, 16]))
    base_ids = _ids(batch['trl_base_goals'])
    starts = _ids(batch['observations'])
    offsets = batch['trl_base_offsets'].astype(np.int64)
    np.testing.assert_array_equal(base_ids, starts + offsets)
    assert np.all((offsets >= 1) & (offsets <= 8))


def test_valid_starts_use_max_path_and_sequence_horizons():
    sampler = PathBridgerDataset(_dataset(), _config(horizon=5, sequence_horizon=10))
    np.testing.assert_array_equal(sampler.valid_starts, np.asarray([0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21]))
    with pytest.raises(ValueError, match='cannot cross'):
        sampler.sample(1, idxs=np.asarray([6]))


@pytest.mark.parametrize('value,message', [
    ((0, 1, 0), 'must be'),
    ((0, -0.1, 1.1, 0), 'non-negative'),
    ((0, 0.4, 0, 0.5), 'sum to one'),
    ((0, 0.5, 0.5, 0), 'cannot combine'),
])
def test_endpoint_goal_mix_validation(value, message):
    with pytest.raises(ValueError, match=message):
        PathBridgerDataset(_dataset(), _config(actor_p=value))


def test_rejects_short_triangle_horizons():
    with pytest.raises(ValueError, match='horizon'):
        PathBridgerDataset(_dataset(), _config(horizon=4))
    with pytest.raises(ValueError, match='sequence_horizon'):
        PathBridgerDataset(_dataset(), _config(sequence_horizon=4))
