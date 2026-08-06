from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

from utils.af_data import ActionFreeTrajectoryData, OnlineReplayBuffer
from utils.datasets import Dataset, action_free_view
from utils.goal_representation import goal_representation, goal_representation_np
from utils.provenance import AlgorithmMetadata


def _dataset():
    observations = np.arange(30, dtype=np.float32).reshape(10, 3)
    terminals = np.zeros(10, dtype=np.float32)
    terminals[[4, 9]] = 1.0
    return Dataset.create(
        observations=observations,
        terminals=terminals,
        actions=np.ones((10, 2), dtype=np.float32),
        rewards=np.ones(10, dtype=np.float32),
        timeouts=np.ones(10, dtype=np.float32),
    )


def test_action_free_view_and_sampler_never_expose_action_or_logged_reward():
    view = action_free_view(_dataset())
    assert set(view) == {'observations', 'terminals'}
    data = ActionFreeTrajectoryData(
        _dataset(), env_name='antmaze-medium-navigate-v0', seed=3
    )
    batch = data.sample(32, subgoal_steps=2)
    assert 'actions' not in batch
    assert data.offline_fields_seen == ('observations', 'terminals')
    assert np.all(batch['indices'] + 1 <= data.episodes.terminal_for_state[batch['indices']])
    assert not np.any(batch['goal_components'] == 0)
    np.testing.assert_array_equal(
        batch['rewards'] == 0.0,
        batch['indices'] + 1 == batch['goal_indices'],
    )


def test_sequence_sampler_is_left_padded_and_action_free():
    data = ActionFreeTrajectoryData(
        _dataset(), env_name='antmaze-medium-navigate-v0', seed=4
    )
    batch = data.sample_sequences(8, context_length=4)
    assert batch['histories'].shape == (8, 4, 3)
    assert batch['history_masks'].shape == (8, 4)
    assert 'actions' not in batch
    np.testing.assert_array_equal(
        batch['remaining'],
        data.episodes.terminal_for_state[batch['indices']] - batch['indices'],
    )


def test_replay_future_her_stays_inside_episode():
    replay = OnlineReplayBuffer(8, (2,), (1,), seed=0)
    for episode in range(2):
        for step in range(3):
            observation = np.asarray([episode, step], np.float32)
            next_observation = np.asarray([episode, step + 1], np.float32)
            replay.add(
                observation=observation,
                action=np.asarray([0.0], np.float32),
                next_observation=next_observation,
                goal=np.asarray([episode, 3], np.float32),
                reward=-1.0,
                mask=1.0,
                episode_id=episode,
                timestep=step,
            )
    batch = replay.sample(128, her_probability=1.0)
    np.testing.assert_array_equal(batch['observations'][:, 0], batch['goals'][:, 0])
    assert batch['replay/her_relabel_fraction'] == pytest.approx(1.0)
    assert batch['replay/her_success_fraction'] > 0.0


def test_her_includes_immediate_positive_anchor():
    replay = OnlineReplayBuffer(2, (1,), (1,), seed=7)
    replay.add(
        observation=np.asarray([0.0], np.float32),
        action=np.asarray([0.0], np.float32),
        next_observation=np.asarray([1.0], np.float32),
        goal=np.asarray([9.0], np.float32),
        reward=-1.0,
        mask=1.0,
        episode_id=0,
        timestep=0,
    )
    batch = replay.sample(16, her_probability=1.0)
    np.testing.assert_array_equal(batch['goals'], np.ones((16, 1), np.float32))
    np.testing.assert_array_equal(batch['rewards'], np.zeros(16, np.float32))
    np.testing.assert_array_equal(batch['masks'], np.zeros(16, np.float32))
    assert batch['replay/her_success_fraction'] == pytest.approx(1.0)


def _scene_state(index: int) -> np.ndarray:
    observation = np.zeros((40,), dtype=np.float32)
    observation[19:22] = np.asarray([index, index + 0.1, index + 0.2])
    observation[28 + index % 2] = 1.0
    observation[32 + (index // 2) % 2] = 1.0
    observation[36] = index / 10.0
    observation[38] = index / 20.0
    return observation


def test_scene_her_uses_oracle_projector_and_preserves_behavior_goal():
    states = np.stack([_scene_state(index) for index in range(4)])
    terminals = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    data = ActionFreeTrajectoryData(
        Dataset.create(observations=states, terminals=terminals),
        env_name='scene-play-v0',
        seed=0,
    )
    projected_np = goal_representation_np(
        states, mode='phi', env_name='scene-play-v0'
    )
    projected_jax = goal_representation(
        jnp.asarray(states), mode='phi', env_name='scene-play-v0'
    )
    assert projected_np.shape == (4, 7)
    np.testing.assert_array_equal(projected_np, np.asarray(projected_jax))

    replay = OnlineReplayBuffer(4, (40,), (1,), seed=2)
    for step in range(3):
        replay.add(
            observation=states[step],
            action=np.zeros((1,), dtype=np.float32),
            next_observation=states[step + 1],
            goal=states[-1],
            reward=-1.0,
            mask=1.0,
            episode_id=0,
            timestep=step,
        )
    batch = replay.sample(
        512,
        her_probability=1.0,
        goal_projector=data.project_goals,
    )
    np.testing.assert_array_equal(
        batch['behavior_goals'],
        np.broadcast_to(states[-1], batch['behavior_goals'].shape),
    )
    assert 0.0 < batch['replay/her_success_fraction'] < 1.0


def test_provenance_rejects_action_leakage():
    metadata = AlgorithmMetadata(
        algorithm='bad',
        port_kind='official_port',
        paper_url='https://example.test',
        official_repo_url=None,
        official_repo_commit=None,
        offline_fields_seen=('observations', 'actions'),
        online_modules_updated=('policy',),
    )
    with pytest.raises(ValueError, match='violates'):
        metadata.validate()
