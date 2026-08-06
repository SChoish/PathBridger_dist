from __future__ import annotations

from types import SimpleNamespace

import jax
import numpy as np
import pytest

from agents.online_idm import parameter_digest
from agents.pixel_lapo import PixelLAPOAgent, get_config
from envs.env_utils import _as_pixel_dataset, make_pixel_env_and_datasets
from scripts.make_pixel_lapo_manifest import ENVIRONMENTS, SUITES, build_rows
from utils.pixel_data import ActionFreePixelTrajectoryData, PixelReplayBuffer
from utils.pixel_evaluation import evaluate_pixel_policy


def _pixels(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (6, 32, 32, 3), dtype=np.uint8)


def _config():
    config = get_config().to_dict()
    config.update(
        feature_dim=16,
        hidden_dims=(16,),
        num_codebooks=2,
        num_codes=8,
        code_dim=4,
        offline_batch_size=2,
        online_batch_size=2,
    )
    return config


def test_pixel_loader_and_sampler_enforce_action_free_uint8_boundary():
    raw = {
        'observations': _pixels(),
        'terminals': np.array([0, 0, 1, 0, 0, 1], np.float32),
        'actions': np.ones((6, 2), np.float32),
        'rewards': np.ones(6, np.float32),
        'qpos': np.ones((6, 4), np.float32),
    }
    dataset = _as_pixel_dataset(raw, split='training')
    assert set(dataset) == {'observations', 'terminals'}
    assert dataset['observations'].dtype == np.uint8
    data = ActionFreePixelTrajectoryData(dataset, seed=3)
    assert data.offline_fields_seen == ('observations', 'terminals')
    batch = data.sample(64)
    assert set(batch) == {
        'observations',
        'next_observations',
        'goals',
        'rewards',
        'masks',
        'indices',
        'goal_indices',
    }
    assert np.all(batch['goal_indices'] > batch['indices'])
    assert np.all(
        batch['goal_indices'] <= data.episodes.terminal_for_state[batch['indices']]
    )
    np.testing.assert_array_equal(
        batch['rewards'] == 0.0,
        batch['goal_indices'] == batch['indices'] + 1,
    )
    with pytest.raises(ValueError, match='uint8'):
        _as_pixel_dataset(
            {**raw, 'observations': raw['observations'].astype(np.float32)},
            split='training',
        )


def test_compact_double_terminal_markers_keep_last_valid_transitions():
    data = ActionFreePixelTrajectoryData(
        {
            'observations': _pixels(),
            'terminals': np.array([0, 1, 1, 0, 1, 1], np.float32),
        }
    )
    np.testing.assert_array_equal(
        data.episodes.transition_indices,
        np.array([0, 1, 3, 4], np.int64),
    )
    np.testing.assert_array_equal(
        data.episodes.terminal_for_state,
        np.array([2, 2, 2, 5, 5, 5], np.int64),
    )


def test_pixel_loader_does_not_download_without_explicit_opt_in(tmp_path):
    with pytest.raises(FileNotFoundError, match='allow_dataset_download'):
        make_pixel_env_and_datasets(
            'visual-antmaze-medium-navigate-v0',
            dataset_dir=tmp_path,
        )


def test_pixel_replay_preserves_compact_storage_and_samples():
    replay = PixelReplayBuffer(3, (32, 32, 3), (2,), seed=0)
    frame = _pixels()[0]
    replay.add(
        observation=frame,
        action=np.array([0.1, -0.2], np.float32),
        next_observation=frame,
        goal=frame,
        reward=-1.0,
        mask=1.0,
    )
    assert replay.observations.dtype == np.uint8
    assert replay.allocated_bytes == sum(
        array.nbytes
        for array in (
            replay.observations,
            replay.next_observations,
            replay.goals,
            replay.actions,
            replay.rewards,
            replay.masks,
        )
    )
    batch = replay.sample(2)
    assert batch['observations'].shape == (2, 32, 32, 3)
    assert batch['actions'].shape == (2, 2)


def test_pixel_lapo_updates_only_the_declared_stage_module():
    images = _pixels()
    data = ActionFreePixelTrajectoryData(
        {
            'observations': images,
            'terminals': np.array([0, 0, 1, 0, 0, 1], np.float32),
            'actions': np.ones((6, 2), np.float32),
        },
        seed=0,
    )
    batch = data.sample(2)
    agent = PixelLAPOAgent.create(0, images[:2], 2, _config())

    policy_before = parameter_digest(agent.network.params['modules_latent_policy'])
    decoder_before = parameter_digest(agent.network.params['modules_decoder'])
    latent_before = parameter_digest(agent.network.params['modules_latent_model'])
    agent, stage1_info = agent.offline_update(batch, stage=1)
    assert np.isfinite(float(stage1_info['loss/total']))
    assert parameter_digest(agent.network.params['modules_latent_model']) != latent_before
    assert parameter_digest(agent.network.params['modules_latent_policy']) == policy_before
    assert parameter_digest(agent.network.params['modules_decoder']) == decoder_before

    latent_after_stage1 = parameter_digest(agent.network.params['modules_latent_model'])
    agent, stage2_info = agent.offline_update(batch, stage=2)
    assert np.isfinite(float(stage2_info['loss/total']))
    assert parameter_digest(agent.network.params['modules_latent_model']) == latent_after_stage1
    assert parameter_digest(agent.network.params['modules_latent_policy']) != policy_before
    assert parameter_digest(agent.network.params['modules_decoder']) == decoder_before

    policy_after_stage2 = parameter_digest(agent.network.params['modules_latent_policy'])
    online_batch = {
        **batch,
        'actions': np.array([[0.2, -0.3], [-0.1, 0.4]], np.float32),
    }
    agent, online_info = agent.online_update(online_batch)
    assert np.isfinite(float(online_info['loss/total']))
    assert parameter_digest(agent.network.params['modules_latent_model']) == latent_after_stage1
    assert parameter_digest(agent.network.params['modules_latent_policy']) == policy_after_stage2
    assert parameter_digest(agent.network.params['modules_decoder']) != decoder_before
    actions = agent.sample_actions(
        images[:2], images[2:4], seed=jax.random.PRNGKey(4), temperature=0.0
    )
    assert actions.shape == (2, 2)
    assert np.all(np.isfinite(actions))


class _PixelEvalEnv:
    def __init__(self):
        self.action_space = SimpleNamespace(
            low=np.full((2,), -1.0, np.float32),
            high=np.full((2,), 1.0, np.float32),
        )
        self.spec = SimpleNamespace(max_episode_steps=2)
        self.frame = np.zeros((32, 32, 3), np.uint8)

    def reset(self, **kwargs):
        return self.frame.copy(), {'goal': self.frame.copy()}

    def step(self, action):
        return self.frame.copy(), 0.0, False, False, {'success': True}


class _ZeroPixelPolicy:
    def sample_actions(self, observations, goals, seed, temperature):
        assert observations.shape == (1, 32, 32, 3)
        assert goals.shape == (1, 32, 32, 3)
        return np.zeros((1, 2), np.float32)


def test_pixel_evaluation_preserves_images_and_any_step_success():
    metrics = evaluate_pixel_policy(
        _ZeroPixelPolicy(),
        _PixelEvalEnv(),
        task_ids=(1, 2),
        episodes_per_task=2,
        seed=0,
    )
    assert metrics['overall_success'] == 1.0
    assert metrics['num_tasks'] == 2


def test_pixel_manifest_is_isolated_and_has_locked_dimensions(tmp_path):
    expected = {'p0_smoke': 3, 'pilot': 12, 'screening': 40}
    for name, count in expected.items():
        rows = build_rows(root=tmp_path, python='python', suite_name=name)
        assert len(rows) == count
        assert {row['algorithm'] for row in rows} == {'gc_pixel_lapo'}
        assert all('train_pixel_lapo.py' in row['command'] for row in rows)
        assert all('train_af.py' not in row['command'] for row in rows)
    assert len(ENVIRONMENTS) == 8
    assert len(SUITES['screening']['seeds']) == 5
