from __future__ import annotations

import jax
import numpy as np

from agents.online_idm import parameter_digest
from agents.pixel_registry import (
    PIXEL_ALGORITHMS,
    create_pixel_algorithm,
    get_pixel_config,
    pixel_algorithm_metadata,
)
from scripts.make_pixel_manifest import SUITES, build_rows
from utils.af_checkpoints import load_af_checkpoint, restore_af_agent, save_af_checkpoint
from utils.pixel_data import ActionFreePixelTrajectoryData


def _pixels(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (6, 32, 32, 3), dtype=np.uint8)


def _data(seed=0):
    return ActionFreePixelTrajectoryData(
        {
            'observations': _pixels(seed),
            'terminals': np.array([0, 0, 1, 0, 0, 1], np.float32),
            'actions': np.ones((6, 2), np.float32),
        },
        seed=seed,
    )


def _stacked_data(seed=0):
    return ActionFreePixelTrajectoryData(
        {
            'observations': _pixels(seed),
            'terminals': np.array([0, 0, 1, 0, 0, 1], np.float32),
        },
        seed=seed,
        frame_stack=3,
    )


def _config(name):
    config = get_pixel_config(name)
    config.update(
        feature_dim=16,
        hidden_dims=(16,),
        offline_batch_size=2,
        online_batch_size=2,
        offline_steps=1,
        frame_stack=1,
    )
    if name == 'gc_pixel_lapo_decoder':
        config.update(num_codebooks=2, num_codes=8, code_dim=4)
    return config


def _online_batch(data):
    batch = data.sample(2)
    return {
        **batch,
        'actions': np.array([[0.2, -0.3], [-0.1, 0.4]], np.float32),
    }


def test_pixel_registry_declares_information_and_online_update_boundaries():
    assert PIXEL_ALGORITHMS == (
        'pixel_pathbridger_online_idm',
        'gc_pixel_lapo_decoder',
        'gc_pixel_drqv2',
        'vip_style_frozen_gc_drqv2',
        'vip_style_finetuned_gc_drqv2',
        'gc_pixel_apv_style_drq',
    )
    assert pixel_algorithm_metadata('gc_pixel_drqv2').offline_fields_seen == ()
    assert pixel_algorithm_metadata('gc_pixel_lapo_decoder').online_modules_updated == (
        'decoder',
    )
    assert pixel_algorithm_metadata(
        'vip_style_frozen_gc_drqv2'
    ).online_modules_updated == ('actor', 'critic')
    assert 'action_dynamics' in pixel_algorithm_metadata(
        'gc_pixel_apv_style_drq'
    ).online_modules_updated
    for name in PIXEL_ALGORITHMS:
        assert pixel_algorithm_metadata(name).port_kind in {
            'online_only',
            'goal_conditioned_adaptation',
            'proposed',
        }


def test_pixel_registry_keeps_legacy_names_as_honest_aliases():
    assert pixel_algorithm_metadata('gc_pixel_lapo').algorithm == (
        'gc_pixel_lapo_decoder'
    )
    assert pixel_algorithm_metadata('gc_pixel_apv').algorithm == (
        'gc_pixel_apv_style_drq'
    )


def test_pixel_pathbridger_freezes_offline_path_and_updates_only_idm_online():
    name = 'pixel_pathbridger_online_idm'
    data = _stacked_data()
    config = _config(name)
    config.update(idm_hidden_dims=(16,), path_horizon=5, frame_stack=3)
    agent, _ = create_pixel_algorithm(
        name,
        seed=0,
        example_images=data.example_images,
        action_dim=2,
        config=config,
    )
    offline_names = ('encoder', 'bridge', 'world_decoder')
    offline_before = {
        module: parameter_digest(agent.network.params[f'modules_{module}'])
        for module in offline_names
    }
    idm_before = parameter_digest(agent.network.params['modules_idm'])
    agent, offline_info = agent.offline_update(data.sample(2, path_horizon=5))
    assert np.isfinite(float(offline_info['loss/total']))
    assert any(
        parameter_digest(agent.network.params[f'modules_{module}'])
        != offline_before[module]
        for module in offline_names
    )
    assert parameter_digest(agent.network.params['modules_idm']) == idm_before

    frozen_before_online = {
        module: parameter_digest(agent.network.params[f'modules_{module}'])
        for module in (*offline_names, 'target_encoder')
    }
    batch = data.sample(2, path_horizon=5)
    batch['actions'] = np.array([[0.2, -0.3], [0.1, -0.4]], np.float32)
    agent, online_info = agent.online_update(batch)
    assert np.isfinite(float(online_info['loss/total']))
    assert parameter_digest(agent.network.params['modules_idm']) != idm_before
    for module, digest in frozen_before_online.items():
        assert parameter_digest(agent.network.params[f'modules_{module}']) == digest
    actions = agent.sample_actions(
        batch['observations'],
        batch['goals'],
        seed=jax.random.PRNGKey(3),
        temperature=0.0,
    )
    assert actions.shape == (2, 2)


def test_vip_frozen_and_finetuned_have_distinct_online_encoder_behavior():
    data = _data()
    images = data.observations[:2]
    for name, should_change in (
        ('vip_style_frozen_gc_drqv2', False),
        ('vip_style_finetuned_gc_drqv2', True),
    ):
        agent, _ = create_pixel_algorithm(
            name,
            seed=0,
            example_images=images,
            action_dim=2,
            config=_config(name),
        )
        agent, offline_info = agent.offline_update(data.sample(2))
        assert np.isfinite(float(offline_info['loss/total']))
        before = parameter_digest(agent.network.params['modules_encoder'])
        agent, online_info = agent.online_update(_online_batch(data))
        assert np.isfinite(float(online_info['loss/total']))
        changed = parameter_digest(agent.network.params['modules_encoder']) != before
        assert changed is should_change


def test_apv_pretrains_video_modules_and_learns_action_dynamics_online():
    name = 'gc_pixel_apv_style_drq'
    data = _data()
    agent, _ = create_pixel_algorithm(
        name,
        seed=0,
        example_images=data.observations[:2],
        action_dim=2,
        config=_config(name),
    )
    video_before = parameter_digest(agent.network.params['modules_video_predictor'])
    dynamics_before = parameter_digest(agent.network.params['modules_action_dynamics'])
    agent, offline_info = agent.offline_update(data.sample(2))
    assert np.isfinite(float(offline_info['loss/total']))
    assert parameter_digest(agent.network.params['modules_video_predictor']) != video_before
    assert parameter_digest(agent.network.params['modules_action_dynamics']) == dynamics_before
    agent, online_info = agent.online_update(_online_batch(data))
    assert np.isfinite(float(online_info['loss/total']))
    assert parameter_digest(agent.network.params['modules_action_dynamics']) != dynamics_before
    actions = agent.sample_actions(
        data.observations[:2],
        data.observations[2:4],
        seed=jax.random.PRNGKey(2),
        temperature=0.0,
    )
    assert actions.shape == (2, 2)


def test_generic_pixel_checkpoint_round_trip(tmp_path):
    name = 'gc_pixel_drqv2'
    images = _pixels()[:2]
    config = _config(name)
    agent, _ = create_pixel_algorithm(
        name,
        seed=0,
        example_images=images,
        action_dim=2,
        config=config,
    )
    checkpoint = tmp_path / 'checkpoint.pkl'
    save_af_checkpoint(
        checkpoint,
        algorithm=name,
        agent=agent,
        step=7,
        config=config,
        metadata={'observation_modality': 'rgb_uint8'},
    )
    payload = load_af_checkpoint(checkpoint)
    template, _ = create_pixel_algorithm(
        name,
        seed=1,
        example_images=images,
        action_dim=2,
        config=config,
    )
    restored = restore_af_agent(template, payload)
    assert payload['step'] == 7
    assert parameter_digest(restored.network.params) == parameter_digest(
        agent.network.params
    )


def test_pixel_manifest_matches_all_algorithms_and_locked_dimensions(tmp_path):
    expected = {'p0_smoke': 18, 'pilot': 72, 'screening': 240}
    for name, count in expected.items():
        rows = build_rows(root=tmp_path, python='python', suite_name=name)
        assert len(rows) == count
        assert {row['algorithm'] for row in rows} == set(PIXEL_ALGORITHMS)
        assert all('train_pixel.py' in row['command'] for row in rows)
        assert all('train_af.py' not in row['command'] for row in rows)
    assert len(SUITES['screening']['seeds']) == 5
