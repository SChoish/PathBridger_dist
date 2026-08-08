from __future__ import annotations

import jax
import numpy as np
import pytest

from agents.online_idm import parameter_digest
from agents.pixel_registry import (
    PIXEL_ALGORITHMS,
    create_pixel_algorithm,
    get_pixel_config,
    pixel_algorithm_metadata,
    pixel_method_scope,
)
from agents.pixel_trl_critic_locks import (
    PIXEL_PBF_GAP_SEARCH,
    PIXEL_PBF_NT_SEARCH,
    PIXEL_TRL_CRITIC_LOCKS,
    apply_pixel_pbf_locks,
    trl_critic_lock_for_env,
)
from agents.pixel_pathbridger import (
    CompactImpalaPathEncoder,
    ImpalaSmallEncoder,
    _crop_with_offsets,
)
from scripts.make_pixel_manifest import SUITES, build_rows
from utils.af_checkpoints import load_af_checkpoint, restore_af_agent, save_af_checkpoint
from utils.pixel_data import ActionFreePixelTrajectoryData, PixelTrajectoryData


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
            'terminals': np.array([0, 0, 0, 0, 0, 1], np.float32),
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
    if name == 'pixel_pbf':
        config.update(
            value_hidden_dims=(16,),
            endpoint_horizon=5,
            endpoint_flow_steps=2,
            path_rep_dim=4,
            p_aug=0.0,
        )
    if name in ('pixel_hiql', 'pixel_ota'):
        config.update(
            actor_hidden_dims=(16,),
            value_hidden_dims=(16,),
            rep_dim=4,
            p_aug=0.0,
        )
    return config


def _online_batch(data):
    batch = data.sample(2)
    return {
        **batch,
        'actions': np.array([[0.2, -0.3], [-0.1, 0.4]], np.float32),
    }


def test_pixel_registry_declares_information_and_online_update_boundaries():
    assert PIXEL_ALGORITHMS == (
        'pixel_pbf',
        'pixel_hiql',
        'pixel_ota',
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
            'full_action',
        }
        scope = pixel_method_scope(name)
        assert tuple(pixel_algorithm_metadata(name).online_modules_updated) == (
            scope['online_trainable_modules']
        )

    assert pixel_method_scope('pixel_pbf')['offline_trainable_modules'] == (
        'encoder', 'endpoint', 'bridge', 'value', 'idm'
    )
    assert pixel_method_scope('pixel_pbf')['online_trainable_modules'] == ()
    assert pixel_method_scope('pixel_hiql')['online_trainable_modules'] == ()
    assert pixel_method_scope('pixel_ota')['online_trainable_modules'] == ()
    assert pixel_method_scope('gc_pixel_lapo_decoder')[
        'online_trainable_modules'
    ] == ('decoder',)
    assert pixel_method_scope('gc_pixel_drqv2')[
        'offline_trainable_modules'
    ] == ()
    assert 'encoder' in pixel_method_scope('vip_style_frozen_gc_drqv2')[
        'online_frozen_modules'
    ]
    assert 'encoder' in pixel_method_scope('vip_style_finetuned_gc_drqv2')[
        'online_trainable_modules'
    ]
    assert pixel_method_scope('gc_pixel_apv_style_drq')[
        'online_frozen_modules'
    ] == ('video_predictor', 'world_decoder')


def test_pixel_registry_keeps_legacy_names_as_honest_aliases():
    assert pixel_algorithm_metadata('gc_pixel_lapo').algorithm == (
        'gc_pixel_lapo_decoder'
    )
    assert pixel_algorithm_metadata('gc_pixel_apv').algorithm == (
        'gc_pixel_apv_style_drq'
    )


def test_pixel_method_names_reject_conflicting_training_regimes():
    images = _pixels()[:2]

    frozen_vip = _config('vip_style_frozen_gc_drqv2')
    frozen_vip['freeze_encoder_online'] = False
    with pytest.raises(ValueError, match='requires freeze_encoder_online=True'):
        create_pixel_algorithm(
            'vip_style_frozen_gc_drqv2',
            seed=0,
            example_images=images,
            action_dim=2,
            config=frozen_vip,
        )

    apv = _config('gc_pixel_apv_style_drq')
    apv['pretraining'] = 'none'
    with pytest.raises(ValueError, match="requires pretraining='apv'"):
        create_pixel_algorithm(
            'gc_pixel_apv_style_drq',
            seed=0,
            example_images=images,
            action_dim=2,
            config=apv,
        )


def test_full_offline_pixel_pbf_trains_idm_from_dataset_actions():
    name = 'pixel_pbf'
    data = PixelTrajectoryData(
        {
            'observations': _pixels(),
            'terminals': np.array([0, 0, 0, 0, 0, 1], np.float32),
            'actions': np.linspace(-0.5, 0.5, 12, dtype=np.float32).reshape(6, 2),
        },
        seed=2,
        frame_stack=3,
    )
    config = _config(name)
    config.update(idm_hidden_dims=(16,), path_horizon=5, frame_stack=3)
    agent, _ = create_pixel_algorithm(
        name,
        seed=0,
        example_images=data.example_images,
        action_dim=2,
        config=config,
    )
    assert isinstance(
        agent.network.model_def.modules['encoder'], CompactImpalaPathEncoder
    )
    before = parameter_digest(agent.network.params['modules_idm'])
    batch = data.sample(
        2,
        path_horizon=5,
        endpoint_horizon=5,
        discount=0.99,
        value_geom_sample=True,
        pbf_indices_only=True,
    )
    agent, info = agent.offline_update_indexed(
        batch,
        np.asarray(data.observations),
        np.asarray(data.episodes.initial_for_state, dtype=np.int32),
    )
    assert np.isfinite(float(info['loss/total']))
    assert np.isfinite(float(info['idm/loss']))
    assert parameter_digest(agent.network.params['modules_idm']) != before


def test_pixel_pbf_compact_representation_and_geometry_gradient_routing():
    data = PixelTrajectoryData(
        {
            'observations': _pixels(),
            'terminals': np.array([0, 0, 0, 0, 0, 1], np.float32),
            'actions': np.linspace(-0.5, 0.5, 12, dtype=np.float32).reshape(6, 2),
        },
        seed=3,
        frame_stack=1,
    )
    config = _config('pixel_pbf')
    assert config['normalize_path_rep'] is True
    assert config['stop_planner_rep_grad'] is True
    assert config['geometry_target_source'] == 'online'
    agent, _ = create_pixel_algorithm(
        'pixel_pbf',
        seed=4,
        example_images=data.example_images,
        action_dim=2,
        config=config,
    )
    encoded = agent.network.select('encoder')(data.example_images)
    assert encoded.shape == (2, 4)
    assert np.allclose(
        np.linalg.norm(np.asarray(encoded), axis=-1), np.sqrt(4), atol=1e-5
    )
    path = agent.latent_path(
        data.example_images[:1],
        data.example_images[1:2],
        seed=jax.random.PRNGKey(8),
        num_candidates=2,
        endpoint_temperature=0.25,
    )
    assert np.allclose(
        np.linalg.norm(np.asarray(path), axis=-1), np.sqrt(4), atol=1e-5
    )

    batch = data.sample(
        2,
        path_horizon=5,
        endpoint_horizon=5,
        discount=0.99,
        value_geom_sample=True,
    )
    # Digest comparisons are GPU-nondeterministic; check the architectural
    # invariant directly: endpoint/bridge losses must not train the encoder.
    assert float(agent.planner_encoder_grad_norm(batch)) < 1e-5
    changed_geometry = dict(batch)
    changed_geometry['endpoint_targets'] = 255 - batch['endpoint_targets']
    changed_geometry['bridge_targets'] = 255 - batch['bridge_targets']
    assert float(agent.planner_encoder_grad_norm(changed_geometry)) < 1e-5

    updated, info = agent.offline_update(batch)
    changed, changed_info = agent.offline_update(changed_geometry)
    assert parameter_digest(updated.network.params['modules_endpoint']) != (
        parameter_digest(changed.network.params['modules_endpoint'])
    )
    for key in (
        'repr/cross_sample_std',
        'repr/effective_rank',
        'repr/one_step_distance',
        'repr/random_pair_distance',
        'idm/mean_action_baseline_mse',
        'action/abs_mean',
        'action/std',
    ):
        assert np.isfinite(float(info[key]))
        assert np.isfinite(float(changed_info[key]))


def test_pixel_pbf_fast_update_skips_log_only_diagnostics():
    data = PixelTrajectoryData(
        {
            'observations': _pixels(),
            'terminals': np.array([0, 0, 0, 0, 0, 1], np.float32),
            'actions': np.linspace(-0.5, 0.5, 12, dtype=np.float32).reshape(6, 2),
        },
        seed=5,
        frame_stack=1,
    )
    config = _config('pixel_pbf')
    agent, _ = create_pixel_algorithm(
        'pixel_pbf',
        seed=6,
        example_images=data.example_images,
        action_dim=2,
        config=config,
    )
    batch = data.sample(
        2,
        path_horizon=5,
        endpoint_horizon=5,
        discount=0.99,
        value_geom_sample=True,
    )
    _, info = agent.offline_update(batch, full_metrics=False)
    assert np.isfinite(float(info['loss/total']))
    assert 'repr/effective_rank' not in info
    assert 'idm/mean_action_baseline_mse' not in info
    assert 'endpoint/value_gap_mean' not in info


def test_pixel_pbf_random_crop_is_consistent_across_path_images():
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    images = np.stack([image, image])
    paths = np.repeat(images[:, None], 5, axis=1)
    offsets = np.array([[0, 6], [5, 1]], dtype=np.int32)
    cropped_images = np.asarray(_crop_with_offsets(images, offsets))
    cropped_paths = np.asarray(_crop_with_offsets(paths, offsets))
    assert np.array_equal(cropped_paths[:, 0], cropped_images)
    assert np.array_equal(cropped_paths[:, 4], cropped_images)


def test_legacy_raw_pixel_pbf_config_keeps_checkpoint_parameter_tree(tmp_path):
    config = _config('pixel_pbf')
    for key in (
        'path_rep_dim',
        'normalize_path_rep',
        'stop_planner_rep_grad',
        'geometry_target_source',
        'log_rep_diagnostics',
    ):
        config.pop(key)
    config['legacy_raw_representation'] = True
    agent, resolved = create_pixel_algorithm(
        'pixel_pbf',
        seed=5,
        example_images=_pixels()[:2],
        action_dim=2,
        config=config,
    )
    encoder_params = agent.network.params['modules_encoder']
    assert isinstance(agent.network.model_def.modules['encoder'], ImpalaSmallEncoder)
    assert 'visual_encoder' not in encoder_params
    assert resolved['path_rep_dim'] == resolved['feature_dim']
    assert resolved['normalize_path_rep'] is False
    checkpoint = tmp_path / 'legacy.pkl'
    legacy_payload_config = dict(config)
    legacy_payload_config.pop('legacy_raw_representation')
    save_af_checkpoint(
        checkpoint,
        algorithm='pixel_pbf',
        agent=agent,
        step=17,
        config=legacy_payload_config,
        metadata={'observation_modality': 'rgb_uint8'},
    )
    payload = load_af_checkpoint(checkpoint)
    restore_config = dict(payload['config'])
    assert 'path_rep_dim' not in restore_config
    restore_config['legacy_raw_representation'] = True
    template, _ = create_pixel_algorithm(
        'pixel_pbf',
        seed=6,
        example_images=_pixels()[:2],
        action_dim=2,
        config=restore_config,
    )
    restored = restore_af_agent(template, payload)
    assert parameter_digest(restored.network.params) == parameter_digest(
        agent.network.params
    )


@pytest.mark.parametrize('name', ('pixel_hiql', 'pixel_ota'))
def test_pixel_hierarchical_baselines_train_and_act(name):
    frames = _pixels()
    data = PixelTrajectoryData(
        {
            'observations': frames,
            'terminals': np.array([0, 0, 1, 0, 0, 1], np.float32),
            'actions': np.linspace(-0.5, 0.5, 12, dtype=np.float32).reshape(6, 2),
        },
        seed=7,
        frame_stack=1,
    )
    config = _config(name)
    config.update(subgoal_steps=2, abstraction_factor=2)
    agent, _ = create_pixel_algorithm(
        name,
        seed=0,
        example_images=data.example_images,
        action_dim=2,
        config=config,
    )
    assert 'goal_rep' not in agent.network.params[
        'modules_low_value' if name == 'pixel_ota' else 'modules_value'
    ]
    batch = data.sample_hierarchical(2, **config)
    before = parameter_digest(agent.network.params['modules_goal_rep'])
    agent, info = agent.offline_update(batch)
    assert np.isfinite(float(info['loss/total']))
    assert np.isfinite(float(info['low_actor/loss']))
    assert parameter_digest(agent.network.params['modules_goal_rep']) != before
    actions = agent.sample_actions(
        data.example_images[:1],
        data.example_images[1:2],
        seed=jax.random.PRNGKey(1),
        temperature=0.0,
    )
    assert actions.shape == (1, 2)
    assert np.all(np.isfinite(np.asarray(actions)))
    if name == 'pixel_ota':
        assert 'high_value/loss' in info
        assert np.all(
            batch['high_value_option_indices']
            <= data.episodes.terminal_for_state[batch['indices']]
        )


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


def test_pixel_pbf_anchor_restore_keeps_branch_nt_config(tmp_path):
    name = 'pixel_pbf'
    config = _config(name)
    images = _pixels()[:2]
    anchor, _ = create_pixel_algorithm(
        name, seed=0, example_images=images, action_dim=2, config=config
    )
    checkpoint = tmp_path / 'step_0.pkl'
    save_af_checkpoint(
        checkpoint,
        algorithm=name,
        agent=anchor,
        step=0,
        config=config,
        metadata={'phase': 'offline'},
        phase='offline',
    )
    branch_config = {**config, 'eval_num_candidates': 16, 'eval_temperature': 0.5}
    branch, _ = create_pixel_algorithm(
        name, seed=1, example_images=images, action_dim=2, config=branch_config
    )
    restored = restore_af_agent(branch, load_af_checkpoint(checkpoint))
    assert restored.config['eval_num_candidates'] == 16
    assert restored.config['eval_temperature'] == 0.5
    assert parameter_digest(restored.network.params) == parameter_digest(
        anchor.network.params
    )


def test_pixel_manifest_matches_all_algorithms_and_locked_dimensions(tmp_path):
    expected = {'pilot': 60, 'screening': 200}
    for name, count in expected.items():
        rows = build_rows(root=tmp_path, python='python', suite_name=name)
        assert len(rows) == count
        assert {row['algorithm'] for row in rows} == set(PIXEL_ALGORITHMS) - {
            'pixel_pbf',
            'pixel_hiql',
            'pixel_ota',
        }
        assert all('train_pixel.py' in row['command'] for row in rows)
        assert all('train_af.py' not in row['command'] for row in rows)
        assert all('--resume_keep=1' in row['command'] for row in rows)
        assert all('--save_replay' in row['command'] for row in rows)
        assert all('--dataset_dir=' in row['command'] for row in rows)
    lapo_rows = build_rows(
        root=tmp_path,
        python='python',
        suite_name='pilot',
        algorithm='gc_pixel_lapo_decoder',
    )
    assert len(lapo_rows) == 12
    assert {row['algorithm'] for row in lapo_rows} == {'gc_pixel_lapo_decoder'}
    assert all(
        '--algorithm=gc_pixel_lapo_decoder' in row['command']
        for row in lapo_rows
    )
    assert len(SUITES['screening']['seeds']) == 5


def test_trl_critic_locks_cover_visual_suite_and_match_paper_lambda():
    assert set(PIXEL_TRL_CRITIC_LOCKS) >= {
        'visual-antmaze-large-navigate-v0',
        'visual-cube-double-play-v0',
        'visual-puzzle-4x4-play-v0',
        'visual-scene-play-v0',
    }
    assert trl_critic_lock_for_env('visual-antmaze-large-navigate-v0')[
        'value_distance_weight_power'
    ] == 0.0
    assert trl_critic_lock_for_env('visual-puzzle-4x4-play-v0')[
        'value_distance_weight_power'
    ] == 2.0
    for lock in PIXEL_TRL_CRITIC_LOCKS.values():
        assert lock['expectile'] == 0.7
        assert lock['value_p_trajgoal'] == 1.0
        assert lock['value_p_curgoal'] == 0.0
        assert lock['value_p_randomgoal'] == 0.0
        assert lock['value_hidden_dims'] == (512, 512, 512)
        assert lock['value_layer_norm'] is True
        assert lock['value_learning_rate'] == 3e-4
        assert lock['value_tau'] == 0.005


def test_pixel_pbf_search_grid_and_locks_are_explicit():
    assert PIXEL_PBF_GAP_SEARCH == (5.0, 10.0)
    assert PIXEL_PBF_NT_SEARCH == (
        (1, 0.0),
        (2, 0.25),
        (16, 0.5),
        (32, 1.0),
    )
    locked = apply_pixel_pbf_locks(
        'visual-cube-double-play-v0',
        {'endpoint_value_scale': 5.0, 'eval_num_candidates': 16},
    )
    assert locked['endpoint_horizon'] == 40
    assert locked['encoder'] == 'impala_small'
    assert locked['feature_dim'] == 512
    assert locked['path_rep_dim'] == 32
    assert locked['normalize_path_rep'] is True
    assert locked['stop_planner_rep_grad'] is True
    assert locked['geometry_target_source'] == 'online'
    assert locked['offline_batch_size'] == 256
    assert locked['frame_stack'] == 1
    assert locked['p_aug'] == 0.5
    assert locked['path_horizon'] == 5
    assert locked['endpoint_value_scale'] == 5.0
    with pytest.raises(ValueError, match='Only gap and'):
        apply_pixel_pbf_locks(
            'visual-cube-double-play-v0', {'discount': 0.95}
        )


def test_impala_small_encoder_matches_official_stack_shape():
    module = ImpalaSmallEncoder(feature_dim=32)
    variables = module.init(
        jax.random.PRNGKey(9), np.zeros((2, 32, 32, 9), dtype=np.uint8)
    )
    features = module.apply(
        variables, np.zeros((2, 32, 32, 9), dtype=np.uint8)
    )
    assert features.shape == (2, 32)
    stack_names = [name for name in variables['params'] if name.startswith('ImpalaResidualStack')]
    assert len(stack_names) == 3


def test_pixel_pbf_sampler_uses_k_endpoint_and_first_five_bridge_targets():
    rng = np.random.default_rng(8)
    frames = rng.integers(0, 256, (12, 32, 32, 3), dtype=np.uint8)
    data = ActionFreePixelTrajectoryData(
        {
            'observations': frames,
            'terminals': np.array([0] * 11 + [1], np.float32),
        },
        seed=4,
        frame_stack=1,
    )
    batch = data.sample(
        32,
        path_horizon=5,
        endpoint_horizon=6,
        discount=0.99,
        value_geom_sample=False,
    )
    assert np.all(batch['indices'] + 6 <= 11)
    assert np.array_equal(
        batch['endpoint_target_indices'],
        np.minimum(batch['indices'] + 6, batch['endpoint_goal_indices']),
    )
    expected_bridge = batch['indices'][:, None] + np.arange(1, 6)[None]
    expected_bridge = np.minimum(
        expected_bridge, batch['endpoint_target_indices'][:, None]
    )
    assert np.array_equal(batch['path_indices'], expected_bridge)
    assert np.all((batch['base_offsets'] >= 1) & (batch['base_offsets'] <= 5))
    assert np.array_equal(
        batch['transitive_valids'], (batch['value_offsets'] > 5).astype(np.float32)
    )


def test_pixel_pbf_non_geometric_future_sampler_matches_state_formula():
    frames = np.zeros((8, 32, 32, 3), dtype=np.uint8)
    data = ActionFreePixelTrajectoryData(
        {
            'observations': frames,
            'terminals': np.array([0, 0, 0, 0, 0, 0, 0, 1], np.float32),
        },
        seed=123,
    )
    indices = np.array([0, 2, 5], dtype=np.int64)
    expected_rng = np.random.default_rng(123)
    distances = expected_rng.random(len(indices))
    expected_goals = np.round(
        (indices + 1) * distances + 7 * (1.0 - distances)
    ).astype(np.int64)
    offsets = data._sample_future_offsets(
        indices, discount=0.99, geom_sample=False
    )
    assert np.array_equal(indices + offsets, expected_goals)
