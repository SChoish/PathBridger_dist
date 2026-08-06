from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import agents.pathbridger as pathbridger
from agents.af_guide import AFGuideAgent, get_config as afguide_config
from agents.gc_actor_critic import GoalConditionedActorCritic, get_config as gc_config
from agents.mscp import MSCPAgent, get_config as mscp_config
from agents.online_idm import OnlineIDMAgent, PBFOnlineIDMPolicy
from agents.oso_decqn import OSODecQNAgent, discretize_deltas, get_config as oso_config
from agents.passive_hiql import PassiveHIQLAgent, get_config as hiql_config
from utils.af_checkpoints import (
    load_af_checkpoint,
    restore_af_agent,
    save_af_checkpoint,
)


OBS = jnp.asarray([[0.0, 0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.4]], jnp.float32)
ACT = jnp.asarray([[0.0, 0.1], [0.1, -0.1]], jnp.float32)


def _online_batch():
    return {
        'observations': OBS,
        'next_observations': OBS + 0.02,
        'actions': ACT,
        'goals': OBS + 0.2,
        'rewards': jnp.asarray([-1.0, 0.0], jnp.float32),
        'masks': jnp.asarray([1.0, 0.0], jnp.float32),
        'desired_next': OBS + 0.03,
    }


def _offline_batch():
    batch = _online_batch()
    batch.pop('actions')
    batch['fast_targets'] = OBS + 0.02
    batch['slow_targets'] = OBS + 0.1
    return batch


def test_gc_sac_and_td3_update_and_act():
    for algorithm in ('sac', 'td3'):
        config = gc_config(algorithm).to_dict()
        config['hidden_dims'] = (8,)
        agent = GoalConditionedActorCritic.create(0, OBS, ACT, config)
        agent, info = agent.update(_online_batch())
        assert np.isfinite(np.asarray(info['loss/total']))
        assert agent.sample_actions(OBS, OBS + 0.2, seed=jax.random.PRNGKey(1)).shape == ACT.shape


def test_passive_hiql_low_policy_changes_only_online():
    config = hiql_config().to_dict()
    config['hidden_dims'] = (8,)
    agent = PassiveHIQLAgent.create(0, OBS, ACT.shape[-1], config)
    low_before = agent.network.params['modules_low_policy']
    offline_agent, _ = agent.offline_update(_offline_batch())
    for left, right in zip(jax.tree_util.tree_leaves(low_before), jax.tree_util.tree_leaves(offline_agent.network.params['modules_low_policy'])):
        np.testing.assert_array_equal(left, right)
    online_agent, _ = offline_agent.online_update(_online_batch())
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(
            jax.tree_util.tree_leaves(offline_agent.network.params['modules_low_policy']),
            jax.tree_util.tree_leaves(online_agent.network.params['modules_low_policy']),
        )
    )


def test_mscp_afguide_and_oso_smoke():
    mcfg = mscp_config().to_dict()
    mcfg['hidden_dims'] = (8,)
    mscp = MSCPAgent.create(0, OBS, ACT.shape[-1], mcfg)
    mscp, _ = mscp.offline_update(_offline_batch())
    mscp, _ = mscp.online_update(_online_batch())
    assert mscp.sample_actions(OBS, OBS + 0.2, seed=jax.random.PRNGKey(2)).shape == ACT.shape

    acfg = afguide_config().to_dict()
    acfg.update(context_length=3, embed_dim=8, num_blocks=1, hidden_dims=(8,))
    guide = AFGuideAgent.create(0, OBS, ACT.shape[-1], jnp.ones(OBS.shape[-1]), acfg)
    sequence_batch = {
        'histories': jnp.broadcast_to(OBS[:, None, :], (2, 3, 4)),
        'history_masks': jnp.ones((2, 3)),
        'goals': OBS + 0.2,
        'remaining': jnp.ones((2,)),
        'target_deltas': jnp.ones_like(OBS) * 0.02,
    }
    guide, _ = guide.offline_update(sequence_batch)
    guide_critic_before = guide.network.params['modules_guide_critic']
    invalid_guide_batch = {
        **_online_batch(),
        'desired_next_valid': jnp.zeros((len(OBS),), jnp.float32),
    }
    guide, invalid_info = guide.online_update(invalid_guide_batch)
    assert float(invalid_info['guide/valid_target_fraction']) == 0.0
    for left, right in zip(
        jax.tree_util.tree_leaves(guide_critic_before),
        jax.tree_util.tree_leaves(guide.network.params['modules_guide_critic']),
    ):
        np.testing.assert_array_equal(left, right)
    guide_batch = {
        **_online_batch(),
        'desired_next_valid': jnp.ones((len(OBS),), jnp.float32),
    }
    guide, valid_info = guide.online_update(guide_batch)
    assert float(valid_info['guide/valid_target_fraction']) == 1.0
    assert guide.sample_actions(OBS, OBS + 0.2, seed=jax.random.PRNGKey(3)).shape == ACT.shape

    ocfg = oso_config().to_dict()
    ocfg.update(hidden_dims=(8,), online_hidden_dims=(8,), idm_hidden_dims=(8,))
    oso = OSODecQNAgent.create(0, OBS, ACT.shape[-1], jnp.ones(4) * 0.1, ocfg)
    oso_batch = _offline_batch()
    oso_batch['delta_bins'] = jnp.asarray(
        discretize_deltas(oso_batch['next_observations'] - oso_batch['observations'], np.ones(4) * 0.1)
    )
    oso, _ = oso.offline_update(oso_batch)
    oso, _ = oso.online_update(_online_batch())
    assert oso.sample_actions(OBS, OBS + 0.2, seed=jax.random.PRNGKey(4)).shape == ACT.shape


def test_action_free_pbf_has_no_idm_and_composes_online(monkeypatch):
    for name in (
        'ScalarTransitiveValue',
        'FlowEndpointProposer',
        'BridgeResidual',
    ):
        module = getattr(pathbridger, name)
        monkeypatch.setattr(pathbridger, name, partial(module, hidden_dims=(8,)))
    config = pathbridger.get_config().to_dict()
    config.update(
        offline_action_free=True,
        endpoint_distribution='flow',
        horizon=5,
        eval_num_candidates=1,
        eval_temperature=0.0,
    )
    planner = pathbridger.PathBridgerAgent.create(0, OBS, None, config)
    assert 'modules_idm' not in planner.network.params
    idm = OnlineIDMAgent.create(
        1,
        OBS,
        ACT.shape[-1],
        {'hidden_dims': (8,), 'learning_rate': 3e-4, 'loss': 'mse'},
    )
    policy = PBFOnlineIDMPolicy(planner, idm, 1, 0.0, 1)
    actions = policy.sample_actions(OBS, OBS + 0.2, seed=jax.random.PRNGKey(5))
    assert actions.shape == ACT.shape
    chunk_policy = PBFOnlineIDMPolicy(planner, idm, 1, 0.0, 5)
    action_chunk = chunk_policy.sample_actions(
        OBS, OBS + 0.2, seed=jax.random.PRNGKey(6)
    )
    assert action_chunk.shape == (len(OBS), 5, ACT.shape[-1])


def test_component_checkpoint_round_trip(tmp_path):
    config = gc_config('sac').to_dict()
    config['hidden_dims'] = (8,)
    template = GoalConditionedActorCritic.create(0, OBS, ACT, config)
    trained, _ = template.update(_online_batch())
    path = tmp_path / 'step_1.pkl'
    save_af_checkpoint(
        path,
        algorithm='gc_sac',
        agent=trained,
        step=1,
        config=config,
        metadata={'offline_fields_seen': []},
    )
    payload = load_af_checkpoint(path)
    restored = restore_af_agent(template, payload)
    expected = trained.sample_actions(
        OBS, OBS + 0.2, seed=jax.random.PRNGKey(6), temperature=0.0
    )
    actual = restored.sample_actions(
        OBS, OBS + 0.2, seed=jax.random.PRNGKey(6), temperature=0.0
    )
    np.testing.assert_allclose(actual, expected)
    assert payload['algorithm'] == 'gc_sac'
    assert payload['step'] == 1
