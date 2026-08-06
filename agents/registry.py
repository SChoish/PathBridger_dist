"""Construction and provenance registry for the benchmark suite."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from agents.af_guide import AFGuideAgent, get_config as afguide_config
from agents.gc_actor_critic import GoalConditionedActorCritic, get_config as gc_config
from agents.mscp import MSCPAgent, get_config as mscp_config
from agents.oso_decqn import OSODecQNAgent, get_config as oso_config
from agents.passive_hiql import PassiveHIQLAgent, get_config as hiql_config
from utils.provenance import AlgorithmMetadata


ALGORITHMS = (
    'pbf_online_idm',
    'gc_mscp',
    'passive_hiql',
    'gc_af_guide',
    'gc_oso_decqn',
    'gc_sac',
    'gc_td3',
    'gc_rlpd',
)


DEFAULT_OFFLINE_STEPS = {
    'pbf_online_idm': 1_000_000,
    'gc_mscp': 1_000_000,
    'passive_hiql': 1_000_000,
    'gc_af_guide': 50_000,
    'gc_oso_decqn': 3_000_000,
    'gc_sac': 0,
    'gc_td3': 0,
    'gc_rlpd': 1_000_000,
}


def get_algorithm_config(name: str) -> dict[str, Any]:
    name = str(name).lower()
    if name == 'gc_mscp':
        return mscp_config().to_dict()
    if name == 'passive_hiql':
        return hiql_config().to_dict()
    if name == 'gc_af_guide':
        return afguide_config().to_dict()
    if name == 'gc_oso_decqn':
        return oso_config().to_dict()
    if name in ('gc_sac', 'gc_rlpd'):
        return gc_config('sac').to_dict()
    if name == 'gc_td3':
        return gc_config('td3').to_dict()
    if name == 'pbf_online_idm':
        return {}
    raise ValueError(f'Unknown algorithm {name!r}; choose from {ALGORITHMS}.')


def create_algorithm(
    name: str,
    *,
    seed: int,
    ex_observations: Any,
    action_dim: int,
    state_scale: Any,
    delta_scale: Any,
    config: dict[str, Any] | None = None,
):
    name = str(name).lower()
    resolved = get_algorithm_config(name)
    if config:
        resolved.update(config)
    observations = jnp.asarray(ex_observations, dtype=jnp.float32)
    if name == 'gc_mscp':
        return MSCPAgent.create(seed, observations, action_dim, resolved), resolved
    if name == 'passive_hiql':
        return PassiveHIQLAgent.create(seed, observations, action_dim, resolved), resolved
    if name == 'gc_af_guide':
        return AFGuideAgent.create(
            seed, observations, action_dim, state_scale, resolved
        ), resolved
    if name == 'gc_oso_decqn':
        return OSODecQNAgent.create(
            seed, observations, action_dim, delta_scale, resolved
        ), resolved
    if name in ('gc_sac', 'gc_td3', 'gc_rlpd'):
        example_actions = jnp.zeros(
            (len(observations), int(action_dim)), dtype=jnp.float32
        )
        return GoalConditionedActorCritic.create(
            seed, observations, example_actions, resolved
        ), resolved
    raise ValueError('PBF is constructed from its environment-specific config and checkpoint.')


def algorithm_metadata(name: str) -> AlgorithmMetadata:
    name = str(name).lower()
    records = {
        'pbf_online_idm': dict(
            port_kind='proposed',
            paper_url='',
            official_repo_url=None,
            official_repo_commit=None,
            online_modules_updated=('idm',),
        ),
        'gc_mscp': dict(
            port_kind='paper_reimplementation',
            paper_url='https://proceedings.mlr.press/v235/wu24j.html',
            official_repo_url='https://github.com/ChengjieWU/MSCP',
            official_repo_commit='81bc98f889ac058691dc67f8f585982140b559c1',
            online_modules_updated=('low_policy', 'value'),
        ),
        'passive_hiql': dict(
            port_kind='goal_conditioned_adaptation',
            paper_url='https://papers.nips.cc/paper_files/paper/2023/hash/6d7c4a0727e089ed6cdd3151cbe8d8ba-Abstract-Conference.html',
            official_repo_url='https://github.com/seohongpark/HIQL',
            official_repo_commit=None,
            online_modules_updated=('low_policy', 'value'),
        ),
        'gc_af_guide': dict(
            port_kind='goal_conditioned_adaptation',
            paper_url='https://arxiv.org/abs/2301.12876',
            official_repo_url='https://github.com/Vision-CAIR/AF-Guide',
            official_repo_commit='8579489dea345b2aaed8eafafaed3b16daef7683',
            online_modules_updated=('actor', 'critic', 'guide_critic'),
        ),
        'gc_oso_decqn': dict(
            port_kind='paper_reimplementation',
            paper_url='https://arxiv.org/abs/2602.00629',
            official_repo_url=None,
            official_repo_commit=None,
            online_modules_updated=('td3', 'idm'),
        ),
        'gc_sac': dict(
            port_kind='online_only',
            paper_url='',
            official_repo_url=None,
            official_repo_commit=None,
            online_modules_updated=('actor', 'critic'),
        ),
        'gc_td3': dict(
            port_kind='online_only',
            paper_url='',
            official_repo_url=None,
            official_repo_commit=None,
            online_modules_updated=('actor', 'critic'),
        ),
        'gc_rlpd': dict(
            port_kind='full_action',
            paper_url='https://arxiv.org/abs/2302.02948',
            official_repo_url='https://github.com/ikostrikov/rlpd',
            official_repo_commit=None,
            online_modules_updated=('actor', 'critic'),
            uses_offline_actions=True,
        ),
    }
    if name not in records:
        raise ValueError(f'Unknown algorithm {name!r}.')
    record = records[name]
    action_free_fields = () if name in ('gc_sac', 'gc_td3') else ('observations', 'terminals')
    if name == 'gc_rlpd':
        action_free_fields = ('observations', 'terminals', 'actions')
    metadata = AlgorithmMetadata(
        algorithm=name,
        offline_fields_seen=action_free_fields,
        uses_offline_logged_rewards=False,
        **record,
    )
    metadata.validate()
    return metadata


__all__ = [
    'ALGORITHMS',
    'DEFAULT_OFFLINE_STEPS',
    'algorithm_metadata',
    'create_algorithm',
    'get_algorithm_config',
]
