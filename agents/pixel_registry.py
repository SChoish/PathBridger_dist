"""Construction and provenance registry for isolated visual benchmarks."""

from __future__ import annotations

from typing import Any

from agents.pixel_drq import PixelDrQAgent, get_config as drq_config
from agents.pixel_lapo import PixelLAPOAgent, get_config as lapo_config
from agents.pixel_pathbridger import (
    PixelPathBridgerAgent,
    get_config as pathbridger_config,
)
from utils.provenance import AlgorithmMetadata


PIXEL_ALGORITHMS = (
    'pixel_pbf',
    'gc_pixel_lapo_decoder',
    'gc_pixel_drqv2',
    'vip_style_frozen_gc_drqv2',
    'vip_style_finetuned_gc_drqv2',
    'gc_pixel_apv_style_drq',
)
PIXEL_ALGORITHM_ALIASES = {
    'pixel_pathbridger': 'pixel_pbf',
    'gc_pixel_lapo': 'gc_pixel_lapo_decoder',
    'vip_frozen_gc_drqv2': 'vip_style_frozen_gc_drqv2',
    'vip_finetuned_gc_drqv2': 'vip_style_finetuned_gc_drqv2',
    'gc_pixel_apv': 'gc_pixel_apv_style_drq',
}

# This is the executable training contract for the visual comparison.  Direct
# gradient updates are listed separately from target-network EMA updates; the
# latter are intentionally not reported as trainable method modules.
PIXEL_METHOD_SCOPES = {
    'pixel_pbf': {
        'offline_trainable_modules': (
            'encoder',
            'endpoint',
            'bridge',
            'value',
            'idm',
        ),
        'online_trainable_modules': (),
        'online_frozen_modules': (
            'encoder',
            'target_encoder',
            'endpoint',
            'bridge',
            'value',
            'target_value',
            'idm',
        ),
    },
    'gc_pixel_lapo_decoder': {
        'offline_trainable_modules': ('latent_model', 'latent_policy'),
        'online_trainable_modules': ('decoder',),
        'online_frozen_modules': ('latent_model', 'latent_policy'),
    },
    'gc_pixel_drqv2': {
        'offline_trainable_modules': (),
        'online_trainable_modules': ('encoder', 'actor', 'critic'),
        'online_frozen_modules': (
            'video_predictor',
            'action_dynamics',
            'world_decoder',
        ),
    },
    'vip_style_frozen_gc_drqv2': {
        'offline_trainable_modules': ('encoder',),
        'online_trainable_modules': ('actor', 'critic'),
        'online_frozen_modules': (
            'encoder',
            'target_encoder',
            'video_predictor',
            'action_dynamics',
            'world_decoder',
        ),
    },
    'vip_style_finetuned_gc_drqv2': {
        'offline_trainable_modules': ('encoder',),
        'online_trainable_modules': ('encoder', 'actor', 'critic'),
        'online_frozen_modules': (
            'video_predictor',
            'action_dynamics',
            'world_decoder',
        ),
    },
    'gc_pixel_apv_style_drq': {
        'offline_trainable_modules': (
            'encoder',
            'video_predictor',
            'world_decoder',
        ),
        'online_trainable_modules': (
            'encoder',
            'actor',
            'critic',
            'action_dynamics',
        ),
        'online_frozen_modules': ('video_predictor', 'world_decoder'),
    },
}

_PIXEL_METHOD_CONFIG_CONTRACTS = {
    'pixel_pbf': {'offline_action_free': False},
    'gc_pixel_drqv2': {
        'pretraining': 'none',
        'freeze_encoder_online': False,
    },
    'vip_style_frozen_gc_drqv2': {
        'pretraining': 'vip',
        'freeze_encoder_online': True,
    },
    'vip_style_finetuned_gc_drqv2': {
        'pretraining': 'vip',
        'freeze_encoder_online': False,
    },
    'gc_pixel_apv_style_drq': {
        'pretraining': 'apv',
        'freeze_encoder_online': False,
    },
}


def canonical_pixel_algorithm(name: str) -> str:
    name = str(name).lower()
    return PIXEL_ALGORITHM_ALIASES.get(name, name)


def pixel_method_scope(name: str) -> dict[str, tuple[str, ...]]:
    """Return the immutable phase/module contract for one visual method."""

    name = canonical_pixel_algorithm(name)
    if name not in PIXEL_METHOD_SCOPES:
        raise ValueError(f'Unknown pixel algorithm {name!r}.')
    return {
        key: tuple(value) for key, value in PIXEL_METHOD_SCOPES[name].items()
    }


def _validate_method_config(name: str, config: dict[str, Any]) -> None:
    for key, expected in _PIXEL_METHOD_CONFIG_CONTRACTS.get(name, {}).items():
        actual = config.get(key)
        if actual != expected:
            raise ValueError(
                f'{name} requires {key}={expected!r}, got {actual!r}. '
                'Choose the matching algorithm name instead of overriding its '
                'training regime.'
            )


def get_pixel_config(name: str) -> dict[str, Any]:
    name = canonical_pixel_algorithm(name)
    if name == 'pixel_pbf':
        config = pathbridger_config().to_dict()
        config['offline_action_free'] = False
        return config
    if name == 'gc_pixel_lapo_decoder':
        return lapo_config().to_dict()
    if name == 'gc_pixel_drqv2':
        return drq_config('none').to_dict()
    if name == 'vip_style_frozen_gc_drqv2':
        return drq_config('vip', freeze_encoder_online=True).to_dict()
    if name == 'vip_style_finetuned_gc_drqv2':
        return drq_config('vip', freeze_encoder_online=False).to_dict()
    if name == 'gc_pixel_apv_style_drq':
        return drq_config('apv', freeze_encoder_online=False).to_dict()
    raise ValueError(f'Unknown pixel algorithm {name!r}.')


def create_pixel_algorithm(
    name: str,
    *,
    seed: int,
    example_images: Any,
    action_dim: int,
    config: dict[str, Any] | None = None,
):
    name = canonical_pixel_algorithm(name)
    resolved = get_pixel_config(name)
    if config:
        resolved.update(config)
    _validate_method_config(name, resolved)
    if name == 'pixel_pbf':
        return (
            PixelPathBridgerAgent.create(
                seed, example_images, action_dim, resolved
            ),
            resolved,
        )
    if name == 'gc_pixel_lapo_decoder':
        return (
            PixelLAPOAgent.create(seed, example_images, action_dim, resolved),
            resolved,
        )
    if name in PIXEL_ALGORITHMS:
        return (
            PixelDrQAgent.create(seed, example_images, action_dim, resolved),
            resolved,
        )
    raise ValueError(f'Unknown pixel algorithm {name!r}.')


def pixel_algorithm_metadata(name: str) -> AlgorithmMetadata:
    name = canonical_pixel_algorithm(name)
    records = {
        'pixel_pbf': dict(
            port_kind='full_action',
            paper_url='',
            official_repo_url=None,
            official_repo_commit=None,
            offline_fields_seen=('observations', 'terminals', 'actions'),
            online_modules_updated=(),
            uses_offline_actions=True,
            implementation_notes=(
                'Full-data offline PBF from pb_bundle with an IMPALA-small '
                'pixel encoder and EMA target encoder. TransV, rectified-flow '
                'endpoint proposal, endpoint-pinned bridge, and IDM are '
                'trained jointly from offline images and actions.'
            ),
        ),
        'gc_pixel_lapo_decoder': dict(
            port_kind='goal_conditioned_adaptation',
            paper_url='https://arxiv.org/abs/2312.10812',
            official_repo_url='https://github.com/schmidtdominik/LAPO',
            official_repo_commit='c3844f7e8c92e900bf7547a265f14089ac68b121',
            offline_fields_seen=('observations', 'terminals'),
            online_modules_updated=('decoder',),
            implementation_notes='Continuous goal-image OGBench adaptation. Unlike native online LAPO, this controlled variant grounds latent codes with a decoder only.',
        ),
        'gc_pixel_drqv2': dict(
            port_kind='online_only',
            paper_url='https://arxiv.org/abs/2107.09645',
            official_repo_url='https://github.com/facebookresearch/drqv2',
            official_repo_commit='c0c650b76c6e5d22a7eb5f2edffd1440fe94f8ef',
            offline_fields_seen=(),
            online_modules_updated=('encoder', 'actor', 'critic'),
            implementation_notes='Goal-image OGBench adaptation of DrQ-v2; no video pretraining.',
        ),
        'vip_style_frozen_gc_drqv2': dict(
            port_kind='goal_conditioned_adaptation',
            paper_url='https://arxiv.org/abs/2210.00030',
            official_repo_url='https://github.com/facebookresearch/vip',
            official_repo_commit='81b052014591bb157cfb036fc3bd0b213653e86b',
            offline_fields_seen=('observations', 'terminals'),
            online_modules_updated=('actor', 'critic'),
            implementation_notes='OGBench-trained VIP-style temporal value encoder frozen during goal-conditioned DrQ-v2 control; not the released Ego4D checkpoint.',
        ),
        'vip_style_finetuned_gc_drqv2': dict(
            port_kind='goal_conditioned_adaptation',
            paper_url='https://arxiv.org/abs/2210.00030',
            official_repo_url='https://github.com/facebookresearch/vip',
            official_repo_commit='81b052014591bb157cfb036fc3bd0b213653e86b',
            offline_fields_seen=('observations', 'terminals'),
            online_modules_updated=('encoder', 'actor', 'critic'),
            implementation_notes='OGBench-trained VIP-style temporal value encoder fine-tuned with goal-conditioned DrQ-v2 and episode-local future-image HER.',
        ),
        'gc_pixel_apv_style_drq': dict(
            port_kind='goal_conditioned_adaptation',
            paper_url='https://proceedings.mlr.press/v162/seo22a.html',
            official_repo_url='https://github.com/younggyoseo/apv',
            official_repo_commit='3efa6218c58cad05479d8d50f173b1afb34664ae',
            offline_fields_seen=('observations', 'terminals'),
            online_modules_updated=(
                'encoder',
                'actor',
                'critic',
                'action_dynamics',
            ),
            implementation_notes='JAX OGBench adaptation: action-free latent video prediction followed by action-conditioned latent dynamics plus goal-conditioned DrQ-v2, not native DreamerV2 APV.',
        ),
    }
    if name not in records:
        raise ValueError(f'Unknown pixel algorithm {name!r}.')
    expected_online = pixel_method_scope(name)['online_trainable_modules']
    declared_online = tuple(records[name]['online_modules_updated'])
    if declared_online != expected_online:
        raise RuntimeError(
            f'Pixel method scope mismatch for {name}: metadata={declared_online} '
            f'vs executable={expected_online}.'
        )
    metadata = AlgorithmMetadata(
        algorithm=name,
        uses_offline_logged_rewards=False,
        **records[name],
    )
    metadata.validate()
    return metadata


__all__ = [
    'PIXEL_ALGORITHMS',
    'PIXEL_ALGORITHM_ALIASES',
    'PIXEL_METHOD_SCOPES',
    'canonical_pixel_algorithm',
    'create_pixel_algorithm',
    'get_pixel_config',
    'pixel_method_scope',
    'pixel_algorithm_metadata',
]
