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
    'pixel_pathbridger_online_idm',
    'gc_pixel_lapo_decoder',
    'gc_pixel_drqv2',
    'vip_style_frozen_gc_drqv2',
    'vip_style_finetuned_gc_drqv2',
    'gc_pixel_apv_style_drq',
)
PIXEL_ALGORITHM_ALIASES = {
    'gc_pixel_lapo': 'gc_pixel_lapo_decoder',
    'vip_frozen_gc_drqv2': 'vip_style_frozen_gc_drqv2',
    'vip_finetuned_gc_drqv2': 'vip_style_finetuned_gc_drqv2',
    'gc_pixel_apv': 'gc_pixel_apv_style_drq',
}


def canonical_pixel_algorithm(name: str) -> str:
    name = str(name).lower()
    return PIXEL_ALGORITHM_ALIASES.get(name, name)


def get_pixel_config(name: str) -> dict[str, Any]:
    name = canonical_pixel_algorithm(name)
    if name == 'pixel_pathbridger_online_idm':
        return pathbridger_config().to_dict()
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
    if name == 'pixel_pathbridger_online_idm':
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
        'pixel_pathbridger_online_idm': dict(
            port_kind='proposed',
            paper_url='',
            official_repo_url=None,
            official_repo_commit=None,
            offline_fields_seen=('observations', 'terminals'),
            online_modules_updated=('idm',),
            implementation_notes='Proposed visual extension: an action-free endpoint-pinned latent path bridge is frozen online, while a separately initialized IDM is grounded from new RGB/action transitions.',
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
    'canonical_pixel_algorithm',
    'create_pixel_algorithm',
    'get_pixel_config',
    'pixel_algorithm_metadata',
]
