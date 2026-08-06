"""Collection-time exploration schedules for deterministic AF baselines."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def collection_noise_std(
    name: str,
    step: int,
    random_steps: int,
    config: Mapping[str, Any],
) -> float:
    """Return scheduled Gaussian noise for deterministic collection policies."""

    if name not in ('gc_td3', 'gc_oso_decqn_factorized'):
        return 0.0
    progress = min(
        max(int(step) - int(random_steps), 0)
        / max(int(config['collection_noise_decay_steps']), 1),
        1.0,
    )
    return float(config['collection_noise_initial']) * (1.0 - progress) + float(
        config['collection_noise_final']
    ) * progress


__all__ = ['collection_noise_std']
