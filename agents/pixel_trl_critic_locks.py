"""Env-wise TRL critic locks and PBF endpoint scales for pixel PathBridger."""

from __future__ import annotations

from typing import Any


# TRL paper critic hyperparameters (lam / discount / expectile / value_p_*).
# endpoint_value_scale comes from PathBridger PBF paper configs, not TRL.
PIXEL_TRL_CRITIC_LOCKS: dict[str, dict[str, Any]] = {
    'visual-antmaze-medium-navigate-v0': {
        'value_distance_weight_power': 0.0,
        'discount': 0.99,
        'expectile': 0.7,
        'value_geom_sample': True,
        'value_p_curgoal': 0.0,
        'value_p_trajgoal': 1.0,
        'value_p_randomgoal': 0.0,
        'endpoint_value_scale': 10.0,
    },
    'visual-antmaze-large-navigate-v0': {
        'value_distance_weight_power': 0.0,
        'discount': 0.99,
        'expectile': 0.7,
        'value_geom_sample': True,
        'value_p_curgoal': 0.0,
        'value_p_trajgoal': 1.0,
        'value_p_randomgoal': 0.0,
        'endpoint_value_scale': 10.0,
    },
    'visual-cube-single-play-v0': {
        'value_distance_weight_power': 0.7,
        'discount': 0.99,
        'expectile': 0.7,
        'value_geom_sample': True,
        'value_p_curgoal': 0.0,
        'value_p_trajgoal': 1.0,
        'value_p_randomgoal': 0.0,
        'endpoint_value_scale': 5.0,
    },
    'visual-cube-double-play-v0': {
        'value_distance_weight_power': 1.0,
        'discount': 0.99,
        'expectile': 0.7,
        'value_geom_sample': True,
        'value_p_curgoal': 0.0,
        'value_p_trajgoal': 1.0,
        'value_p_randomgoal': 0.0,
        'endpoint_value_scale': 10.0,
    },
    'visual-cube-triple-play-v0': {
        'value_distance_weight_power': 1.0,
        'discount': 0.99,
        'expectile': 0.7,
        'value_geom_sample': True,
        'value_p_curgoal': 0.0,
        'value_p_trajgoal': 1.0,
        'value_p_randomgoal': 0.0,
        'endpoint_value_scale': 10.0,
    },
    'visual-puzzle-3x3-play-v0': {
        'value_distance_weight_power': 0.5,
        'discount': 0.99,
        'expectile': 0.7,
        'value_geom_sample': True,
        'value_p_curgoal': 0.0,
        'value_p_trajgoal': 1.0,
        'value_p_randomgoal': 0.0,
        'endpoint_value_scale': 10.0,
    },
    'visual-puzzle-4x4-play-v0': {
        'value_distance_weight_power': 2.0,
        'discount': 0.99,
        'expectile': 0.7,
        'value_geom_sample': True,
        'value_p_curgoal': 0.0,
        'value_p_trajgoal': 1.0,
        'value_p_randomgoal': 0.0,
        'endpoint_value_scale': 10.0,
    },
    'visual-scene-play-v0': {
        'value_distance_weight_power': 1.0,
        'discount': 0.99,
        'expectile': 0.7,
        'value_geom_sample': True,
        'value_p_curgoal': 0.0,
        'value_p_trajgoal': 1.0,
        'value_p_randomgoal': 0.0,
        'endpoint_value_scale': 5.0,
    },
}


def trl_critic_lock_for_env(env_name: str) -> dict[str, Any]:
    if env_name not in PIXEL_TRL_CRITIC_LOCKS:
        raise KeyError(
            f'No TRL critic lock for {env_name!r}. '
            f'Known: {sorted(PIXEL_TRL_CRITIC_LOCKS)}'
        )
    return dict(PIXEL_TRL_CRITIC_LOCKS[env_name])


__all__ = ['PIXEL_TRL_CRITIC_LOCKS', 'trl_critic_lock_for_env']
