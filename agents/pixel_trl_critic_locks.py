"""Exact TRL critic and PathBridger locks for pixel PBF tuning."""

from __future__ import annotations

from typing import Any


_TRL_SHARED: dict[str, Any] = {
    'discount': 0.99,
    'expectile': 0.7,
    'value_geom_sample': True,
    'value_p_curgoal': 0.0,
    'value_p_trajgoal': 1.0,
    'value_p_randomgoal': 0.0,
    'value_hidden_dims': (512, 512, 512),
    'value_layer_norm': True,
    'value_learning_rate': 3e-4,
    'value_tau': 0.005,
}


def _trl_lock(distance_power: float) -> dict[str, Any]:
    return {**_TRL_SHARED, 'value_distance_weight_power': distance_power}


# Exact critic settings from the official TRL commands.  Pixel image batch size
# remains a visual-memory exception and is deliberately not part of this lock.
PIXEL_TRL_CRITIC_LOCKS: dict[str, dict[str, Any]] = {
    'visual-antmaze-medium-navigate-v0': _trl_lock(0.0),
    'visual-antmaze-large-navigate-v0': _trl_lock(0.0),
    'visual-cube-single-play-v0': _trl_lock(0.7),
    'visual-cube-double-play-v0': _trl_lock(1.0),
    'visual-cube-triple-play-v0': _trl_lock(1.0),
    'visual-puzzle-3x3-play-v0': _trl_lock(0.5),
    'visual-puzzle-4x4-play-v0': _trl_lock(2.0),
    'visual-scene-play-v0': _trl_lock(1.0),
}

# K is environment-specific in PBF; h_a, flow integration, architecture, and
# optimizer are shared paper settings and are not sweep axes.
PIXEL_PBF_ENDPOINT_HORIZONS = {
    'visual-antmaze-medium-navigate-v0': 25,
    'visual-antmaze-large-navigate-v0': 25,
    'visual-cube-single-play-v0': 40,
    'visual-cube-double-play-v0': 40,
    'visual-cube-triple-play-v0': 40,
    'visual-puzzle-3x3-play-v0': 25,
    'visual-puzzle-4x4-play-v0': 25,
    'visual-scene-play-v0': 25,
}

PIXEL_PBF_GAP_SEARCH = (5.0, 10.0)
PIXEL_PBF_NT_SEARCH = ((1, 0.0), (2, 0.25), (16, 0.5), (32, 1.0))


def _pixel_augmentation_probability(env_name: str) -> float:
    return 0.5 if any(
        token in env_name for token in ('cube', 'scene', 'puzzle')
    ) else 0.0


def trl_critic_lock_for_env(env_name: str) -> dict[str, Any]:
    if env_name not in PIXEL_TRL_CRITIC_LOCKS:
        raise KeyError(
            f'No TRL critic lock for {env_name!r}. '
            f'Known: {sorted(PIXEL_TRL_CRITIC_LOCKS)}'
        )
    return dict(PIXEL_TRL_CRITIC_LOCKS[env_name])


def pixel_pbf_lock_for_env(env_name: str) -> dict[str, Any]:
    critic = trl_critic_lock_for_env(env_name)
    return {
        **critic,
        'encoder': 'impala_small',
        'feature_dim': 512,
        'path_rep_dim': 32,
        'normalize_path_rep': True,
        'stop_planner_rep_grad': True,
        'geometry_target_source': 'online',
        'log_rep_diagnostics': True,
        'frame_stack': 1,
        'offline_batch_size': 256,
        'p_aug': _pixel_augmentation_probability(env_name),
        'encoder_learning_rate': 3e-4,
        'path_horizon': 5,
        'endpoint_horizon': PIXEL_PBF_ENDPOINT_HORIZONS[env_name],
        'endpoint_flow_steps': 8,
        'hidden_dims': (512, 512, 512),
        'bridge_layer_norm': True,
        'learning_rate': 3e-4,
        'tau': 0.005,
        'idm_hidden_dims': (512, 512, 512),
        'idm_layer_norm': True,
        'idm_learning_rate': 3e-4,
    }


def apply_pixel_pbf_locks(
    env_name: str, overrides: dict[str, Any]
) -> dict[str, Any]:
    """Merge locks and reject silent changes to fixed TRL/PBF settings."""

    result = dict(overrides)
    for key, expected in pixel_pbf_lock_for_env(env_name).items():
        if key in result:
            actual = result[key]
            comparable_actual = tuple(actual) if isinstance(actual, list) else actual
            if comparable_actual != expected:
                raise ValueError(
                    f'Pixel PBF locks {key}={expected!r} for {env_name}; '
                    f'got override {actual!r}. Only gap and (N, T) are tuned.'
                )
        result[key] = expected
    return result


__all__ = [
    'PIXEL_PBF_ENDPOINT_HORIZONS',
    'PIXEL_PBF_GAP_SEARCH',
    'PIXEL_PBF_NT_SEARCH',
    'PIXEL_TRL_CRITIC_LOCKS',
    'apply_pixel_pbf_locks',
    'pixel_pbf_lock_for_env',
    'trl_critic_lock_for_env',
]
