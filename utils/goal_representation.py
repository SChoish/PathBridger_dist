"""Minimal goal-representation helpers for state-based OGBench tasks.

``phi`` is the task-relevant achieved-goal representation used by PathBridger:
maze position, puzzle button states, cube positions, or Scene's oracle ordering.
The full compact observation remains available through ``mode='full'``.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp

_MANIP_ARM_JOINT_DIM = 6
_MANIP_HEAD_DIM = 2 * _MANIP_ARM_JOINT_DIM + 3 + 1 + 1 + 1 + 1
_MANIP_CUBE_STRIDE = 3 + 4 + 1 + 1
_MANIP_BUTTON_STRIDE = 2 + 1 + 1
_SCENE_TAIL_DIM = 4
_OGBENCH_SCENE_LAYOUT = (1, 2, 2)
_MAZE_GOAL_XY_INDICES = (0, 1)


def _env_goal_phi_kind(env_name: str | None) -> str:
    if env_name is None or not str(env_name).strip():
        raise ValueError(
            "goal_representation='phi' requires a non-empty OGBench environment name."
        )
    name = str(env_name).lower()
    if 'humanoidmaze' in name:
        return 'humanoidmaze'
    if 'antmaze' in name:
        return 'antmaze'
    if 'puzzle' in name:
        return 'puzzle'
    if 'scene' in name:
        return 'scene'
    if 'cube' in name:
        return 'cube'
    raise ValueError(
        f"goal_representation='phi' does not support env_name={env_name!r}; "
        "expected antmaze, humanoidmaze, puzzle, cube, or scene."
    )


def _cube_position_indices(obs_dim: int) -> tuple[int, ...]:
    remainder = int(obs_dim) - _MANIP_HEAD_DIM
    if remainder < _MANIP_CUBE_STRIDE or remainder % _MANIP_CUBE_STRIDE:
        return ()
    indices: list[int] = []
    for start in range(_MANIP_HEAD_DIM, int(obs_dim), _MANIP_CUBE_STRIDE):
        indices.extend((start, start + 1, start + 2))
    return tuple(indices)


def _button_state_indices(obs_dim: int) -> tuple[int, ...]:
    remainder = int(obs_dim) - _MANIP_HEAD_DIM
    if remainder <= 0 or remainder % _MANIP_BUTTON_STRIDE:
        return ()
    return tuple(
        _MANIP_HEAD_DIM + i * _MANIP_BUTTON_STRIDE + 1
        for i in range(remainder // _MANIP_BUTTON_STRIDE)
    )


def _expected_scene_obs_dim(num_cubes: int, num_buttons: int, num_states: int) -> int:
    return (
        _MANIP_HEAD_DIM
        + num_cubes * _MANIP_CUBE_STRIDE
        + num_buttons * (num_states + 2)
        + _SCENE_TAIL_DIM
    )


def _scene_layout(obs_dim: int, env_name: str | None) -> tuple[int, int, int]:
    """Infer ``(num_cubes, num_buttons, button_states)`` for compact Scene."""

    obs_dim = int(obs_dim)
    if env_name is not None and 'scene' in str(env_name).lower():
        expected = _expected_scene_obs_dim(*_OGBENCH_SCENE_LAYOUT)
        if obs_dim != expected:
            raise ValueError(
                f'Scene obs_dim={obs_dim} does not match the OGBench '
                f'{_OGBENCH_SCENE_LAYOUT} layout (expected {expected}).'
            )
        return _OGBENCH_SCENE_LAYOUT

    middle = obs_dim - _MANIP_HEAD_DIM - _SCENE_TAIL_DIM
    if middle < 0:
        raise ValueError(f'Scene compact obs_dim={obs_dim} is too small.')
    solutions: list[tuple[int, int, int]] = []
    for num_cubes in range(middle // _MANIP_CUBE_STRIDE + 1):
        remainder = middle - num_cubes * _MANIP_CUBE_STRIDE
        for num_states in range(2, 16):
            button_stride = num_states + 2
            if remainder % button_stride:
                continue
            num_buttons = remainder // button_stride
            if num_buttons >= 1:
                solutions.append((num_cubes, num_buttons, num_states))
    if not solutions:
        raise ValueError(f'Could not infer a compact Scene layout from obs_dim={obs_dim}.')
    if len(solutions) > 1:
        if _OGBENCH_SCENE_LAYOUT in solutions:
            return _OGBENCH_SCENE_LAYOUT
        raise ValueError(f'Ambiguous compact Scene layout for obs_dim={obs_dim}: {solutions}.')
    return solutions[0]


def _scene_phi(goals: jnp.ndarray, obs_dim: int, env_name: str | None) -> jnp.ndarray:
    """Match ``SceneEnv.compute_oracle_observation`` field ordering."""

    num_cubes, num_buttons, num_states = _scene_layout(obs_dim, env_name)
    cursor = _MANIP_HEAD_DIM
    parts: list[jnp.ndarray] = []

    cube_indices: list[int] = []
    for _ in range(num_cubes):
        cube_indices.extend((cursor, cursor + 1, cursor + 2))
        cursor += _MANIP_CUBE_STRIDE
    if cube_indices:
        parts.append(jnp.take(goals, jnp.asarray(cube_indices, dtype=jnp.int32), axis=-1))

    button_scalars: list[jnp.ndarray] = []
    button_stride = num_states + 2
    for _ in range(num_buttons):
        one_hot = goals[..., cursor : cursor + num_states]
        button_scalars.append(jnp.argmax(one_hot, axis=-1).astype(jnp.float32))
        cursor += button_stride
    if button_scalars:
        parts.append(jnp.stack(button_scalars, axis=-1))

    expected_cursor = obs_dim - _SCENE_TAIL_DIM
    if cursor != expected_cursor:
        raise ValueError(
            f'Inconsistent Scene layout: parsed through {cursor}, expected {expected_cursor}.'
        )
    # Drawer and window scaled positions are the first element of each tail pair.
    tail_indices = jnp.asarray([obs_dim - 4, obs_dim - 2], dtype=jnp.int32)
    parts.append(jnp.take(goals, tail_indices, axis=-1))
    return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=-1)


def normalize_phi_goal_obs_indices(raw: object) -> tuple[int, ...]:
    """Normalize YAML/ConfigDict phi indices to a tuple."""

    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise TypeError(
            f'phi_goal_obs_indices must be a list or tuple, got {type(raw).__name__}.'
        )
    indices = tuple(int(index) for index in raw)
    if any(index < 0 for index in indices):
        raise ValueError(f'phi_goal_obs_indices must be non-negative, got {indices}.')
    return indices


def infer_phi_goal_obs_indices(
    env_name: str | None, obs_dim: int | None = None
) -> tuple[int, ...]:
    """Infer task-relevant compact-observation indices when representable as a slice.

    Scene returns an empty tuple because its phi includes an argmax over each
    button's one-hot block; :func:`goal_representation` constructs it directly.
    Unknown or incomplete inputs return an empty tuple so config initialization
    can defer validation until the observation dimension is known.
    """

    if env_name is None or not str(env_name).strip():
        return ()
    try:
        kind = _env_goal_phi_kind(env_name)
    except ValueError:
        return ()
    if kind in ('antmaze', 'humanoidmaze'):
        return _MAZE_GOAL_XY_INDICES
    if obs_dim is None:
        return ()
    if kind == 'puzzle':
        return _button_state_indices(int(obs_dim))
    if kind == 'cube':
        return _cube_position_indices(int(obs_dim))
    if kind == 'scene':
        return ()
    raise AssertionError(f'Unhandled phi kind {kind!r}.')


def assert_phi_goal_obs_indices(
    obs_dim: int,
    mode: str,
    phi_goal_obs_indices: Sequence[int] | tuple[int, ...] = (),
    *,
    where: str = 'goal representation',
    env_name: str | None = None,
) -> None:
    """Validate a goal representation against its compact observation shape."""

    # Indices remain in serialized configs for readability.  The environment
    # family is authoritative, matching the research implementation.
    normalize_phi_goal_obs_indices(phi_goal_obs_indices)
    mode = str(mode).lower()
    if mode in ('full', 'raw', 'none', ''):
        return
    if mode not in ('phi', 'auto', 'goal_phi'):
        raise ValueError(f"{where}: unknown goal representation {mode!r}.")

    obs_dim = int(obs_dim)
    try:
        kind = _env_goal_phi_kind(env_name)
        if kind == 'puzzle' and not _button_state_indices(obs_dim):
            raise ValueError(
                f'obs_dim={obs_dim} is incompatible with compact puzzle observations.'
            )
        if kind == 'cube' and not _cube_position_indices(obs_dim):
            raise ValueError(
                f'obs_dim={obs_dim} is incompatible with compact cube observations.'
            )
        if kind == 'scene':
            _scene_layout(obs_dim, env_name)
        if kind in ('antmaze', 'humanoidmaze') and obs_dim < 2:
            raise ValueError(f'obs_dim={obs_dim} is too small for planar maze goals.')
    except ValueError as exc:
        raise ValueError(f'{where}: {exc}') from exc


def goal_representation(
    goals: jnp.ndarray | None,
    mode: str,
    phi_goal_obs_indices: Sequence[int] | tuple[int, ...] = (),
    *,
    env_name: str | None = None,
) -> jnp.ndarray | None:
    """Map full compact goal observations to ``full`` or task-relevant ``phi``."""

    if goals is None:
        return None
    normalize_phi_goal_obs_indices(phi_goal_obs_indices)
    mode = str(mode).lower()
    if mode in ('full', 'raw', 'none', ''):
        return goals
    if mode not in ('phi', 'auto', 'goal_phi'):
        raise ValueError(f"Unknown goal_representation={mode!r}; expected 'full' or 'phi'.")

    obs_dim = int(goals.shape[-1])
    kind = _env_goal_phi_kind(env_name)
    if kind == 'puzzle':
        indices = _button_state_indices(obs_dim)
    elif kind == 'cube':
        indices = _cube_position_indices(obs_dim)
    elif kind in ('antmaze', 'humanoidmaze'):
        indices = _MAZE_GOAL_XY_INDICES if obs_dim >= 2 else ()
    elif kind == 'scene':
        return _scene_phi(goals, obs_dim, env_name)
    else:
        raise AssertionError(f'Unhandled phi kind {kind!r}.')

    if not indices:
        raise ValueError(
            f"goal_representation='phi' is incompatible with env={env_name!r}, "
            f'obs_dim={obs_dim}.'
        )
    return jnp.take(goals, jnp.asarray(indices, dtype=jnp.int32), axis=-1)


__all__ = [
    'assert_phi_goal_obs_indices',
    'goal_representation',
    'infer_phi_goal_obs_indices',
    'normalize_phi_goal_obs_indices',
]
