"""State-based multitask OGBench evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import numpy as np

DEFAULT_TASK_IDS = (1, 2, 3, 4, 5)
ACTION_CHUNK_HORIZON = 5


def _max_episode_steps(env: Any) -> int:
    spec = getattr(env, 'spec', None)
    max_steps = getattr(spec, 'max_episode_steps', None)
    if max_steps is None:
        max_steps = getattr(env, '_max_episode_steps', None)
    if max_steps is None:
        raise ValueError('Evaluation environment must expose `spec.max_episode_steps`.')
    max_steps = int(max_steps)
    if max_steps < 1:
        raise ValueError(f'Environment max episode length must be positive, got {max_steps}.')
    return max_steps


def _info_success(info: Any) -> bool:
    if not isinstance(info, Mapping):
        return False
    value = np.asarray(info.get('success', False))
    return bool(np.any(value))


def _sample_action_chunk(
    agent: Any,
    observation: np.ndarray,
    goal: np.ndarray,
    *,
    num_candidates: int,
    temperature: float,
    seed,
) -> np.ndarray:
    action_chunks = agent.sample_action_chunks(
        observations=observation.reshape(1, -1),
        goals=goal.reshape(1, -1),
        num_candidates=num_candidates,
        temperature=temperature,
        seed=seed,
    )
    action_chunks = np.asarray(jax.device_get(action_chunks), dtype=np.float32)
    if action_chunks.ndim != 3 or action_chunks.shape[0] != 1:
        raise ValueError(
            'agent.sample_action_chunks must return shape [1, 5, action_dim], '
            f'got {action_chunks.shape}.'
        )
    action_chunk = np.squeeze(action_chunks, axis=0)
    if action_chunk.shape[0] != ACTION_CHUNK_HORIZON:
        raise ValueError(
            f'PathBridger executes {ACTION_CHUNK_HORIZON}-step action chunks, got shape {action_chunk.shape}.'
        )
    return action_chunk


def _evaluate_episode(
    agent: Any,
    env: Any,
    observation: np.ndarray,
    goal: np.ndarray,
    *,
    action_low: np.ndarray,
    action_high: np.ndarray,
    max_episode_steps: int,
    num_candidates: int,
    temperature: float,
    rng,
) -> tuple[bool, Any]:
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    goal = np.asarray(goal, dtype=np.float32).reshape(-1)
    success = False
    terminated = False
    truncated = False
    step = 0

    while step < max_episode_steps and not (terminated or truncated):
        rng, sample_seed = jax.random.split(rng)
        action_chunk = _sample_action_chunk(
            agent,
            observation,
            goal,
            num_candidates=num_candidates,
            temperature=temperature,
            seed=sample_seed,
        )

        for action in action_chunk:
            if step >= max_episode_steps or terminated or truncated:
                break
            action = np.clip(action, action_low, action_high)
            observation, _, terminated, truncated, info = env.step(action)
            observation = np.asarray(observation, dtype=np.float32).reshape(-1)
            success = success or _info_success(info)
            terminated = bool(terminated)
            truncated = bool(truncated)
            step += 1

    return success, rng


def evaluate(
    agent: Any,
    env: Any,
    *,
    task_ids: Sequence[int] = DEFAULT_TASK_IDS,
    episodes_per_task: int = 10,
    num_candidates: int = 1,
    temperature: float = 1.0,
    seed: int = 0,
) -> dict[str, float | int]:
    """Evaluate PathBridger on OGBench tasks using five-step IDM chunks.

    Each replanning call receives batched state and goal arrays and a fresh JAX
    random key. An episode succeeds when the environment reports
    ``info['success']`` on any step.
    """
    task_ids = tuple(int(task_id) for task_id in task_ids)
    if not task_ids:
        raise ValueError('task_ids must contain at least one task.')
    if int(episodes_per_task) < 1:
        raise ValueError('episodes_per_task must be at least 1.')
    if int(num_candidates) < 1:
        raise ValueError('num_candidates must be at least 1.')
    if float(temperature) < 0:
        raise ValueError('temperature must be non-negative.')

    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    max_episode_steps = _max_episode_steps(env)
    rng = jax.random.PRNGKey(int(seed))
    metrics: dict[str, float | int] = {}
    task_success_rates = []

    for task_id in task_ids:
        successes = []
        for _ in range(int(episodes_per_task)):
            observation, info = env.reset(options={'task_id': task_id, 'render_goal': False})
            if not isinstance(info, Mapping) or 'goal' not in info:
                raise RuntimeError(f'Environment reset for task {task_id} did not return info["goal"].')
            success, rng = _evaluate_episode(
                agent,
                env,
                observation,
                info['goal'],
                action_low=action_low,
                action_high=action_high,
                max_episode_steps=max_episode_steps,
                num_candidates=int(num_candidates),
                temperature=float(temperature),
                rng=rng,
            )
            successes.append(float(success))

        task_success = float(np.mean(successes))
        metrics[f'task_{task_id}_success'] = task_success
        task_success_rates.append(task_success)

    metrics['overall_success'] = float(np.mean(task_success_rates))
    metrics['num_tasks'] = len(task_ids)
    metrics['episodes_per_task'] = int(episodes_per_task)
    return metrics


__all__ = ['ACTION_CHUNK_HORIZON', 'DEFAULT_TASK_IDS', 'evaluate']
