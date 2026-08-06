"""Policy-agnostic OGBench evaluation for offline-to-online agents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import numpy as np

from utils.evaluation import DEFAULT_TASK_IDS, _info_success, _max_episode_steps


def evaluate_policy(
    policy: Any,
    env: Any,
    *,
    task_ids: Sequence[int] = DEFAULT_TASK_IDS,
    episodes_per_task: int = 10,
    seed: int = 0,
    execute_horizon: int = 1,
) -> dict[str, float | int]:
    """Evaluate an object exposing ``sample_actions(obs, goals, seed)``."""

    if execute_horizon < 1:
        raise ValueError('execute_horizon must be positive.')
    if episodes_per_task < 1:
        raise ValueError('episodes_per_task must be positive.')
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    max_steps = _max_episode_steps(env)
    rng = jax.random.PRNGKey(int(seed))
    metrics: dict[str, float | int] = {}
    task_rates = []
    for task_id in tuple(int(value) for value in task_ids):
        successes = []
        for episode in range(int(episodes_per_task)):
            observation, info = env.reset(
                seed=int(seed) * 100_000 + task_id * 1_000 + episode,
                options={'task_id': task_id, 'render_goal': False},
            )
            if not isinstance(info, Mapping) or 'goal' not in info:
                raise RuntimeError('OGBench reset must provide info["goal"].')
            goal = np.asarray(info['goal'], dtype=np.float32).reshape(-1)
            observation = np.asarray(observation, dtype=np.float32).reshape(-1)
            success = False
            terminated = truncated = False
            step = 0
            while step < max_steps and not (terminated or truncated):
                rng, action_rng = jax.random.split(rng)
                actions = policy.sample_actions(
                    observation[None, :], goal[None, :], seed=action_rng, temperature=0.0
                )
                actions = np.asarray(jax.device_get(actions), dtype=np.float32)
                if actions.ndim == 1:
                    actions = actions[None, :]
                if actions.ndim == 3:
                    actions = actions[0]
                elif actions.ndim == 2:
                    actions = actions[:1]
                else:
                    raise ValueError(f'Unexpected policy action shape {actions.shape}.')
                for action in actions[:execute_horizon]:
                    observation, _, terminated, truncated, step_info = env.step(
                        np.clip(action, action_low, action_high)
                    )
                    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
                    success = success or _info_success(step_info)
                    step += 1
                    if step >= max_steps or terminated or truncated:
                        break
            successes.append(float(success))
        rate = float(np.mean(successes))
        metrics[f'task_{task_id}_success'] = rate
        task_rates.append(rate)
    metrics['overall_success'] = float(np.mean(task_rates))
    metrics['num_tasks'] = len(task_rates)
    metrics['episodes_per_task'] = int(episodes_per_task)
    return metrics


__all__ = ['evaluate_policy']
