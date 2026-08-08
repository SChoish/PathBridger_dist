"""Policy-agnostic evaluation for goal-conditioned visual OGBench agents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import numpy as np

from utils.evaluation import DEFAULT_TASK_IDS, _info_success, _max_episode_steps
from utils.pixel_data import repeat_pixel_frame, stack_pixel_history


def _frame(value: Any, *, name: str) -> np.ndarray:
    frame = np.asarray(value)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f'{name} must have shape [H, W, 3], got {frame.shape}.')
    if frame.dtype != np.uint8:
        raise ValueError(f'{name} must be uint8, got {frame.dtype}.')
    return frame


def evaluate_pixel_policy(
    policy: Any,
    env: Any,
    *,
    task_ids: Sequence[int] = DEFAULT_TASK_IDS,
    episodes_per_task: int = 10,
    seed: int = 0,
) -> dict[str, float | int]:
    """Evaluate pixels, executing PBF chunks and one-step actions otherwise."""

    task_ids = tuple(int(value) for value in task_ids)
    if not task_ids:
        raise ValueError('task_ids must contain at least one task.')
    if episodes_per_task < 1:
        raise ValueError('episodes_per_task must be positive.')
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    max_steps = _max_episode_steps(env)
    config = getattr(policy, 'config', {})
    frame_stack = int(config.get('frame_stack', 1))
    if frame_stack < 1:
        raise ValueError('Pixel policy frame_stack must be positive.')
    rng = jax.random.PRNGKey(int(seed))
    metrics: dict[str, float | int] = {}
    task_rates = []
    for task_id in task_ids:
        successes = []
        for episode in range(int(episodes_per_task)):
            observation, reset_info = env.reset(
                seed=int(seed) * 100_000 + task_id * 1_000 + episode,
                options={'task_id': task_id, 'render_goal': False},
            )
            if not isinstance(reset_info, Mapping) or 'goal' not in reset_info:
                raise RuntimeError('Visual OGBench reset must provide info["goal"].')
            observation = _frame(observation, name='observation')
            goal = _frame(reset_info['goal'], name='goal')
            history = [observation.copy()]
            policy_goal = repeat_pixel_frame(goal, frame_stack)
            success = False
            terminated = truncated = False
            step = 0
            while step < max_steps and not (terminated or truncated):
                rng, action_rng = jax.random.split(rng)
                policy_observation = stack_pixel_history(history, frame_stack)
                if hasattr(policy, 'sample_action_chunks'):
                    actions = policy.sample_action_chunks(
                        policy_observation[None, ...],
                        policy_goal[None, ...],
                        seed=action_rng,
                    )
                    actions = np.asarray(
                        jax.device_get(actions), dtype=np.float32
                    )
                    if actions.ndim != 3 or actions.shape[0] != 1:
                        raise ValueError(
                            'Pixel PBF must return [1, 5, action_dim], '
                            f'got {actions.shape}.'
                        )
                    action_chunk = actions[0]
                else:
                    actions = policy.sample_actions(
                        policy_observation[None, ...],
                        policy_goal[None, ...],
                        seed=action_rng,
                        temperature=0.0,
                    )
                    actions = np.asarray(
                        jax.device_get(actions), dtype=np.float32
                    )
                    if actions.ndim != 2 or actions.shape[0] != 1:
                        raise ValueError(
                            'Pixel policy must return [1, action_dim], '
                            f'got {actions.shape}.'
                        )
                    action_chunk = actions
                for action in action_chunk:
                    if step >= max_steps or terminated or truncated:
                        break
                    observation, _, terminated, truncated, info = env.step(
                        np.clip(action, action_low, action_high)
                    )
                    observation = _frame(observation, name='observation')
                    history.append(observation.copy())
                    success = success or _info_success(info)
                    step += 1
            successes.append(float(success))
        rate = float(np.mean(successes))
        metrics[f'task_{task_id}_success'] = rate
        task_rates.append(rate)
    metrics['overall_success'] = float(np.mean(task_rates))
    metrics['num_tasks'] = len(task_rates)
    metrics['episodes_per_task'] = int(episodes_per_task)
    return metrics


__all__ = ['evaluate_pixel_policy']
