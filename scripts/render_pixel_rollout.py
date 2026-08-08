#!/usr/bin/env python3
"""Render pixel PBF rollouts to mp4 (obs | goal side-by-side)."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import jax
import numpy as np
from absl import app, flags
from PIL import Image, ImageDraw, ImageFont

from agents.pixel_registry import (
    PIXEL_ALGORITHMS,
    canonical_pixel_algorithm,
    create_pixel_algorithm,
)
from utils.af_checkpoints import load_af_checkpoint, restore_af_agent
from utils.evaluation import DEFAULT_TASK_IDS, _info_success, _max_episode_steps
from utils.pixel_data import repeat_pixel_frame, stack_pixel_history


FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint', '', 'Checkpoint .pkl path.')
flags.DEFINE_string('output_dir', '', 'Directory for mp4 + summary json.')
flags.DEFINE_integer('episodes', 1, 'Episodes per task.')
flags.DEFINE_string('task_ids', '1,2,3', 'Comma-separated task ids.')
flags.DEFINE_integer('seed', 0, 'Base seed.')
flags.DEFINE_integer('num_candidates', -1, 'Override Best-of-N; -1 = ckpt.')
flags.DEFINE_float('endpoint_temperature', -1.0, 'Override T; <0 = ckpt.')
flags.DEFINE_float('fps', 20.0, 'Output video fps.')
flags.DEFINE_integer('scale', 4, 'Nearest-neighbor upsample factor.')


def _parse_tasks(text: str) -> tuple[int, ...]:
    tasks = tuple(int(x) for x in text.split(',') if x.strip())
    if not tasks:
        raise ValueError('--task_ids must be non-empty')
    return tasks


def _annotate(frame: np.ndarray, lines: list[str], scale: int) -> np.ndarray:
    img = Image.fromarray(frame)
    if scale > 1:
        img = img.resize(
            (img.width * scale, img.height * scale), Image.Resampling.NEAREST
        )
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    y = 4
    for line in lines:
        draw.text((4, y), line, fill=(255, 255, 0), font=font)
        y += 12
    return np.asarray(img, dtype=np.uint8)


def _compose(obs: np.ndarray, goal: np.ndarray, lines: list[str], scale: int) -> np.ndarray:
    obs_a = _annotate(obs, lines, scale)
    goal_a = _annotate(goal, ['goal'], scale)
    gap = np.zeros((obs_a.shape[0], 4, 3), dtype=np.uint8)
    return np.concatenate([obs_a, gap, goal_a], axis=1)


def main(_):
    if not FLAGS.checkpoint:
        raise ValueError('--checkpoint is required')
    if not FLAGS.output_dir:
        raise ValueError('--output_dir is required')
    out_dir = Path(FLAGS.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = load_af_checkpoint(FLAGS.checkpoint)
    algorithm = canonical_pixel_algorithm(str(payload['algorithm']))
    if algorithm not in PIXEL_ALGORITHMS:
        raise ValueError(f'unsupported algorithm {algorithm}')
    metadata = payload['metadata']
    env_name = metadata['env_name']

    import ogbench

    env = ogbench.make_env_and_datasets(env_name, env_only=True)
    observation, info = env.reset(
        seed=FLAGS.seed, options={'task_id': 1, 'render_goal': False}
    )
    config = dict(payload['config'])
    if FLAGS.num_candidates >= 0:
        config['eval_num_candidates'] = int(FLAGS.num_candidates)
    if FLAGS.endpoint_temperature >= 0.0:
        config['eval_temperature'] = float(FLAGS.endpoint_temperature)
    frame_stack = int(config.get('frame_stack', 1))
    examples = np.stack(
        [
            repeat_pixel_frame(np.asarray(observation, dtype=np.uint8), frame_stack),
            repeat_pixel_frame(np.asarray(info['goal'], dtype=np.uint8), frame_stack),
        ]
    )
    template, _ = create_pixel_algorithm(
        algorithm,
        seed=FLAGS.seed,
        example_images=examples,
        action_dim=int(np.prod(env.action_space.shape)),
        config=config,
    )
    agent = restore_af_agent(template, payload)
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    max_steps = _max_episode_steps(env)
    n = int(config.get('eval_num_candidates', 1))
    t = float(config.get('eval_temperature', 0.0))
    tasks = _parse_tasks(FLAGS.task_ids)
    rng = jax.random.PRNGKey(int(FLAGS.seed))
    summary = []

    print(
        f'[render] env={env_name} N={n} T={t} tasks={tasks} '
        f'episodes={FLAGS.episodes} ckpt_step={payload["step"]}',
        flush=True,
    )

    for task_id in tasks:
        for episode in range(int(FLAGS.episodes)):
            observation, reset_info = env.reset(
                seed=int(FLAGS.seed) * 100_000 + task_id * 1_000 + episode,
                options={'task_id': int(task_id), 'render_goal': False},
            )
            observation = np.asarray(observation, dtype=np.uint8)
            goal = np.asarray(reset_info['goal'], dtype=np.uint8)
            history = [observation.copy()]
            policy_goal = repeat_pixel_frame(goal, frame_stack)
            success = False
            terminated = truncated = False
            step = 0
            frames = [
                _compose(
                    observation,
                    goal,
                    [f'task={task_id} ep={episode}', f'step=0 success=0'],
                    FLAGS.scale,
                )
            ]
            while step < max_steps and not (terminated or truncated):
                rng, action_rng = jax.random.split(rng)
                policy_observation = stack_pixel_history(history, frame_stack)
                actions = agent.sample_action_chunks(
                    policy_observation[None, ...],
                    policy_goal[None, ...],
                    seed=action_rng,
                    num_candidates=n,
                    temperature=t,
                )
                action_chunk = np.asarray(jax.device_get(actions), dtype=np.float32)[0]
                for action in action_chunk:
                    if step >= max_steps or terminated or truncated:
                        break
                    observation, _, terminated, truncated, info = env.step(
                        np.clip(action, action_low, action_high)
                    )
                    observation = np.asarray(observation, dtype=np.uint8)
                    history.append(observation.copy())
                    success = success or _info_success(info)
                    step += 1
                    frames.append(
                        _compose(
                            observation,
                            goal,
                            [
                                f'task={task_id} ep={episode}',
                                f'step={step} success={int(success)}',
                            ],
                            FLAGS.scale,
                        )
                    )
            tag = f'task{task_id}_ep{episode}_s{int(success)}'
            mp4 = out_dir / f'{tag}.mp4'
            imageio.mimsave(mp4, frames, fps=float(FLAGS.fps))
            # also a short gif of last 80 frames for quick preview
            gif = out_dir / f'{tag}.gif'
            imageio.mimsave(gif, frames[::max(1, len(frames) // 80)], fps=10)
            row = {
                'task_id': int(task_id),
                'episode': int(episode),
                'success': int(success),
                'steps': int(step),
                'mp4': str(mp4),
                'gif': str(gif),
            }
            summary.append(row)
            print(f'[render] wrote {mp4} success={int(success)} steps={step}', flush=True)

    (out_dir / 'summary.json').write_text(
        json.dumps(
            {
                'checkpoint': str(Path(FLAGS.checkpoint).resolve()),
                'env_name': env_name,
                'num_candidates': n,
                'endpoint_temperature': t,
                'rollouts': summary,
            },
            indent=2,
        )
        + '\n'
    )
    print(f'[render] done → {out_dir}', flush=True)


if __name__ == '__main__':
    app.run(main)
