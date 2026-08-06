"""Evaluate a saved goal-conditioned pixel LAPO checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from absl import app, flags

from agents.pixel_lapo import PixelLAPOAgent
from utils.af_checkpoints import load_af_checkpoint, restore_af_agent
from utils.pixel_evaluation import evaluate_pixel_policy


FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint', '', 'Exact train_pixel_lapo checkpoint path.')
flags.DEFINE_integer('episodes', 50, 'Episodes per official task.')
flags.DEFINE_integer('seed', 0, 'Evaluation seed.')
flags.DEFINE_string('output_path', '', 'Optional JSON output.')


def main(_):
    if not FLAGS.checkpoint:
        raise ValueError('--checkpoint is required.')
    payload = load_af_checkpoint(FLAGS.checkpoint)
    if payload['algorithm'] != 'gc_pixel_lapo':
        raise ValueError('Checkpoint is not a gc_pixel_lapo run.')
    metadata = payload['metadata']
    if metadata.get('observation_modality') != 'rgb_uint8':
        raise ValueError('Checkpoint does not declare the RGB uint8 pixel protocol.')

    import ogbench

    env = ogbench.make_env_and_datasets(metadata['env_name'], env_only=True)
    observation, info = env.reset(
        seed=FLAGS.seed, options={'task_id': 1, 'render_goal': False}
    )
    examples = np.stack(
        [np.asarray(observation, dtype=np.uint8), np.asarray(info['goal'], dtype=np.uint8)]
    )
    template = PixelLAPOAgent.create(
        FLAGS.seed,
        examples,
        int(np.prod(env.action_space.shape)),
        payload['config'],
    )
    agent = restore_af_agent(template, payload)
    metrics = evaluate_pixel_policy(
        agent,
        env,
        episodes_per_task=FLAGS.episodes,
        seed=FLAGS.seed,
    )
    result = {
        'algorithm': payload['algorithm'],
        'checkpoint_step': int(payload['step']),
        'env_name': metadata['env_name'],
        'protocol_id': metadata['protocol_id'],
        **metrics,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if FLAGS.output_path:
        path = Path(FLAGS.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + '\n')


def run():
    app.run(main)


if __name__ == '__main__':
    run()
