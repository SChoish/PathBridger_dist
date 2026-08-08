"""Evaluate any checkpoint produced by the unified visual benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from absl import app, flags

from agents.pixel_registry import (
    PIXEL_ALGORITHMS,
    canonical_pixel_algorithm,
    create_pixel_algorithm,
)
from utils.af_checkpoints import load_af_checkpoint, restore_af_agent
from utils.pixel_data import repeat_pixel_frame
from utils.pixel_evaluation import evaluate_pixel_policy


FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint', '', 'Exact train_pixel.py checkpoint path.')
flags.DEFINE_integer('episodes', 50, 'Episodes per official task.')
flags.DEFINE_integer('seed', 0, 'Evaluation seed.')
flags.DEFINE_string('output_path', '', 'Optional JSON output.')
flags.DEFINE_integer(
    'num_candidates', -1, 'PBF endpoint candidates; -1 uses checkpoint config.'
)
flags.DEFINE_float(
    'endpoint_temperature',
    -1.0,
    'PBF endpoint temperature; negative uses checkpoint config.',
)
flags.DEFINE_boolean(
    'log_each_episode',
    True,
    'Print one line after every episode (task/ep/success/steps/running rate).',
)


def main(_):
    if not FLAGS.checkpoint:
        raise ValueError('--checkpoint is required.')
    if FLAGS.episodes < 1:
        raise ValueError('--episodes must be positive.')
    payload = load_af_checkpoint(FLAGS.checkpoint)
    checkpoint_algorithm = str(payload['algorithm'])
    algorithm = canonical_pixel_algorithm(checkpoint_algorithm)
    if algorithm not in PIXEL_ALGORITHMS:
        raise ValueError(
            f'Checkpoint algorithm {checkpoint_algorithm!r} is not a pixel algorithm.'
        )
    metadata = payload['metadata']
    if metadata.get('observation_modality') != 'rgb_uint8':
        raise ValueError('Checkpoint does not declare the RGB uint8 pixel protocol.')

    import ogbench

    env = ogbench.make_env_and_datasets(metadata['env_name'], env_only=True)
    observation, info = env.reset(
        seed=FLAGS.seed, options={'task_id': 1, 'render_goal': False}
    )
    config = dict(payload['config'])
    if algorithm == 'pixel_pbf' and 'path_rep_dim' not in config:
        # The first pixel-PBF checkpoints predate the compact normalized path
        # representation. They remain evaluable with their original tree.
        config['legacy_raw_representation'] = True
    if FLAGS.num_candidates >= 0:
        if FLAGS.num_candidates < 1:
            raise ValueError('--num_candidates must be positive or -1.')
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
    episodes_path = None
    if FLAGS.output_path:
        path = Path(FLAGS.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        episodes_path = path.with_name(path.stem + '_episodes.jsonl')
    print(
        f'[eval] start env={metadata["env_name"]} '
        f'episodes_per_task={FLAGS.episodes} '
        f'N={int(config.get("eval_num_candidates", 1))} '
        f'T={float(config.get("eval_temperature", 0.0))} '
        f'ckpt_step={int(payload["step"])}'
        + (f' episode_log={episodes_path}' if episodes_path else ''),
        flush=True,
    )
    metrics = evaluate_pixel_policy(
        agent,
        env,
        episodes_per_task=FLAGS.episodes,
        seed=FLAGS.seed,
        log_each_episode=bool(FLAGS.log_each_episode),
        episode_log_path=episodes_path,
    )
    result = {
        'algorithm': algorithm,
        'checkpoint_step': int(payload['step']),
        'env_name': metadata['env_name'],
        'port_kind': metadata['port_kind'],
        'protocol_id': metadata['protocol_id'],
        'num_candidates': int(config.get('eval_num_candidates', 1)),
        'endpoint_temperature': float(config.get('eval_temperature', 0.0)),
        **metrics,
    }
    output = json.dumps(result, indent=2, sort_keys=True)
    print(output, flush=True)
    if FLAGS.output_path:
        path = Path(FLAGS.output_path)
        path.write_text(output + '\n')
        print(f'[eval] wrote summary → {path}', flush=True)


def run():
    app.run(main)


if __name__ == '__main__':
    run()
