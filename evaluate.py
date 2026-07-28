"""Evaluate one unified PathBridger checkpoint on the five OGBench tasks."""

from __future__ import annotations

import json
from pathlib import Path

from absl import app, flags
from ml_collections import config_flags

from agents import PathBridgerAgent
from envs.env_utils import make_env_and_datasets
from utils.datasets import PathBridgerDataset
from utils.evaluation import DEFAULT_TASK_IDS, evaluate
from utils.flax_utils import resolve_checkpoint, restore_agent

FLAGS = flags.FLAGS
_DEFAULT_CONFIG = str(
    Path(__file__).resolve().parent / 'configs' / 'pbf' / 'antmaze_medium.py'
)

flags.DEFINE_string('checkpoint_dir', '', 'Checkpoint directory or exact .pkl file.')
flags.DEFINE_integer(
    'checkpoint_step',
    0,
    'Checkpoint step; required for a directory and inferred from an exact params_<step>.pkl file.',
)
flags.DEFINE_string('dataset_dir', '', 'Optional OGBench dataset directory.')
flags.DEFINE_integer('episodes', 50, 'Episodes for each of the five predefined tasks.')
flags.DEFINE_integer('seed', 0, 'Evaluation seed.')
flags.DEFINE_string('output_path', '', 'Optional JSON result path.')

config_flags.DEFINE_config_file(
    'agent',
    _DEFAULT_CONFIG,
    'PathBridger paper-reproduction config used by the checkpoint.',
    lock_config=False,
)


def main(_):
    if not FLAGS.checkpoint_dir:
        raise ValueError('checkpoint_dir is required.')
    if FLAGS.checkpoint_step < 0:
        raise ValueError('checkpoint_step cannot be negative.')
    if FLAGS.episodes < 1:
        raise ValueError('episodes must be at least 1.')
    _, checkpoint_step = resolve_checkpoint(
        FLAGS.checkpoint_dir,
        FLAGS.checkpoint_step,
    )

    config = FLAGS.agent
    env, train_data, _ = make_env_and_datasets(
        str(config.env_name),
        dataset_dir=FLAGS.dataset_dir or None,
    )
    dataset = PathBridgerDataset(train_data, config)
    example_batch = dataset.sample(1)
    agent = PathBridgerAgent.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )
    agent = restore_agent(agent, FLAGS.checkpoint_dir, FLAGS.checkpoint_step)

    metrics = evaluate(
        agent,
        env,
        task_ids=DEFAULT_TASK_IDS,
        episodes_per_task=FLAGS.episodes,
        num_candidates=int(config.eval_num_candidates),
        temperature=float(config.eval_temperature),
        seed=FLAGS.seed,
    )
    result = {
        'checkpoint_step': checkpoint_step,
        'env_name': str(config.env_name),
        'endpoint_distribution': str(config.endpoint_distribution),
        'eval_num_candidates': int(config.eval_num_candidates),
        'eval_temperature': float(config.eval_temperature),
        **{f'evaluation/{key}': value for key, value in metrics.items()},
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if FLAGS.output_path:
        output_path = Path(FLAGS.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as file:
            file.write(text)
            file.write('\n')


def run():
    """Run the command-line evaluation entry point."""

    app.run(main)


if __name__ == '__main__':
    run()
