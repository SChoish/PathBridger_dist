"""Train PathBridger on a fixed state-based OGBench dataset."""

from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import jax
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

from agents import PathBridgerAgent
from envs.env_utils import make_env_and_datasets
from utils.datasets import PathBridgerDataset
from utils.evaluation import DEFAULT_TASK_IDS, evaluate
from utils.flax_utils import resolve_checkpoint, restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb

FLAGS = flags.FLAGS

_BATCH_SIZE = 1024
_DEFAULT_CONFIG = str(
    Path(__file__).resolve().parent / 'configs' / 'pbf' / 'antmaze_medium.py'
)

flags.DEFINE_string('run_group', 'Debug', 'Experiment group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('save_dir', 'exp/', 'Root output directory.')
flags.DEFINE_string('dataset_dir', '', 'Optional OGBench dataset directory.')
flags.DEFINE_string('restore_path', '', 'Checkpoint directory or exact .pkl file.')
flags.DEFINE_integer(
    'restore_step',
    0,
    'Checkpoint step; required for a directory and inferred from an exact params_<step>.pkl file.',
)
flags.DEFINE_integer('train_steps', 1_000_000, 'Number of gradient updates.')
flags.DEFINE_integer('log_interval', 5_000, 'Training CSV/W&B logging interval.')
flags.DEFINE_integer('eval_interval', 100_000, 'Environment evaluation interval; 0 disables intermediate eval.')
flags.DEFINE_integer('save_interval', 100_000, 'Checkpoint interval; 0 saves only the final checkpoint.')
flags.DEFINE_integer('eval_episodes', 50, 'Evaluation episodes for each of the five OGBench tasks.')
flags.DEFINE_boolean('use_wandb', False, 'Enable optional Weights & Biases logging.')
flags.DEFINE_boolean('use_tqdm', True, 'Show a training progress bar.')
flags.DEFINE_boolean(
    'async_prefetch',
    True,
    'Overlap host batch sampling with the accelerator update.',
)

config_flags.DEFINE_config_file(
    'agent',
    _DEFAULT_CONFIG,
    'PathBridger agent and paper-reproduction environment config.',
    lock_config=False,
)


def _host_metrics(info) -> dict[str, float]:
    metrics = {}
    for key, value in info.items():
        array = np.asarray(jax.device_get(value))
        if array.size != 1:
            raise ValueError(f'Training metric {key!r} must be scalar, got shape {array.shape}.')
        metrics[str(key)] = float(array.reshape(()))
    return metrics


def _write_run_config(run_dir: str, config) -> None:
    payload = {
        'flags': get_flag_dict(),
        'agent': config.to_dict() if hasattr(config, 'to_dict') else dict(config),
    }
    with open(os.path.join(run_dir, 'flags.json'), 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write('\n')


def _validate_runtime_flags() -> None:
    if FLAGS.train_steps < 1:
        raise ValueError('train_steps must be at least 1.')
    if FLAGS.restore_step < 0:
        raise ValueError('restore_step cannot be negative.')
    if FLAGS.restore_step and not FLAGS.restore_path:
        raise ValueError('restore_step requires restore_path.')
    if FLAGS.log_interval < 1:
        raise ValueError('log_interval must be at least 1.')
    if FLAGS.eval_interval < 0 or FLAGS.save_interval < 0:
        raise ValueError('eval_interval and save_interval cannot be negative.')
    if FLAGS.eval_episodes < 1:
        raise ValueError('eval_episodes must be at least 1.')


def main(_):
    _validate_runtime_flags()
    config = FLAGS.agent
    restore_step = 0
    if FLAGS.restore_path:
        _, restore_step = resolve_checkpoint(
            FLAGS.restore_path,
            FLAGS.restore_step,
        )
        if restore_step >= FLAGS.train_steps:
            raise ValueError('The restored step must be smaller than train_steps.')

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

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
    if FLAGS.restore_path:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_step)

    exp_name = get_exp_name(
        FLAGS.seed,
        env_name=str(config.env_name),
        agent_name=f'pathbridger_{config.endpoint_distribution}',
    )
    run_dir = os.path.abspath(
        os.path.join(FLAGS.save_dir, 'pathbridger', FLAGS.run_group, exp_name)
    )
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    _write_run_config(run_dir, config)

    wandb_run = None
    if FLAGS.use_wandb:
        wandb_run = setup_wandb(
            project='PathBridger',
            group=FLAGS.run_group,
            name=exp_name,
            config={'flags': get_flag_dict(), 'agent': config.to_dict()},
            directory=run_dir,
        )

    start_step = restore_step + 1
    steps = range(start_step, FLAGS.train_steps + 1)
    if FLAGS.use_tqdm:
        steps = tqdm.tqdm(steps, smoothing=0.1, dynamic_ncols=True)

    train_logger = CsvLogger(os.path.join(run_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(run_dir, 'eval.csv'))
    start_time = time.time()
    interval_start = start_time
    prefetch_pool = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix='train-prefetch')
        if FLAGS.async_prefetch
        else None
    )

    def _submit_batch() -> Future:
        return prefetch_pool.submit(dataset.sample, _BATCH_SIZE)

    next_batch_future = _submit_batch() if prefetch_pool is not None else None

    try:
        for step in steps:
            if next_batch_future is None:
                batch = dataset.sample(_BATCH_SIZE)
            else:
                batch = next_batch_future.result()
                next_batch_future = None

            is_final = step == FLAGS.train_steps
            do_save = is_final or (
                FLAGS.save_interval > 0
                and step % FLAGS.save_interval == 0
            )
            # Do not sample past a checkpoint before its NumPy RNG state is
            # serialized. This preserves exact batch-stream resume semantics.
            if prefetch_pool is not None and not do_save:
                next_batch_future = _submit_batch()

            agent, update_info = agent.update(batch)

            if step % FLAGS.log_interval == 0 or is_final:
                train_metrics = _host_metrics(update_info)
                train_metrics['time/interval_seconds'] = time.time() - interval_start
                train_metrics['time/total_seconds'] = time.time() - start_time
                interval_start = time.time()
                train_logger.log(train_metrics, step=step)
                if wandb_run is not None:
                    wandb_run.log({f'training/{key}': value for key, value in train_metrics.items()}, step=step)

            do_eval = is_final or (FLAGS.eval_interval > 0 and step % FLAGS.eval_interval == 0)
            if do_eval:
                eval_info = evaluate(
                    agent,
                    env,
                    task_ids=DEFAULT_TASK_IDS,
                    episodes_per_task=FLAGS.eval_episodes,
                    num_candidates=int(config.eval_num_candidates),
                    temperature=float(config.eval_temperature),
                    seed=FLAGS.seed,
                )
                eval_metrics = {f'evaluation/{key}': value for key, value in eval_info.items()}
                eval_logger.log(eval_metrics, step=step)
                if wandb_run is not None:
                    wandb_run.log(eval_metrics, step=step)

            if do_save:
                save_agent(agent, checkpoint_dir, step)
                if prefetch_pool is not None and not is_final:
                    next_batch_future = _submit_batch()
    finally:
        if prefetch_pool is not None:
            prefetch_pool.shutdown(wait=True)
        train_logger.close()
        eval_logger.close()
        if wandb_run is not None:
            wandb_run.finish()

    print(f'Run saved to {run_dir}')


def run():
    """Run the command-line training entry point."""

    app.run(main)


if __name__ == '__main__':
    run()
