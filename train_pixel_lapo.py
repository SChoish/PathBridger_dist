"""Train the isolated goal-conditioned pixel LAPO OGBench adaptation."""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import jax
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

from agents.online_idm import parameter_digest
from agents.pixel_lapo import PixelLAPOAgent, get_config
from envs.env_utils import make_pixel_env_and_datasets
from utils.af_checkpoints import save_af_checkpoint
from utils.evaluation import DEFAULT_TASK_IDS, _info_success, _max_episode_steps
from utils.log_utils import CsvLogger, get_exp_name
from utils.pixel_data import ActionFreePixelTrajectoryData, PixelReplayBuffer
from utils.pixel_evaluation import evaluate_pixel_policy


FLAGS = flags.FLAGS
PROTOCOL_VERSION = 'pixel_lapo_o2o_v1'
ALGORITHM = 'gc_pixel_lapo'

flags.DEFINE_string(
    'env_name',
    'visual-antmaze-medium-navigate-v0',
    'Official visual-* OGBench dataset name.',
)
flags.DEFINE_string('dataset_dir', '', 'Optional OGBench visual dataset/cache directory.')
flags.DEFINE_bool(
    'allow_dataset_download',
    False,
    'Allow OGBench to download a missing visual dataset (disabled by default).',
)
flags.DEFINE_integer('seed', 0, 'Pipeline seed.')
flags.DEFINE_string('save_dir', 'exp/', 'Experiment root.')
flags.DEFINE_string('run_group', 'pixel_lapo', 'Pixel experiment group.')
flags.DEFINE_string('protocol_suite', 'adhoc', 'Pixel benchmark suite.')
flags.DEFINE_integer('stage1_steps', -1, 'Latent IDM/world-model steps; -1 uses config.')
flags.DEFINE_integer('stage2_steps', -1, 'Latent-policy BC steps; -1 uses config.')
flags.DEFINE_integer('online_steps', 1_000_000, 'Primitive online environment steps.')
flags.DEFINE_integer('random_steps', 10_000, 'Random decoder-grounding bootstrap steps.')
flags.DEFINE_integer('update_start', 1_000, 'Replay size at first decoder update.')
flags.DEFINE_integer('replay_capacity', 20_000, 'Number of uint8 visual transitions.')
flags.DEFINE_integer('log_interval', 1_000, 'CSV logging interval.')
flags.DEFINE_string(
    'eval_steps',
    '0,10000,25000,50000,100000,250000,500000,1000000',
    'Comma-separated online evaluation/checkpoint steps.',
)
flags.DEFINE_integer('eval_episodes', 10, 'Episodes per task; zero disables evaluation.')
flags.DEFINE_bool('use_tqdm', True, 'Display progress bars.')
config_flags.DEFINE_config_dict('lapo', get_config(), lock_config=False)


def _host_metrics(info):
    return {
        key: float(np.asarray(jax.device_get(value)))
        for key, value in info.items()
        if np.asarray(jax.device_get(value)).size == 1
    }


def _parse_eval_steps(value: str, maximum: int) -> tuple[int, ...]:
    steps = sorted({int(item.strip()) for item in value.split(',') if item.strip()})
    if any(step < 0 for step in steps):
        raise ValueError('eval_steps cannot contain negative values.')
    return tuple(step for step in steps if step <= maximum)


def _stable_hash(payload) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def _frame(value, *, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 3 or result.shape[-1] != 3 or result.dtype != np.uint8:
        raise ValueError(f'{name} must be an HWC uint8 RGB frame, got {result.shape}/{result.dtype}.')
    return result


def main(_):
    if FLAGS.online_steps < 0 or FLAGS.random_steps < 0:
        raise ValueError('online_steps and random_steps must be non-negative.')
    if FLAGS.random_steps > FLAGS.online_steps:
        raise ValueError('random_steps cannot exceed online_steps.')
    if FLAGS.update_start < 1 or FLAGS.replay_capacity < FLAGS.update_start:
        raise ValueError('Require replay_capacity >= update_start >= 1.')
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    task_rng = np.random.default_rng(FLAGS.seed + 17_003)
    exploration_rng = np.random.default_rng(FLAGS.seed + 91_009)

    env, train_dataset, _ = make_pixel_env_and_datasets(
        FLAGS.env_name,
        dataset_dir=FLAGS.dataset_dir or None,
        allow_download=FLAGS.allow_dataset_download,
    )
    pixel_data = ActionFreePixelTrajectoryData(train_dataset, seed=FLAGS.seed)
    action_shape = tuple(int(value) for value in env.action_space.shape)
    action_dim = int(np.prod(action_shape))
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    if not (np.allclose(action_low, -1.0) and np.allclose(action_high, 1.0)):
        raise ValueError('Pixel LAPO currently requires OGBench actions bounded by [-1, 1].')

    config = FLAGS.lapo.to_dict()
    stage1_steps = int(FLAGS.stage1_steps if FLAGS.stage1_steps >= 0 else config['stage1_steps'])
    stage2_steps = int(FLAGS.stage2_steps if FLAGS.stage2_steps >= 0 else config['stage2_steps'])
    batch_size = int(config['offline_batch_size'])
    agent = PixelLAPOAgent.create(
        FLAGS.seed,
        pixel_data.observations[:2],
        action_dim,
        config,
    )

    exp_name = get_exp_name(FLAGS.seed, env_name=FLAGS.env_name, agent_name=ALGORITHM)
    run_dir = Path(FLAGS.save_dir).resolve() / 'pixel_lapo_o2o' / FLAGS.run_group / exp_name
    checkpoint_dir = run_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    stage1_logger = CsvLogger(run_dir / 'offline_stage1.csv')
    iterator = range(1, stage1_steps + 1)
    if FLAGS.use_tqdm and stage1_steps:
        iterator = tqdm.tqdm(iterator, desc='pixel-lapo-stage1', dynamic_ncols=True)
    for step in iterator:
        agent, info = agent.offline_update(pixel_data.sample(batch_size), stage=1)
        if step % FLAGS.log_interval == 0 or step == stage1_steps:
            stage1_logger.log(_host_metrics(info), step)
    stage1_logger.close()

    stage2_logger = CsvLogger(run_dir / 'offline_stage2.csv')
    iterator = range(1, stage2_steps + 1)
    if FLAGS.use_tqdm and stage2_steps:
        iterator = tqdm.tqdm(iterator, desc='pixel-lapo-stage2', dynamic_ncols=True)
    for step in iterator:
        agent, info = agent.offline_update(pixel_data.sample(batch_size), stage=2)
        if step % FLAGS.log_interval == 0 or step == stage2_steps:
            stage2_logger.log(_host_metrics(info), step)
    stage2_logger.close()

    latent_digest = parameter_digest(agent.network.params['modules_latent_model'])
    policy_digest = parameter_digest(agent.network.params['modules_latent_policy'])
    eval_steps = _parse_eval_steps(FLAGS.eval_steps, FLAGS.online_steps)
    protocol_payload = {
        'protocol_version': PROTOCOL_VERSION,
        'suite': str(FLAGS.protocol_suite),
        'run_group': str(FLAGS.run_group),
        'online_steps': int(FLAGS.online_steps),
        'random_steps': int(FLAGS.random_steps),
        'update_start': int(FLAGS.update_start),
        'replay_capacity': int(FLAGS.replay_capacity),
        'eval_steps': list(eval_steps),
        'eval_episodes_per_task': int(FLAGS.eval_episodes),
        'evaluation_task_ids': list(DEFAULT_TASK_IDS),
    }
    protocol_id = f'{PROTOCOL_VERSION}:{_stable_hash(protocol_payload)[:16]}'
    config_payload = {
        **protocol_payload,
        'algorithm': ALGORITHM,
        'env_name': FLAGS.env_name,
        'stage1_steps': stage1_steps,
        'stage2_steps': stage2_steps,
        'config': config,
    }
    run_metadata = {
        'algorithm': ALGORITHM,
        'method_status': 'continuous-control OGBench adaptation of LAPO, not an official reproduction',
        'protocol_version': PROTOCOL_VERSION,
        'protocol_id': protocol_id,
        'protocol_suite': str(FLAGS.protocol_suite),
        'run_group': str(FLAGS.run_group),
        'config_hash': _stable_hash(config_payload),
        'env_name': str(FLAGS.env_name),
        'seed': int(FLAGS.seed),
        'observation_modality': 'rgb_uint8',
        'offline_fields_seen': list(pixel_data.offline_fields_seen),
        'uses_offline_actions': False,
        'offline_modules_trained': ['latent_model', 'latent_policy'],
        'online_modules_updated': ['decoder'],
        'online_data_fields_used': ['observations', 'next_observations', 'actions'],
        'stage1_steps': stage1_steps,
        'stage2_steps': stage2_steps,
        'online_steps': int(FLAGS.online_steps),
        'random_steps': int(FLAGS.random_steps),
        'update_start': int(FLAGS.update_start),
        'replay_capacity': int(FLAGS.replay_capacity),
        'eval_steps': list(eval_steps),
        'eval_episodes_per_task': int(FLAGS.eval_episodes),
        'evaluation_task_ids': list(DEFAULT_TASK_IDS),
        'online_goal_distribution': 'uniform_over_official_task_ids_per_episode',
        'image_shape': list(pixel_data.image_shape),
        'action_dim': action_dim,
        'config': config,
        'latent_model_initial_digest': latent_digest,
        'latent_policy_initial_digest': policy_digest,
        'lapo_paper': 'https://arxiv.org/abs/2312.10812',
        'lapo_repository': 'https://github.com/schmidtdominik/LAPO',
        'lapo_reference_commit': 'c3844f7e8c92e900bf7547a265f14089ac68b121',
    }
    (run_dir / 'metadata.json').write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + '\n'
    )

    replay = PixelReplayBuffer(
        FLAGS.replay_capacity,
        pixel_data.image_shape,
        action_shape,
        seed=FLAGS.seed,
    )
    run_metadata['replay_allocated_bytes'] = replay.allocated_bytes
    eval_env = None
    if FLAGS.eval_episodes > 0:
        import ogbench

        eval_env = ogbench.make_env_and_datasets(FLAGS.env_name, env_only=True)
    online_logger = CsvLogger(run_dir / 'online.csv')
    eval_logger = CsvLogger(run_dir / 'eval.csv')
    max_episode_steps = _max_episode_steps(env)
    timestep = 0
    task_id = int(task_rng.choice(DEFAULT_TASK_IDS))
    observation, reset_info = env.reset(
        seed=FLAGS.seed, options={'task_id': task_id, 'render_goal': False}
    )
    observation = _frame(observation, name='observation')
    goal = _frame(reset_info['goal'], name='goal')
    start_time = time.time()
    last_info = {}

    def assert_offline_frozen():
        if parameter_digest(agent.network.params['modules_latent_model']) != latent_digest:
            raise RuntimeError('Frozen pixel latent model changed during online decoder training.')
        if parameter_digest(agent.network.params['modules_latent_policy']) != policy_digest:
            raise RuntimeError('Frozen pixel latent policy changed during online decoder training.')

    def evaluate_and_save(step):
        assert_offline_frozen()
        checkpoint_metadata = {
            **run_metadata,
            'latent_model_final_digest': parameter_digest(
                agent.network.params['modules_latent_model']
            ),
            'latent_policy_final_digest': parameter_digest(
                agent.network.params['modules_latent_policy']
            ),
            'offline_modules_frozen_verified': True,
        }
        if eval_env is not None:
            metrics = evaluate_pixel_policy(
                agent,
                eval_env,
                episodes_per_task=FLAGS.eval_episodes,
                seed=FLAGS.seed,
            )
            eval_logger.log(metrics, step)
        save_af_checkpoint(
            checkpoint_dir / f'step_{step}.pkl',
            algorithm=ALGORITHM,
            agent=agent,
            step=step,
            config=config,
            metadata=checkpoint_metadata,
        )

    if 0 in eval_steps:
        evaluate_and_save(0)

    iterator = range(1, FLAGS.online_steps + 1)
    if FLAGS.use_tqdm and FLAGS.online_steps:
        iterator = tqdm.tqdm(iterator, desc='pixel-lapo-online-decoder', dynamic_ncols=True)
    for step in iterator:
        if step <= FLAGS.random_steps:
            action = exploration_rng.uniform(action_low, action_high).astype(np.float32)
        else:
            action = np.asarray(
                jax.device_get(
                    agent.sample_actions(
                        observation[None, ...],
                        goal[None, ...],
                        seed=jax.random.PRNGKey(FLAGS.seed * 1_000_003 + step),
                        temperature=1.0,
                    )[0]
                ),
                dtype=np.float32,
            )
        action = np.clip(action, action_low, action_high)
        next_observation, _, terminated, truncated, info = env.step(action)
        next_observation = _frame(next_observation, name='next_observation')
        success = _info_success(info)
        replay.add(
            observation=observation,
            action=action,
            next_observation=next_observation,
            goal=goal,
            reward=0.0 if success else -1.0,
            mask=0.0 if success or bool(terminated) else 1.0,
        )
        observation = next_observation
        timestep += 1

        if replay.size >= FLAGS.update_start:
            agent, last_info = agent.online_update(
                replay.sample(int(config['online_batch_size']))
            )

        if terminated or truncated or timestep >= max_episode_steps:
            timestep = 0
            task_id = int(task_rng.choice(DEFAULT_TASK_IDS))
            observation, reset_info = env.reset(
                options={'task_id': task_id, 'render_goal': False}
            )
            observation = _frame(observation, name='observation')
            goal = _frame(reset_info['goal'], name='goal')

        if step % FLAGS.log_interval == 0 or step == FLAGS.online_steps:
            metrics = _host_metrics(last_info)
            metrics.update(
                replay_size=replay.size,
                replay_allocated_bytes=replay.allocated_bytes,
                elapsed_seconds=time.time() - start_time,
                task_id=task_id,
            )
            online_logger.log(metrics, step)
        if step in eval_steps:
            evaluate_and_save(step)

    assert_offline_frozen()
    run_metadata.update(
        latent_model_final_digest=parameter_digest(agent.network.params['modules_latent_model']),
        latent_policy_final_digest=parameter_digest(agent.network.params['modules_latent_policy']),
        offline_modules_frozen_verified=True,
    )
    (run_dir / 'metadata.json').write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + '\n'
    )
    online_logger.close()
    eval_logger.close()
    print(f'Pixel LAPO run saved to {run_dir}')


def run():
    app.run(main)


if __name__ == '__main__':
    run()
