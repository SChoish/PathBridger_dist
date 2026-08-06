"""Unified action-free offline-to-online visual OGBench benchmark runner."""

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

from agents.online_idm import parameter_digest
from agents.pixel_registry import (
    PIXEL_ALGORITHMS,
    create_pixel_algorithm,
    pixel_algorithm_metadata,
)
from envs.env_utils import make_pixel_env_and_datasets
from utils.af_checkpoints import save_af_checkpoint
from utils.evaluation import DEFAULT_TASK_IDS, _info_success, _max_episode_steps
from utils.log_utils import CsvLogger, get_exp_name
from utils.pixel_data import (
    ActionFreePixelTrajectoryData,
    PixelReplayBuffer,
    repeat_pixel_frame,
    stack_pixel_history,
)
from utils.pixel_evaluation import evaluate_pixel_policy


FLAGS = flags.FLAGS
PROTOCOL_VERSION = 'pixel_o2o_v3'

flags.DEFINE_enum(
    'algorithm',
    'pixel_pathbridger_online_idm',
    PIXEL_ALGORITHMS,
    'Visual algorithm. Adaptation status is recorded in metadata.json.',
)
flags.DEFINE_string(
    'env_name',
    'visual-antmaze-medium-navigate-v0',
    'Official visual-* OGBench environment/dataset name.',
)
flags.DEFINE_string('dataset_dir', '', 'Optional OGBench visual dataset/cache directory.')
flags.DEFINE_bool(
    'allow_dataset_download',
    False,
    'Allow OGBench to download missing visual data. Disabled by default.',
)
flags.DEFINE_integer('seed', 0, 'Pipeline seed.')
flags.DEFINE_string('save_dir', 'exp/', 'Experiment root.')
flags.DEFINE_string('run_group', 'pixel_benchmark', 'Visual experiment group.')
flags.DEFINE_string('protocol_suite', 'adhoc', 'Visual benchmark suite.')
flags.DEFINE_integer(
    'offline_steps',
    -1,
    'VIP/APV action-free pretraining steps; -1 uses the algorithm config.',
)
flags.DEFINE_integer(
    'lapo_stage1_steps',
    -1,
    'LAPO latent-model steps; -1 uses the algorithm config.',
)
flags.DEFINE_integer(
    'lapo_stage2_steps',
    -1,
    'LAPO latent-policy steps; -1 uses the algorithm config.',
)
flags.DEFINE_string(
    'config_json',
    '',
    'Optional JSON object of flat algorithm-config overrides.',
)
flags.DEFINE_integer('online_steps', 1_000_000, 'Primitive online environment steps.')
flags.DEFINE_integer('random_steps', 10_000, 'Uniform-random bootstrap steps.')
flags.DEFINE_integer('update_start', 1_000, 'Replay size at the first online update.')
flags.DEFINE_integer('replay_capacity', 50_000, 'Number of indexed visual transitions.')
flags.DEFINE_integer('frame_stack', 3, 'Consecutive RGB frames per policy observation.')
flags.DEFINE_float('her_probability', 0.8, 'Future-image HER relabel probability.')
flags.DEFINE_integer('log_interval', 1_000, 'CSV logging interval.')
flags.DEFINE_string(
    'eval_steps',
    '0,10000,25000,50000,100000,250000,500000,1000000',
    'Comma-separated online evaluation/checkpoint steps.',
)
flags.DEFINE_integer('eval_episodes', 10, 'Episodes per task; zero disables evaluation.')
flags.DEFINE_bool('use_tqdm', True, 'Display progress bars.')


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


def _write_metadata(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def _offline_module_names(algorithm: str) -> tuple[str, ...]:
    if algorithm == 'pixel_pathbridger_online_idm':
        return ('encoder', 'bridge', 'world_decoder')
    if algorithm == 'gc_pixel_lapo_decoder':
        return ('latent_model', 'latent_policy')
    if algorithm.startswith('vip_'):
        return ('encoder',)
    if algorithm == 'gc_pixel_apv_style_drq':
        return ('encoder', 'video_predictor', 'world_decoder')
    return ()


def _frozen_module_names(algorithm: str) -> tuple[str, ...]:
    if algorithm == 'pixel_pathbridger_online_idm':
        return ('encoder', 'target_encoder', 'bridge', 'world_decoder')
    if algorithm == 'gc_pixel_lapo_decoder':
        return ('latent_model', 'latent_policy')
    if algorithm == 'vip_style_frozen_gc_drqv2':
        return ('encoder', 'target_encoder')
    return ()


def _module_digest(agent, name: str) -> str:
    return parameter_digest(agent.network.params[f'modules_{name}'])


def _make_env_and_data(algorithm: str):
    metadata = pixel_algorithm_metadata(algorithm)
    if metadata.offline_fields_seen:
        env, train_dataset, _ = make_pixel_env_and_datasets(
            FLAGS.env_name,
            dataset_dir=FLAGS.dataset_dir or None,
            allow_download=FLAGS.allow_dataset_download,
        )
        data = ActionFreePixelTrajectoryData(
            train_dataset,
            seed=FLAGS.seed,
            frame_stack=FLAGS.frame_stack,
        )
        return env, data, data.example_images

    import ogbench

    env = ogbench.make_env_and_datasets(FLAGS.env_name, env_only=True)
    observation, info = env.reset(
        seed=FLAGS.seed,
        options={'task_id': int(DEFAULT_TASK_IDS[0]), 'render_goal': False},
    )
    observation = _frame(observation, name='observation')
    goal = _frame(info['goal'], name='goal')
    return env, None, np.stack(
        [
            repeat_pixel_frame(observation, FLAGS.frame_stack),
            repeat_pixel_frame(goal, FLAGS.frame_stack),
        ]
    )


def main(_):
    if FLAGS.online_steps < 0 or FLAGS.random_steps < 0:
        raise ValueError('online_steps and random_steps must be non-negative.')
    if FLAGS.random_steps > FLAGS.online_steps:
        raise ValueError('random_steps cannot exceed online_steps.')
    if FLAGS.update_start < 1 or FLAGS.replay_capacity < FLAGS.update_start:
        raise ValueError('Require replay_capacity >= update_start >= 1.')
    if FLAGS.eval_episodes < 0:
        raise ValueError('eval_episodes cannot be negative.')
    if FLAGS.frame_stack < 1:
        raise ValueError('frame_stack must be positive.')
    if not 0.0 <= FLAGS.her_probability <= 1.0:
        raise ValueError('her_probability must lie in [0, 1].')

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    task_rng = np.random.default_rng(FLAGS.seed + 17_003)
    exploration_rng = np.random.default_rng(FLAGS.seed + 91_009)
    algorithm = str(FLAGS.algorithm)
    algorithm_metadata = pixel_algorithm_metadata(algorithm)
    env, pixel_data, example_images = _make_env_and_data(algorithm)

    action_shape = tuple(int(value) for value in env.action_space.shape)
    action_dim = int(np.prod(action_shape))
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    if not (np.allclose(action_low, -1.0) and np.allclose(action_high, 1.0)):
        raise ValueError('Visual benchmarks require OGBench actions bounded by [-1, 1].')

    overrides = json.loads(FLAGS.config_json) if FLAGS.config_json else {}
    if not isinstance(overrides, dict):
        raise ValueError('config_json must decode to a JSON object.')
    overrides.setdefault('frame_stack', int(FLAGS.frame_stack))
    agent, config = create_pixel_algorithm(
        algorithm,
        seed=FLAGS.seed,
        example_images=example_images,
        action_dim=action_dim,
        config=overrides,
    )
    batch_size = int(config['offline_batch_size'])
    offline_steps = int(
        FLAGS.offline_steps if FLAGS.offline_steps >= 0 else config.get('offline_steps', 0)
    )
    stage1_steps = int(
        FLAGS.lapo_stage1_steps
        if FLAGS.lapo_stage1_steps >= 0
        else config.get('stage1_steps', 0)
    )
    stage2_steps = int(
        FLAGS.lapo_stage2_steps
        if FLAGS.lapo_stage2_steps >= 0
        else config.get('stage2_steps', 0)
    )
    if min(offline_steps, stage1_steps, stage2_steps) < 0:
        raise ValueError('Offline training steps cannot be negative.')
    if algorithm != 'gc_pixel_lapo_decoder' and (stage1_steps or stage2_steps):
        stage1_steps = stage2_steps = 0
    if algorithm == 'gc_pixel_lapo_decoder':
        offline_steps = stage1_steps + stage2_steps
    if pixel_data is None and offline_steps:
        raise ValueError(f'{algorithm} is online-only and cannot use offline_steps.')

    exp_name = get_exp_name(FLAGS.seed, env_name=FLAGS.env_name, agent_name=algorithm)
    run_dir = (
        Path(FLAGS.save_dir).resolve()
        / 'pixel_o2o'
        / FLAGS.run_group
        / exp_name
    )
    checkpoint_dir = run_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if algorithm == 'gc_pixel_lapo_decoder':
        for stage, steps in ((1, stage1_steps), (2, stage2_steps)):
            logger = CsvLogger(run_dir / f'offline_stage{stage}.csv')
            iterator = range(1, steps + 1)
            if FLAGS.use_tqdm and steps:
                iterator = tqdm.tqdm(
                    iterator, desc=f'{algorithm}-stage{stage}', dynamic_ncols=True
                )
            for step in iterator:
                agent, info = agent.offline_update(
                    pixel_data.sample(
                        batch_size,
                        path_horizon=int(config.get('path_horizon', 5)),
                    ),
                    stage=stage,
                )
                if step % FLAGS.log_interval == 0 or step == steps:
                    logger.log(_host_metrics(info), step)
            logger.close()
    elif offline_steps:
        logger = CsvLogger(run_dir / 'offline.csv')
        iterator = range(1, offline_steps + 1)
        if FLAGS.use_tqdm:
            iterator = tqdm.tqdm(iterator, desc=f'{algorithm}-offline', dynamic_ncols=True)
        for step in iterator:
            agent, info = agent.offline_update(
                pixel_data.sample(
                    batch_size,
                    path_horizon=int(config.get('path_horizon', 5)),
                )
            )
            if step % FLAGS.log_interval == 0 or step == offline_steps:
                logger.log(_host_metrics(info), step)
        logger.close()

    frozen_initial = {
        name: _module_digest(agent, name)
        for name in _frozen_module_names(algorithm)
    }
    eval_steps = _parse_eval_steps(FLAGS.eval_steps, FLAGS.online_steps)
    image_shape = tuple(int(value) for value in example_images.shape[1:])
    protocol_payload = {
        'protocol_version': PROTOCOL_VERSION,
        'suite': str(FLAGS.protocol_suite),
        'run_group': str(FLAGS.run_group),
        'online_steps': int(FLAGS.online_steps),
        'random_steps': int(FLAGS.random_steps),
        'update_start': int(FLAGS.update_start),
        'replay_capacity': int(FLAGS.replay_capacity),
        'frame_stack': int(FLAGS.frame_stack),
        'her_probability': float(FLAGS.her_probability),
        'eval_steps': list(eval_steps),
        'eval_episodes_per_task': int(FLAGS.eval_episodes),
        'evaluation_task_ids': list(DEFAULT_TASK_IDS),
    }
    protocol_id = f'{PROTOCOL_VERSION}:{_stable_hash(protocol_payload)[:16]}'
    config_payload = {
        **protocol_payload,
        'algorithm': algorithm,
        'env_name': FLAGS.env_name,
        'offline_steps': offline_steps,
        'lapo_stage1_steps': stage1_steps,
        'lapo_stage2_steps': stage2_steps,
        'config': config,
    }
    run_metadata = {
        **algorithm_metadata.to_dict(),
        'protocol_version': PROTOCOL_VERSION,
        'protocol_id': protocol_id,
        'protocol_suite': str(FLAGS.protocol_suite),
        'run_group': str(FLAGS.run_group),
        'config_hash': _stable_hash(config_payload),
        'env_name': str(FLAGS.env_name),
        'seed': int(FLAGS.seed),
        'observation_modality': 'rgb_uint8',
        'offline_fields_seen': (
            [] if pixel_data is None else list(pixel_data.offline_fields_seen)
        ),
        'offline_modules_trained': list(_offline_module_names(algorithm)),
        'online_data_fields_used': [
            'observations',
            'next_observations',
            'goals',
            'actions',
            'rewards',
            'masks',
        ],
        'offline_steps': offline_steps,
        'lapo_stage1_steps': stage1_steps,
        'lapo_stage2_steps': stage2_steps,
        'online_steps': int(FLAGS.online_steps),
        'random_steps': int(FLAGS.random_steps),
        'update_start': int(FLAGS.update_start),
        'replay_capacity': int(FLAGS.replay_capacity),
        'frame_stack': int(FLAGS.frame_stack),
        'her_probability': float(FLAGS.her_probability),
        'eval_steps': list(eval_steps),
        'eval_episodes_per_task': int(FLAGS.eval_episodes),
        'evaluation_task_ids': list(DEFAULT_TASK_IDS),
        'online_goal_distribution': 'uniform_over_official_task_ids_per_episode',
        'image_shape': list(image_shape),
        'action_dim': action_dim,
        'config': config,
        'frozen_module_initial_digests': frozen_initial,
        'run_completed': False,
    }

    replay = PixelReplayBuffer(
        FLAGS.replay_capacity,
        tuple(int(value) for value in env.observation_space.shape),
        action_shape,
        seed=FLAGS.seed,
        frame_stack=FLAGS.frame_stack,
    )
    run_metadata['replay_allocated_bytes'] = replay.allocated_bytes
    _write_metadata(run_dir / 'metadata.json', run_metadata)

    eval_env = None
    if FLAGS.eval_episodes > 0:
        import ogbench

        eval_env = ogbench.make_env_and_datasets(FLAGS.env_name, env_only=True)
    online_logger = CsvLogger(run_dir / 'online.csv')
    eval_logger = CsvLogger(run_dir / 'eval.csv')
    max_episode_steps = _max_episode_steps(env)
    episode_id = 0
    timestep = 0
    task_id = int(task_rng.choice(DEFAULT_TASK_IDS))
    observation, reset_info = env.reset(
        seed=FLAGS.seed, options={'task_id': task_id, 'render_goal': False}
    )
    observation = _frame(observation, name='observation')
    goal = _frame(reset_info['goal'], name='goal')
    frame_history = [observation.copy()]
    start_time = time.time()
    last_info = {}

    def assert_frozen_modules() -> None:
        for name, expected in frozen_initial.items():
            actual = _module_digest(agent, name)
            if actual != expected:
                raise RuntimeError(f'Frozen pixel module {name!r} changed online.')

    def checkpoint_metadata() -> dict:
        assert_frozen_modules()
        return {
            **run_metadata,
            'frozen_module_final_digests': {
                name: _module_digest(agent, name) for name in frozen_initial
            },
            'offline_modules_frozen_verified': True,
        }

    def evaluate_and_save(step: int) -> None:
        metadata = checkpoint_metadata()
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
            algorithm=algorithm,
            agent=agent,
            step=step,
            config=config,
            metadata=metadata,
        )

    if 0 in eval_steps:
        evaluate_and_save(0)

    iterator = range(1, FLAGS.online_steps + 1)
    if FLAGS.use_tqdm and FLAGS.online_steps:
        iterator = tqdm.tqdm(iterator, desc=f'{algorithm}-online', dynamic_ncols=True)
    for step in iterator:
        policy_observation = stack_pixel_history(frame_history, FLAGS.frame_stack)
        policy_goal = repeat_pixel_frame(goal, FLAGS.frame_stack)
        if step <= FLAGS.random_steps:
            action = exploration_rng.uniform(action_low, action_high).astype(np.float32)
        else:
            action = np.asarray(
                jax.device_get(
                    agent.sample_actions(
                        policy_observation[None, ...],
                        policy_goal[None, ...],
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
            episode_id=episode_id,
            timestep=timestep,
        )
        frame_history.append(next_observation.copy())
        observation = next_observation
        timestep += 1

        if replay.size >= FLAGS.update_start:
            online_batch = replay.sample(
                int(config['online_batch_size']),
                her_probability=float(FLAGS.her_probability),
            )
            replay_info = {
                key: online_batch.pop(key)
                for key in tuple(online_batch)
                if key.startswith('replay/')
            }
            agent, last_info = agent.online_update(online_batch)
            last_info = {**last_info, **replay_info}

        if terminated or truncated or timestep >= max_episode_steps:
            episode_id += 1
            timestep = 0
            task_id = int(task_rng.choice(DEFAULT_TASK_IDS))
            observation, reset_info = env.reset(
                options={'task_id': task_id, 'render_goal': False}
            )
            observation = _frame(observation, name='observation')
            goal = _frame(reset_info['goal'], name='goal')
            frame_history = [observation.copy()]

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

    if FLAGS.online_steps not in eval_steps:
        evaluate_and_save(FLAGS.online_steps)
    final_metadata = checkpoint_metadata()
    run_metadata.update(
        frozen_module_final_digests=final_metadata['frozen_module_final_digests'],
        offline_modules_frozen_verified=True,
        run_completed=True,
    )
    _write_metadata(run_dir / 'metadata.json', run_metadata)
    online_logger.close()
    eval_logger.close()
    print(f'Pixel benchmark run saved to {run_dir}')


def run():
    app.run(main)


if __name__ == '__main__':
    run()
