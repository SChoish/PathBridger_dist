"""Unified action-free offline-to-online visual OGBench benchmark runner."""

from __future__ import annotations

import hashlib
import json
import random
import signal
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
    pixel_method_scope,
)
from agents.pixel_trl_critic_locks import apply_pixel_pbf_locks
from envs.env_utils import make_pixel_env_and_datasets
from utils.af_checkpoints import (
    load_af_checkpoint,
    restore_af_agent,
    save_af_checkpoint,
)
from utils.evaluation import DEFAULT_TASK_IDS, _info_success, _max_episode_steps
from utils.log_utils import CsvLogger, get_exp_name
from utils.pixel_data import (
    ActionFreePixelTrajectoryData,
    PixelTrajectoryData,
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
flags.DEFINE_string(
    'restore_path',
    '',
    'Optional checkpoint (.pkl) or checkpoints/ directory for resume.',
)
flags.DEFINE_integer(
    'restore_step',
    -1,
    'Checkpoint step; -1 infers from path or latest matching file.',
)
flags.DEFINE_bool(
    'resume_in_place',
    True,
    'When restoring, append into the same run_dir inferred from the checkpoint.',
)
flags.DEFINE_integer(
    'resume_interval',
    50_000,
    'Periodic resume_step_*.pkl interval online; 0 disables periodic resume saves.',
)
flags.DEFINE_integer('resume_keep', 2, 'Number of resume_step_*.pkl files to retain.')
flags.DEFINE_bool(
    'save_replay',
    False,
    'Include replay in periodic resume checkpoints. Soft-stop always saves replay.',
)


def _install_soft_stop(stop_requested: dict) -> None:
    def _request_stop(signum, _frame):
        if stop_requested['flag']:
            return
        stop_requested['flag'] = True
        stop_requested['signum'] = int(signum)
        print(
            f'[signal] signum={signum} — will emergency-save after current step',
            flush=True,
        )

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    try:
        signal.signal(signal.SIGHUP, _request_stop)
    except (ValueError, OSError):
        pass


def _resolve_restore_path(path: str, step: int) -> tuple[str | None, int | None]:
    if not path:
        return None, None
    candidate = Path(path)
    if candidate.is_dir():
        if step >= 0:
            for name in (
                f'resume_step_{step}.pkl',
                f'step_{step}.pkl',
                f'offline_step_{step}.pkl',
            ):
                target = candidate / name
                if target.is_file():
                    return str(target), step
            raise FileNotFoundError(
                f'No resume/step/offline checkpoint for step {step} under {candidate}'
            )
        matches = sorted(
            list(candidate.glob('resume_step_*.pkl'))
            + list(candidate.glob('step_*.pkl'))
            + list(candidate.glob('offline_step_*.pkl')),
            key=lambda item: item.stat().st_mtime,
        )
        if not matches:
            raise FileNotFoundError(f'No step_*.pkl under {candidate}')
        ckpt = matches[-1]
        return str(ckpt), int(ckpt.stem.rsplit('_', 1)[-1])
    if candidate.is_file():
        if step < 0:
            stem = candidate.stem
            if stem.startswith(('step_', 'resume_step_', 'offline_step_')):
                step = int(stem.rsplit('_', 1)[-1])
            else:
                step = 0
        return str(candidate), step
    raise FileNotFoundError(path)


def _checkpoint_run_dir(path: str | Path) -> Path:
    checkpoint = Path(path).resolve()
    if checkpoint.parent.name != 'checkpoints':
        raise ValueError(
            'In-place resume requires a checkpoint under a run '
            f'checkpoints directory, got {checkpoint}.'
        )
    return checkpoint.parent.parent


def _rotate_resume_checkpoints(checkpoint_dir: Path, keep: int) -> None:
    if keep < 1:
        raise ValueError('resume_keep must be positive.')
    paths = sorted(
        checkpoint_dir.glob('resume_step_*.pkl'),
        key=lambda path: int(path.stem.rsplit('_', 1)[-1]),
    )
    for path in paths[:-keep]:
        path.unlink()


def _write_emergency_marker(run_dir: Path, step: int, signum: int | None, phase: str) -> None:
    marker = run_dir / f'EMERGENCY_SAVE_{phase}_step{step}'
    marker.write_text(f'signal={signum} phase={phase} step={step}\n', encoding='utf-8')
    print(
        f'[signal] emergency_save_done phase={phase} step={step} signum={signum}',
        flush=True,
    )


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
        raise ValueError(
            f'{name} must be an HWC uint8 RGB frame, got {result.shape}/{result.dtype}.'
        )
    return result


def _write_metadata(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def _offline_module_names(algorithm: str) -> tuple[str, ...]:
    return pixel_method_scope(algorithm)['offline_trainable_modules']


def _frozen_module_names(algorithm: str) -> tuple[str, ...]:
    return pixel_method_scope(algorithm)['online_frozen_modules']


def _module_digest(agent, name: str) -> str:
    return parameter_digest(agent.network.params[f'modules_{name}'])


def _sample_offline_batch(pixel_data, batch_size: int, config: dict):
    kwargs = {'path_horizon': int(config.get('path_horizon', 5))}
    if 'endpoint_horizon' in config:
        kwargs['endpoint_horizon'] = int(config['endpoint_horizon'])
    if 'value_distance_weight_power' in config or 'discount' in config:
        kwargs.update(
            discount=float(config.get('discount', 0.99)),
            value_geom_sample=bool(config.get('value_geom_sample', True)),
            value_p_curgoal=float(config.get('value_p_curgoal', 0.0)),
            value_p_trajgoal=float(config.get('value_p_trajgoal', 1.0)),
            value_p_randomgoal=float(config.get('value_p_randomgoal', 0.0)),
        )
    return pixel_data.sample(batch_size, **kwargs)


def _make_env_and_data(algorithm: str):
    metadata = pixel_algorithm_metadata(algorithm)
    if metadata.offline_fields_seen:
        full_offline = algorithm == 'pixel_pbf'
        env, train_dataset, _ = make_pixel_env_and_datasets(
            FLAGS.env_name,
            dataset_dir=FLAGS.dataset_dir or None,
            allow_download=FLAGS.allow_dataset_download,
            action_free=not full_offline,
        )
        data_cls = PixelTrajectoryData if full_offline else ActionFreePixelTrajectoryData
        data = data_cls(
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
    if FLAGS.resume_keep < 1:
        raise ValueError('resume_keep must be positive.')

    stop_requested = {'flag': False, 'signum': None}
    _install_soft_stop(stop_requested)

    restore_file, restore_step = _resolve_restore_path(
        FLAGS.restore_path, int(FLAGS.restore_step)
    )
    restore_payload = None
    if restore_file is not None:
        restore_payload = load_af_checkpoint(restore_file)
        if str(restore_payload.get('algorithm')) != str(FLAGS.algorithm):
            raise ValueError(
                f'Restore algorithm mismatch: ckpt={restore_payload.get("algorithm")} '
                f'vs flags={FLAGS.algorithm}'
            )
        print(
            f'[resume] loaded {restore_file} phase={restore_payload.get("phase")} '
            f'step={restore_payload.get("step")}',
            flush=True,
        )

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    task_rng = np.random.default_rng(FLAGS.seed + 17_003)
    exploration_rng = np.random.default_rng(FLAGS.seed + 91_009)
    algorithm = str(FLAGS.algorithm)
    algorithm_metadata = pixel_algorithm_metadata(algorithm)
    method_scope = pixel_method_scope(algorithm)
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
    if algorithm in ('pixel_pbf', 'pixel_pathbridger_online_idm'):
        overrides = apply_pixel_pbf_locks(FLAGS.env_name, overrides)
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
    resume_in_place = bool(restore_file and FLAGS.resume_in_place)
    if resume_in_place:
        run_dir = _checkpoint_run_dir(restore_file)
    else:
        run_dir = (
            Path(FLAGS.save_dir).resolve()
            / 'pixel_o2o'
            / FLAGS.run_group
            / exp_name
        )
    checkpoint_dir = run_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    restore_phase = str(restore_payload.get('phase')) if restore_payload else ''
    restore_runtime = (restore_payload or {}).get('runtime') or {}
    offline_start = 1
    lapo_stage_start = 1
    skip_offline = restore_phase == 'online'
    if restore_payload is not None and restore_phase == 'offline':
        agent = restore_af_agent(agent, restore_payload)
        offline_start = int(restore_payload['step']) + 1
        lapo_stage_start = int(restore_runtime.get('lapo_stage', 1))
        if algorithm == 'gc_pixel_lapo_decoder' and lapo_stage_start == 2:
            # stage2 steps are 1-indexed within the stage.
            offline_start = int(restore_runtime.get('stage_step', restore_payload['step'])) + 1

    def save_offline_checkpoint(step: int, *, lapo_stage: int | None = None, stage_step: int | None = None):
        runtime = {
            'offline_step': int(step),
            'lapo_stage': int(lapo_stage) if lapo_stage is not None else None,
            'stage_step': int(stage_step) if stage_step is not None else int(step),
        }
        save_af_checkpoint(
            checkpoint_dir / f'offline_step_{step}.pkl',
            algorithm=algorithm,
            agent=agent,
            step=step,
            config=config,
            metadata={'phase': 'offline', 'algorithm': algorithm},
            runtime=runtime,
            phase='offline',
        )

    if not skip_offline and algorithm == 'gc_pixel_lapo_decoder':
        absolute_step = 0
        for stage, steps in ((1, stage1_steps), (2, stage2_steps)):
            if stage < lapo_stage_start:
                absolute_step += steps
                continue
            stage_begin = offline_start if stage == lapo_stage_start else 1
            logger = CsvLogger(
                run_dir / f'offline_stage{stage}.csv',
                resume=resume_in_place and stage_begin > 1,
            )
            iterator = range(stage_begin, steps + 1)
            if FLAGS.use_tqdm and steps:
                iterator = tqdm.tqdm(
                    iterator, desc=f'{algorithm}-stage{stage}', dynamic_ncols=True
                )
            for stage_step in iterator:
                absolute_step = (stage1_steps if stage == 2 else 0) + stage_step
                agent, info = agent.offline_update(
                    _sample_offline_batch(pixel_data, batch_size, config),
                    stage=stage,
                )
                if stage_step % FLAGS.log_interval == 0 or stage_step == steps:
                    logger.log(_host_metrics(info), absolute_step)
                if stop_requested['flag']:
                    save_offline_checkpoint(
                        absolute_step, lapo_stage=stage, stage_step=stage_step
                    )
                    _write_emergency_marker(
                        run_dir, absolute_step, stop_requested['signum'], 'offline'
                    )
                    logger.close()
                    print(f'Run soft-stopped (offline) at {run_dir}', flush=True)
                    return
            logger.close()
            offline_start = 1
    elif not skip_offline and offline_steps:
        logger = CsvLogger(
            run_dir / 'offline.csv',
            resume=resume_in_place and offline_start > 1,
        )
        iterator = range(offline_start, offline_steps + 1)
        if FLAGS.use_tqdm:
            iterator = tqdm.tqdm(iterator, desc=f'{algorithm}-offline', dynamic_ncols=True)
        for step in iterator:
            agent, info = agent.offline_update(
                _sample_offline_batch(pixel_data, batch_size, config)
            )
            if step % FLAGS.log_interval == 0 or step == offline_steps:
                logger.log(_host_metrics(info), step)
            if stop_requested['flag']:
                save_offline_checkpoint(step)
                _write_emergency_marker(
                    run_dir, step, stop_requested['signum'], 'offline'
                )
                logger.close()
                print(f'Run soft-stopped (offline) at {run_dir}', flush=True)
                return
        logger.close()
    elif skip_offline and restore_payload is not None:
        agent = restore_af_agent(agent, restore_payload)

    frozen_initial = {
        name: _module_digest(agent, name)
        for name in _frozen_module_names(algorithm)
    }
    if restore_payload is not None and restore_phase == 'online':
        parent_meta = restore_payload.get('metadata') or {}
        parent_digests = parent_meta.get('frozen_module_initial_digests')
        if parent_digests:
            frozen_initial = {key: str(value) for key, value in parent_digests.items()}

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
        'online_modules_trained': list(
            method_scope['online_trainable_modules']
        ),
        'online_modules_frozen': list(method_scope['online_frozen_modules']),
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
        'dataset_dir': str(FLAGS.dataset_dir or ''),
        'trl_critic_lock': {
            'value_distance_weight_power': float(
                config.get('value_distance_weight_power', 0.0)
            ),
            'discount': float(config.get('discount', 0.99)),
            'expectile': float(config.get('expectile', 0.7)),
            'value_geom_sample': bool(config.get('value_geom_sample', True)),
            'value_p_curgoal': float(config.get('value_p_curgoal', 0.0)),
            'value_p_trajgoal': float(config.get('value_p_trajgoal', 1.0)),
            'value_p_randomgoal': float(config.get('value_p_randomgoal', 0.0)),
            'value_hidden_dims': list(config['value_hidden_dims']),
            'value_layer_norm': bool(config['value_layer_norm']),
            'value_learning_rate': float(config['value_learning_rate']),
            'value_tau': float(config['value_tau']),
        }
        if algorithm in ('pixel_pbf', 'pixel_pathbridger_online_idm')
        else None,
        'pixel_pbf_tune': {
            'encoder': str(config['encoder']),
            'feature_dim': int(config['feature_dim']),
            'gap': float(config['endpoint_value_scale']),
            'num_candidates': int(config['eval_num_candidates']),
            'endpoint_temperature': float(config['eval_temperature']),
            'endpoint_horizon': int(config['endpoint_horizon']),
            'path_horizon': int(config['path_horizon']),
            'endpoint_flow_steps': int(config['endpoint_flow_steps']),
        }
        if algorithm in ('pixel_pbf', 'pixel_pathbridger_online_idm')
        else None,
    }
    if restore_file is not None:
        resume_events = list((restore_payload.get('metadata') or {}).get('resume_events', []))
        resume_events.append(
            {
                'checkpoint': str(Path(restore_file).resolve()),
                'resume_step': int(restore_payload['step']),
                'phase': restore_phase,
                'in_place': resume_in_place,
                'timestamp': int(time.time()),
            }
        )
        run_metadata.update(
            resume_checkpoint=str(Path(restore_file).resolve()),
            resume_step=int(restore_payload['step']),
            resume_in_place=resume_in_place,
            resume_events=resume_events,
        )

    replay = PixelReplayBuffer(
        FLAGS.replay_capacity,
        tuple(int(value) for value in env.observation_space.shape),
        action_shape,
        seed=FLAGS.seed,
        frame_stack=FLAGS.frame_stack,
    )
    online_start_step = 1
    if restore_payload is not None and restore_phase == 'online':
        if restore_payload.get('replay') is not None:
            replay.load_state_dict(restore_payload['replay'])
        else:
            print(
                '[resume] warning: online ckpt has no replay; buffer starts empty',
                flush=True,
            )
        online_start_step = int(restore_payload['step']) + 1
        if restore_runtime.get('task_rng_state') is not None:
            task_rng.bit_generator.state = restore_runtime['task_rng_state']
        if restore_runtime.get('exploration_rng_state') is not None:
            exploration_rng.bit_generator.state = restore_runtime[
                'exploration_rng_state'
            ]

    run_metadata['replay_allocated_bytes'] = replay.allocated_bytes
    _write_metadata(run_dir / 'metadata.json', run_metadata)

    eval_env = None
    if FLAGS.eval_episodes > 0:
        import ogbench

        eval_env = ogbench.make_env_and_datasets(FLAGS.env_name, env_only=True)
    online_logger = CsvLogger(
        run_dir / 'online.csv',
        resume=resume_in_place and online_start_step > 1,
    )
    eval_logger = CsvLogger(
        run_dir / 'eval.csv',
        resume=resume_in_place and online_start_step > 1,
    )
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
    if restore_payload is not None and restore_phase == 'online' and restore_runtime:
        episode_id = int(restore_runtime.get('episode_id', episode_id)) + 1
        timestep = 0
        task_id = int(task_rng.choice(DEFAULT_TASK_IDS))
        observation, reset_info = env.reset(
            options={'task_id': task_id, 'render_goal': False}
        )
        observation = _frame(observation, name='observation')
        goal = _frame(reset_info['goal'], name='goal')
        frame_history = [observation.copy()]
        print(
            f'[resume] fresh episode_id={episode_id} task_id={task_id} '
            '(mid-episode env state not restored)',
            flush=True,
        )
    start_time = time.time()
    last_info = {}
    pending_action_chunk: list[np.ndarray] = []

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

    def _runtime_state(step: int) -> dict:
        return {
            'online_step': int(step),
            'episode_id': int(episode_id),
            'timestep': int(timestep),
            'task_id': int(task_id),
            'task_rng_state': task_rng.bit_generator.state,
            'exploration_rng_state': exploration_rng.bit_generator.state,
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
            phase='online',
        )

    def save_resume_checkpoint(step: int, *, force_replay: bool = False) -> None:
        include_replay = bool(force_replay or FLAGS.save_replay)
        save_af_checkpoint(
            checkpoint_dir / f'resume_step_{step}.pkl',
            algorithm=algorithm,
            agent=agent,
            step=step,
            config=config,
            metadata=checkpoint_metadata(),
            replay_state=replay.state_dict() if include_replay else None,
            runtime=_runtime_state(step),
            phase='online',
        )
        _rotate_resume_checkpoints(checkpoint_dir, int(FLAGS.resume_keep))

    if 0 in eval_steps and online_start_step <= 1:
        evaluate_and_save(0)

    if online_start_step > FLAGS.online_steps:
        print(
            f'[resume] already past online_steps={FLAGS.online_steps}; nothing to do',
            flush=True,
        )
        online_logger.close()
        eval_logger.close()
        return

    iterator = range(online_start_step, FLAGS.online_steps + 1)
    if FLAGS.use_tqdm and FLAGS.online_steps:
        iterator = tqdm.tqdm(iterator, desc=f'{algorithm}-online', dynamic_ncols=True)
    for step in iterator:
        policy_observation = stack_pixel_history(frame_history, FLAGS.frame_stack)
        policy_goal = repeat_pixel_frame(goal, FLAGS.frame_stack)
        if step <= FLAGS.random_steps:
            pending_action_chunk.clear()
            action = exploration_rng.uniform(action_low, action_high).astype(np.float32)
        elif algorithm == 'pixel_pathbridger_online_idm':
            if not pending_action_chunk:
                planned = np.asarray(
                    jax.device_get(
                        agent.sample_action_chunks(
                            policy_observation[None, ...],
                            policy_goal[None, ...],
                            seed=jax.random.PRNGKey(
                                FLAGS.seed * 1_000_003 + step
                            ),
                            action_temperature=1.0,
                        )[0]
                    ),
                    dtype=np.float32,
                )
                pending_action_chunk.extend(planned)
            action = pending_action_chunk.pop(0)
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
            pending_action_chunk.clear()
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

        emergency = bool(stop_requested['flag'])
        periodic_resume = bool(
            FLAGS.resume_interval > 0
            and (
                step % int(FLAGS.resume_interval) == 0
                or step == FLAGS.online_steps
            )
            and (FLAGS.save_replay or emergency)
        )
        # Soft-stop always writes a replay-bearing resume ckpt.
        if emergency or (periodic_resume and FLAGS.save_replay):
            save_resume_checkpoint(step, force_replay=emergency)
        elif (
            FLAGS.resume_interval > 0
            and step % int(FLAGS.resume_interval) == 0
            and not FLAGS.save_replay
        ):
            # Lightweight agent-only resume marker when replay dump is disabled.
            save_resume_checkpoint(step, force_replay=False)
        if emergency:
            _write_emergency_marker(
                run_dir, step, stop_requested['signum'], 'online'
            )
            online_logger.close()
            eval_logger.close()
            print(f'Run soft-stopped (online) at {run_dir}', flush=True)
            return

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
