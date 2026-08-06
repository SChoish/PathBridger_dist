"""Train the action-free offline-to-online PBF study and its baselines."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import signal
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

from agents.af_guide import AFGuideAgent
from agents.online_idm import OnlineIDMAgent, PBFOnlineIDMPolicy, get_config as idm_config, parameter_digest
from agents.oso_decqn import OSODecQNAgent, discretize_deltas
from agents.pathbridger import PathBridgerAgent
from agents.registry import (
    ALGORITHMS,
    DEFAULT_OFFLINE_STEPS,
    algorithm_metadata,
    create_algorithm,
)
from envs.env_utils import make_env_and_datasets
from utils.af_checkpoints import (
    load_af_checkpoint,
    restore_af_agent,
    restore_af_planner,
    save_af_checkpoint,
)
from utils.af_data import (
    REPLAY_METRIC_KEYS,
    ActionFreeTrajectoryData,
    OnlineReplayBuffer,
)
from utils.af_evaluation import evaluate_policy
from utils.datasets import PathBridgerDataset, action_free_view, observation_state_scale
from utils.evaluation import DEFAULT_TASK_IDS, _info_success, _max_episode_steps
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name


FLAGS = flags.FLAGS
PROTOCOL_VERSION = 'af_o2o_v2'
_DEFAULT_PBF_CONFIG = str(
    Path(__file__).resolve().parent / 'configs' / 'pbf_af' / 'antmaze_medium.py'
)

flags.DEFINE_enum('algorithm', 'pbf_online_idm', ALGORITHMS, 'Algorithm to train.')
flags.DEFINE_string('env_name', 'antmaze-medium-navigate-v0', 'OGBench dataset name for non-PBF algorithms.')
flags.DEFINE_string('dataset_dir', '', 'Optional OGBench dataset/cache directory.')
flags.DEFINE_integer('seed', 0, 'Pipeline seed.')
flags.DEFINE_string('save_dir', 'exp/', 'Experiment root.')
flags.DEFINE_string('run_group', 'af_main', 'Experiment group.')
flags.DEFINE_string('protocol_suite', 'adhoc', 'Benchmark suite recorded in metadata.')
flags.DEFINE_integer('offline_steps', -1, 'Offline updates; -1 uses the algorithm-faithful default.')
flags.DEFINE_integer('online_steps', 1_000_000, 'Primitive online environment steps.')
flags.DEFINE_integer('random_steps', 10_000, 'Random bootstrap steps included in the online budget.')
flags.DEFINE_integer('update_start', 1_000, 'Replay size at first online update.')
flags.DEFINE_integer('replay_capacity', 1_000_000, 'Online replay capacity.')
flags.DEFINE_integer('log_interval', 1_000, 'CSV logging interval.')
flags.DEFINE_string(
    'eval_steps',
    '0,10000,25000,50000,100000,250000,500000,1000000',
    'Comma-separated online evaluation/checkpoint steps.',
)
flags.DEFINE_integer('eval_episodes', 10, 'Episodes per task at intermediate evaluation; zero disables evaluation.')
flags.DEFINE_bool(
    'save_replay',
    True,
    'Include replay arrays in online checkpoints (needed for exact online resume).',
)
flags.DEFINE_bool('use_tqdm', True, 'Display progress bars.')
flags.DEFINE_string('pbf_restore_path', '', 'Optional pretrained action-free PBF checkpoint.')
flags.DEFINE_integer('pbf_restore_step', 0, 'PBF checkpoint step, or infer from exact filename.')
flags.DEFINE_string(
    'af_restore_path',
    '',
    'Optional AF checkpoint (.pkl) or directory for online/offline resume.',
)
flags.DEFINE_integer(
    'af_restore_step',
    -1,
    'AF checkpoint step; -1 infers from path or latest step_*.pkl in a directory.',
)
flags.DEFINE_integer(
    'pbf_execute_horizon',
    1,
    'Primitive actions decoded per PBF plan; supported values are 1 and 5.',
)
config_flags.DEFINE_config_file(
    'pbf', _DEFAULT_PBF_CONFIG, 'Environment-specific action-free PBF config.', lock_config=False
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


def _resolve_af_restore_path(path: str, step: int) -> tuple[str | None, int | None]:
    if not path:
        return None, None
    candidate = Path(path)
    if candidate.is_dir():
        if step >= 0:
            online = candidate / f'step_{step}.pkl'
            offline = candidate / f'offline_step_{step}.pkl'
            if online.is_file():
                return str(online), step
            if offline.is_file():
                return str(offline), step
            raise FileNotFoundError(
                f'No step_{step}.pkl or offline_step_{step}.pkl under {candidate}'
            )
        matches = sorted(
            list(candidate.glob('step_*.pkl'))
            + list(candidate.glob('offline_step_*.pkl')),
            key=lambda item: item.stat().st_mtime,
        )
        if not matches:
            raise FileNotFoundError(f'No step_*.pkl under {candidate}')
        ckpt = matches[-1]
        stem = ckpt.stem
        step = int(stem.rsplit('_', 1)[-1])
        return str(ckpt), step
    if candidate.is_file():
        if step < 0:
            stem = candidate.stem
            if stem.startswith('step_') or stem.startswith('offline_step_'):
                step = int(stem.rsplit('_', 1)[-1])
            else:
                step = 0
        return str(candidate), step
    raise FileNotFoundError(path)


def _write_emergency_marker(run_dir: str, step: int, signum: int | None, phase: str) -> None:
    marker = Path(run_dir) / f'EMERGENCY_SAVE_{phase}_step{step}'
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
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _delta_scale(data: ActionFreeTrajectoryData) -> np.ndarray:
    indices = data.episodes.transition_indices
    deltas = data.observations[indices + 1] - data.observations[indices]
    return np.maximum(np.std(deltas.astype(np.float64), axis=0), 1e-3).astype(np.float32)


def _offline_batch(name, data, full_action_data, batch_size, config, delta_scale):
    if name == 'gc_af_guide':
        return data.sample_sequences(
            batch_size, context_length=int(config['context_length'])
        )
    batch = data.sample(
        batch_size,
        subgoal_steps=int(config.get('subgoal_steps', 10)),
    )
    if name == 'gc_oso_decqn_factorized':
        batch['delta_bins'] = discretize_deltas(
            batch['next_observations'] - batch['observations'],
            delta_scale,
            float(config['discretization_threshold']),
        )
    if name == 'gc_sac_50_50':
        if full_action_data is None:
            raise ValueError(
                'GC-SAC-50/50 requires its explicitly scoped full-action dataset.'
            )
        batch['actions'] = np.asarray(
            full_action_data['actions'][batch['indices']], dtype=np.float32
        )
    return batch


def _concat_batches(left, right):
    common = set(left) & set(right)
    required = {'observations', 'next_observations', 'actions', 'goals', 'rewards', 'masks'}
    return {
        key: np.concatenate([np.asarray(left[key]), np.asarray(right[key])], axis=0)
        for key in sorted(common & required)
    }


def _history_arrays(history, context_length, state_dim):
    history = [np.asarray(state, dtype=np.float32) for state in history[-context_length:]]
    output = np.empty((context_length, state_dim), dtype=np.float32)
    mask = np.zeros((context_length,), dtype=np.float32)
    output[-len(history) :] = history
    output[: context_length - len(history)] = history[0]
    mask[-len(history) :] = 1.0
    return output[None, ...], mask[None, ...]


def _make_eval_env(env_name):
    import ogbench

    return ogbench.make_env_and_datasets(env_name, env_only=True)


def main(_):
    name = str(FLAGS.algorithm)
    if FLAGS.online_steps < 0 or FLAGS.random_steps < 0:
        raise ValueError('online_steps and random_steps must be non-negative.')
    if FLAGS.random_steps > FLAGS.online_steps:
        raise ValueError('random_steps cannot exceed online_steps.')
    if FLAGS.pbf_execute_horizon not in (1, 5):
        raise ValueError('pbf_execute_horizon must be 1 or 5.')
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    task_rng = np.random.default_rng(FLAGS.seed + 17_003)
    exploration_rng = np.random.default_rng(FLAGS.seed + 91_009)

    pbf_config = FLAGS.pbf
    env_name = str(pbf_config.env_name) if name == 'pbf_online_idm' else FLAGS.env_name
    env, train_data, _ = make_env_and_datasets(
        env_name, dataset_dir=FLAGS.dataset_dir or None
    )
    action_dim = int(np.prod(env.action_space.shape))
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    af_data = ActionFreeTrajectoryData(
        train_data, env_name=env_name, seed=FLAGS.seed
    )
    state_scale = observation_state_scale(train_data)
    delta_scale = _delta_scale(af_data)
    example_observations = af_data.observations[:2]
    metadata = algorithm_metadata(name)
    full_action_data = train_data if name == 'gc_sac_50_50' else None

    exp_name = get_exp_name(FLAGS.seed, env_name=env_name, agent_name=name)
    run_dir = os.path.abspath(
        os.path.join(FLAGS.save_dir, 'pbf_af_o2o', FLAGS.run_group, exp_name)
    )
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    stop_requested = {'flag': False, 'signum': None}
    _install_soft_stop(stop_requested)
    af_restore_file, af_restore_step = _resolve_af_restore_path(
        FLAGS.af_restore_path, int(FLAGS.af_restore_step)
    )
    af_payload = None
    skip_offline = False
    online_start_step = 1
    offline_start_step = 1
    if af_restore_file is not None:
        af_payload = load_af_checkpoint(af_restore_file)
        if str(af_payload.get('algorithm')) != name:
            raise ValueError(
                f'AF restore algorithm mismatch: ckpt={af_payload.get("algorithm")} '
                f'vs flags={name}'
            )
        phase = str(af_payload.get('phase', 'online'))
        if phase == 'online':
            skip_offline = True
            online_start_step = int(af_payload['step']) + 1
            print(
                f'[resume] online from step={online_start_step} '
                f'ckpt={af_restore_file}',
                flush=True,
            )
        elif phase == 'offline':
            offline_start_step = int(af_payload['step']) + 1
            print(
                f'[resume] offline from step={offline_start_step} '
                f'ckpt={af_restore_file}',
                flush=True,
            )
        else:
            raise ValueError(f'Unsupported AF checkpoint phase={phase!r}')

    planner = None
    planner_config = None
    planner_digest = None
    if name == 'pbf_online_idm':
        planner_config = pbf_config.to_dict()
        if not bool(planner_config.get('offline_action_free', False)):
            raise ValueError('The PBF config must set offline_action_free=True.')
        state_only = action_free_view(train_data)
        pbf_dataset = PathBridgerDataset(
            state_only, pbf_config, require_actions=False
        )
        planner = PathBridgerAgent.create(
            FLAGS.seed,
            pbf_dataset.sample(1)['observations'],
            None,
            pbf_config,
            state_scale=state_scale,
        )
        if FLAGS.pbf_restore_path:
            planner = restore_agent(
                planner, FLAGS.pbf_restore_path, FLAGS.pbf_restore_step
            )
        resolved_offline_steps = (
            int(FLAGS.offline_steps)
            if FLAGS.offline_steps >= 0
            else (0 if FLAGS.pbf_restore_path or skip_offline else DEFAULT_OFFLINE_STEPS[name])
        )
        if skip_offline:
            resolved_offline_steps = 0
        offline_logger = CsvLogger(os.path.join(run_dir, 'offline.csv'))
        offline_start = offline_start_step
        if FLAGS.pbf_restore_path and FLAGS.pbf_restore_step > 0 and not skip_offline:
            offline_start = max(offline_start, int(FLAGS.pbf_restore_step) + 1)
        iterator = range(offline_start, resolved_offline_steps + 1)
        if FLAGS.use_tqdm and resolved_offline_steps:
            iterator = tqdm.tqdm(iterator, desc='offline-pbf', dynamic_ncols=True)
        for step in iterator:
            do_log = step % FLAGS.log_interval == 0 or step == resolved_offline_steps
            planner, info = planner.update(
                pbf_dataset.sample(1024), full_metrics=do_log
            )
            if do_log:
                offline_logger.log(_host_metrics(info), step)
            emergency = bool(stop_requested['flag'])
            if step % 100_000 == 0 or step == resolved_offline_steps or emergency:
                save_agent(planner, os.path.join(run_dir, 'pbf_checkpoints'), step)
                if emergency:
                    _write_emergency_marker(
                        run_dir, step, stop_requested['signum'], 'offline'
                    )
                    offline_logger.close()
                    print(f'Run soft-stopped (offline) at {run_dir}', flush=True)
                    return
        offline_logger.close()
        controller_config = idm_config().to_dict()
        controller_config['execute_horizon'] = int(FLAGS.pbf_execute_horizon)
        agent = OnlineIDMAgent.create(
            FLAGS.seed + 1, example_observations, action_dim, controller_config
        )
        resolved_config = controller_config
        planner_digest = parameter_digest(planner.network.params)
    else:
        agent, resolved_config = create_algorithm(
            name,
            seed=FLAGS.seed,
            ex_observations=example_observations,
            action_dim=action_dim,
            state_scale=state_scale,
            delta_scale=delta_scale,
        )
        resolved_offline_steps = (
            int(FLAGS.offline_steps)
            if FLAGS.offline_steps >= 0
            else (0 if skip_offline else DEFAULT_OFFLINE_STEPS[name])
        )
        if skip_offline:
            resolved_offline_steps = 0
        if (
            af_payload is not None
            and str(af_payload.get('phase')) == 'offline'
            and not skip_offline
        ):
            agent = restore_af_agent(agent, af_payload)
        offline_logger = CsvLogger(os.path.join(run_dir, 'offline.csv'))
        batch_size = int(
            resolved_config.get(
                'offline_batch_size', resolved_config.get('batch_size', 1024)
            )
        )
        iterator = range(offline_start_step, resolved_offline_steps + 1)
        if FLAGS.use_tqdm and resolved_offline_steps:
            iterator = tqdm.tqdm(iterator, desc=f'offline-{name}', dynamic_ncols=True)
        for step in iterator:
            batch = _offline_batch(
                name,
                af_data,
                full_action_data,
                batch_size,
                resolved_config,
                delta_scale,
            )
            agent, info = agent.offline_update(batch)
            if step % FLAGS.log_interval == 0 or step == resolved_offline_steps:
                offline_logger.log(_host_metrics(info), step)
            if stop_requested['flag']:
                save_af_checkpoint(
                    os.path.join(checkpoint_dir, f'offline_step_{step}.pkl'),
                    algorithm=name,
                    agent=agent,
                    step=step,
                    config=resolved_config,
                    metadata={'phase': 'offline'},
                    phase='offline',
                )
                _write_emergency_marker(
                    run_dir, step, stop_requested['signum'], 'offline'
                )
                offline_logger.close()
                print(f'Run soft-stopped (offline) at {run_dir}', flush=True)
                return
        offline_logger.close()

    eval_step_list = list(_parse_eval_steps(FLAGS.eval_steps, FLAGS.online_steps))
    protocol_payload = {
        'protocol_version': PROTOCOL_VERSION,
        'suite': str(FLAGS.protocol_suite),
        'run_group': str(FLAGS.run_group),
        'online_steps': int(FLAGS.online_steps),
        'random_steps': int(FLAGS.random_steps),
        'update_start': int(FLAGS.update_start),
        'replay_capacity': int(FLAGS.replay_capacity),
        'eval_steps': eval_step_list,
        'eval_episodes_per_task': int(FLAGS.eval_episodes),
        'evaluation_task_ids': list(DEFAULT_TASK_IDS),
    }
    protocol_id = f'{PROTOCOL_VERSION}:{_stable_hash(protocol_payload)[:16]}'
    config_payload = {
        **protocol_payload,
        'algorithm': name,
        'env_name': env_name,
        'offline_steps': int(resolved_offline_steps),
        'config': resolved_config,
        'pbf_config': planner_config,
    }
    run_metadata = metadata.to_dict()
    run_metadata.update(
        protocol_version=PROTOCOL_VERSION,
        protocol_id=protocol_id,
        protocol_suite=str(FLAGS.protocol_suite),
        run_group=str(FLAGS.run_group),
        config_hash=_stable_hash(config_payload),
        env_name=env_name,
        seed=int(FLAGS.seed),
        offline_steps=int(resolved_offline_steps),
        online_steps=int(FLAGS.online_steps),
        random_steps=int(FLAGS.random_steps),
        update_start=int(FLAGS.update_start),
        replay_capacity=int(FLAGS.replay_capacity),
        eval_steps=eval_step_list,
        eval_episodes_per_task=int(FLAGS.eval_episodes),
        evaluation_task_ids=list(DEFAULT_TASK_IDS),
        online_goal_distribution='uniform_over_official_task_ids_per_episode',
        state_scale=np.asarray(state_scale).tolist(),
        delta_scale=np.asarray(delta_scale).tolist(),
        config=resolved_config,
        pbf_config=planner_config,
        planner_initial_digest=planner_digest,
    )
    Path(run_dir, 'metadata.json').write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + '\n'
    )

    replay = OnlineReplayBuffer(
        FLAGS.replay_capacity,
        tuple(env.observation_space.shape),
        tuple(env.action_space.shape),
        seed=FLAGS.seed,
    )
    eval_env = _make_eval_env(env_name) if FLAGS.eval_episodes > 0 else None
    eval_steps = _parse_eval_steps(FLAGS.eval_steps, FLAGS.online_steps)
    online_logger = CsvLogger(os.path.join(run_dir, 'online.csv'))
    eval_logger = CsvLogger(os.path.join(run_dir, 'eval.csv'))
    max_episode_steps = _max_episode_steps(env)
    episode_id = 0
    timestep = 0
    task_id = int(task_rng.choice(DEFAULT_TASK_IDS))
    observation, reset_info = env.reset(
        seed=FLAGS.seed, options={'task_id': task_id, 'render_goal': False}
    )
    goal = np.asarray(reset_info['goal'], dtype=np.float32)
    observation = np.asarray(observation, dtype=np.float32)
    history = [observation.copy()]
    pending_actions: list[np.ndarray] = []
    pending_desired_next: list[np.ndarray] = []
    start_time = time.time()
    last_info = {}

    if af_payload is not None and str(af_payload.get('phase', 'online')) == 'online':
        agent = restore_af_agent(agent, af_payload)
        if planner is not None:
            planner = restore_af_planner(planner, af_payload)
            planner_digest = parameter_digest(planner.network.params)
            run_metadata['planner_initial_digest'] = planner_digest
        if af_payload.get('replay') is not None:
            replay.load_state_dict(af_payload['replay'])
        elif FLAGS.save_replay:
            print(
                '[resume] warning: online ckpt has no replay; buffer starts empty',
                flush=True,
            )
        runtime = af_payload.get('runtime') or {}
        if runtime:
            # OGBench cannot rebind mid-episode physics; keep agent/replay/RNGs
            # and open a fresh episode at the restored RNG state.
            if runtime.get('task_rng_state') is not None:
                task_rng.bit_generator.state = runtime['task_rng_state']
            if runtime.get('exploration_rng_state') is not None:
                exploration_rng.bit_generator.state = runtime['exploration_rng_state']
            episode_id = int(runtime.get('episode_id', episode_id)) + 1
            timestep = 0
            pending_actions = []
            pending_desired_next = []
            task_id = int(task_rng.choice(DEFAULT_TASK_IDS))
            observation, reset_info = env.reset(
                options={'task_id': task_id, 'render_goal': False}
            )
            observation = np.asarray(observation, dtype=np.float32)
            goal = np.asarray(reset_info['goal'], dtype=np.float32)
            history = [observation.copy()]
            print(
                f'[resume] fresh episode_id={episode_id} task_id={task_id} '
                '(mid-episode env state not restored)',
                flush=True,
            )

    def current_policy():
        if name == 'pbf_online_idm':
            return PBFOnlineIDMPolicy(
                planner=planner,
                idm=agent,
                num_candidates=int(pbf_config.eval_num_candidates),
                endpoint_temperature=float(pbf_config.eval_temperature),
                execute_horizon=int(resolved_config['execute_horizon']),
            )
        return agent

    def _runtime_state(step: int) -> dict:
        return {
            'online_step': int(step),
            'episode_id': int(episode_id),
            'timestep': int(timestep),
            'task_id': int(task_id),
            'observation': np.asarray(observation, dtype=np.float32),
            'goal': np.asarray(goal, dtype=np.float32),
            'history': [np.asarray(item, dtype=np.float32) for item in history],
            'pending_actions': [
                np.asarray(item, dtype=np.float32) for item in pending_actions
            ],
            'pending_desired_next': [
                np.asarray(item, dtype=np.float32) for item in pending_desired_next
            ],
            'task_rng_state': task_rng.bit_generator.state,
            'exploration_rng_state': exploration_rng.bit_generator.state,
        }

    def evaluate_and_save(step, *, force_replay: bool = False):
        if eval_env is not None and not force_replay:
            metrics = evaluate_policy(
                current_policy(),
                eval_env,
                episodes_per_task=FLAGS.eval_episodes,
                seed=FLAGS.seed,
                execute_horizon=(
                    int(resolved_config['execute_horizon'])
                    if name == 'pbf_online_idm'
                    else 1
                ),
            )
            eval_logger.log(metrics, step)
        elif eval_env is not None and force_replay:
            # Soft-stop: skip expensive eval; still persist weights/replay/runtime.
            pass
        save_af_checkpoint(
            os.path.join(checkpoint_dir, f'step_{step}.pkl'),
            algorithm=name,
            agent=agent,
            step=step,
            config=resolved_config,
            metadata=run_metadata,
            planner=planner,
            planner_config=planner_config,
            replay_state=(
                replay.state_dict()
                if (FLAGS.save_replay or force_replay)
                else None
            ),
            runtime=_runtime_state(step),
            phase='online',
        )

    if 0 in eval_steps and online_start_step <= 1 and af_payload is None:
        evaluate_and_save(0)

    iterator = range(online_start_step, FLAGS.online_steps + 1)
    if FLAGS.use_tqdm and FLAGS.online_steps:
        iterator = tqdm.tqdm(iterator, desc=f'online-{name}', dynamic_ncols=True)
    for step in iterator:
        desired_next = observation.copy()
        desired_next_valid = False
        if step <= FLAGS.random_steps:
            action = exploration_rng.uniform(action_low, action_high).astype(np.float32)
        elif name == 'pbf_online_idm':
            policy = current_policy()
            if not pending_actions:
                action_seed = jax.random.PRNGKey(FLAGS.seed * 1_000_003 + step)
                prefix = policy.desired_prefix(
                    jnp.asarray(observation[None, :]),
                    jnp.asarray(goal[None, :]),
                    seed=action_seed,
                )
                decoded = np.asarray(
                    jax.device_get(policy.decode_prefix(prefix)), dtype=np.float32
                )
                if decoded.ndim == 2:
                    decoded = decoded[:, None, :]
                horizon = int(resolved_config['execute_horizon'])
                pending_actions = [item.copy() for item in decoded[0, :horizon]]
                prefix_host = np.asarray(jax.device_get(prefix[0]), dtype=np.float32)
                pending_desired_next = [
                    item.copy() for item in prefix_host[1 : horizon + 1]
                ]
            action = pending_actions.pop(0)
            desired_next = pending_desired_next.pop(0)
            desired_next_valid = True
            decay = min(
                max(step - FLAGS.random_steps, 0)
                / max(int(resolved_config['exploration_decay_steps']), 1),
                1.0,
            )
            noise_std = (
                float(resolved_config['exploration_std_initial']) * (1.0 - decay)
                + float(resolved_config['exploration_std_final']) * decay
            )
            action = action + exploration_rng.normal(
                0.0, noise_std, size=action.shape
            )
        else:
            if isinstance(agent, OSODecQNAgent):
                agent = dataclasses.replace(agent, online_steps=step - 1)
            if isinstance(agent, AFGuideAgent):
                histories, history_masks = _history_arrays(
                    history,
                    int(resolved_config['context_length']),
                    observation.shape[-1],
                )
                desired_next = np.asarray(
                    agent.plan_next(
                        jnp.asarray(histories),
                        jnp.asarray(goal[None, :]),
                        jnp.asarray([max_episode_steps - timestep], jnp.float32),
                        jnp.asarray(history_masks),
                    )[0]
                )
                desired_next_valid = True
            action = np.asarray(
                jax.device_get(
                    agent.sample_actions(
                        jnp.asarray(observation[None, :]),
                        jnp.asarray(goal[None, :]),
                        seed=jax.random.PRNGKey(FLAGS.seed * 1_000_003 + step),
                        temperature=1.0,
                    )[0]
                ),
                dtype=np.float32,
            )
        action = np.clip(action, action_low, action_high).astype(np.float32)
        next_observation, _, terminated, truncated, info = env.step(action)
        next_observation = np.asarray(next_observation, dtype=np.float32)
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
            desired_next=desired_next,
            desired_next_valid=desired_next_valid,
        )
        history.append(next_observation.copy())
        observation = next_observation
        timestep += 1

        if replay.size >= FLAGS.update_start:
            batch_size = int(
                resolved_config.get(
                    'online_batch_size', resolved_config.get('batch_size', 256)
                )
            )
            online_batch = replay.sample(
                batch_size,
                her_probability=(
                    0.0
                    if name == 'pbf_online_idm'
                    else float(resolved_config.get('her_probability', 0.8))
                ),
                goal_projector=af_data.project_goals,
            )
            replay_metrics = {
                key: online_batch.pop(key) for key in REPLAY_METRIC_KEYS
            }
            if name == 'pbf_online_idm':
                idm_batch = {
                    key: online_batch[key]
                    for key in ('observations', 'next_observations', 'actions')
                }
                agent, last_info = agent.online_update(idm_batch)
            elif name == 'gc_mscp_style':
                offline_batch = _offline_batch(
                    name,
                    af_data,
                    None,
                    batch_size,
                    resolved_config,
                    delta_scale,
                )
                agent, last_info = agent.mixed_online_update(
                    online_batch,
                    offline_batch,
                    tune_planners=bool(resolved_config['tune_planners_online']),
                )
            elif name == 'gc_oso_decqn_factorized':
                offline_batch = _offline_batch(
                    name,
                    af_data,
                    None,
                    batch_size,
                    resolved_config,
                    delta_scale,
                )
                agent, last_info = agent.online_update(online_batch, offline_batch)
            elif name == 'gc_sac_50_50':
                half_size = max(batch_size // 2, 1)
                offline_batch = _offline_batch(
                    name,
                    af_data,
                    full_action_data,
                    half_size,
                    resolved_config,
                    delta_scale,
                )
                half_online = {
                    key: value[:half_size]
                    for key, value in online_batch.items()
                    if np.asarray(value).ndim > 0
                    and np.asarray(value).shape[0] == batch_size
                }
                agent, last_info = agent.online_update(
                    _concat_batches(half_online, offline_batch)
                )
            else:
                agent, last_info = agent.online_update(online_batch)
            last_info = {**last_info, **replay_metrics}

        if terminated or truncated or timestep >= max_episode_steps:
            episode_id += 1
            timestep = 0
            task_id = int(task_rng.choice(DEFAULT_TASK_IDS))
            observation, reset_info = env.reset(
                options={'task_id': task_id, 'render_goal': False}
            )
            observation = np.asarray(observation, dtype=np.float32)
            goal = np.asarray(reset_info['goal'], dtype=np.float32)
            history = [observation.copy()]
            pending_actions.clear()
            pending_desired_next.clear()

        if step % FLAGS.log_interval == 0 or step == FLAGS.online_steps:
            metrics = _host_metrics(last_info)
            metrics.update(
                replay_size=replay.size,
                elapsed_seconds=time.time() - start_time,
                task_id=task_id,
            )
            online_logger.log(metrics, step)
        emergency = bool(stop_requested['flag'])
        if step in eval_steps:
            evaluate_and_save(step)
        elif emergency:
            evaluate_and_save(step, force_replay=True)
        if emergency:
            _write_emergency_marker(
                run_dir, step, stop_requested['signum'], 'online'
            )
            online_logger.close()
            eval_logger.close()
            print(f'Run soft-stopped (online) at {run_dir}', flush=True)
            return

    online_logger.close()
    eval_logger.close()
    if planner is not None:
        planner_final_digest = parameter_digest(planner.network.params)
        if planner_final_digest != planner_digest:
            raise RuntimeError('Frozen PBF parameters changed during online IDM training.')
        run_metadata.update(
            planner_final_digest=planner_final_digest,
            planner_frozen_verified=True,
        )
        Path(run_dir, 'metadata.json').write_text(
            json.dumps(run_metadata, indent=2, sort_keys=True) + '\n'
        )
    print(f'Run saved to {run_dir}')


def run():
    app.run(main)


if __name__ == '__main__':
    run()
