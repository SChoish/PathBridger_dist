"""Evaluate a component-wise action-free offline-to-online checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import flax
import numpy as np
from absl import app, flags

from agents.online_idm import OnlineIDMAgent, PBFOnlineIDMPolicy
from agents.pathbridger import PathBridgerAgent
from agents.registry import create_algorithm
from envs.env_utils import make_env_and_datasets
from utils.af_checkpoints import load_af_checkpoint, restore_af_agent
from utils.af_data import ActionFreeTrajectoryData
from utils.af_evaluation import evaluate_policy
from utils.datasets import observation_state_scale


FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint', '', 'Exact train_af checkpoint path.')
flags.DEFINE_string('dataset_dir', '', 'Optional OGBench dataset/cache directory.')
flags.DEFINE_integer('episodes', 50, 'Episodes per official task.')
flags.DEFINE_integer('seed', 0, 'Evaluation seed.')
flags.DEFINE_string('output_path', '', 'Optional JSON output.')


def main(_):
    if not FLAGS.checkpoint:
        raise ValueError('--checkpoint is required.')
    payload = load_af_checkpoint(FLAGS.checkpoint)
    metadata = payload['metadata']
    env_name = metadata['env_name']
    env, train_data, _ = make_env_and_datasets(
        env_name, dataset_dir=FLAGS.dataset_dir or None
    )
    af_data = ActionFreeTrajectoryData(train_data, env_name=env_name, seed=FLAGS.seed)
    state_scale = observation_state_scale(train_data)
    indices = af_data.episodes.transition_indices
    deltas = af_data.observations[indices + 1] - af_data.observations[indices]
    delta_scale = np.maximum(np.std(deltas, axis=0), 1e-3)
    example = af_data.observations[:2]
    action_dim = int(np.prod(env.action_space.shape))
    name = payload['algorithm']
    if name == 'pbf_online_idm':
        planner_config = payload['planner_config']
        planner = PathBridgerAgent.create(
            FLAGS.seed, example, None, planner_config, state_scale=state_scale
        )
        planner = flax.serialization.from_state_dict(planner, payload['planner'])
        idm = OnlineIDMAgent.create(
            FLAGS.seed + 1, example, action_dim, payload['config']
        )
        idm = restore_af_agent(idm, payload)
        policy = PBFOnlineIDMPolicy(
            planner,
            idm,
            int(planner_config['eval_num_candidates']),
            float(planner_config['eval_temperature']),
            int(payload['config']['execute_horizon']),
        )
        execute_horizon = int(payload['config']['execute_horizon'])
    else:
        template, _ = create_algorithm(
            name,
            seed=FLAGS.seed,
            ex_observations=example,
            action_dim=action_dim,
            state_scale=state_scale,
            delta_scale=delta_scale,
            config=payload['config'],
        )
        policy = restore_af_agent(template, payload)
        execute_horizon = 1
    metrics = evaluate_policy(
        policy,
        env,
        episodes_per_task=FLAGS.episodes,
        seed=FLAGS.seed,
        execute_horizon=execute_horizon,
    )
    result = {
        'algorithm': name,
        'checkpoint_step': int(payload['step']),
        'env_name': env_name,
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
