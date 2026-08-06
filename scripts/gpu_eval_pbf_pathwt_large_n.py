#!/usr/bin/env python3
"""GPU eval for large-N NT cells of one paper_s0_pbf_pathwt checkpoint.

Only evaluates N in {8,16,32} x T in {0.25,0.5,1.0} (9 cells). Small-N /
(T=0,N=1) stay on the CPU watcher. Skips cells that already have JSON so it
can run concurrently with cpu_eval_pbf_pathwt_ckpt.py.

Does not force step DONE; if the full 19-cell grid is already on disk when
this finishes, writes cpu_eval/step_{step}.DONE so the CPU watcher can move on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Device must be fixed before JAX import.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--gpu', type=str, default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
_pre_args, _ = _pre.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(_pre_args.gpu)
os.environ.pop('JAX_PLATFORMS', None)
os.environ.setdefault('MUJOCO_GL', os.environ.get('MUJOCO_GL', 'egl'))

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml_collections import config_dict  # noqa: E402

from agents import PathBridgerAgent  # noqa: E402
from agents.pathbridger import get_config  # noqa: E402
from envs.env_utils import make_env_and_datasets  # noqa: E402
from utils.datasets import PathBridgerDataset  # noqa: E402
from utils.evaluation import DEFAULT_TASK_IDS, evaluate  # noqa: E402
from utils.flax_utils import restore_agent  # noqa: E402

FLOW_NT_GRID = (
    (1, 0.0),
    *[(n, t) for n in (1, 2, 4) for t in (0.25, 0.5, 1.0)],
    *[(n, t) for n in (8, 16, 32) for t in (0.25, 0.5, 1.0)],
)


def _t_tok(t: float) -> str:
    return str(int(t)) if float(t) == int(t) else str(t).replace('.', 'p')


def _load_agent_config(run_dir: Path) -> tuple[config_dict.ConfigDict, dict]:
    flags = json.loads((run_dir / 'flags.json').read_text(encoding='utf-8'))
    saved = dict(flags.get('agent') or {})
    config = get_config()
    config.update(saved)
    if 'prefix_model' not in saved:
        config.prefix_model = 'deterministic'
    # Stochastic bridge eval: sample M=4 prefixes, pick via EMA TransV chain.
    if str(config.get('prefix_model', 'deterministic')) != 'deterministic':
        config.eval_prefix_selection = 'transv_chain'
        config.eval_num_prefix_samples = 4
        config.eval_include_deterministic_prefix = False
    return config, flags


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=str, required=True)
    p.add_argument('--checkpoint-step', type=int, required=True)
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dataset-dir', type=str, default='')
    p.add_argument('--gpu', type=str, default=_pre_args.gpu)
    p.add_argument('--min-n', type=int, default=8)
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    step = int(args.checkpoint_step)
    ckpt_dir = run_dir / 'checkpoints'
    ckpt = ckpt_dir / f'params_{step}.pkl'
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    grid = tuple((n, t) for n, t in FLOW_NT_GRID if int(n) >= int(args.min_n))
    if not grid:
        print(f'nothing to do min_n={args.min_n}')
        return

    eval_dir = run_dir / 'eval_results'
    eval_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        (n, t)
        for n, t in grid
        if not (eval_dir / f'epoch{step}_t{_t_tok(t)}_n{n}.json').is_file()
    ]
    if not pending:
        print(f'skip all large-N present step={step} min_n={args.min_n}')
    else:
        config, flags_payload = _load_agent_config(run_dir)
        env, train_data, _ = make_env_and_datasets(
            str(config.env_name),
            dataset_dir=args.dataset_dir or None,
        )
        dataset = PathBridgerDataset(train_data, config)
        example_batch = dataset.sample(1)
        create_kwargs = {}
        if flags_payload.get('state_scale') is not None:
            create_kwargs['state_scale'] = flags_payload['state_scale']
        agent = PathBridgerAgent.create(
            args.seed,
            example_batch['observations'],
            example_batch['actions'],
            config,
            **create_kwargs,
        )
        agent = restore_agent(agent, str(ckpt_dir), step)

        for n_cand, temp in pending:
            t_tok = _t_tok(temp)
            out = eval_dir / f'epoch{step}_t{t_tok}_n{n_cand}.json'
            if out.is_file():
                print(f'skip existing {out.name}')
                continue
            metrics = evaluate(
                agent,
                env,
                task_ids=DEFAULT_TASK_IDS,
                episodes_per_task=int(args.episodes),
                num_candidates=int(n_cand),
                temperature=float(temp),
                seed=int(args.seed),
            )
            result = {
                'checkpoint_step': step,
                'env_name': str(config.env_name),
                'endpoint_distribution': str(config.endpoint_distribution),
                'eval_num_candidates': int(n_cand),
                'eval_temperature': float(temp),
                'eval_device': 'gpu',
                **{f'evaluation/{k}': float(v) for k, v in metrics.items()},
            }
            tmp = out.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
            tmp.replace(out)
            print(
                f'step={step} n={n_cand} t={temp:g} gpu={args.gpu} '
                f'overall={metrics["overall_success"]:.4f}'
            )

    missing = [
        (n, t)
        for n, t in FLOW_NT_GRID
        if not (eval_dir / f'epoch{step}_t{_t_tok(t)}_n{n}.json').is_file()
    ]
    marker_dir = run_dir / 'cpu_eval'
    marker_dir.mkdir(parents=True, exist_ok=True)
    if not missing:
        done_marker = marker_dir / f'step_{step}.DONE'
        if not done_marker.is_file():
            done_marker.write_text('ok\n', encoding='utf-8')
            print(f'DONE full grid present step={step}')
    else:
        print(f'large-N pass finished; still missing {len(missing)} cells (CPU will fill)')


if __name__ == '__main__':
    main()
