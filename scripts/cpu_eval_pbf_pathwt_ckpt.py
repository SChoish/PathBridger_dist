#!/usr/bin/env python3
"""CPU NT eval for one pathbridger checkpoint.

Default ``--grid=best`` evaluates only the paper Best-(N,T) cell for the env
(mean @800/900/1M reporting). ``--grid=full`` keeps the legacy 19-cell NT sweep:
  (N=1, T=0) + N in {1,2,4,8,16,32} x T in {0.25,0.5,1.0}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
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

# Paper / provenance PBF Best-(N,T) per env (seed-mean table).
BEST_NT_BY_ENV = {
    'cube-single-play-v0': (1, 0.0),
    'cube-double-play-v0': (8, 0.25),
    'cube-triple-play-v0': (1, 0.0),
    'puzzle-3x3-play-v0': (32, 1.0),
    'puzzle-4x4-play-v0': (16, 1.0),
    'antmaze-medium-navigate-v0': (8, 0.25),
    'antmaze-large-navigate-v0': (32, 0.5),
    'scene-play-v0': (8, 0.5),
}


def _t_tok(t: float) -> str:
    return str(int(t)) if float(t) == int(t) else str(t).replace('.', 'p')


def _best_nt(env_name: str) -> tuple[int, float]:
    key = str(env_name).strip()
    if key not in BEST_NT_BY_ENV:
        raise KeyError(f'no Best-(N,T) mapping for env_name={env_name!r}')
    return BEST_NT_BY_ENV[key]


def _load_agent_config(run_dir: Path) -> tuple[config_dict.ConfigDict, dict]:
    """Merge run flags over get_config() so older wave1 flags.json (no prefix_*) work.

    Wave1 paper_s0_pbf_pathwt flags omit prefix_model and related keys; defaults
    match training intent (deterministic prefix). Saved overrides (beta, horizon,
    env_name, …) always win.
    """
    flags = json.loads((run_dir / 'flags.json').read_text(encoding='utf-8'))
    saved = dict(flags.get('agent') or {})
    config = get_config()
    config.update(saved)
    if 'prefix_model' not in saved:
        config.prefix_model = 'deterministic'
    if str(config.get('prefix_model', 'deterministic')) != 'deterministic':
        config.eval_prefix_selection = 'transv_chain'
        if 'eval_num_prefix_samples' not in saved:
            config.eval_num_prefix_samples = 4
        config.eval_include_deterministic_prefix = False
    return config, flags


def _cell_path(
    eval_dir: Path,
    step: int,
    n_cand: int,
    temp: float,
    m: int | None,
    prefix_temp: float | None = None,
) -> Path:
    """Legacy M=4 cells omit ``_m``; M sweeps use ``_m{M}``; M×prefix-T uses ``_m{M}_pt{T}``."""
    base = f'epoch{step}_t{_t_tok(temp)}_n{n_cand}'
    if m is None:
        return eval_dir / f'{base}.json'
    if prefix_temp is None:
        return eval_dir / f'{base}_m{int(m)}.json'
    return eval_dir / f'{base}_m{int(m)}_pt{_t_tok(float(prefix_temp))}.json'


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=str, required=True)
    p.add_argument('--checkpoint-step', type=int, required=True)
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dataset-dir', type=str, default='')
    p.add_argument(
        '--grid',
        choices=('best', 'full'),
        default=os.environ.get('EVAL_GRID', 'best'),
        help='best=paper Best-(N,T) only; full=19-cell NT sweep',
    )
    p.add_argument(
        '--prefix-samples',
        type=int,
        default=None,
        help=(
            'Override eval_num_prefix_samples (bridge/prefix candidates M). '
            'When set, writes epoch*_m{M}.json and step_{step}_m{M}.DONE so '
            'legacy M=4 cells are not overwritten.'
        ),
    )
    p.add_argument(
        '--prefix-temperature',
        type=float,
        default=None,
        help=(
            'Override eval_prefix_temperature (bridge flow sampling T). '
            'When set with --prefix-samples, writes *_m{M}_pt{T}.json.'
        ),
    )
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    step = int(args.checkpoint_step)
    ckpt_dir = run_dir / 'checkpoints'
    ckpt = ckpt_dir / f'params_{step}.pkl'
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    m_override = None if args.prefix_samples is None else int(args.prefix_samples)
    if m_override is not None and m_override < 1:
        raise ValueError('--prefix-samples must be >= 1')
    pt_override = (
        None if args.prefix_temperature is None else float(args.prefix_temperature)
    )
    if pt_override is not None and pt_override < 0.0:
        raise ValueError('--prefix-temperature must be >= 0')
    if pt_override is not None and m_override is None:
        raise ValueError('--prefix-temperature requires --prefix-samples')

    marker_dir = run_dir / 'cpu_eval'
    marker_dir.mkdir(parents=True, exist_ok=True)
    if m_override is None:
        done_marker = marker_dir / f'step_{step}.DONE'
    elif pt_override is None:
        done_marker = marker_dir / f'step_{step}_m{m_override}.DONE'
    else:
        done_marker = marker_dir / f'step_{step}_m{m_override}_pt{_t_tok(pt_override)}.DONE'
    if done_marker.is_file():
        print(f'skip already done {done_marker}')
        return

    config, flags_payload = _load_agent_config(run_dir)
    if m_override is not None:
        if str(config.get('prefix_model', 'deterministic')) == 'deterministic':
            raise ValueError('--prefix-samples requires a stochastic prefix_model')
        config.eval_prefix_selection = 'transv_chain'
        config.eval_num_prefix_samples = m_override
        config.eval_include_deterministic_prefix = False
        if pt_override is not None:
            config.eval_prefix_temperature = pt_override
    if args.grid == 'best':
        grid = (_best_nt(str(config.env_name)),)
    else:
        grid = FLOW_NT_GRID

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

    eval_dir = run_dir / 'eval_results'
    eval_dir.mkdir(parents=True, exist_ok=True)
    prefix_m = int(config.get('eval_num_prefix_samples', 1))
    prefix_temp = float(config.get('eval_prefix_temperature', 1.0))
    overalls = []
    for n_cand, temp in grid:
        out = _cell_path(
            eval_dir, step, int(n_cand), float(temp), m_override, pt_override,
        )
        if out.is_file():
            cell = json.loads(out.read_text(encoding='utf-8'))
            overalls.append(float(cell.get('evaluation/overall_success', 0.0)))
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
            'prefix_model': str(config.prefix_model),
            'eval_prefix_selection': str(config.eval_prefix_selection),
            'eval_num_prefix_samples': prefix_m,
            'eval_prefix_temperature': prefix_temp,
            'eval_num_candidates': int(n_cand),
            'eval_temperature': float(temp),
            'eval_grid': args.grid,
            'eval_episodes': int(args.episodes),
            **{f'evaluation/{k}': float(v) for k, v in metrics.items()},
        }
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        overalls.append(float(metrics['overall_success']))
        print(
            f'step={step} n={n_cand} t={temp:g} m={prefix_m} pt={prefix_temp:g} '
            f'overall={metrics["overall_success"]:.4f}'
        )

    missing = [
        (n, t)
        for n, t in grid
        if not _cell_path(
            eval_dir, step, int(n), float(t), m_override, pt_override,
        ).is_file()
    ]
    if missing:
        raise RuntimeError(f'eval grid incomplete for step={step}: {missing}')
    summary = {
        'checkpoint_step': step,
        'eval_grid': args.grid,
        'eval_num_prefix_samples': prefix_m,
        'eval_prefix_temperature': prefix_temp,
        'eval_episodes': int(args.episodes),
        'nt_grid': [{'n': n, 't': t} for n, t in grid],
        'overall_success_mean': float(sum(overalls) / len(overalls)),
        'overall_success_max': float(max(overalls)),
    }
    if m_override is None:
        summary_name = f'step_{step}_nt_summary.json'
    elif pt_override is None:
        summary_name = f'step_{step}_m{m_override}_nt_summary.json'
    else:
        summary_name = f'step_{step}_m{m_override}_pt{_t_tok(pt_override)}_nt_summary.json'
    (marker_dir / summary_name).write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8',
    )
    done_marker.write_text(
        f'ok grid={args.grid} m={prefix_m} pt={prefix_temp} ep={args.episodes}\n',
        encoding='utf-8',
    )
    print(
        f'DONE step={step} grid={args.grid} m={prefix_m} pt={prefix_temp:g} '
        f'mean={summary["overall_success_mean"]:.4f} '
        f'max={summary["overall_success_max"]:.4f}'
    )


if __name__ == '__main__':
    main()
