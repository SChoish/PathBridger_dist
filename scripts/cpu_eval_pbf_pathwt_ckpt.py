#!/usr/bin/env python3
"""CPU NT-grid eval for one pathbridger (PBF flow + temporal path weighting) checkpoint.

Full protocol NT grid (19 cells), matching the PathBridger PBF sweeps:
  (N=1, T=0) + N in {1,2,4,8,16,32} x T in {0.25,0.5,1.0}

Writes per-cell JSON under eval_results/ and marks cpu_eval/step_{step}.DONE
when the full grid for that step is complete.
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

# Full grid search (not the sparse Best-N 4-point subset).
# Order is intentional: when an 800k/900k/1M ckpt appears, evaluate low-N
# first ((1,0) then N=1/2/4), then N>=8. GPU large-N sidecar only starts a
# step after these small cells exist (see watch_gpu_eval_large_n_*.sh).
FLOW_NT_GRID = (
    (1, 0.0),
    *[(n, t) for n in (1, 2, 4) for t in (0.25, 0.5, 1.0)],
    *[(n, t) for n in (8, 16, 32) for t in (0.25, 0.5, 1.0)],
)


def _t_tok(t: float) -> str:
    return str(int(t)) if float(t) == int(t) else str(t).replace('.', 'p')


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
    # Explicit fallback if get_config ever drops the key.
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
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    step = int(args.checkpoint_step)
    ckpt_dir = run_dir / 'checkpoints'
    ckpt = ckpt_dir / f'params_{step}.pkl'
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    marker_dir = run_dir / 'cpu_eval'
    marker_dir.mkdir(parents=True, exist_ok=True)
    done_marker = marker_dir / f'step_{step}.DONE'
    if done_marker.is_file():
        print(f'skip already done {done_marker}')
        return

    config, flags_payload = _load_agent_config(run_dir)
    env, train_data, _ = make_env_and_datasets(
        str(config.env_name),
        dataset_dir=args.dataset_dir or None,
    )
    dataset = PathBridgerDataset(train_data, config)
    example_batch = dataset.sample(1)
    # Wave1 runs predate state_scale in flags.json; omit → create() uses ones,
    # matching the in-memory trainers that wrote those checkpoints.
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
    overalls = []
    for n_cand, temp in FLOW_NT_GRID:
        t_tok = _t_tok(temp)
        out = eval_dir / f'epoch{step}_t{t_tok}_n{n_cand}.json'
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
            'eval_num_candidates': int(n_cand),
            'eval_temperature': float(temp),
            **{f'evaluation/{k}': float(v) for k, v in metrics.items()},
        }
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        overalls.append(float(metrics['overall_success']))
        print(f'step={step} n={n_cand} t={temp:g} overall={metrics["overall_success"]:.4f}')

    # Full grid present?
    missing = [
        (n, t)
        for n, t in FLOW_NT_GRID
        if not (eval_dir / f'epoch{step}_t{_t_tok(t)}_n{n}.json').is_file()
    ]
    if missing:
        raise RuntimeError(f'NT grid incomplete for step={step}: {missing}')
    summary = {
        'checkpoint_step': step,
        'nt_grid': [{'n': n, 't': t} for n, t in FLOW_NT_GRID],
        'overall_success_mean': float(sum(overalls) / len(overalls)),
        'overall_success_max': float(max(overalls)),
    }
    (marker_dir / f'step_{step}_nt_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8',
    )
    done_marker.write_text('ok\n', encoding='utf-8')
    print(
        f'DONE step={step} mean={summary["overall_success_mean"]:.4f} '
        f'max={summary["overall_success_max"]:.4f}'
    )


if __name__ == '__main__':
    main()
