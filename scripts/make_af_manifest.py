"""Write staged action-free benchmark command manifests."""

from __future__ import annotations

import argparse
import csv
import shlex
from pathlib import Path


ENVIRONMENTS = {
    'antmaze-medium-navigate-v0': 'antmaze_medium',
    'antmaze-large-navigate-v0': 'antmaze_large',
    'cube-single-play-v0': 'cube_single',
    'cube-double-play-v0': 'cube_double',
    'cube-triple-play-v0': 'cube_triple',
    'puzzle-3x3-play-v0': 'puzzle_3x3',
    'puzzle-4x4-play-v0': 'puzzle_4x4',
    'scene-play-v0': 'scene',
}
ALGORITHMS = (
    'pbf_online_idm',
    'gc_mscp_style',
    'hiql_endpoint_online',
    'gc_af_guide',
    'gc_oso_decqn_factorized',
    'gc_sac',
    'gc_td3',
)
FULL_ACTION_ALGORITHMS = ('gc_sac_50_50',)
EVAL_STEPS = '0,10000,25000,50000,100000,250000,500000,1000000'
SUITES = {
    'p0_smoke': {
        'environments': (
            'antmaze-medium-navigate-v0',
            'cube-double-play-v0',
            'scene-play-v0',
        ),
        'seeds': (0,),
        'online_steps': 50_000,
        'eval_steps': '0,10000,25000,50000',
        'run_group': 'p0_smoke_50k',
    },
    'pilot': {
        'environments': (
            'antmaze-large-navigate-v0',
            'cube-double-play-v0',
            'puzzle-4x4-play-v0',
            'scene-play-v0',
        ),
        'seeds': (0, 1, 2),
        'online_steps': 250_000,
        'eval_steps': '0,10000,25000,50000,100000,250000',
        'run_group': 'pilot_250k',
    },
    'screening': {
        'environments': tuple(ENVIRONMENTS),
        'seeds': (0, 1, 2, 3, 4),
        'online_steps': 1_000_000,
        'eval_steps': EVAL_STEPS,
        'run_group': 'pre_tuning_screening_1m',
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='benchmark_manifest.csv')
    parser.add_argument('--python', default='python')
    parser.add_argument('--suite', choices=tuple(SUITES), default='p0_smoke')
    parser.add_argument('--run_group', default='')
    parser.add_argument('--include_full_action', action='store_true')
    parser.add_argument('--offline_steps', type=int, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    suite = SUITES[args.suite]
    run_group = args.run_group or suite['run_group']
    rows = []
    algorithms = ALGORITHMS + (FULL_ACTION_ALGORITHMS if args.include_full_action else ())
    for algorithm in algorithms:
        for environment in suite['environments']:
            config_name = ENVIRONMENTS[environment]
            for seed in suite['seeds']:
                command = [
                    args.python,
                    str(root / 'train_af.py'),
                    f'--algorithm={algorithm}',
                    f'--seed={seed}',
                    f'--run_group={run_group}',
                    f'--protocol_suite={args.suite}',
                    f'--online_steps={suite["online_steps"]}',
                    '--random_steps=10000',
                    '--replay_capacity=1000000',
                    f'--eval_steps={suite["eval_steps"]}',
                    '--eval_episodes=10',
                ]
                if algorithm == 'pbf_online_idm':
                    command.append(
                        f'--pbf={root / "configs" / "pbf_af" / (config_name + ".py")}'
                    )
                else:
                    command.append(f'--env_name={environment}')
                if args.offline_steps is not None:
                    command.append(f'--offline_steps={args.offline_steps}')
                rows.append(
                    {
                        'suite': args.suite,
                        'algorithm': algorithm,
                        'environment': environment,
                        'seed': seed,
                        'command': shlex.join(command),
                    }
                )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} {args.suite} runs to {output}')


if __name__ == '__main__':
    main()
