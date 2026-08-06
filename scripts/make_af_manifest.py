"""Write the locked 8-environment x 5-seed benchmark command manifest."""

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
    'gc_mscp',
    'passive_hiql',
    'gc_af_guide',
    'gc_oso_decqn',
    'gc_sac',
    'gc_td3',
)
FULL_ACTION_ALGORITHMS = ('gc_rlpd',)
EVAL_STEPS = '0,10000,25000,50000,100000,250000,500000,1000000'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='benchmark_manifest.csv')
    parser.add_argument('--python', default='python')
    parser.add_argument('--run_group', default='main_1m')
    parser.add_argument('--include_full_action', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = []
    algorithms = ALGORITHMS + (FULL_ACTION_ALGORITHMS if args.include_full_action else ())
    for algorithm in algorithms:
        for environment, config_name in ENVIRONMENTS.items():
            for seed in range(5):
                command = [
                    args.python,
                    str(root / 'train_af.py'),
                    f'--algorithm={algorithm}',
                    f'--seed={seed}',
                    f'--run_group={args.run_group}',
                    '--online_steps=1000000',
                    '--random_steps=10000',
                    '--replay_capacity=1000000',
                    f'--eval_steps={EVAL_STEPS}',
                    '--eval_episodes=10',
                ]
                if algorithm == 'pbf_online_idm':
                    command.append(
                        f'--pbf={root / "configs" / "pbf_af" / (config_name + ".py")}'
                    )
                else:
                    command.append(f'--env_name={environment}')
                rows.append(
                    {
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
    print(f'Wrote {len(rows)} runs to {output}')


if __name__ == '__main__':
    main()
