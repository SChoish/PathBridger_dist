"""Write isolated visual OGBench manifests for the pixel LAPO extension."""

from __future__ import annotations

import argparse
import csv
import shlex
from pathlib import Path


ENVIRONMENTS = (
    'visual-antmaze-medium-navigate-v0',
    'visual-antmaze-large-navigate-v0',
    'visual-cube-single-play-v0',
    'visual-cube-double-play-v0',
    'visual-cube-triple-play-v0',
    'visual-puzzle-3x3-play-v0',
    'visual-puzzle-4x4-play-v0',
    'visual-scene-play-v0',
)
ALGORITHM = 'gc_pixel_lapo_decoder'
EVAL_STEPS = '0,10000,25000,50000,100000,250000,500000,1000000'
SUITES = {
    'p0_smoke': {
        'environments': (
            'visual-antmaze-medium-navigate-v0',
            'visual-cube-double-play-v0',
            'visual-scene-play-v0',
        ),
        'seeds': (0,),
        'online_steps': 50_000,
        'eval_steps': '0,10000,25000,50000',
        'run_group': 'pixel_p0_smoke_50k',
    },
    'pilot': {
        'environments': (
            'visual-antmaze-large-navigate-v0',
            'visual-cube-double-play-v0',
            'visual-puzzle-4x4-play-v0',
            'visual-scene-play-v0',
        ),
        'seeds': (0, 1, 2),
        'online_steps': 250_000,
        'eval_steps': '0,10000,25000,50000,100000,250000',
        'run_group': 'pixel_pilot_250k',
    },
    'screening': {
        'environments': ENVIRONMENTS,
        'seeds': (0, 1, 2, 3, 4),
        'online_steps': 1_000_000,
        'eval_steps': EVAL_STEPS,
        'run_group': 'pixel_screening_1m',
    },
}


def build_rows(*, root: Path, python: str, suite_name: str, run_group: str = ''):
    suite = SUITES[suite_name]
    group = run_group or suite['run_group']
    rows = []
    for environment in suite['environments']:
        for seed in suite['seeds']:
            command = [
                python,
                str(root / 'train_pixel_lapo.py'),
                f'--env_name={environment}',
                f'--seed={seed}',
                f'--run_group={group}',
                f'--protocol_suite={suite_name}',
                f'--online_steps={suite["online_steps"]}',
                '--random_steps=10000',
                '--replay_capacity=20000',
                f'--eval_steps={suite["eval_steps"]}',
                '--eval_episodes=10',
            ]
            rows.append(
                {
                    'suite': suite_name,
                    'algorithm': ALGORITHM,
                    'environment': environment,
                    'seed': seed,
                    'command': shlex.join(command),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='pixel_lapo_manifest.csv')
    parser.add_argument('--python', default='python')
    parser.add_argument('--suite', choices=tuple(SUITES), default='p0_smoke')
    parser.add_argument('--run_group', default='')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = build_rows(
        root=root,
        python=args.python,
        suite_name=args.suite,
        run_group=args.run_group,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} isolated pixel LAPO runs to {output}')


if __name__ == '__main__':
    main()
