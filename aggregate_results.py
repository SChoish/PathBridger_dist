"""Aggregate train_af evaluation curves without mixing information regimes."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from absl import app, flags


FLAGS = flags.FLAGS
flags.DEFINE_string('root', 'exp/pbf_af_o2o', 'Root containing train_af runs.')
flags.DEFINE_string('output_dir', '', 'Output directory; defaults to <root>/aggregate.')
flags.DEFINE_integer('bootstrap_samples', 10_000, 'Stratified bootstrap resamples.')
flags.DEFINE_integer('seed', 0, 'Bootstrap seed.')


def _read_curve(path: Path):
    with path.open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    points = []
    for row in rows:
        if row.get('step', '') and row.get('overall_success', ''):
            points.append((int(row['step']), float(row['overall_success'])))
    points.sort()
    return points


def _auc(points, budget):
    if not points:
        return float('nan')
    steps = np.asarray([point[0] for point in points], dtype=np.float64)
    values = np.asarray([point[1] for point in points], dtype=np.float64)
    keep = steps <= budget
    steps, values = steps[keep], values[keep]
    if len(steps) == 0:
        return float('nan')
    if steps[0] > 0:
        steps = np.concatenate([[0.0], steps])
        values = np.concatenate([[values[0]], values])
    if steps[-1] < budget:
        steps = np.concatenate([steps, [float(budget)]])
        values = np.concatenate([values, [values[-1]]])
    return float(np.trapezoid(values, steps) / max(float(budget), 1.0))


def _iqm(values):
    values = np.sort(np.asarray(values, dtype=np.float64))
    if not len(values):
        return float('nan')
    lower = int(np.floor(0.25 * len(values)))
    upper = int(np.ceil(0.75 * len(values)))
    return float(np.mean(values[lower:upper]))


def _stratified_ci(by_env, samples, seed):
    rng = np.random.default_rng(seed)
    envs = sorted(by_env)
    if not envs:
        return [float('nan'), float('nan')]
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled = []
        for env in rng.choice(envs, size=len(envs), replace=True):
            values = np.asarray(by_env[env], dtype=np.float64)
            sampled.append(float(rng.choice(values)))
        draws[index] = _iqm(sampled)
    return [float(value) for value in np.percentile(draws, [2.5, 97.5])]


def _result_group(port_kind):
    if port_kind == 'online_only':
        return 'online_only'
    if port_kind == 'full_action':
        return 'full_action_upper_bound'
    return 'action_free_main'


def main(_):
    root = Path(FLAGS.root)
    output_dir = Path(FLAGS.output_dir) if FLAGS.output_dir else root / 'aggregate'
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for metadata_path in root.rglob('metadata.json'):
        run_dir = metadata_path.parent
        eval_path = run_dir / 'eval.csv'
        if not eval_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text())
        points = _read_curve(eval_path)
        if not points:
            continue
        budget = int(metadata.get('online_steps', points[-1][0]))
        # Never turn an interrupted run into an apparently complete curve by
        # carrying its last score forward to the requested budget.
        if points[-1][0] < budget:
            continue
        records.append(
            {
                'algorithm': metadata['algorithm'],
                'group': _result_group(metadata['port_kind']),
                'port_kind': metadata['port_kind'],
                'env_name': metadata['env_name'],
                'seed': int(metadata['seed']),
                'auc_250k': (
                    _auc(points, 250_000) if budget >= 250_000 else float('nan')
                ),
                'auc_full': _auc(points, budget),
                'final_success': float(points[-1][1]),
                'final_step': int(points[-1][0]),
                'run_dir': str(run_dir),
            }
        )
    records.sort(key=lambda row: (row['group'], row['algorithm'], row['env_name'], row['seed']))
    if records:
        with (output_dir / 'runs.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    summaries = []
    by_algorithm = defaultdict(list)
    for record in records:
        by_algorithm[record['algorithm']].append(record)
    for algorithm, rows in sorted(by_algorithm.items()):
        final_by_env = defaultdict(list)
        auc_250k_by_env = defaultdict(list)
        auc_full_by_env = defaultdict(list)
        for row in rows:
            final_by_env[row['env_name']].append(row['final_success'])
            auc_250k_by_env[row['env_name']].append(row['auc_250k'])
            auc_full_by_env[row['env_name']].append(row['auc_full'])
        final_values = [row['final_success'] for row in rows]
        auc_250k_values = [row['auc_250k'] for row in rows]
        auc_full_values = [row['auc_full'] for row in rows]
        summaries.append(
            {
                'algorithm': algorithm,
                'group': rows[0]['group'],
                'port_kind': rows[0]['port_kind'],
                'num_runs': len(rows),
                'num_envs': len(final_by_env),
                'final_iqm': _iqm(final_values),
                'final_iqm_ci_low': _stratified_ci(final_by_env, FLAGS.bootstrap_samples, FLAGS.seed)[0],
                'final_iqm_ci_high': _stratified_ci(final_by_env, FLAGS.bootstrap_samples, FLAGS.seed)[1],
                'auc_250k_iqm': _iqm(auc_250k_values),
                'auc_250k_iqm_ci_low': _stratified_ci(
                    auc_250k_by_env, FLAGS.bootstrap_samples, FLAGS.seed + 1
                )[0],
                'auc_250k_iqm_ci_high': _stratified_ci(
                    auc_250k_by_env, FLAGS.bootstrap_samples, FLAGS.seed + 1
                )[1],
                'auc_full_iqm': _iqm(auc_full_values),
                'auc_full_iqm_ci_low': _stratified_ci(
                    auc_full_by_env, FLAGS.bootstrap_samples, FLAGS.seed + 2
                )[0],
                'auc_full_iqm_ci_high': _stratified_ci(
                    auc_full_by_env, FLAGS.bootstrap_samples, FLAGS.seed + 2
                )[1],
            }
        )
    if summaries:
        with (output_dir / 'summary.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    (output_dir / 'summary.json').write_text(
        json.dumps({'runs': records, 'summary': summaries}, indent=2, sort_keys=True) + '\n'
    )
    print(f'Aggregated {len(records)} runs into {output_dir}')


def run():
    app.run(main)


if __name__ == '__main__':
    run()
