"""Aggregate train_af evaluation curves without mixing information regimes."""

from __future__ import annotations

import csv
import itertools
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
flags.DEFINE_bool(
    'require_balanced_matrix',
    True,
    'Require every algorithm summary to have the same seeds for every environment.',
)


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


def _score_matrix(rows, metric):
    envs = sorted({row['env_name'] for row in rows})
    seeds = sorted({int(row['seed']) for row in rows})
    values = {
        (int(row['seed']), row['env_name']): float(row[metric]) for row in rows
    }
    missing = [
        (seed, env) for seed in seeds for env in envs if (seed, env) not in values
    ]
    if missing:
        raise ValueError(
            f'Unbalanced seed x environment matrix for {metric}; missing {missing[:8]}.'
        )
    matrix = np.asarray(
        [[values[(seed, env)] for env in envs] for seed in seeds],
        dtype=np.float64,
    )
    return matrix, tuple(seeds), tuple(envs)


def _stratified_ci(matrix, samples, seed):
    """Rliable-compatible bootstrap over runs, stratified by environment."""

    rng = np.random.default_rng(seed)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size:
        return [float('nan'), float('nan')]
    num_runs, num_envs = matrix.shape
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        run_indices = rng.integers(0, num_runs, size=(num_runs, num_envs))
        sampled = np.take_along_axis(matrix, run_indices, axis=0)
        draws[index] = _iqm(sampled.reshape(-1))
    return [float(value) for value in np.percentile(draws, [2.5, 97.5])]


def _probability_of_improvement(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError('Probability matrices must share the environment dimension.')
    comparisons = left[:, None, :] - right[None, :, :]
    return float(np.mean((comparisons > 0.0) + 0.5 * (comparisons == 0.0)))


def _probability_ci(left, right, samples, seed):
    rng = np.random.default_rng(seed)
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        left_indices = rng.integers(0, left.shape[0], size=left.shape)
        right_indices = rng.integers(0, right.shape[0], size=right.shape)
        left_sample = np.take_along_axis(left, left_indices, axis=0)
        right_sample = np.take_along_axis(right, right_indices, axis=0)
        draws[index] = _probability_of_improvement(left_sample, right_sample)
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
    matrices = {}
    by_algorithm = defaultdict(list)
    for record in records:
        by_algorithm[record['algorithm']].append(record)
    for algorithm, rows in sorted(by_algorithm.items()):
        try:
            final_matrix, seeds, envs = _score_matrix(rows, 'final_success')
            auc_250k_matrix, _, _ = _score_matrix(rows, 'auc_250k')
            auc_full_matrix, _, _ = _score_matrix(rows, 'auc_full')
        except ValueError:
            if FLAGS.require_balanced_matrix:
                raise
            continue
        final_ci = _stratified_ci(
            final_matrix, FLAGS.bootstrap_samples, FLAGS.seed
        )
        auc_250k_ci = (
            _stratified_ci(
                auc_250k_matrix, FLAGS.bootstrap_samples, FLAGS.seed + 1
            )
            if np.all(np.isfinite(auc_250k_matrix))
            else [float('nan'), float('nan')]
        )
        auc_full_ci = _stratified_ci(
            auc_full_matrix, FLAGS.bootstrap_samples, FLAGS.seed + 2
        )
        comparison_metric = (
            'auc_250k'
            if np.all(np.isfinite(auc_250k_matrix))
            else 'auc_full'
        )
        matrices[algorithm] = {
            'group': rows[0]['group'],
            'envs': envs,
            'metric': comparison_metric,
            'scores': (
                auc_250k_matrix
                if comparison_metric == 'auc_250k'
                else auc_full_matrix
            ),
        }
        summaries.append(
            {
                'algorithm': algorithm,
                'group': rows[0]['group'],
                'port_kind': rows[0]['port_kind'],
                'num_runs': len(rows),
                'num_envs': len(envs),
                'num_seeds': len(seeds),
                'final_iqm': _iqm(final_matrix.reshape(-1)),
                'final_iqm_ci_low': final_ci[0],
                'final_iqm_ci_high': final_ci[1],
                'auc_250k_iqm': _iqm(auc_250k_matrix.reshape(-1)),
                'auc_250k_iqm_ci_low': auc_250k_ci[0],
                'auc_250k_iqm_ci_high': auc_250k_ci[1],
                'auc_full_iqm': _iqm(auc_full_matrix.reshape(-1)),
                'auc_full_iqm_ci_low': auc_full_ci[0],
                'auc_full_iqm_ci_high': auc_full_ci[1],
            }
        )
    if summaries:
        with (output_dir / 'summary.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)

    comparisons = []
    for left_name, right_name in itertools.combinations(sorted(matrices), 2):
        left = matrices[left_name]
        right = matrices[right_name]
        if (
            left['group'] != right['group']
            or left['envs'] != right['envs']
            or left['metric'] != right['metric']
        ):
            continue
        probability = _probability_of_improvement(
            left['scores'], right['scores']
        )
        interval = _probability_ci(
            left['scores'],
            right['scores'],
            FLAGS.bootstrap_samples,
            FLAGS.seed + 3,
        )
        comparisons.append(
            {
                'algorithm_x': left_name,
                'algorithm_y': right_name,
                'group': left['group'],
                'metric': left['metric'],
                'probability_x_better': probability,
                'ci_low': interval[0],
                'ci_high': interval[1],
            }
        )
    if comparisons:
        with (output_dir / 'comparisons.csv').open(
            'w', newline='', encoding='utf-8'
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
    (output_dir / 'summary.json').write_text(
        json.dumps(
            {'runs': records, 'summary': summaries, 'comparisons': comparisons},
            indent=2,
            sort_keys=True,
        )
        + '\n'
    )
    print(f'Aggregated {len(records)} runs into {output_dir}')


def run():
    app.run(main)


if __name__ == '__main__':
    run()
