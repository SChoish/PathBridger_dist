from __future__ import annotations

import numpy as np
import pytest

from agents.online_idm import get_config as idm_config
from agents.registry import ALGORITHMS, algorithm_metadata
from aggregate_results import (
    _auc,
    _probability_of_improvement,
    _score_matrix,
    _stratified_ci,
)
from scripts.make_af_manifest import ALGORITHMS as MAIN_ALGORITHMS
from scripts.make_af_manifest import ENVIRONMENTS, EVAL_STEPS, SUITES
from utils.af_data import FORBIDDEN_OFFLINE_FIELDS
from utils.af_evaluation import evaluate_policy
from utils.af_exploration import collection_noise_std


def test_action_free_information_regimes_and_online_only_idm():
    for name in ALGORITHMS:
        metadata = algorithm_metadata(name)
        if metadata.port_kind not in ('full_action', 'online_only'):
            assert not (set(metadata.offline_fields_seen) & FORBIDDEN_OFFLINE_FIELDS)
            assert not metadata.uses_offline_actions
    proposed = algorithm_metadata('pbf_online_idm')
    assert proposed.online_modules_updated == ('idm',)
    config = idm_config()
    assert config.loss == 'l1'
    assert config.learning_rate == 1e-3


def test_locked_main_manifest_dimensions_and_schedule():
    assert len(MAIN_ALGORITHMS) * len(ENVIRONMENTS) * 5 == 280
    assert EVAL_STEPS == '0,10000,25000,50000,100000,250000,500000,1000000'
    assert len(MAIN_ALGORITHMS) * len(SUITES['p0_smoke']['environments']) == 21
    assert (
        len(MAIN_ALGORITHMS)
        * len(SUITES['pilot']['environments'])
        * len(SUITES['pilot']['seeds'])
        == 84
    )


def test_auc_is_normalized_trapezoid():
    points = [(0, 0.0), (100_000, 0.5), (250_000, 1.0)]
    assert _auc(points, 250_000) == pytest.approx(0.55)


def test_rliable_style_matrix_and_stratified_bootstrap():
    rows = [
        {'seed': seed, 'env_name': env, 'score': float(seed + env_index)}
        for seed in (0, 1)
        for env_index, env in enumerate(('env_a', 'env_b'))
    ]
    matrix, seeds, envs = _score_matrix(rows, 'score')
    assert matrix.shape == (2, 2)
    assert seeds == (0, 1)
    assert envs == ('env_a', 'env_b')
    constant_interval = _stratified_ci(np.ones((3, 2)), 20, 0)
    assert constant_interval == pytest.approx([1.0, 1.0])
    assert _probability_of_improvement(
        np.ones((2, 2)), np.zeros((3, 2))
    ) == pytest.approx(1.0)


def test_score_matrix_rejects_unbalanced_runs():
    rows = [
        {'seed': 0, 'env_name': 'a', 'score': 1.0},
        {'seed': 0, 'env_name': 'b', 'score': 1.0},
        {'seed': 1, 'env_name': 'a', 'score': 1.0},
    ]
    with pytest.raises(ValueError, match='Unbalanced'):
        _score_matrix(rows, 'score')


def test_score_matrix_rejects_duplicate_runs():
    rows = [
        {'seed': 0, 'env_name': 'a', 'score': 1.0},
        {'seed': 0, 'env_name': 'a', 'score': 2.0},
    ]
    with pytest.raises(ValueError, match='Duplicate'):
        _score_matrix(rows, 'score')


def test_evaluation_rejects_empty_protocol():
    with pytest.raises(ValueError, match='episodes_per_task'):
        evaluate_policy(object(), None, episodes_per_task=0)


def test_deterministic_collection_noise_schedule():
    config = {
        'collection_noise_initial': 0.2,
        'collection_noise_final': 0.05,
        'collection_noise_decay_steps': 100_000,
    }
    assert collection_noise_std('gc_sac', 10_001, 10_000, config) == 0.0
    assert collection_noise_std('gc_td3', 10_000, 10_000, config) == pytest.approx(0.2)
    assert collection_noise_std('gc_td3', 60_000, 10_000, config) == pytest.approx(0.125)
    assert collection_noise_std(
        'gc_oso_decqn_factorized', 110_000, 10_000, config
    ) == pytest.approx(0.05)
