from __future__ import annotations

import pytest

from agents.online_idm import get_config as idm_config
from agents.registry import ALGORITHMS, algorithm_metadata
from aggregate_results import _auc
from scripts.make_af_manifest import ALGORITHMS as MAIN_ALGORITHMS
from scripts.make_af_manifest import ENVIRONMENTS, EVAL_STEPS
from utils.af_data import FORBIDDEN_OFFLINE_FIELDS
from utils.af_evaluation import evaluate_policy


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


def test_auc_is_normalized_trapezoid():
    points = [(0, 0.0), (100_000, 0.5), (250_000, 1.0)]
    assert _auc(points, 250_000) == pytest.approx(0.55)


def test_evaluation_rejects_empty_protocol():
    with pytest.raises(ValueError, match='episodes_per_task'):
        evaluate_policy(object(), None, episodes_per_task=0)
