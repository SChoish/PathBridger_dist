"""Dataset-directory isolation tests with a fake OGBench module."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from envs.env_utils import make_env_and_datasets


def _raw_dataset(marker: float) -> dict[str, np.ndarray]:
    return {
        'observations': np.full((2, 3), marker, dtype=np.float32),
        'actions': np.zeros((2, 1), dtype=np.float32),
        'terminals': np.asarray([0.0, 1.0], dtype=np.float32),
    }


def test_shared_cache_selects_only_the_requested_dataset_subtree(
    tmp_path,
    monkeypatch,
):
    requested = 'cube-double-play-v0'
    other = 'scene-play-v0'
    paths = [
        tmp_path / requested / 'train-0.npz',
        tmp_path / requested / 'val-0.npz',
        tmp_path / other / 'train-0.npz',
        tmp_path / other / 'val-0.npz',
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    loaded: list[str] = []

    def load_dataset(path, *, compact_dataset):
        assert compact_dataset is True
        loaded.append(str(path))
        marker = 1.0 if requested in str(path) else 9.0
        return _raw_dataset(marker)

    fake_ogbench = SimpleNamespace(
        load_dataset=load_dataset,
        make_env_and_datasets=lambda name, **kwargs: object(),
    )
    monkeypatch.setitem(sys.modules, 'ogbench', fake_ogbench)

    _, train, validation = make_env_and_datasets(requested, tmp_path)

    assert len(loaded) == 2
    assert all(requested in path for path in loaded)
    np.testing.assert_array_equal(train['observations'], _raw_dataset(1.0)['observations'])
    np.testing.assert_array_equal(
        validation['observations'],
        _raw_dataset(1.0)['observations'],
    )

