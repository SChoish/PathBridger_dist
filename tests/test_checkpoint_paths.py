"""Checkpoint path semantics for installations with the ML stack."""

from __future__ import annotations

import pytest


pytest.importorskip('flax')
pytest.importorskip('jax')
pytest.importorskip('optax')

from utils.flax_utils import resolve_checkpoint  # noqa: E402


def test_exact_checkpoint_infers_and_validates_step(tmp_path):
    exact = tmp_path / 'params_800000.pkl'
    assert resolve_checkpoint(exact, 0) == (exact, 800_000)
    assert resolve_checkpoint(exact, 800_000) == (exact, 800_000)
    with pytest.raises(ValueError, match='filename implies step'):
        resolve_checkpoint(exact, 900_000)


def test_checkpoint_directory_requires_step(tmp_path):
    with pytest.raises(ValueError, match='requires an explicit positive step'):
        resolve_checkpoint(tmp_path, 0)
    assert resolve_checkpoint(tmp_path, 12) == (
        tmp_path / 'params_12.pkl',
        12,
    )

