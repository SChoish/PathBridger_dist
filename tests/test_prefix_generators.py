"""Numerical tests for joint stochastic bridge-prefix helpers."""

from __future__ import annotations

import numpy as np
import pytest


jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')
pytest.importorskip('flax')

from agents.prefix_generators import low_rank_gaussian_nll  # noqa: E402


def _dense_gaussian_nll(target, mean, sigma, factor):
    covariance = np.diag(np.square(sigma)) + factor @ factor.T
    residual = target - mean
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign > 0
    return 0.5 * (
        target.size * np.log(2.0 * np.pi)
        + logdet
        + residual @ np.linalg.solve(covariance, residual)
    )


def test_low_rank_nll_matches_dense_covariance():
    rng = np.random.default_rng(4)
    targets = rng.normal(size=(3, 6)).astype(np.float32)
    means = rng.normal(size=(3, 6)).astype(np.float32)
    sigmas = (0.2 + rng.random(size=(3, 6))).astype(np.float32)
    factors = (0.3 * rng.normal(size=(3, 6, 2))).astype(np.float32)

    actual = np.asarray(
        low_rank_gaussian_nll(targets, means, sigmas, factors)
    )
    expected = np.asarray([
        _dense_gaussian_nll(targets[i], means[i], sigmas[i], factors[i])
        for i in range(3)
    ])
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_zero_factor_matches_diagonal_gaussian():
    targets = jnp.asarray([[1.0, -2.0, 0.5]], dtype=jnp.float32)
    means = jnp.asarray([[0.2, -0.1, 0.3]], dtype=jnp.float32)
    sigmas = jnp.asarray([[0.5, 0.7, 1.2]], dtype=jnp.float32)
    factors = jnp.zeros((1, 3, 2), dtype=jnp.float32)
    normalized = (targets - means) / sigmas
    expected = 0.5 * (
        3 * jnp.log(2.0 * jnp.pi)
        + 2.0 * jnp.sum(jnp.log(sigmas), axis=-1)
        + jnp.sum(jnp.square(normalized), axis=-1)
    )
    np.testing.assert_allclose(
        low_rank_gaussian_nll(targets, means, sigmas, factors),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_near_collinear_factor_loss_and_gradient_are_finite():
    targets = jnp.asarray([[0.5, -0.2, 0.1, 0.8]], dtype=jnp.float32)
    means = jnp.zeros_like(targets)
    sigmas = jnp.full_like(targets, 1e-3)
    base = jnp.asarray([1.0, 2.0, -1.0, 0.5], dtype=jnp.float32)
    factors = jnp.stack((base, base * (1.0 + 1e-6)), axis=-1)[None, :, :]

    def loss_fn(current_factors):
        return low_rank_gaussian_nll(
            targets,
            means,
            sigmas,
            current_factors,
        ).mean()

    loss = loss_fn(factors)
    gradient = jax.grad(loss_fn)(factors)
    assert np.isfinite(np.asarray(loss)).all()
    assert np.isfinite(np.asarray(gradient)).all()
