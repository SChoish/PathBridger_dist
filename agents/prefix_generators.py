"""Joint stochastic generators for five-step bridge residuals."""

from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from utils.networks import MLP


_HIDDEN_DIMS = (512, 512, 512)
_LAYER_NORM = True


def _validate_low_rank_shapes(
    targets: jnp.ndarray,
    means: jnp.ndarray,
    sigmas: jnp.ndarray,
    factors: jnp.ndarray,
) -> None:
    """Validate event and low-rank axes without materializing covariance."""

    if targets.shape != means.shape or targets.shape != sigmas.shape:
        raise ValueError(
            'targets, means, and sigmas must have identical shapes; got '
            f'{targets.shape}, {means.shape}, and {sigmas.shape}.'
        )
    if factors.shape[:-1] != targets.shape:
        raise ValueError(
            'factors must have shape [..., event_dim, rank]; got '
            f'{factors.shape} for targets {targets.shape}.'
        )


def low_rank_gaussian_nll(
    targets: jnp.ndarray,
    means: jnp.ndarray,
    sigmas: jnp.ndarray,
    factors: jnp.ndarray,
) -> jnp.ndarray:
    """Stable per-example NLL for ``diag(sigmas**2) + U U^T``.

    Woodbury and the matrix-determinant lemma reduce the only factorization
    from ``event_dim`` to ``rank``.
    """

    _validate_low_rank_shapes(targets, means, sigmas, factors)
    if targets.ndim < 2:
        raise ValueError('low-rank Gaussian inputs need a batch and event axis.')
    if factors.shape[-1] == 0:
        normalized = (targets - means) / sigmas
        return 0.5 * (
            targets.shape[-1] * jnp.log(2.0 * jnp.pi)
            + 2.0 * jnp.sum(jnp.log(sigmas), axis=-1)
            + jnp.sum(jnp.square(normalized), axis=-1)
        )

    normalized = (targets - means) / sigmas
    whitened_factors = factors / sigmas[..., :, None]
    rank = factors.shape[-1]
    gram = (
        jnp.eye(rank, dtype=targets.dtype)
        + jnp.einsum('...pr,...ps->...rs', whitened_factors, whitened_factors)
    )
    cholesky = jnp.linalg.cholesky(gram)
    projected = jnp.einsum('...pr,...p->...r', whitened_factors, normalized)
    solved = solve_triangular(
        cholesky,
        projected[..., None],
        lower=True,
    )[..., 0]
    mahalanobis = jnp.maximum(
        jnp.sum(jnp.square(normalized), axis=-1)
        - jnp.sum(jnp.square(solved), axis=-1),
        0.0,
    )
    log_determinant = (
        2.0 * jnp.sum(jnp.log(sigmas), axis=-1)
        + 2.0
        * jnp.sum(
            jnp.log(jnp.diagonal(cholesky, axis1=-2, axis2=-1)),
            axis=-1,
        )
    )
    return 0.5 * (
        targets.shape[-1] * jnp.log(2.0 * jnp.pi)
        + log_determinant
        + mahalanobis
    )


class LowRankGaussianPrefix(nn.Module):
    """Conditional low-rank Gaussian over one flattened residual prefix."""

    event_dim: int
    rank: int
    sigma_floor: float
    hidden_dims: Sequence[int] = _HIDDEN_DIMS

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        endpoint_displacements: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        inputs = jnp.concatenate([observations, endpoint_displacements], axis=-1)
        hidden = MLP(
            tuple(self.hidden_dims),
            activate_final=True,
            layer_norm=_LAYER_NORM,
        )(inputs)
        means = nn.Dense(self.event_dim, name='means')(hidden)
        raw_sigmas = nn.Dense(self.event_dim, name='raw_sigmas')(hidden)
        sigmas = nn.softplus(raw_sigmas) + float(self.sigma_floor)
        factors = nn.Dense(self.event_dim * self.rank, name='factors')(hidden)
        factors = factors.reshape(*factors.shape[:-1], self.event_dim, self.rank)
        return means, sigmas, factors


class JointFlowPrefix(nn.Module):
    """Conditional velocity field over one flattened residual prefix."""

    event_dim: int
    hidden_dims: Sequence[int] = _HIDDEN_DIMS

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        endpoint_displacements: jnp.ndarray,
        noisy_residuals: jnp.ndarray,
        times: jnp.ndarray,
    ) -> jnp.ndarray:
        times = jnp.asarray(times, dtype=jnp.float32)
        if times.ndim == noisy_residuals.ndim - 1:
            times = times[..., None]
        inputs = jnp.concatenate(
            [observations, endpoint_displacements, noisy_residuals, times],
            axis=-1,
        )
        return MLP(
            (*self.hidden_dims, self.event_dim),
            activate_final=False,
            layer_norm=_LAYER_NORM,
        )(inputs)


__all__ = [
    'JointFlowPrefix',
    'LowRankGaussianPrefix',
    'low_rank_gaussian_nll',
]
