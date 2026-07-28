"""Small neural-network building blocks used by PathBridger."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import flax.linen as nn


def default_init(scale: float = 1.0) -> Callable[..., Any]:
    """Return the default dense-layer initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


class MLP(nn.Module):
    """A GELU MLP with optional LayerNorm after each activated layer."""

    hidden_dims: Sequence[int]
    activate_final: bool = False
    layer_norm: bool = False
    kernel_init: Callable[..., Any] = default_init()

    @nn.compact
    def __call__(self, x):
        for index, width in enumerate(self.hidden_dims):
            x = nn.Dense(width, kernel_init=self.kernel_init)(x)
            if index < len(self.hidden_dims) - 1 or self.activate_final:
                x = nn.gelu(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return x


__all__ = ['MLP', 'default_init']
