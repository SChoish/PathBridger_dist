"""Online-only inverse dynamics and frozen-PBF execution wrapper."""

from __future__ import annotations

import dataclasses
import hashlib
from functools import partial
from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from agents.pathbridger import InverseDynamics, PathBridgerAgent
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field


class OnlineIDMAgent(flax.struct.PyTreeNode):
    """Deterministic IDM whose optimizer contains no planner parameters."""

    rng: Any
    network: TrainState
    config: Any = nonpytree_field()

    @partial(jax.jit, static_argnames=())
    def update(
        self, batch: dict[str, jnp.ndarray]
    ) -> tuple['OnlineIDMAgent', dict[str, jnp.ndarray]]:
        required = ('observations', 'next_observations', 'actions')
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f'Online IDM batch is missing {missing}.')

        def loss_fn(params):
            predictions = self.network.select('idm')(
                batch['observations'], batch['next_observations'], params=params
            )
            errors = predictions - batch['actions']
            if str(self.config.get('loss', 'mse')) == 'l1':
                per_example = jnp.sum(jnp.abs(errors), axis=-1)
            else:
                per_example = jnp.sum(jnp.square(errors), axis=-1)
            loss = per_example.mean()
            return loss, {
                'idm/loss': loss,
                'idm/action_mse': jnp.mean(jnp.square(errors)),
                'idm/action_l1': jnp.mean(jnp.abs(errors)),
                'idm/action_abs_mean': jnp.mean(jnp.abs(predictions)),
            }

        network, info = self.network.apply_loss_fn(loss_fn)
        return self.replace(network=network), info

    online_update = update

    @jax.jit
    def predict(
        self,
        observations: jnp.ndarray,
        desired_next_observations: jnp.ndarray,
    ) -> jnp.ndarray:
        return jnp.clip(
            self.network.select('idm')(observations, desired_next_observations),
            -1.0,
            1.0,
        )

    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations: jnp.ndarray,
        action_dim: int,
        config: dict[str, Any] | ml_collections.ConfigDict,
    ) -> 'OnlineIDMAgent':
        config = dict(config)
        observations = jnp.asarray(ex_observations, dtype=jnp.float32)
        if observations.ndim != 2:
            raise ValueError('ex_observations must have shape [B, D].')
        module = InverseDynamics(
            action_dim=int(action_dim),
            hidden_dims=tuple(config['hidden_dims']),
        )
        model = ModuleDict({'idm': module})
        rng = jax.random.PRNGKey(int(seed))
        rng, init_rng = jax.random.split(rng)
        params = model.init(
            init_rng,
            idm=(observations, observations),
        )['params']
        network = TrainState.create(
            model,
            params,
            tx=optax.adam(float(config['learning_rate'])),
        )
        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(config),
        )


def parameter_digest(tree: Any) -> str:
    """Stable digest used to audit that the PBF tree stayed frozen."""

    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.asarray(jax.device_get(leaf))
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class PBFOnlineIDMPolicy:
    """Compose one frozen action-free PBF with one mutable online IDM."""

    planner: PathBridgerAgent
    idm: OnlineIDMAgent
    num_candidates: int
    endpoint_temperature: float
    execute_horizon: int = 1

    def __post_init__(self):
        if not bool(self.planner.config.get('offline_action_free', False)):
            raise ValueError('PBFOnlineIDMPolicy requires an action-free PBF checkpoint.')
        if 'modules_idm' in self.planner.network.params:
            raise ValueError('Frozen action-free PBF unexpectedly contains IDM parameters.')
        if self.execute_horizon not in (1, 5):
            raise ValueError('execute_horizon must be 1 or 5.')

    def desired_prefix(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        *,
        seed: jax.Array,
    ) -> jnp.ndarray:
        return self.planner.sample_state_prefix(
            observations,
            goals,
            seed=seed,
            num_candidates=int(self.num_candidates),
            temperature=float(self.endpoint_temperature),
        )

    def sample_actions(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        seed: jax.Array | None = None,
        temperature: float = 0.0,
    ) -> jnp.ndarray:
        del temperature
        if seed is None:
            seed = jax.random.PRNGKey(0)
        prefix = self.desired_prefix(observations, goals, seed=seed)
        return self.decode_prefix(prefix)

    def decode_prefix(self, prefix: jnp.ndarray) -> jnp.ndarray:
        """Decode one planned prefix into the configured primitive action chunk."""

        if prefix.ndim == 2:
            prefix = prefix[None, ...]
        batch_size, _, state_dim = prefix.shape
        horizon = int(self.execute_horizon)
        if prefix.shape[1] < horizon + 1:
            raise ValueError(
                f'Prefix length {prefix.shape[1]} is too short for horizon {horizon}.'
            )
        current = prefix[:, :horizon, :].reshape(batch_size * horizon, state_dim)
        following = prefix[:, 1 : horizon + 1, :].reshape(
            batch_size * horizon, state_dim
        )
        actions = self.idm.predict(current, following).reshape(batch_size, horizon, -1)
        return actions[:, 0, :] if horizon == 1 else actions


def get_config() -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict(
        dict(
            hidden_dims=(512, 512, 512),
            learning_rate=1e-3,
            batch_size=512,
            replay_capacity=1_000_000,
            update_start=1_000,
            random_steps=10_000,
            execute_horizon=1,
            exploration_std_initial=0.2,
            exploration_std_final=0.05,
            exploration_decay_steps=90_000,
            loss='l1',
        )
    )


__all__ = [
    'OnlineIDMAgent',
    'PBFOnlineIDMPolicy',
    'get_config',
    'parameter_digest',
]
