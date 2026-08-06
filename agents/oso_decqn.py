"""Goal-conditioned OSO-DecQN paper reimplementation for OGBench."""

from __future__ import annotations

import dataclasses
from functools import partial
from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from agents.gc_actor_critic import GoalConditionedActorCritic, _replace_subtree, get_config as gc_config
from agents.online_idm import OnlineIDMAgent
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP


NUM_DELTA_BINS = 3


def discretize_deltas(
    deltas: Any,
    scale: Any,
    threshold: float = 0.25,
) -> np.ndarray:
    """Map normalized per-dimension deltas to negative/zero/positive bins."""

    deltas = np.asarray(deltas, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    normalized = deltas / np.maximum(scale, 1e-6)
    return np.where(
        normalized < -float(threshold),
        0,
        np.where(normalized > float(threshold), 2, 1),
    ).astype(np.int32)


class FactorizedDeltaQ(nn.Module):
    state_dim: int
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations, goals):
        inputs = jnp.concatenate([observations, goals], axis=-1)
        shape = (*inputs.shape[:-1], self.state_dim, NUM_DELTA_BINS)
        q1 = MLP((*self.hidden_dims, self.state_dim * NUM_DELTA_BINS), name='q1')(
            inputs
        ).reshape(shape)
        q2 = MLP((*self.hidden_dims, self.state_dim * NUM_DELTA_BINS), name='q2')(
            inputs
        ).reshape(shape)
        return q1, q2


class OSOStateAgent(flax.struct.PyTreeNode):
    rng: Any
    network: TrainState
    delta_scale: Any
    config: Any = nonpytree_field()

    @partial(jax.jit, static_argnames=())
    def offline_update(self, batch):
        bins = jnp.asarray(batch['delta_bins'], dtype=jnp.int32)

        def loss_fn(params):
            next_q1, next_q2 = self.network.select('target_q')(
                batch['next_observations'], batch['goals']
            )
            next_total = jnp.mean(
                jnp.max(jnp.minimum(next_q1, next_q2), axis=-1), axis=-1
            )
            target = jax.lax.stop_gradient(
                batch['rewards']
                + float(self.config['discount']) * batch['masks'] * next_total
            )
            q1, q2 = self.network.select('q')(
                batch['observations'], batch['goals'], params=params
            )
            selected1 = jnp.take_along_axis(q1, bins[..., None], axis=-1).squeeze(-1)
            selected2 = jnp.take_along_axis(q2, bins[..., None], axis=-1).squeeze(-1)
            total1 = jnp.mean(selected1, axis=-1)
            total2 = jnp.mean(selected2, axis=-1)
            bellman = jnp.mean(jnp.square(total1 - target) + jnp.square(total2 - target))
            cql1 = jnp.mean(jax.nn.logsumexp(q1, axis=-1) - selected1)
            cql2 = jnp.mean(jax.nn.logsumexp(q2, axis=-1) - selected2)
            conservative = cql1 + cql2
            total = bellman + float(self.config['conservative_weight']) * conservative
            return total, {
                'loss/total': total,
                'state_q/bellman': bellman,
                'state_q/conservative': conservative,
                'state_q/q_mean': 0.5 * (total1.mean() + total2.mean()),
            }

        network, info = self.network.apply_loss_fn(loss_fn)
        tau = float(self.config['tau'])
        target_params = jax.tree_util.tree_map(
            lambda value, old: tau * value + (1.0 - tau) * old,
            network.params['modules_q'],
            network.params['modules_target_q'],
        )
        network = network.replace(
            params=_replace_subtree(network.params, 'target_q', target_params)
        )
        return self.replace(network=network), info

    @jax.jit
    def desired_next(self, observations, goals):
        q1, q2 = self.network.select('q')(observations, goals)
        bins = jnp.argmax(jnp.minimum(q1, q2), axis=-1)
        directions = bins.astype(observations.dtype) - 1.0
        return observations + directions * self.delta_scale

    @classmethod
    def create(cls, seed, ex_observations, delta_scale, config):
        config = dict(config)
        observations = jnp.asarray(ex_observations, dtype=jnp.float32)
        state_dim = int(observations.shape[-1])
        module = FactorizedDeltaQ(state_dim, tuple(config['hidden_dims']))
        target_module = FactorizedDeltaQ(state_dim, tuple(config['hidden_dims']))
        model = ModuleDict({'q': module, 'target_q': target_module})
        rng = jax.random.PRNGKey(int(seed))
        rng, init_rng = jax.random.split(rng)
        params = model.init(
            init_rng,
            q=(observations, observations),
            target_q=(observations, observations),
        )['params']
        params = _replace_subtree(params, 'target_q', params['modules_q'])
        network = TrainState.create(
            model, params, tx=optax.adam(float(config['offline_learning_rate']))
        )
        return cls(
            rng=rng,
            network=network,
            delta_scale=jnp.asarray(delta_scale, dtype=jnp.float32),
            config=flax.core.FrozenDict(config),
        )


@dataclasses.dataclass(frozen=True)
class OSODecQNAgent:
    state_policy: OSOStateAgent
    online_actor: GoalConditionedActorCritic
    idm: OnlineIDMAgent
    config: dict[str, Any]
    online_steps: int = 0

    def offline_update(self, batch):
        state_policy, info = self.state_policy.offline_update(batch)
        return dataclasses.replace(self, state_policy=state_policy), info

    def online_update(self, batch, offline_batch=None):
        actor, actor_info = self.online_actor.online_update(batch)
        idm_batch = {
            'observations': batch['observations'],
            'next_observations': batch['next_observations'],
            'actions': batch['actions'],
        }
        if bool(self.config.get('native_pseudo_action', False)) and offline_batch is not None:
            pseudo_actions = actor.sample_actions(
                offline_batch['observations'],
                offline_batch['goals'],
                seed=jax.random.PRNGKey(self.online_steps),
                temperature=0.0,
            )
            idm_batch = {
                'observations': jnp.concatenate(
                    [idm_batch['observations'], offline_batch['observations']], axis=0
                ),
                'next_observations': jnp.concatenate(
                    [idm_batch['next_observations'], offline_batch['next_observations']], axis=0
                ),
                'actions': jnp.concatenate([idm_batch['actions'], pseudo_actions], axis=0),
            }
        idm, idm_info = self.idm.online_update(idm_batch)
        info = {
            **{f'online/{key}': value for key, value in actor_info.items()},
            **idm_info,
            'switch/beta': jnp.asarray(self.beta, dtype=jnp.float32),
        }
        return dataclasses.replace(
            self,
            online_actor=actor,
            idm=idm,
            online_steps=self.online_steps + 1,
        ), info

    @property
    def beta(self) -> float:
        decreases = self.online_steps // int(self.config['beta_interval'])
        return max(
            float(self.config['beta_min']),
            float(self.config['beta_initial'])
            - decreases * float(self.config['beta_decrement']),
        )

    def sample_actions(self, observations, goals, seed=None, temperature=0.0):
        if seed is None:
            seed = jax.random.PRNGKey(0)
        switch_seed, actor_seed = jax.random.split(seed)
        actor_actions = self.online_actor.sample_actions(
            observations, goals, seed=actor_seed, temperature=temperature
        )
        desired = self.state_policy.desired_next(observations, goals)
        guided_actions = self.idm.predict(observations, desired)
        use_guide = jax.random.bernoulli(
            switch_seed, p=self.beta, shape=actor_actions.shape[:-1]
        )
        return jnp.where(use_guide[..., None], guided_actions, actor_actions)

    @classmethod
    def create(cls, seed, ex_observations, action_dim, delta_scale, config):
        config = dict(config)
        state_policy = OSOStateAgent.create(
            seed, ex_observations, delta_scale, config
        )
        actor_config = gc_config('td3').to_dict()
        actor_config.update(
            learning_rate=float(config['online_learning_rate']),
            hidden_dims=tuple(config['online_hidden_dims']),
            batch_size=int(config['online_batch_size']),
        )
        actor = GoalConditionedActorCritic.create(
            seed + 1,
            ex_observations,
            jnp.zeros((len(ex_observations), int(action_dim)), dtype=jnp.float32),
            actor_config,
        )
        idm_config = {
            'hidden_dims': tuple(config['idm_hidden_dims']),
            'learning_rate': float(config['idm_learning_rate']),
            'loss': 'l1',
        }
        idm = OnlineIDMAgent.create(
            seed + 2, ex_observations, int(action_dim), idm_config
        )
        return cls(
            state_policy=state_policy,
            online_actor=actor,
            idm=idm,
            config=config,
        )


def get_config() -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict(
        dict(
            offline_learning_rate=3e-4,
            online_learning_rate=3e-4,
            idm_learning_rate=1e-3,
            hidden_dims=(512, 512, 512),
            online_hidden_dims=(256, 256),
            idm_hidden_dims=(512, 512, 512),
            discount=0.99,
            tau=0.005,
            conservative_weight=1.0,
            discretization_threshold=0.25,
            offline_batch_size=512,
            online_batch_size=512,
            beta_initial=0.5,
            beta_min=0.0,
            beta_decrement=0.1,
            beta_interval=100_000,
            collection_noise_initial=0.2,
            collection_noise_final=0.05,
            collection_noise_decay_steps=100_000,
            native_pseudo_action=False,
        )
    )


__all__ = [
    'FactorizedDeltaQ',
    'OSODecQNAgent',
    'OSOStateAgent',
    'discretize_deltas',
    'get_config',
]
