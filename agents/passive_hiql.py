"""Passive-HIQL online baseline for action-free offline trajectories.

Offline updates touch only the goal-conditioned value and high-level state
policy.  The low-level action policy is initialized but receives gradients only
from real online ``(s, a, s')`` transitions.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.gc_actor_critic import GaussianActor, _replace_subtree, _sample_tanh_gaussian
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP


class TwinGoalValue(nn.Module):
    hidden_dims: Sequence[int] = (512, 512, 512)

    @nn.compact
    def __call__(self, observations: jnp.ndarray, goals: jnp.ndarray):
        inputs = jnp.concatenate([observations, goals], axis=-1)
        v1 = MLP((*self.hidden_dims, 1), name='v1')(inputs).squeeze(-1)
        v2 = MLP((*self.hidden_dims, 1), name='v2')(inputs).squeeze(-1)
        return v1, v2


class HighStatePolicy(nn.Module):
    state_dim: int
    hidden_dims: Sequence[int] = (512, 512, 512)

    @nn.compact
    def __call__(self, observations: jnp.ndarray, goals: jnp.ndarray):
        inputs = jnp.concatenate([observations, goals], axis=-1)
        delta = MLP((*self.hidden_dims, self.state_dim))(inputs)
        return observations + delta


class PassiveHIQLAgent(flax.struct.PyTreeNode):
    rng: Any
    network: TrainState
    config: Any = nonpytree_field()

    @staticmethod
    def _expectile(diff: jnp.ndarray, expectile: float) -> jnp.ndarray:
        weights = jnp.where(diff >= 0.0, expectile, 1.0 - expectile)
        return weights * jnp.square(diff)

    def _value_loss(self, batch, grad_params):
        next_v1, next_v2 = self.network.select('target_value')(
            batch['next_observations'], batch['goals']
        )
        target = batch['rewards'] + float(self.config['discount']) * batch['masks'] * jnp.minimum(next_v1, next_v2)
        target = jax.lax.stop_gradient(target)
        v1, v2 = self.network.select('value')(
            batch['observations'], batch['goals'], params=grad_params
        )
        loss = jnp.mean(
            self._expectile(target - v1, float(self.config['expectile']))
            + self._expectile(target - v2, float(self.config['expectile']))
        )
        return loss, {'value/loss': loss, 'value/mean': 0.5 * (v1.mean() + v2.mean())}

    @partial(jax.jit, static_argnames=())
    def offline_update(self, batch):
        """Train value and state-as-action high policy without actions."""

        new_rng, _ = jax.random.split(self.rng)

        def loss_fn(params):
            value_loss, info = self._value_loss(batch, params)
            predicted = self.network.select('high_policy')(
                batch['observations'], batch['goals'], params=params
            )
            current_v = jnp.mean(
                jnp.stack(self.network.select('target_value')(batch['observations'], batch['goals'])), axis=0
            )
            target_v = jnp.mean(
                jnp.stack(self.network.select('target_value')(batch['slow_targets'], batch['goals'])), axis=0
            )
            weights = jnp.minimum(
                jnp.exp(float(self.config['high_alpha']) * (target_v - current_v)),
                100.0,
            )
            high_error = jnp.sum(jnp.square(predicted - batch['slow_targets']), axis=-1)
            high_loss = jnp.mean(jax.lax.stop_gradient(weights) * high_error)
            total = value_loss + high_loss
            return total, {
                'loss/total': total,
                **info,
                'high/loss': high_loss,
                'high/weight_mean': weights.mean(),
            }

        network, info = self.network.apply_loss_fn(loss_fn)
        network = self._update_target(network)
        return self.replace(rng=new_rng, network=network), info

    @partial(jax.jit, static_argnames=())
    def online_update(self, batch):
        """Tune value and learn the low-level action policy online."""

        new_rng, action_seed = jax.random.split(self.rng)

        def loss_fn(params):
            value_loss, info = self._value_loss(batch, params)
            subgoals = self.network.select('high_policy')(
                batch['observations'], batch['goals']
            )
            mean, log_std = self.network.select('low_policy')(
                batch['observations'], subgoals, params=params
            )
            clipped_actions = jnp.clip(batch['actions'], -0.999, 0.999)
            pre_tanh = jnp.arctanh(clipped_actions)
            log_prob = -0.5 * (
                jnp.square((pre_tanh - mean) / jnp.exp(log_std))
                + 2.0 * log_std
                + jnp.log(2.0 * jnp.pi)
            ) - jnp.log(jnp.maximum(1.0 - jnp.square(clipped_actions), 1e-6))
            log_prob = jnp.sum(log_prob, axis=-1)
            current_v = jnp.mean(
                jnp.stack(self.network.select('target_value')(batch['observations'], batch['goals'])), axis=0
            )
            next_v = jnp.mean(
                jnp.stack(self.network.select('target_value')(batch['next_observations'], batch['goals'])), axis=0
            )
            weights = jnp.minimum(
                jnp.exp(float(self.config['low_alpha']) * (next_v - current_v)),
                100.0,
            )
            low_loss = -jnp.mean(jax.lax.stop_gradient(weights) * log_prob)
            total = value_loss + low_loss
            return total, {
                'loss/total': total,
                **info,
                'low/loss': low_loss,
                'low/log_prob': log_prob.mean(),
                'low/weight_mean': weights.mean(),
            }

        network, info = self.network.apply_loss_fn(loss_fn)
        network = self._update_target(network)
        return self.replace(rng=new_rng, network=network), info

    def _update_target(self, network: TrainState) -> TrainState:
        tau = float(self.config['tau'])
        target = jax.tree_util.tree_map(
            lambda value, old: tau * value + (1.0 - tau) * old,
            network.params['modules_value'],
            network.params['modules_target_value'],
        )
        return network.replace(params=_replace_subtree(network.params, 'target_value', target))

    @partial(jax.jit, static_argnames=('temperature',))
    def sample_actions(self, observations, goals, seed=None, temperature=0.0):
        if seed is None:
            seed = jax.random.PRNGKey(0)
        subgoals = self.network.select('high_policy')(observations, goals)
        mean, log_std = self.network.select('low_policy')(observations, subgoals)
        actions, _ = _sample_tanh_gaussian(mean, log_std, seed, temperature)
        return jnp.clip(actions, -1.0, 1.0)

    @classmethod
    def create(cls, seed, ex_observations, action_dim, config):
        config = dict(config)
        observations = jnp.asarray(ex_observations, dtype=jnp.float32)
        goals = observations
        state_dim = int(observations.shape[-1])
        hidden_dims = tuple(config['hidden_dims'])
        value_def = TwinGoalValue(hidden_dims)
        target_value_def = TwinGoalValue(hidden_dims)
        high_def = HighStatePolicy(state_dim, hidden_dims)
        low_def = GaussianActor(int(action_dim), hidden_dims)
        model = ModuleDict(
            {
                'value': value_def,
                'target_value': target_value_def,
                'high_policy': high_def,
                'low_policy': low_def,
            }
        )
        rng = jax.random.PRNGKey(int(seed))
        rng, init_rng = jax.random.split(rng)
        params = model.init(
            init_rng,
            value=(observations, goals),
            target_value=(observations, goals),
            high_policy=(observations, goals),
            low_policy=(observations, goals),
        )['params']
        params = _replace_subtree(params, 'target_value', params['modules_value'])
        network = TrainState.create(
            model, params, tx=optax.adam(float(config['learning_rate']))
        )
        return cls(rng=rng, network=network, config=flax.core.FrozenDict(config))


def get_config() -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict(
        dict(
            learning_rate=3e-4,
            hidden_dims=(512, 512, 512),
            discount=0.99,
            tau=0.005,
            expectile=0.7,
            high_alpha=3.0,
            low_alpha=3.0,
            subgoal_steps=10,
            batch_size=1024,
        )
    )


__all__ = ['PassiveHIQLAgent', 'get_config']
