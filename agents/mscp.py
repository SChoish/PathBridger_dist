"""Goal-conditioned Multiscale State-Centric Planner baseline.

This OGBench port preserves MSCP's core decomposition: a goal-conditioned
value, a slow state planner, a pessimistically regularized one-step fast state
planner, and an action policy learned only after online interaction begins.
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
from agents.passive_hiql import TwinGoalValue
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP


class StatePlanner(nn.Module):
    state_dim: int
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations, goals):
        inputs = jnp.concatenate([observations, goals], axis=-1)
        delta = MLP((*self.hidden_dims, self.state_dim))(inputs)
        return observations + delta


class MSCPAgent(flax.struct.PyTreeNode):
    rng: Any
    network: TrainState
    config: Any = nonpytree_field()

    def _value_loss(self, batch, params):
        next_values = self.network.select('target_value')(
            batch['next_observations'], batch['goals']
        )
        target = batch['rewards'] + float(self.config['discount']) * batch['masks'] * jnp.minimum(*next_values)
        target = jax.lax.stop_gradient(target)
        v1, v2 = self.network.select('value')(
            batch['observations'], batch['goals'], params=params
        )
        expectile = float(self.config['expectile'])
        diff1, diff2 = target - v1, target - v2
        loss = jnp.mean(
            jnp.where(diff1 >= 0, expectile, 1 - expectile) * jnp.square(diff1)
            + jnp.where(diff2 >= 0, expectile, 1 - expectile) * jnp.square(diff2)
        )
        return loss, {
            'value/loss': loss,
            'value/mean': 0.5 * (v1.mean() + v2.mean()),
            'value/target_mean': target.mean(),
            'value/target_std': target.std(),
        }

    @partial(jax.jit, static_argnames=())
    def offline_update(self, batch):
        def loss_fn(params):
            value_loss, value_info = self._value_loss(batch, params)
            slow = self.network.select('slow_planner')(
                batch['observations'], batch['goals'], params=params
            )
            fast = self.network.select('fast_planner')(
                batch['observations'], batch['slow_targets'], params=params
            )
            slow_loss = jnp.mean(jnp.sum(jnp.square(slow - batch['slow_targets']), axis=-1))
            fast_error = jnp.sum(jnp.square(fast - batch['fast_targets']), axis=-1)
            predicted_values = jnp.minimum(
                *self.network.select('target_value')(fast, batch['slow_targets'])
            )
            data_values = jnp.minimum(
                *self.network.select('target_value')(batch['fast_targets'], batch['slow_targets'])
            )
            pessimism = jnp.mean(jax.nn.relu(predicted_values - data_values))
            fast_loss = fast_error.mean() + float(self.config['fast_pessimism']) * pessimism
            total = value_loss + slow_loss + fast_loss
            return total, {
                'loss/total': total,
                **value_info,
                'slow/loss': slow_loss,
                'fast/loss': fast_loss,
                'fast/pessimism': pessimism,
            }

        network, info = self.network.apply_loss_fn(
            loss_fn,
            trainable_modules=('value', 'slow_planner', 'fast_planner'),
        )
        network = self._update_target(network)
        return self.replace(network=network), info

    @partial(jax.jit, static_argnames=('tune_planners',))
    def mixed_online_update(self, online_batch, offline_batch, tune_planners=False):
        """Use a 50:50 offline/online value mixture and online actions."""

        def loss_fn(params):
            online_value, online_info = self._value_loss(online_batch, params)
            offline_value, _ = self._value_loss(offline_batch, params)
            behavior_goals = online_batch.get(
                'behavior_goals', online_batch['goals']
            )
            slow = self.network.select('slow_planner')(
                online_batch['observations'],
                behavior_goals,
                params=params if tune_planners else None,
            )
            desired_next = self.network.select('fast_planner')(
                online_batch['observations'],
                slow,
                params=params if tune_planners else None,
            )
            mean, log_std = self.network.select('low_policy')(
                online_batch['observations'], desired_next, params=params
            )
            actions = jnp.clip(online_batch['actions'], -0.999, 0.999)
            pre_tanh = jnp.arctanh(actions)
            log_prob = -0.5 * (
                jnp.square((pre_tanh - mean) / jnp.exp(log_std))
                + 2.0 * log_std
                + jnp.log(2.0 * jnp.pi)
            ) - jnp.log(jnp.maximum(1.0 - jnp.square(actions), 1e-6))
            low_loss = -jnp.sum(log_prob, axis=-1).mean()
            planner_loss = jnp.zeros((), dtype=low_loss.dtype)
            if tune_planners:
                planner_loss = jnp.mean(
                    jnp.sum(
                        jnp.square(
                            desired_next - online_batch['next_observations']
                        ),
                        axis=-1,
                    )
                )
            value_loss = 0.5 * (online_value + offline_value)
            total = value_loss + low_loss + planner_loss
            return total, {
                'loss/total': total,
                **online_info,
                'value/offline_loss': offline_value,
                'low/loss': low_loss,
                'planner/online_loss': planner_loss,
            }

        trainable_modules = ('value', 'low_policy')
        if tune_planners:
            trainable_modules += ('slow_planner', 'fast_planner')
        network, info = self.network.apply_loss_fn(
            loss_fn, trainable_modules=trainable_modules
        )
        network = self._update_target(network)
        return self.replace(network=network), info

    def online_update(self, batch):
        """Compatibility update when no offline mixture batch is supplied."""

        offline_like = {
            **batch,
            'slow_targets': batch['next_observations'],
            'fast_targets': batch['next_observations'],
        }
        return self.mixed_online_update(batch, offline_like, tune_planners=False)

    def _update_target(self, network):
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
        slow = self.network.select('slow_planner')(observations, goals)
        desired_next = self.network.select('fast_planner')(observations, slow)
        mean, log_std = self.network.select('low_policy')(observations, desired_next)
        actions, _ = _sample_tanh_gaussian(mean, log_std, seed, temperature)
        return jnp.clip(actions, -1.0, 1.0)

    @classmethod
    def create(cls, seed, ex_observations, action_dim, config):
        config = dict(config)
        observations = jnp.asarray(ex_observations, dtype=jnp.float32)
        state_dim = int(observations.shape[-1])
        hidden_dims = tuple(config['hidden_dims'])
        value_def = TwinGoalValue(hidden_dims)
        target_value_def = TwinGoalValue(hidden_dims)
        slow_def = StatePlanner(state_dim, hidden_dims)
        fast_def = StatePlanner(state_dim, hidden_dims)
        low_def = GaussianActor(int(action_dim), hidden_dims)
        model = ModuleDict(
            {
                'value': value_def,
                'target_value': target_value_def,
                'slow_planner': slow_def,
                'fast_planner': fast_def,
                'low_policy': low_def,
            }
        )
        rng = jax.random.PRNGKey(int(seed))
        rng, init_rng = jax.random.split(rng)
        params = model.init(
            init_rng,
            value=(observations, observations),
            target_value=(observations, observations),
            slow_planner=(observations, observations),
            fast_planner=(observations, observations),
            low_policy=(observations, observations),
        )['params']
        params = _replace_subtree(params, 'target_value', params['modules_value'])
        learning_rate = float(config['learning_rate'])
        network = TrainState.create(
            model,
            params,
            tx={
                name: optax.adam(learning_rate)
                for name in (
                    'value',
                    'slow_planner',
                    'fast_planner',
                    'low_policy',
                )
            },
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
            fast_pessimism=1.0,
            subgoal_steps=10,
            batch_size=1024,
            tune_planners_online=False,
        )
    )


__all__ = ['MSCPAgent', 'StatePlanner', 'get_config']
