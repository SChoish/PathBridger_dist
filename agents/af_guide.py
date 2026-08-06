"""Goal-conditioned AF-Guide OGBench port.

The offline Action-Free Decision Transformer predicts a desired next-state
delta from state history, explicit goal, and remaining horizon.  Online Guided
SAC uses separate environment and guide critics.
"""

from __future__ import annotations

from functools import partial

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.gc_actor_critic import (
    GaussianActor,
    TwinGoalCritic,
    _replace_subtree,
    _sample_tanh_gaussian,
)
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP


class TransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, inputs, attention_mask=None):
        normalized = nn.LayerNorm()(inputs)
        attended = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
            dropout_rate=0.0,
        )(normalized, mask=attention_mask)
        hidden = inputs + attended
        feed_forward = MLP((4 * self.embed_dim, self.embed_dim))(
            nn.LayerNorm()(hidden)
        )
        return hidden + feed_forward


class GoalConditionedAFDT(nn.Module):
    state_dim: int
    context_length: int = 20
    embed_dim: int = 128
    num_blocks: int = 3
    num_heads: int = 1

    @nn.compact
    def __call__(self, histories, goals, remaining, history_masks=None):
        batch_size, context_length, _ = histories.shape
        if context_length != self.context_length:
            raise ValueError(
                f'AFDT expected context {self.context_length}, got {context_length}.'
            )
        state_tokens = nn.Dense(self.embed_dim, name='state_embedding')(histories)
        goal_token = nn.Dense(self.embed_dim, name='goal_embedding')(goals)
        remaining = jnp.asarray(remaining, dtype=histories.dtype).reshape(batch_size, 1)
        horizon_token = nn.Dense(self.embed_dim, name='horizon_embedding')(
            jnp.log1p(remaining)
        )
        positions = self.param(
            'position_embedding',
            nn.initializers.normal(stddev=0.02),
            (self.context_length, self.embed_dim),
        )
        tokens = (
            state_tokens
            + goal_token[:, None, :]
            + horizon_token[:, None, :]
            + positions[None, :, :]
        )
        if history_masks is None:
            history_masks = jnp.ones(
                histories.shape[:2], dtype=histories.dtype
            )
        tokens = tokens * history_masks[..., None]
        attention_mask = nn.make_attention_mask(
            history_masks > 0,
            history_masks > 0,
        )
        for block in range(self.num_blocks):
            tokens = TransformerBlock(
                self.embed_dim, self.num_heads, name=f'block_{block}'
            )(tokens, attention_mask)
        return nn.Dense(self.state_dim, name='delta_head')(
            nn.LayerNorm()(tokens[:, -1, :])
        )


class AFGuideAgent(flax.struct.PyTreeNode):
    rng: object
    network: TrainState
    state_scale: object
    config: object = nonpytree_field()

    @partial(jax.jit, static_argnames=())
    def offline_update(self, batch):
        """Update AFDT only; no action input is accepted or referenced."""

        def loss_fn(params):
            predictions = self.network.select('afdt')(
                batch['histories'],
                batch['goals'],
                batch['remaining'],
                batch['history_masks'],
                params=params,
            )
            normalized_error = (
                predictions - batch['target_deltas']
            ) / self.state_scale
            loss = jnp.mean(jnp.sum(jnp.square(normalized_error), axis=-1))
            return loss, {
                'loss/total': loss,
                'afdt/loss': loss,
                'afdt/delta_rms': jnp.sqrt(jnp.mean(jnp.square(predictions))),
            }

        network, info = self.network.apply_loss_fn(
            loss_fn, trainable_modules=('afdt',)
        )
        return self.replace(network=network), info

    @partial(jax.jit, static_argnames=())
    def online_update(self, batch):
        (
            new_rng,
            next_env_seed,
            next_guide_seed,
            actor_env_seed,
            actor_guide_seed,
        ) = jax.random.split(self.rng, 5)

        def loss_fn(params):
            guide_valid = jnp.asarray(
                batch.get(
                    'desired_next_valid',
                    jnp.ones(batch['rewards'].shape, dtype=jnp.float32),
                ),
                dtype=jnp.float32,
            )
            valid_count = jnp.maximum(jnp.sum(guide_valid), 1.0)
            valid_fraction = jnp.mean(guide_valid)
            behavior_goals = batch.get('behavior_goals', batch['goals'])
            behavior_masks = batch.get('behavior_masks', batch['masks'])
            next_mean, next_log_std = self.network.select('actor')(
                batch['next_observations'], batch['goals']
            )
            next_actions, next_log_prob = _sample_tanh_gaussian(
                next_mean, next_log_std, next_env_seed, 1.0
            )
            target_q = jnp.minimum(
                *self.network.select('target_critic')(
                    batch['next_observations'], batch['goals'], next_actions
                )
            ) - float(self.config['entropy_coefficient']) * next_log_prob
            env_target = jax.lax.stop_gradient(
                batch['rewards']
                + float(self.config['discount']) * batch['masks'] * target_q
            )
            guide_reward = -jnp.sqrt(
                jnp.sum(
                    jnp.square(
                        (batch['next_observations'] - batch['desired_next'])
                        / self.state_scale
                    ),
                    axis=-1,
                )
                + 1e-6
            )
            next_guide_mean, next_guide_log_std = self.network.select('actor')(
                batch['next_observations'], behavior_goals
            )
            next_guide_actions, _ = _sample_tanh_gaussian(
                next_guide_mean,
                next_guide_log_std,
                next_guide_seed,
                1.0,
            )
            target_guide = jnp.minimum(
                *self.network.select('target_guide_critic')(
                    batch['next_observations'],
                    behavior_goals,
                    next_guide_actions,
                )
            )
            guide_target = jax.lax.stop_gradient(
                guide_reward
                + float(self.config['discount'])
                * behavior_masks
                * target_guide
            )
            q1, q2 = self.network.select('critic')(
                batch['observations'], batch['goals'], batch['actions'], params=params
            )
            g1, g2 = self.network.select('guide_critic')(
                batch['observations'],
                behavior_goals,
                batch['actions'],
                params=params,
            )
            env_critic_loss = jnp.mean(
                jnp.square(q1 - env_target) + jnp.square(q2 - env_target)
            )
            guide_critic_loss = jnp.sum(
                guide_valid
                * (jnp.square(g1 - guide_target) + jnp.square(g2 - guide_target))
            ) / valid_count
            critic_loss = env_critic_loss + guide_critic_loss
            mean, log_std = self.network.select('actor')(
                batch['observations'], batch['goals'], params=params
            )
            actions, log_prob = _sample_tanh_gaussian(
                mean, log_std, actor_env_seed, 1.0
            )
            actor_q = jnp.minimum(
                *self.network.select('critic')(
                    batch['observations'], batch['goals'], actions
                )
            )
            guide_mean, guide_log_std = self.network.select('actor')(
                batch['observations'], behavior_goals, params=params
            )
            guide_actions, _ = _sample_tanh_gaussian(
                guide_mean,
                guide_log_std,
                actor_guide_seed,
                1.0,
            )
            actor_guide = jnp.minimum(
                *self.network.select('guide_critic')(
                    batch['observations'], behavior_goals, guide_actions
                )
            )
            env_actor_loss = jnp.mean(
                float(self.config['entropy_coefficient']) * log_prob - actor_q
            )
            guide_actor_loss = -float(self.config['guide_weight']) * jnp.sum(
                guide_valid * actor_guide
            ) / valid_count
            actor_loss = env_actor_loss + guide_actor_loss
            total = critic_loss + actor_loss
            return total, {
                'loss/total': total,
                'critic/loss': critic_loss,
                'critic/q_mean': 0.5 * (q1.mean() + q2.mean()),
                'critic/q_std': jnp.stack([q1, q2]).std(),
                'actor/loss': actor_loss,
                'guide/actor_loss': guide_actor_loss,
                'guide/reward': jnp.sum(guide_valid * guide_reward) / valid_count,
                'guide/q_mean': 0.5 * (g1.mean() + g2.mean()),
                'guide/critic_loss': guide_critic_loss,
                'guide/valid_target_fraction': valid_fraction,
            }

        network, info = self.network.apply_loss_fn(
            loss_fn,
            trainable_modules=('actor', 'critic', 'guide_critic'),
        )
        network = self._update_targets(network)
        return self.replace(rng=new_rng, network=network), info

    def _update_targets(self, network):
        tau = float(self.config['tau'])
        for source, target in (
            ('critic', 'target_critic'),
            ('guide_critic', 'target_guide_critic'),
        ):
            updated = jax.tree_util.tree_map(
                lambda value, old: tau * value + (1.0 - tau) * old,
                network.params[f'modules_{source}'],
                network.params[f'modules_{target}'],
            )
            network = network.replace(
                params=_replace_subtree(network.params, target, updated)
            )
        return network

    @partial(jax.jit, static_argnames=())
    def plan_next(self, histories, goals, remaining, history_masks=None):
        if history_masks is None:
            history_masks = jnp.ones(histories.shape[:2], dtype=histories.dtype)
        delta = self.network.select('afdt')(
            histories, goals, remaining, history_masks
        )
        return histories[:, -1, :] + delta

    @partial(jax.jit, static_argnames=('temperature',))
    def sample_actions(self, observations, goals, seed=None, temperature=0.0):
        if seed is None:
            seed = jax.random.PRNGKey(0)
        mean, log_std = self.network.select('actor')(observations, goals)
        actions, _ = _sample_tanh_gaussian(mean, log_std, seed, temperature)
        return jnp.clip(actions, -1.0, 1.0)

    @classmethod
    def create(cls, seed, ex_observations, action_dim, state_scale, config):
        config = dict(config)
        observations = jnp.asarray(ex_observations, dtype=jnp.float32)
        batch_size, state_dim = observations.shape
        context_length = int(config['context_length'])
        histories = jnp.broadcast_to(
            observations[:, None, :], (batch_size, context_length, state_dim)
        )
        masks = jnp.ones((batch_size, context_length), dtype=jnp.float32)
        remaining = jnp.ones((batch_size,), dtype=jnp.float32)
        afdt = GoalConditionedAFDT(
            state_dim=state_dim,
            context_length=context_length,
            embed_dim=int(config['embed_dim']),
            num_blocks=int(config['num_blocks']),
            num_heads=int(config['num_heads']),
        )
        actor = GaussianActor(int(action_dim), tuple(config['hidden_dims']))
        critic = TwinGoalCritic(tuple(config['hidden_dims']))
        target_critic = TwinGoalCritic(tuple(config['hidden_dims']))
        guide_critic = TwinGoalCritic(tuple(config['hidden_dims']))
        target_guide_critic = TwinGoalCritic(tuple(config['hidden_dims']))
        actions = jnp.zeros((batch_size, int(action_dim)), dtype=jnp.float32)
        model = ModuleDict(
            {
                'afdt': afdt,
                'actor': actor,
                'critic': critic,
                'target_critic': target_critic,
                'guide_critic': guide_critic,
                'target_guide_critic': target_guide_critic,
            }
        )
        rng = jax.random.PRNGKey(int(seed))
        rng, init_rng = jax.random.split(rng)
        params = model.init(
            init_rng,
            afdt=(histories, observations, remaining, masks),
            actor=(observations, observations),
            critic=(observations, observations, actions),
            target_critic=(observations, observations, actions),
            guide_critic=(observations, observations, actions),
            target_guide_critic=(observations, observations, actions),
        )['params']
        params = _replace_subtree(params, 'target_critic', params['modules_critic'])
        params = _replace_subtree(
            params, 'target_guide_critic', params['modules_guide_critic']
        )
        learning_rate = float(config['learning_rate'])
        network = TrainState.create(
            model,
            params,
            tx={
                name: optax.adam(learning_rate)
                for name in ('afdt', 'actor', 'critic', 'guide_critic')
            },
        )
        scale = jnp.asarray(state_scale, dtype=jnp.float32)
        return cls(
            rng=rng,
            network=network,
            state_scale=scale,
            config=flax.core.FrozenDict(config),
        )


def get_config() -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict(
        dict(
            learning_rate=1e-4,
            context_length=20,
            num_blocks=3,
            embed_dim=128,
            num_heads=1,
            hidden_dims=(256, 256),
            discount=0.99,
            tau=0.005,
            entropy_coefficient=0.1,
            guide_weight=1.0,
            batch_size=64,
        )
    )


__all__ = ['AFGuideAgent', 'GoalConditionedAFDT', 'get_config']
