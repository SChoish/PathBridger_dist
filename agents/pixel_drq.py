"""Goal-conditioned visual control and action-free pixel pretraining ports.

This module provides one shared DrQ-v2-style online controller and two explicit
OGBench adaptations:

* VIP-style temporal value pretraining from action-free RGB trajectories;
* APV-style latent video prediction followed by online action-conditioned
  latent dynamics.

They are controlled ports, not exact reproductions of the original task suites.
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

from agents.gc_actor_critic import _replace_subtree
from agents.pixel_lapo import PixelDecoder
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, default_init


def random_shift(images: jnp.ndarray, seed: jax.Array, padding: int = 4):
    """Apply independent edge-padded random translations to NHWC images."""

    padding = int(padding)
    if padding <= 0:
        return images
    height, width, channels = images.shape[1:]
    keys = jax.random.split(seed, images.shape[0])

    def crop(image, key):
        padded = jnp.pad(
            image,
            ((padding, padding), (padding, padding), (0, 0)),
            mode='edge',
        )
        offset = jax.random.randint(
            key, (2,), minval=0, maxval=2 * padding + 1
        )
        return jax.lax.dynamic_slice(
            padded,
            (offset[0], offset[1], 0),
            (height, width, channels),
        )

    return jax.vmap(crop)(images, keys)


class DrQEncoder(nn.Module):
    feature_dim: int = 128
    channels: Sequence[int] = (32, 32, 32, 32)

    @nn.compact
    def __call__(self, images):
        hidden = jnp.asarray(images, dtype=jnp.float32) / 255.0 - 0.5
        for index, channels in enumerate(self.channels):
            hidden = nn.Conv(
                int(channels),
                kernel_size=(3, 3),
                strides=(2, 2),
                padding='SAME',
                name=f'conv_{index}',
            )(hidden)
            hidden = nn.relu(hidden)
        hidden = hidden.reshape((hidden.shape[0], -1))
        hidden = nn.Dense(self.feature_dim, name='projection')(hidden)
        return nn.LayerNorm(name='feature_norm')(hidden)


class PixelDeterministicActor(nn.Module):
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, observation_features, goal_features):
        inputs = jnp.concatenate([observation_features, goal_features], axis=-1)
        return jnp.tanh(
            MLP(
                (*self.hidden_dims, self.action_dim),
                activate_final=False,
                kernel_init=default_init(0.01),
            )(inputs)
        )


class PixelTwinCritic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, observation_features, goal_features, actions):
        inputs = jnp.concatenate(
            [observation_features, goal_features, actions], axis=-1
        )
        q1 = MLP((*self.hidden_dims, 1), name='q1')(inputs).squeeze(-1)
        q2 = MLP((*self.hidden_dims, 1), name='q2')(inputs).squeeze(-1)
        return q1, q2


class LatentVideoPredictor(nn.Module):
    feature_dim: int
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, features):
        return MLP((*self.hidden_dims, self.feature_dim), name='predictor')(
            features
        )


class ActionConditionedDynamics(nn.Module):
    feature_dim: int
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, features, actions):
        return MLP((*self.hidden_dims, self.feature_dim), name='dynamics')(
            jnp.concatenate([features, actions], axis=-1)
        )


class PixelDrQAgent(flax.struct.PyTreeNode):
    """DrQ-v2-style controller with optional VIP/APV action-free pretraining."""

    rng: Any
    network: TrainState
    config: Any = nonpytree_field()

    def _target_update(self, network, pairs):
        tau = float(self.config['tau'])
        for source, target in pairs:
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
    def offline_update(self, batch):
        """Update only the declared action-free visual pretraining modules."""

        pretraining = str(self.config['pretraining'])
        if pretraining == 'none':
            zero = jnp.zeros((), dtype=jnp.float32)
            return self, {'loss/total': zero, 'offline/skipped': jnp.ones(())}

        if pretraining == 'vip':
            def loss_fn(params):
                current = self.network.select('encoder')(
                    batch['observations'], params=params
                )
                next_features = self.network.select('target_encoder')(
                    batch['next_observations']
                )
                goal_features = self.network.select('target_encoder')(
                    batch['goals']
                )
                current_value = -jnp.linalg.norm(
                    current - goal_features, axis=-1
                )
                next_value = -jnp.linalg.norm(
                    next_features - goal_features, axis=-1
                )
                target = batch['rewards'] + float(
                    self.config['vip_discount']
                ) * batch['masks'] * next_value
                value_loss = jnp.mean(
                    jnp.square(current_value - jax.lax.stop_gradient(target))
                )
                anchor_loss = jnp.sum(
                    (1.0 - batch['masks']) * jnp.square(current_value)
                ) / jnp.maximum(jnp.sum(1.0 - batch['masks']), 1.0)
                loss = value_loss + float(
                    self.config['vip_anchor_weight']
                ) * anchor_loss
                return loss, {
                    'loss/total': loss,
                    'vip/value_loss': value_loss,
                    'vip/anchor_loss': anchor_loss,
                    'vip/value_mean': jnp.mean(current_value),
                }

            network, info = self.network.apply_loss_fn(
                loss_fn, trainable_modules=('encoder',)
            )
        elif pretraining == 'apv':
            def loss_fn(params):
                features = self.network.select('encoder')(
                    batch['observations'], params=params
                )
                target_features = self.network.select('target_encoder')(
                    batch['next_observations']
                )
                predicted = self.network.select('video_predictor')(
                    features, params=params
                )
                reconstruction = self.network.select('world_decoder')(
                    predicted, params=params
                )
                feature_loss = jnp.mean(
                    jnp.square(predicted - jax.lax.stop_gradient(target_features))
                )
                target_pixels = jnp.asarray(
                    batch['next_observations'], dtype=jnp.float32
                ) / 255.0
                reconstruction_loss = jnp.mean(
                    jnp.square(reconstruction - target_pixels)
                )
                loss = feature_loss + float(
                    self.config['reconstruction_weight']
                ) * reconstruction_loss
                return loss, {
                    'loss/total': loss,
                    'apv/feature_loss': feature_loss,
                    'apv/reconstruction_loss': reconstruction_loss,
                }

            network, info = self.network.apply_loss_fn(
                loss_fn,
                trainable_modules=('encoder', 'video_predictor', 'world_decoder'),
            )
        else:
            raise ValueError(f'Unknown pixel pretraining mode {pretraining!r}.')

        network = self._target_update(network, (('encoder', 'target_encoder'),))
        return self.replace(network=network), info

    def online_update(self, batch):
        update_actor = int(self.network.step) % int(self.config['policy_delay']) == 0
        return self._online_update(batch, update_actor=update_actor)

    @partial(jax.jit, static_argnames=('update_actor',))
    def _online_update(self, batch, *, update_actor: bool):
        new_rng, obs_seed, next_seed, goal_seed, target_seed, actor_seed = (
            jax.random.split(self.rng, 6)
        )
        padding = int(self.config['augmentation_padding'])
        observations = random_shift(batch['observations'], obs_seed, padding)
        next_observations = random_shift(
            batch['next_observations'], next_seed, padding
        )
        goals = random_shift(batch['goals'], goal_seed, padding)
        train_encoder = not bool(self.config['freeze_encoder_online'])
        pretraining = str(self.config['pretraining'])

        def loss_fn(params):
            encoder_params = params if train_encoder else None
            features = self.network.select('encoder')(
                observations, params=encoder_params
            )
            goal_features = self.network.select('encoder')(
                goals, params=encoder_params
            )
            target_next = self.network.select('target_encoder')(
                next_observations
            )
            target_goal = self.network.select('target_encoder')(goals)
            target_actions = self.network.select('target_actor')(
                target_next, target_goal
            )
            noise = jnp.clip(
                jax.random.normal(target_seed, target_actions.shape)
                * float(self.config['target_noise']),
                -float(self.config['target_noise_clip']),
                float(self.config['target_noise_clip']),
            )
            target_actions = jnp.clip(target_actions + noise, -1.0, 1.0)
            target_q1, target_q2 = self.network.select('target_critic')(
                target_next, target_goal, target_actions
            )
            target = batch['rewards'] + float(self.config['discount']) * batch[
                'masks'
            ] * jnp.minimum(target_q1, target_q2)
            q1, q2 = self.network.select('critic')(
                features,
                goal_features,
                batch['actions'],
                params=params,
            )
            critic_loss = jnp.mean(
                jnp.square(q1 - jax.lax.stop_gradient(target))
                + jnp.square(q2 - jax.lax.stop_gradient(target))
            )
            auxiliary_loss = jnp.zeros((), dtype=critic_loss.dtype)
            if pretraining == 'apv':
                predicted_next = self.network.select('action_dynamics')(
                    features, batch['actions'], params=params
                )
                auxiliary_loss = jnp.mean(
                    jnp.square(
                        predicted_next - jax.lax.stop_gradient(target_next)
                    )
                )
            if update_actor:
                actor_features = jax.lax.stop_gradient(
                    self.network.select('encoder')(observations)
                )
                actor_goals = jax.lax.stop_gradient(
                    self.network.select('encoder')(goals)
                )
                actions = self.network.select('actor')(
                    actor_features, actor_goals, params=params
                )
                actor_q1, _ = self.network.select('critic')(
                    actor_features, actor_goals, actions
                )
                actor_loss = -jnp.mean(actor_q1)
            else:
                actor_loss = jnp.zeros((), dtype=critic_loss.dtype)
            total = (
                critic_loss
                + actor_loss
                + float(self.config['action_dynamics_weight']) * auxiliary_loss
            )
            return total, {
                'loss/total': total,
                'critic/loss': critic_loss,
                'critic/q_mean': 0.5 * (jnp.mean(q1) + jnp.mean(q2)),
                'actor/loss': actor_loss,
                'actor/updated': jnp.asarray(update_actor, jnp.float32),
                'apv/action_dynamics_loss': auxiliary_loss,
            }

        trainable = ['critic']
        if train_encoder:
            trainable.append('encoder')
        if pretraining == 'apv':
            trainable.append('action_dynamics')
        if update_actor:
            trainable.append('actor')
        network, info = self.network.apply_loss_fn(
            loss_fn, trainable_modules=tuple(trainable)
        )
        targets = [('critic', 'target_critic')]
        if train_encoder:
            targets.append(('encoder', 'target_encoder'))
        if update_actor:
            targets.append(('actor', 'target_actor'))
        network = self._target_update(network, tuple(targets))
        return self.replace(rng=new_rng, network=network), info

    @partial(jax.jit, static_argnames=('temperature',))
    def sample_actions(self, observations, goals, seed=None, temperature=0.0):
        if seed is None:
            seed = jax.random.PRNGKey(0)
        features = self.network.select('encoder')(observations)
        goal_features = self.network.select('encoder')(goals)
        actions = self.network.select('actor')(features, goal_features)
        if temperature > 0.0:
            actions = actions + float(temperature) * float(
                self.config['exploration_std']
            ) * jax.random.normal(seed, actions.shape)
        return jnp.clip(actions, -1.0, 1.0)

    @classmethod
    def create(cls, seed, example_images, action_dim, config):
        config = dict(config)
        images = jnp.asarray(example_images, dtype=jnp.uint8)
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError('Pixel DrQ examples must have shape [B, H, W, 3].')
        image_shape = tuple(int(value) for value in images.shape[1:])
        feature_dim = int(config['feature_dim'])
        hidden_dims = tuple(config['hidden_dims'])
        encoder = DrQEncoder(feature_dim)
        actor = PixelDeterministicActor(int(action_dim), hidden_dims)
        critic = PixelTwinCritic(hidden_dims)
        video_predictor = LatentVideoPredictor(feature_dim, hidden_dims)
        action_dynamics = ActionConditionedDynamics(feature_dim, hidden_dims)
        world_decoder = PixelDecoder(image_shape)
        model = ModuleDict(
            {
                'encoder': encoder,
                'target_encoder': encoder,
                'actor': actor,
                'target_actor': actor,
                'critic': critic,
                'target_critic': critic,
                'video_predictor': video_predictor,
                'action_dynamics': action_dynamics,
                'world_decoder': world_decoder,
            }
        )
        features = jnp.zeros((len(images), feature_dim), jnp.float32)
        actions = jnp.zeros((len(images), int(action_dim)), jnp.float32)
        rng = jax.random.PRNGKey(int(seed))
        rng, init_rng = jax.random.split(rng)
        params = model.init(
            init_rng,
            encoder=(images,),
            target_encoder=(images,),
            actor=(features, features),
            target_actor=(features, features),
            critic=(features, features, actions),
            target_critic=(features, features, actions),
            video_predictor=(features,),
            action_dynamics=(features, actions),
            world_decoder=(features,),
        )['params']
        for source, target in (
            ('encoder', 'target_encoder'),
            ('actor', 'target_actor'),
            ('critic', 'target_critic'),
        ):
            params = _replace_subtree(
                params, target, params[f'modules_{source}']
            )
        network = TrainState.create(
            model,
            params,
            tx={
                'encoder': optax.adam(float(config['encoder_learning_rate'])),
                'actor': optax.adam(float(config['learning_rate'])),
                'critic': optax.adam(float(config['learning_rate'])),
                'video_predictor': optax.adam(
                    float(config['encoder_learning_rate'])
                ),
                'action_dynamics': optax.adam(float(config['learning_rate'])),
                'world_decoder': optax.adam(
                    float(config['encoder_learning_rate'])
                ),
            },
        )
        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(config),
        )


def get_config(pretraining='none', *, freeze_encoder_online=False):
    return ml_collections.ConfigDict(
        dict(
            pretraining=str(pretraining),
            freeze_encoder_online=bool(freeze_encoder_online),
            feature_dim=128,
            hidden_dims=(256, 256),
            learning_rate=3e-4,
            encoder_learning_rate=1e-4,
            discount=0.99,
            tau=0.01,
            policy_delay=2,
            target_noise=0.2,
            target_noise_clip=0.3,
            exploration_std=0.1,
            augmentation_padding=4,
            offline_batch_size=128,
            online_batch_size=256,
            offline_steps=100_000 if pretraining != 'none' else 0,
            vip_discount=0.98,
            vip_anchor_weight=1.0,
            reconstruction_weight=1.0,
            action_dynamics_weight=1.0 if pretraining == 'apv' else 0.0,
        )
    )


__all__ = [
    'DrQEncoder',
    'PixelDrQAgent',
    'PixelDeterministicActor',
    'PixelTwinCritic',
    'get_config',
    'random_shift',
]
