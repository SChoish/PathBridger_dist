"""Latent visual PathBridger with frozen offline paths and online-only IDM."""

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
from agents.pixel_drq import DrQEncoder
from agents.pixel_lapo import PixelDecoder
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, default_init


class LatentPathBridge(nn.Module):
    feature_dim: int
    path_horizon: int = 5
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, current_features, goal_features):
        residual = MLP(
            (*self.hidden_dims, self.path_horizon * self.feature_dim),
            activate_final=False,
            name='residual_path',
        )(jnp.concatenate([current_features, goal_features], axis=-1))
        residual = residual.reshape(
            (-1, self.path_horizon, self.feature_dim)
        )
        fractions = jnp.linspace(
            1.0 / self.path_horizon, 1.0, self.path_horizon
        )[None, :, None]
        base = (
            current_features[:, None, :] * (1.0 - fractions)
            + goal_features[:, None, :] * fractions
        )
        # Endpoint pinning preserves the goal while the residual models the
        # non-linear intermediate path.
        return base + residual * (1.0 - fractions)


class LatentInverseDynamics(nn.Module):
    action_dim: int
    hidden_dims: Sequence[int] = (512, 512, 512)

    @nn.compact
    def __call__(self, current_features, desired_next_features):
        inputs = jnp.concatenate(
            [current_features, desired_next_features], axis=-1
        )
        return jnp.tanh(
            MLP(
                (*self.hidden_dims, self.action_dim),
                activate_final=False,
                kernel_init=default_init(0.01),
                name='idm',
            )(inputs)
        )


class PixelPathBridgerAgent(flax.struct.PyTreeNode):
    """Action-free latent path learner with a separately optimized online IDM."""

    rng: Any
    network: TrainState
    config: Any = nonpytree_field()

    def _update_target_encoder(self, network):
        tau = float(self.config['tau'])
        target = jax.tree_util.tree_map(
            lambda value, old: tau * value + (1.0 - tau) * old,
            network.params['modules_encoder'],
            network.params['modules_target_encoder'],
        )
        return network.replace(
            params=_replace_subtree(network.params, 'target_encoder', target)
        )

    @partial(jax.jit, static_argnames=())
    def offline_update(self, batch):
        path_horizon = int(self.config['path_horizon'])

        def loss_fn(params):
            current = self.network.select('encoder')(
                batch['observations'], params=params
            )
            goal = self.network.select('encoder')(
                batch['goals'], params=params
            )
            predicted_path = self.network.select('bridge')(
                current, goal, params=params
            )
            path_images = batch['path_observations']
            flat_images = path_images.reshape((-1, *path_images.shape[2:]))
            target_path = self.network.select('target_encoder')(
                flat_images
            ).reshape((-1, path_horizon, current.shape[-1]))
            feature_loss = jnp.mean(
                jnp.square(
                    predicted_path - jax.lax.stop_gradient(target_path)
                )
            )
            reconstructed = self.network.select('world_decoder')(
                predicted_path.reshape((-1, current.shape[-1])),
                params=params,
            ).reshape(path_images.shape)
            reconstruction_loss = jnp.mean(
                jnp.square(
                    reconstructed
                    - jnp.asarray(path_images, dtype=jnp.float32) / 255.0
                )
            )
            endpoint_loss = jnp.mean(
                jnp.square(predicted_path[:, -1] - goal)
            )
            feature_std = jnp.std(current, axis=0)
            variance_loss = jnp.mean(jnp.square(jax.nn.relu(1.0 - feature_std)))
            total = (
                feature_loss
                + float(self.config['reconstruction_weight'])
                * reconstruction_loss
                + float(self.config['endpoint_weight']) * endpoint_loss
                + float(self.config['variance_weight']) * variance_loss
            )
            return total, {
                'loss/total': total,
                'path/feature_loss': feature_loss,
                'path/reconstruction_loss': reconstruction_loss,
                'path/endpoint_loss': endpoint_loss,
                'path/variance_loss': variance_loss,
            }

        network, info = self.network.apply_loss_fn(
            loss_fn,
            trainable_modules=('encoder', 'bridge', 'world_decoder'),
        )
        network = self._update_target_encoder(network)
        return self.replace(network=network), info

    @partial(jax.jit, static_argnames=())
    def online_update(self, batch):
        """Ground actual online transitions without updating the visual path prior."""

        def loss_fn(params):
            current = self.network.select('encoder')(batch['observations'])
            following = self.network.select('encoder')(
                batch['next_observations']
            )
            predicted = self.network.select('idm')(
                current, following, params=params
            )
            errors = predicted - batch['actions']
            loss = jnp.mean(jnp.sum(jnp.abs(errors), axis=-1))
            return loss, {
                'loss/total': loss,
                'idm/action_l1': jnp.mean(jnp.abs(errors)),
                'idm/action_mse': jnp.mean(jnp.square(errors)),
                'idm/action_abs_mean': jnp.mean(jnp.abs(predicted)),
            }

        network, info = self.network.apply_loss_fn(
            loss_fn, trainable_modules=('idm',)
        )
        return self.replace(network=network), info

    @partial(jax.jit, static_argnames=())
    def latent_path(self, observations, goals):
        current = self.network.select('encoder')(observations)
        goal = self.network.select('encoder')(goals)
        return self.network.select('bridge')(current, goal)

    @partial(jax.jit, static_argnames=('temperature',))
    def sample_actions(self, observations, goals, seed=None, temperature=0.0):
        if seed is None:
            seed = jax.random.PRNGKey(0)
        current = self.network.select('encoder')(observations)
        goal = self.network.select('encoder')(goals)
        path = self.network.select('bridge')(current, goal)
        actions = self.network.select('idm')(current, path[:, 0])
        if temperature > 0.0:
            actions = actions + float(temperature) * float(
                self.config['exploration_std']
            ) * jax.random.normal(seed, actions.shape)
        return jnp.clip(actions, -1.0, 1.0)

    @classmethod
    def create(cls, seed, example_images, action_dim, config):
        config = dict(config)
        images = jnp.asarray(example_images, dtype=jnp.uint8)
        expected_channels = 3 * int(config['frame_stack'])
        if images.ndim != 4 or images.shape[-1] != expected_channels:
            raise ValueError(
                'Pixel PathBridger examples must be [B, H, W, '
                f'{expected_channels}] for frame_stack={config["frame_stack"]}.'
            )
        feature_dim = int(config['feature_dim'])
        hidden_dims = tuple(config['hidden_dims'])
        encoder = DrQEncoder(feature_dim)
        bridge = LatentPathBridge(
            feature_dim=feature_dim,
            path_horizon=int(config['path_horizon']),
            hidden_dims=hidden_dims,
        )
        world_decoder = PixelDecoder(tuple(int(v) for v in images.shape[1:]))
        idm = LatentInverseDynamics(
            int(action_dim), tuple(config['idm_hidden_dims'])
        )
        model = ModuleDict(
            {
                'encoder': encoder,
                'target_encoder': encoder,
                'bridge': bridge,
                'world_decoder': world_decoder,
                'idm': idm,
            }
        )
        features = jnp.zeros((len(images), feature_dim), jnp.float32)
        rng = jax.random.PRNGKey(int(seed))
        rng, init_rng = jax.random.split(rng)
        params = model.init(
            init_rng,
            encoder=(images,),
            target_encoder=(images,),
            bridge=(features, features),
            world_decoder=(features,),
            idm=(features, features),
        )['params']
        params = _replace_subtree(
            params, 'target_encoder', params['modules_encoder']
        )
        network = TrainState.create(
            model,
            params,
            tx={
                'encoder': optax.adam(float(config['encoder_learning_rate'])),
                'bridge': optax.adam(float(config['learning_rate'])),
                'world_decoder': optax.adam(
                    float(config['encoder_learning_rate'])
                ),
                'idm': optax.adam(float(config['idm_learning_rate'])),
            },
        )
        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(config),
        )


def get_config():
    return ml_collections.ConfigDict(
        dict(
            frame_stack=3,
            feature_dim=128,
            hidden_dims=(256, 256),
            idm_hidden_dims=(512, 512, 512),
            path_horizon=5,
            learning_rate=3e-4,
            encoder_learning_rate=1e-4,
            idm_learning_rate=1e-3,
            tau=0.01,
            reconstruction_weight=1.0,
            endpoint_weight=1.0,
            variance_weight=0.1,
            exploration_std=0.1,
            offline_batch_size=128,
            online_batch_size=256,
            offline_steps=100_000,
        )
    )


__all__ = [
    'LatentInverseDynamics',
    'LatentPathBridge',
    'PixelPathBridgerAgent',
    'get_config',
]
