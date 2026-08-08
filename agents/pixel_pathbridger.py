"""Latent visual PathBridger with TransV critic, frozen paths, and online IDM."""

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

_VALUE_EPS = 1e-6
_ENDPOINT_WEIGHT_CAP = 100.0


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


class LatentTransitiveValue(nn.Module):
    """Bounded latent state-goal value returned as a logit."""

    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, current_features, goal_features):
        inputs = jnp.concatenate([current_features, goal_features], axis=-1)
        return MLP(
            (*self.hidden_dims, 1),
            activate_final=False,
            name='latent_value',
        )(inputs).squeeze(-1)


def _expectile_bce(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    expectile: float,
) -> jnp.ndarray:
    predictions = jax.nn.sigmoid(logits)
    weights = jnp.where(predictions > targets, 1.0 - expectile, expectile)
    return weights * optax.sigmoid_binary_cross_entropy(logits, targets)


class PixelPathBridgerAgent(flax.struct.PyTreeNode):
    """Action-free latent path learner with latent TransV and online-only IDM."""

    rng: Any
    network: TrainState
    config: Any = nonpytree_field()

    def _ema_update(self, network, *, source: str, target: str, tau: float):
        updated = jax.tree_util.tree_map(
            lambda value, old: tau * value + (1.0 - tau) * old,
            network.params[f'modules_{source}'],
            network.params[f'modules_{target}'],
        )
        return network.replace(
            params=_replace_subtree(network.params, target, updated)
        )

    def _distance_weight_from_values(self, target_values: jnp.ndarray):
        target_values = jax.lax.stop_gradient(target_values)
        safe_values = jnp.clip(target_values, _VALUE_EPS, 1.0)
        distance = jnp.log(safe_values) / jnp.log(
            jnp.asarray(self.config['discount'], dtype=jnp.float32)
        )
        weights = jnp.power(
            1.0 + distance,
            -jnp.asarray(
                self.config['value_distance_weight_power'],
                dtype=jnp.float32,
            ),
        )
        return jax.lax.stop_gradient(weights), target_values

    def _encode(self, images, params=None, *, module: str = 'encoder'):
        return self.network.select(module)(images, params=params)

    def value_loss(self, batch, grad_params):
        discount = jnp.asarray(self.config['discount'], dtype=jnp.float32)
        expectile = float(self.config['expectile'])
        observations = batch['observations']
        next_observations = batch['next_observations']
        value_goals = batch['value_goals']
        base_goals = batch['base_goals']
        subgoals = batch['transitive_subgoals']

        current = self._encode(observations, grad_params)
        next_feats = self._encode(next_observations, grad_params)
        value_goal_feats = self._encode(value_goals, grad_params)
        base_goal_feats = self._encode(base_goals, grad_params)
        subgoal_feats = self._encode(subgoals, grad_params)

        online_logits = self.network.select('value')(
            jnp.concatenate([current, current, current], axis=0),
            jnp.concatenate(
                [current, base_goal_feats, value_goal_feats], axis=0
            ),
            params=grad_params,
        )
        self_logits, base_logits, transitive_logits = jnp.split(
            online_logits, 3, axis=0
        )
        self_loss = optax.sigmoid_binary_cross_entropy(
            self_logits, jnp.ones_like(self_logits)
        ).mean()

        base_offsets = jnp.asarray(batch['base_offsets'], dtype=jnp.float32)
        base_targets = jnp.power(discount, base_offsets)
        base_bce = optax.sigmoid_binary_cross_entropy(base_logits, base_targets)

        left_offsets = jnp.asarray(batch['transitive_offsets'], dtype=jnp.float32)
        total_offsets = jnp.asarray(batch['value_offsets'], dtype=jnp.float32)
        right_offsets = total_offsets - left_offsets

        use_distance_weights = (
            float(self.config['value_distance_weight_power']) != 0.0
        )
        target_current = self._encode(observations, module='target_encoder')
        target_sub = self._encode(subgoals, module='target_encoder')
        target_value_goals = self._encode(value_goals, module='target_encoder')
        target_base_goals = self._encode(base_goals, module='target_encoder')
        target_states = [target_current, target_sub]
        target_goals = [target_sub, target_value_goals]
        if use_distance_weights:
            target_states.extend([target_current, target_current])
            target_goals.extend([target_base_goals, target_value_goals])
        target_logits = self.network.select('target_value')(
            jnp.concatenate(target_states, axis=0),
            jnp.concatenate(target_goals, axis=0),
        )
        target_values = jnp.split(
            jax.nn.sigmoid(target_logits), len(target_states), axis=0
        )
        target_left, target_right = target_values[:2]
        if use_distance_weights:
            base_distance_weights, _ = self._distance_weight_from_values(
                target_values[2]
            )
            transitive_distance_weights, _ = self._distance_weight_from_values(
                target_values[3]
            )
        else:
            base_distance_weights = jnp.ones_like(base_bce)
            transitive_distance_weights = jnp.ones_like(base_bce)
        base_loss = jnp.mean(base_distance_weights * base_bce)

        exact_left = jnp.power(discount, left_offsets)
        exact_right = jnp.power(discount, right_offsets)
        mixed_left = jnp.where(left_offsets <= 1.0, exact_left, target_left)
        mixed_right = jnp.where(right_offsets <= 1.0, exact_right, target_right)
        transitive_targets = jax.lax.stop_gradient(mixed_left * mixed_right)
        transitive_bce = _expectile_bce(
            transitive_logits, transitive_targets, expectile
        )
        transitive_valids = jnp.asarray(
            batch['transitive_valids'], dtype=jnp.float32
        )
        transitive_loss = jnp.sum(
            transitive_valids * transitive_distance_weights * transitive_bce
        ) / jnp.maximum(jnp.sum(transitive_valids), 1.0)

        # Keep next_feats in the graph so encoder sees 1-step transitions.
        _ = next_feats
        loss = self_loss + base_loss + transitive_loss
        return loss, {
            'value/loss': loss,
            'value/self_loss': self_loss,
            'value/base_loss': base_loss,
            'value/transitive_loss': transitive_loss,
            'value/base_distance_weight_mean': base_distance_weights.mean(),
            'value/transitive_distance_weight_mean': (
                transitive_distance_weights.mean()
            ),
        }

    def _endpoint_weights(self, current, goals, endpoints):
        value_logits = self.network.select('target_value')(
            jnp.concatenate([current, endpoints], axis=0),
            jnp.concatenate([goals, goals], axis=0),
        )
        current_values, endpoint_values = jnp.split(
            jax.nn.sigmoid(value_logits), 2, axis=0
        )
        value_gap = endpoint_values - current_values
        weights = jnp.minimum(
            _ENDPOINT_WEIGHT_CAP,
            jnp.exp(
                jnp.asarray(
                    self.config['endpoint_value_scale'], dtype=jnp.float32
                )
                * value_gap
            ),
        )
        return jax.lax.stop_gradient(weights), jax.lax.stop_gradient(value_gap)

    @partial(jax.jit, static_argnames=())
    def offline_update(self, batch):
        path_horizon = int(self.config['path_horizon'])

        def loss_fn(params):
            value_loss, value_info = self.value_loss(batch, params)
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
            endpoint_weights, value_gap = self._endpoint_weights(
                current, goal, predicted_path[:, -1]
            )
            feature_errors = jnp.mean(
                jnp.square(
                    predicted_path - jax.lax.stop_gradient(target_path)
                ),
                axis=(1, 2),
            )
            feature_loss = jnp.mean(endpoint_weights * feature_errors)
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
                endpoint_weights * jnp.mean(
                    jnp.square(predicted_path[:, -1] - goal), axis=-1
                )
            )
            feature_std = jnp.std(current, axis=0)
            variance_loss = jnp.mean(jnp.square(jax.nn.relu(1.0 - feature_std)))
            total = (
                value_loss
                + feature_loss
                + float(self.config['reconstruction_weight'])
                * reconstruction_loss
                + float(self.config['endpoint_weight']) * endpoint_loss
                + float(self.config['variance_weight']) * variance_loss
            )
            info = {
                'loss/total': total,
                'path/feature_loss': feature_loss,
                'path/reconstruction_loss': reconstruction_loss,
                'path/endpoint_loss': endpoint_loss,
                'path/variance_loss': variance_loss,
                'path/endpoint_weight_mean': endpoint_weights.mean(),
                'path/value_gap_mean': value_gap.mean(),
                **value_info,
            }
            return total, info

        network, info = self.network.apply_loss_fn(
            loss_fn,
            trainable_modules=(
                'encoder',
                'bridge',
                'world_decoder',
                'value',
            ),
        )
        network = self._ema_update(
            network,
            source='encoder',
            target='target_encoder',
            tau=float(self.config['tau']),
        )
        network = self._ema_update(
            network,
            source='value',
            target='target_value',
            tau=float(self.config['value_tau']),
        )
        return self.replace(network=network), info

    @partial(jax.jit, static_argnames=())
    def online_update(self, batch):
        """Ground actual online transitions without updating the visual prior."""

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
        value_hidden = tuple(
            config.get('value_hidden_dims', config['hidden_dims'])
        )
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
        value = LatentTransitiveValue(value_hidden)
        # Target modules need a distinct Flax instance. Reusing ``value`` here
        # binds only one of the two ModuleDict names and drops ``modules_value``.
        target_value = LatentTransitiveValue(value_hidden)
        model = ModuleDict(
            {
                'encoder': encoder,
                'target_encoder': encoder,
                'bridge': bridge,
                'world_decoder': world_decoder,
                'value': value,
                'target_value': target_value,
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
            value=(features, features),
            target_value=(features, features),
            idm=(features, features),
        )['params']
        params = _replace_subtree(
            params, 'target_encoder', params['modules_encoder']
        )
        params = _replace_subtree(
            params, 'target_value', params['modules_value']
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
                'value': optax.adam(
                    float(config.get('value_learning_rate', config['learning_rate']))
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
            value_hidden_dims=(256, 256),
            idm_hidden_dims=(512, 512, 512),
            path_horizon=5,
            learning_rate=3e-4,
            encoder_learning_rate=1e-4,
            value_learning_rate=3e-4,
            idm_learning_rate=1e-3,
            tau=0.01,
            value_tau=0.005,
            reconstruction_weight=1.0,
            endpoint_weight=1.0,
            variance_weight=0.1,
            exploration_std=0.1,
            offline_batch_size=128,
            online_batch_size=256,
            offline_steps=100_000,
            # TRL / PathBridger critic locks (override per env in the tune queue).
            discount=0.99,
            expectile=0.7,
            value_distance_weight_power=0.0,
            endpoint_value_scale=10.0,
            value_geom_sample=True,
            value_p_curgoal=0.0,
            value_p_trajgoal=1.0,
            value_p_randomgoal=0.0,
        )
    )


__all__ = [
    'LatentInverseDynamics',
    'LatentPathBridge',
    'LatentTransitiveValue',
    'PixelPathBridgerAgent',
    'get_config',
]
