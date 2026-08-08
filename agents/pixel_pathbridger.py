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
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP

_VALUE_EPS = 1e-6
_ENDPOINT_WEIGHT_CAP = 5.0
_BRIDGE_ALPHA_POWER = 0.8


class ImpalaResidualStack(nn.Module):
    """One official IMPALA-small residual stack."""

    channels: int

    @nn.compact
    def __call__(self, inputs):
        initializer = nn.initializers.xavier_uniform()
        hidden = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
            strides=1,
            padding='SAME',
            kernel_init=initializer,
        )(inputs)
        hidden = nn.max_pool(
            hidden, window_shape=(3, 3), strides=(2, 2), padding='SAME'
        )
        residual = hidden
        hidden = nn.relu(hidden)
        hidden = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
            strides=1,
            padding='SAME',
            kernel_init=initializer,
        )(hidden)
        hidden = nn.relu(hidden)
        hidden = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
            strides=1,
            padding='SAME',
            kernel_init=initializer,
        )(hidden)
        return hidden + residual


class ImpalaSmallEncoder(nn.Module):
    """Official IMPALA-small image encoder (one block per 16/32/32 stack)."""

    feature_dim: int = 512

    @nn.compact
    def __call__(self, images):
        hidden = jnp.asarray(images, dtype=jnp.float32) / 255.0
        for channels in (16, 32, 32):
            hidden = ImpalaResidualStack(channels)(hidden)
        hidden = nn.relu(hidden)
        hidden = hidden.reshape((hidden.shape[0], -1))
        return MLP((self.feature_dim,), activate_final=True)(hidden)


def _length_normalize(value: jnp.ndarray) -> jnp.ndarray:
    """Normalize a compact path representation to radius ``sqrt(dim)``."""

    norm = jnp.linalg.norm(value, axis=-1, keepdims=True)
    return value / jnp.maximum(norm, 1e-6) * jnp.sqrt(value.shape[-1])


class CompactImpalaPathEncoder(nn.Module):
    """IMPALA visual features followed by a compact normalized path head."""

    feature_dim: int = 512
    path_rep_dim: int = 32

    @nn.compact
    def __call__(self, images):
        features = ImpalaSmallEncoder(
            self.feature_dim, name='visual_encoder'
        )(images)
        representation = nn.Dense(
            self.path_rep_dim, name='path_projector'
        )(features)
        return _length_normalize(representation)


def _stack_resident_frames(frames, initial_for_state, indices, frame_stack: int):
    """Build channel-stacked observations from a device-resident frame store."""

    indices = jnp.asarray(indices, dtype=jnp.int32)
    flat_indices = indices.reshape(-1)
    initials = initial_for_state[flat_indices]
    offsets = jnp.arange(frame_stack - 1, -1, -1, dtype=jnp.int32)
    history = jnp.maximum(
        flat_indices[:, None] - offsets[None, :], initials[:, None]
    )
    pixels = frames[history]
    pixels = jnp.transpose(pixels, (0, 2, 3, 1, 4)).reshape(
        flat_indices.shape[0],
        frames.shape[1],
        frames.shape[2],
        frame_stack * frames.shape[3],
    )
    return pixels.reshape((*indices.shape, *pixels.shape[1:]))


def _crop_with_offsets(images, offsets, padding: int = 3):
    """Apply one episode-consistent DrQ crop offset per sampled transition."""

    images = jnp.asarray(images)
    if images.ndim not in (4, 5):
        raise ValueError(f'Expected [B,H,W,C] or [B,T,H,W,C], got {images.shape}.')
    height, width, channels = images.shape[-3:]

    def crop_one(image, offset):
        padded = jnp.pad(
            image,
            ((padding, padding), (padding, padding), (0, 0)),
            mode='edge',
        )
        return jax.lax.dynamic_slice(
            padded,
            (offset[0], offset[1], 0),
            (height, width, channels),
        )

    if images.ndim == 4:
        return jax.vmap(crop_one)(images, offsets)

    def crop_path(path, offset):
        return jax.vmap(lambda image: crop_one(image, offset))(path)

    return jax.vmap(crop_path)(images, offsets)


class LatentPathBridge(nn.Module):
    feature_dim: int
    path_horizon: int = 5
    endpoint_horizon: int = 25
    hidden_dims: Sequence[int] = (512, 512, 512)
    layer_norm: bool = True

    @nn.compact
    def __call__(self, current_features, endpoint_features):
        indices = jnp.arange(1, self.path_horizon + 1, dtype=jnp.float32)
        times = indices / jnp.asarray(self.endpoint_horizon, dtype=jnp.float32)
        times = jnp.broadcast_to(
            times[None, :], (current_features.shape[0], self.path_horizon)
        )
        current = jnp.broadcast_to(
            current_features[:, None, :],
            (current_features.shape[0], self.path_horizon, self.feature_dim),
        )
        displacement = endpoint_features - current_features
        displacements = jnp.broadcast_to(
            displacement[:, None, :], current.shape
        )
        residuals = MLP(
            (*self.hidden_dims, self.feature_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
            name='residual_path',
        )(
            jnp.concatenate(
                [current, displacements, times[..., None]], axis=-1
            )
        )
        alphas = jnp.power(times, _BRIDGE_ALPHA_POWER)
        masks = times * (1.0 - times)
        return (
            current
            + alphas[..., None] * displacements
            + masks[..., None] * residuals
        )


class LatentFlowEndpoint(nn.Module):
    """Conditional rectified-flow velocity for a K-step latent endpoint."""

    feature_dim: int
    hidden_dims: Sequence[int] = (512, 512, 512)
    layer_norm: bool = True

    @nn.compact
    def __call__(
        self,
        current_features,
        goal_features,
        noisy_displacements,
        times,
    ):
        times = jnp.asarray(times, dtype=jnp.float32)
        if times.ndim == noisy_displacements.ndim - 1:
            times = times[..., None]
        inputs = jnp.concatenate(
            [current_features, goal_features, noisy_displacements, times],
            axis=-1,
        )
        return MLP(
            (*self.hidden_dims, self.feature_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
            name='endpoint_velocity',
        )(inputs)


class LatentInverseDynamics(nn.Module):
    action_dim: int
    hidden_dims: Sequence[int] = (512, 512, 512)
    layer_norm: bool = True

    @nn.compact
    def __call__(self, current_features, desired_next_features):
        inputs = jnp.concatenate(
            [current_features, desired_next_features], axis=-1
        )
        return MLP(
            (*self.hidden_dims, self.action_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
            name='idm',
        )(inputs)


class LatentTransitiveValue(nn.Module):
    """Bounded latent state-goal value returned as a logit."""

    hidden_dims: Sequence[int] = (512, 512, 512)
    layer_norm: bool = True

    @nn.compact
    def __call__(self, current_features, goal_features):
        inputs = jnp.concatenate([current_features, goal_features], axis=-1)
        return MLP(
            (*self.hidden_dims, 1),
            activate_final=False,
            layer_norm=self.layer_norm,
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
    """Pixel PBF with latent TransV, flow endpoints, bridge, and IDM."""

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

    def _project_path_rep(self, value):
        if bool(self.config.get('normalize_path_rep', False)):
            return _length_normalize(value)
        return value

    def _augment_batch(self, batch, seed):
        """Use the same random crop for every image belonging to one row."""

        probability = float(self.config.get('p_aug', 0.0))
        if probability <= 0.0:
            return batch
        apply_seed, offset_seed = jax.random.split(seed)
        offsets = jax.random.randint(
            offset_seed,
            (batch['observations'].shape[0], 2),
            minval=0,
            maxval=7,
            dtype=jnp.int32,
        )
        apply_crop = jax.random.bernoulli(apply_seed, probability)
        image_keys = (
            'observations',
            'next_observations',
            'value_goals',
            'base_goals',
            'transitive_subgoals',
            'endpoint_goals',
            'endpoint_targets',
            'bridge_targets',
        )
        augmented = dict(batch)
        for key in image_keys:
            if key not in batch:
                continue
            cropped = _crop_with_offsets(batch[key], offsets)
            augmented[key] = jax.lax.select(apply_crop, cropped, batch[key])
        return augmented

    @staticmethod
    def _representation_diagnostics(current, following):
        """Cheap collapse indicators for the compact path representation."""

        centered = current - current.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered / jnp.maximum(current.shape[0] - 1, 1)
        eigenvalues = jnp.maximum(jnp.linalg.eigvalsh(covariance), 0.0)
        probabilities = eigenvalues / jnp.maximum(eigenvalues.sum(), 1e-8)
        effective_rank = jnp.exp(
            -jnp.sum(probabilities * jnp.log(jnp.maximum(probabilities, 1e-8)))
        )
        random_pairs = jnp.roll(current, shift=1, axis=0)
        return {
            'repr/cross_sample_std': jnp.mean(jnp.std(current, axis=0)),
            'repr/effective_rank': effective_rank,
            'repr/norm': jnp.linalg.norm(current, axis=-1).mean(),
            'repr/one_step_distance': jnp.linalg.norm(
                following - current, axis=-1
            ).mean(),
            'repr/random_pair_distance': jnp.linalg.norm(
                random_pairs - current, axis=-1
            ).mean(),
        }

    def value_loss(self, batch, grad_params, features=None):
        discount = jnp.asarray(self.config['discount'], dtype=jnp.float32)
        expectile = float(self.config['expectile'])
        observations = batch['observations']
        value_goals = batch['value_goals']
        base_goals = batch['base_goals']
        subgoals = batch['transitive_subgoals']

        if features is None:
            current = self._encode(observations, grad_params)
            value_goal_feats = self._encode(value_goals, grad_params)
            base_goal_feats = self._encode(base_goals, grad_params)
            subgoal_feats = self._encode(subgoals, grad_params)
            target_current = self._encode(observations, module='target_encoder')
            target_sub = self._encode(subgoals, module='target_encoder')
            target_value_goals = self._encode(
                value_goals, module='target_encoder'
            )
            target_base_goals = self._encode(
                base_goals, module='target_encoder'
            )
        else:
            current = features['current']
            value_goal_feats = features['value_goals']
            base_goal_feats = features['base_goals']
            subgoal_feats = features['subgoals']
            target_current = features['target_current']
            target_sub = features['target_subgoals']
            target_value_goals = features['target_value_goals']
            target_base_goals = features['target_base_goals']

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
        mixed_left = jnp.where(left_offsets <= 5.0, exact_left, target_left)
        mixed_right = jnp.where(right_offsets <= 5.0, exact_right, target_right)
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

    def _sample_endpoint_candidates(
        self,
        current,
        goals,
        seed,
        *,
        num_candidates: int,
        temperature: float,
    ):
        batch_size, feature_dim = current.shape
        noise = jax.random.normal(
            seed,
            (batch_size, num_candidates, feature_dim),
            dtype=current.dtype,
        )
        displacements = jnp.asarray(temperature, current.dtype) * noise
        repeated_current = jnp.broadcast_to(
            current[:, None, :], displacements.shape
        ).reshape(batch_size * num_candidates, feature_dim)
        repeated_goals = jnp.broadcast_to(
            goals[:, None, :], displacements.shape
        ).reshape(batch_size * num_candidates, feature_dim)
        flat = displacements.reshape(batch_size * num_candidates, feature_dim)
        flow_steps = int(self.config['endpoint_flow_steps'])
        for flow_step in range(flow_steps):
            time = jnp.full(
                (batch_size * num_candidates, 1),
                flow_step / flow_steps,
                dtype=current.dtype,
            )
            velocity = self.network.select('endpoint')(
                repeated_current, repeated_goals, flat, time
            )
            flat = flat + velocity / flow_steps
        endpoints = self._project_path_rep(repeated_current + flat)
        return endpoints.reshape(batch_size, num_candidates, feature_dim)

    def _select_endpoint(
        self,
        current,
        goals,
        seed,
        *,
        num_candidates: int,
        temperature: float,
    ):
        candidates = self._sample_endpoint_candidates(
            current,
            goals,
            seed,
            num_candidates=num_candidates,
            temperature=temperature,
        )
        batch_size, _, feature_dim = candidates.shape
        flat_current = jnp.broadcast_to(
            current[:, None, :], candidates.shape
        ).reshape(-1, feature_dim)
        flat_goals = jnp.broadcast_to(
            goals[:, None, :], candidates.shape
        ).reshape(-1, feature_dim)
        flat_candidates = candidates.reshape(-1, feature_dim)
        to_endpoint = jax.nn.sigmoid(
            self.network.select('value')(flat_current, flat_candidates)
        )
        to_goal = jax.nn.sigmoid(
            self.network.select('value')(flat_candidates, flat_goals)
        )
        scores = (to_endpoint * to_goal).reshape(batch_size, num_candidates)
        best = jnp.argmax(scores, axis=1)
        return jnp.take_along_axis(
            candidates, best[:, None, None], axis=1
        )[:, 0, :]

    def _materialize_indexed_batch(self, batch, frames, initial_for_state):
        frame_stack = int(self.config['frame_stack'])
        materialized = {
            'observations': _stack_resident_frames(
                frames,
                initial_for_state,
                batch['observation_indices'],
                frame_stack,
            ),
            'endpoint_goals': _stack_resident_frames(
                frames,
                initial_for_state,
                batch['endpoint_goal_indices'],
                frame_stack,
            ),
            'endpoint_targets': _stack_resident_frames(
                frames,
                initial_for_state,
                batch['endpoint_target_indices'],
                frame_stack,
            ),
            'bridge_targets': _stack_resident_frames(
                frames,
                initial_for_state,
                batch['bridge_indices'],
                frame_stack,
            ),
            'value_goals': _stack_resident_frames(
                frames,
                initial_for_state,
                batch['value_goal_indices'],
                frame_stack,
            ),
            'base_goals': _stack_resident_frames(
                frames,
                initial_for_state,
                batch['base_indices'],
                frame_stack,
            ),
            'transitive_subgoals': _stack_resident_frames(
                frames,
                initial_for_state,
                batch['transitive_indices'],
                frame_stack,
            ),
            'value_offsets': batch['value_offsets'],
            'base_offsets': batch['base_offsets'],
            'transitive_offsets': batch['transitive_offsets'],
            'transitive_valids': batch['transitive_valids'],
        }
        if not bool(self.config.get('offline_action_free', False)):
            materialized['next_observations'] = _stack_resident_frames(
                frames,
                initial_for_state,
                batch['next_observation_indices'],
                frame_stack,
            )
            materialized['actions'] = batch['actions']
        return materialized

    def _encode_offline_bundle(self, batch, params, path_horizon: int):
        """Encode the offline batch into online/target feature bundles."""

        geometry_source = str(
            self.config.get('geometry_target_source', 'target')
        )
        online_images = [
            batch['observations'],
            batch['value_goals'],
            batch['base_goals'],
            batch['transitive_subgoals'],
            batch['endpoint_goals'],
        ]
        if not bool(self.config.get('offline_action_free', False)):
            online_images.append(batch['next_observations'])
        path_images = batch['bridge_targets']
        flat_path_images = path_images.reshape((-1, *path_images.shape[2:]))
        if geometry_source == 'online':
            online_images.extend([batch['endpoint_targets'], flat_path_images])
        online_features = self.network.select('encoder')(
            jnp.concatenate(online_images, axis=0), params=params
        )
        batch_size = batch['observations'].shape[0]
        regular_online_parts = 5 + int(
            not bool(self.config.get('offline_action_free', False))
        )
        online_parts = [
            online_features[index * batch_size : (index + 1) * batch_size]
            for index in range(regular_online_parts)
        ]
        current, value_goal, base_goal, subgoal, goal = online_parts[:5]
        following = None
        if not bool(self.config.get('offline_action_free', False)):
            following = online_parts[5]
        if geometry_source == 'online':
            geometry_start = regular_online_parts * batch_size
            target_endpoint = online_features[
                geometry_start : geometry_start + batch_size
            ]
            target_path = online_features[geometry_start + batch_size :].reshape(
                (-1, path_horizon, current.shape[-1])
            )
        target_images = [
            batch['observations'],
            batch['transitive_subgoals'],
            batch['value_goals'],
            batch['base_goals'],
        ]
        if geometry_source == 'target':
            target_images.extend([batch['endpoint_targets'], flat_path_images])
        target_features = self.network.select('target_encoder')(
            jnp.concatenate(target_images, axis=0)
        )
        target_current = target_features[0 * batch_size : 1 * batch_size]
        target_subgoal = target_features[1 * batch_size : 2 * batch_size]
        target_value_goal = target_features[2 * batch_size : 3 * batch_size]
        target_base_goal = target_features[3 * batch_size : 4 * batch_size]
        if geometry_source == 'target':
            target_endpoint = target_features[4 * batch_size : 5 * batch_size]
            target_path = target_features[5 * batch_size :].reshape(
                (-1, path_horizon, current.shape[-1])
            )
        return {
            'current': current,
            'value_goal': value_goal,
            'base_goal': base_goal,
            'subgoal': subgoal,
            'goal': goal,
            'following': following,
            'target_current': target_current,
            'target_subgoal': target_subgoal,
            'target_value_goal': target_value_goal,
            'target_base_goal': target_base_goal,
            'target_endpoint': target_endpoint,
            'target_path': target_path,
        }

    def _planner_regression_loss(
        self,
        params,
        *,
        current,
        goal,
        target_endpoint,
        target_path,
        noise_rng,
        time_rng,
    ):
        """Endpoint-flow + bridge losses with production stop-gradient routing.

        Feature/action reductions use a mean over the last axis so loss scale is
        independent of ``path_rep_dim`` / action dimension.
        """

        planner_current = current
        planner_goal = goal
        if bool(self.config.get('stop_planner_rep_grad', False)):
            planner_current = jax.lax.stop_gradient(planner_current)
            planner_goal = jax.lax.stop_gradient(planner_goal)
        target_endpoint = jax.lax.stop_gradient(target_endpoint)
        target_path = jax.lax.stop_gradient(target_path)
        endpoint_weights, value_gap = self._endpoint_weights(
            planner_current, planner_goal, target_endpoint
        )
        displacement_targets = jax.lax.stop_gradient(
            target_endpoint - planner_current
        )
        noise = jax.random.normal(
            noise_rng,
            displacement_targets.shape,
            dtype=displacement_targets.dtype,
        )
        times = jax.random.uniform(
            time_rng,
            (displacement_targets.shape[0], 1),
            dtype=displacement_targets.dtype,
        )
        noisy = (1.0 - times) * noise + times * displacement_targets
        target_velocity = displacement_targets - noise
        predicted_velocity = self.network.select('endpoint')(
            planner_current, planner_goal, noisy, times, params=params
        )
        flow_errors = jnp.mean(
            jnp.square(predicted_velocity - target_velocity), axis=-1
        )
        endpoint_loss = jnp.mean(endpoint_weights * flow_errors)
        predicted_path = self._project_path_rep(
            self.network.select('bridge')(
                planner_current, target_endpoint, params=params
            )
        )
        feature_errors = jnp.mean(
            jnp.abs(predicted_path - target_path),
            axis=-1,
        )
        feature_loss = feature_errors.mean(axis=1).mean()
        return endpoint_loss + feature_loss, {
            'endpoint_loss': endpoint_loss,
            'feature_loss': feature_loss,
            'flow_errors': flow_errors,
            'feature_errors': feature_errors,
            'endpoint_weights': endpoint_weights,
            'value_gap': value_gap,
        }

    def _offline_update_impl(self, batch, *, full_metrics: bool = True):
        path_horizon = int(self.config['path_horizon'])
        new_rng, noise_rng, time_rng, aug_rng = jax.random.split(self.rng, 4)
        batch = self._augment_batch(batch, aug_rng)

        def loss_fn(params):
            features = self._encode_offline_bundle(batch, params, path_horizon)
            current = features['current']
            following = features['following']
            value_loss, value_info = self.value_loss(
                batch,
                params,
                features={
                    'current': current,
                    'value_goals': features['value_goal'],
                    'base_goals': features['base_goal'],
                    'subgoals': features['subgoal'],
                    'target_current': features['target_current'],
                    'target_subgoals': features['target_subgoal'],
                    'target_value_goals': features['target_value_goal'],
                    'target_base_goals': features['target_base_goal'],
                },
            )
            planner_loss, planner_info = self._planner_regression_loss(
                params,
                current=current,
                goal=features['goal'],
                target_endpoint=features['target_endpoint'],
                target_path=features['target_path'],
                noise_rng=noise_rng,
                time_rng=time_rng,
            )
            endpoint_loss = planner_info['endpoint_loss']
            feature_loss = planner_info['feature_loss']
            if bool(self.config.get('offline_action_free', False)):
                idm_loss = jnp.zeros((), dtype=feature_loss.dtype)
                idm_info = {}
            else:
                if 'actions' not in batch:
                    raise KeyError('Full offline pixel PBF batch requires actions.')
                predicted_actions = self.network.select('idm')(
                    current, following, params=params
                )
                squared_errors = jnp.square(predicted_actions - batch['actions'])
                idm_loss = squared_errors.mean()
                idm_info = {'idm/loss': idm_loss}
                if full_metrics:
                    idm_info.update({
                        'idm/action_mse': squared_errors.mean(),
                        'idm/mean_action_baseline_mse': jnp.mean(
                            jnp.square(
                                batch['actions']
                                - batch['actions'].mean(axis=0, keepdims=True)
                            )
                        ),
                        'action/abs_mean': jnp.mean(jnp.abs(predicted_actions)),
                        'action/std': jnp.std(predicted_actions),
                    })
            total = value_loss + planner_loss + idm_loss
            info = {
                'loss/total': total,
                'bridge/loss': feature_loss,
                **idm_info,
                **value_info,
            }
            if full_metrics:
                info.update({
                    'bridge/prefix_l1': planner_info['feature_errors'].mean(),
                    'endpoint/flow_matching_loss': endpoint_loss,
                    'endpoint/flow_error_mean': planner_info['flow_errors'].mean(),
                    'endpoint/weight_mean': planner_info['endpoint_weights'].mean(),
                    'endpoint/value_gap_mean': planner_info['value_gap'].mean(),
                })
            if (
                full_metrics
                and following is not None
                and bool(self.config.get('log_rep_diagnostics', False))
            ):
                info.update(
                    self._representation_diagnostics(current, following)
                )
            return total, info

        trainable_modules = ['encoder', 'endpoint', 'bridge', 'value']
        if not bool(self.config.get('offline_action_free', False)):
            trainable_modules.append('idm')
        network, info = self.network.apply_loss_fn(
            loss_fn,
            trainable_modules=tuple(trainable_modules),
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
        return self.replace(network=network, rng=new_rng), info

    @partial(jax.jit, static_argnames=())
    def planner_encoder_grad_norm(self, batch):
        """L2 norm of encoder grads from endpoint+bridge only (≈0 when routed)."""

        path_horizon = int(self.config['path_horizon'])
        _, noise_rng, time_rng, aug_rng = jax.random.split(self.rng, 4)
        batch = self._augment_batch(batch, aug_rng)

        def planner_loss(params):
            features = self._encode_offline_bundle(batch, params, path_horizon)
            loss, _ = self._planner_regression_loss(
                params,
                current=features['current'],
                goal=features['goal'],
                target_endpoint=features['target_endpoint'],
                target_path=features['target_path'],
                noise_rng=noise_rng,
                time_rng=time_rng,
            )
            return loss

        grads = jax.grad(planner_loss)(self.network.params)
        return optax.tree.norm(grads['modules_encoder'])

    @partial(jax.jit, static_argnames=('full_metrics',))
    def offline_update(self, batch, *, full_metrics: bool = True):
        return self._offline_update_impl(batch, full_metrics=full_metrics)

    @partial(jax.jit, static_argnames=('full_metrics',))
    def offline_update_indexed(
        self,
        batch,
        frames,
        initial_for_state,
        *,
        full_metrics: bool = True,
    ):
        materialized = self._materialize_indexed_batch(
            batch, frames, initial_for_state
        )
        return self._offline_update_impl(materialized, full_metrics=full_metrics)

    @partial(jax.jit, static_argnames=())
    def online_update(self, batch):
        """Ground actual online transitions without updating the visual prior."""

        def loss_fn(params):
            images = jnp.concatenate(
                [batch['observations'], batch['next_observations']], axis=0
            )
            features = jax.lax.stop_gradient(
                self.network.select('encoder')(images)
            )
            current, following = jnp.split(features, 2, axis=0)
            predicted = self.network.select('idm')(
                current, following, params=params
            )
            errors = predicted - batch['actions']
            squared_errors = jnp.square(errors)
            loss = jnp.sum(squared_errors, axis=-1).mean()
            return loss, {
                'loss/total': loss,
                'idm/loss': loss,
                'idm/action_mse': squared_errors.mean(),
                'idm/action_abs_mean': jnp.mean(jnp.abs(predicted)),
            }

        network, info = self.network.apply_loss_fn(
            loss_fn, trainable_modules=('idm',)
        )
        return self.replace(network=network), info

    @partial(
        jax.jit,
        static_argnames=('num_candidates', 'endpoint_temperature'),
    )
    def latent_path(
        self,
        observations,
        goals,
        seed=None,
        num_candidates=None,
        endpoint_temperature=None,
    ):
        if seed is None:
            seed = jax.random.PRNGKey(0)
        if num_candidates is None:
            num_candidates = int(self.config['eval_num_candidates'])
        if endpoint_temperature is None:
            endpoint_temperature = float(self.config['eval_temperature'])
        current = self.network.select('encoder')(observations)
        goal = self.network.select('encoder')(goals)
        endpoint = self._select_endpoint(
            current,
            goal,
            seed,
            num_candidates=int(num_candidates),
            temperature=float(endpoint_temperature),
        )
        return self._project_path_rep(
            self.network.select('bridge')(current, endpoint)
        )

    @partial(
        jax.jit,
        static_argnames=('num_candidates', 'temperature', 'action_temperature'),
    )
    def sample_action_chunks(
        self,
        observations,
        goals,
        seed=None,
        num_candidates=None,
        temperature=None,
        action_temperature=0.0,
    ):
        if seed is None:
            seed = jax.random.PRNGKey(0)
        endpoint_seed, action_seed = jax.random.split(seed)
        if num_candidates is None:
            num_candidates = int(self.config['eval_num_candidates'])
        if temperature is None:
            temperature = float(self.config['eval_temperature'])
        current = self.network.select('encoder')(observations)
        goal = self.network.select('encoder')(goals)
        endpoint = self._select_endpoint(
            current,
            goal,
            endpoint_seed,
            num_candidates=int(num_candidates),
            temperature=float(temperature),
        )
        path = self._project_path_rep(
            self.network.select('bridge')(current, endpoint)
        )
        current_states = jnp.concatenate(
            [current[:, None, :], path[:, :-1, :]], axis=1
        )
        actions = self.network.select('idm')(
            current_states.reshape((-1, current.shape[-1])),
            path.reshape((-1, current.shape[-1])),
        ).reshape((current.shape[0], int(self.config['path_horizon']), -1))
        if action_temperature > 0.0:
            actions = actions + float(action_temperature) * float(
                self.config['exploration_std']
            ) * jax.random.normal(action_seed, actions.shape)
        return jnp.clip(actions, -1.0, 1.0)

    def sample_actions(
        self,
        observations,
        goals,
        seed=None,
        temperature=0.0,
        num_candidates=None,
        endpoint_temperature=None,
    ):
        """Compatibility API returning the first action of a planned chunk."""

        return self.sample_action_chunks(
            observations,
            goals,
            seed=seed,
            num_candidates=num_candidates,
            temperature=endpoint_temperature,
            action_temperature=temperature,
        )[:, 0, :]

    @classmethod
    def create(cls, seed, example_images, action_dim, config):
        config = dict(config)
        legacy_representation = bool(
            config.pop('legacy_raw_representation', False)
        )
        if legacy_representation:
            # Pre-revision checkpoints used the raw IMPALA output. Preserve
            # their exact parameter tree for standalone evaluation.
            config['path_rep_dim'] = int(config['feature_dim'])
            config['normalize_path_rep'] = False
            config['stop_planner_rep_grad'] = False
            config['geometry_target_source'] = 'target'
            config['log_rep_diagnostics'] = False
            config.setdefault('p_aug', 0.0)
        path_horizon = int(config['path_horizon'])
        endpoint_horizon = int(config['endpoint_horizon'])
        if path_horizon < 1 or endpoint_horizon < path_horizon:
            raise ValueError(
                'Require endpoint_horizon >= path_horizon >= 1, got '
                f'{endpoint_horizon} and {path_horizon}.'
            )
        if int(config['endpoint_flow_steps']) < 1:
            raise ValueError('endpoint_flow_steps must be positive.')
        if int(config['eval_num_candidates']) < 1:
            raise ValueError('eval_num_candidates must be positive.')
        if float(config['eval_temperature']) < 0.0:
            raise ValueError('eval_temperature cannot be negative.')
        if not 0.0 < float(config['discount']) < 1.0:
            raise ValueError('discount must lie in (0, 1).')
        images = jnp.asarray(example_images, dtype=jnp.uint8)
        expected_channels = 3 * int(config['frame_stack'])
        if images.ndim != 4 or images.shape[-1] != expected_channels:
            raise ValueError(
                'Pixel PathBridger examples must be [B, H, W, '
                f'{expected_channels}] for frame_stack={config["frame_stack"]}.'
            )
        feature_dim = int(config['feature_dim'])
        path_rep_dim = int(config['path_rep_dim'])
        if path_rep_dim < 2:
            raise ValueError('path_rep_dim must be at least two.')
        if str(config['geometry_target_source']) not in ('online', 'target'):
            raise ValueError("geometry_target_source must be 'online' or 'target'.")
        if not 0.0 <= float(config.get('p_aug', 0.0)) <= 1.0:
            raise ValueError('p_aug must lie in [0, 1].')
        if config.get('encoder') != 'impala_small':
            raise ValueError(
                "Pixel PBF requires encoder='impala_small', got "
                f"{config.get('encoder')!r}."
            )
        hidden_dims = tuple(config['hidden_dims'])
        value_hidden = tuple(
            config.get('value_hidden_dims', config['hidden_dims'])
        )
        if bool(config['normalize_path_rep']):
            encoder = CompactImpalaPathEncoder(feature_dim, path_rep_dim)
        else:
            if path_rep_dim != feature_dim:
                raise ValueError(
                    'A non-normalized representation requires '
                    'path_rep_dim == feature_dim.'
                )
            encoder = ImpalaSmallEncoder(feature_dim)
        bridge = LatentPathBridge(
            feature_dim=path_rep_dim,
            path_horizon=path_horizon,
            endpoint_horizon=endpoint_horizon,
            hidden_dims=hidden_dims,
            layer_norm=bool(config.get('bridge_layer_norm', True)),
        )
        endpoint = LatentFlowEndpoint(
            feature_dim=path_rep_dim,
            hidden_dims=hidden_dims,
            layer_norm=bool(config.get('bridge_layer_norm', True)),
        )
        idm = LatentInverseDynamics(
            int(action_dim),
            tuple(config['idm_hidden_dims']),
            layer_norm=bool(config.get('idm_layer_norm', True)),
        )
        value = LatentTransitiveValue(
            value_hidden,
            layer_norm=bool(config.get('value_layer_norm', True)),
        )
        # Target modules need a distinct Flax instance. Reusing ``value`` here
        # binds only one of the two ModuleDict names and drops ``modules_value``.
        target_value = LatentTransitiveValue(
            value_hidden,
            layer_norm=bool(config.get('value_layer_norm', True)),
        )
        model = ModuleDict(
            {
                'encoder': encoder,
                'target_encoder': encoder,
                'endpoint': endpoint,
                'bridge': bridge,
                'value': value,
                'target_value': target_value,
                'idm': idm,
            }
        )
        features = jnp.zeros((len(images), path_rep_dim), jnp.float32)
        rng = jax.random.PRNGKey(int(seed))
        rng, init_rng = jax.random.split(rng)
        params = model.init(
            init_rng,
            encoder=(images,),
            target_encoder=(images,),
            endpoint=(features, features, features, jnp.zeros((len(images), 1))),
            bridge=(features, features),
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
                'endpoint': optax.adam(float(config['learning_rate'])),
                'bridge': optax.adam(float(config['learning_rate'])),
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
            frame_stack=1,
            encoder='impala_small',
            feature_dim=512,
            path_rep_dim=32,
            normalize_path_rep=True,
            stop_planner_rep_grad=True,
            geometry_target_source='online',
            log_rep_diagnostics=True,
            # Env-specific in train_pixel / pixel_pbf locks (manip=0.5, maze=0.0).
            p_aug=0.0,
            hidden_dims=(512, 512, 512),
            bridge_layer_norm=True,
            value_hidden_dims=(512, 512, 512),
            value_layer_norm=True,
            idm_hidden_dims=(512, 512, 512),
            idm_layer_norm=True,
            path_horizon=5,
            endpoint_horizon=25,
            endpoint_flow_steps=8,
            learning_rate=3e-4,
            encoder_learning_rate=3e-4,
            value_learning_rate=3e-4,
            idm_learning_rate=3e-4,
            tau=0.005,
            value_tau=0.005,
            exploration_std=0.1,
            offline_batch_size=256,
            online_batch_size=256,
            offline_steps=500_000,
            offline_action_free=False,
            # TRL / PathBridger critic locks (override per env in the tune queue).
            discount=0.99,
            expectile=0.7,
            value_distance_weight_power=0.0,
            endpoint_value_scale=10.0,
            value_geom_sample=True,
            value_p_curgoal=0.0,
            value_p_trajgoal=1.0,
            value_p_randomgoal=0.0,
            eval_num_candidates=1,
            eval_temperature=0.0,
        )
    )


__all__ = [
    'CompactImpalaPathEncoder',
    'LatentInverseDynamics',
    'ImpalaSmallEncoder',
    'LatentFlowEndpoint',
    'LatentPathBridge',
    'LatentTransitiveValue',
    'PixelPathBridgerAgent',
    'get_config',
]
