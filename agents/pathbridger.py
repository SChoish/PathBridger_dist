"""PBF PathBridger with PathFlower's triangular chunk-Q critic.

The endpoint flow, endpoint-pinned state bridge, and inverse dynamics model
are retained. The critic uses base chunk-Q regression, a triangular chunk-Q
decomposition through an offline split state, and an expectile value backup.
At inference, endpoint candidates are decoded into five-step IDM chunks and
ranked directly by triangular Q.
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

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.goal_representation import (
    assert_phi_goal_obs_indices,
    goal_representation,
    infer_phi_goal_obs_indices,
)
from utils.networks import MLP


# These are algorithm constants, not experiment options.
_HIDDEN_DIMS = (512, 512, 512)
_CRITIC_HIDDEN_DIMS = (512, 512, 512)
_LAYER_NORM = True
_LEARNING_RATE = 3e-4
_ACTION_HORIZON = 5
_ENDPOINT_WEIGHT_CAP = 5.0
_FLOW_STEPS = 8
_BRIDGE_ALPHA_POWER = 0.8
_GAUSSIAN_LOG_STD_MIN = -5.0
_GAUSSIAN_LOG_STD_MAX = 1.0
_Q_VALUE_EPS = 1e-6

_REQUIRED_BATCH_KEYS = (
    'observations',
    'next_observations',
    'actions',
    'trajectory',
    'endpoint_goals',
    'endpoint_targets',
    'value_goals',
    'value_offsets',
    'action_chunk_actions',
    'trl_base_goals',
    'trl_base_offsets',
    'trl_split_observations',
    'trl_split_goals',
    'trl_split_action_chunk_actions',
    'trl_split_offsets',
    'trl_valid_mask',
)


def _bce_expectile_loss(logits, targets, tau):
    probabilities = jax.nn.sigmoid(logits)
    weights = jnp.where(targets >= probabilities, float(tau), 1.0 - float(tau))
    return weights * optax.sigmoid_binary_cross_entropy(logits, targets)


class ScalarValueNet(nn.Module):
    """Implicit state-goal value used by endpoint weighting and Triangle-Q."""

    hidden_dims: Sequence[int] = _CRITIC_HIDDEN_DIMS

    @nn.compact
    def __call__(self, observations: jnp.ndarray, goals: jnp.ndarray) -> jnp.ndarray:
        goal_inputs = goal_representation(goals, 'full')
        inputs = jnp.concatenate([observations, goal_inputs], axis=-1)
        return MLP(
            (*self.hidden_dims, 1),
            activate_final=False,
            layer_norm=_LAYER_NORM,
        )(inputs).squeeze(-1)


class BinaryChunkCritic(nn.Module):
    """Ensemble of bounded critics over a fixed-length action chunk."""

    action_size: int
    hidden_dims: Sequence[int] = _CRITIC_HIDDEN_DIMS
    num_qs: int = 2

    @nn.compact
    def __call__(self, observations, goals, actions):
        actions = jnp.asarray(actions)
        if actions.ndim > 2:
            actions = actions.reshape(actions.shape[0], -1)
        if actions.shape[-1] != self.action_size:
            raise ValueError(
                f'Expected flattened action size {self.action_size}, got {actions.shape[-1]}.'
            )
        inputs = jnp.concatenate([
            observations, goal_representation(goals, 'full'), actions,
        ], axis=-1)
        hidden = MLP(
            tuple(self.hidden_dims), activate_final=True, layer_norm=_LAYER_NORM,
        )(inputs)
        return jnp.stack([
            nn.Dense(1, name=f'q_head_{index}')(hidden).squeeze(-1)
            for index in range(self.num_qs)
        ], axis=0)


class GaussianEndpointProposer(nn.Module):
    """Diagonal-Gaussian model of the K-step endpoint displacement."""

    state_dim: int
    env_name: str
    phi_goal_obs_indices: tuple[int, ...]
    hidden_dims: Sequence[int] = _HIDDEN_DIMS

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        goal_inputs = goal_representation(
            goals,
            'phi',
            self.phi_goal_obs_indices,
            env_name=self.env_name,
        )
        inputs = jnp.concatenate([observations, goal_inputs], axis=-1)
        hidden = MLP(
            tuple(self.hidden_dims),
            activate_final=True,
            layer_norm=_LAYER_NORM,
        )(inputs)
        mean = nn.Dense(self.state_dim, name='mean')(hidden)
        log_std = nn.Dense(self.state_dim, name='log_std')(hidden)
        log_std = jnp.clip(
            log_std,
            _GAUSSIAN_LOG_STD_MIN,
            _GAUSSIAN_LOG_STD_MAX,
        )
        return mean, log_std


class FlowEndpointProposer(nn.Module):
    """Conditional rectified-flow velocity field for endpoint displacements."""

    state_dim: int
    env_name: str
    phi_goal_obs_indices: tuple[int, ...]
    hidden_dims: Sequence[int] = _HIDDEN_DIMS

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        noisy_displacements: jnp.ndarray,
        times: jnp.ndarray,
    ) -> jnp.ndarray:
        goal_inputs = goal_representation(
            goals,
            'phi',
            self.phi_goal_obs_indices,
            env_name=self.env_name,
        )
        times = jnp.asarray(times, dtype=jnp.float32)
        if times.ndim == noisy_displacements.ndim - 1:
            times = times[..., None]
        inputs = jnp.concatenate(
            [observations, goal_inputs, noisy_displacements, times],
            axis=-1,
        )
        return MLP(
            (*self.hidden_dims, self.state_dim),
            activate_final=False,
            layer_norm=_LAYER_NORM,
        )(inputs)


class BridgeResidual(nn.Module):
    """Interior deformation of a fixed, endpoint-pinned reference bridge."""

    state_dim: int
    hidden_dims: Sequence[int] = _HIDDEN_DIMS

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        endpoint_displacements: jnp.ndarray,
        times: jnp.ndarray,
    ) -> jnp.ndarray:
        times = jnp.asarray(times, dtype=jnp.float32)
        observations = jnp.broadcast_to(
            observations[:, None, :],
            (*times.shape, observations.shape[-1]),
        )
        endpoint_displacements = jnp.broadcast_to(
            endpoint_displacements[:, None, :],
            (*times.shape, endpoint_displacements.shape[-1]),
        )
        inputs = jnp.concatenate(
            [observations, endpoint_displacements, times[..., None]],
            axis=-1,
        )
        return MLP(
            (*self.hidden_dims, self.state_dim),
            activate_final=False,
            layer_norm=_LAYER_NORM,
        )(inputs)


class InverseDynamics(nn.Module):
    """Decode one desired state transition into an action."""

    action_dim: int
    hidden_dims: Sequence[int] = _HIDDEN_DIMS

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        next_observations: jnp.ndarray,
    ) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, next_observations], axis=-1)
        return MLP(
            (*self.hidden_dims, self.action_dim),
            activate_final=False,
            layer_norm=_LAYER_NORM,
        )(inputs)


def _replace_module_params(params: Any, module_name: str, value: Any) -> Any:
    """Replace one ModuleDict subtree without changing the container type."""

    is_frozen = isinstance(params, flax.core.FrozenDict)
    mutable = flax.core.unfreeze(params) if is_frozen else dict(params)
    mutable[f'modules_{module_name}'] = value
    return flax.core.freeze(mutable) if is_frozen else mutable


class PathBridgerAgent(flax.struct.PyTreeNode):
    """Joint PBF bridge/IDM and triangular-Q training state."""

    rng: Any
    network: TrainState
    config: Any = nonpytree_field()

    def _aggregate_q(self, qs: jnp.ndarray) -> jnp.ndarray:
        method = str(self.config['q_agg']).lower()
        if method == 'min':
            return jnp.min(qs, axis=0)
        if method == 'mean':
            return jnp.mean(qs, axis=0)
        raise ValueError(f"q_agg must be 'min' or 'mean', got {method!r}")

    def _valid_mask(self, batch):
        valids = batch.get('valids')
        if valids is None:
            return jnp.ones((batch['observations'].shape[0],), dtype=jnp.float32)
        valids = jnp.asarray(valids, dtype=jnp.float32)
        if valids.ndim == 1:
            return valids
        return valids.reshape(valids.shape[0], -1)[:, -1]

    @staticmethod
    def _weighted_mean(values, weights):
        return jnp.sum(values * weights) / jnp.maximum(jnp.sum(weights), 1e-6)

    def triangle_q_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
        full_metrics: bool = True,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """PathFlower base-Q, triangular-Q, and expectile-V objectives."""

        valid_mask = self._valid_mask(batch)
        eps = float(self.config.get('q_value_eps', _Q_VALUE_EPS))
        discount = float(self.config['discount'])
        observations = batch['observations']
        goals = batch['value_goals']
        actions = batch['action_chunk_actions']

        # Both online-Q objectives use the same network and action chunks. A
        # single larger GEMM is substantially more efficient on accelerators
        # than launching the 512-wide MLP twice at batch size B.
        batch_size = observations.shape[0]
        online_logits = self.network.select('action_critic')(
            jnp.concatenate([observations, observations], axis=0),
            jnp.concatenate([batch['trl_base_goals'], goals], axis=0),
            jnp.concatenate([actions, actions], axis=0),
            params=grad_params,
        )
        base_logits, triangle_logits = jnp.split(
            online_logits, [batch_size], axis=1,
        )
        base_target = jnp.clip(
            jnp.power(discount, jnp.asarray(batch['trl_base_offsets'], dtype=jnp.float32)),
            eps,
            1.0,
        )
        base_loss_per = jnp.mean(
            optax.sigmoid_binary_cross_entropy(base_logits, base_target[None, :]),
            axis=0,
        )
        base_loss = self._weighted_mean(base_loss_per, valid_mask)

        # Likewise, all three target-Q evaluations share parameters. Batch
        # them to avoid three small, sequential MLP launches.
        target_logits = self.network.select('target_action_critic')(
            jnp.concatenate([
                observations,
                batch['trl_split_observations'],
                observations,
            ], axis=0),
            jnp.concatenate([
                batch['trl_split_goals'],
                goals,
                goals,
            ], axis=0),
            jnp.concatenate([
                actions,
                batch['trl_split_action_chunk_actions'],
                actions,
            ], axis=0),
        )
        left_logits, right_logits, target_q_logits = jnp.split(
            target_logits, [batch_size, 2 * batch_size], axis=1,
        )
        left_q = jnp.clip(
            self._aggregate_q(jax.nn.sigmoid(left_logits)), eps, 1.0,
        )
        right_q = jnp.clip(
            self._aggregate_q(jax.nn.sigmoid(right_logits)), eps, 1.0,
        )
        triangle_target = jax.lax.stop_gradient(
            jnp.clip(left_q * right_q, eps, 1.0)
        )
        triangle_valid = (
            jnp.asarray(batch['trl_valid_mask'], dtype=jnp.float32) * valid_mask
        )
        distances = jnp.maximum(
            jnp.asarray(batch['value_offsets'], dtype=jnp.float32), 0.0,
        )
        distance_weights = jax.lax.stop_gradient(
            1.0 / jnp.power(
                1.0 + distances,
                float(self.config['value_distance_weight_power']),
            )
        )
        triangle_loss_per = jnp.mean(
            _bce_expectile_loss(
                triangle_logits,
                triangle_target[None, :],
                float(self.config['tau_q']),
            ),
            axis=0,
        )
        triangle_loss = self._weighted_mean(
            triangle_loss_per,
            triangle_valid * distance_weights,
        )

        value_logits = self.network.select('value')(
            observations, goals, params=grad_params,
        )
        value_target = jax.lax.stop_gradient(jnp.clip(
            self._aggregate_q(jax.nn.sigmoid(target_q_logits)), eps, 1.0,
        ))
        value_loss = self._weighted_mean(
            _bce_expectile_loss(
                value_logits,
                value_target,
                float(self.config['tau_v']),
            ),
            valid_mask,
        )

        loss = (
            float(self.config['lambda_q_base']) * base_loss
            + float(self.config['lambda_q_tri']) * triangle_loss
            + float(self.config['lambda_v']) * value_loss
        )
        info = {
            'triangle_q/loss': loss,
            'triangle_q/base_loss': base_loss,
            'triangle_q/triangle_loss': triangle_loss,
            'triangle_q/value_loss': value_loss,
        }
        if full_metrics:
            base_q = self._aggregate_q(jax.nn.sigmoid(base_logits))
            triangle_q = self._aggregate_q(jax.nn.sigmoid(triangle_logits))
            values = jax.nn.sigmoid(value_logits)
            info.update({
                'triangle_q/base_pred_mean': base_q.mean(),
                'triangle_q/base_target_mean': base_target.mean(),
                'triangle_q/pred_mean': triangle_q.mean(),
                'triangle_q/target_mean': triangle_target.mean(),
                'triangle_q/left_mean': left_q.mean(),
                'triangle_q/right_mean': right_q.mean(),
                'triangle_q/value_mean': values.mean(),
                'triangle_q/value_target_mean': value_target.mean(),
                'triangle_q/valid_fraction': triangle_valid.mean(),
                'triangle_q/distance_weight_mean': distance_weights.mean(),
            })
        return loss, info

    def _endpoint_weights(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        endpoint_targets: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        batch_size = observations.shape[0]
        target_values = jax.nn.sigmoid(
            self.network.select('target_value')(
                jnp.concatenate([observations, endpoint_targets], axis=0),
                jnp.concatenate([goals, goals], axis=0),
            )
        )
        current_values, endpoint_values = jnp.split(
            target_values, [batch_size], axis=0,
        )
        value_gap = endpoint_values - current_values
        weights = jnp.minimum(
            _ENDPOINT_WEIGHT_CAP,
            jnp.exp(
                jnp.asarray(
                    self.config['endpoint_value_scale'],
                    dtype=jnp.float32,
                )
                * value_gap
            ),
        )
        return jax.lax.stop_gradient(weights), jax.lax.stop_gradient(value_gap)

    def endpoint_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
        rng: jax.Array,
        full_metrics: bool = True,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Weighted PBG likelihood or PBF flow-matching loss."""

        observations = batch['observations']
        goals = batch['endpoint_goals']
        endpoint_targets = batch['endpoint_targets']
        displacement_targets = endpoint_targets - observations
        weights, value_gap = self._endpoint_weights(
            observations,
            goals,
            endpoint_targets,
        )

        if self.config['endpoint_distribution'] == 'gaussian':
            means, log_stds = self.network.select('endpoint')(
                observations,
                goals,
                params=grad_params,
            )
            inverse_variances = jnp.exp(-2.0 * log_stds)
            nll = 0.5 * jnp.sum(
                jnp.square(displacement_targets - means)
                * inverse_variances
                + 2.0 * log_stds
                + jnp.log(2.0 * jnp.pi),
                axis=-1,
            )
            loss = jnp.mean(weights * nll)
            info = {
                'endpoint/loss': loss,
            }
            if full_metrics:
                info['endpoint/nll'] = nll.mean()
                info['endpoint/std_mean'] = jnp.exp(log_stds).mean()
        else:
            noise_rng, time_rng = jax.random.split(rng)
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
            noisy_displacements = (
                (1.0 - times) * noise + times * displacement_targets
            )
            target_velocities = displacement_targets - noise
            predicted_velocities = self.network.select('endpoint')(
                observations,
                goals,
                noisy_displacements,
                times,
                params=grad_params,
            )
            flow_errors = jnp.sum(
                jnp.square(predicted_velocities - target_velocities),
                axis=-1,
            )
            loss = jnp.mean(weights * flow_errors)
            info = {
                'endpoint/loss': loss,
            }
            if full_metrics:
                info['endpoint/flow_matching_loss'] = flow_errors.mean()
                info['endpoint/flow_time_mean'] = times.mean()

        if full_metrics:
            info.update({
                'endpoint/weight_mean': weights.mean(),
                'endpoint/weight_max': weights.max(),
                'endpoint/value_gap_mean': value_gap.mean(),
            })
        return loss, info

    def _construct_bridge_at_indices(
        self,
        observations: jnp.ndarray,
        endpoints: jnp.ndarray,
        indices: jnp.ndarray,
        *,
        params: Any | None = None,
    ) -> jnp.ndarray:
        """Construct only the requested bridge indices."""

        horizon = int(self.config['horizon'])
        indices = jnp.asarray(indices, dtype=jnp.float32)
        times = indices / jnp.asarray(horizon, dtype=jnp.float32)
        times = jnp.broadcast_to(
            times[None, :],
            (observations.shape[0], indices.shape[0]),
        )
        alphas = jnp.power(times, _BRIDGE_ALPHA_POWER)
        masks = (
            indices * (jnp.asarray(horizon, dtype=jnp.float32) - indices)
            / jnp.asarray(horizon * horizon, dtype=jnp.float32)
        )
        endpoint_displacements = endpoints - observations
        residuals = self.network.select('bridge')(
            observations,
            endpoint_displacements,
            times,
            params=params,
        )
        displacements = (
            alphas[..., None] * endpoint_displacements[:, None, :]
            + masks[None, :, None] * residuals
        )
        bridge = observations[:, None, :] + displacements
        # Preserve requested endpoint pins exactly, including under
        # finite-precision arithmetic.
        bridge = jnp.where(
            (indices == 0)[None, :, None], observations[:, None, :], bridge,
        )
        bridge = jnp.where(
            (indices == horizon)[None, :, None], endpoints[:, None, :], bridge,
        )
        return bridge

    def _construct_bridge(
        self,
        observations: jnp.ndarray,
        endpoints: jnp.ndarray,
        *,
        params: Any | None = None,
    ) -> jnp.ndarray:
        """Construct the full bridge for inference and public callers."""

        return self._construct_bridge_at_indices(
            observations,
            endpoints,
            jnp.arange(int(self.config['horizon']) + 1),
            params=params,
        )

    @jax.jit
    def construct_bridge(
        self,
        observations: jnp.ndarray,
        endpoints: jnp.ndarray,
    ) -> jnp.ndarray:
        """Construct a deterministic endpoint-pinned absolute-state bridge."""

        squeeze = observations.ndim == 1
        if squeeze:
            observations = observations[None, :]
            endpoints = endpoints[None, :]
        bridge = self._construct_bridge(observations, endpoints)
        return bridge[0] if squeeze else bridge

    def bridge_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
        full_metrics: bool = True,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Mean L1 reconstruction for exactly bridge indices 1 through 5."""

        trajectory = batch['trajectory']
        prefix = self._construct_bridge_at_indices(
            batch['observations'],
            trajectory[:, -1, :],
            jnp.arange(1, _ACTION_HORIZON + 1),
            params=grad_params,
        )
        prefix_errors = jnp.sum(
            jnp.abs(prefix - trajectory[:, 1 : _ACTION_HORIZON + 1, :]),
            axis=-1,
        )
        loss = prefix_errors.mean()
        info = {
            'bridge/loss': loss,
        }
        if full_metrics:
            info['bridge/prefix_l1'] = loss
            # The endpoint is pinned by construction, so this invariant is
            # exactly zero and does not require constructing the full bridge.
            info['bridge/endpoint_error'] = jnp.zeros((), dtype=loss.dtype)
        return loss, info

    def idm_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
        full_metrics: bool = True,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Adjacent-transition inverse-dynamics MSE."""

        predicted_actions = self.network.select('idm')(
            batch['observations'],
            batch['next_observations'],
            params=grad_params,
        )
        squared_errors = jnp.square(predicted_actions - batch['actions'])
        loss = jnp.sum(squared_errors, axis=-1).mean()
        info = {
            'idm/loss': loss,
            'idm/squared_l2': loss,
        }
        if full_metrics:
            info['idm/action_mse'] = squared_errors.mean()
        return loss, info

    @partial(jax.jit, static_argnames=('full_metrics',))
    def total_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
        rng: jax.Array | None = None,
        full_metrics: bool = True,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Sum Triangle-Q, endpoint, bridge, and IDM objectives."""

        rng = self.rng if rng is None else rng
        critic_loss, critic_info = self.triangle_q_loss(
            batch, grad_params, full_metrics,
        )
        endpoint_loss, endpoint_info = self.endpoint_loss(
            batch,
            grad_params,
            rng,
            full_metrics,
        )
        bridge_loss, bridge_info = self.bridge_loss(
            batch, grad_params, full_metrics,
        )
        idm_loss, idm_info = self.idm_loss(batch, grad_params, full_metrics)
        loss = critic_loss + endpoint_loss + bridge_loss + idm_loss
        info = {
            'loss/total': loss,
            **critic_info,
            **endpoint_info,
            **bridge_info,
            **idm_info,
        }
        return loss, info

    def _ema_targets(self, network: TrainState) -> TrainState:
        tau = float(self.config['tau'])
        params = network.params
        updated_q = jax.tree_util.tree_map(
            lambda source, old: tau * source + (1.0 - tau) * old,
            params['modules_action_critic'],
            params['modules_target_action_critic'],
        )
        updated_v = jax.tree_util.tree_map(
            lambda source, old: tau * source + (1.0 - tau) * old,
            params['modules_value'],
            params['modules_target_value'],
        )
        params = _replace_module_params(params, 'target_action_critic', updated_q)
        params = _replace_module_params(params, 'target_value', updated_v)
        return network.replace(params=params)

    def update(
        self,
        batch: dict[str, jnp.ndarray],
        full_metrics: bool = True,
    ) -> tuple['PathBridgerAgent', dict[str, jnp.ndarray]]:
        """Apply one joint gradient update and EMA updates for Q and V."""

        missing = [key for key in _REQUIRED_BATCH_KEYS if key not in batch]
        if missing:
            raise KeyError(f'PathBridger batch is missing keys: {missing}')
        expected_length = int(self.config['horizon']) + 1
        if int(batch['trajectory'].shape[1]) != expected_length:
            raise ValueError(
                'trajectory must have shape [B, horizon + 1, D]; '
                f'expected length {expected_length}, got '
                f'{batch["trajectory"].shape[1]}.'
            )
        return self._update_impl(batch, full_metrics=full_metrics)

    @partial(jax.jit, static_argnames=('full_metrics',))
    def _update_impl(
        self,
        batch: dict[str, jnp.ndarray],
        full_metrics: bool = True,
    ) -> tuple['PathBridgerAgent', dict[str, jnp.ndarray]]:
        new_rng, loss_rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(
                batch, grad_params, rng=loss_rng, full_metrics=full_metrics,
            )

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        new_network = self._ema_targets(new_network)
        return self.replace(rng=new_rng, network=new_network), info

    def _flow_endpoint_samples(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        initial_noise: jnp.ndarray,
    ) -> jnp.ndarray:
        """Integrate the endpoint flow with the fixed eight Euler steps."""

        batch_size, num_candidates, state_dim = initial_noise.shape
        flat_observations = jnp.broadcast_to(
            observations[:, None, :],
            (batch_size, num_candidates, state_dim),
        ).reshape(batch_size * num_candidates, state_dim)
        flat_goals = jnp.broadcast_to(
            goals[:, None, :],
            (batch_size, num_candidates, goals.shape[-1]),
        ).reshape(batch_size * num_candidates, goals.shape[-1])
        displacements = initial_noise.reshape(
            batch_size * num_candidates,
            state_dim,
        )
        step_size = jnp.asarray(1.0 / _FLOW_STEPS, dtype=jnp.float32)
        for step in range(_FLOW_STEPS):
            times = jnp.full(
                (displacements.shape[0], 1),
                step / _FLOW_STEPS,
                dtype=jnp.float32,
            )
            velocities = self.network.select('endpoint')(
                flat_observations,
                flat_goals,
                displacements,
                times,
            )
            displacements = displacements + step_size * velocities
        return displacements.reshape(batch_size, num_candidates, state_dim)

    def _sample_endpoint_candidates(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        seed: jax.Array,
        *,
        num_candidates: int,
        temperature: float,
    ) -> jnp.ndarray:
        batch_size, state_dim = observations.shape
        noise = jax.random.normal(
            seed,
            (batch_size, num_candidates, state_dim),
            dtype=observations.dtype,
        )
        temperature_array = jnp.asarray(temperature, dtype=observations.dtype)

        if self.config['endpoint_distribution'] == 'gaussian':
            means, log_stds = self.network.select('endpoint')(
                observations,
                goals,
            )
            displacements = (
                means[:, None, :]
                + temperature_array
                * jnp.exp(log_stds)[:, None, :]
                * noise
            )
        else:
            displacements = self._flow_endpoint_samples(
                observations,
                goals,
                temperature_array * noise,
            )
        return observations[:, None, :] + displacements

    @partial(
        jax.jit,
        static_argnames=('num_candidates', 'temperature'),
    )
    def sample_action_chunks(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        seed: jax.Array | None = None,
        num_candidates: int | None = None,
        temperature: float | None = None,
    ) -> jnp.ndarray:
        """Sample PBF endpoints, decode five-step prefixes, and rank by Q."""

        if seed is None:
            seed = jax.random.PRNGKey(0)
        if num_candidates is None:
            num_candidates = int(self.config['eval_num_candidates'])
        if temperature is None:
            temperature = float(self.config['eval_temperature'])
        if num_candidates < 1:
            raise ValueError('num_candidates must be at least one.')
        if temperature < 0.0:
            raise ValueError('temperature must be non-negative.')

        squeeze = observations.ndim == 1
        if squeeze:
            observations = observations[None, :]
            goals = goals[None, :]

        candidates = self._sample_endpoint_candidates(
            observations,
            goals,
            seed,
            num_candidates=num_candidates,
            temperature=temperature,
        )
        batch_size, _, state_dim = candidates.shape
        repeated_observations = jnp.broadcast_to(
            observations[:, None, :],
            candidates.shape,
        )
        repeated_goals = jnp.broadcast_to(
            goals[:, None, :],
            (batch_size, num_candidates, goals.shape[-1]),
        )
        flat_observations = repeated_observations.reshape(
            batch_size * num_candidates,
            state_dim,
        )
        flat_candidates = candidates.reshape(
            batch_size * num_candidates,
            state_dim,
        )
        flat_goals = repeated_goals.reshape(
            batch_size * num_candidates,
            goals.shape[-1],
        )
        # Decode every proposed bridge through h, then let triangular Q select.
        flat_bridges = self._construct_bridge(flat_observations, flat_candidates)
        action_chunk_horizon = int(self.config['action_chunk_horizon'])
        prefix = flat_bridges[:, : action_chunk_horizon + 1, :]
        current_states = prefix[:, :-1, :].reshape(
            batch_size * num_candidates * action_chunk_horizon, state_dim,
        )
        next_states = prefix[:, 1:, :].reshape(
            batch_size * num_candidates * action_chunk_horizon, state_dim,
        )
        candidate_actions = self.network.select('idm')(current_states, next_states)
        candidate_actions = candidate_actions.reshape(
            batch_size, num_candidates, action_chunk_horizon, -1,
        )
        q_logits = self.network.select('action_critic')(
            flat_observations,
            flat_goals,
            candidate_actions.reshape(batch_size * num_candidates, -1),
        )
        scores = self._aggregate_q(jax.nn.sigmoid(q_logits)).reshape(
            batch_size, num_candidates,
        )
        best_indices = jnp.argmax(scores, axis=1)
        actions = jnp.take_along_axis(
            candidate_actions,
            best_indices[:, None, None, None],
            axis=1,
        )[:, 0]
        return actions[0] if squeeze else actions

    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations: jnp.ndarray,
        ex_actions: jnp.ndarray,
        config: dict[str, Any],
    ) -> 'PathBridgerAgent':
        """Initialize all PathBridger modules and the joint optimizer."""

        config = dict(config)
        endpoint_distribution = str(
            config['endpoint_distribution']
        ).lower()
        if endpoint_distribution not in ('flow', 'gaussian'):
            raise ValueError(
                "endpoint_distribution must be 'flow' or 'gaussian', got "
                f'{endpoint_distribution!r}.'
            )
        config['endpoint_distribution'] = endpoint_distribution

        horizon = int(config['horizon'])
        action_chunk_horizon = int(config['action_chunk_horizon'])
        sequence_horizon = int(config['sequence_horizon'])
        if action_chunk_horizon != _ACTION_HORIZON:
            raise ValueError(
                f'Triangle-Q uses {_ACTION_HORIZON}-step chunks, got '
                f'{action_chunk_horizon}.'
            )
        if sequence_horizon < action_chunk_horizon:
            raise ValueError('sequence_horizon must cover the action chunk.')
        if int(config['num_qs']) < 1:
            raise ValueError('num_qs must be at least one.')
        if str(config['q_agg']).lower() not in ('min', 'mean'):
            raise ValueError("q_agg must be 'min' or 'mean'.")
        if horizon < _ACTION_HORIZON:
            raise ValueError(
                f'horizon must be at least {_ACTION_HORIZON}, got {horizon}.'
            )
        discount = float(config['discount'])
        if not 0.0 < discount < 1.0:
            raise ValueError(f'discount must be in (0, 1), got {discount}.')
        if int(config['eval_num_candidates']) < 1:
            raise ValueError('eval_num_candidates must be at least one.')
        if float(config['eval_temperature']) < 0.0:
            raise ValueError('eval_temperature must be non-negative.')

        observations = jnp.asarray(ex_observations, dtype=jnp.float32)
        actions = jnp.asarray(ex_actions, dtype=jnp.float32)
        if observations.ndim != 2 or actions.ndim != 2:
            raise ValueError(
                'ex_observations and ex_actions must be batched rank-2 arrays.'
            )
        if observations.shape[0] != actions.shape[0]:
            raise ValueError(
                'ex_observations and ex_actions must have equal batch sizes.'
            )
        state_dim = int(observations.shape[-1])
        action_dim = int(actions.shape[-1])
        env_name = str(config['env_name'])
        phi_goal_obs_indices = infer_phi_goal_obs_indices(env_name, state_dim)
        assert_phi_goal_obs_indices(
            state_dim,
            'phi',
            phi_goal_obs_indices,
            where='PathBridgerAgent.create (endpoint goal representation)',
            env_name=env_name,
        )
        value_def = ScalarValueNet()
        target_value_def = ScalarValueNet()
        action_critic_def = BinaryChunkCritic(
            action_size=action_chunk_horizon * action_dim,
            num_qs=int(config['num_qs']),
        )
        target_action_critic_def = BinaryChunkCritic(
            action_size=action_chunk_horizon * action_dim,
            num_qs=int(config['num_qs']),
        )
        if endpoint_distribution == 'gaussian':
            endpoint_def = GaussianEndpointProposer(
                state_dim=state_dim,
                env_name=env_name,
                phi_goal_obs_indices=phi_goal_obs_indices,
            )
            endpoint_args = (observations, observations)
        else:
            endpoint_def = FlowEndpointProposer(
                state_dim=state_dim,
                env_name=env_name,
                phi_goal_obs_indices=phi_goal_obs_indices,
            )
            endpoint_args = (
                observations,
                observations,
                jnp.zeros_like(observations),
                jnp.zeros((observations.shape[0], 1), dtype=jnp.float32),
            )
        bridge_def = BridgeResidual(state_dim=state_dim)
        idm_def = InverseDynamics(action_dim=action_dim)

        bridge_times = jnp.broadcast_to(
            jnp.linspace(
                0.0,
                1.0,
                horizon + 1,
                dtype=jnp.float32,
            )[None, :],
            (observations.shape[0], horizon + 1),
        )
        network_info = {
            'value': (value_def, (observations, observations)),
            'target_value': (
                target_value_def,
                (observations, observations),
            ),
            'action_critic': (
                action_critic_def,
                (observations, observations, jnp.zeros(
                    (observations.shape[0], action_chunk_horizon * action_dim),
                    dtype=jnp.float32,
                )),
            ),
            'target_action_critic': (
                target_action_critic_def,
                (observations, observations, jnp.zeros(
                    (observations.shape[0], action_chunk_horizon * action_dim),
                    dtype=jnp.float32,
                )),
            ),
            'endpoint': (endpoint_def, endpoint_args),
            'bridge': (
                bridge_def,
                (observations, jnp.zeros_like(observations), bridge_times),
            ),
            'idm': (idm_def, (observations, observations)),
        }
        network_def = ModuleDict(
            {name: definition for name, (definition, _) in network_info.items()}
        )
        network_args = {
            name: arguments for name, (_, arguments) in network_info.items()
        }

        rng = jax.random.PRNGKey(int(seed))
        rng, init_rng = jax.random.split(rng)
        network_params = network_def.init(
            init_rng,
            **network_args,
        )['params']
        network_params = _replace_module_params(
            network_params,
            'target_action_critic',
            network_params['modules_action_critic'],
        )
        network_params = _replace_module_params(
            network_params,
            'target_value',
            network_params['modules_value'],
        )
        network = TrainState.create(
            network_def,
            network_params,
            tx=optax.adam(_LEARNING_RATE),
        )
        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(config),
        )


def get_config() -> ml_collections.ConfigDict:
    """Return PBF AntMaze-medium settings with PathFlower Triangle-Q."""

    return ml_collections.ConfigDict(
        dict(
            env_name='antmaze-medium-navigate-v0',
            endpoint_distribution='flow',
            horizon=25,
            discount=0.999,
            actor_p=(0.0, 0.0, 1.0, 0.0),
            sequence_horizon=25,
            action_chunk_horizon=5,
            value_geom_sample=True,
            num_qs=2,
            q_agg='mean',
            tau=0.005,
            tau_q=0.7,
            tau_v=0.7,
            lambda_q_base=1.0,
            lambda_q_tri=1.0,
            lambda_v=1.0,
            value_distance_weight_power=0.0,
            q_value_eps=1e-6,
            endpoint_value_scale=10.0,
            eval_num_candidates=8,
            eval_temperature=0.25,
        )
    )


__all__ = [
    'BridgeResidual',
    'FlowEndpointProposer',
    'GaussianEndpointProposer',
    'InverseDynamics',
    'PathBridgerAgent',
    'ScalarValueNet',
    'BinaryChunkCritic',
    'get_config',
]
