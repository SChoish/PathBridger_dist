"""PathBridger's complete bridge policy in one agent.

The released agent deliberately contains no actor or action-conditioned
critic.  It learns exactly four policy-side components:

* a scalar transitive value with an EMA target,
* a Gaussian or rectified-flow endpoint proposer,
* an endpoint-pinned residual bridge, and
* an inverse-dynamics model (IDM).
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
_LAYER_NORM = True
_LEARNING_RATE = 3e-4
_ACTION_HORIZON = 5
_BASE_HORIZON = 5
_TAU_V = 0.7
_TARGET_TAU = 0.005
_ENDPOINT_WEIGHT_CAP = 5.0
_FLOW_STEPS = 8
_BRIDGE_ALPHA_POWER = 0.8
_GAUSSIAN_LOG_STD_MIN = -5.0
_GAUSSIAN_LOG_STD_MAX = 1.0
_VALUE_EPS = 1e-6

_REQUIRED_BATCH_KEYS = (
    'observations',
    'next_observations',
    'actions',
    'bridge_targets',
    'endpoint_goals',
    'endpoint_targets',
    'value_goals',
    'value_offsets',
    'base_goals',
    'base_offsets',
    'transitive_subgoals',
    'transitive_offsets',
    'transitive_valids',
)


class ScalarTransitiveValue(nn.Module):
    """A bounded state-goal value, returned as a logit."""

    hidden_dims: Sequence[int] = _HIDDEN_DIMS

    @nn.compact
    def __call__(self, observations: jnp.ndarray, goals: jnp.ndarray) -> jnp.ndarray:
        goal_inputs = goal_representation(goals, 'full')
        inputs = jnp.concatenate([observations, goal_inputs], axis=-1)
        return MLP(
            (*self.hidden_dims, 1),
            activate_final=False,
            layer_norm=_LAYER_NORM,
        )(inputs).squeeze(-1)


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


def _expectile_bce(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
) -> jnp.ndarray:
    """Expectile-weighted BCE used by the transitive constituent target."""

    predictions = jax.nn.sigmoid(logits)
    weights = jnp.where(predictions > targets, 1.0 - _TAU_V, _TAU_V)
    return weights * optax.sigmoid_binary_cross_entropy(logits, targets)


def _replace_module_params(params: Any, module_name: str, value: Any) -> Any:
    """Replace one ModuleDict subtree without changing the container type."""

    is_frozen = isinstance(params, flax.core.FrozenDict)
    mutable = flax.core.unfreeze(params) if is_frozen else dict(params)
    mutable[f'modules_{module_name}'] = value
    return flax.core.freeze(mutable) if is_frozen else mutable


class PathBridgerAgent(flax.struct.PyTreeNode):
    """Actor-free PathBridger training and evaluation state."""

    rng: Any
    network: TrainState
    config: Any = nonpytree_field()

    def _distance_weight_from_values(
        self,
        target_values: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the un-clipped TRL distance weight from the target value."""

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

    def value_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Self, base, and transitive scalar-value BCE losses."""

        observations = batch['observations']
        discount = jnp.asarray(self.config['discount'], dtype=jnp.float32)

        online_logits = self.network.select('value')(
            jnp.concatenate([observations, observations, observations], axis=0),
            jnp.concatenate(
                [observations, batch['base_goals'], batch['value_goals']],
                axis=0,
            ),
            params=grad_params,
        )
        self_logits, base_logits, transitive_logits = jnp.split(
            online_logits,
            3,
            axis=0,
        )
        self_targets = jnp.ones_like(self_logits)
        self_loss = optax.sigmoid_binary_cross_entropy(
            self_logits,
            self_targets,
        ).mean()

        base_offsets = jnp.asarray(batch['base_offsets'], dtype=jnp.float32)
        base_targets = jnp.power(discount, base_offsets)
        base_bce = optax.sigmoid_binary_cross_entropy(
            base_logits,
            base_targets,
        )

        goals = batch['value_goals']
        subgoals = batch['transitive_subgoals']
        left_offsets = jnp.asarray(
            batch['transitive_offsets'],
            dtype=jnp.float32,
        )
        total_offsets = jnp.asarray(batch['value_offsets'], dtype=jnp.float32)
        right_offsets = total_offsets - left_offsets

        use_distance_weights = (
            float(self.config['value_distance_weight_power']) != 0.0
        )
        target_states = [observations, subgoals]
        target_goals = [subgoals, goals]
        if use_distance_weights:
            target_states.extend([observations, observations])
            target_goals.extend([batch['base_goals'], goals])
        target_logits = self.network.select('target_value')(
            jnp.concatenate(target_states, axis=0),
            jnp.concatenate(target_goals, axis=0),
        )
        target_values = jnp.split(
            jax.nn.sigmoid(target_logits),
            len(target_states),
            axis=0,
        )
        target_left, target_right = target_values[:2]
        if use_distance_weights:
            base_distance_weights, base_target_values = (
                self._distance_weight_from_values(target_values[2])
            )
            transitive_distance_weights, transitive_target_values = (
                self._distance_weight_from_values(target_values[3])
            )
        else:
            base_distance_weights = jnp.ones_like(base_bce)
            transitive_distance_weights = jnp.ones_like(base_bce)
        base_loss = jnp.mean(base_distance_weights * base_bce)

        exact_left = jnp.power(discount, left_offsets)
        exact_right = jnp.power(discount, right_offsets)
        mixed_left = jnp.where(
            left_offsets <= _BASE_HORIZON,
            exact_left,
            target_left,
        )
        mixed_right = jnp.where(
            right_offsets <= _BASE_HORIZON,
            exact_right,
            target_right,
        )
        transitive_targets = jax.lax.stop_gradient(mixed_left * mixed_right)

        transitive_bce = _expectile_bce(
            transitive_logits,
            transitive_targets,
        )
        transitive_valids = jnp.asarray(
            batch['transitive_valids'],
            dtype=jnp.float32,
        )
        if transitive_valids.ndim > 1:
            transitive_valids = transitive_valids.reshape(
                transitive_valids.shape[0],
                -1,
            )[:, 0]
        transitive_numerator = jnp.sum(
            transitive_valids
            * transitive_distance_weights
            * transitive_bce
        )
        transitive_loss = transitive_numerator / jnp.maximum(
            jnp.sum(transitive_valids),
            1.0,
        )

        loss = self_loss + base_loss + transitive_loss
        info = {
            'value/loss': loss,
            'value/self_loss': self_loss,
            'value/base_loss': base_loss,
            'value/transitive_loss': transitive_loss,
            'value/self_mean': jax.nn.sigmoid(self_logits).mean(),
            'value/base_mean': jax.nn.sigmoid(base_logits).mean(),
            'value/base_target_mean': base_targets.mean(),
            'value/base_distance_weight_mean': base_distance_weights.mean(),
            'value/transitive_mean': jax.nn.sigmoid(
                transitive_logits
            ).mean(),
            'value/transitive_target_mean': transitive_targets.mean(),
            'value/transitive_distance_weight_mean': (
                transitive_distance_weights.mean()
            ),
            'value/transitive_valid_fraction': transitive_valids.mean(),
        }
        if use_distance_weights:
            info['value/base_target_value_mean'] = base_target_values.mean()
            info['value/transitive_target_value_mean'] = (
                transitive_target_values.mean()
            )
        return loss, info

    def _endpoint_weights(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        endpoint_targets: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        value_logits = self.network.select('target_value')(
            jnp.concatenate([observations, endpoint_targets], axis=0),
            jnp.concatenate([goals, goals], axis=0),
        )
        current_values, endpoint_values = jnp.split(
            jax.nn.sigmoid(value_logits),
            2,
            axis=0,
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
                'endpoint/nll': nll.mean(),
                'endpoint/std_mean': jnp.exp(log_stds).mean(),
            }
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
                'endpoint/flow_matching_loss': flow_errors.mean(),
                'endpoint/flow_time_mean': times.mean(),
            }

        info.update(
            {
                'endpoint/weight_mean': weights.mean(),
                'endpoint/weight_max': weights.max(),
                'endpoint/value_gap_mean': value_gap.mean(),
            }
        )
        return loss, info

    def _bridge_states_at_indices(
        self,
        observations: jnp.ndarray,
        endpoints: jnp.ndarray,
        indices: jnp.ndarray,
        *,
        params: Any | None = None,
    ) -> jnp.ndarray:
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
        return observations[:, None, :] + displacements

    def _construct_bridge(
        self,
        observations: jnp.ndarray,
        endpoints: jnp.ndarray,
        *,
        params: Any | None = None,
    ) -> jnp.ndarray:
        horizon = int(self.config['horizon'])
        bridge = self._bridge_states_at_indices(
            observations,
            endpoints,
            jnp.arange(horizon + 1, dtype=jnp.float32),
            params=params,
        )
        # Preserve both pins exactly, including under finite-precision arithmetic.
        bridge = bridge.at[:, 0, :].set(observations)
        bridge = bridge.at[:, -1, :].set(endpoints)
        return bridge

    def _construct_bridge_prefix(
        self,
        observations: jnp.ndarray,
        endpoints: jnp.ndarray,
        *,
        params: Any | None = None,
    ) -> jnp.ndarray:
        """Construct only the six states consumed by the five-step IDM policy."""

        future_states = self._bridge_states_at_indices(
            observations,
            endpoints,
            jnp.arange(1, _ACTION_HORIZON + 1, dtype=jnp.float32),
            params=params,
        )
        if int(self.config['horizon']) == _ACTION_HORIZON:
            future_states = future_states.at[:, -1, :].set(endpoints)
        return jnp.concatenate([observations[:, None, :], future_states], axis=1)

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
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Mean L1 reconstruction for exactly bridge indices 1 through 5."""

        predicted = self._construct_bridge_prefix(
            batch['observations'],
            batch['endpoint_targets'],
            params=grad_params,
        )
        prefix_errors = jnp.sum(
            jnp.abs(predicted[:, 1 : _ACTION_HORIZON + 1, :]
                    - batch['bridge_targets']),
            axis=-1,
        )
        loss = prefix_errors.mean()
        return loss, {
            'bridge/loss': loss,
            'bridge/prefix_l1': prefix_errors.mean(),
        }

    def idm_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Adjacent-transition inverse-dynamics MSE."""

        predicted_actions = self.network.select('idm')(
            batch['observations'],
            batch['next_observations'],
            params=grad_params,
        )
        squared_errors = jnp.square(predicted_actions - batch['actions'])
        loss = jnp.sum(squared_errors, axis=-1).mean()
        return loss, {
            'idm/loss': loss,
            'idm/squared_l2': loss,
            'idm/action_mse': squared_errors.mean(),
        }

    @jax.jit
    def total_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
        rng: jax.Array | None = None,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Sum all four fixed-coefficient PathBridger objectives."""

        rng = self.rng if rng is None else rng
        value_loss, value_info = self.value_loss(batch, grad_params)
        endpoint_loss, endpoint_info = self.endpoint_loss(
            batch,
            grad_params,
            rng,
        )
        bridge_loss, bridge_info = self.bridge_loss(batch, grad_params)
        idm_loss, idm_info = self.idm_loss(batch, grad_params)
        loss = value_loss + endpoint_loss + bridge_loss + idm_loss
        info = {
            'loss/total': loss,
            **value_info,
            **endpoint_info,
            **bridge_info,
            **idm_info,
        }
        return loss, info

    def _ema_target_value(self, network: TrainState) -> TrainState:
        online_params = network.params['modules_value']
        target_params = network.params['modules_target_value']
        updated_target = jax.tree_util.tree_map(
            lambda online, target: (
                _TARGET_TAU * online + (1.0 - _TARGET_TAU) * target
            ),
            online_params,
            target_params,
        )
        return network.replace(
            params=_replace_module_params(
                network.params,
                'target_value',
                updated_target,
            )
        )

    def update(
        self,
        batch: dict[str, jnp.ndarray],
    ) -> tuple['PathBridgerAgent', dict[str, jnp.ndarray]]:
        """Apply one joint gradient update and one value-target EMA update."""

        missing = [key for key in _REQUIRED_BATCH_KEYS if key not in batch]
        if missing:
            raise KeyError(f'PathBridger batch is missing keys: {missing}')
        if int(batch['bridge_targets'].shape[1]) != _ACTION_HORIZON:
            raise ValueError(
                'bridge_targets must have shape [B, action_horizon, D]; '
                f'expected length {_ACTION_HORIZON}, got '
                f'{batch["bridge_targets"].shape[1]}.'
            )
        return self._update_impl(batch)

    @jax.jit
    def _update_impl(
        self,
        batch: dict[str, jnp.ndarray],
    ) -> tuple['PathBridgerAgent', dict[str, jnp.ndarray]]:
        new_rng, loss_rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=loss_rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        new_network = self._ema_target_value(new_network)
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
        """Sample endpoints, select by online value product, and decode actions."""

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
        if num_candidates == 1:
            selected_endpoints = candidates[:, 0, :]
        else:
            flat_observations = jnp.broadcast_to(
                observations[:, None, :],
                candidates.shape,
            ).reshape(batch_size * num_candidates, state_dim)
            flat_candidates = candidates.reshape(
                batch_size * num_candidates,
                state_dim,
            )
            flat_goals = jnp.broadcast_to(
                goals[:, None, :],
                (batch_size, num_candidates, goals.shape[-1]),
            ).reshape(batch_size * num_candidates, goals.shape[-1])
            value_logits = self.network.select('value')(
                jnp.concatenate([flat_observations, flat_candidates], axis=0),
                jnp.concatenate([flat_candidates, flat_goals], axis=0),
            )
            values_to_endpoint, values_to_goal = jnp.split(
                jax.nn.sigmoid(value_logits),
                2,
                axis=0,
            )
            scores = (values_to_endpoint * values_to_goal).reshape(
                batch_size,
                num_candidates,
            )
            best_indices = jnp.argmax(scores, axis=1)
            selected_endpoints = jnp.take_along_axis(
                candidates,
                best_indices[:, None, None],
                axis=1,
            )[:, 0, :]

        prefix = self._construct_bridge_prefix(
            observations,
            selected_endpoints,
        )
        current_states = prefix[:, :-1, :].reshape(
            batch_size * _ACTION_HORIZON,
            state_dim,
        )
        next_states = prefix[:, 1:, :].reshape(
            batch_size * _ACTION_HORIZON,
            state_dim,
        )
        actions = self.network.select('idm')(current_states, next_states)
        actions = actions.reshape(batch_size, _ACTION_HORIZON, -1)
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
        value_def = ScalarTransitiveValue()
        target_value_def = ScalarTransitiveValue()
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
    """Return the paper's PBF AntMaze-medium settings as clean defaults."""

    return ml_collections.ConfigDict(
        dict(
            env_name='antmaze-medium-navigate-v0',
            endpoint_distribution='flow',
            horizon=25,
            discount=0.99,
            actor_p=(0.0, 0.0, 1.0, 0.0),
            critic_p=(0.0, 1.0, 0.0, 0.0),
            endpoint_value_scale=10.0,
            value_distance_weight_power=0.0,
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
    'ScalarTransitiveValue',
    'get_config',
]
