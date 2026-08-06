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

from agents.prefix_generators import (
    JointFlowPrefix,
    LowRankGaussianPrefix,
    low_rank_gaussian_nll,
)
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
_PREFIX_STREAM_ID = 0x50524658

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
_REQUIRED_STATE_BATCH_KEYS = tuple(
    key for key in _REQUIRED_BATCH_KEYS if key != 'actions'
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


def _temporal_path_weights(
    distances: jnp.ndarray,
    active_transitions: jnp.ndarray,
    strength: jnp.ndarray,
    min_weight: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return one-sided geodesic energies and path weights."""

    active = jnp.asarray(active_transitions, dtype=jnp.float32)
    current = distances[:, :-1]
    following = distances[:, 1:]
    distance_deltas = following - current
    progress_targets = jnp.minimum(1.0, jax.lax.stop_gradient(current))
    defects = jnp.where(
        active > 0.0,
        progress_targets + distance_deltas,
        0.0,
    )
    active_counts = jnp.maximum(jnp.sum(active, axis=1), 1.0)
    energies = jnp.sum(jax.nn.relu(defects) * active, axis=1) / active_counts
    weights = jnp.clip(
        jnp.exp(-jax.lax.stop_gradient(strength * energies)),
        float(min_weight),
        1.0,
    )
    return jax.lax.stop_gradient(weights), energies, defects, distance_deltas


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
    state_scale: Any
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

    def _path_weight_strength(self) -> jnp.ndarray:
        """Linearly enable temporal path weighting after value warm-up."""

        beta = float(self.config['path_weight_beta'])
        warmup = int(self.config['path_weight_warmup'])
        ramp = int(self.config['path_weight_ramp'])
        step = jnp.asarray(self.network.step, dtype=jnp.float32)
        if ramp == 0:
            progress = jnp.asarray(step >= warmup, dtype=jnp.float32)
        else:
            progress = jnp.clip((step - warmup) / float(ramp), 0.0, 1.0)
        return jnp.asarray(beta, dtype=jnp.float32) * progress

    def _dataset_path_weights(
        self,
        observations: jnp.ndarray,
        bridge_targets: jnp.ndarray,
        endpoints: jnp.ndarray,
        *,
        full_metrics: bool,
    ) -> tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]:
        """Score the real five-step prefix against its padded endpoint."""

        prefix = jnp.concatenate(
            [observations[:, None, :], bridge_targets],
            axis=1,
        )
        current_states = prefix[:, :-1, :]
        active = jnp.any(
            jnp.abs(current_states - endpoints[:, None, :]) > 1e-6,
            axis=-1,
        ).astype(jnp.float32)
        strength = self._path_weight_strength()

        def evaluate_geometry(_):
            batch_size, prefix_length, state_dim = prefix.shape
            flat_states = prefix.reshape(batch_size * prefix_length, state_dim)
            flat_endpoints = jnp.broadcast_to(
                endpoints[:, None, :],
                prefix.shape,
            ).reshape(batch_size * prefix_length, state_dim)
            values = jax.nn.sigmoid(
                self.network.select('target_value')(
                    flat_states,
                    flat_endpoints,
                )
            ).reshape(batch_size, prefix_length)
            values = jax.lax.stop_gradient(values)
            safe_values = jnp.clip(values, _VALUE_EPS, 1.0)
            distances = jnp.log(safe_values) / jnp.log(
                jnp.asarray(self.config['discount'], dtype=jnp.float32)
            )
            distance_cap = (
                float(self.config['path_distance_cap_multiplier'])
                * float(self.config['horizon'])
            )
            distances = jnp.clip(distances, 0.0, distance_cap)
            return _temporal_path_weights(
                distances,
                active,
                strength,
                float(self.config['path_weight_min']),
            )

        def skip_geometry(_):
            batch_size = observations.shape[0]
            transition_shape = (batch_size, _ACTION_HORIZON)
            return (
                jnp.ones((batch_size,), dtype=jnp.float32),
                jnp.zeros((batch_size,), dtype=jnp.float32),
                jnp.zeros(transition_shape, dtype=jnp.float32),
                jnp.zeros(transition_shape, dtype=jnp.float32),
            )

        # In the common unweighted configuration, keep the geometry branch out
        # of the compiled training executable altogether.  Full-metric updates
        # still evaluate it so diagnostics retain their previous semantics.
        if float(self.config['path_weight_beta']) == 0.0 and not full_metrics:
            weights, energies, defects, distance_deltas = skip_geometry(None)
        else:
            should_evaluate = jnp.logical_or(
                strength > 0.0,
                jnp.asarray(full_metrics),
            )
            weights, energies, defects, distance_deltas = jax.lax.cond(
                should_evaluate,
                evaluate_geometry,
                skip_geometry,
                operand=None,
            )
        return weights, energies, defects, distance_deltas, active, strength

    def value_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
        full_metrics: bool = True,
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
        need_distance_values = use_distance_weights or full_metrics
        target_states = [observations, subgoals]
        target_goals = [subgoals, goals]
        if need_distance_values:
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
            base_distance_weights, _ = self._distance_weight_from_values(
                target_values[2]
            )
            transitive_distance_weights, _ = (
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
        }
        if full_metrics:
            info.update({
                'value/self_mean': jax.nn.sigmoid(self_logits).mean(),
                'value/base_mean': jax.nn.sigmoid(base_logits).mean(),
                'value/base_target_mean': base_targets.mean(),
                'value/base_target_value_mean': target_values[2].mean(),
                'value/base_distance_weight_mean': base_distance_weights.mean(),
                'value/transitive_mean': jax.nn.sigmoid(
                    transitive_logits
                ).mean(),
                'value/transitive_target_mean': transitive_targets.mean(),
                'value/transitive_target_value_mean': (
                    target_values[3].mean()
                ),
                'value/transitive_distance_weight_mean': (
                    transitive_distance_weights.mean()
                ),
                'value/transitive_valid_fraction': transitive_valids.mean(),
            })
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
        bridge = jnp.where(
            (indices == 0)[None, :, None],
            observations[:, None, :],
            bridge,
        )
        bridge = jnp.where(
            (indices == horizon)[None, :, None],
            endpoints[:, None, :],
            bridge,
        )
        return bridge

    def _construct_bridge(
        self,
        observations: jnp.ndarray,
        endpoints: jnp.ndarray,
        *,
        params: Any | None = None,
    ) -> jnp.ndarray:
        """Construct the full endpoint-pinned bridge for inference."""

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
        prefix: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Mean L1 reconstruction for exactly bridge indices 1 through 5."""

        bridge_targets = batch['bridge_targets']
        if prefix is None:
            prefix = self._construct_bridge_at_indices(
                batch['observations'],
                batch['endpoint_targets'],
                jnp.arange(1, _ACTION_HORIZON + 1),
                params=grad_params,
            )
        prefix_errors = jnp.sum(
            jnp.abs(prefix - bridge_targets),
            axis=-1,
        )
        per_sample_loss = prefix_errors.mean(axis=1)
        (
            path_weights,
            path_energies,
            defects,
            distance_deltas,
            active,
            strength,
        ) = self._dataset_path_weights(
            batch['observations'],
            bridge_targets,
            batch['endpoint_targets'],
            full_metrics=full_metrics,
        )
        loss = jnp.sum(path_weights * per_sample_loss) / jnp.maximum(
            jnp.sum(path_weights),
            _VALUE_EPS,
        )
        info = {
            'bridge/loss': loss,
        }
        if full_metrics:
            active_count = jnp.maximum(jnp.sum(active), 1.0)
            weight_sum = jnp.sum(path_weights)
            ess = jnp.square(weight_sum) / jnp.maximum(
                path_weights.shape[0] * jnp.sum(jnp.square(path_weights)),
                _VALUE_EPS,
            )
            info.update({
                'bridge/prefix_l1': prefix_errors.mean(),
                'bridge/endpoint_error': jnp.zeros((), dtype=loss.dtype),
                'bridge/path_weight_strength': strength,
                'bridge/path_weight_mean': path_weights.mean(),
                'bridge/path_weight_min': path_weights.min(),
                'bridge/path_weight_ess_fraction': ess,
                'bridge/path_energy_mean': path_energies.mean(),
                'bridge/path_active_fraction': active.mean(),
                'bridge/path_monotonic_violation_fraction': jnp.sum(
                    (distance_deltas > 0.0) * active
                ) / active_count,
                'bridge/path_negative_defect_fraction': jnp.sum(
                    (defects < 0.0) * active
                ) / active_count,
            })
        return loss, info

    def prefix_distribution_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
        rng: jax.Array,
        full_metrics: bool = True,
        reference: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Learn an unweighted joint distribution of normalized bridge residuals."""

        prefix_model = self.config['prefix_model']
        if prefix_model == 'deterministic':
            zero = jnp.zeros((), dtype=batch['observations'].dtype)
            return zero, {}

        event_steps = _ACTION_HORIZON - int(
            int(self.config['horizon']) == _ACTION_HORIZON
        )
        observations = batch['observations']
        endpoints = batch['endpoint_targets']
        if reference is None:
            reference = self._construct_bridge_at_indices(
                observations,
                endpoints,
                jnp.arange(1, event_steps + 1),
                params=grad_params,
            )
        reference = jax.lax.stop_gradient(reference[:, :event_steps, :])
        residual_targets = (
            batch['bridge_targets'][:, :event_steps, :] - reference
        ) / self.state_scale[None, None, :]
        flat_targets = residual_targets.reshape(residual_targets.shape[0], -1)
        endpoint_displacements = endpoints - observations

        if prefix_model == 'low_rank_gaussian':
            means, sigmas, factors = self.network.select('prefix')(
                observations,
                endpoint_displacements,
                params=grad_params,
            )
            nll = low_rank_gaussian_nll(flat_targets, means, sigmas, factors)
            loss = nll.mean()
            info = {'prefix/loss': loss}
            if full_metrics:
                info.update({
                    'prefix/nll': loss,
                    'prefix/target_rms': jnp.sqrt(
                        jnp.mean(jnp.square(flat_targets))
                    ),
                    'prefix/mean_rms': jnp.sqrt(jnp.mean(jnp.square(means))),
                    'prefix/sigma_mean': sigmas.mean(),
                    'prefix/factor_rms': jnp.sqrt(
                        jnp.mean(jnp.square(factors))
                    ),
                })
            return loss, info

        noise_rng, time_rng = jax.random.split(rng)
        noise = jax.random.normal(
            noise_rng,
            flat_targets.shape,
            dtype=flat_targets.dtype,
        )
        times = jax.random.uniform(
            time_rng,
            (flat_targets.shape[0], 1),
            dtype=flat_targets.dtype,
        )
        noisy_residuals = (1.0 - times) * noise + times * flat_targets
        target_velocities = flat_targets - noise
        predicted_velocities = self.network.select('prefix')(
            observations,
            endpoint_displacements,
            noisy_residuals,
            times,
            params=grad_params,
        )
        per_example_errors = jnp.sum(
            jnp.square(predicted_velocities - target_velocities),
            axis=-1,
        )
        loss = per_example_errors.mean()
        info = {'prefix/loss': loss}
        if full_metrics:
            info.update({
                'prefix/flow_matching_loss': loss,
                'prefix/target_rms': jnp.sqrt(
                    jnp.mean(jnp.square(flat_targets))
                ),
                'prefix/noise_rms': jnp.sqrt(jnp.mean(jnp.square(noise))),
                'prefix/velocity_rms': jnp.sqrt(
                    jnp.mean(jnp.square(predicted_velocities))
                ),
            })
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
        """Sum all four fixed-coefficient PathBridger objectives."""

        rng = self.rng if rng is None else rng
        value_loss, value_info = self.value_loss(batch, grad_params, full_metrics)
        endpoint_loss, endpoint_info = self.endpoint_loss(
            batch,
            grad_params,
            rng,
            full_metrics,
        )
        bridge_prefix = self._construct_bridge_at_indices(
            batch['observations'],
            batch['endpoint_targets'],
            jnp.arange(1, _ACTION_HORIZON + 1),
            params=grad_params,
        )
        bridge_loss, bridge_info = self.bridge_loss(
            batch,
            grad_params,
            full_metrics,
            prefix=bridge_prefix,
        )
        if bool(self.config.get('offline_action_free', False)):
            idm_loss = jnp.zeros((), dtype=value_loss.dtype)
            idm_info = {}
        else:
            idm_loss, idm_info = self.idm_loss(batch, grad_params, full_metrics)
        prefix_rng = jax.random.fold_in(rng, _PREFIX_STREAM_ID)
        prefix_loss, prefix_info = self.prefix_distribution_loss(
            batch,
            grad_params,
            prefix_rng,
            full_metrics,
            reference=bridge_prefix,
        )
        weighted_prefix_loss = (
            jnp.asarray(self.config['prefix_loss_weight'], dtype=prefix_loss.dtype)
            * prefix_loss
        )
        loss = (
            value_loss
            + endpoint_loss
            + bridge_loss
            + idm_loss
            + weighted_prefix_loss
        )
        info = {
            'loss/total': loss,
            **value_info,
            **endpoint_info,
            **bridge_info,
            **idm_info,
            **prefix_info,
        }
        if self.config['prefix_model'] != 'deterministic':
            info['loss/prefix_weighted'] = weighted_prefix_loss
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
        full_metrics: bool = True,
    ) -> tuple['PathBridgerAgent', dict[str, jnp.ndarray]]:
        """Apply one joint gradient update and one value-target EMA update."""

        required = (
            _REQUIRED_STATE_BATCH_KEYS
            if bool(self.config.get('offline_action_free', False))
            else _REQUIRED_BATCH_KEYS
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f'PathBridger batch is missing keys: {missing}')
        if int(batch['bridge_targets'].shape[1]) != _ACTION_HORIZON:
            raise ValueError(
                'bridge_targets must have shape [B, 5, D]; '
                f'got {batch["bridge_targets"].shape}.'
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

    def _sample_stochastic_prefixes(
        self,
        observations: jnp.ndarray,
        endpoints: jnp.ndarray,
        seed: jax.Array,
        *,
        num_samples: int,
        temperature: float,
    ) -> jnp.ndarray:
        """Sample absolute prefixes for one already-selected endpoint per row."""

        batch_size, state_dim = observations.shape
        event_steps = _ACTION_HORIZON - int(
            int(self.config['horizon']) == _ACTION_HORIZON
        )
        event_dim = event_steps * state_dim
        endpoint_displacements = endpoints - observations
        reference = self._construct_bridge_at_indices(
            observations,
            endpoints,
            jnp.arange(1, _ACTION_HORIZON + 1),
        )
        temperature_array = jnp.asarray(temperature, dtype=observations.dtype)

        if self.config['prefix_model'] == 'low_rank_gaussian':
            low_rank_rng, diagonal_rng = jax.random.split(seed)
            means, sigmas, factors = self.network.select('prefix')(
                observations,
                endpoint_displacements,
            )
            low_rank_noise = jax.random.normal(
                low_rank_rng,
                (batch_size, num_samples, int(self.config['prefix_rank'])),
                dtype=observations.dtype,
            )
            diagonal_noise = jax.random.normal(
                diagonal_rng,
                (batch_size, num_samples, event_dim),
                dtype=observations.dtype,
            )
            correlated = jnp.einsum(
                'bpr,bmr->bmp',
                factors,
                low_rank_noise,
            )
            normalized = means[:, None, :] + temperature_array * (
                correlated + sigmas[:, None, :] * diagonal_noise
            )
        else:
            normalized = temperature_array * jax.random.normal(
                seed,
                (batch_size, num_samples, event_dim),
                dtype=observations.dtype,
            )
            flat_observations = jnp.broadcast_to(
                observations[:, None, :],
                (batch_size, num_samples, state_dim),
            ).reshape(batch_size * num_samples, state_dim)
            flat_displacements = jnp.broadcast_to(
                endpoint_displacements[:, None, :],
                (batch_size, num_samples, state_dim),
            ).reshape(batch_size * num_samples, state_dim)
            flat_normalized = normalized.reshape(batch_size * num_samples, event_dim)
            flow_steps = int(self.config['prefix_flow_steps'])
            step_size = jnp.asarray(1.0 / flow_steps, dtype=observations.dtype)
            for step in range(flow_steps):
                times = jnp.full(
                    (batch_size * num_samples, 1),
                    step / flow_steps,
                    dtype=observations.dtype,
                )
                velocities = self.network.select('prefix')(
                    flat_observations,
                    flat_displacements,
                    flat_normalized,
                    times,
                )
                flat_normalized = flat_normalized + step_size * velocities
            normalized = flat_normalized.reshape(batch_size, num_samples, event_dim)

        normalized = normalized.reshape(
            batch_size,
            num_samples,
            event_steps,
            state_dim,
        )
        prefixes = jnp.broadcast_to(
            reference[:, None, :, :],
            (batch_size, num_samples, _ACTION_HORIZON, state_dim),
        )
        stochastic_states = (
            reference[:, None, :event_steps, :]
            + self.state_scale[None, None, None, :] * normalized
        )
        prefixes = prefixes.at[:, :, :event_steps, :].set(stochastic_states)
        starts = jnp.broadcast_to(
            observations[:, None, None, :],
            (batch_size, num_samples, 1, state_dim),
        )
        return jnp.concatenate([starts, prefixes], axis=2)

    def _sample_prefixes(
        self,
        observations: jnp.ndarray,
        endpoints: jnp.ndarray,
        seed: jax.Array,
        *,
        num_samples: int,
        temperature: float,
        include_deterministic: bool,
    ) -> jnp.ndarray:
        deterministic_states = self._construct_bridge_at_indices(
            observations,
            endpoints,
            jnp.arange(0, _ACTION_HORIZON + 1),
        )[:, None, :, :]
        if self.config['prefix_model'] == 'deterministic':
            return deterministic_states

        stochastic = self._sample_stochastic_prefixes(
            observations,
            endpoints,
            seed,
            num_samples=num_samples,
            temperature=temperature,
        )
        if include_deterministic:
            return jnp.concatenate([deterministic_states, stochastic], axis=1)
        return stochastic

    @partial(
        jax.jit,
        static_argnames=('num_samples', 'temperature', 'include_deterministic'),
    )
    def sample_prefixes(
        self,
        observations: jnp.ndarray,
        endpoints: jnp.ndarray,
        seed: jax.Array | None = None,
        *,
        num_samples: int = 1,
        temperature: float = 1.0,
        include_deterministic: bool = False,
    ) -> jnp.ndarray:
        """Sample endpoint-conditioned absolute state prefixes."""

        if num_samples < 1:
            raise ValueError('num_samples must be at least one.')
        if temperature < 0.0:
            raise ValueError('temperature must be non-negative.')
        if seed is None:
            seed = jax.random.PRNGKey(0)
        squeeze = observations.ndim == 1
        if squeeze:
            observations = observations[None, :]
            endpoints = endpoints[None, :]
        prefixes = self._sample_prefixes(
            observations,
            endpoints,
            seed,
            num_samples=num_samples,
            temperature=temperature,
            include_deterministic=include_deterministic,
        )
        return prefixes[0] if squeeze else prefixes

    def _transv_chain_scores(
        self,
        prefixes: jnp.ndarray,
        endpoints: jnp.ndarray,
    ) -> jnp.ndarray:
        """Score prefix candidates with the EMA TransV multiplicative chain."""

        batch_size, num_prefixes, _, state_dim = prefixes.shape
        current = prefixes[:, :, :-1, :].reshape(-1, state_dim)
        following = prefixes[:, :, 1:, :].reshape(-1, state_dim)
        transition_values = jax.nn.sigmoid(
            self.network.select('target_value')(current, following)
        ).reshape(batch_size, num_prefixes, _ACTION_HORIZON)
        final_states = prefixes[:, :, -1, :].reshape(-1, state_dim)
        repeated_endpoints = jnp.broadcast_to(
            endpoints[:, None, :],
            (batch_size, num_prefixes, state_dim),
        ).reshape(-1, state_dim)
        remainder_values = jax.nn.sigmoid(
            self.network.select('target_value')(final_states, repeated_endpoints)
        ).reshape(batch_size, num_prefixes)
        return (
            jnp.sum(jnp.log(jnp.clip(transition_values, _VALUE_EPS, 1.0)), axis=-1)
            + jnp.log(jnp.clip(remainder_values, _VALUE_EPS, 1.0))
        )

    @partial(
        jax.jit,
        static_argnames=('num_candidates', 'temperature'),
    )
    def sample_state_prefix(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        seed: jax.Array | None = None,
        num_candidates: int | None = None,
        temperature: float | None = None,
    ) -> jnp.ndarray:
        """Plan one absolute six-state prefix without decoding actions."""

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
        repeated_observations = jnp.broadcast_to(observations[:, None, :], candidates.shape)
        repeated_goals = jnp.broadcast_to(
            goals[:, None, :],
            (batch_size, num_candidates, goals.shape[-1]),
        )
        flat_observations = repeated_observations.reshape(batch_size * num_candidates, state_dim)
        flat_candidates = candidates.reshape(batch_size * num_candidates, state_dim)
        flat_goals = repeated_goals.reshape(batch_size * num_candidates, goals.shape[-1])
        values_to_endpoint = jax.nn.sigmoid(
            self.network.select('value')(flat_observations, flat_candidates)
        )
        values_to_goal = jax.nn.sigmoid(
            self.network.select('value')(flat_candidates, flat_goals)
        )
        scores = (values_to_endpoint * values_to_goal).reshape(batch_size, num_candidates)
        best_indices = jnp.argmax(scores, axis=1)
        selected_endpoints = jnp.take_along_axis(
            candidates, best_indices[:, None, None], axis=1
        )[:, 0, :]

        prefix_seed = jax.random.fold_in(seed, _PREFIX_STREAM_ID)
        if self.config['prefix_model'] == 'deterministic':
            prefix_candidates = self._sample_prefixes(
                observations,
                selected_endpoints,
                prefix_seed,
                num_samples=1,
                temperature=0.0,
                include_deterministic=True,
            )
            prefix = prefix_candidates[:, 0, :, :]
        elif self.config['eval_prefix_selection'] == 'sample_one':
            prefix_candidates = self._sample_prefixes(
                observations,
                selected_endpoints,
                prefix_seed,
                num_samples=1,
                temperature=float(self.config['eval_prefix_temperature']),
                include_deterministic=False,
            )
            prefix = prefix_candidates[:, 0, :, :]
        else:
            prefix_candidates = self._sample_prefixes(
                observations,
                selected_endpoints,
                prefix_seed,
                num_samples=int(self.config['eval_num_prefix_samples']),
                temperature=float(self.config['eval_prefix_temperature']),
                include_deterministic=bool(self.config['eval_include_deterministic_prefix']),
            )
            chain_scores = self._transv_chain_scores(prefix_candidates, selected_endpoints)
            best_prefix_indices = jnp.argmax(chain_scores, axis=1)
            prefix = jnp.take_along_axis(
                prefix_candidates,
                best_prefix_indices[:, None, None, None],
                axis=1,
            )[:, 0, :, :]
        return prefix[0] if squeeze else prefix

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
        """Select an endpoint first, sample its prefix, then decode actions."""

        if bool(self.config.get('offline_action_free', False)):
            raise ValueError(
                'An action-free PBF checkpoint has no IDM. Use sample_state_prefix '
                'with a separately trained online IDM.'
            )

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
        values_to_endpoint = jax.nn.sigmoid(
            self.network.select('value')(flat_observations, flat_candidates)
        )
        values_to_goal = jax.nn.sigmoid(
            self.network.select('value')(flat_candidates, flat_goals)
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

        prefix_seed = jax.random.fold_in(seed, _PREFIX_STREAM_ID)
        if self.config['prefix_model'] == 'deterministic':
            prefix_candidates = self._sample_prefixes(
                observations,
                selected_endpoints,
                prefix_seed,
                num_samples=1,
                temperature=0.0,
                include_deterministic=True,
            )
            prefix = prefix_candidates[:, 0, :, :]
        elif self.config['eval_prefix_selection'] == 'sample_one':
            prefix_candidates = self._sample_prefixes(
                observations,
                selected_endpoints,
                prefix_seed,
                num_samples=1,
                temperature=float(self.config['eval_prefix_temperature']),
                include_deterministic=False,
            )
            prefix = prefix_candidates[:, 0, :, :]
        else:
            prefix_candidates = self._sample_prefixes(
                observations,
                selected_endpoints,
                prefix_seed,
                num_samples=int(self.config['eval_num_prefix_samples']),
                temperature=float(self.config['eval_prefix_temperature']),
                include_deterministic=bool(
                    self.config['eval_include_deterministic_prefix']
                ),
            )
            chain_scores = self._transv_chain_scores(
                prefix_candidates,
                selected_endpoints,
            )
            best_prefix_indices = jnp.argmax(chain_scores, axis=1)
            prefix = jnp.take_along_axis(
                prefix_candidates,
                best_prefix_indices[:, None, None, None],
                axis=1,
            )[:, 0, :, :]

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
        ex_actions: jnp.ndarray | None,
        config: dict[str, Any],
        state_scale: jnp.ndarray | None = None,
    ) -> 'PathBridgerAgent':
        """Initialize all PathBridger modules and the joint optimizer."""

        config = dict(config)
        config.setdefault('offline_action_free', False)
        endpoint_distribution = str(
            config['endpoint_distribution']
        ).lower()
        if endpoint_distribution not in ('flow', 'gaussian'):
            raise ValueError(
                "endpoint_distribution must be 'flow' or 'gaussian', got "
                f'{endpoint_distribution!r}.'
            )
        config['endpoint_distribution'] = endpoint_distribution

        prefix_model = str(config['prefix_model']).lower()
        if prefix_model not in ('deterministic', 'low_rank_gaussian', 'joint_flow'):
            raise ValueError(
                "prefix_model must be 'deterministic', 'low_rank_gaussian', or "
                f"'joint_flow', got {prefix_model!r}."
            )
        config['prefix_model'] = prefix_model
        if float(config['prefix_loss_weight']) < 0.0:
            raise ValueError('prefix_loss_weight cannot be negative.')
        if int(config['prefix_rank']) < 1:
            raise ValueError('prefix_rank must be at least one.')
        if float(config['prefix_sigma_floor']) <= 0.0:
            raise ValueError('prefix_sigma_floor must be positive.')
        if float(config['prefix_scale_floor']) <= 0.0:
            raise ValueError('prefix_scale_floor must be positive.')
        if int(config['prefix_flow_steps']) < 1:
            raise ValueError('prefix_flow_steps must be at least one.')
        prefix_selection = str(config['eval_prefix_selection']).lower()
        if prefix_selection not in ('sample_one', 'transv_chain'):
            raise ValueError(
                "eval_prefix_selection must be 'sample_one' or 'transv_chain'."
            )
        config['eval_prefix_selection'] = prefix_selection
        if int(config['eval_num_prefix_samples']) < 1:
            raise ValueError('eval_num_prefix_samples must be at least one.')
        if float(config['eval_prefix_temperature']) < 0.0:
            raise ValueError('eval_prefix_temperature must be non-negative.')

        horizon = int(config['horizon'])
        if horizon < _ACTION_HORIZON:
            raise ValueError(
                f'horizon must be at least {_ACTION_HORIZON}, got {horizon}.'
            )
        discount = float(config['discount'])
        if not 0.0 < discount < 1.0:
            raise ValueError(f'discount must be in (0, 1), got {discount}.')
        if float(config['path_weight_beta']) < 0.0:
            raise ValueError('path_weight_beta cannot be negative.')
        if not 0.0 < float(config['path_weight_min']) <= 1.0:
            raise ValueError('path_weight_min must lie in (0, 1].')
        if int(config['path_weight_warmup']) < 0:
            raise ValueError('path_weight_warmup cannot be negative.')
        if int(config['path_weight_ramp']) < 0:
            raise ValueError('path_weight_ramp cannot be negative.')
        if float(config['path_distance_cap_multiplier']) <= 0.0:
            raise ValueError('path_distance_cap_multiplier must be positive.')
        if int(config['eval_num_candidates']) < 1:
            raise ValueError('eval_num_candidates must be at least one.')
        if float(config['eval_temperature']) < 0.0:
            raise ValueError('eval_temperature must be non-negative.')

        observations = jnp.asarray(ex_observations, dtype=jnp.float32)
        action_free = bool(config['offline_action_free'])
        if observations.ndim != 2:
            raise ValueError('ex_observations must be a batched rank-2 array.')
        actions = None if ex_actions is None else jnp.asarray(ex_actions, dtype=jnp.float32)
        if not action_free:
            if actions is None or actions.ndim != 2:
                raise ValueError(
                    'Action-conditioned PathBridger requires batched rank-2 ex_actions.'
                )
            if observations.shape[0] != actions.shape[0]:
                raise ValueError(
                    'ex_observations and ex_actions must have equal batch sizes.'
                )
        state_dim = int(observations.shape[-1])
        if state_scale is None:
            state_scale_array = jnp.ones((state_dim,), dtype=jnp.float32)
        else:
            state_scale_array = jnp.asarray(state_scale, dtype=jnp.float32)
        if state_scale_array.shape != (state_dim,):
            raise ValueError(
                f'state_scale must have shape ({state_dim},), got '
                f'{state_scale_array.shape}.'
            )
        if not bool(jnp.all(jnp.isfinite(state_scale_array))):
            raise ValueError('state_scale must contain only finite values.')
        if not bool(jnp.all(state_scale_array > 0.0)):
            raise ValueError('state_scale values must be positive.')
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
        event_steps = _ACTION_HORIZON - int(horizon == _ACTION_HORIZON)
        prefix_event_dim = event_steps * state_dim
        prefix_def = None
        prefix_args = None
        if prefix_model == 'low_rank_gaussian':
            prefix_def = LowRankGaussianPrefix(
                event_dim=prefix_event_dim,
                rank=int(config['prefix_rank']),
                sigma_floor=float(config['prefix_sigma_floor']),
            )
            prefix_args = (observations, jnp.zeros_like(observations))
        elif prefix_model == 'joint_flow':
            prefix_def = JointFlowPrefix(event_dim=prefix_event_dim)
            prefix_args = (
                observations,
                jnp.zeros_like(observations),
                jnp.zeros(
                    (observations.shape[0], prefix_event_dim),
                    dtype=jnp.float32,
                ),
                jnp.zeros((observations.shape[0], 1), dtype=jnp.float32),
            )

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
        }
        if not action_free:
            idm_def = InverseDynamics(action_dim=int(actions.shape[-1]))
            network_info['idm'] = (idm_def, (observations, observations))
        if prefix_def is not None:
            network_info['prefix'] = (prefix_def, prefix_args)
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
            state_scale=state_scale_array,
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
            path_weight_beta=0.0,
            path_weight_min=0.1,
            path_weight_warmup=100_000,
            path_weight_ramp=100_000,
            path_distance_cap_multiplier=2.0,
            prefix_model='deterministic',
            prefix_loss_weight=1.0,
            prefix_rank=8,
            prefix_sigma_floor=1e-3,
            prefix_scale_floor=1e-3,
            prefix_flow_steps=8,
            eval_num_candidates=8,
            eval_temperature=0.25,
            eval_prefix_selection='sample_one',
            eval_num_prefix_samples=1,
            eval_prefix_temperature=1.0,
            eval_include_deterministic_prefix=False,
            offline_action_free=False,
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
