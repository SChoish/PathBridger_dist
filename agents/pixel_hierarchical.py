"""Official-style pixel HIQL and OTA baselines for offline OGBench.

The implementation follows the public OGBench HIQL and OTA losses while
keeping this repository's agent/checkpoint API.  Pixel inputs use independent
IMPALA-small towers and a compact length-normalized ``phi([s; g])`` rather
than the single shared 512-D PBF latent.
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
from agents.pixel_pathbridger import ImpalaSmallEncoder
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, default_init


def _length_normalize(value: jnp.ndarray) -> jnp.ndarray:
    norm = jnp.linalg.norm(value, axis=-1, keepdims=True)
    return value / jnp.maximum(norm, 1e-6) * jnp.sqrt(value.shape[-1])


def _normal_log_prob(value: jnp.ndarray, mean: jnp.ndarray) -> jnp.ndarray:
    """Log probability under the unit-variance Gaussian used officially."""

    return -0.5 * jnp.sum(
        jnp.square(value - mean) + jnp.log(2.0 * jnp.pi), axis=-1
    )


class GoalRepresentation(nn.Module):
    """State-dependent compact goal representation ``phi([s; g])``."""

    rep_dim: int = 10
    feature_dim: int = 512
    hidden_dims: Sequence[int] = (512, 512, 512)
    layer_norm: bool = True

    @nn.compact
    def __call__(self, state_goal_images: jnp.ndarray) -> jnp.ndarray:
        hidden = ImpalaSmallEncoder(self.feature_dim)(state_goal_images)
        representation = MLP(
            (*self.hidden_dims, self.rep_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(hidden)
        return _length_normalize(representation)


class PixelTwinValue(nn.Module):
    """Twin V(s, phi([s; g])) with a value-specific state tower."""

    rep_dim: int = 10
    feature_dim: int = 512
    hidden_dims: Sequence[int] = (512, 512, 512)
    layer_norm: bool = True

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, goals: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        state_features = ImpalaSmallEncoder(
            self.feature_dim, name='state_encoder'
        )(observations)
        goal_features = GoalRepresentation(
            rep_dim=self.rep_dim,
            feature_dim=self.feature_dim,
            hidden_dims=self.hidden_dims,
            layer_norm=self.layer_norm,
            name='goal_rep',
        )(jnp.concatenate([observations, goals], axis=-1))
        inputs = jnp.concatenate([state_features, goal_features], axis=-1)
        value1 = MLP(
            (*self.hidden_dims, 1),
            layer_norm=self.layer_norm,
            name='value1',
        )(inputs).squeeze(-1)
        value2 = MLP(
            (*self.hidden_dims, 1),
            layer_norm=self.layer_norm,
            name='value2',
        )(inputs).squeeze(-1)
        return value1, value2


class PixelLowActor(nn.Module):
    """Primitive action actor pi_low(encoder_low(s), phi([s; w]))."""

    action_dim: int
    feature_dim: int = 512
    hidden_dims: Sequence[int] = (512, 512, 512)

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, encoded_goals: jnp.ndarray
    ) -> jnp.ndarray:
        state_features = ImpalaSmallEncoder(
            self.feature_dim, name='state_encoder'
        )(observations)
        hidden = MLP(self.hidden_dims, activate_final=True)(
            jnp.concatenate([state_features, encoded_goals], axis=-1)
        )
        return nn.Dense(
            self.action_dim,
            kernel_init=default_init(0.01),
            name='mean',
        )(hidden)


class PixelHighActor(nn.Module):
    """High actor predicting a compact waypoint from early-fused [s; g]."""

    rep_dim: int = 10
    feature_dim: int = 512
    hidden_dims: Sequence[int] = (512, 512, 512)

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, goals: jnp.ndarray
    ) -> jnp.ndarray:
        features = ImpalaSmallEncoder(self.feature_dim)(
            jnp.concatenate([observations, goals], axis=-1)
        )
        hidden = MLP(self.hidden_dims, activate_final=True)(features)
        return nn.Dense(
            self.rep_dim,
            kernel_init=default_init(0.01),
            name='mean',
        )(hidden)


class PixelHierarchicalAgent(flax.struct.PyTreeNode):
    """Shared implementation of official-style pixel HIQL and pixel OTA."""

    rng: Any
    network: TrainState
    config: Any = nonpytree_field()

    @staticmethod
    def _expectile_loss(advantage, difference, expectile):
        weight = jnp.where(advantage >= 0, expectile, 1.0 - expectile)
        return weight * jnp.square(difference)

    def _value_loss(
        self,
        batch,
        grad_params,
        *,
        value_name: str,
        target_name: str,
        observations_key: str,
        goals_key: str,
        rewards_key: str,
        masks_key: str,
        discount: float,
    ):
        next_v1_t, next_v2_t = self.network.select(target_name)(
            batch[observations_key], batch[goals_key]
        )
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        target = batch[rewards_key] + discount * batch[masks_key] * next_v_t
        v1_t, v2_t = self.network.select(target_name)(
            batch['observations'], batch[goals_key]
        )
        advantage = target - (v1_t + v2_t) / 2.0
        q1 = batch[rewards_key] + discount * batch[masks_key] * next_v1_t
        q2 = batch[rewards_key] + discount * batch[masks_key] * next_v2_t
        v1, v2 = self.network.select(value_name)(
            batch['observations'], batch[goals_key], params=grad_params
        )
        loss = (
            self._expectile_loss(
                advantage, q1 - v1, float(self.config['expectile'])
            ).mean()
            + self._expectile_loss(
                advantage, q2 - v2, float(self.config['expectile'])
            ).mean()
        )
        values = (v1 + v2) / 2.0
        return loss, {
            'loss': loss,
            'v_mean': values.mean(),
            'v_max': values.max(),
            'v_min': values.min(),
            'adv_mean': advantage.mean(),
        }

    def _low_actor_loss(self, batch, grad_params, value_name: str):
        v1, v2 = self.network.select(value_name)(
            batch['observations'], batch['low_actor_goals']
        )
        nv1, nv2 = self.network.select(value_name)(
            batch['next_observations'], batch['low_actor_goals']
        )
        advantage = (nv1 + nv2 - v1 - v2) / 2.0
        weights = jnp.minimum(
            jnp.exp(advantage * float(self.config['low_alpha'])), 100.0
        )
        goal_reps = self.network.select('goal_rep')(
            jnp.concatenate(
                [batch['observations'], batch['low_actor_goals']], axis=-1
            ),
            params=grad_params,
        )
        if not bool(self.config['low_actor_rep_grad']):
            goal_reps = jax.lax.stop_gradient(goal_reps)
        means = self.network.select('low_actor')(
            batch['observations'], goal_reps, params=grad_params
        )
        log_prob = _normal_log_prob(batch['actions'], means)
        loss = -(weights * log_prob).mean()
        return loss, {
            'loss': loss,
            'adv_mean': advantage.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean(jnp.square(means - batch['actions'])),
            'action_abs_mean': jnp.mean(jnp.abs(means)),
        }

    def _high_actor_loss(self, batch, grad_params, value_name: str):
        v1, v2 = self.network.select(value_name)(
            batch['observations'], batch['high_actor_goals']
        )
        nv1, nv2 = self.network.select(value_name)(
            batch['high_actor_targets'], batch['high_actor_goals']
        )
        advantage = (nv1 + nv2 - v1 - v2) / 2.0
        weights = jnp.minimum(
            jnp.exp(advantage * float(self.config['high_alpha'])), 100.0
        )
        means = self.network.select('high_actor')(
            batch['observations'], batch['high_actor_goals'], params=grad_params
        )
        targets = jax.lax.stop_gradient(
            self.network.select('goal_rep')(
                jnp.concatenate(
                    [batch['observations'], batch['high_actor_targets']], axis=-1
                )
            )
        )
        log_prob = _normal_log_prob(targets, means)
        loss = -(weights * log_prob).mean()
        return loss, {
            'loss': loss,
            'adv_mean': advantage.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean(jnp.square(means - targets)),
            'rep_norm': jnp.linalg.norm(targets, axis=-1).mean(),
        }

    @partial(jax.jit, static_argnames=())
    def offline_update(self, batch):
        new_rng, _ = jax.random.split(self.rng)
        is_ota = str(self.config['agent_name']) == 'ota'

        def loss_fn(grad_params):
            info = {}
            if is_ota:
                low_value_loss, low_info = self._value_loss(
                    batch,
                    grad_params,
                    value_name='low_value',
                    target_name='target_low_value',
                    observations_key='next_observations',
                    goals_key='value_goals',
                    rewards_key='rewards',
                    masks_key='masks',
                    discount=float(self.config['discount']),
                )
                high_value_loss, high_info = self._value_loss(
                    batch,
                    grad_params,
                    value_name='high_value',
                    target_name='target_high_value',
                    observations_key='high_value_option_observations',
                    goals_key='high_value_goals',
                    rewards_key='high_value_rewards',
                    masks_key='high_value_masks',
                    # OTA intentionally applies one high-level discount, not gamma^n.
                    discount=float(self.config['discount']),
                )
                value_loss = low_value_loss + high_value_loss
                value_name = 'low_value'
                high_actor_value_name = 'high_value'
                info.update({f'low_value/{key}': value for key, value in low_info.items()})
                info.update({f'high_value/{key}': value for key, value in high_info.items()})
            else:
                value_loss, value_info = self._value_loss(
                    batch,
                    grad_params,
                    value_name='value',
                    target_name='target_value',
                    observations_key='next_observations',
                    goals_key='value_goals',
                    rewards_key='rewards',
                    masks_key='masks',
                    discount=float(self.config['discount']),
                )
                value_name = high_actor_value_name = 'value'
                info.update({f'value/{key}': value for key, value in value_info.items()})

            low_loss, low_info = self._low_actor_loss(
                batch, grad_params, value_name
            )
            high_loss, high_info = self._high_actor_loss(
                batch, grad_params, high_actor_value_name
            )
            info.update({f'low_actor/{key}': value for key, value in low_info.items()})
            info.update({f'high_actor/{key}': value for key, value in high_info.items()})
            total = value_loss + low_loss + high_loss
            info['loss/total'] = total
            return total, info

        network, info = self.network.apply_loss_fn(loss_fn)
        tau = float(self.config['tau'])
        pairs = (
            (('low_value', 'target_low_value'), ('high_value', 'target_high_value'))
            if is_ota
            else (('value', 'target_value'),)
        )
        for source, target in pairs:
            target_params = jax.tree_util.tree_map(
                lambda value, old: tau * value + (1.0 - tau) * old,
                network.params[f'modules_{source}'],
                network.params[f'modules_{target}'],
            )
            network = network.replace(
                params=_replace_subtree(network.params, target, target_params)
            )
        return self.replace(rng=new_rng, network=network), info

    @partial(jax.jit, static_argnames=())
    def online_update(self, batch):
        """HIQL/OTA are frozen offline baselines in this protocol."""

        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return self, {'loss/total': zero, 'offline_policy/frozen': 1.0 + zero}

    @partial(jax.jit, static_argnames=('temperature',))
    def sample_actions(
        self,
        observations,
        goals,
        seed=None,
        temperature: float = 0.0,
    ):
        if seed is None:
            seed = jax.random.PRNGKey(0)
        high_seed, low_seed = jax.random.split(seed)
        high_mean = self.network.select('high_actor')(observations, goals)
        high_noise = jax.random.normal(high_seed, high_mean.shape)
        goal_reps = _length_normalize(high_mean + temperature * high_noise)
        low_mean = self.network.select('low_actor')(observations, goal_reps)
        low_noise = jax.random.normal(low_seed, low_mean.shape)
        return jnp.clip(low_mean + temperature * low_noise, -1.0, 1.0)

    @classmethod
    def create(
        cls,
        seed: int,
        example_images,
        action_dim: int,
        config: dict[str, Any] | ml_collections.ConfigDict,
    ) -> 'PixelHierarchicalAgent':
        config = dict(config)
        mode = str(config['agent_name'])
        if mode not in ('hiql', 'ota'):
            raise ValueError("agent_name must be 'hiql' or 'ota'.")
        examples = jnp.asarray(example_images)
        if examples.ndim != 4:
            raise ValueError('Pixel examples must be rank-four NHWC images.')
        hidden = tuple(config['actor_hidden_dims'])
        value_hidden = tuple(config['value_hidden_dims'])
        feature_dim = int(config['feature_dim'])
        rep_dim = int(config['rep_dim'])
        layer_norm = bool(config['layer_norm'])
        goal_rep = GoalRepresentation(rep_dim, feature_dim, value_hidden, layer_norm)
        low_actor = PixelLowActor(int(action_dim), feature_dim, hidden)
        high_actor = PixelHighActor(rep_dim, feature_dim, hidden)
        modules = {
            'goal_rep': goal_rep,
            'low_actor': low_actor,
            'high_actor': high_actor,
        }
        init_args = {
            'goal_rep': jnp.concatenate([examples, examples], axis=-1),
            'low_actor': (examples, jnp.zeros((len(examples), rep_dim))),
            'high_actor': (examples, examples),
        }
        if mode == 'ota':
            for name in ('low_value', 'target_low_value', 'high_value', 'target_high_value'):
                modules[name] = PixelTwinValue(
                    rep_dim, feature_dim, value_hidden, layer_norm
                )
                init_args[name] = (examples, examples)
            target_pairs = (
                ('low_value', 'target_low_value'),
                ('high_value', 'target_high_value'),
            )
        else:
            modules['value'] = PixelTwinValue(
                rep_dim, feature_dim, value_hidden, layer_norm
            )
            modules['target_value'] = PixelTwinValue(
                rep_dim, feature_dim, value_hidden, layer_norm
            )
            init_args['value'] = (examples, examples)
            init_args['target_value'] = (examples, examples)
            target_pairs = (('value', 'target_value'),)
        model = ModuleDict(modules)
        rng = jax.random.PRNGKey(int(seed))
        rng, init_rng = jax.random.split(rng)
        params = model.init(init_rng, **init_args)['params']
        for source, target in target_pairs:
            params = _replace_subtree(params, target, params[f'modules_{source}'])
        network = TrainState.create(
            model, params, tx=optax.adam(float(config['learning_rate']))
        )
        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(config),
        )


PixelHIQLAgent = PixelHierarchicalAgent
PixelOTAAgent = PixelHierarchicalAgent


def get_config(agent_name: str = 'hiql') -> ml_collections.ConfigDict:
    if agent_name not in ('hiql', 'ota'):
        raise ValueError(agent_name)
    config = dict(
        agent_name=agent_name,
        learning_rate=3e-4,
        offline_batch_size=256,
        online_batch_size=256,
        offline_steps=500_000,
        actor_hidden_dims=(512, 512, 512),
        value_hidden_dims=(512, 512, 512),
        feature_dim=512,
        layer_norm=True,
        discount=0.99,
        tau=0.005,
        expectile=0.7,
        low_alpha=3.0,
        high_alpha=3.0,
        subgoal_steps=25,
        rep_dim=10,
        low_actor_rep_grad=True,
        const_std=True,
        value_p_curgoal=0.2,
        value_p_trajgoal=0.5,
        value_p_randomgoal=0.3,
        value_geom_sample=True,
        actor_p_curgoal=0.0,
        actor_p_trajgoal=1.0,
        actor_p_randomgoal=0.0,
        actor_geom_sample=False,
        gc_negative=True,
        p_aug=0.0,
        frame_stack=1,
    )
    if agent_name == 'ota':
        config['abstraction_factor'] = 5
    return ml_collections.ConfigDict(config)


__all__ = [
    'GoalRepresentation',
    'PixelHIQLAgent',
    'PixelOTAAgent',
    'PixelHierarchicalAgent',
    'get_config',
]
