"""Goal-conditioned SAC and TD3 used by the shared online protocol."""

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
from utils.networks import MLP, default_init


class GaussianActor(nn.Module):
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, observations: jnp.ndarray, goals: jnp.ndarray):
        inputs = jnp.concatenate([observations, goals], axis=-1)
        hidden = MLP((*self.hidden_dims,), activate_final=True)(inputs)
        mean = nn.Dense(
            self.action_dim,
            kernel_init=default_init(0.01),
            name='mean',
        )(hidden)
        log_std = nn.Dense(
            self.action_dim,
            kernel_init=default_init(0.01),
            name='log_std',
        )(hidden)
        return mean, jnp.clip(log_std, -5.0, 2.0)


class DeterministicActor(nn.Module):
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, observations: jnp.ndarray, goals: jnp.ndarray):
        inputs = jnp.concatenate([observations, goals], axis=-1)
        return jnp.tanh(
            MLP(
                (*self.hidden_dims, self.action_dim),
                activate_final=False,
                kernel_init=default_init(0.01),
            )(inputs)
        )


class TwinGoalCritic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        inputs = jnp.concatenate([observations, goals, actions], axis=-1)
        q1 = MLP((*self.hidden_dims, 1), name='q1')(inputs).squeeze(-1)
        q2 = MLP((*self.hidden_dims, 1), name='q2')(inputs).squeeze(-1)
        return q1, q2


def _sample_tanh_gaussian(
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    seed: jax.Array,
    temperature: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    temperature = jnp.asarray(temperature, dtype=mean.dtype)
    noise = jax.random.normal(seed, mean.shape, dtype=mean.dtype)
    pre_tanh = mean + temperature * jnp.exp(log_std) * noise
    actions = jnp.tanh(pre_tanh)
    gaussian_log_prob = -0.5 * (
        jnp.square((pre_tanh - mean) / jnp.exp(log_std))
        + 2.0 * log_std
        + jnp.log(2.0 * jnp.pi)
    )
    correction = jnp.log(jnp.maximum(1.0 - jnp.square(actions), 1e-6))
    log_prob = jnp.sum(gaussian_log_prob - correction, axis=-1)
    return actions, log_prob


def _replace_subtree(params: Any, name: str, value: Any) -> Any:
    frozen = isinstance(params, flax.core.FrozenDict)
    mutable = flax.core.unfreeze(params) if frozen else dict(params)
    mutable[f'modules_{name}'] = value
    return flax.core.freeze(mutable) if frozen else mutable


class GoalConditionedActorCritic(flax.struct.PyTreeNode):
    """A compact GC-SAC/GC-TD3 implementation with HER-ready batches."""

    rng: Any
    network: TrainState
    config: Any = nonpytree_field()

    def _actor_sample(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        seed: jax.Array,
        *,
        params: Any | None = None,
        target: bool = False,
        temperature: float = 1.0,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        name = 'target_actor' if target else 'actor'
        if self.config['algorithm'] == 'sac':
            mean, log_std = self.network.select(name)(
                observations, goals, params=params
            )
            return _sample_tanh_gaussian(mean, log_std, seed, temperature)
        actions = self.network.select(name)(observations, goals, params=params)
        return actions, jnp.zeros(actions.shape[:-1], dtype=actions.dtype)

    def critic_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
        seed: jax.Array,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        next_actions, next_log_prob = self._actor_sample(
            batch['next_observations'],
            batch['goals'],
            seed,
            target=self.config['algorithm'] == 'td3',
            temperature=1.0,
        )
        if self.config['algorithm'] == 'td3':
            noise = jnp.clip(
                jax.random.normal(seed, next_actions.shape)
                * float(self.config['target_noise']),
                -float(self.config['target_noise_clip']),
                float(self.config['target_noise_clip']),
            )
            next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)
        target_q1, target_q2 = self.network.select('target_critic')(
            batch['next_observations'], batch['goals'], next_actions
        )
        target_q = jnp.minimum(target_q1, target_q2)
        if self.config['algorithm'] == 'sac':
            target_q = target_q - float(self.config['entropy_coefficient']) * next_log_prob
        target = batch['rewards'] + (
            float(self.config['discount']) * batch['masks'] * target_q
        )
        target = jax.lax.stop_gradient(target)
        q1, q2 = self.network.select('critic')(
            batch['observations'],
            batch['goals'],
            batch['actions'],
            params=grad_params,
        )
        loss = jnp.mean(jnp.square(q1 - target) + jnp.square(q2 - target))
        return loss, {
            'critic/loss': loss,
            'critic/q_mean': 0.5 * (q1.mean() + q2.mean()),
            'critic/q_std': jnp.stack([q1, q2]).std(),
            'critic/target_mean': target.mean(),
            'critic/target_std': target.std(),
        }

    def actor_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
        seed: jax.Array,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        actions, log_prob = self._actor_sample(
            batch['observations'],
            batch['goals'],
            seed,
            params=grad_params,
            temperature=1.0,
        )
        # Critic parameters are constants here, while gradients still flow
        # through its action input into the actor.
        q1, q2 = self.network.select('critic')(
            batch['observations'], batch['goals'], actions
        )
        q = jnp.minimum(q1, q2)
        if self.config['algorithm'] == 'sac':
            loss = jnp.mean(
                float(self.config['entropy_coefficient']) * log_prob - q
            )
        else:
            loss = -q1.mean()
        return loss, {
            'actor/loss': loss,
            'actor/action_abs_mean': jnp.abs(actions).mean(),
            'actor/entropy': -log_prob.mean(),
        }

    def update(
        self, batch: dict[str, jnp.ndarray]
    ) -> tuple['GoalConditionedActorCritic', dict[str, jnp.ndarray]]:
        update_actor = self.config['algorithm'] == 'sac' or (
            int(self.network.step) % int(self.config['policy_delay']) == 0
        )
        return self._update_impl(batch, update_actor=update_actor)

    @partial(jax.jit, static_argnames=('update_actor',))
    def _update_impl(
        self,
        batch: dict[str, jnp.ndarray],
        *,
        update_actor: bool,
    ) -> tuple['GoalConditionedActorCritic', dict[str, jnp.ndarray]]:
        new_rng, critic_seed, actor_seed = jax.random.split(self.rng, 3)

        def loss_fn(grad_params):
            critic_loss, info = self.critic_loss(batch, grad_params, critic_seed)
            if update_actor:
                actor_loss, actor_info = self.actor_loss(
                    batch, grad_params, actor_seed
                )
            else:
                actor_loss = jnp.zeros((), dtype=critic_loss.dtype)
                actor_info = {
                    'actor/loss': actor_loss,
                    'actor/action_abs_mean': actor_loss,
                    'actor/entropy': actor_loss,
                }
            total = critic_loss + actor_loss
            return total, {'loss/total': total, **info, **actor_info}

        trainable_modules = ('critic', 'actor') if update_actor else ('critic',)
        network, info = self.network.apply_loss_fn(
            loss_fn, trainable_modules=trainable_modules
        )
        tau = float(self.config['tau'])
        target_pairs = [('critic', 'target_critic')]
        if update_actor:
            target_pairs.append(('actor', 'target_actor'))
        for source, target in target_pairs:
            updated = jax.tree_util.tree_map(
                lambda value, target_value: tau * value + (1.0 - tau) * target_value,
                network.params[f'modules_{source}'],
                network.params[f'modules_{target}'],
            )
            network = network.replace(
                params=_replace_subtree(network.params, target, updated)
            )
        info['actor/updated'] = jnp.asarray(update_actor, dtype=jnp.float32)
        return self.replace(rng=new_rng, network=network), info

    online_update = update
    offline_update = update

    @partial(jax.jit, static_argnames=('temperature',))
    def sample_actions(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        seed: jax.Array | None = None,
        temperature: float = 0.0,
    ) -> jnp.ndarray:
        if seed is None:
            seed = jax.random.PRNGKey(0)
        actions, _ = self._actor_sample(
            observations,
            goals,
            seed,
            temperature=temperature,
        )
        return jnp.clip(actions, -1.0, 1.0)

    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations: jnp.ndarray,
        ex_actions: jnp.ndarray,
        config: dict[str, Any] | ml_collections.ConfigDict,
    ) -> 'GoalConditionedActorCritic':
        config = dict(config)
        algorithm = str(config.get('algorithm', 'sac')).lower()
        if algorithm not in ('sac', 'td3'):
            raise ValueError("algorithm must be 'sac' or 'td3'.")
        config['algorithm'] = algorithm
        observations = jnp.asarray(ex_observations, dtype=jnp.float32)
        actions = jnp.asarray(ex_actions, dtype=jnp.float32)
        goals = observations
        if observations.ndim != 2 or actions.ndim != 2:
            raise ValueError('Example observations/actions must be rank two.')
        hidden_dims = tuple(config['hidden_dims'])
        action_dim = int(actions.shape[-1])
        actor_def: nn.Module
        if algorithm == 'sac':
            actor_def = GaussianActor(action_dim, hidden_dims)
            actor_args = (observations, goals)
        else:
            actor_def = DeterministicActor(action_dim, hidden_dims)
            actor_args = (observations, goals)
        critic_def = TwinGoalCritic(hidden_dims)
        modules = {
            'actor': actor_def,
            'target_actor': actor_def,
            'critic': critic_def,
            'target_critic': critic_def,
        }
        model = ModuleDict(modules)
        init_args = {
            'actor': actor_args,
            'target_actor': actor_args,
            'critic': (observations, goals, actions),
            'target_critic': (observations, goals, actions),
        }
        rng = jax.random.PRNGKey(int(seed))
        rng, init_rng = jax.random.split(rng)
        params = model.init(init_rng, **init_args)['params']
        params = _replace_subtree(params, 'target_actor', params['modules_actor'])
        params = _replace_subtree(params, 'target_critic', params['modules_critic'])
        learning_rate = float(config['learning_rate'])
        network = TrainState.create(
            model,
            params,
            tx={
                'actor': optax.adam(learning_rate),
                'critic': optax.adam(learning_rate),
            },
        )
        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(config),
        )


def get_config(algorithm: str = 'sac') -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict(
        dict(
            algorithm=algorithm,
            learning_rate=3e-4,
            hidden_dims=(256, 256),
            discount=0.99,
            tau=0.005,
            entropy_coefficient=0.1,
            policy_delay=2,
            target_noise=0.2,
            target_noise_clip=0.5,
            collection_noise_initial=0.2,
            collection_noise_final=0.05,
            collection_noise_decay_steps=100_000,
            batch_size=256,
            her_probability=0.8,
        )
    )


__all__ = [
    'DeterministicActor',
    'GaussianActor',
    'GoalConditionedActorCritic',
    'TwinGoalCritic',
    'get_config',
]
