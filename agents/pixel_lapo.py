"""Goal-conditioned pixel LAPO adaptation for continuous-control OGBench.

The implementation preserves LAPO's information flow while changing its
Procgen-specific CNN/discrete-action stage into an explicitly labeled OGBench
adaptation:

1. learn a vector-quantized latent IDM and visual world model from RGB pairs;
2. clone inferred latent actions with a goal-conditioned latent policy;
3. freeze both offline modules and ground latent actions to continuous actions
   with a decoder trained only on newly collected ``(image, action, image')``.
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

from agents.gc_actor_critic import _sample_tanh_gaussian
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP


def _pixels(inputs: jnp.ndarray) -> jnp.ndarray:
    return jnp.asarray(inputs, dtype=jnp.float32) / 255.0


class PixelEncoder(nn.Module):
    feature_dim: int = 256
    channels: Sequence[int] = (32, 64, 128, 128)

    @nn.compact
    def __call__(self, images: jnp.ndarray) -> jnp.ndarray:
        hidden = _pixels(images)
        for index, features in enumerate(self.channels):
            hidden = nn.Conv(
                int(features),
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='SAME',
                name=f'conv_{index}',
            )(hidden)
            hidden = nn.silu(hidden)
        hidden = hidden.reshape((hidden.shape[0], -1))
        hidden = nn.Dense(self.feature_dim, name='projection')(hidden)
        return nn.LayerNorm(name='feature_norm')(hidden)


class PixelDecoder(nn.Module):
    image_shape: tuple[int, int, int]
    channels: Sequence[int] = (128, 64, 32)

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        height, width, output_channels = self.image_shape
        base_height, base_width = height // 16, width // 16
        hidden = nn.Dense(
            base_height * base_width * int(self.channels[0]), name='projection'
        )(features)
        hidden = hidden.reshape(
            (features.shape[0], base_height, base_width, int(self.channels[0]))
        )
        for index, channels in enumerate(self.channels[1:]):
            hidden = nn.ConvTranspose(
                int(channels),
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='SAME',
                name=f'deconv_{index}',
            )(hidden)
            hidden = nn.silu(hidden)
        hidden = nn.ConvTranspose(
            16,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='SAME',
            name='deconv_final_hidden',
        )(hidden)
        hidden = nn.silu(hidden)
        logits = nn.ConvTranspose(
            output_channels,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='SAME',
            name='pixels',
        )(hidden)
        return nn.sigmoid(logits)


class PixelLatentModel(nn.Module):
    image_shape: tuple[int, int, int]
    feature_dim: int = 256
    hidden_dims: Sequence[int] = (256, 256)
    num_codebooks: int = 2
    num_codes: int = 64
    code_dim: int = 16

    @nn.compact
    def __call__(self, observations, next_observations):
        encoder = PixelEncoder(self.feature_dim, name='encoder')
        current_features = encoder(observations)
        next_features = encoder(next_observations)
        prequantized = MLP(
            (*self.hidden_dims, self.num_codebooks * self.code_dim),
            name='latent_idm',
        )(jnp.concatenate([current_features, next_features], axis=-1))
        prequantized = prequantized.reshape(
            (-1, self.num_codebooks, self.code_dim)
        )
        codebook = self.param(
            'codebook',
            nn.initializers.uniform(scale=1.0 / max(self.num_codes, 1)),
            (self.num_codebooks, self.num_codes, self.code_dim),
        )
        distances = jnp.sum(
            jnp.square(
                prequantized[:, :, None, :] - codebook[None, :, :, :]
            ),
            axis=-1,
        )
        indices = jnp.argmin(distances, axis=-1)
        expanded_codebook = jnp.broadcast_to(
            codebook[None, ...],
            (prequantized.shape[0], *codebook.shape),
        )
        quantized = jnp.take_along_axis(
            expanded_codebook,
            indices[:, :, None, None],
            axis=2,
        ).squeeze(axis=2)
        straight_through = prequantized + jax.lax.stop_gradient(
            quantized - prequantized
        )
        predicted_features = MLP(
            (*self.hidden_dims, self.feature_dim), name='world_model'
        )(
            jnp.concatenate(
                [current_features, straight_through.reshape((len(observations), -1))],
                axis=-1,
            )
        )
        reconstruction = PixelDecoder(
            self.image_shape, name='world_decoder'
        )(predicted_features)
        return {
            'current_features': current_features,
            'next_features': next_features,
            'predicted_features': predicted_features,
            'prequantized': prequantized,
            'quantized': quantized,
            'indices': indices,
            'reconstruction': reconstruction,
        }


class GoalConditionedLatentPolicy(nn.Module):
    num_codebooks: int
    num_codes: int
    feature_dim: int = 256
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, observations, goals):
        encoder = PixelEncoder(self.feature_dim, name='encoder')
        observation_features = encoder(observations)
        goal_features = encoder(goals)
        logits = MLP(
            (*self.hidden_dims, self.num_codebooks * self.num_codes),
            name='policy',
        )(jnp.concatenate([observation_features, goal_features], axis=-1))
        return logits.reshape((-1, self.num_codebooks, self.num_codes))


class ContinuousLatentDecoder(nn.Module):
    action_dim: int
    feature_dim: int = 256
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, observations, latent_embeddings):
        features = PixelEncoder(self.feature_dim, name='encoder')(observations)
        inputs = jnp.concatenate(
            [features, latent_embeddings.reshape((len(observations), -1))],
            axis=-1,
        )
        hidden = MLP((*self.hidden_dims,), activate_final=True, name='trunk')(
            inputs
        )
        mean = nn.Dense(self.action_dim, name='mean')(hidden)
        log_std = nn.Dense(self.action_dim, name='log_std')(hidden)
        return mean, jnp.clip(log_std, -5.0, 2.0)


def _gather_codebook(codebook, indices):
    expanded = jnp.broadcast_to(
        codebook[None, ...], (indices.shape[0], *codebook.shape)
    )
    return jnp.take_along_axis(
        expanded, indices[:, :, None, None], axis=2
    ).squeeze(axis=2)


class PixelLAPOAgent(flax.struct.PyTreeNode):
    """Three-stage action-free pixel latent-action agent."""

    rng: Any
    network: TrainState
    config: Any = nonpytree_field()

    def _codebook(self):
        return self.network.params['modules_latent_model']['codebook']

    @partial(jax.jit, static_argnames=('stage',))
    def offline_update(self, batch, *, stage: int):
        if stage not in (1, 2):
            raise ValueError('Pixel LAPO offline stage must be 1 or 2.')

        if stage == 1:
            def loss_fn(params):
                outputs = self.network.select('latent_model')(
                    batch['observations'],
                    batch['next_observations'],
                    params=params,
                )
                target_pixels = _pixels(batch['next_observations'])
                reconstruction_loss = jnp.mean(
                    jnp.square(outputs['reconstruction'] - target_pixels)
                )
                feature_loss = jnp.mean(
                    jnp.square(
                        outputs['predicted_features']
                        - jax.lax.stop_gradient(outputs['next_features'])
                    )
                )
                commitment_loss = jnp.mean(
                    jnp.square(
                        outputs['prequantized']
                        - jax.lax.stop_gradient(outputs['quantized'])
                    )
                )
                codebook_loss = jnp.mean(
                    jnp.square(
                        jax.lax.stop_gradient(outputs['prequantized'])
                        - outputs['quantized']
                    )
                )
                total = (
                    float(self.config['reconstruction_weight'])
                    * reconstruction_loss
                    + float(self.config['feature_weight']) * feature_loss
                    + float(self.config['commitment_weight']) * commitment_loss
                    + codebook_loss
                )
                one_hot = jax.nn.one_hot(
                    outputs['indices'], int(self.config['num_codes'])
                )
                probabilities = jnp.mean(one_hot, axis=(0, 1))
                perplexity = jnp.exp(
                    -jnp.sum(
                        probabilities * jnp.log(probabilities + 1e-8)
                    )
                )
                return total, {
                    'loss/total': total,
                    'stage1/reconstruction_loss': reconstruction_loss,
                    'stage1/feature_loss': feature_loss,
                    'stage1/commitment_loss': commitment_loss,
                    'stage1/codebook_loss': codebook_loss,
                    'stage1/code_perplexity': perplexity,
                }

            network, info = self.network.apply_loss_fn(
                loss_fn, trainable_modules=('latent_model',)
            )
            return self.replace(network=network), info

        def loss_fn(params):
            targets = self.network.select('latent_model')(
                batch['observations'], batch['next_observations']
            )['indices']
            logits = self.network.select('latent_policy')(
                batch['observations'], batch['goals'], params=params
            )
            loss = jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(logits, targets)
            )
            accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == targets)
            return loss, {
                'loss/total': loss,
                'stage2/latent_bc_loss': loss,
                'stage2/latent_accuracy': accuracy,
            }

        network, info = self.network.apply_loss_fn(
            loss_fn, trainable_modules=('latent_policy',)
        )
        return self.replace(network=network), info

    @partial(jax.jit, static_argnames=())
    def online_update(self, batch):
        """Train only the continuous latent-to-action grounding decoder."""

        def loss_fn(params):
            latent_indices = self.network.select('latent_model')(
                batch['observations'], batch['next_observations']
            )['indices']
            latent_embeddings = _gather_codebook(
                self._codebook(), latent_indices
            )
            mean, log_std = self.network.select('decoder')(
                batch['observations'], latent_embeddings, params=params
            )
            actions = jnp.clip(batch['actions'], -0.999, 0.999)
            pre_tanh = jnp.arctanh(actions)
            log_prob = -0.5 * (
                jnp.square((pre_tanh - mean) / jnp.exp(log_std))
                + 2.0 * log_std
                + jnp.log(2.0 * jnp.pi)
            ) - jnp.log(jnp.maximum(1.0 - jnp.square(actions), 1e-6))
            loss = -jnp.mean(jnp.sum(log_prob, axis=-1))
            predicted_actions = jnp.tanh(mean)
            action_l2 = jnp.sqrt(
                jnp.mean(jnp.square(predicted_actions - actions))
            )
            return loss, {
                'loss/total': loss,
                'decoder/nll': loss,
                'decoder/action_l2': action_l2,
                'decoder/action_abs_mean': jnp.mean(jnp.abs(predicted_actions)),
            }

        network, info = self.network.apply_loss_fn(
            loss_fn, trainable_modules=('decoder',)
        )
        return self.replace(network=network), info

    @partial(jax.jit, static_argnames=('temperature',))
    def sample_actions(self, observations, goals, seed=None, temperature=0.0):
        if seed is None:
            seed = jax.random.PRNGKey(0)
        latent_seed, action_seed = jax.random.split(seed)
        logits = self.network.select('latent_policy')(observations, goals)
        if temperature == 0.0:
            latent_indices = jnp.argmax(logits, axis=-1)
        else:
            latent_indices = jax.random.categorical(
                latent_seed, logits / float(temperature), axis=-1
            )
        latent_embeddings = _gather_codebook(
            self._codebook(), latent_indices
        )
        mean, log_std = self.network.select('decoder')(
            observations, latent_embeddings
        )
        actions, _ = _sample_tanh_gaussian(
            mean, log_std, action_seed, float(temperature)
        )
        return jnp.clip(actions, -1.0, 1.0)

    @classmethod
    def create(
        cls,
        seed: int,
        example_images: jnp.ndarray,
        action_dim: int,
        config: dict[str, Any] | ml_collections.ConfigDict,
    ) -> 'PixelLAPOAgent':
        config = dict(config)
        images = jnp.asarray(example_images, dtype=jnp.uint8)
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError('Pixel LAPO examples must have shape [B, H, W, 3].')
        image_shape = tuple(int(value) for value in images.shape[1:])
        if image_shape[0] % 16 or image_shape[1] % 16:
            raise ValueError('Image height and width must be divisible by 16.')
        latent_model = PixelLatentModel(
            image_shape=image_shape,
            feature_dim=int(config['feature_dim']),
            hidden_dims=tuple(config['hidden_dims']),
            num_codebooks=int(config['num_codebooks']),
            num_codes=int(config['num_codes']),
            code_dim=int(config['code_dim']),
        )
        latent_policy = GoalConditionedLatentPolicy(
            num_codebooks=int(config['num_codebooks']),
            num_codes=int(config['num_codes']),
            feature_dim=int(config['feature_dim']),
            hidden_dims=tuple(config['hidden_dims']),
        )
        decoder = ContinuousLatentDecoder(
            int(action_dim),
            feature_dim=int(config['feature_dim']),
            hidden_dims=tuple(config['hidden_dims']),
        )
        model = ModuleDict(
            {
                'latent_model': latent_model,
                'latent_policy': latent_policy,
                'decoder': decoder,
            }
        )
        dummy_embeddings = jnp.zeros(
            (
                len(images),
                int(config['num_codebooks']),
                int(config['code_dim']),
            ),
            dtype=jnp.float32,
        )
        rng = jax.random.PRNGKey(int(seed))
        rng, init_rng = jax.random.split(rng)
        params = model.init(
            init_rng,
            latent_model=(images, images),
            latent_policy=(images, images),
            decoder=(images, dummy_embeddings),
        )['params']
        network = TrainState.create(
            model,
            params,
            tx={
                'latent_model': optax.adam(float(config['stage1_learning_rate'])),
                'latent_policy': optax.adam(float(config['stage2_learning_rate'])),
                'decoder': optax.adam(float(config['decoder_learning_rate'])),
            },
        )
        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(config),
        )


def get_config() -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict(
        dict(
            feature_dim=256,
            hidden_dims=(256, 256),
            num_codebooks=2,
            num_codes=64,
            code_dim=16,
            stage1_learning_rate=3e-4,
            stage2_learning_rate=2e-4,
            decoder_learning_rate=3e-4,
            reconstruction_weight=1.0,
            feature_weight=1.0,
            commitment_weight=0.05,
            stage1_steps=50_000,
            stage2_steps=60_000,
            offline_batch_size=128,
            online_batch_size=128,
        )
    )


__all__ = [
    'ContinuousLatentDecoder',
    'GoalConditionedLatentPolicy',
    'PixelEncoder',
    'PixelLAPOAgent',
    'PixelLatentModel',
    'get_config',
]
