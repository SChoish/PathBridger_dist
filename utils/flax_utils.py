"""Minimal Flax training-state and checkpoint utilities."""

from __future__ import annotations

import functools
import os
import pickle
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import flax
import flax.linen as nn
import jax
import numpy as np
import optax

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


class ModuleDict(nn.Module):
    """Expose a dictionary of Flax modules through one shared parameter tree."""

    modules: dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name: str | None = None, **kwargs):
        if name is not None:
            return self.modules[name](*args, **kwargs)

        if kwargs.keys() != self.modules.keys():
            raise ValueError(
                'Initialization arguments must exactly match the module names: '
                f'expected {tuple(self.modules)}, got {tuple(kwargs)}.'
            )

        outputs = {}
        for module_name, module_args in kwargs.items():
            module = self.modules[module_name]
            if isinstance(module_args, Mapping):
                outputs[module_name] = module(**module_args)
            elif isinstance(module_args, Sequence):
                outputs[module_name] = module(*module_args)
            else:
                outputs[module_name] = module(module_args)
        return outputs


class TrainState(flax.struct.PyTreeNode):
    """Parameters, optimizer state, and the callable Flax module definition."""

    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(
        cls,
        model_def: nn.Module,
        params: Any,
        tx: optax.GradientTransformation | None = None,
        **kwargs,
    ):
        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=None if tx is None else tx.init(params),
            **kwargs,
        )

    def __call__(self, *args, params=None, method: str | None = None, **kwargs):
        variables = {'params': self.params if params is None else params}
        method_fn = None if method is None else getattr(self.model_def, method)
        return self.apply_fn(variables, *args, method=method_fn, **kwargs)

    def select(self, name: str):
        """Return a callable bound to one module in a :class:`ModuleDict`."""
        return functools.partial(self, name=name)

    def apply_gradients(self, grads, **kwargs):
        if self.tx is None:
            raise ValueError('Cannot apply gradients without an optimizer.')
        updates, opt_state = self.tx.update(grads, self.opt_state, self.params)
        params = optax.apply_updates(self.params, updates)
        return self.replace(step=self.step + 1, params=params, opt_state=opt_state, **kwargs)

    def apply_loss_fn(self, loss_fn):
        """Differentiate ``loss_fn(params)`` and apply one optimizer update."""
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)
        return self.apply_gradients(grads), info


_CHECKPOINT_NAME = re.compile(r'^params_(\d+)\.pkl$')


def resolve_checkpoint(
    path: str | os.PathLike[str],
    step: int = 0,
) -> tuple[Path, int]:
    """Resolve a checkpoint path and its unambiguous training step.

    A zero step means "infer from an exact ``params_<step>.pkl`` filename".
    Checkpoint directories always require an explicit positive step.
    """

    path = Path(path)
    if path.suffix == '.pkl':
        match = _CHECKPOINT_NAME.fullmatch(path.name)
        inferred_step = int(match.group(1)) if match is not None else None
        requested_step = int(step)
        if requested_step < 0:
            raise ValueError('Checkpoint step cannot be negative.')
        if requested_step == 0:
            if inferred_step is None:
                raise ValueError(
                    'An exact checkpoint with a nonstandard filename requires '
                    'an explicit positive step.'
                )
            requested_step = inferred_step
        elif inferred_step is not None and requested_step != inferred_step:
            raise ValueError(
                f'Checkpoint filename implies step {inferred_step}, but '
                f'step {requested_step} was requested.'
            )
        return path, requested_step

    step = int(step)
    if step < 1:
        raise ValueError(
            'A checkpoint directory requires an explicit positive step.'
        )
    return path / f'params_{step}.pkl', step


def _checkpoint_metadata(agent: Any) -> dict[str, Any]:
    """Serialize architecture settings and residual normalization metadata."""

    config = agent.config
    prefix_model = str(config['prefix_model'])
    metadata = {
        'endpoint_distribution': str(config['endpoint_distribution']),
        'prefix_model': prefix_model,
        'horizon': int(config['horizon']),
        'state_scale': np.asarray(agent.state_scale, dtype=np.float32),
        'offline_action_free': bool(config.get('offline_action_free', False)),
    }
    if prefix_model == 'low_rank_gaussian':
        metadata.update({
            'prefix_rank': int(config['prefix_rank']),
            'prefix_sigma_floor': float(config['prefix_sigma_floor']),
        })
    elif prefix_model == 'joint_flow':
        metadata['prefix_flow_steps'] = int(config['prefix_flow_steps'])
    return metadata


def _validate_checkpoint_metadata(agent: Any, metadata: Any) -> None:
    """Reject stochastic checkpoint/config mismatches before deserialization."""

    prefix_model = str(agent.config['prefix_model'])
    if metadata is None:
        if prefix_model != 'deterministic':
            raise ValueError(
                'A stochastic prefix checkpoint requires architecture metadata.'
            )
        return
    if not isinstance(metadata, Mapping):
        raise ValueError('Invalid PathBridger checkpoint metadata.')

    expected = _checkpoint_metadata(agent)
    keys = ['endpoint_distribution', 'prefix_model', 'horizon', 'offline_action_free']
    if prefix_model == 'low_rank_gaussian':
        keys.extend(['prefix_rank', 'prefix_sigma_floor'])
    elif prefix_model == 'joint_flow':
        keys.append('prefix_flow_steps')
    mismatches = [
        key
        for key in keys
        if key not in metadata or metadata[key] != expected[key]
    ]
    if mismatches:
        details = ', '.join(
            f'{key}: checkpoint={metadata.get(key)!r}, config={expected[key]!r}'
            for key in mismatches
        )
        raise ValueError(f'Checkpoint architecture mismatch ({details}).')

    checkpoint_scale = np.asarray(metadata.get('state_scale'), dtype=np.float32)
    expected_scale = np.asarray(expected['state_scale'], dtype=np.float32)
    if checkpoint_scale.shape != expected_scale.shape or not np.allclose(
        checkpoint_scale,
        expected_scale,
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError(
            'Checkpoint state_scale does not match the current training dataset.'
        )


def save_agent(agent: Any, save_dir: str | os.PathLike[str], step: int) -> str:
    """Save the unified agent and host sampler states."""
    save_path, _ = resolve_checkpoint(save_dir, step)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'agent': flax.serialization.to_state_dict(agent),
        'metadata': _checkpoint_metadata(agent),
        'numpy_random_state': np.random.get_state(),
        'python_random_state': random.getstate(),
    }
    temporary_path = save_path.with_suffix('.pkl.tmp')
    with temporary_path.open('wb') as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_path, save_path)
    return str(save_path)


def restore_agent(
    agent: Any,
    restore_path: str | os.PathLike[str],
    step: int = 0,
) -> Any:
    """Restore an agent and, when present, its host sampler states."""
    checkpoint_path, _ = resolve_checkpoint(restore_path, step)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    with checkpoint_path.open('rb') as file:
        payload = pickle.load(file)
    if not isinstance(payload, dict) or 'agent' not in payload:
        raise ValueError(f'Invalid PathBridger checkpoint: {checkpoint_path}')
    _validate_checkpoint_metadata(agent, payload.get('metadata'))
    agent_state = payload['agent']
    if isinstance(agent_state, Mapping) and 'state_scale' not in agent_state:
        if str(agent.config['prefix_model']) != 'deterministic':
            raise ValueError('Legacy checkpoints cannot restore a stochastic prefix model.')
        agent_state = dict(agent_state)
        agent_state['state_scale'] = flax.serialization.to_state_dict(
            agent.state_scale
        )
    restored_agent = flax.serialization.from_state_dict(agent, agent_state)
    if 'numpy_random_state' in payload:
        np.random.set_state(payload['numpy_random_state'])
    if 'python_random_state' in payload:
        random.setstate(payload['python_random_state'])
    return restored_agent


__all__ = [
    'ModuleDict',
    'TrainState',
    'nonpytree_field',
    'resolve_checkpoint',
    'restore_agent',
    'save_agent',
]
