"""Component-wise checkpoints for action-free offline-to-online runs."""

from __future__ import annotations

import dataclasses
import os
import pickle
from pathlib import Path
from typing import Any

import flax

from agents.oso_decqn import OSODecQNAgent


def _agent_state(agent: Any) -> dict[str, Any]:
    if isinstance(agent, OSODecQNAgent):
        return {
            'kind': 'oso',
            'state_policy': flax.serialization.to_state_dict(agent.state_policy),
            'online_actor': flax.serialization.to_state_dict(agent.online_actor),
            'idm': flax.serialization.to_state_dict(agent.idm),
            'online_steps': int(agent.online_steps),
        }
    return {'kind': 'flax', 'state': flax.serialization.to_state_dict(agent)}


def _restore_agent(template: Any, state: dict[str, Any]) -> Any:
    if state['kind'] == 'oso':
        if not isinstance(template, OSODecQNAgent):
            raise TypeError('OSO checkpoint requires an OSODecQNAgent template.')
        return dataclasses.replace(
            template,
            state_policy=flax.serialization.from_state_dict(
                template.state_policy, state['state_policy']
            ),
            online_actor=flax.serialization.from_state_dict(
                template.online_actor, state['online_actor']
            ),
            idm=flax.serialization.from_state_dict(template.idm, state['idm']),
            online_steps=int(state['online_steps']),
        )
    return flax.serialization.from_state_dict(template, state['state'])


def save_af_checkpoint(
    path: str | os.PathLike[str],
    *,
    algorithm: str,
    agent: Any,
    step: int,
    config: dict[str, Any],
    metadata: dict[str, Any],
    planner: Any | None = None,
    planner_config: dict[str, Any] | None = None,
    replay_state: dict[str, Any] | None = None,
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format_version': 1,
        'algorithm': str(algorithm),
        'step': int(step),
        'config': dict(config),
        'metadata': dict(metadata),
        'agent': _agent_state(agent),
        'planner': None if planner is None else flax.serialization.to_state_dict(planner),
        'planner_config': planner_config,
        'replay': replay_state,
    }
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)
    return str(path)


def load_af_checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).open('rb') as file:
        payload = pickle.load(file)
    if not isinstance(payload, dict) or payload.get('format_version') != 1:
        raise ValueError(f'Invalid action-free checkpoint: {path}')
    return payload


def restore_af_agent(template: Any, payload: dict[str, Any]) -> Any:
    return _restore_agent(template, payload['agent'])


__all__ = ['load_af_checkpoint', 'restore_af_agent', 'save_af_checkpoint']
