"""Aligned PBF bridge supervision and PathFlower Triangle-Q batches."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np


class Dataset(Mapping[str, np.ndarray]):
    """Immutable mapping around a compact state-only offline dataset."""

    @classmethod
    def create(cls, freeze: bool = True, **fields: Any):
        return cls(fields, freeze=freeze)

    def __init__(self, fields: Mapping[str, Any], *, freeze: bool = True):
        if 'observations' not in fields:
            raise ValueError("Dataset requires an 'observations' field.")
        arrays = {}
        size = None
        for key, value in fields.items():
            array = np.asarray(value)
            if array.ndim == 0:
                raise ValueError(f'Dataset field {key!r} needs a batch dimension.')
            if size is None:
                size = len(array)
            elif len(array) != size:
                raise ValueError(f'Dataset field {key!r} has inconsistent length.')
            if freeze:
                array = array.view()
                array.setflags(write=False)
            arrays[str(key)] = array
        if not size:
            raise ValueError('Dataset cannot be empty.')
        self._data = MappingProxyType(arrays)
        self.size = int(size)

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def get_random_idxs(self, count):
        return np.random.randint(self.size, size=int(count), dtype=np.int64)


@dataclass(frozen=True)
class PathBridgerDatasetConfig:
    horizon: int
    sequence_horizon: int
    action_chunk_horizon: int
    discount: float
    actor_p: tuple[float, float, float, float]
    value_geom_sample: bool = True


def _config_get(config: Any, key: str):
    if isinstance(config, Mapping):
        if key not in config:
            raise ValueError(f'PathBridger-TriangleQ config is missing {key!r}.')
        return config[key]
    try:
        return getattr(config, key)
    except AttributeError as error:
        raise ValueError(f'PathBridger-TriangleQ config is missing {key!r}.') from error


def _validate_goal_mix(value):
    probabilities = np.asarray(value, dtype=np.float64)
    if probabilities.shape != (4,):
        raise ValueError('actor_p must be (p_cur,p_geom,p_traj,p_rand).')
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError('actor_p probabilities must be finite and non-negative.')
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-6, rtol=0):
        raise ValueError('actor_p probabilities must sum to one.')
    if probabilities[1] > 0 and probabilities[2] > 0:
        raise ValueError('actor_p cannot combine geometric and uniform future sampling.')
    return tuple(float(item) for item in probabilities)


@dataclass
class PathBridgerDataset:
    """Sample endpoint/bridge/IDM and triangular-Q fields at shared starts."""

    dataset: Dataset
    config: Any

    def __post_init__(self):
        self.horizon = int(_config_get(self.config, 'horizon'))
        self.sequence_horizon = int(_config_get(self.config, 'sequence_horizon'))
        self.action_horizon = int(_config_get(self.config, 'action_chunk_horizon'))
        self.discount = float(_config_get(self.config, 'discount'))
        self.actor_p = _validate_goal_mix(_config_get(self.config, 'actor_p'))
        self.value_geom_sample = bool(_config_get(self.config, 'value_geom_sample'))
        if self.horizon < self.action_horizon:
            raise ValueError('horizon must cover action_chunk_horizon.')
        if self.sequence_horizon < self.action_horizon:
            raise ValueError('sequence_horizon must cover action_chunk_horizon.')
        if not 0.0 < self.discount < 1.0:
            raise ValueError('discount must lie in (0,1).')

        observations = np.asarray(self.dataset['observations'])
        actions = np.asarray(self.dataset['actions'])
        terminals = np.asarray(self.dataset['terminals']).reshape(-1)
        if observations.ndim != 2 or actions.ndim != 2:
            raise ValueError('PathBridger-TriangleQ requires rank-2 states and actions.')
        self.terminal_locs = np.flatnonzero(terminals > 0).astype(np.int64)
        if not len(self.terminal_locs) or int(self.terminal_locs[-1]) != self.dataset.size - 1:
            raise ValueError('The final compact-dataset observation must be terminal.')
        self.initial_locs = np.concatenate([
            np.asarray([0], dtype=np.int64), self.terminal_locs[:-1] + 1,
        ])
        self._final_for_idx = np.empty(self.dataset.size, dtype=np.int64)
        required_horizon = max(
            self.horizon,
            self.sequence_horizon,
            self.action_horizon,
        )
        valid_parts = []
        for start, final in zip(self.initial_locs, self.terminal_locs):
            self._final_for_idx[start : final + 1] = final
            last = int(final) - required_horizon
            if last >= int(start):
                valid_parts.append(np.arange(start, last + 1, dtype=np.int64))
        if not valid_parts:
            raise ValueError(f'No episode admits required horizon {required_horizon}.')
        self.valid_starts = np.concatenate(valid_parts)
        self._path_offsets = np.arange(self.horizon + 1, dtype=np.int64)
        self._action_offsets = np.arange(self.action_horizon, dtype=np.int64)

    def _observations(self, idxs):
        return np.asarray(self.dataset['observations'][idxs], dtype=np.float32)

    @staticmethod
    def _uniform_positive_offsets(max_offsets):
        max_offsets = np.asarray(max_offsets, dtype=np.int64)
        return 1 + np.floor(np.random.random(len(max_offsets)) * max_offsets).astype(np.int64)

    def _sample_endpoint_goals(self, idxs, finals):
        p_cur, p_geom, p_traj, p_rand = self.actor_p
        draws = np.random.random(len(idxs))
        current = draws < p_cur
        future = (draws >= p_cur) & (draws < p_cur + p_geom + p_traj)
        random = ~(current | future)
        goal_idxs = idxs.copy()
        if np.any(future):
            if p_geom > 0:
                offsets = np.random.geometric(1.0 - self.discount, size=int(future.sum()))
            else:
                offsets = self._uniform_positive_offsets(finals[future] - idxs[future])
            goal_idxs[future] = np.minimum(idxs[future] + offsets, finals[future])
        if np.any(random):
            goal_idxs[random] = self.dataset.get_random_idxs(int(random.sum()))
        return goal_idxs.astype(np.int64), random

    def _sample_value_goals(self, idxs, finals):
        if self.value_geom_sample:
            offsets = np.random.geometric(1.0 - self.discount, size=len(idxs))
            return np.minimum(idxs + offsets, finals).astype(np.int64)
        return (
            idxs + self._uniform_positive_offsets(finals - idxs)
        ).astype(np.int64)

    def _action_chunks(self, idxs):
        finals = self._final_for_idx[idxs]
        if np.any(idxs + self.action_horizon > finals):
            raise ValueError('Action chunk crosses an episode boundary.')
        action_idxs = idxs[:, None] + self._action_offsets[None, :]
        actions = np.asarray(self.dataset['actions'][action_idxs], dtype=np.float32)
        return actions.reshape(len(idxs), -1)

    def _validate_starts(self, idxs):
        idxs = np.asarray(idxs, dtype=np.int64)
        if idxs.ndim != 1 or not len(idxs):
            raise ValueError('idxs must be a non-empty rank-1 array.')
        if np.any(idxs < 0) or np.any(idxs >= self.dataset.size):
            raise IndexError(f'Sample starts must lie in [0,{self.dataset.size}).')
        required = max(self.horizon, self.sequence_horizon, self.action_horizon)
        if np.any(idxs + required > self._final_for_idx[idxs]):
            raise ValueError('A training window cannot cross an episode boundary.')
        return idxs

    def sample(self, batch_size: int, idxs=None):
        batch_size = int(batch_size)
        if idxs is None:
            choices = np.random.randint(len(self.valid_starts), size=batch_size)
            idxs = self.valid_starts[choices]
        idxs = self._validate_starts(idxs)
        if len(idxs) != batch_size:
            raise ValueError('batch_size does not match explicit idxs.')

        finals = self._final_for_idx[idxs]
        endpoint_goal_idxs, random_endpoint = self._sample_endpoint_goals(idxs, finals)
        endpoint_target_idxs = np.minimum(idxs + self.horizon, endpoint_goal_idxs)
        endpoint_target_idxs[random_endpoint] = idxs[random_endpoint] + self.horizon
        trajectory_idxs = np.minimum(
            idxs[:, None] + self._path_offsets[None, :],
            endpoint_target_idxs[:, None],
        )

        value_goal_idxs = self._sample_value_goals(idxs, finals)
        value_offsets = (value_goal_idxs - idxs).astype(np.float32)
        split_low = idxs + 1
        split_high = np.minimum(value_goal_idxs - 1, finals - self.action_horizon)
        triangle_valid = (value_goal_idxs > idxs + 1) & (split_high >= split_low)
        split_idxs = idxs.copy()
        if np.any(triangle_valid):
            lo = split_low[triangle_valid]
            hi = split_high[triangle_valid]
            split_idxs[triangle_valid] = lo + np.random.randint(
                0, hi - lo + 1, size=int(triangle_valid.sum()),
            )

        max_base_offsets = np.maximum(
            1,
            np.minimum(self.sequence_horizon, finals - idxs),
        )
        base_offsets = self._uniform_positive_offsets(max_base_offsets)
        base_goal_idxs = idxs + base_offsets

        return {
            'observations': self._observations(idxs),
            'next_observations': self._observations(idxs + 1),
            'actions': np.asarray(self.dataset['actions'][idxs], dtype=np.float32),
            'trajectory': self._observations(trajectory_idxs),
            'endpoint_goals': self._observations(endpoint_goal_idxs),
            'endpoint_targets': self._observations(endpoint_target_idxs),
            'value_goals': self._observations(value_goal_idxs),
            'value_offsets': value_offsets,
            'action_chunk_actions': self._action_chunks(idxs),
            'valids': np.ones((batch_size, self.action_horizon), dtype=np.float32),
            'trl_base_goals': self._observations(base_goal_idxs),
            'trl_base_offsets': base_offsets.astype(np.float32),
            'trl_split_observations': self._observations(split_idxs),
            'trl_split_goals': self._observations(split_idxs),
            'trl_split_action_chunk_actions': self._action_chunks(split_idxs),
            'trl_split_offsets': (split_idxs - idxs).astype(np.float32),
            'trl_valid_mask': triangle_valid.astype(np.float32),
        }


__all__ = ['Dataset', 'PathBridgerDataset', 'PathBridgerDatasetConfig']
