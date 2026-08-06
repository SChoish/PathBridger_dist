"""Strict action-free pixel trajectories and compact online pixel replay."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class PixelEpisodeIndex:
    """Episode boundaries for compact OGBench pixel trajectories."""

    initial_for_state: np.ndarray
    terminal_for_state: np.ndarray
    transition_indices: np.ndarray

    @classmethod
    def from_terminals(cls, terminals: Any) -> 'PixelEpisodeIndex':
        terminals = np.asarray(terminals, dtype=np.float32).reshape(-1)
        terminal_mask = terminals > 0
        candidates = np.flatnonzero(terminal_mask).astype(np.int64)
        # OGBench's compact conversion marks both the last valid transition
        # state and the following terminal observation.  Treat each consecutive
        # run as one boundary and keep its last observation, so the transition
        # into that terminal observation remains available for learning.
        terminal_indices = candidates[
            (candidates == len(terminals) - 1)
            | ~terminal_mask[np.minimum(candidates + 1, len(terminals) - 1)]
        ]
        if not len(terminal_indices) or terminal_indices[-1] != len(terminals) - 1:
            raise ValueError('The final compact pixel observation must be terminal.')
        initial_indices = np.concatenate(
            [np.asarray([0], np.int64), terminal_indices[:-1] + 1]
        )
        initial_for_state = np.empty(len(terminals), np.int64)
        terminal_for_state = np.empty(len(terminals), np.int64)
        transitions = []
        for initial, terminal in zip(initial_indices, terminal_indices):
            initial_for_state[initial : terminal + 1] = initial
            terminal_for_state[initial : terminal + 1] = terminal
            if terminal > initial:
                transitions.append(np.arange(initial, terminal, dtype=np.int64))
        if not transitions:
            raise ValueError('Pixel data must contain at least one valid transition.')
        return cls(
            initial_for_state=initial_for_state,
            terminal_for_state=terminal_for_state,
            transition_indices=np.concatenate(transitions),
        )


def _validate_pixels(array: Any, *, where: str) -> np.ndarray:
    pixels = np.asarray(array)
    if pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise ValueError(
            f'{where} must have shape [N, H, W, 3], got {pixels.shape}.'
        )
    if pixels.dtype != np.uint8:
        raise ValueError(f'{where} must use uint8 storage, got {pixels.dtype}.')
    if pixels.shape[1] % 16 or pixels.shape[2] % 16:
        raise ValueError('Pixel height and width must be divisible by 16.')
    return pixels


class ActionFreePixelTrajectoryData:
    """Offline visual trajectories exposing only pixels and terminals.

    The constructor copies references to exactly two whitelisted fields.  It
    never retains actions, rewards, simulator state, or privileged metadata.
    """

    def __init__(self, dataset: Mapping[str, Any], *, seed: int = 0):
        source_keys = {str(key).lower(): key for key in dataset}
        if not {'observations', 'terminals'} <= set(source_keys):
            raise ValueError('Pixel trajectories require observations and terminals.')
        observations = _validate_pixels(
            dataset[source_keys['observations']], where='observations'
        )
        terminals = np.asarray(
            dataset[source_keys['terminals']], dtype=np.float32
        ).reshape(-1)
        if len(observations) != len(terminals):
            raise ValueError('Pixel observations and terminals must have equal lengths.')
        self.observations = observations.view()
        self.observations.setflags(write=False)
        self.terminals = terminals.view()
        self.terminals.setflags(write=False)
        self.episodes = PixelEpisodeIndex.from_terminals(terminals)
        self.rng = np.random.default_rng(int(seed))
        self.offline_fields_seen = ('observations', 'terminals')

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.observations.shape[1:])

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        choices = self.rng.integers(
            0, len(self.episodes.transition_indices), size=int(batch_size)
        )
        indices = self.episodes.transition_indices[choices]
        finals = self.episodes.terminal_for_state[indices]
        offsets = 1 + np.floor(
            self.rng.random(len(indices)) * (finals - indices)
        ).astype(np.int64)
        goal_indices = indices + offsets
        successes = goal_indices == indices + 1
        return {
            'observations': self.observations[indices],
            'next_observations': self.observations[indices + 1],
            'goals': self.observations[goal_indices],
            'rewards': successes.astype(np.float32) - 1.0,
            'masks': 1.0 - successes.astype(np.float32),
            'indices': indices,
            'goal_indices': goal_indices,
        }


class PixelReplayBuffer:
    """Memory-conscious uint8 replay for continuous-action visual control."""

    def __init__(
        self,
        capacity: int,
        image_shape: tuple[int, int, int],
        action_shape: tuple[int, ...],
        *,
        seed: int = 0,
    ):
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError('Replay capacity must be positive.')
        image_shape = tuple(int(value) for value in image_shape)
        _validate_pixels(
            np.empty((1, *image_shape), dtype=np.uint8), where='image_shape'
        )
        self.observations = np.empty((self.capacity, *image_shape), np.uint8)
        self.next_observations = np.empty_like(self.observations)
        self.goals = np.empty_like(self.observations)
        self.actions = np.empty((self.capacity, *action_shape), np.float32)
        self.rewards = np.empty((self.capacity,), np.float32)
        self.masks = np.empty((self.capacity,), np.float32)
        self.pointer = 0
        self.size = 0
        self.rng = np.random.default_rng(int(seed))

    @property
    def allocated_bytes(self) -> int:
        return int(
            self.observations.nbytes
            + self.next_observations.nbytes
            + self.goals.nbytes
            + self.actions.nbytes
            + self.rewards.nbytes
            + self.masks.nbytes
        )

    def add(
        self,
        *,
        observation: Any,
        action: Any,
        next_observation: Any,
        goal: Any,
        reward: float,
        mask: float,
    ) -> None:
        slot = self.pointer
        self.observations[slot] = observation
        self.actions[slot] = action
        self.next_observations[slot] = next_observation
        self.goals[slot] = goal
        self.rewards[slot] = reward
        self.masks[slot] = mask
        self.pointer = (slot + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if self.size < 1:
            raise ValueError('Cannot sample an empty pixel replay.')
        indices = self.rng.integers(0, self.size, size=int(batch_size))
        return {
            'observations': self.observations[indices],
            'actions': self.actions[indices],
            'next_observations': self.next_observations[indices],
            'goals': self.goals[indices],
            'rewards': self.rewards[indices],
            'masks': self.masks[indices],
        }


__all__ = [
    'ActionFreePixelTrajectoryData',
    'PixelEpisodeIndex',
    'PixelReplayBuffer',
]
