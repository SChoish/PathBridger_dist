"""Shared state-only offline data and online replay utilities.

The action-free boundary is enforced when :class:`ActionFreeTrajectoryData`
is constructed: action, reward, and return arrays are not retained as object
attributes, so downstream algorithms cannot accidentally recover them.
"""

from __future__ import annotations

import bisect
import dataclasses
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np

from utils.datasets import Dataset
from utils.goal_representation import infer_phi_goal_obs_indices


FORBIDDEN_OFFLINE_FIELDS = frozenset(
    {'actions', 'rewards', 'returns', 'return_to_go', 'rtg'}
)
REPLAY_METRIC_KEYS = (
    'replay/her_relabel_fraction',
    'replay/her_success_fraction',
    'replay/commanded_success_fraction',
    'replay/desired_next_valid_fraction',
    'replay/desired_next_l2',
)


@dataclasses.dataclass(frozen=True)
class EpisodeIndex:
    """Compact-trajectory boundary lookup."""

    initial: np.ndarray
    terminal: np.ndarray
    initial_for_state: np.ndarray
    terminal_for_state: np.ndarray
    transition_indices: np.ndarray

    @classmethod
    def from_terminals(cls, terminals: Any) -> 'EpisodeIndex':
        terminals = np.asarray(terminals).reshape(-1)
        terminal = np.flatnonzero(terminals > 0).astype(np.int64)
        if not len(terminal) or int(terminal[-1]) != len(terminals) - 1:
            raise ValueError('The last compact-dataset state must be terminal.')
        initial = np.concatenate(
            [np.asarray([0], dtype=np.int64), terminal[:-1] + 1]
        )
        initial_for_state = np.empty(len(terminals), dtype=np.int64)
        terminal_for_state = np.empty(len(terminals), dtype=np.int64)
        transition_parts = []
        for start, final in zip(initial, terminal):
            initial_for_state[start : final + 1] = start
            terminal_for_state[start : final + 1] = final
            if final > start:
                transition_parts.append(np.arange(start, final, dtype=np.int64))
        if not transition_parts:
            raise ValueError('Action-free data must contain at least one transition.')
        return cls(
            initial=initial,
            terminal=terminal,
            initial_for_state=initial_for_state,
            terminal_for_state=terminal_for_state,
            transition_indices=np.concatenate(transition_parts),
        )


class ActionFreeTrajectoryData:
    """Immutable state trajectory with goal-conditioned batch samplers."""

    def __init__(
        self,
        dataset: Dataset | Mapping[str, Any],
        *,
        env_name: str,
        discount: float = 0.99,
        seed: int = 0,
    ):
        fields = {str(key).lower(): key for key in dataset}
        if 'observations' not in fields or 'terminals' not in fields:
            raise ValueError('Action-free data requires observations and terminals.')
        observations = np.asarray(dataset[fields['observations']], dtype=np.float32)
        terminals = np.asarray(dataset[fields['terminals']], dtype=np.float32)
        if observations.ndim != 2:
            raise ValueError(
                f'Only state vectors are supported, got {observations.shape}.'
            )
        if len(terminals) != len(observations):
            raise ValueError('observations and terminals must have equal lengths.')
        self.observations = observations.view()
        self.observations.setflags(write=False)
        self.terminals = terminals.view()
        self.terminals.setflags(write=False)
        self.episodes = EpisodeIndex.from_terminals(self.terminals)
        self.env_name = str(env_name)
        self.discount = float(discount)
        self.rng = np.random.default_rng(int(seed))
        self.size = len(observations)
        self.phi_indices = np.asarray(
            infer_phi_goal_obs_indices(self.env_name, observations.shape[-1]),
            dtype=np.int64,
        )
        self.offline_fields_seen = ('observations', 'terminals')

    def _sample_transition_indices(self, batch_size: int) -> np.ndarray:
        choices = self.rng.integers(
            0, len(self.episodes.transition_indices), size=int(batch_size)
        )
        return self.episodes.transition_indices[choices]

    def _sample_goal_indices(
        self,
        indices: np.ndarray,
        *,
        future_probability: float,
        current_probability: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if future_probability < 0 or current_probability < 0:
            raise ValueError('Goal probabilities must be non-negative.')
        if future_probability + current_probability > 1.0 + 1e-8:
            raise ValueError('Goal probabilities cannot sum above one.')
        batch_size = len(indices)
        draws = self.rng.random(batch_size)
        goal_indices = self.rng.choice(
            self.episodes.transition_indices, size=batch_size
        ).astype(np.int64)
        components = np.full(batch_size, 2, dtype=np.int8)

        current = draws < current_probability
        goal_indices[current] = indices[current]
        components[current] = 0

        future = (draws >= current_probability) & (
            draws < current_probability + future_probability
        )
        if np.any(future):
            starts = indices[future]
            finals = self.episodes.terminal_for_state[starts]
            max_offsets = finals - starts
            offsets = 1 + np.floor(
                self.rng.random(len(starts)) * max_offsets
            ).astype(np.int64)
            goal_indices[future] = starts + offsets
            components[future] = 1
        return goal_indices, components

    def sample(
        self,
        batch_size: int,
        *,
        future_probability: float = 0.9,
        current_probability: float = 0.0,
        subgoal_steps: int = 10,
    ) -> dict[str, np.ndarray]:
        """Sample supervision under the transition reward ``r(s', g)``.

        Current-state goals are disabled by default because they are not
        terminal anchors under this convention.  Immediate future states
        provide the zero-reward Bellman anchors instead.
        """

        indices = self._sample_transition_indices(batch_size)
        next_indices = indices + 1
        goal_indices, components = self._sample_goal_indices(
            indices,
            future_probability=future_probability,
            current_probability=current_probability,
        )
        same_episode = (
            self.episodes.initial_for_state[indices]
            == self.episodes.initial_for_state[goal_indices]
        )
        ordered = same_episode & (goal_indices >= indices)
        slow_target_indices = np.minimum(
            indices + int(subgoal_steps),
            self.episodes.terminal_for_state[indices],
        )
        slow_target_indices = np.where(
            ordered,
            np.minimum(slow_target_indices, goal_indices),
            slow_target_indices,
        )
        successes = next_indices == goal_indices
        rewards = successes.astype(np.float32) - 1.0
        masks = 1.0 - successes.astype(np.float32)
        return {
            'observations': self.observations[indices],
            'next_observations': self.observations[next_indices],
            'goals': self.observations[goal_indices],
            'rewards': rewards,
            'masks': masks,
            'fast_targets': self.observations[next_indices],
            'slow_targets': self.observations[slow_target_indices],
            'indices': indices,
            'goal_indices': goal_indices,
            'goal_components': components,
        }

    def sample_sequences(
        self,
        batch_size: int,
        *,
        context_length: int = 20,
        future_probability: float = 1.0,
    ) -> dict[str, np.ndarray]:
        """Sample left-padded histories for GC-AFDT."""

        context_length = int(context_length)
        if context_length < 1:
            raise ValueError('context_length must be positive.')
        indices = self._sample_transition_indices(batch_size)
        goal_indices, _ = self._sample_goal_indices(
            indices,
            future_probability=future_probability,
            current_probability=0.0,
        )
        histories = np.empty(
            (len(indices), context_length, self.observations.shape[-1]),
            dtype=np.float32,
        )
        history_masks = np.zeros((len(indices), context_length), dtype=np.float32)
        for row, index in enumerate(indices):
            start = max(
                int(self.episodes.initial_for_state[index]),
                int(index) - context_length + 1,
            )
            sequence = self.observations[start : index + 1]
            histories[row, -len(sequence) :] = sequence
            histories[row, : context_length - len(sequence)] = sequence[0]
            history_masks[row, -len(sequence) :] = 1.0
        remaining = np.maximum(goal_indices - indices, 0).astype(np.float32)
        return {
            'histories': histories,
            'history_masks': history_masks,
            'observations': self.observations[indices],
            'goals': self.observations[goal_indices],
            'remaining': remaining,
            'target_deltas': self.observations[indices + 1] - self.observations[indices],
        }


class OnlineReplayBuffer:
    """Fixed-capacity replay with episode-aware future HER."""

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        *,
        seed: int = 0,
    ):
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError('Replay capacity must be positive.')
        self.rng = np.random.default_rng(int(seed))
        self.observations = np.zeros((self.capacity, *observation_shape), np.float32)
        self.actions = np.zeros((self.capacity, *action_shape), np.float32)
        self.next_observations = np.zeros_like(self.observations)
        self.goals = np.zeros_like(self.observations)
        self.rewards = np.zeros((self.capacity,), np.float32)
        self.masks = np.ones((self.capacity,), np.float32)
        self.desired_next = np.zeros_like(self.observations)
        self.desired_next_valid = np.zeros((self.capacity,), np.bool_)
        self.episode_ids = np.full((self.capacity,), -1, np.int64)
        self.timesteps = np.full((self.capacity,), -1, np.int64)
        self.pointer = 0
        self.size = 0
        self._episode_slots: dict[int, list[tuple[int, int]]] = defaultdict(list)

    def _remove_old_slot(self, slot: int) -> None:
        old_episode = int(self.episode_ids[slot])
        if old_episode < 0:
            return
        entries = self._episode_slots[old_episode]
        old = (int(self.timesteps[slot]), slot)
        position = bisect.bisect_left(entries, old)
        if position < len(entries) and entries[position] == old:
            entries.pop(position)
        if not entries:
            del self._episode_slots[old_episode]

    def add(
        self,
        *,
        observation: Any,
        action: Any,
        next_observation: Any,
        goal: Any,
        reward: float,
        mask: float,
        episode_id: int,
        timestep: int,
        desired_next: Any | None = None,
        desired_next_valid: bool = False,
    ) -> None:
        slot = self.pointer
        self._remove_old_slot(slot)
        self.observations[slot] = observation
        self.actions[slot] = action
        self.next_observations[slot] = next_observation
        self.goals[slot] = goal
        self.rewards[slot] = reward
        self.masks[slot] = mask
        self.desired_next[slot] = (
            next_observation if desired_next is None else desired_next
        )
        self.desired_next_valid[slot] = bool(desired_next_valid)
        self.episode_ids[slot] = int(episode_id)
        self.timesteps[slot] = int(timestep)
        bisect.insort(
            self._episode_slots[int(episode_id)],
            (int(timestep), slot),
        )
        self.pointer = (slot + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        *,
        her_probability: float = 0.0,
        goal_indices: np.ndarray | None = None,
        success_tolerance: float = 1e-5,
    ) -> dict[str, np.ndarray]:
        if self.size < 1:
            raise ValueError('Cannot sample an empty replay buffer.')
        valid_slots = np.flatnonzero(self.episode_ids >= 0)
        indices = self.rng.choice(valid_slots, size=int(batch_size), replace=True)
        goals = self.goals[indices].copy()
        rewards = self.rewards[indices].copy()
        masks = self.masks[indices].copy()
        commanded_success_fraction = np.mean(rewards >= 0.0, dtype=np.float32)
        relabel = self.rng.random(len(indices)) < float(her_probability)
        relabeled = np.zeros(len(indices), dtype=np.bool_)
        her_successes = np.zeros(len(indices), dtype=np.bool_)
        for row in np.flatnonzero(relabel):
            slot = int(indices[row])
            episode = int(self.episode_ids[slot])
            entries = self._episode_slots[episode]
            # Include the current transition.  Its achieved next state is the
            # necessary immediate positive anchor g = s_{t+1}.
            position = bisect.bisect_left(
                entries, (int(self.timesteps[slot]), -1)
            )
            if position >= len(entries):
                continue
            future_timestep, future_slot = entries[
                int(self.rng.integers(position, len(entries)))
            ]
            del future_timestep
            relabeled[row] = True
            goals[row] = self.next_observations[future_slot]
            achieved = self.next_observations[slot]
            if goal_indices is not None:
                achieved = achieved[goal_indices]
                target = goals[row][goal_indices]
            else:
                target = goals[row]
            success = bool(np.linalg.norm(achieved - target) <= success_tolerance)
            her_successes[row] = success
            rewards[row] = 0.0 if success else -1.0
            masks[row] = 0.0 if success else 1.0
        num_relabeled = int(np.sum(relabeled))
        desired_valid = self.desired_next_valid[indices]
        desired_errors = np.linalg.norm(
            self.next_observations[indices] - self.desired_next[indices], axis=-1
        )
        desired_valid_count = int(np.sum(desired_valid))
        return {
            'observations': self.observations[indices],
            'actions': self.actions[indices],
            'next_observations': self.next_observations[indices],
            'goals': goals,
            'rewards': rewards,
            'masks': masks,
            'desired_next': self.desired_next[indices],
            'desired_next_valid': self.desired_next_valid[indices].astype(np.float32),
            'replay/her_relabel_fraction': np.asarray(
                np.mean(relabeled), dtype=np.float32
            ),
            'replay/her_success_fraction': np.asarray(
                np.sum(her_successes) / max(num_relabeled, 1), dtype=np.float32
            ),
            'replay/commanded_success_fraction': np.asarray(
                commanded_success_fraction, dtype=np.float32
            ),
            'replay/desired_next_valid_fraction': np.asarray(
                np.mean(desired_valid), dtype=np.float32
            ),
            'replay/desired_next_l2': np.asarray(
                np.sum(desired_errors * desired_valid)
                / max(desired_valid_count, 1),
                dtype=np.float32,
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            'capacity': self.capacity,
            'pointer': self.pointer,
            'size': self.size,
            'observations': self.observations,
            'actions': self.actions,
            'next_observations': self.next_observations,
            'goals': self.goals,
            'rewards': self.rewards,
            'masks': self.masks,
            'desired_next': self.desired_next,
            'desired_next_valid': self.desired_next_valid,
            'episode_ids': self.episode_ids,
            'timesteps': self.timesteps,
            'rng_state': self.rng.bit_generator.state,
        }


__all__ = [
    'ActionFreeTrajectoryData',
    'EpisodeIndex',
    'FORBIDDEN_OFFLINE_FIELDS',
    'OnlineReplayBuffer',
    'REPLAY_METRIC_KEYS',
]
