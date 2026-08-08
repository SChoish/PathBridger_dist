"""Strict action-free pixel trajectories and compact online pixel replay."""

from __future__ import annotations

import bisect
import dataclasses
from collections import defaultdict
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


def repeat_pixel_frame(frame: np.ndarray, count: int) -> np.ndarray:
    return np.concatenate([frame] * int(count), axis=-1)


def stack_pixel_history(frames: list[np.ndarray], count: int) -> np.ndarray:
    if not frames:
        raise ValueError('Pixel frame history cannot be empty.')
    selected = list(frames[-int(count) :])
    selected = [selected[0]] * (int(count) - len(selected)) + selected
    return np.concatenate(selected, axis=-1)


class ActionFreePixelTrajectoryData:
    """Offline visual trajectories exposing only pixels and terminals.

    The constructor copies references to exactly two whitelisted fields.  It
    never retains actions, rewards, simulator state, or privileged metadata.
    """

    def __init__(
        self,
        dataset: Mapping[str, Any],
        *,
        seed: int = 0,
        frame_stack: int = 1,
    ):
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
        self.frame_stack = int(frame_stack)
        if self.frame_stack < 1:
            raise ValueError('frame_stack must be positive.')
        self.offline_fields_seen = ('observations', 'terminals')

    @property
    def image_shape(self) -> tuple[int, int, int]:
        height, width, channels = self.observations.shape[1:]
        return int(height), int(width), int(channels) * self.frame_stack

    def stack_indices(self, indices: Any) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        initials = self.episodes.initial_for_state[indices]
        offsets = np.arange(self.frame_stack - 1, -1, -1, dtype=np.int64)
        history = np.maximum(indices[:, None] - offsets[None, :], initials[:, None])
        frames = self.observations[history]
        return np.concatenate(
            [frames[:, position] for position in range(self.frame_stack)],
            axis=-1,
        )

    @property
    def example_images(self) -> np.ndarray:
        indices = self.episodes.transition_indices[:2]
        if len(indices) < 2:
            indices = np.repeat(indices, 2)
        return self.stack_indices(indices)

    def _sample_future_offsets(
        self,
        indices: np.ndarray,
        *,
        discount: float,
        geom_sample: bool,
    ) -> np.ndarray:
        """Sample positive within-episode offsets for goals / value targets."""

        finals = self.episodes.terminal_for_state[indices]
        remaining = np.maximum(finals - indices, 1)
        if geom_sample:
            # TRL-style geometric horizon with success at the terminal capped.
            probs = 1.0 - float(discount)
            probs = min(max(probs, 1e-6), 1.0 - 1e-6)
            offsets = self.rng.geometric(p=probs, size=len(indices))
            offsets = np.minimum(offsets, remaining).astype(np.int64)
            offsets = np.maximum(offsets, 1)
        else:
            offsets = 1 + np.floor(
                self.rng.random(len(indices)) * remaining
            ).astype(np.int64)
        return offsets

    def sample(
        self,
        batch_size: int,
        *,
        path_horizon: int = 5,
        endpoint_horizon: int | None = None,
        discount: float = 0.99,
        value_geom_sample: bool = True,
        value_p_curgoal: float = 0.0,
        value_p_trajgoal: float = 1.0,
        value_p_randomgoal: float = 0.0,
        pbf_indices_only: bool = False,
    ) -> dict[str, np.ndarray]:
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError('batch_size must be positive.')
        path_horizon = int(path_horizon)
        if path_horizon < 1:
            raise ValueError('path_horizon must be positive.')

        transition_indices = self.episodes.transition_indices
        if endpoint_horizon is None:
            valid_starts = transition_indices
        else:
            endpoint_horizon = int(endpoint_horizon)
            if endpoint_horizon < path_horizon:
                raise ValueError(
                    'endpoint_horizon must be at least path_horizon, got '
                    f'{endpoint_horizon} < {path_horizon}.'
                )
            valid_starts = transition_indices[
                transition_indices + endpoint_horizon
                <= self.episodes.terminal_for_state[transition_indices]
            ]
            if not len(valid_starts):
                raise ValueError(
                    'No pixel episode contains a full '
                    f'horizon-{endpoint_horizon} PathBridger window.'
                )
        choices = self.rng.integers(
            0, len(valid_starts), size=batch_size
        )
        indices = valid_starts[choices]
        finals = self.episodes.terminal_for_state[indices]

        # PathBridger's endpoint conditioning goal is an ordinary future state.
        endpoint_goal_offsets = self._sample_future_offsets(
            indices, discount=discount, geom_sample=False
        )
        endpoint_goal_indices = indices + endpoint_goal_offsets
        if endpoint_horizon is None:
            endpoint_target_indices = endpoint_goal_indices
            fractions = np.arange(1, path_horizon + 1, dtype=np.float64)
            fractions /= float(path_horizon)
            bridge_indices = indices[:, None] + np.ceil(
                endpoint_goal_offsets[:, None] * fractions[None, :]
            ).astype(np.int64)
        else:
            endpoint_target_indices = np.minimum(
                indices + endpoint_horizon, endpoint_goal_indices
            )
            bridge_indices = indices[:, None] + np.arange(
                1, path_horizon + 1, dtype=np.int64
            )[None, :]
            bridge_indices = np.minimum(
                bridge_indices, endpoint_target_indices[:, None]
            )
        mix = np.asarray(
            [value_p_curgoal, value_p_trajgoal, value_p_randomgoal],
            dtype=np.float64,
        )
        if mix.sum() <= 0:
            raise ValueError('Value goal mixture probabilities must sum to > 0.')
        mix = mix / mix.sum()
        components = self.rng.choice(3, size=len(indices), p=mix)
        value_offsets = np.zeros(len(indices), dtype=np.int64)
        value_goal_indices = indices.copy()
        traj_mask = components == 1
        if np.any(traj_mask):
            traj_offsets = self._sample_future_offsets(
                indices[traj_mask],
                discount=discount,
                geom_sample=bool(value_geom_sample),
            )
            value_offsets[traj_mask] = traj_offsets
            value_goal_indices[traj_mask] = indices[traj_mask] + traj_offsets
        rand_mask = components == 2
        if np.any(rand_mask):
            rand_choices = self.rng.integers(
                0, len(self.episodes.transition_indices), size=int(np.sum(rand_mask))
            )
            value_goal_indices[rand_mask] = self.episodes.transition_indices[
                rand_choices
            ]
            value_offsets[rand_mask] = np.maximum(
                value_goal_indices[rand_mask] - indices[rand_mask], 0
            )
        # TRL/PBF short anchors use a uniformly sampled offset in [1, 5].
        remaining = finals - indices
        base_max_offsets = np.minimum(5, remaining)
        base_offsets = 1 + np.floor(
            self.rng.random(len(indices)) * base_max_offsets
        ).astype(np.int64)
        base_indices = indices + base_offsets

        # Transitive splits are strictly internal and only active for pairs
        # longer than the fixed five-step base horizon.
        transitive_valids = value_offsets > 5
        transitive_offsets = np.zeros(len(indices), dtype=np.int64)
        if np.any(transitive_valids):
            max_offsets = value_offsets[transitive_valids] - 1
            transitive_offsets[transitive_valids] = 1 + np.floor(
                self.rng.random(np.sum(transitive_valids)) * max_offsets
            ).astype(np.int64)
        transitive_indices = indices + transitive_offsets

        if pbf_indices_only:
            if endpoint_horizon is None:
                raise ValueError(
                    'Indexed PBF batches require an explicit endpoint_horizon.'
                )
            return {
                'observation_indices': indices,
                'next_observation_indices': indices + 1,
                'endpoint_goal_indices': endpoint_goal_indices,
                'endpoint_target_indices': endpoint_target_indices,
                'bridge_indices': bridge_indices,
                'value_goal_indices': value_goal_indices,
                'base_indices': base_indices,
                'transitive_indices': transitive_indices,
                'value_offsets': value_offsets.astype(np.float32),
                'base_offsets': base_offsets.astype(np.float32),
                'transitive_offsets': transitive_offsets.astype(np.float32),
                'transitive_valids': transitive_valids.astype(np.float32),
            }

        bridge_targets = self.stack_indices(bridge_indices.reshape(-1)).reshape(
            len(indices), path_horizon, *self.image_shape
        )
        successes = endpoint_goal_indices == indices + 1

        result = {
            'observations': self.stack_indices(indices),
            'next_observations': self.stack_indices(indices + 1),
            'goals': self.stack_indices(endpoint_goal_indices),
            'rewards': successes.astype(np.float32) - 1.0,
            'masks': 1.0 - successes.astype(np.float32),
            'indices': indices,
            'goal_indices': endpoint_goal_indices,
            'path_indices': bridge_indices,
            'path_observations': bridge_targets,
            'value_goals': self.stack_indices(value_goal_indices),
            'value_offsets': value_offsets.astype(np.float32),
            'base_goals': self.stack_indices(base_indices),
            'base_offsets': base_offsets.astype(np.float32),
            'transitive_subgoals': self.stack_indices(transitive_indices),
            'transitive_offsets': transitive_offsets.astype(np.float32),
            'transitive_valids': transitive_valids.astype(np.float32),
        }
        if endpoint_horizon is not None:
            result.update(
                endpoint_goals=self.stack_indices(endpoint_goal_indices),
                endpoint_targets=self.stack_indices(endpoint_target_indices),
                bridge_targets=bridge_targets,
                endpoint_goal_indices=endpoint_goal_indices,
                endpoint_target_indices=endpoint_target_indices,
            )
        return result


class PixelTrajectoryData(ActionFreePixelTrajectoryData):
    """Full offline pixel trajectories with actions but no privileged state."""

    def __init__(
        self,
        dataset: Mapping[str, Any],
        *,
        seed: int = 0,
        frame_stack: int = 1,
    ):
        if 'actions' not in dataset:
            raise ValueError('Full offline pixel trajectories require actions.')
        actions = np.asarray(dataset['actions'], dtype=np.float32)
        if actions.ndim != 2 or len(actions) != len(dataset['observations']):
            raise ValueError(
                'Full offline pixel actions must have shape [N, A], got '
                f'{actions.shape}.'
            )
        super().__init__(dataset, seed=seed, frame_stack=frame_stack)
        self.actions = actions.view()
        self.actions.setflags(write=False)
        self.offline_fields_seen = ('observations', 'terminals', 'actions')

    def sample(self, batch_size: int, **kwargs) -> dict[str, np.ndarray]:
        batch = super().sample(batch_size, **kwargs)
        index_key = (
            'observation_indices'
            if bool(kwargs.get('pbf_indices_only', False))
            else 'indices'
        )
        batch['actions'] = self.actions[batch[index_key]]
        return batch

    def _hierarchical_goal_indices(
        self, indices, *, discount, p_current, p_trajectory, p_random, geometric
    ):
        probabilities = np.asarray(
            [p_current, p_trajectory, p_random], dtype=np.float64
        )
        if not np.isclose(probabilities.sum(), 1.0):
            raise ValueError('Hierarchical goal probabilities must sum to one.')
        finals = self.episodes.terminal_for_state[indices]
        random_indices = self.episodes.transition_indices[
            self.rng.integers(
                0, len(self.episodes.transition_indices), size=len(indices)
            )
        ]
        if geometric:
            probability = np.clip(1.0 - float(discount), 1e-6, 1.0 - 1e-6)
            offsets = self.rng.geometric(probability, size=len(indices))
            trajectory_indices = np.minimum(indices + offsets, finals)
        else:
            distances = self.rng.random(len(indices))
            trajectory_indices = np.round(
                np.minimum(indices + 1, finals) * distances
                + finals * (1.0 - distances)
            ).astype(np.int64)
        components = self.rng.choice(3, size=len(indices), p=probabilities)
        return np.where(
            components == 0,
            indices,
            np.where(components == 1, trajectory_indices, random_indices),
        )

    def _augment_hierarchical(self, batch, keys):
        padding = 3
        offsets = self.rng.integers(
            0, 2 * padding + 1, size=(len(batch[keys[0]]), 2)
        )
        for key in keys:
            images = np.asarray(batch[key])
            padded = np.pad(
                images,
                ((0, 0), (padding, padding), (padding, padding), (0, 0)),
                mode='edge',
            )
            shifted = np.empty_like(images)
            height, width = images.shape[1:3]
            for row, (top, left) in enumerate(offsets):
                shifted[row] = padded[
                    row, top : top + height, left : left + width
                ]
            batch[key] = shifted

    def sample_hierarchical(self, batch_size: int, **config):
        """Sample official HGCDataset fields required by HIQL or OTA."""

        agent_name = str(config['agent_name'])
        if agent_name not in ('hiql', 'ota'):
            raise ValueError("agent_name must be 'hiql' or 'ota'.")
        transitions = self.episodes.transition_indices
        indices = transitions[
            self.rng.integers(0, len(transitions), size=int(batch_size))
        ]
        finals = self.episodes.terminal_for_state[indices]
        discount = float(config.get('discount', 0.99))
        subgoal_steps = int(config.get('subgoal_steps', 25))
        value_goals = self._hierarchical_goal_indices(
            indices,
            discount=discount,
            p_current=float(config.get('value_p_curgoal', 0.2)),
            p_trajectory=float(config.get('value_p_trajgoal', 0.5)),
            p_random=float(config.get('value_p_randomgoal', 0.3)),
            geometric=bool(config.get('value_geom_sample', True)),
        )
        successes = indices == value_goals
        low_goals = np.minimum(indices + subgoal_steps, finals)
        if bool(config.get('actor_geom_sample', False)):
            probability = np.clip(1.0 - discount, 1e-6, 1.0 - 1e-6)
            actor_goals = np.minimum(
                indices + self.rng.geometric(probability, size=len(indices)),
                finals,
            )
        else:
            distances = self.rng.random(len(indices))
            actor_goals = np.round(
                np.minimum(indices + 1, finals) * distances
                + finals * (1.0 - distances)
            ).astype(np.int64)
        actor_targets = np.minimum(indices + subgoal_steps, actor_goals)
        random_actor_goals = transitions[
            self.rng.integers(0, len(transitions), size=len(indices))
        ]
        pick_random = self.rng.random(len(indices)) < float(
            config.get('actor_p_randomgoal', 0.0)
        )
        high_actor_goals = np.where(
            pick_random, random_actor_goals, actor_goals
        )
        high_actor_targets = np.where(
            pick_random, np.minimum(indices + subgoal_steps, finals), actor_targets
        )
        batch = {
            'observations': self.stack_indices(indices),
            'next_observations': self.stack_indices(indices + 1),
            'actions': self.actions[indices],
            'value_goals': self.stack_indices(value_goals),
            'rewards': successes.astype(np.float32) - 1.0,
            'masks': 1.0 - successes.astype(np.float32),
            'low_actor_goals': self.stack_indices(low_goals),
            'high_actor_goals': self.stack_indices(high_actor_goals),
            'high_actor_targets': self.stack_indices(high_actor_targets),
            'indices': indices,
            'value_goal_indices': value_goals,
            'low_actor_goal_indices': low_goals,
            'high_actor_goal_indices': high_actor_goals,
            'high_actor_target_indices': high_actor_targets,
        }
        if agent_name == 'ota':
            random_goals = transitions[
                self.rng.integers(0, len(transitions), size=len(indices))
            ]
            if bool(config.get('value_geom_sample', True)):
                probability = np.clip(1.0 - discount, 1e-6, 1.0 - 1e-6)
                trajectory_goals = np.minimum(
                    indices + self.rng.geometric(probability, size=len(indices)),
                    finals,
                )
            else:
                distances = self.rng.random(len(indices))
                trajectory_goals = np.round(
                    np.minimum(indices + 1, finals) * distances
                    + finals * (1.0 - distances)
                ).astype(np.int64)
            p_current = float(config.get('value_p_curgoal', 0.2))
            p_trajectory = float(config.get('value_p_trajgoal', 0.5))
            trajectory_mask = self.rng.random(len(indices)) < (
                p_trajectory / (1.0 - p_current + 1e-6)
            )
            factor = int(config.get('abstraction_factor', 5))
            option_indices = np.where(
                trajectory_mask,
                np.minimum(indices + factor, trajectory_goals),
                np.minimum(indices + factor, finals),
            )
            high_value_goals = np.where(
                trajectory_mask, trajectory_goals, random_goals
            )
            high_value_goals = np.where(
                self.rng.random(len(indices)) < p_current,
                option_indices,
                high_value_goals,
            )
            high_successes = option_indices == high_value_goals
            batch.update(
                high_value_goals=self.stack_indices(high_value_goals),
                high_value_option_observations=self.stack_indices(option_indices),
                high_value_rewards=high_successes.astype(np.float32) - 1.0,
                high_value_masks=1.0 - high_successes.astype(np.float32),
                high_value_goal_indices=high_value_goals,
                high_value_option_indices=option_indices,
            )
        if self.rng.random() < float(config.get('p_aug', 0.0)):
            self._augment_hierarchical(
                batch,
                (
                    'observations', 'next_observations', 'value_goals',
                    'low_actor_goals', 'high_actor_goals', 'high_actor_targets',
                ),
            )
        return batch


class PixelReplayBuffer:
    """Episode-aware indexed raw-frame replay with future-image HER."""

    def __init__(
        self,
        capacity: int,
        image_shape: tuple[int, int, int],
        action_shape: tuple[int, ...],
        *,
        seed: int = 0,
        frame_stack: int = 1,
    ):
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError('Replay capacity must be positive.')
        self.raw_image_shape = tuple(int(value) for value in image_shape)
        _validate_pixels(
            np.empty((1, *self.raw_image_shape), dtype=np.uint8),
            where='image_shape',
        )
        self.frame_stack = int(frame_stack)
        if self.frame_stack < 1:
            raise ValueError('frame_stack must be positive.')
        self.observation_frame_ids = np.full(self.capacity, -1, np.int64)
        self.next_frame_ids = np.full(self.capacity, -1, np.int64)
        self.behavior_goal_frame_ids = np.full(self.capacity, -1, np.int64)
        self.actions = np.empty((self.capacity, *action_shape), np.float32)
        self.rewards = np.empty((self.capacity,), np.float32)
        self.masks = np.empty((self.capacity,), np.float32)
        self.episode_ids = np.full(self.capacity, -1, np.int64)
        self.timesteps = np.full(self.capacity, -1, np.int64)
        self.pointer = 0
        self.size = 0
        self.rng = np.random.default_rng(int(seed))
        self._frames: dict[int, np.ndarray] = {}
        self._frame_references: dict[int, int] = {}
        self._next_frame_id = 0
        self._episode_slots: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self._episode_goal_ids: dict[int, int] = {}

    @property
    def image_shape(self) -> tuple[int, int, int]:
        height, width, channels = self.raw_image_shape
        return height, width, channels * self.frame_stack

    def _new_frame(self, value: Any, *, where: str) -> int:
        frame = np.asarray(value)
        if frame.shape != self.raw_image_shape or frame.dtype != np.uint8:
            raise ValueError(
                f'{where} must be {self.raw_image_shape} uint8, '
                f'got {frame.shape}/{frame.dtype}.'
            )
        frame_id = self._next_frame_id
        self._next_frame_id += 1
        self._frames[frame_id] = frame.copy()
        self._frame_references[frame_id] = 0
        return frame_id

    def _retain(self, frame_id: int) -> None:
        self._frame_references[frame_id] += 1

    def _release(self, frame_id: int) -> None:
        if frame_id < 0:
            return
        references = self._frame_references[frame_id] - 1
        if references <= 0:
            del self._frame_references[frame_id]
            del self._frames[frame_id]
        else:
            self._frame_references[frame_id] = references

    def _remove_old_slot(self, slot: int) -> None:
        episode = int(self.episode_ids[slot])
        if episode < 0:
            return
        entries = self._episode_slots[episode]
        old = (int(self.timesteps[slot]), slot)
        position = bisect.bisect_left(entries, old)
        if position < len(entries) and entries[position] == old:
            entries.pop(position)
        for frame_id in (
            int(self.observation_frame_ids[slot]),
            int(self.next_frame_ids[slot]),
            int(self.behavior_goal_frame_ids[slot]),
        ):
            self._release(frame_id)
        if not entries:
            del self._episode_slots[episode]
            self._episode_goal_ids.pop(episode, None)

    def _history_frame_ids(self, slot: int, *, next_state: bool) -> list[int]:
        episode = int(self.episode_ids[slot])
        timestep = int(self.timesteps[slot])
        entries = self._episode_slots[episode]
        earliest_slot = entries[0][1]
        earliest_id = int(self.observation_frame_ids[earliest_slot])
        end = timestep + int(next_state)
        result = []
        for state_time in range(end - self.frame_stack + 1, end + 1):
            if next_state and state_time == timestep + 1:
                result.append(int(self.next_frame_ids[slot]))
                continue
            position = bisect.bisect_right(entries, (state_time, self.capacity)) - 1
            if position < 0:
                result.append(earliest_id)
            else:
                result.append(int(self.observation_frame_ids[entries[position][1]]))
        return result

    def _stack_slots(self, slots: np.ndarray, *, next_state: bool) -> np.ndarray:
        return np.stack(
            [
                np.concatenate(
                    [self._frames[index] for index in self._history_frame_ids(int(slot), next_state=next_state)],
                    axis=-1,
                )
                for slot in slots
            ]
        )

    def _stack_goal_ids(self, frame_ids: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                repeat_pixel_frame(self._frames[int(frame_id)], self.frame_stack)
                for frame_id in frame_ids
            ]
        )

    @property
    def allocated_bytes(self) -> int:
        return int(
            sum(frame.nbytes for frame in self._frames.values())
            + self.observation_frame_ids.nbytes
            + self.next_frame_ids.nbytes
            + self.behavior_goal_frame_ids.nbytes
            + self.actions.nbytes
            + self.rewards.nbytes
            + self.masks.nbytes
            + self.episode_ids.nbytes
            + self.timesteps.nbytes
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
        episode_id: int,
        timestep: int,
    ) -> None:
        slot = self.pointer
        self._remove_old_slot(slot)
        episode_id = int(episode_id)
        timestep = int(timestep)
        entries = self._episode_slots[episode_id]
        if any(existing_timestep == timestep for existing_timestep, _ in entries):
            raise ValueError(
                f'Duplicate timestep {timestep} in pixel episode {episode_id}.'
            )
        observation_id = None
        if entries and entries[-1][0] == timestep - 1:
            previous_id = int(self.next_frame_ids[entries[-1][1]])
            if np.array_equal(self._frames[previous_id], np.asarray(observation)):
                observation_id = previous_id
        if observation_id is None:
            observation_id = self._new_frame(observation, where='observation')
        next_id = self._new_frame(next_observation, where='next_observation')
        goal_id = self._episode_goal_ids.get(episode_id)
        if goal_id is None:
            goal_id = self._new_frame(goal, where='goal')
            self._episode_goal_ids[episode_id] = goal_id
        elif not np.array_equal(self._frames[goal_id], np.asarray(goal)):
            raise ValueError('The behavior goal changed within one pixel episode.')
        for frame_id in (observation_id, next_id, goal_id):
            self._retain(frame_id)
        self.observation_frame_ids[slot] = observation_id
        self.next_frame_ids[slot] = next_id
        self.behavior_goal_frame_ids[slot] = goal_id
        self.actions[slot] = action
        self.rewards[slot] = reward
        self.masks[slot] = mask
        self.episode_ids[slot] = episode_id
        self.timesteps[slot] = timestep
        bisect.insort(entries, (timestep, slot))
        self.pointer = (slot + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        *,
        her_probability: float = 0.8,
    ) -> dict[str, np.ndarray]:
        if self.size < 1:
            raise ValueError('Cannot sample an empty pixel replay.')
        valid_slots = np.flatnonzero(self.episode_ids >= 0)
        indices = self.rng.choice(valid_slots, size=int(batch_size), replace=True)
        behavior_goal_ids = self.behavior_goal_frame_ids[indices].copy()
        goal_ids = behavior_goal_ids.copy()
        rewards = self.rewards[indices].copy()
        masks = self.masks[indices].copy()
        relabeled = np.zeros(len(indices), dtype=np.bool_)
        successes = np.zeros(len(indices), dtype=np.bool_)
        requested = self.rng.random(len(indices)) < float(her_probability)
        for row in np.flatnonzero(requested):
            slot = int(indices[row])
            entries = self._episode_slots[int(self.episode_ids[slot])]
            position = bisect.bisect_left(
                entries, (int(self.timesteps[slot]), -1)
            )
            if position >= len(entries):
                continue
            _, future_slot = entries[int(self.rng.integers(position, len(entries)))]
            relabeled[row] = True
            goal_ids[row] = self.next_frame_ids[future_slot]
            success = future_slot == slot
            successes[row] = success
            rewards[row] = 0.0 if success else -1.0
            masks[row] = 0.0 if success else 1.0
        relabeled_count = max(int(np.sum(relabeled)), 1)
        return {
            'observations': self._stack_slots(indices, next_state=False),
            'actions': self.actions[indices],
            'next_observations': self._stack_slots(indices, next_state=True),
            'goals': self._stack_goal_ids(goal_ids),
            'behavior_goals': self._stack_goal_ids(behavior_goal_ids),
            'rewards': rewards,
            'masks': masks,
            'behavior_rewards': self.rewards[indices].copy(),
            'behavior_masks': self.masks[indices].copy(),
            'indices': indices,
            'replay/her_relabel_fraction': np.asarray(
                np.mean(relabeled), dtype=np.float32
            ),
            'replay/her_success_fraction': np.asarray(
                np.sum(successes) / relabeled_count, dtype=np.float32
            ),
            'replay/commanded_success_fraction': np.asarray(
                np.mean(self.rewards[indices] >= 0.0), dtype=np.float32
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        slots = np.flatnonzero(self.episode_ids >= 0).astype(np.int64)
        live_frame_ids = sorted(self._frames)
        return {
            'format_version': 1,
            'capacity': self.capacity,
            'raw_image_shape': self.raw_image_shape,
            'action_shape': tuple(int(value) for value in self.actions.shape[1:]),
            'frame_stack': self.frame_stack,
            'pointer': self.pointer,
            'size': self.size,
            'next_frame_id': self._next_frame_id,
            'slots': slots,
            'observation_frame_ids': self.observation_frame_ids[slots].copy(),
            'next_frame_ids': self.next_frame_ids[slots].copy(),
            'behavior_goal_frame_ids': self.behavior_goal_frame_ids[slots].copy(),
            'actions': self.actions[slots].copy(),
            'rewards': self.rewards[slots].copy(),
            'masks': self.masks[slots].copy(),
            'episode_ids': self.episode_ids[slots].copy(),
            'timesteps': self.timesteps[slots].copy(),
            'frames': {int(fid): self._frames[fid].copy() for fid in live_frame_ids},
            'frame_references': {
                int(fid): int(self._frame_references[fid]) for fid in live_frame_ids
            },
            'episode_slots': {
                int(episode): list(entries)
                for episode, entries in self._episode_slots.items()
            },
            'episode_goal_ids': {
                int(episode): int(frame_id)
                for episode, frame_id in self._episode_goal_ids.items()
            },
            'rng_state': self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state['capacity']) != self.capacity:
            raise ValueError(
                f'Replay capacity mismatch: ckpt={state["capacity"]} vs '
                f'buffer={self.capacity}.'
            )
        if tuple(state['raw_image_shape']) != self.raw_image_shape:
            raise ValueError('Replay image_shape mismatch.')
        if int(state['frame_stack']) != self.frame_stack:
            raise ValueError('Replay frame_stack mismatch.')
        slots = np.asarray(state['slots'], dtype=np.int64)
        if len(slots) != int(state['size']):
            raise ValueError(
                f'Compact replay slot count {len(slots)} does not match '
                f'size {state["size"]}.'
            )
        self.observation_frame_ids.fill(-1)
        self.next_frame_ids.fill(-1)
        self.behavior_goal_frame_ids.fill(-1)
        self.actions.fill(0.0)
        self.rewards.fill(0.0)
        self.masks.fill(1.0)
        self.episode_ids.fill(-1)
        self.timesteps.fill(-1)
        self.observation_frame_ids[slots] = np.asarray(
            state['observation_frame_ids'], dtype=np.int64
        )
        self.next_frame_ids[slots] = np.asarray(state['next_frame_ids'], dtype=np.int64)
        self.behavior_goal_frame_ids[slots] = np.asarray(
            state['behavior_goal_frame_ids'], dtype=np.int64
        )
        self.actions[slots] = np.asarray(state['actions'], dtype=np.float32)
        self.rewards[slots] = np.asarray(state['rewards'], dtype=np.float32)
        self.masks[slots] = np.asarray(state['masks'], dtype=np.float32)
        self.episode_ids[slots] = np.asarray(state['episode_ids'], dtype=np.int64)
        self.timesteps[slots] = np.asarray(state['timesteps'], dtype=np.int64)
        self.pointer = int(state['pointer'])
        self.size = int(state['size'])
        self._next_frame_id = int(state['next_frame_id'])
        self._frames = {
            int(fid): np.asarray(frame, dtype=np.uint8).copy()
            for fid, frame in dict(state['frames']).items()
        }
        self._frame_references = {
            int(fid): int(count)
            for fid, count in dict(state['frame_references']).items()
        }
        self._episode_slots = defaultdict(list)
        for episode, entries in dict(state['episode_slots']).items():
            self._episode_slots[int(episode)] = [
                (int(timestep), int(slot)) for timestep, slot in entries
            ]
        self._episode_goal_ids = {
            int(episode): int(frame_id)
            for episode, frame_id in dict(state['episode_goal_ids']).items()
        }
        self.rng.bit_generator.state = state['rng_state']


__all__ = [
    'ActionFreePixelTrajectoryData',
    'PixelTrajectoryData',
    'PixelEpisodeIndex',
    'PixelReplayBuffer',
    'repeat_pixel_frame',
    'stack_pixel_history',
]
