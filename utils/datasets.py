"""State-only offline dataset utilities for PathBridger.

The sampler intentionally exposes only the supervision used by the final
PathBridger objective.  It assumes OGBench's compact layout: observations are a
single state sequence and ``terminals`` marks the final state of every episode.
No image augmentation, frame stacking, replay storage, actor target, or action
chunk target is implemented here.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

_ACTION_HORIZON = 5
_BRIDGE_OFFSETS = np.arange(1, _ACTION_HORIZON + 1, dtype=np.int64)
_CURRENT_GOAL = 0
_FUTURE_GOAL = 1
_RANDOM_GOAL = 2


class Dataset(Mapping[str, np.ndarray]):
    """An immutable mapping of equally sized NumPy arrays.

    The mapping cannot be changed after construction and each stored array is
    exposed through a read-only view.  Indexing a dataset returns an ordinary
    mutable batch dictionary, which is convenient for JAX training code.
    """

    @classmethod
    def create(cls, freeze: bool = True, **fields: Any) -> "Dataset":
        """Create a dataset from named array fields.

        ``freeze=False`` is accepted for API compatibility, but the mapping
        itself remains immutable.  PathBridger's offline loaders always use the
        default read-only arrays.
        """

        return cls(fields, freeze=freeze)

    def __init__(self, fields: Mapping[str, Any], *, freeze: bool = True):
        if "observations" not in fields:
            raise ValueError("Dataset requires an 'observations' field.")
        if not fields:
            raise ValueError("Dataset cannot be empty.")

        arrays: dict[str, np.ndarray] = {}
        size: int | None = None
        for key, value in fields.items():
            array = np.asarray(value)
            if array.ndim == 0:
                raise ValueError(f"Dataset field {key!r} must have a leading batch dimension.")
            if size is None:
                size = len(array)
            elif len(array) != size:
                raise ValueError(
                    f"Dataset fields must have equal lengths; {key!r} has {len(array)}, expected {size}."
                )
            if freeze:
                array = array.view()
                array.setflags(write=False)
            arrays[str(key)] = array

        if size is None or size == 0:
            raise ValueError("Dataset must contain at least one transition.")
        self._data = MappingProxyType(arrays)
        self.size = size

        if "valids" in arrays:
            valids = np.asarray(arrays["valids"]).reshape(-1)
            self._valid_idxs = np.flatnonzero(valids > 0)
            if len(self._valid_idxs) == 0:
                raise ValueError("Dataset contains no valid transitions.")
        else:
            self._valid_idxs = None

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get_random_idxs(self, num_idxs: int) -> np.ndarray:
        """Sample transition indices, respecting compact-dataset ``valids``."""

        num_idxs = int(num_idxs)
        if num_idxs < 1:
            raise ValueError(f"num_idxs must be positive, got {num_idxs}.")
        if self._valid_idxs is None:
            return np.random.randint(0, self.size, size=num_idxs, dtype=np.int64)
        choices = np.random.randint(0, len(self._valid_idxs), size=num_idxs)
        return self._valid_idxs[choices]

    def get_subset(self, idxs: Any) -> dict[str, np.ndarray]:
        """Return fields at ``idxs`` and infer compact next observations."""

        idxs = np.asarray(idxs, dtype=np.int64)
        if np.any(idxs < 0) or np.any(idxs >= self.size):
            raise IndexError(f"Dataset indices must lie in [0, {self.size}); got {idxs}.")
        result = {key: value[idxs] for key, value in self._data.items()}
        if "next_observations" not in result:
            next_idxs = np.minimum(idxs + 1, self.size - 1)
            result["next_observations"] = self._data["observations"][next_idxs]
        return result

    def sample(self, batch_size: int, idxs: Any | None = None) -> dict[str, np.ndarray]:
        """Sample a transition batch."""

        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        return self.get_subset(idxs)


@dataclasses.dataclass(frozen=True)
class PathBridgerDatasetConfig:
    """The complete sampler configuration."""

    horizon: int
    discount: float
    actor_p: tuple[float, float, float, float]
    critic_p: tuple[float, float, float, float]


def _config_get(config: Any, key: str) -> Any:
    """Read one setting from a mapping, ConfigDict, or dataclass-like object."""

    if isinstance(config, Mapping):
        try:
            return config[key]
        except KeyError as exc:
            raise ValueError(f"PathBridgerDataset config is missing {key!r}.") from exc
    try:
        return getattr(config, key)
    except AttributeError as exc:
        raise ValueError(f"PathBridgerDataset config is missing {key!r}.") from exc


def observation_state_scale(dataset: Dataset, floor: float = 1e-3) -> np.ndarray:
    """Return per-state-dimension training standard deviations with a floor."""

    floor = float(floor)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError(f"state scale floor must be positive and finite, got {floor}.")
    observations = np.asarray(dataset["observations"], dtype=np.float32)
    if observations.ndim != 2:
        raise ValueError(
            "state scale requires observations with shape [N, D], got "
            f"{observations.shape}."
        )
    scale = np.std(observations.astype(np.float64), axis=0)
    return np.maximum(scale, floor).astype(np.float32)


def _validate_goal_mix(value: Any, *, name: str) -> tuple[float, float, float, float]:
    """Validate a ``(p_cur, p_geom, p_traj, p_rand)`` goal-sampling mix."""

    try:
        probabilities = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a numeric 4-tuple ordered as "
            "(p_cur, p_geom, p_traj, p_rand)."
        ) from exc
    if probabilities.shape != (4,):
        raise ValueError(
            f"{name} must be a 4-tuple ordered as "
            f"(p_cur, p_geom, p_traj, p_rand), got shape {probabilities.shape}."
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError(f"{name} probabilities must be finite, got {tuple(probabilities)}.")
    if np.any(probabilities < 0.0):
        raise ValueError(f"{name} probabilities must be non-negative, got {tuple(probabilities)}.")
    total = float(np.sum(probabilities))
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-6):
        raise ValueError(f"{name} probabilities must sum to 1, got {total}.")
    if probabilities[1] > 0.0 and probabilities[2] > 0.0:
        raise ValueError(
            f"{name} cannot enable geometric and ordinary trajectory-future "
            "sampling at the same time."
        )
    probabilities = probabilities / total
    return tuple(float(probability) for probability in probabilities)


@dataclasses.dataclass
class PathBridgerDataset:
    """Add PathBridger hindsight supervision to a compact offline dataset.

    ``actor_p`` controls endpoint/actor goals and ``critic_p`` controls scalar
    value/critic goals.  Both use the paper's four-component order
    ``(p_cur, p_geom, p_traj, p_rand)``; there is no separate geometric-sampling
    boolean.
    """

    dataset: Dataset
    config: Any
    require_actions: bool = True

    def __post_init__(self) -> None:
        self.horizon = int(_config_get(self.config, "horizon"))
        self.discount = float(_config_get(self.config, "discount"))
        self.actor_p = _validate_goal_mix(
            _config_get(self.config, "actor_p"),
            name="actor_p",
        )
        self.critic_p = _validate_goal_mix(
            _config_get(self.config, "critic_p"),
            name="critic_p",
        )
        if self.horizon < _ACTION_HORIZON:
            raise ValueError(
                f"horizon must be at least {_ACTION_HORIZON}, got {self.horizon}."
            )
        if not 0.0 < self.discount < 1.0:
            raise ValueError(f"discount must lie in (0, 1), got {self.discount}.")
        if self.critic_p[0] > 0.0 or self.critic_p[3] > 0.0:
            raise ValueError(
                "critic_p must place all probability on one ordered future-goal "
                "component (p_geom or p_traj) for the transitive value objective."
            )

        observations = np.asarray(self.dataset["observations"])
        if observations.ndim != 2:
            raise ValueError(
                "PathBridger_dist supports state-vector observations only; "
                f"expected observations with shape [N, D], got {observations.shape}."
            )
        if self.require_actions and "actions" not in self.dataset:
            raise ValueError("PathBridgerDataset requires an 'actions' field.")
        if "terminals" not in self.dataset:
            raise ValueError(
                "PathBridgerDataset requires compact OGBench 'terminals' to preserve episode boundaries."
            )

        terminals = np.asarray(self.dataset["terminals"])
        if terminals.ndim != 1:
            raise ValueError(f"terminals must have shape [N], got {terminals.shape}.")
        self.size = self.dataset.size
        self.terminal_locs = np.flatnonzero(terminals > 0).astype(np.int64)
        if len(self.terminal_locs) == 0 or int(self.terminal_locs[-1]) != self.size - 1:
            raise ValueError("The final compact-dataset observation must be marked terminal.")
        self.initial_locs = np.concatenate(
            [np.asarray([0], dtype=np.int64), self.terminal_locs[:-1] + 1]
        )

        # Cache the episode terminal for every state and all starts whose full
        # K-window is present in one episode.
        self._final_for_idx = np.empty(self.size, dtype=np.int64)
        valid_parts: list[np.ndarray] = []
        for start, final in zip(self.initial_locs, self.terminal_locs):
            self._final_for_idx[start : final + 1] = final
            last_start = int(final) - self.horizon
            if last_start >= int(start):
                valid_parts.append(np.arange(start, last_start + 1, dtype=np.int64))
        if not valid_parts:
            raise ValueError(
                f"No episode contains a full horizon-{self.horizon} state window."
            )
        self.valid_starts = np.concatenate(valid_parts)

    def _validate_starts(self, idxs: Any) -> np.ndarray:
        idxs = np.asarray(idxs, dtype=np.int64)
        if idxs.ndim != 1 or len(idxs) == 0:
            raise ValueError(f"idxs must be a non-empty 1D array, got shape {idxs.shape}.")
        if np.any(idxs < 0) or np.any(idxs >= self.size):
            raise IndexError(f"Sample starts must lie in [0, {self.size}); got {idxs}.")
        finals = self._final_for_idx[idxs]
        bad = idxs + self.horizon > finals
        if np.any(bad):
            row = int(np.flatnonzero(bad)[0])
            raise ValueError(
                "A PathBridger training window cannot cross an episode boundary: "
                f"start={int(idxs[row])}, horizon={self.horizon}, terminal={int(finals[row])}."
            )
        return idxs

    @staticmethod
    def _uniform_positive_offsets(max_offsets: np.ndarray) -> np.ndarray:
        """Uniformly sample an integer in ``[1, max_offset]`` per row."""

        max_offsets = np.asarray(max_offsets, dtype=np.int64)
        if np.any(max_offsets < 1):
            raise ValueError("Positive future sampling requires at least one remaining state.")
        return 1 + np.floor(np.random.random(len(max_offsets)) * max_offsets).astype(np.int64)

    def _sample_goal_indices(
        self,
        idxs: np.ndarray,
        finals: np.ndarray,
        probabilities: tuple[float, float, float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample current, future, or random goals from one paper-format mix."""

        p_cur, p_geom, p_traj, p_rand = probabilities
        p_future = p_geom + p_traj
        batch_size = len(idxs)

        if p_cur == 1.0:
            components = np.full(batch_size, _CURRENT_GOAL, dtype=np.int8)
        elif p_future == 1.0:
            components = np.full(batch_size, _FUTURE_GOAL, dtype=np.int8)
        elif p_rand == 1.0:
            components = np.full(batch_size, _RANDOM_GOAL, dtype=np.int8)
        else:
            draws = np.random.random(batch_size)
            components = np.full(batch_size, _RANDOM_GOAL, dtype=np.int8)
            components[draws < p_cur + p_future] = _FUTURE_GOAL
            components[draws < p_cur] = _CURRENT_GOAL

        goal_idxs = idxs.copy()
        future_mask = components == _FUTURE_GOAL
        if np.any(future_mask):
            future_idxs = idxs[future_mask]
            future_finals = finals[future_mask]
            if p_geom > 0.0:
                offsets = np.random.geometric(
                    p=1.0 - self.discount,
                    size=len(future_idxs),
                ).astype(np.int64)
                goal_idxs[future_mask] = np.minimum(
                    future_idxs + offsets,
                    future_finals,
                )
            else:
                distances = np.random.random(len(future_idxs))
                goal_idxs[future_mask] = np.round(
                    (future_idxs + 1) * distances
                    + future_finals * (1.0 - distances)
                ).astype(np.int64)

        random_mask = components == _RANDOM_GOAL
        if np.any(random_mask):
            goal_idxs[random_mask] = self.dataset.get_random_idxs(
                int(np.sum(random_mask))
            )
        return goal_idxs, components

    def sample(self, batch_size: int, idxs: Any | None = None) -> dict[str, np.ndarray]:
        """Sample one PathBridger training batch.

        Endpoint and value goals follow ``actor_p`` and ``critic_p``.  The paper
        defaults select ordinary trajectory-future endpoints and geometric
        value goals, respectively.  Short base pairs use offsets 1--5, and
        transitive splits are active only for value pairs longer than five
        transitions.
        """

        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if idxs is None:
            choices = np.random.randint(0, len(self.valid_starts), size=batch_size)
            idxs = self.valid_starts[choices]
        else:
            idxs = self._validate_starts(idxs)
            if len(idxs) != batch_size:
                raise ValueError(
                    f"batch_size={batch_size} does not match the {len(idxs)} provided indices."
                )
        idxs = self._validate_starts(idxs)

        observations = np.asarray(self.dataset["observations"])
        finals = self._final_for_idx[idxs]
        remaining = finals - idxs

        # Endpoint proposer supervision follows actor_p.  A random conditioning
        # goal keeps the actual K-step endpoint on the source trajectory.
        endpoint_goal_idxs, endpoint_components = self._sample_goal_indices(
            idxs,
            finals,
            self.actor_p,
        )
        endpoint_target_idxs = np.minimum(idxs + self.horizon, endpoint_goal_idxs)
        random_endpoint_rows = endpoint_components == _RANDOM_GOAL
        endpoint_target_idxs[random_endpoint_rows] = (
            idxs[random_endpoint_rows] + self.horizon
        )
        bridge_target_idxs = idxs[:, None] + _BRIDGE_OFFSETS[None, :]
        bridge_target_idxs = np.minimum(
            bridge_target_idxs,
            endpoint_target_idxs[:, None],
        )

        # The final transitive value objective requires ordered future pairs.
        value_goal_idxs, _ = self._sample_goal_indices(
            idxs,
            finals,
            self.critic_p,
        )
        value_offsets = value_goal_idxs - idxs

        # The short anchor horizon H_b is fixed to five in the final method.
        base_max_offsets = np.minimum(5, remaining)
        base_offsets = self._uniform_positive_offsets(base_max_offsets)
        base_goal_idxs = idxs + base_offsets

        # TRL splits are strictly internal to long pairs.  Invalid rows use the
        # current state and zero offset; their contribution is removed by the
        # explicit float mask.
        transitive_valids = value_offsets > 5
        transitive_offsets = np.zeros(batch_size, dtype=np.int64)
        if np.any(transitive_valids):
            max_split_offsets = value_offsets[transitive_valids] - 1
            transitive_offsets[transitive_valids] = self._uniform_positive_offsets(
                max_split_offsets
            )
        transitive_idxs = idxs + transitive_offsets

        state_batch = {
            "observations": np.asarray(observations[idxs], dtype=np.float32),
            "next_observations": np.asarray(observations[idxs + 1], dtype=np.float32),
            "bridge_targets": np.asarray(
                observations[bridge_target_idxs],
                dtype=np.float32,
            ),
            "endpoint_goals": np.asarray(observations[endpoint_goal_idxs], dtype=np.float32),
            "endpoint_targets": np.asarray(observations[endpoint_target_idxs], dtype=np.float32),
            "value_goals": np.asarray(observations[value_goal_idxs], dtype=np.float32),
            "value_offsets": value_offsets.astype(np.float32),
            "base_goals": np.asarray(observations[base_goal_idxs], dtype=np.float32),
            "base_offsets": base_offsets.astype(np.float32),
            "transitive_subgoals": np.asarray(observations[transitive_idxs], dtype=np.float32),
            "transitive_offsets": transitive_offsets.astype(np.float32),
            "transitive_valids": transitive_valids.astype(np.float32),
        }
        if self.require_actions:
            # Preserve the original explicit public contract for legacy PBF.
            return {
                "observations": state_batch["observations"],
                "next_observations": state_batch["next_observations"],
                "actions": np.asarray(
                    self.dataset["actions"][idxs], dtype=np.float32
                ),
                "bridge_targets": state_batch["bridge_targets"],
                "endpoint_goals": state_batch["endpoint_goals"],
                "endpoint_targets": state_batch["endpoint_targets"],
                "value_goals": state_batch["value_goals"],
                "value_offsets": state_batch["value_offsets"],
                "base_goals": state_batch["base_goals"],
                "base_offsets": state_batch["base_offsets"],
                "transitive_subgoals": state_batch["transitive_subgoals"],
                "transitive_offsets": state_batch["transitive_offsets"],
                "transitive_valids": state_batch["transitive_valids"],
            }
        return state_batch


def action_free_view(dataset: Dataset) -> Dataset:
    """Return an immutable offline view with action/reward supervision removed."""

    forbidden = {'actions', 'rewards', 'returns', 'return_to_go', 'rtg'}
    fields = {key: dataset[key] for key in dataset if key.lower() not in forbidden}
    return Dataset.create(**fields)


__all__ = [
    "Dataset",
    "PathBridgerDataset",
    "PathBridgerDatasetConfig",
    "action_free_view",
    "observation_state_scale",
]
