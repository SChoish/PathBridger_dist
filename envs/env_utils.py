"""OGBench environment and compact state/pixel dataset loading.

The state and visual entry points are intentionally separate.  The visual
loader exposes either a strict action-free view or a full-action view while
always dropping rewards, simulator state, and every other privileged field.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from utils.datasets import Dataset


def _is_validation_file(path: Path) -> bool:
    stem = path.stem.lower()
    parent_names = {parent.name.lower() for parent in path.parents}
    return (
        stem.endswith(('-val', '_val'))
        or stem.startswith(('val-', 'val_', 'validation-', 'validation_'))
        or '-val-' in stem
        or '_val_' in stem
        or 'val' in parent_names
        or 'validation' in parent_names
    )


def _merge_compact_shards(
    ogbench: Any,
    paths: Sequence[Path],
    *,
    observation_dtype: Any = np.float32,
) -> dict[str, np.ndarray]:
    if not paths:
        raise ValueError('Cannot merge an empty shard list.')

    merged_parts: dict[str, list[np.ndarray]] = {}
    expected_keys: set[str] | None = None
    for path in paths:
        raw = ogbench.load_dataset(
            str(path),
            ob_dtype=observation_dtype,
            compact_dataset=True,
        )
        if not isinstance(raw, Mapping):
            raise TypeError(f'OGBench shard {path} did not contain a dataset mapping.')
        keys = set(raw)
        if expected_keys is None:
            expected_keys = keys
            merged_parts = {key: [] for key in raw}
        elif keys != expected_keys:
            missing = sorted(expected_keys - keys)
            extra = sorted(keys - expected_keys)
            raise ValueError(
                f'Inconsistent fields in shard {path}: missing={missing}, extra={extra}.'
            )
        for key, value in raw.items():
            array = np.asarray(value)
            if array.ndim == 0:
                raise ValueError(f'Shard field {key!r} in {path} has no batch dimension.')
            merged_parts[key].append(array)

    return {
        key: np.concatenate(parts, axis=0)
        for key, parts in merged_parts.items()
    }


def _as_state_dataset(raw: Mapping[str, Any], *, split: str) -> Dataset:
    dataset = Dataset.create(**raw)
    observations = np.asarray(dataset['observations'])
    if observations.ndim != 2:
        raise ValueError(
            'PathBridger is state-based: '
            f'{split} observations must have shape [N, D], got {observations.shape}.'
        )
    return dataset


def _as_pixel_dataset(
    raw: Mapping[str, Any], *, split: str, action_free: bool = True
) -> Dataset:
    """Build either the strict action-free or full offline pixel view."""

    if 'observations' not in raw or 'terminals' not in raw:
        raise ValueError(
            f'{split} visual data must contain observations and terminals.'
        )
    observations = np.asarray(raw['observations'])
    terminals = np.asarray(raw['terminals'], dtype=np.float32).reshape(-1)
    if observations.ndim != 4 or observations.shape[-1] != 3:
        raise ValueError(
            f'{split} visual observations must have shape [N, H, W, 3], '
            f'got {observations.shape}.'
        )
    if observations.dtype != np.uint8:
        raise ValueError(
            f'{split} visual observations must be uint8, got {observations.dtype}.'
        )
    if len(observations) != len(terminals):
        raise ValueError(
            f'{split} visual observations and terminals have different lengths.'
        )
    fields = {'observations': observations, 'terminals': terminals}
    if not action_free:
        if 'actions' not in raw:
            raise ValueError(f'{split} full pixel data must contain actions.')
        actions = np.asarray(raw['actions'], dtype=np.float32)
        if actions.ndim != 2 or len(actions) != len(observations):
            raise ValueError(
                f'{split} pixel actions must have shape [N, A], got '
                f'{actions.shape} for N={len(observations)}.'
            )
        fields['actions'] = actions
    # Never expose qpos, qvel, rewards, or other privileged NPZ fields.
    return Dataset.create(**fields)


def _local_npz_paths(
    dataset_name: str,
    resolved_dir: Path | None,
) -> list[Path]:
    if resolved_dir is None or not resolved_dir.is_dir():
        return []
    npz_paths = sorted(path for path in resolved_dir.rglob('*.npz') if path.is_file())
    name = str(dataset_name).lower()
    named_paths = [
        path
        for path in npz_paths
        if path.stem.lower() == name
        or path.stem.lower().startswith(f'{name}-')
        or path.stem.lower().startswith(f'{name}_')
        or name in {parent.name.lower() for parent in path.parents}
    ]
    if named_paths:
        return named_paths

    conventional_roots = {'train', 'val', 'validation'}
    relative_parts = [path.relative_to(resolved_dir).parts for path in npz_paths]
    conventional_layout = all(
        len(parts) == 1 or parts[0].lower() in conventional_roots
        for parts in relative_parts
    )
    has_train_and_val = (
        any(not _is_validation_file(path) for path in npz_paths)
        and any(_is_validation_file(path) for path in npz_paths)
    )
    if resolved_dir.name.lower() == name or (
        conventional_layout and has_train_and_val
    ):
        return npz_paths
    return []


def make_env_and_datasets(
    dataset_name: str,
    dataset_dir: str | Path | None = None,
    **env_kwargs: Any,
) -> tuple[Any, Dataset, Dataset]:
    """Create an OGBench environment and immutable compact train/val datasets.

    When ``dataset_dir`` contains NPZ files, every train shard and validation
    shard is loaded with ``compact_dataset=True`` and concatenated in sorted path
    order.  Validation files may use ``*-val.npz``/``*_val.npz`` names or live
    under a ``val``/``validation`` directory.  If the directory contains no NPZ
    files, it is passed to OGBench so OGBench can resolve or download its normal
    dataset files there.
    """

    if not str(dataset_name).strip():
        raise ValueError('dataset_name must be a non-empty OGBench dataset name.')

    try:
        import ogbench
    except ImportError as exc:
        raise ImportError(
            'OGBench is required to create PathBridger environments and datasets.'
        ) from exc

    resolved_dir: Path | None = None
    if dataset_dir is not None and str(dataset_dir).strip():
        resolved_dir = Path(dataset_dir).expanduser()

    npz_paths = _local_npz_paths(dataset_name, resolved_dir)

    if npz_paths:
        train_paths = [path for path in npz_paths if not _is_validation_file(path)]
        val_paths = [path for path in npz_paths if _is_validation_file(path)]
        if not train_paths or not val_paths:
            raise ValueError(
                f'dataset_dir={resolved_dir} must contain both train and validation NPZ files; '
                f'found {len(train_paths)} train and {len(val_paths)} validation files.'
            )
        env = ogbench.make_env_and_datasets(dataset_name, env_only=True, **env_kwargs)
        train_raw = _merge_compact_shards(ogbench, train_paths)
        val_raw = _merge_compact_shards(ogbench, val_paths)
    else:
        load_kwargs = dict(env_kwargs)
        if resolved_dir is not None:
            load_kwargs['dataset_dir'] = str(resolved_dir)
        env, train_raw, val_raw = ogbench.make_env_and_datasets(
            dataset_name,
            compact_dataset=True,
            **load_kwargs,
        )

    train_dataset = _as_state_dataset(train_raw, split='training')
    val_dataset = _as_state_dataset(val_raw, split='validation')
    return env, train_dataset, val_dataset


def make_pixel_env_and_datasets(
    dataset_name: str,
    dataset_dir: str | Path | None = None,
    *,
    allow_download: bool = False,
    action_free: bool = True,
    **env_kwargs: Any,
) -> tuple[Any, Dataset, Dataset]:
    """Create an OGBench visual environment and strict pixel datasets.

    This entry point accepts only ``visual-*`` OGBench tasks, keeps RGB arrays
    as compact ``uint8`` values.  ``action_free=True`` returns only images and
    terminals; full offline mode additionally returns actions.
    """

    if 'visual-' not in str(dataset_name).lower():
        raise ValueError(
            'Pixel benchmarks require an official visual-* OGBench dataset name.'
        )
    try:
        import ogbench
    except ImportError as exc:
        raise ImportError(
            'OGBench is required to create visual environments and datasets.'
        ) from exc

    resolved_dir = None
    if dataset_dir is not None and str(dataset_dir).strip():
        resolved_dir = Path(dataset_dir).expanduser()
    lookup_dir = resolved_dir or Path('~/.ogbench/data').expanduser()
    npz_paths = _local_npz_paths(dataset_name, lookup_dir)
    if npz_paths:
        train_paths = [path for path in npz_paths if not _is_validation_file(path)]
        val_paths = [path for path in npz_paths if _is_validation_file(path)]
        if not train_paths or not val_paths:
            raise ValueError(
                f'dataset_dir={resolved_dir} must contain both train and validation '
                f'NPZ files; found {len(train_paths)} train and {len(val_paths)} '
                'validation files.'
            )
        env = ogbench.make_env_and_datasets(dataset_name, env_only=True, **env_kwargs)
        train_raw = _merge_compact_shards(
            ogbench, train_paths, observation_dtype=np.uint8
        )
        val_raw = _merge_compact_shards(
            ogbench, val_paths, observation_dtype=np.uint8
        )
    elif allow_download:
        load_kwargs = dict(env_kwargs)
        if resolved_dir is not None:
            load_kwargs['dataset_dir'] = str(resolved_dir)
        env, train_raw, val_raw = ogbench.make_env_and_datasets(
            dataset_name,
            compact_dataset=True,
            **load_kwargs,
        )
    else:
        raise FileNotFoundError(
            f'No local NPZ files found for {dataset_name!r} under {lookup_dir}. '
            'Provision the visual dataset explicitly or pass '
            '--allow_dataset_download=true.'
        )
    return (
        env,
        _as_pixel_dataset(train_raw, split='training', action_free=action_free),
        _as_pixel_dataset(val_raw, split='validation', action_free=action_free),
    )


__all__ = ['make_env_and_datasets', 'make_pixel_env_and_datasets']
