"""Local experiment logging with optional Weights & Biases integration."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any


def _serializable(value: Any) -> Any:
    """Convert common flag/config values to JSON-compatible Python values."""
    if hasattr(value, 'to_dict'):
        value = value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'item'):
        try:
            return _serializable(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, 'tolist'):
        try:
            return _serializable(value.tolist())
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def get_flag_dict(flag_values=None) -> dict[str, Any]:
    """Return all top-level flags as a JSON-serializable dictionary."""
    if flag_values is None:
        from absl import flags

        flag_values = flags.FLAGS

    if isinstance(flag_values, Mapping):
        raw = dict(flag_values)
    else:
        raw = {}
        for name in flag_values:
            if '.' not in name:
                raw[name] = getattr(flag_values, name)
    return {name: _serializable(value) for name, value in raw.items()}


def save_flag_dict(path: str | os.PathLike[str], flag_values=None) -> str:
    """Serialize flags to a readable JSON file and return its path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as file:
        json.dump(get_flag_dict(flag_values), file, indent=2, sort_keys=True)
        file.write('\n')
    return str(output_path)


def _slug(value: Any) -> str:
    return str(value).strip().replace('/', '_').replace('\\', '_').replace(' ', '_')


def get_exp_name(seed: int, env_name: str | None = None, agent_name: str | None = None) -> str:
    """Build a concise, filesystem-safe experiment name."""
    parts = [_slug(value) for value in (agent_name, env_name) if value]
    parts.append(f'sd{int(seed):03d}')
    if job_id := os.environ.get('SLURM_JOB_ID'):
        parts.append(f'job{_slug(job_id)}')
    if process_id := os.environ.get('SLURM_PROCID'):
        parts.append(f'proc{_slug(process_id)}')
    parts.append(datetime.now().strftime('%Y%m%d_%H%M%S'))
    return '_'.join(parts)


def _csv_value(value: Any) -> Any:
    value = _serializable(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(',', ':'))
    return value


class CsvLogger:
    """Write scalar metric dictionaries to a local CSV file."""

    def __init__(self, path: str | os.PathLike[str], *, resume: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._writer = None
        self._header: list[str] | None = None

        if resume and self.path.is_file() and self.path.stat().st_size:
            with self.path.open('r', newline='', encoding='utf-8') as file:
                self._header = next(csv.reader(file))
            self._open('a')

    def _open(self, mode: str) -> None:
        self._file = self.path.open(mode, newline='', encoding='utf-8')
        if self._header is not None:
            self._writer = csv.DictWriter(self._file, fieldnames=self._header, extrasaction='ignore')

    def log(self, row: Mapping[str, Any], step: int) -> None:
        record = {str(key): _csv_value(value) for key, value in row.items()}
        record['step'] = int(step)

        if self._file is None:
            self._header = list(record)
            self._open('w')
            self._writer.writeheader()

        self._writer.writerow(record)
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def setup_wandb(
    *,
    project: str = 'PathBridger',
    entity: str | None = None,
    group: str | None = None,
    name: str | None = None,
    config: Mapping[str, Any] | None = None,
    mode: str = 'online',
    directory: str | os.PathLike[str] | None = None,
):
    """Initialize an optional W&B run without making W&B a base dependency."""
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError('Weights & Biases is optional; install PathBridger with the `tracking` extra.') from error

    wandb_dir = str(directory) if directory is not None else tempfile.mkdtemp(prefix='pathbridger-wandb-')
    run_config = get_flag_dict() if config is None else _serializable(config)
    return wandb.init(
        project=project,
        entity=entity,
        group=group,
        name=name,
        config=run_config,
        mode=mode,
        dir=wandb_dir,
    )


__all__ = ['CsvLogger', 'get_exp_name', 'get_flag_dict', 'save_flag_dict', 'setup_wandb']
