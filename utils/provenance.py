"""Result provenance and information-boundary validation."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from utils.af_data import FORBIDDEN_OFFLINE_FIELDS


PORT_KINDS = frozenset(
    {'proposed', 'official_port', 'goal_conditioned_adaptation', 'paper_reimplementation', 'online_only', 'full_action'}
)


@dataclasses.dataclass(frozen=True)
class AlgorithmMetadata:
    algorithm: str
    port_kind: str
    paper_url: str
    official_repo_url: str | None
    official_repo_commit: str | None
    offline_fields_seen: tuple[str, ...]
    online_modules_updated: tuple[str, ...]
    uses_offline_actions: bool = False
    uses_offline_logged_rewards: bool = False

    def validate(self) -> None:
        if not self.algorithm.strip():
            raise ValueError('algorithm cannot be empty.')
        if self.port_kind not in PORT_KINDS:
            raise ValueError(f'Unknown port_kind {self.port_kind!r}.')
        fields = {field.lower() for field in self.offline_fields_seen}
        leaked = sorted(fields & FORBIDDEN_OFFLINE_FIELDS)
        action_free = self.port_kind != 'full_action' and self.port_kind != 'online_only'
        if action_free and (leaked or self.uses_offline_actions or self.uses_offline_logged_rewards):
            raise ValueError(
                f'Action-free algorithm {self.algorithm} violates its offline contract: {leaked}.'
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return dataclasses.asdict(self)

    def write_json(self, path: str | Path, **run_fields: Any) -> None:
        payload = {**self.to_dict(), **run_fields}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
        temporary.replace(path)


__all__ = ['AlgorithmMetadata', 'PORT_KINDS']
