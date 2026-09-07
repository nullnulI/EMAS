from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MapThorEpisode:
    """One MAP-THOR task/floorplan/seed evaluation unit."""

    episode_id: str
    task_name: str
    instruction: str
    floorplan: str
    task_id: int
    task_category: int
    task_type: str
    timeout: int
    seed: int = 0
    agent_count: int = 2
    checklist: tuple[str, ...] = ()
    config_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def task_dir_name(self) -> str:
        return self.task_name

    def result_dir(self, root: Path) -> Path:
        return root / self.episode_id
