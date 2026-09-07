from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from .episode import MapThorEpisode


def resolve_mapthor_root(explicit: str | Path | None = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("MAPTHOR_ROOT"):
        candidates.append(Path(os.environ["MAPTHOR_ROOT"]))
    project_root = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            project_root / "benchmark" / "mapthor_assets",
            project_root / "external" / "LLaMAR",
            project_root / "LLaMAR",
        ]
    )
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "AI2Thor" / "Tasks").is_dir() and (resolved / "configs").is_dir():
            return resolved
    searched = ", ".join(str(item) for item in candidates)
    raise FileNotFoundError(
        "MAP-THOR assets were not found. Clone https://github.com/nsidn98/LLaMAR "
        f"and pass --mapthor-root. Searched: {searched}"
    )


def _load_module(path: Path, logical_name: str, mapthor_root: Path) -> ModuleType:
    if str(mapthor_root) not in sys.path:
        sys.path.insert(0, str(mapthor_root))
    module_name = f"emas_mapthor_{logical_name}_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_official_components(
    mapthor_root: Path,
    episode: MapThorEpisode,
) -> tuple[Any, Any]:
    task_dir = mapthor_root / "AI2Thor" / "Tasks" / episode.task_dir_name
    initializer_path = task_dir / f"{episode.floorplan}.py"
    checker_path = task_dir / "checker.py"
    if not initializer_path.exists():
        raise FileNotFoundError(f"Missing MAP-THOR initializer: {initializer_path}")
    if not checker_path.exists():
        raise FileNotFoundError(f"Missing MAP-THOR checker: {checker_path}")
    initializer_module = _load_module(
        initializer_path, f"{episode.task_name}_{episode.floorplan}_initializer", mapthor_root
    )
    checker_module = _load_module(checker_path, f"{episode.task_name}_checker", mapthor_root)
    return initializer_module.SceneInitializer(), checker_module.Checker()


def initialize_episode(
    controller: Any,
    episode: MapThorEpisode,
    mapthor_root: Path,
) -> Any:
    """Apply the official task/floorplan initializer and prepare its checker."""

    initializer, checker = load_official_components(mapthor_root, episode)
    event = initializer.preinit(controller.last_event, controller)
    object_ids = [
        str(obj["objectId"])
        for obj in (controller.last_event.metadata.get("objects") or [])
        if obj.get("objectId")
    ]
    if hasattr(checker, "all_objects"):
        checker.all_objects(object_ids, episode.floorplan)
    return checker


def snapshot_objects(controller: Any) -> list[dict[str, Any]]:
    metadata = getattr(getattr(controller, "last_event", None), "metadata", {}) or {}
    return [dict(item) for item in metadata.get("objects") or []]
