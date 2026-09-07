from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from .episode import MapThorEpisode


CONFIG_GLOB = "config_type*.json"
TASK_NAME_ALIASES = {
    # The public config contains this stale name while the official Tasks
    # directory uses the name below.
    "2_put_all_potatoes_bowl": "2_put_all_tomatoes_potatoes_fridge",
}
INSTRUCTION_OVERRIDES = {
    # Upstream config_type2 accidentally repeats the drawer instruction.
    "2_open_all_cabinets": "Open all the cabinets",
}



def _category(task_name: str, config_path: Path) -> int:
    match = re.match(r"(\d+)_", task_name)
    if match:
        return int(match.group(1))
    match = re.search(r"type(\d+)", config_path.stem)
    return int(match.group(1)) if match else 0


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def discover_config_files(config_dir: Path) -> list[Path]:
    files = sorted(config_dir.glob(CONFIG_GLOB))
    if not files:
        raise FileNotFoundError(
            f"No {CONFIG_GLOB} files found under {config_dir}. "
            "Point --mapthor-root at the official LLaMAR checkout or pass "
            "--mapthor-config-dir."
        )
    return files


def load_episodes(
    config_dir: Path,
    *,
    task_categories: Iterable[int] | None = None,
    task_ids: Iterable[int] | None = None,
    task_names: Iterable[str] | None = None,
    floorplan_indices: Iterable[int] | None = None,
    seeds: Iterable[int] = (0,),
    agent_count: int = 2,
    tasks_root: Path | None = None,
    require_assets: bool = True,
) -> list[MapThorEpisode]:
    """Expand official MAP-THOR JSON configs into concrete episodes."""

    category_filter = set(task_categories or [])
    id_filter = set(task_ids or [])
    name_filter = set(task_names or [])
    floor_filter = set(floorplan_indices or [])
    episodes: list[MapThorEpisode] = []

    for config_path in discover_config_files(config_dir):
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        for raw in payload.get("tasks") or []:
            configured_name = str(raw["task_name"])
            task_name = TASK_NAME_ALIASES.get(configured_name, configured_name)
            category = _category(task_name, config_path)
            task_id = int(raw.get("task_id", 0))
            if category_filter and category not in category_filter:
                continue
            if id_filter and task_id not in id_filter:
                continue
            if name_filter and task_name not in name_filter and configured_name not in name_filter:
                continue

            floorplans = [str(item) for item in raw.get("task_floorplans") or []]
            for floor_index, floorplan in enumerate(floorplans):
                if floor_filter and floor_index not in floor_filter:
                    continue
                if tasks_root is not None and require_assets:
                    initializer = tasks_root / task_name / f"{floorplan}.py"
                    checker = tasks_root / task_name / "checker.py"
                    if not initializer.exists() or not checker.exists():
                        continue

                for seed in seeds:
                    episode_id = _safe_id(
                        f"type{category}_task{task_id}_{task_name}_{floorplan}_seed{seed}_agents{agent_count}"
                    )
                    episodes.append(
                        MapThorEpisode(
                            episode_id=episode_id,
                            task_name=task_name,
                            instruction=INSTRUCTION_OVERRIDES.get(task_name, str(raw["task_description"])),
                            floorplan=floorplan,
                            task_id=task_id,
                            task_category=category,
                            task_type=str(raw.get("task_type") or "unknown"),
                            timeout=int(raw.get("task_timeout") or 30),
                            seed=int(seed),
                            agent_count=int(agent_count),
                            checklist=tuple(str(item) for item in raw.get("task_checklist") or []),
                            config_path=str(config_path),
                            metadata={
                                "configured_task_name": configured_name,
                                "floorplan_index": floor_index,
                                "task_complexity": raw.get("task_complexity"),
                            },
                        )
                    )
    return episodes


def select_episode(
    config_dir: Path,
    *,
    task_id: int | None,
    task_name: str | None,
    floorplan_index: int,
    seed: int,
    agent_count: int,
    tasks_root: Path,
) -> MapThorEpisode:
    episodes = load_episodes(
        config_dir,
        task_ids=[task_id] if task_id is not None else None,
        task_names=[task_name] if task_name else None,
        floorplan_indices=[floorplan_index],
        seeds=[seed],
        agent_count=agent_count,
        tasks_root=tasks_root,
        require_assets=True,
    )
    if not episodes:
        selector = f"task_id={task_id}" if task_id is not None else f"task_name={task_name!r}"
        raise LookupError(
            f"No runnable MAP-THOR episode for {selector}, floorplan_index={floorplan_index}."
        )
    if len(episodes) > 1 and task_id is None and not task_name:
        raise LookupError("Task selection is ambiguous; pass --mapthor-task-id or --mapthor-task-name.")
    return episodes[0]
