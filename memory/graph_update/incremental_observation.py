"""Prepare observation datasets that contain only previously unseen objects."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .merge import load_json


OBJECT_ID_FIELDS = ("ai2thor_object_id", "objectId", "object_id")
STRUCTURAL_OBJECT_TYPES = {"floor", "wall", "ceiling", "room"}


def is_structural_object(obj: dict[str, Any]) -> bool:
    object_id = str(obj.get("objectId") or "").lower()
    object_type = str(obj.get("objectType") or obj.get("name") or "").lower()
    return object_id.startswith("room|") or object_type in STRUCTURAL_OBJECT_TYPES


def scenegraph_object_ids(nodes: Sequence[dict[str, Any]]) -> set[str]:
    """Return stable AI2-THOR ids already represented by scene-graph nodes."""

    object_ids: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict):
            continue
        for field in OBJECT_ID_FIELDS:
            value = node.get(field)
            if value not in (None, ""):
                object_ids.add(str(value))
                break
    return object_ids


def partition_observed_objects(
    observed_objects: Sequence[dict[str, Any]],
    known_object_ids: set[str],
    observed_object_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split AI2-THOR metadata into unseen and already recorded objects."""

    new_objects: list[dict[str, Any]] = []
    known_objects: list[dict[str, Any]] = []
    for obj in observed_objects:
        if not isinstance(obj, dict) or not obj.get("objectId"):
            continue
        if is_structural_object(obj):
            continue
        object_id = str(obj["objectId"])
        if observed_object_ids is not None and object_id not in observed_object_ids:
            continue
        if observed_object_ids is None and obj.get("visible") is False:
            continue
        target = known_objects if object_id in known_object_ids else new_objects
        target.append(deepcopy(obj))
    return new_objects, known_objects


def _link_dataset(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source_path in source.rglob("*"):
        relative_path = source_path.relative_to(source)
        destination_path = destination / relative_path
        if source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
        elif source_path.name != "obj_meta.json":
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            if not destination_path.exists():
                os.symlink(source_path.resolve(), destination_path)


def _color_key(value: Any) -> tuple[int, int, int]:
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = list(value)
    if len(parts) != 3:
        raise ValueError(f"Instance color must have three channels, got {value!r}")
    return tuple(int(channel) for channel in parts)


def mask_known_instances(
    scene_dir: Path,
    known_object_ids: set[str],
    instance_color_maps: Mapping[str, Mapping[Any, str]],
) -> int:
    """Black out known simulator instances before GroundingDINO sees RGB frames."""

    import imageio.v2 as imageio
    import numpy as np

    masked_frames = 0
    for frame_name, color_map in instance_color_maps.items():
        color_path = scene_dir / "color" / frame_name
        instance_path = scene_dir / "instance" / frame_name
        if not color_path.exists() or not instance_path.exists():
            continue

        color = np.asarray(imageio.imread(color_path)).copy()
        instances = np.asarray(imageio.imread(instance_path))
        changed = False
        for instance_color, object_id in color_map.items():
            if str(object_id) not in known_object_ids:
                continue
            mask = np.all(instances[..., :3] == np.asarray(_color_key(instance_color)), axis=-1)
            if np.any(mask):
                color[mask] = 0
                changed = True
        if changed:
            color_path.unlink()
            imageio.imwrite(color_path, color)
            masked_frames += 1
    return masked_frames


def prepare_incremental_observation_dataset(
    *,
    dataset_root: Path,
    scene_id: str,
    stored_scenegraph_path: Path,
    output_root: Path,
    instance_color_maps: Mapping[str, Mapping[Any, str]] | None = None,
    observed_object_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Create a lightweight dataset view whose detector vocabulary is unseen objects.

    The original ConceptGraphs pipeline uses ``obj_meta.json`` when
    ``class_set=scene``. Replacing that file in a linked dataset narrows the
    detector prompt without modifying ConceptGraphs itself. Supplying instance
    color maps additionally removes known same-class instances from RGB input.
    """

    source_scene_dir = Path(dataset_root) / scene_id
    observed_metadata_path = source_scene_dir / "obj_meta.json"
    stored_nodes = load_json(stored_scenegraph_path) if stored_scenegraph_path.exists() else []
    observed_objects = load_json(observed_metadata_path) if observed_metadata_path.exists() else []
    known_ids = scenegraph_object_ids(stored_nodes)
    new_objects, known_objects = partition_observed_objects(
        observed_objects,
        known_ids,
        observed_object_ids=observed_object_ids,
    )

    output_scene_dir = Path(output_root) / scene_id
    _link_dataset(source_scene_dir, output_scene_dir)
    (output_scene_dir / "obj_meta.json").write_text(
        json.dumps(new_objects, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    masked_frames = 0
    if instance_color_maps and known_ids:
        masked_frames = mask_known_instances(output_scene_dir, known_ids, instance_color_maps)

    return {
        "dataset_root": str(output_root),
        "scene_id": scene_id,
        "scene_dir": str(output_scene_dir),
        "new_object_ids": [str(obj["objectId"]) for obj in new_objects],
        "new_object_types": sorted({str(obj.get("objectType")) for obj in new_objects if obj.get("objectType")}),
        "known_object_ids": sorted(known_ids),
        "num_new_objects": len(new_objects),
        "num_known_observed_objects": len(known_objects),
        "num_masked_frames": masked_frames,
        "can_filter_by_identity": bool(known_ids),
    }
