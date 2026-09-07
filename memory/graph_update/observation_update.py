"""Observation-driven scene graph generation entry points.

This file intentionally keeps heavy ConceptGraphs imports inside functions so
the execution-delta path can run without loading VLM/SLAM dependencies.
"""

from __future__ import annotations

from argparse import Namespace
from copy import copy
from pathlib import Path
from typing import Any

from .incremental_observation import prepare_incremental_observation_dataset
from .execution_delta import infer_relations_for_new_objects
from .merge import load_json, save_json


def _metadata_object_to_scenegraph_node(obj: dict[str, Any], index: int) -> dict[str, Any]:
    object_type = str(obj.get("objectType") or obj.get("name") or "object")
    object_id = str(obj.get("objectId") or obj.get("name") or f"observed_object_{index}")
    position = obj.get("position") if isinstance(obj.get("position"), dict) else {}
    axis_aligned_bbox = obj.get("axisAlignedBoundingBox") if isinstance(obj.get("axisAlignedBoundingBox"), dict) else {}
    center = axis_aligned_bbox.get("center") if isinstance(axis_aligned_bbox.get("center"), dict) else position
    size = axis_aligned_bbox.get("size") if isinstance(axis_aligned_bbox.get("size"), dict) else {}

    node: dict[str, Any] = {
        "id": index,
        "original_id": index,
        "pruned_id": index,
        "ai2thor_object_id": object_id,
        "objectId": object_id,
        "object_tag": object_type,
        "caption": f"AI2-THOR observed {object_type}",
        "possible_tags": [object_type],
        "merge_source": "ai2thor_metadata_fallback",
    }
    if center:
        node["bbox_center"] = [center.get("x", 0.0), center.get("y", 0.0), center.get("z", 0.0)]
    if size:
        node["bbox_extent"] = [size.get("x", 0.0), size.get("y", 0.0), size.get("z", 0.0)]
    return node


def _ensure_observed_scenegraph_has_nodes(result: dict[str, Any], prepared: dict[str, Any], cachedir: Path) -> dict[str, Any]:
    scene_graph_path = Path(result["scene_graph"]) if result.get("scene_graph") else cachedir / "scene_graph.json"
    existing_nodes: list[dict[str, Any]] = []
    if scene_graph_path.exists():
        loaded = load_json(scene_graph_path)
        if isinstance(loaded, list):
            existing_nodes = [node for node in loaded if isinstance(node, dict)]

    if existing_nodes or int(prepared.get("num_new_objects") or 0) <= 0:
        return result

    obj_meta_path = Path(prepared["dataset_root"]) / str(prepared["scene_id"]) / "obj_meta.json"
    metadata = load_json(obj_meta_path) if obj_meta_path.exists() else []
    fallback_nodes = [
        _metadata_object_to_scenegraph_node(obj, index)
        for index, obj in enumerate(metadata)
        if isinstance(obj, dict) and obj.get("objectId")
    ]
    if not fallback_nodes:
        return result

    save_json(fallback_nodes, scene_graph_path)
    result = {
        **result,
        "scene_graph": str(scene_graph_path),
        "fallback_observed_nodes": True,
        "fallback_observed_node_count": len(fallback_nodes),
        "fallback_reason": "ConceptGraphs incremental stage produced zero nodes; using visible AI2-THOR metadata nodes.",
    }
    return result


def build_observed_scenegraph(
    *,
    dataset_root: Path,
    scene_id: str,
    cachedir: Path,
    args: Namespace,
) -> dict[str, Any]:
    """Run the existing ConceptGraphs scene graph pipeline on saved RGB-D data.

    ``dataset_root`` should contain the AI2-THOR-style folders expected by the
    current dataset loader, such as ``color/``, ``depth/``, ``pose/`` and
    ``obj_meta.json`` under ``scene_id``. The easiest producer today is
    ``planning.datagen.generation.save_observation_dataset``.
    """

    from planning.datagen.generation import run_scenegraph_pipeline

    return run_scenegraph_pipeline(
        dataset_root=Path(dataset_root),
        scene_id=scene_id,
        cachedir=Path(cachedir),
        args=args,
    )


def build_observed_scenegraph_from_stage(stage_info: dict[str, Any], args: Namespace) -> dict[str, Any]:
    """Build a scene graph from a stage dict returned by ``collect_stage``."""

    stage_dir = Path(stage_info["stage_dir"])
    return build_observed_scenegraph(
        dataset_root=Path(stage_info["dataset_root"]),
        scene_id=str(stage_info["scene_id"]),
        cachedir=stage_dir / "sg_cache",
        args=args,
    )


def build_incremental_observed_scenegraph(
    *,
    dataset_root: Path,
    scene_id: str,
    cachedir: Path,
    stored_scenegraph_path: Path,
    args: Namespace,
    filtered_dataset_root: Path | None = None,
    instance_color_maps: dict[str, dict[Any, str]] | None = None,
    observed_object_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Run ConceptGraphs only with object categories absent from the stored graph."""

    filtered_root = filtered_dataset_root or (cachedir / "incremental_dataset")
    prepared = prepare_incremental_observation_dataset(
        dataset_root=dataset_root,
        scene_id=scene_id,
        stored_scenegraph_path=stored_scenegraph_path,
        output_root=filtered_root,
        instance_color_maps=instance_color_maps,
        observed_object_ids=observed_object_ids,
    )
    if prepared["num_new_objects"] == 0:
        return {
            **prepared,
            "skipped": True,
            "reason": "no previously unseen AI2-THOR objects in observation metadata",
            "scene_graph": None,
            "relations": None,
            "relation_deltas": [],
        }

    incremental_args = copy(args)
    incremental_args.class_set = "scene"
    result = build_observed_scenegraph(
        dataset_root=Path(prepared["dataset_root"]),
        scene_id=scene_id,
        cachedir=cachedir,
        args=incremental_args,
    )
    result = _ensure_observed_scenegraph_has_nodes(result, prepared, cachedir)
    observed_metadata = load_json(dataset_root / scene_id / "obj_meta.json")
    relation_deltas = infer_relations_for_new_objects(
        observed_metadata,
        prepared["new_object_ids"],
    )
    return {
        **result,
        **prepared,
        "skipped": False,
        "relation_deltas": [delta.to_dict() for delta in relation_deltas],
    }
