"""Merge observed scene graphs and execution-derived relation deltas."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .schemas import ObjectDelta, RelationDelta

SUPPORTED_RELATION_RECORDS = {"a on b", "b on a", "a in b", "b in a"}


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def canonical_object_id(value: Any) -> int | str | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def node_scenegraph_id(node: dict[str, Any]) -> int | str | None:
    for key in ("original_id", "id", "pruned_id"):
        value = canonical_object_id(node.get(key))
        if value is not None:
            return value
    return None


def node_key(node: dict[str, Any]) -> tuple[str, ...]:
    for key in ("ai2thor_object_id", "objectId", "object_id", "name"):
        value = node.get(key)
        if value not in (None, ""):
            return ("ai2thor_object_id", str(value))

    scenegraph_id = node_scenegraph_id(node)
    if scenegraph_id is not None:
        return ("scenegraph_id", str(scenegraph_id))

    return (
        "object_text",
        str(node.get("object_tag", "")).strip().lower(),
        str(node.get("caption", "")).strip().lower(),
    )


def canonicalize_scenegraph_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    canonical_nodes = []
    for pruned_id, node in enumerate(nodes):
        canonical_node = deepcopy(node)
        canonical_node["pruned_id"] = pruned_id
        canonical_node.setdefault("original_id", canonical_node.get("id", pruned_id))
        canonical_nodes.append(canonical_node)
    return canonical_nodes


def merge_scenegraph_nodes(
    stored_nodes: list[dict[str, Any]],
    observed_nodes: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Merge stored and newly observed nodes.

    Observed nodes replace stored nodes with the same stable key. Unknown nodes
    are appended and then re-pruned with contiguous ``pruned_id`` values.
    """

    nodes_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    order: list[tuple[str, ...]] = []

    for source_name, nodes in (("stored", stored_nodes or []), ("observed", observed_nodes or [])):
        for node in nodes:
            if not isinstance(node, dict):
                continue
            key = node_key(node)
            merged = deepcopy(node)
            merged["merge_source"] = source_name
            if key not in nodes_by_key:
                order.append(key)
            nodes_by_key[key] = merged

    return canonicalize_scenegraph_nodes([nodes_by_key[key] for key in order])


def append_new_scenegraph_nodes(
    stored_nodes: list[dict[str, Any]],
    observed_nodes: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Append incrementally detected nodes without trusting local graph ids.

    A separately built observation graph starts numbering nodes at zero, so its
    ``id/original_id/pruned_id`` values cannot be used to match global nodes.
    Stable AI2-THOR ids are still deduplicated when present.
    """

    merged = [deepcopy(node) for node in stored_nodes if isinstance(node, dict)]
    known_ai2thor_ids = {
        str(node[field])
        for node in merged
        for field in ("ai2thor_object_id", "objectId", "object_id")
        if node.get(field) not in (None, "")
    }
    next_original_id = max(
        (int(value) for node in merged for value in [node_scenegraph_id(node)] if isinstance(value, int)),
        default=-1,
    ) + 1

    for observed in observed_nodes or []:
        if not isinstance(observed, dict):
            continue
        stable_id = next(
            (
                str(observed[field])
                for field in ("ai2thor_object_id", "objectId", "object_id")
                if observed.get(field) not in (None, "")
            ),
            None,
        )
        if stable_id is not None and stable_id in known_ai2thor_ids:
            continue
        node = deepcopy(observed)
        node["observed_original_id"] = node.get("original_id", node.get("id"))
        node["original_id"] = next_original_id
        node["id"] = next_original_id
        node["merge_source"] = "incremental_observation"
        merged.append(node)
        next_original_id += 1
        if stable_id is not None:
            known_ai2thor_ids.add(stable_id)
    return canonicalize_scenegraph_nodes(merged)


def _object_delta_payload(delta: ObjectDelta | dict[str, Any]) -> tuple[str | None, str | None, dict[str, Any]]:
    if isinstance(delta, ObjectDelta):
        return delta.object_id, delta.object_type, delta.changed_fields
    object_id = delta.get("object_id") or delta.get("objectId") or delta.get("ai2thor_object_id")
    object_type = delta.get("object_type") or delta.get("objectType")
    changed_fields = delta.get("changed_fields")
    return (
        str(object_id) if object_id not in (None, "") else None,
        str(object_type) if object_type not in (None, "") else None,
        changed_fields if isinstance(changed_fields, dict) else {},
    )


def _position_vector(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        try:
            return [float(value.get(axis, 0.0)) for axis in ("x", "y", "z")]
        except (TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            return None
    return None


def apply_object_deltas(
    nodes: list[dict[str, Any]],
    object_deltas: list[ObjectDelta | dict[str, Any]],
) -> list[dict[str, Any]]:
    """Patch action-observed properties without replacing graph-only fields."""

    updated = [deepcopy(node) for node in nodes if isinstance(node, dict)]
    nodes_by_object_id: dict[str, dict[str, Any]] = {}
    for node in updated:
        for key in ("ai2thor_object_id", "objectId", "object_id", "name"):
            value = node.get(key)
            if value not in (None, ""):
                nodes_by_object_id[str(value)] = node

    for delta in object_deltas or []:
        object_id, object_type, changed_fields = _object_delta_payload(delta)
        node = nodes_by_object_id.get(str(object_id)) if object_id is not None else None
        if node is None:
            continue
        if object_type:
            node.setdefault("object_tag", object_type)
            node.setdefault("possible_tags", [object_type])

        applied: dict[str, Any] = {}
        for field_name, change in changed_fields.items():
            after = change.get("after") if isinstance(change, dict) and "after" in change else change
            node[field_name] = deepcopy(after)
            applied[field_name] = deepcopy(after)
            if field_name == "position":
                center = _position_vector(after)
                if center is not None:
                    node["bbox_center"] = center
            elif field_name == "axisAlignedBoundingBox" and isinstance(after, dict):
                center = _position_vector(after.get("center"))
                extent = _position_vector(after.get("size"))
                if center is not None:
                    node["bbox_center"] = center
                if extent is not None:
                    node["bbox_extent"] = extent
        if applied:
            node["last_action_state_update"] = applied
            node["merge_source"] = "execution_delta"

    return canonicalize_scenegraph_nodes(updated)


def build_ai2thor_to_scenegraph_id_map(nodes: list[dict[str, Any]]) -> dict[str, int | str]:
    """Build an ``AI2-THOR objectId -> scene graph id`` mapping from node fields."""

    mapping: dict[str, int | str] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        scenegraph_id = node_scenegraph_id(node)
        if scenegraph_id is None:
            continue
        for key in ("ai2thor_object_id", "objectId", "object_id", "name"):
            value = node.get(key)
            if value not in (None, ""):
                mapping[str(value)] = scenegraph_id

        original_id = node.get("original_id")
        if isinstance(original_id, str) and "|" in original_id:
            mapping[original_id] = scenegraph_id
        raw_id = node.get("id")
        if isinstance(raw_id, str) and "|" in raw_id:
            mapping[raw_id] = scenegraph_id
    return mapping


def relation_key(relation: dict[str, Any]) -> tuple[str, str, str] | None:
    relation_name = str(relation.get("object_relation", "")).strip().lower()
    object1 = relation.get("object1") or {}
    object2 = relation.get("object2") or {}
    object1_id = canonical_object_id(object1.get("id"))
    object2_id = canonical_object_id(object2.get("id"))
    if object1_id is None or object2_id is None or not relation_name:
        return None
    return (str(object1_id), str(object2_id), relation_name)


def _map_object_id(object_id: Any, id_map: dict[str, Any] | None) -> Any:
    if id_map is None:
        return object_id
    return id_map.get(str(object_id), object_id)


def _delta_to_record(delta: RelationDelta | dict[str, Any], id_map: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(delta, RelationDelta):
        record = delta.to_record()
    else:
        object1 = delta.get("object1") or {}
        object2 = delta.get("object2") or {}
        record = {
            "object1": {"id": delta.get("object1_id", object1.get("id"))},
            "object2": {"id": delta.get("object2_id", object2.get("id"))},
            "object_relation": delta.get("object_relation"),
        }
        if delta.get("reason"):
            record["reason"] = delta.get("reason")

    record["object1"]["id"] = _map_object_id(record["object1"]["id"], id_map)
    record["object2"]["id"] = _map_object_id(record["object2"]["id"], id_map)
    record["object_relation"] = str(record["object_relation"]).strip().lower()
    return record


def _delta_op(delta: RelationDelta | dict[str, Any]) -> str:
    if isinstance(delta, RelationDelta):
        return delta.op
    return str(delta.get("op", "")).strip().lower()


def apply_relation_deltas(
    relations: list[dict[str, Any]],
    relation_deltas: list[RelationDelta | dict[str, Any]],
    *,
    id_map: dict[str, Any] | None = None,
    remove_existing_between_pair: bool = True,
) -> list[dict[str, Any]]:
    """Apply relation deltas to relation records.

    Only supported spatial relations are actively updated. Other relation
    records, such as VLM "none of these" explanations, are preserved.
    """

    relation_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    passthrough: list[dict[str, Any]] = []

    for relation in relations or []:
        if not isinstance(relation, dict):
            continue
        key = relation_key(relation)
        relation_name = str(relation.get("object_relation", "")).strip().lower()
        if key is None or relation_name not in SUPPORTED_RELATION_RECORDS:
            passthrough.append(deepcopy(relation))
            continue
        relation_index[key] = deepcopy(relation)

    for delta in relation_deltas or []:
        op = _delta_op(delta)
        record = _delta_to_record(delta, id_map=id_map)
        key = relation_key(record)
        relation_name = str(record.get("object_relation", "")).strip().lower()
        if key is None or relation_name not in SUPPORTED_RELATION_RECORDS:
            continue

        object1_id, object2_id, _ = key
        if op == "remove":
            relation_index.pop(key, None)
            continue
        if op == "add":
            if remove_existing_between_pair:
                stale_keys = [
                    existing_key
                    for existing_key in relation_index
                    if existing_key[0] == object1_id and existing_key[1] == object2_id
                ]
                for stale_key in stale_keys:
                    relation_index.pop(stale_key, None)
            relation_index[key] = record

    return passthrough + list(relation_index.values())


def update_scenegraph_files(
    *,
    stored_scenegraph_path: Path,
    stored_relations_path: Path,
    output_dir: Path,
    observed_scenegraph_path: Path | None = None,
    relation_deltas: list[RelationDelta | dict[str, Any]] | None = None,
    object_deltas: list[ObjectDelta | dict[str, Any]] | None = None,
    id_map: dict[str, Any] | None = None,
    append_observed_only: bool = False,
) -> dict[str, Any]:
    """Merge scene graph files and write an updated cache directory."""

    stored_nodes = load_json(stored_scenegraph_path) if stored_scenegraph_path.exists() else []
    stored_relations = load_json(stored_relations_path) if stored_relations_path.exists() else []
    observed_nodes = []
    if observed_scenegraph_path is not None and observed_scenegraph_path.exists():
        observed_nodes = load_json(observed_scenegraph_path)

    if append_observed_only:
        merged_nodes = append_new_scenegraph_nodes(stored_nodes, observed_nodes)
    else:
        merged_nodes = merge_scenegraph_nodes(stored_nodes, observed_nodes)
    merged_nodes = apply_object_deltas(merged_nodes, object_deltas or [])
    inferred_id_map = build_ai2thor_to_scenegraph_id_map(merged_nodes)
    if id_map:
        inferred_id_map.update(id_map)

    merged_relations = apply_relation_deltas(
        stored_relations,
        relation_deltas or [],
        id_map=inferred_id_map,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    scenegraph_path = output_dir / "scene_graph.json"
    relations_path = output_dir / "cfslam_object_relations.json"
    metadata_path = output_dir / "graph_update_metadata.json"
    save_json(merged_nodes, scenegraph_path)
    save_json(merged_relations, relations_path)
    save_json(
        {
            "scene_graph": str(scenegraph_path),
            "relations": str(relations_path),
            "source_scene_graph": str(stored_scenegraph_path),
            "source_relations": str(stored_relations_path),
            "observed_scene_graph": str(observed_scenegraph_path) if observed_scenegraph_path else None,
            "num_nodes_before": len(stored_nodes),
            "num_nodes_after": len(merged_nodes),
            "num_relations_before": len(stored_relations),
            "num_relations_after": len(merged_relations),
            "num_relation_deltas": len(relation_deltas or []),
            "num_object_deltas": len(object_deltas or []),
        },
        metadata_path,
    )
    return {
        "cachedir": str(output_dir),
        "scene_graph": str(scenegraph_path),
        "relations": str(relations_path),
        "object_relations": str(relations_path),
        "metadata": str(metadata_path),
    }
