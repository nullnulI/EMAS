"""Infer scene-graph edge updates from AI2-THOR execution results.

AI2-THOR does not expose a high-level "skill result" API. The reliable signal is
the before/after event metadata plus the low-level action trace. This module
turns that metadata into local relation deltas such as "Apple on CounterTop".
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .schemas import ObjectDelta, RelationDelta

DEFAULT_DIFF_FIELDS = (
    "position",
    "rotation",
    "parentReceptacles",
    "receptacleObjectIds",
    "isPickedUp",
    "isOpen",
    "openness",
    "isToggled",
    "isBroken",
    "isDirty",
    "isFilledWithLiquid",
    "fillLiquid",
    "isCooked",
    "isSliced",
    "isMoving",
)

SUPPORTED_RELATION_RECORDS = {"a on b", "b on a", "a in b", "b in a"}

CONTAINER_TYPES = {
    "Bowl",
    "Box",
    "Cabinet",
    "CoffeeMachine",
    "Cup",
    "Drawer",
    "Fridge",
    "GarbageCan",
    "Kettle",
    "Microwave",
    "Mug",
    "Pan",
    "Plate",
    "Pot",
    "Safe",
    "Sink",
    "SinkBasin",
    "Toaster",
}

SURFACE_TYPES = {
    "Bathtub",
    "Bed",
    "Bench",
    "Chair",
    "CoffeeTable",
    "CounterTop",
    "Desk",
    "DiningTable",
    "Dresser",
    "Floor",
    "Ottoman",
    "Shelf",
    "SideTable",
    "Sofa",
    "Stool",
    "TVStand",
}


def object_identifier(obj: dict[str, Any]) -> str | None:
    """Return the most stable AI2-THOR identifier available for an object."""

    for key in ("objectId", "name", "id"):
        value = obj.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def object_type_from_id(object_id: str | None) -> str | None:
    if not object_id:
        return None
    return object_id.split("|", 1)[0]


def index_objects(objects_or_metadata: Any) -> dict[str, dict[str, Any]]:
    """Index AI2-THOR metadata objects by ``objectId``.

    The input can be either a raw ``metadata["objects"]`` list or the full
    metadata dict returned by an event.
    """

    if isinstance(objects_or_metadata, dict):
        objects = objects_or_metadata.get("objects") or []
    else:
        objects = objects_or_metadata or []

    indexed: dict[str, dict[str, Any]] = {}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        object_id = object_identifier(obj)
        if object_id:
            indexed[object_id] = obj
    return indexed


def _normalize_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        try:
            return abs(float(left) - float(right)) < 1e-4
        except (TypeError, ValueError):
            return left == right
    if isinstance(left, dict) and isinstance(right, dict):
        keys = set(left) | set(right)
        return all(_same_value(left.get(key), right.get(key)) for key in keys)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return _normalize_list(left) == _normalize_list(right)
    return left == right


def diff_objects(
    pre_objects_or_metadata: Any,
    post_objects_or_metadata: Any,
    *,
    fields: Iterable[str] = DEFAULT_DIFF_FIELDS,
) -> list[ObjectDelta]:
    """Compare AI2-THOR objects before and after execution."""

    pre_index = index_objects(pre_objects_or_metadata)
    post_index = index_objects(post_objects_or_metadata)
    field_names = tuple(fields)
    deltas: list[ObjectDelta] = []

    for object_id in sorted(set(pre_index) | set(post_index)):
        before = pre_index.get(object_id, {})
        after = post_index.get(object_id, {})
        changed_fields: dict[str, dict[str, Any]] = {}
        for field_name in field_names:
            before_value = before.get(field_name)
            after_value = after.get(field_name)
            if not _same_value(before_value, after_value):
                changed_fields[field_name] = {"before": before_value, "after": after_value}
        if changed_fields:
            object_type = after.get("objectType") or before.get("objectType") or object_type_from_id(object_id)
            deltas.append(ObjectDelta(object_id=object_id, object_type=object_type, changed_fields=changed_fields))
    return deltas


def _primary_parent_receptacle(obj: dict[str, Any]) -> str | None:
    parents = [str(parent) for parent in _normalize_list(obj.get("parentReceptacles")) if parent not in (None, "")]
    non_floor = [parent for parent in parents if (object_type_from_id(parent) or "").lower() != "floor"]
    return non_floor[0] if non_floor else (parents[0] if parents else None)


def _relation_for_parent(child: dict[str, Any], parent: dict[str, Any] | None, parent_id: str) -> str:
    parent_type = None
    if parent is not None:
        parent_type = parent.get("objectType") or object_type_from_id(parent_id)
    parent_type = parent_type or object_type_from_id(parent_id)

    if parent_type in CONTAINER_TYPES:
        return "a in b"
    if parent_type in SURFACE_TYPES:
        return "a on b"
    if parent and child.get("objectId") in _normalize_list(parent.get("receptacleObjectIds")):
        return "a in b"
    return "a on b"


def _parent_relation_records(objects_by_id: dict[str, dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for child_id, child in objects_by_id.items():
        if child.get("isPickedUp"):
            continue
        parent_id = _primary_parent_receptacle(child)
        if not parent_id:
            continue
        parent = objects_by_id.get(parent_id)
        relation_name = _relation_for_parent(child, parent, parent_id)
        key = (child_id, parent_id, relation_name)
        records[key] = {
            "object1": {"id": child_id},
            "object2": {"id": parent_id},
            "object_relation": relation_name,
        }
    return records


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            yield from _walk_strings(child)


def extract_focus_object_ids(execution: Any) -> set[str]:
    """Best-effort extraction of AI2-THOR object ids mentioned by a trace."""

    focus = set()
    for value in _walk_strings(execution):
        if "|" in value:
            focus.add(value)
    return focus


def _successful_execution(execution: Any) -> bool:
    """Return whether the trace contains at least one successful low-level action.

    If no action result fields are present, keep the metadata diff usable and
    return True.
    """

    action_results: list[bool] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if "lastActionSuccess" in value:
                action_results.append(bool(value.get("lastActionSuccess")))
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(execution)
    return any(action_results) if action_results else True


def infer_relation_deltas_from_metadata(
    pre_metadata: Any,
    post_metadata: Any,
    *,
    execution: Any | None = None,
    focus_object_ids: Iterable[str] | None = None,
) -> list[RelationDelta]:
    """Infer add/remove relation deltas from pre/post AI2-THOR metadata.

    The relation format matches ``cfslam_object_relations.json``:
    ``object1`` is the moved/contained object, ``object2`` is the support or
    container, and ``object_relation`` is currently ``a on b`` or ``a in b``.
    """

    if execution is not None and not _successful_execution(execution):
        return []

    pre_index = index_objects(pre_metadata)
    post_index = index_objects(post_metadata)
    pre_records = _parent_relation_records(pre_index)
    post_records = _parent_relation_records(post_index)

    changed_object_ids = {delta.object_id for delta in diff_objects(pre_index, post_index)}
    focus_ids = set(str(object_id) for object_id in (focus_object_ids or []))
    if execution is not None:
        focus_ids |= extract_focus_object_ids(execution)
    active_ids = changed_object_ids | focus_ids

    deltas: list[RelationDelta] = []
    for key, record in pre_records.items():
        child_id, parent_id, relation_name = key
        if active_ids and child_id not in active_ids and parent_id not in active_ids:
            continue
        if key not in post_records:
            deltas.append(
                RelationDelta(
                    op="remove",
                    object1_id=child_id,
                    object2_id=parent_id,
                    object_relation=relation_name,
                    reason="relation disappeared in post-execution metadata",
                )
            )

    for key, record in post_records.items():
        child_id, parent_id, relation_name = key
        if active_ids and child_id not in active_ids and parent_id not in active_ids:
            continue
        if key not in pre_records:
            deltas.append(
                RelationDelta(
                    op="add",
                    object1_id=child_id,
                    object2_id=parent_id,
                    object_relation=relation_name,
                    reason="relation appeared in post-execution metadata",
                )
            )
    return deltas


def infer_relations_for_new_objects(
    metadata: Any,
    new_object_ids: Iterable[str],
    *,
    require_existing_parent: bool = True,
) -> list[RelationDelta]:
    """Infer current relations whose child is a newly observed object.

    When ``require_existing_parent`` is true, relations between two new
    objects are excluded, leaving only ``new object -> existing receptacle``.
    """

    new_ids = {str(object_id) for object_id in new_object_ids}
    records = _parent_relation_records(index_objects(metadata))
    deltas: list[RelationDelta] = []
    for child_id, parent_id, relation_name in records:
        if child_id not in new_ids:
            continue
        if require_existing_parent and parent_id in new_ids:
            continue
        deltas.append(
            RelationDelta(
                op="add",
                object1_id=child_id,
                object2_id=parent_id,
                object_relation=relation_name,
                reason="relation inferred for newly observed object",
            )
        )
    return deltas
