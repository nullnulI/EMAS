"""Validate planner selectors and expand them against an episode scene catalogue.

This module deliberately contains no benchmark task phrases or semantic object groups.
The planning model chooses concrete object types from the provided scene catalogue;
this module only validates action affordances and binds stable object IDs.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

from action_contracts import (
    ACTION_CONTRACTS,
    AFFORDANCE_FIELDS,
    action_role_affordances,
    normalize_logical_action,
)
from placement_contracts import compatible_receptacles, placement_compatibility


VALID_QUANTIFIERS = frozenset({"one", "all"})
ACTION_ROLE_AFFORDANCES = action_role_affordances()


def _type_key(value: Any) -> str:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(value or "").replace("_", " "))
    return "".join(re.findall(r"[a-z0-9]+", separated.lower()))


def _catalogue(objects: Iterable[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in objects or []:
        if not isinstance(item, Mapping):
            continue
        object_type = str(item.get("objectType") or item.get("type") or "").strip()
        object_id = str(item.get("objectId") or item.get("id") or "").strip()
        if not object_type or not object_id:
            continue
        record: dict[str, Any] = {"objectType": object_type, "objectId": object_id}
        position = item.get("position")
        record["position"] = deepcopy(position) if isinstance(position, Mapping) else {}
        for field in AFFORDANCE_FIELDS:
            record[field] = bool(item.get(field, False))
        for field in ("parentReceptacles", "receptacleObjectIds"):
            raw_values = item.get(field)
            if isinstance(raw_values, list):
                record[field] = [
                    str(value).strip()
                    for value in raw_values
                    if str(value).strip()
                ]
            else:
                record[field] = []
        result.append(record)
    return sorted(result, key=lambda item: (item["objectType"], item["objectId"]))


def summarize_scene_catalogue(objects: Iterable[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """Return type, affordance, and live containment facts for the planner prompt."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    catalogue = _catalogue(objects)
    by_id = {item["objectId"]: item for item in catalogue}

    def coordinate(position: Mapping[str, Any], axis: str) -> float | None:
        try:
            return round(float(position.get(axis)), 3)
        except (TypeError, ValueError):
            return None

    central_parent_locations: set[tuple[str, float, float]] = set()
    parent_instances_by_type: dict[str, list[tuple[float, float]]] = {}
    for item in catalogue:
        position = item.get("position") if isinstance(item.get("position"), Mapping) else {}
        x, z = coordinate(position, "x"), coordinate(position, "z")
        if x is not None and z is not None:
            parent_instances_by_type.setdefault(item["objectType"], []).append((x, z))
    for parent_type, positions in parent_instances_by_type.items():
        if len(positions) < 2:
            continue
        centre_x = sum(x for x, _ in positions) / len(positions)
        centre_z = sum(z for _, z in positions) / len(positions)
        central_x, central_z = min(
            positions,
            key=lambda value: (value[0] - centre_x) ** 2 + (value[1] - centre_z) ** 2,
        )
        central_parent_locations.add((parent_type, central_x, central_z))
    for item in catalogue:
        grouped.setdefault(item["objectType"], []).append(item)
    summary: list[dict[str, Any]] = []
    for object_type in sorted(grouped):
        instances = grouped[object_type]
        record = {
            "object_type": object_type,
            "count": len(instances),
            "affordances": {
                field: all(bool(item.get(field)) for item in instances)
                for field in AFFORDANCE_FIELDS
            },
        }
        parent_locations: dict[tuple[str, float | None, float | None, float | None], int] = {}
        for instance in instances:
            for parent_id in instance.get("parentReceptacles") or []:
                parent = by_id.get(parent_id) or {}
                parent_type = str(
                    parent.get("objectType") or str(parent_id).split("|", 1)[0]
                ).strip()
                position = parent.get("position") if isinstance(parent.get("position"), Mapping) else {}

                key = (
                    parent_type,
                    coordinate(position, "x"),
                    coordinate(position, "y"),
                    coordinate(position, "z"),
                )
                parent_locations[key] = parent_locations.get(key, 0) + 1
        if parent_locations:
            record["parent_locations"] = [
                {
                    "parent_type": parent_type,
                    "parent_position": {
                        axis: value
                        for axis, value in zip(("x", "y", "z"), (x, y, z))
                        if value is not None
                    },
                    "instance_count": count,
                    **(
                        {"relative_location": "central"}
                        if x is not None and z is not None
                        and (parent_type, x, z) in central_parent_locations
                        else {}
                    ),
                }
                for (parent_type, x, y, z), count in sorted(
                    parent_locations.items(),
                    key=lambda item: (
                        item[0][0],
                        float("inf") if item[0][1] is None else item[0][1],
                        float("inf") if item[0][3] is None else item[0][3],
                    ),
                )
            ]
        summary.append(record)
    return summary


def _selector(grounding: Mapping[str, Any], role: str) -> dict[str, Any] | None:
    value = grounding.get(f"{role}_selector")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        return {"quantifier": "", "object_types": []}
    raw_types = value.get("object_types") or []
    if isinstance(raw_types, str):
        raw_types = [raw_types]
    object_types = []
    for item in raw_types if isinstance(raw_types, list) else []:
        text = str(item).strip()
        if text and text not in object_types:
            object_types.append(text)
    return {
        "quantifier": str(value.get("quantifier") or "").strip().lower(),
        "object_types": object_types,
    }


def _resolve_types(
    requested: list[str], catalogue: list[dict[str, Any]],
) -> tuple[list[str], str | None]:
    by_key: dict[str, list[str]] = {}
    for object_type in sorted({item["objectType"] for item in catalogue}):
        by_key.setdefault(_type_key(object_type), []).append(object_type)
    resolved: list[str] = []
    for value in requested:
        candidates = by_key.get(_type_key(value), [])
        if not candidates:
            return [], f"object type {value!r} is absent from the scene catalogue"
        if len(candidates) != 1:
            return [], f"object type {value!r} is ambiguous: {candidates}"
        if candidates[0] not in resolved:
            resolved.append(candidates[0])
    return resolved, None


def _validate_role(
    selector: dict[str, Any] | None,
    role: str,
    affordance: str | tuple[str, ...],
    catalogue: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    if selector is None:
        return None, f"missing {role}_selector"
    quantifier = selector.get("quantifier")
    if quantifier not in VALID_QUANTIFIERS:
        return None, f"invalid {role} quantifier {quantifier!r}"
    requested_types = selector.get("object_types") or []
    if not requested_types:
        return None, f"missing {role} object_types"
    if role == "destination" and len(requested_types) != 1:
        return None, (
            "destination_selector must contain exactly one object_type; "
            f"got {requested_types}"
        )
    resolved_types, error = _resolve_types(requested_types, catalogue)
    if error:
        return None, error
    selected = [item for item in catalogue if item["objectType"] in resolved_types]
    if not selected:
        return None, f"no scene instances match {role} types {resolved_types}"
    affordances = (affordance,) if isinstance(affordance, str) else tuple(affordance)
    invalid = [
        item["objectId"]
        for item in selected
        if not any(bool(item.get(field)) for field in affordances)
    ]
    if invalid:
        requirement = affordances[0] if len(affordances) == 1 else f"any of {list(affordances)}"
        return None, f"{role} instances do not satisfy {requirement}: {invalid}"
    return {
        "quantifier": quantifier,
        "object_types": resolved_types,
        "instances": selected,
        "affordances": list(affordances),
    }, None


def _generated_id(base: str, index: int, existing: set[str], reserved: set[str]) -> str:
    if index == 0:
        existing.add(base)
        return base
    suffix = index + 1
    while True:
        candidate = f"{base}__instance_{suffix:03d}"
        if candidate not in existing and candidate not in reserved:
            existing.add(candidate)
            return candidate
        suffix += 1


def _bind_role(
    grounding: dict[str, Any], role: str, validated: dict[str, Any], instance: dict[str, Any] | None,
) -> None:
    types = list(validated["object_types"])
    grounding[f"{role}_selector"] = {
        "quantifier": validated["quantifier"], "object_types": types,
    }
    grounding[f"{role}_object_tags"] = [instance["objectType"]] if instance else types
    grounding[f"{role}_object_ids"] = [instance["objectId"]] if instance else []


def _place_relation_already_satisfied(
    source_instance: dict[str, Any] | None,
    destination_instance: dict[str, Any] | None,
) -> bool:
    if not source_instance or not destination_instance:
        return False
    source_id = str(source_instance.get("objectId") or "").strip()
    destination_id = str(destination_instance.get("objectId") or "").strip()
    if not source_id or not destination_id:
        return False
    parent_ids = {
        str(value).strip()
        for value in source_instance.get("parentReceptacles") or []
        if str(value).strip()
    }
    receptacle_child_ids = {
        str(value).strip()
        for value in destination_instance.get("receptacleObjectIds") or []
        if str(value).strip()
    }
    return destination_id in parent_ids or source_id in receptacle_child_ids


def _mark_already_satisfied_place(
    clone: dict[str, Any],
    grounding: dict[str, Any],
    source_instance: dict[str, Any],
    destination_instance: dict[str, Any],
) -> None:
    source_id = str(source_instance.get("objectId") or "").strip()
    destination_id = str(destination_instance.get("objectId") or "").strip()
    grounding["status"] = "satisfied"
    runtime = deepcopy(clone.get("runtime") if isinstance(clone.get("runtime"), Mapping) else {})
    runtime.update({
        "already_satisfied": True,
        "already_satisfied_reason": (
            "source object is already contained by the destination receptacle"
        ),
        "satisfied_relation": {
            "source_object_id": source_id,
            "destination_object_id": destination_id,
            "relation": "contained_by",
        },
    })
    clone["runtime"] = runtime


def _fallback(
    original: dict[str, Any],
    status: str,
    reason: str,
    events: list[dict[str, Any]],
    *,
    violations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output = deepcopy(original)
    diagnostics = output.setdefault("planner_diagnostics", {})
    diagnostics["selector_expansion"] = {
        "status": status, "reason": reason, "events": events,
    }
    if violations:
        diagnostics["selector_expansion"]["violations"] = deepcopy(violations)
    return output


def expand_scene_catalog_task_graph(
    planner_task_graph: Mapping[str, Any],
    object_catalog: Iterable[Mapping[str, Any]] | None,
    *,
    authoritative_destination_types: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Expand valid planner selectors without interpreting the user instruction."""

    original = deepcopy(dict(planner_task_graph))
    if authoritative_destination_types is None:
        diagnostics = original.get("planner_diagnostics")
        root_validation = (
            diagnostics.get("root_intent_validation")
            if isinstance(diagnostics, Mapping)
            else {}
        )
        root_intent = (
            root_validation.get("root_intent")
            if isinstance(root_validation, Mapping)
            else {}
        )
        roles = root_intent.get("roles") if isinstance(root_intent, Mapping) else {}
        authoritative_destination_types = (
            roles.get("destination") if isinstance(roles, Mapping) else []
        )
    authoritative_destination_keys = {
        _type_key(value)
        for value in authoritative_destination_types or []
        if str(value).strip()
    }
    catalogue = _catalogue(object_catalog)
    tasks = [deepcopy(item) for item in original.get("flat_tasks") or [] if isinstance(item, Mapping)]
    if not tasks:
        return _fallback(original, "skipped", "planner graph has no tasks", [])
    selector_tasks = [
        task for task in tasks
        if isinstance(task.get("grounding"), Mapping)
        and any(f"{role}_selector" in task["grounding"] for role in ("source", "destination"))
    ]
    if not selector_tasks:
        return _fallback(original, "not_applicable", "planner emitted no selectors", [])
    required_interaction_selector = any(
        str(task.get("action") or "").strip().lower() in ACTION_ROLE_AFFORDANCES
        for task in selector_tasks
    )
    if not catalogue:
        status = "rejected" if required_interaction_selector else "skipped"
        return _fallback(original, status, "scene catalogue is unavailable", [])

    reserved_ids = {str(task.get("id")) for task in tasks}
    generated_ids: set[str] = set()
    replacements: dict[str, list[str]] = {}
    expanded: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    interaction_selector_types: set[str] = set()
    discovery_selector_types: dict[str, list[str]] = {}

    for task in tasks:
        task_id = str(task.get("id") or "")
        action = normalize_logical_action(task.get("action"))
        grounding = deepcopy(task.get("grounding") if isinstance(task.get("grounding"), Mapping) else {})
        requirements = ACTION_ROLE_AFFORDANCES.get(action)
        has_selector = any(f"{role}_selector" in grounding for role in ("source", "destination"))
        if not has_selector:
            expanded.append(task)
            continue
        if requirements is None:
            if action in {"find", "inspect", "navigate"}:
                resolved_discovery_types: list[str] = []
                for role in ("source", "destination"):
                    selector = _selector(grounding, role)
                    if not selector or not selector.get("object_types"):
                        continue
                    resolved, error = _resolve_types(selector["object_types"], catalogue)
                    if error is None:
                        resolved_discovery_types.extend(resolved)
                discovery_selector_types[task_id] = list(dict.fromkeys(resolved_discovery_types))
                grounding.pop("source_selector", None)
                grounding.pop("destination_selector", None)
                task["grounding"] = grounding
                expanded.append(task)
                events.append({
                    "task_id": task_id, "action": action,
                    "status": "ignored_non_interaction_selector",
                    "selector_types": discovery_selector_types[task_id],
                })
                continue
            return _fallback(original, "rejected", f"task {task_id}: action {action!r} has no selector affordance contract", events)
        for role in ("source", "destination"):
            if role in requirements:
                continue
            selector = _selector(grounding, role)
            if selector and selector.get("object_types"):
                return _fallback(
                    original,
                    "rejected",
                    f"task {task_id}: {role}_selector is not supported for action {action!r}",
                    events,
                )
        validated: dict[str, dict[str, Any]] = {}
        for role, affordance in requirements.items():
            value, error = _validate_role(_selector(grounding, role), role, affordance, catalogue)
            if error:
                return _fallback(original, "rejected", f"task {task_id}: {error}", events)
            assert value is not None
            validated[role] = value
            interaction_selector_types.update(value["object_types"])

        contract = ACTION_CONTRACTS.get(action) or {}
        source = validated.get("source")
        required_source_quantifier = contract.get("source_quantifier")
        if source is not None and required_source_quantifier and source["quantifier"] != required_source_quantifier:
            return _fallback(
                original,
                "rejected",
                f"task {task_id}: source quantifier must be {required_source_quantifier!r} for action {action!r}",
                events,
            )

        destination = validated.get("destination")
        if action == "place" and source is not None and destination is not None:
            unsupported_pairs = [
                (source_type, destination_type)
                for source_type in source["object_types"]
                for destination_type in destination["object_types"]
                if placement_compatibility(source_type, destination_type) is False
            ]
            authoritative_pairs = [
                (source_type, destination_type)
                for source_type, destination_type in unsupported_pairs
                if _type_key(destination_type) in authoritative_destination_keys
            ]
            if authoritative_pairs:
                events.append({
                    "task_id": task_id,
                    "action": action,
                    "status": "authoritative_destination_override",
                    "pairs": [
                        {"source_type": source_type, "destination_type": destination_type}
                        for source_type, destination_type in authoritative_pairs
                    ],
                })
            incompatible_pairs = [
                (source_type, destination_type)
                for source_type, destination_type in unsupported_pairs
                if _type_key(destination_type) not in authoritative_destination_keys
            ]
            if incompatible_pairs:
                pair_text = ", ".join(
                    f"{source_type} -> {destination_type}"
                    for source_type, destination_type in incompatible_pairs
                )
                invalid_values = [
                    {
                        "source_type": source_type,
                        "destination_type": destination_type,
                        "compatible_receptacles": list(
                            compatible_receptacles(source_type) or ()
                        ),
                    }
                    for source_type, destination_type in incompatible_pairs
                ]
                reason = (
                    f"task {task_id}: placement pair is not supported by the "
                    f"AI2-THOR controller: {pair_text}"
                )
                return _fallback(
                    original,
                    "rejected",
                    reason,
                    events,
                    violations=[{
                        "stage": "selector_expansion",
                        "task_id": task_id,
                        "code": "incompatible_receptacle",
                        "field": "grounding.destination_selector",
                        "message": reason,
                        "invalid_values": invalid_values,
                        "required_fix": (
                            "Replace the destination selector with a receptacle type "
                            "that is both semantically appropriate and compatible with "
                            "every selected source type."
                        ),
                    }],
                )

        source_instances = list(source["instances"]) if source and source["quantifier"] == "all" else [None]
        destination = validated.get("destination")
        destination_instance = None
        if destination is not None and destination["quantifier"] == "all":
            return _fallback(original, "rejected", f"task {task_id}: destination quantifier 'all' is not executable", events)
        if destination is not None and len(destination["instances"]) == 1:
            destination_instance = destination["instances"][0]
        source_one_instance = None
        if source is not None and source["quantifier"] == "one" and len(source["instances"]) == 1:
            source_one_instance = source["instances"][0]

        generated: list[str] = []
        previous_id: str | None = None
        dependency_origins = [str(value) for value in task.get("depends_on") or []]
        destination_is_openable = bool(
            destination and destination["instances"]
            and all(bool(item.get("openable")) for item in destination["instances"])
        )
        for index, source_instance in enumerate(source_instances):
            clone = deepcopy(task)
            clone_id = _generated_id(task_id, index, generated_ids, reserved_ids)
            clone["id"] = clone_id
            clone_grounding = deepcopy(grounding)
            if source is not None:
                _bind_role(clone_grounding, "source", source, source_instance or source_one_instance)
            if destination is not None:
                _bind_role(clone_grounding, "destination", destination, destination_instance)
            source_tags = list(clone_grounding.get("source_object_tags") or [])
            destination_tags = list(clone_grounding.get("destination_object_tags") or [])
            clone_grounding["object_tags"] = list(dict.fromkeys([*source_tags, *destination_tags]))
            clone_grounding["status"] = "grounded" if (
                clone_grounding.get("source_object_ids")
                and (destination is None or clone_grounding.get("destination_object_ids"))
            ) else "partial"
            clone["grounding"] = clone_grounding
            bound_source = source_instance or source_one_instance
            bound_source_object_id = (
                str(bound_source.get("objectId") or "").strip()
                if isinstance(bound_source, Mapping)
                else ""
            )
            serialized_after_task_id = (
                previous_id
                if action in {"place", "put"}
                and destination_is_openable
                and previous_id is not None
                else None
            )
            clone_runtime = deepcopy(
                clone.get("runtime")
                if isinstance(clone.get("runtime"), Mapping)
                else {}
            )
            clone_runtime["selector_expansion"] = {
                "origin_task_id": task_id,
                "instance_index": index + 1,
                "bound_source_object_id": bound_source_object_id or None,
                "expanded_dependency_origins": list(dependency_origins),
                "serialized_after_task_id": serialized_after_task_id,
            }
            clone["runtime"] = clone_runtime
            dependencies = list(dependency_origins)
            if action in {"place", "put"} and destination_is_openable and previous_id is not None:
                dependencies = [previous_id]
            clone["depends_on"] = list(dict.fromkeys(dependencies))
            if source_instance is not None:
                clone["name"] = f"{action} {source_instance['objectType']}"
                clone["description"] = f"Execute {action} for the specific object {source_instance['objectId']}."
            if (
                action in {"place", "put"}
                and source_instance is not None
                and destination_instance is not None
                and _place_relation_already_satisfied(source_instance, destination_instance)
            ):
                _mark_already_satisfied_place(
                    clone,
                    clone_grounding,
                    source_instance,
                    destination_instance,
                )
            expanded.append(clone)
            generated.append(clone_id)
            previous_id = clone_id
        replacements[task_id] = generated
        generated_set = set(generated)
        already_satisfied = [
            str(item.get("id"))
            for item in expanded
            if str(item.get("id")) in generated_set
            and bool((item.get("runtime") or {}).get("already_satisfied"))
        ]
        events.append({
            "task_id": task_id,
            "action": action,
            "generated_task_ids": generated,
            "already_satisfied_task_ids": already_satisfied,
            "source_types": list(source["object_types"]) if source else [],
            "source_instance_ids": [item["objectId"] for item in source_instances if item is not None],
            "destination_types": list(destination["object_types"]) if destination else [],
            "destination_object_id": destination_instance["objectId"] if destination_instance else None,
        })

    elided_ids = {
        task_id
        for task_id, selector_types in discovery_selector_types.items()
        if selector_types and set(selector_types).issubset(interaction_selector_types)
    }
    if elided_ids:
        task_by_id = {str(task.get("id") or ""): task for task in expanded}

        def inherited_dependencies(dependency_id: str, visiting: set[str]) -> list[str]:
            if dependency_id not in elided_ids:
                return [dependency_id]
            if dependency_id in visiting:
                return []
            elided_task = task_by_id.get(dependency_id)
            if elided_task is None:
                return []
            inherited: list[str] = []
            for upstream in elided_task.get("depends_on") or []:
                inherited.extend(inherited_dependencies(str(upstream), {*visiting, dependency_id}))
            return inherited

        for task in expanded:
            dependencies: list[str] = []
            for dependency in task.get("depends_on") or []:
                dependencies.extend(inherited_dependencies(str(dependency), set()))
            task["depends_on"] = list(dict.fromkeys(dependencies))
        expanded = [
            task for task in expanded if str(task.get("id") or "") not in elided_ids
        ]
        for event in events:
            if event.get("task_id") in elided_ids:
                event["status"] = "elided_selector_bound_discovery"

    for task in expanded:
        rewritten: list[str] = []
        task_id = str(task.get("id") or "")
        for dependency in task.get("depends_on") or []:
            dependency_id = str(dependency)
            replacement_ids = replacements.get(dependency_id)
            if replacement_ids and task_id not in replacement_ids:
                rewritten.extend(replacement_ids)
            else:
                rewritten.append(dependency_id)
        task["depends_on"] = list(dict.fromkeys(value for value in rewritten if value != task_id))

    output = deepcopy(original)
    output["flat_tasks"] = expanded
    output.setdefault("planner_diagnostics", {})["selector_expansion"] = {
        "status": "expanded",
        "reason": "validated planner selectors against scene affordances",
        "events": events,
        "generated_task_count": len(expanded),
    }
    return output
