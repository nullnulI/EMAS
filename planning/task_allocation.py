"""
Resource-aware execution planning for an immutable semantic Task Graph.

The allocator consumes the immutable task graph produced by
``planning.task_graph`` and returns a separate, validated Execution Plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

EMAS_ROOT = Path(__file__).resolve().parents[1]
CONCEPTGRAPH_REPO_ROOT = EMAS_ROOT / "memory" / "concept-graphs"
for path in (EMAS_ROOT, CONCEPTGRAPH_REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from planning.model_config import DEFAULT_PLANNING_MODEL_PATH
from action_contracts import normalize_logical_action

try:
    from planning.task_graph import parse_json_from_text
except Exception:
    parse_json_from_text = None


__all__ = [
    "TaskAllocationError",
    "build_effective_execution_graph",
    "validate_execution_plan",
    "build_execution_plan",
    "first_execution_unit",
    "build_argparser",
    "main",
]


class TaskAllocationError(RuntimeError):
    """Raised only when the execution-planning model infrastructure fails."""

    code = "task_allocation_failed"

    def __init__(
        self,
        message: str,
        *,
        diagnostics: dict[str, Any],
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code or type(self).code)
        self.diagnostics = deepcopy(diagnostics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "diagnostics": deepcopy(self.diagnostics),
        }


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _compact_vector(value: Any) -> dict | None:
    if not isinstance(value, dict):
        return None
    compact = {}
    for key in ("x", "y", "z"):
        if key in value:
            compact[key] = value[key]
    return compact or None


def _visible_objects(metadata: dict, limit: int = 12) -> list[dict]:
    objects = []
    for obj in metadata.get("objects") or []:
        if not isinstance(obj, dict) or not obj.get("visible"):
            continue
        objects.append(
            {
                "objectId": obj.get("objectId"),
                "objectType": obj.get("objectType"),
                "name": obj.get("name"),
                "distance": obj.get("distance"),
                "position": _compact_vector(obj.get("position")),
                "pickupable": obj.get("pickupable"),
                "openable": obj.get("openable"),
                "isOpen": obj.get("isOpen"),
            }
        )
        if len(objects) >= limit:
            break
    return objects


def _held_objects(metadata: dict) -> list[dict]:
    inventory = metadata.get("inventoryObjects") or metadata.get("inventory") or []
    held = []
    for obj in inventory:
        if isinstance(obj, dict):
            held.append(
                {
                    "objectId": obj.get("objectId"),
                    "objectType": obj.get("objectType"),
                    "name": obj.get("name"),
                }
            )
        else:
            held.append({"objectId": str(obj)})
    return held


def _is_compact_agent_state(metadata: dict) -> bool:
    return "agent_id" in metadata and (
        "visible_objects" in metadata
        or "inventoryObjects" in metadata
        or "inventory" in metadata
        or "held_objects" in metadata
    )


def _summarize_compact_agent_state(metadata: dict, fallback_agent_id: str | int = 0) -> dict:
    summary = deepcopy(metadata)
    summary["agent_id"] = str(summary.get("agent_id", fallback_agent_id))

    nested_metadata = summary.get("metadata") if isinstance(summary.get("metadata"), dict) else {}
    agent = summary.get("agent") or nested_metadata.get("agent") or {}
    if agent:
        summary.setdefault("position", _compact_vector(agent.get("position")))
        summary.setdefault("rotation", _compact_vector(agent.get("rotation")))
        summary.setdefault("cameraHorizon", agent.get("cameraHorizon"))
        summary.setdefault("isStanding", agent.get("isStanding"))

    if not summary.get("inventoryObjects") and nested_metadata:
        summary["inventoryObjects"] = nested_metadata.get("inventoryObjects") or nested_metadata.get("inventory") or []
    summary["held_objects"] = summary.get("held_objects") or _held_objects(summary)

    if not summary.get("visible_objects") and nested_metadata:
        summary["visible_objects"] = _visible_objects(nested_metadata)
    else:
        summary.setdefault("visible_objects", [])

    return summary


def summarize_agent_metadata(metadata: dict, agent_id: str | int = 0) -> dict:
    """Build a small, LLM-friendly summary from an AI2-THOR metadata dict."""

    if _is_compact_agent_state(metadata):
        return _summarize_compact_agent_state(metadata, agent_id)

    agent = metadata.get("agent") or {}
    return {
        "agent_id": str(agent.get("agentId", metadata.get("agentId", agent_id))),
        "agent": agent,
        "position": _compact_vector(agent.get("position")),
        "rotation": _compact_vector(agent.get("rotation")),
        "cameraHorizon": agent.get("cameraHorizon"),
        "isStanding": agent.get("isStanding"),
        "lastAction": metadata.get("lastAction"),
        "lastActionSuccess": metadata.get("lastActionSuccess"),
        "held_objects": _held_objects(metadata),
        "visible_objects": _visible_objects(metadata),
    }


def extract_agent_metadata(source: Any) -> list[dict]:
    """
    Extract agent metadata from common AI2-THOR objects.

    Supported inputs:
    - Controller-like object with ``last_event``
    - Event-like object with ``metadata`` or multi-agent ``events``
    - raw metadata dict with ``agent`` or ``agents``
    - a list/tuple of metadata dicts
    """
    if source is None:
        return []

    if hasattr(source, "last_event"):
        return extract_agent_metadata(source.last_event)

    if hasattr(source, "events") and getattr(source, "events") is not None:
        agents = []
        for index, event in enumerate(source.events):
            if hasattr(event, "metadata"):
                agents.append(summarize_agent_metadata(event.metadata, index))
        if agents:
            return agents

    if hasattr(source, "metadata"):
        return extract_agent_metadata(source.metadata)

    if isinstance(source, (list, tuple)):
        agents = []
        for index, item in enumerate(source):
            if isinstance(item, dict):
                agents.append(summarize_agent_metadata(item, index))
        return agents

    if not isinstance(source, dict):
        raise TypeError(f"Unsupported metadata source: {type(source)!r}")

    if isinstance(source.get("agents"), list):
        agents = []
        for index, item in enumerate(source["agents"]):
            if isinstance(item, dict):
                agents.append(summarize_agent_metadata(item, index))
        return agents

    if "agent" in source:
        return [summarize_agent_metadata(source, 0)]

    raise ValueError("metadata source does not contain an AI2-THOR agent payload")

def _normalized_object_key(value: Any) -> str:
    return "".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _task_object_references(task_graph: dict) -> tuple[set[str], set[str]]:
    object_ids: set[str] = set()
    object_types: set[str] = set()
    for task in _task_index(task_graph).values():
        grounding = task.get("grounding")
        if not isinstance(grounding, dict):
            continue
        for key in (
            "node_ids",
            "source_node_ids",
            "destination_node_ids",
            "source_object_ids",
            "destination_object_ids",
        ):
            object_ids.update(
                str(value) for value in grounding.get(key) or [] if str(value).strip()
            )
        for key in (
            "object_tags",
            "source_object_tags",
            "destination_object_tags",
        ):
            object_types.update(
                _normalized_object_key(value)
                for value in grounding.get(key) or []
                if _normalized_object_key(value)
            )
        for key in ("source_selector", "destination_selector"):
            selector = grounding.get(key)
            if not isinstance(selector, dict):
                continue
            object_types.update(
                _normalized_object_key(value)
                for value in selector.get("object_types") or []
                if _normalized_object_key(value)
            )
    return object_ids, object_types


def _summarize_execution_agents(
    agents: list[dict],
    task_graph: dict,
    max_relevant_objects: int = 12,
) -> list[dict]:
    """Keep only agent state that can affect the current allocation decision."""

    relevant_ids, relevant_types = _task_object_references(task_graph)
    summaries = []
    for agent in agents:
        relevant_visible_objects = []
        for obj in agent.get("visible_objects") or []:
            if not isinstance(obj, dict):
                continue
            object_id = str(obj.get("objectId") or "")
            object_type = _normalized_object_key(obj.get("objectType"))
            object_name = _normalized_object_key(obj.get("name"))
            object_id_type = _normalized_object_key(object_id.split("|", 1)[0])
            if not (
                object_id in relevant_ids
                or object_type in relevant_types
                or object_name in relevant_types
                or object_id_type in relevant_types
            ):
                continue
            relevant_visible_objects.append({
                key: deepcopy(obj.get(key))
                for key in (
                    "objectId",
                    "objectType",
                    "distance",
                    "position",
                    "pickupable",
                    "openable",
                    "isOpen",
                )
                if key in obj
            })
            if len(relevant_visible_objects) >= max_relevant_objects:
                break

        summaries.append({
            "agent_id": str(agent.get("agent_id")),
            "position": deepcopy(agent.get("position")),
            "held_objects": deepcopy(agent.get("held_objects") or []),
            "skills": deepcopy(agent.get("skills") or []),
            "relevant_visible_objects": relevant_visible_objects,
        })
    return summaries


def _parse_json_from_text(text: str) -> Any | None:
    if parse_json_from_text is not None:
        return parse_json_from_text(text)
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _task_index(task_graph: dict) -> dict[str, dict]:
    tasks = task_graph.get("flat_tasks") or []
    if not isinstance(tasks, list):
        raise TypeError("task_graph.flat_tasks must be a list")
    return {
        str(task["id"]): deepcopy(task)
        for task in tasks
        if isinstance(task, dict) and task.get("id")
    }


def _source_type_keys(task: dict[str, Any]) -> set[str]:
    grounding = task.get("grounding") if isinstance(task.get("grounding"), dict) else {}
    selector = grounding.get("source_selector") if isinstance(grounding.get("source_selector"), dict) else {}
    values = selector.get("object_types") or grounding.get("source_object_tags") or []
    if isinstance(values, str):
        values = [values]
    return {_normalized_object_key(value) for value in values if str(value).strip()}


def _summarize_execution_scene_context(
    scene_context: dict | None,
    max_nodes: int = 12,
    max_triples: int = 20,
) -> dict | None:
    if not isinstance(scene_context, dict):
        return None

    compact_nodes = []
    for node in scene_context.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        compact = {
            key: node.get(key)
            for key in ("pruned_id", "original_id", "object_tag", "caption", "class_name")
            if key in node
        }
        compact_nodes.append(compact)
        if len(compact_nodes) >= max_nodes:
            break

    compact_triples = []
    for triple in scene_context.get("triples") or []:
        compact_triples.append(triple)
        if len(compact_triples) >= max_triples:
            break

    return {
        "task": scene_context.get("task"),
        "task_spec": scene_context.get("task_spec"),
        "num_nodes": len(scene_context.get("nodes") or []),
        "num_edges": len(scene_context.get("edges") or []),
        "num_triples": len(scene_context.get("triples") or []),
        "nodes": compact_nodes,
        "triples": compact_triples,
    }


EXECUTION_POLICY = "first_unit_then_replan"
EXECUTION_EDGE_KINDS = {
    "semantic",
    "object_handoff",
    "agent_sequence",
    "resource_mutex",
}


def _execution_progress_sets(
    progress: dict[str, Any] | None,
) -> tuple[set[str], set[str]]:
    value = progress if isinstance(progress, dict) else {}
    completed = {
        str(task_id)
        for task_id in value.get("completed_task_ids", value.get("completed", [])) or []
    }
    failed = {
        str(task_id)
        for task_id in value.get("failed_task_ids", value.get("failed", [])) or []
    }
    return completed, failed


def _selector_expansion_info(task: dict[str, Any]) -> dict[str, Any]:
    runtime = task.get("runtime") if isinstance(task.get("runtime"), dict) else {}
    value = runtime.get("selector_expansion")
    return value if isinstance(value, dict) else {}


def _source_object_ids(task: dict[str, Any]) -> list[str]:
    grounding = task.get("grounding") if isinstance(task.get("grounding"), dict) else {}
    return [
        str(value)
        for value in grounding.get("source_object_ids") or []
        if str(value).strip()
    ]


def _source_key(task: dict[str, Any]) -> str | None:
    object_ids = _source_object_ids(task)
    if len(object_ids) == 1:
        return f"id:{object_ids[0]}"
    source_types = sorted(_source_type_keys(task))
    if len(source_types) == 1:
        return f"type:{source_types[0]}"
    return None


def _destination_key(task: dict[str, Any]) -> str | None:
    grounding = task.get("grounding") if isinstance(task.get("grounding"), dict) else {}
    object_ids = [
        str(value)
        for value in grounding.get("destination_object_ids") or []
        if str(value).strip()
    ]
    if object_ids:
        return f"id:{object_ids[0]}"
    selector = (
        grounding.get("destination_selector")
        if isinstance(grounding.get("destination_selector"), dict)
        else {}
    )
    values = selector.get("object_types") or grounding.get("destination_object_tags") or []
    if isinstance(values, str):
        values = [values]
    normalized = sorted({_normalized_object_key(value) for value in values if str(value).strip()})
    return f"type:{','.join(normalized)}" if normalized else None


def _task_resource_key(task: dict[str, Any]) -> str | None:
    action = normalize_logical_action(task.get("action"))
    if action == "place":
        return _destination_key(task)
    return _source_key(task)


def _clone_family(task_id: str) -> str:
    return re.sub(r"__instance_\d+$", "", str(task_id))


def _selector_quantifier(task: dict[str, Any]) -> str:
    grounding = task.get("grounding") if isinstance(task.get("grounding"), dict) else {}
    selector = (
        grounding.get("source_selector")
        if isinstance(grounding.get("source_selector"), dict)
        else {}
    )
    return str(selector.get("quantifier") or "").strip().lower()


def _ordered_expansion_groups(
    tasks: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for task_id, task in tasks.items():
        info = _selector_expansion_info(task)
        origin = str(
            info.get("origin_task_id")
            or info.get("selector_origin_task_id")
            or ""
        )
        if origin:
            groups[origin].append(task_id)
    for values in groups.values():
        values.sort(
            key=lambda task_id: (
                int(_selector_expansion_info(tasks[task_id]).get("instance_index") or 0),
                task_id,
            )
        )
    return dict(groups)


def _exact_source_bijection(
    pick_ids: list[str],
    place_ids: list[str],
    tasks: dict[str, dict[str, Any]],
) -> dict[str, str] | None:
    if not pick_ids or len(pick_ids) != len(place_ids):
        return None
    picks: dict[str, str] = {}
    places: dict[str, str] = {}
    for task_id in pick_ids:
        values = _source_object_ids(tasks[task_id])
        if len(values) != 1 or values[0] in picks:
            return None
        picks[values[0]] = task_id
    for task_id in place_ids:
        values = _source_object_ids(tasks[task_id])
        if len(values) != 1 or values[0] in places:
            return None
        places[values[0]] = task_id
    if set(picks) != set(places):
        return None
    return {places[object_id]: picks[object_id] for object_id in picks}


def _provenance_has_exact_source_ids(
    task_ids: Iterable[str],
    tasks: dict[str, dict[str, Any]],
) -> bool:
    for task_id in task_ids:
        source_ids = _source_object_ids(tasks[task_id])
        info = _selector_expansion_info(tasks[task_id])
        bound_id = str(info.get("bound_source_object_id") or "")
        if len(source_ids) != 1 or not bound_id or source_ids[0] != bound_id:
            return False
    return True


def _execution_pick_place_pairs(
    tasks: dict[str, dict[str, Any]],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    pairs: dict[str, str] = {}
    events: list[dict[str, Any]] = []
    groups = _ordered_expansion_groups(tasks)
    for place_origin, place_ids in groups.items():
        if not place_ids or any(
            normalize_logical_action(tasks[task_id].get("action")) != "place"
            for task_id in place_ids
        ):
            continue
        dependency_origins = {
            str(origin)
            for task_id in place_ids
            for origin in _selector_expansion_info(tasks[task_id]).get(
                "expanded_dependency_origins", []
            )
            if str(origin)
        }
        for pick_origin in sorted(dependency_origins):
            pick_ids = groups.get(pick_origin, [])
            if not pick_ids or any(
                normalize_logical_action(tasks[task_id].get("action")) != "pick"
                for task_id in pick_ids
            ):
                continue
            if not _provenance_has_exact_source_ids(
                [*pick_ids, *place_ids], tasks
            ):
                events.append(
                    {
                        "status": "rejected",
                        "mode": "provenance",
                        "pick_origin": pick_origin,
                        "place_origin": place_origin,
                        "reason": "bound_source_object_id_does_not_match_grounding",
                    }
                )
                continue
            matched = _exact_source_bijection(pick_ids, place_ids, tasks)
            if matched is None:
                events.append(
                    {
                        "status": "rejected",
                        "mode": "provenance",
                        "pick_origin": pick_origin,
                        "place_origin": place_origin,
                        "reason": "source_object_ids_are_not_a_bijection",
                    }
                )
                continue
            pairs.update(matched)
            events.append(
                {
                    "status": "paired",
                    "mode": "provenance",
                    "pick_origin": pick_origin,
                    "place_origin": place_origin,
                    "pairs": deepcopy(matched),
                }
            )

    if pairs:
        return pairs, events

    # Compatibility is intentionally narrow.  Old expanded graphs did not
    # record selector_expansion provenance, so accept only clone families that
    # still prove an ``all`` expansion, an aggregate Pick barrier, and an exact
    # one-to-one object-ID match.  Object types alone are never sufficient.
    clone_groups: dict[str, list[str]] = defaultdict(list)
    for task_id in tasks:
        clone_groups[_clone_family(task_id)].append(task_id)
    pick_groups = [
        (family, sorted(task_ids))
        for family, task_ids in clone_groups.items()
        if len(task_ids) > 1
        and all(
            normalize_logical_action(tasks[task_id].get("action")) == "pick"
            and _selector_quantifier(tasks[task_id]) == "all"
            for task_id in task_ids
        )
    ]
    place_groups = [
        (family, sorted(task_ids))
        for family, task_ids in clone_groups.items()
        if len(task_ids) > 1
        and all(
            normalize_logical_action(tasks[task_id].get("action")) == "place"
            and _selector_quantifier(tasks[task_id]) == "all"
            for task_id in task_ids
        )
    ]
    for pick_family, pick_ids in pick_groups:
        pick_set = set(pick_ids)
        for place_family, place_ids in place_groups:
            has_aggregate_barrier = any(
                len(
                    pick_set
                    & {
                        str(dependency)
                        for dependency in tasks[place_id].get("depends_on") or []
                    }
                )
                > 1
                for place_id in place_ids
            )
            if not has_aggregate_barrier:
                continue
            matched = _exact_source_bijection(pick_ids, place_ids, tasks)
            if matched is None:
                continue
            pairs.update(matched)
            events.append(
                {
                    "status": "paired",
                    "mode": "legacy_strict",
                    "pick_origin": pick_family,
                    "place_origin": place_family,
                    "pairs": deepcopy(matched),
                }
            )
    return pairs, events


def build_effective_execution_graph(
    task_graph: dict[str, Any],
    progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an independent execution graph; task_graph is never mutated."""

    all_tasks = _task_index(task_graph)
    completed, failed = _execution_progress_sets(progress)
    task_order = [
        task_id
        for task_id in all_tasks
        if task_id not in completed and task_id not in failed
    ]
    tasks = {task_id: deepcopy(all_tasks[task_id]) for task_id in task_order}
    dependencies = {
        task_id: {
            str(dependency)
            for dependency in task.get("depends_on") or []
            if str(dependency) in tasks
        }
        for task_id, task in tasks.items()
    }
    failed_dependencies = {
        task_id: sorted(
            {
                str(dependency)
                for dependency in task.get("depends_on") or []
                if str(dependency) in failed
            }
        )
        for task_id, task in tasks.items()
    }
    failed_dependencies = {
        task_id: values for task_id, values in failed_dependencies.items() if values
    }

    all_pairs, pairing_events = _execution_pick_place_pairs(all_tasks)
    pair_pick_ids = set(all_pairs.values())
    active_pairs = {
        place_id: pick_id
        for place_id, pick_id in all_pairs.items()
        if place_id in tasks
    }
    for place_id, pick_id in active_pairs.items():
        dependencies[place_id] -= pair_pick_ids
        if pick_id in tasks:
            dependencies[place_id].add(pick_id)

    edges: list[dict[str, str]] = []
    for target_id in task_order:
        for source_id in sorted(dependencies[target_id]):
            kind = "semantic"
            if active_pairs.get(target_id) == source_id:
                kind = "object_handoff"
            elif (
                normalize_logical_action(tasks[source_id].get("action")) == "place"
                and normalize_logical_action(tasks[target_id].get("action")) == "place"
                and _destination_key(tasks[source_id]) == _destination_key(tasks[target_id])
                and _destination_key(tasks[source_id]) is not None
            ):
                kind = "resource_mutex"
            edges.append(
                {
                    "from_task_id": source_id,
                    "to_task_id": target_id,
                    "kind": kind,
                }
            )
    return {
        "task": task_graph.get("task"),
        "tasks": tasks,
        "task_order": task_order,
        "dependencies": {
            task_id: sorted(values) for task_id, values in dependencies.items()
        },
        "dependency_edges": edges,
        "pick_for_place": active_pairs,
        "completed_task_ids": sorted(completed),
        "failed_task_ids": sorted(failed),
        "failed_dependencies": failed_dependencies,
        "pairing_events": pairing_events,
    }


def _execution_graph_fingerprint(
    task_graph: dict[str, Any],
    progress: dict[str, Any] | None,
    task_graph_version: int,
) -> str:
    tasks = _task_index(task_graph)
    completed, failed = _execution_progress_sets(progress)
    value = {
        "version": int(task_graph_version),
        "completed": sorted(completed),
        "failed": sorted(failed),
        "tasks": [
            {
                "id": task_id,
                "action": task.get("action"),
                "depends_on": task.get("depends_on") or [],
                "grounding": task.get("grounding") or {},
                "runtime": task.get("runtime") or {},
            }
            for task_id, task in tasks.items()
        ],
    }
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _execution_agents(agent_states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    agents = extract_agent_metadata(agent_states)
    if not agents:
        raise ValueError("agent_states must contain at least one agent")
    original_by_id = {
        str(value.get("agent_id", value.get("robot_id", index))): value
        for index, value in enumerate(agent_states)
        if isinstance(value, dict)
    }
    seen: set[str] = set()
    normalized = []
    for agent in agents:
        agent_id = str(agent.get("agent_id"))
        if agent_id in seen:
            raise ValueError(f"duplicate agent_id {agent_id!r}")
        seen.add(agent_id)
        source = original_by_id.get(agent_id, {})
        item = deepcopy(agent)
        item["agent_id"] = agent_id
        item["inventory_capacity"] = max(
            int(
                source.get("inventory_capacity")
                or source.get("inventoryCapacity")
                or 1
            ),
            1,
        )
        normalized.append(item)
    return normalized


def _inventory_token(item: dict[str, Any]) -> str | None:
    object_id = item.get("objectId") or item.get("object_id") or item.get("id")
    if object_id not in (None, ""):
        return f"id:{object_id}"
    object_type = item.get("objectType") or item.get("object_type") or item.get("type")
    if object_type not in (None, ""):
        return f"type:{_normalized_object_key(object_type)}"
    return None


def _initial_execution_inventory(
    agents: list[dict[str, Any]],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for agent in agents:
        values = (
            agent.get("held_objects")
            or agent.get("inventoryObjects")
            or agent.get("inventory")
            or []
        )
        tokens = []
        for value in values:
            if isinstance(value, dict):
                token = _inventory_token(value)
            else:
                text = str(value)
                token = f"id:{text}" if "|" in text else f"type:{_normalized_object_key(text)}"
            if token:
                tokens.append(token)
        result[str(agent["agent_id"])] = tokens
    return result


def _matching_inventory_token(
    source_key: str | None,
    held: Iterable[str],
) -> str | None:
    if source_key is None:
        return None
    values = list(held)
    if source_key in values:
        return source_key
    if source_key.startswith("type:"):
        wanted = source_key.split(":", 1)[1]
        for token in values:
            raw = token.split(":", 1)[1] if ":" in token else token
            if _normalized_object_key(raw.split("|", 1)[0]) == wanted:
                return token
    return None


def _agent_supports_execution_action(
    agent: dict[str, Any],
    action: str,
) -> bool:
    skills = {
        normalize_logical_action(value)
        for value in agent.get("skills") or []
        if str(value).strip()
    }
    return not skills or action in skills


def _execution_eligibility_error(
    task_id: str,
    agent_id: str,
    *,
    effective: dict[str, Any],
    agents_by_id: dict[str, dict[str, Any]],
    inventory: dict[str, list[str]],
    assigned_agent_by_task: dict[str, str],
) -> str | None:
    task = effective["tasks"][task_id]
    agent = agents_by_id.get(agent_id)
    if agent is None:
        return f"unknown agent_id {agent_id!r}"
    action = normalize_logical_action(task.get("action"))
    if not _agent_supports_execution_action(agent, action):
        return f"agent {agent_id} does not support action {action!r}"

    source_key = _source_key(task)
    own_token = _matching_inventory_token(source_key, inventory[agent_id])
    holders = [
        candidate
        for candidate, held in inventory.items()
        if _matching_inventory_token(source_key, held) is not None
    ]
    capacity = int(agent.get("inventory_capacity") or 1)
    if action == "pick":
        if holders:
            return f"source {source_key!r} is already held by {holders}"
        if len(inventory[agent_id]) >= capacity:
            return f"agent {agent_id} inventory is full"
        return None

    if action == "place":
        pick_id = effective["pick_for_place"].get(task_id)
        if pick_id:
            pick_agent = assigned_agent_by_task.get(pick_id)
            if pick_agent is not None and pick_agent != agent_id:
                return (
                    f"place {task_id} must use the agent {pick_agent} "
                    f"assigned to pick {pick_id}"
                )
            if own_token is None:
                return f"agent {agent_id} does not hold paired source {source_key!r}"
            return None
        if holders and own_token is None:
            return f"source {source_key!r} is held by another agent {holders}"
        if not holders and len(inventory[agent_id]) >= capacity:
            return f"agent {agent_id} lacks capacity for atomic place"
        return None

    if action in {"drop", "move_held"} and own_token is None:
        return f"agent {agent_id} does not hold source {source_key!r}"
    return None


def _apply_execution_transition(
    task_id: str,
    agent_id: str,
    *,
    effective: dict[str, Any],
    inventory: dict[str, list[str]],
) -> None:
    task = effective["tasks"][task_id]
    action = normalize_logical_action(task.get("action"))
    source_key = _source_key(task)
    if action == "pick" and source_key:
        inventory[agent_id].append(source_key)
    elif action in {"place", "drop"}:
        token = _matching_inventory_token(source_key, inventory[agent_id])
        if token is not None:
            inventory[agent_id].remove(token)


def _effective_cycle_task_ids(effective: dict[str, Any]) -> list[str]:
    remaining = set(effective["task_order"])
    dependencies = {
        task_id: set(values)
        for task_id, values in effective["dependencies"].items()
    }
    while remaining:
        ready = {
            task_id
            for task_id in remaining
            if not (dependencies[task_id] & remaining)
        }
        if not ready:
            return sorted(remaining)
        remaining -= ready
    return []


def validate_execution_plan(
    effective: dict[str, Any],
    agents: list[dict[str, Any]],
    raw_plan: Any,
) -> dict[str, Any]:
    """Validate full coverage and simulate resource state for every unit."""

    errors: list[str] = []
    if not isinstance(raw_plan, dict):
        return {
            "status": "invalid",
            "errors": ["execution plan response must be a JSON object"],
            "units": [],
            "covered_task_ids": [],
            "missing_task_ids": list(effective["task_order"]),
        }
    if set(raw_plan) - {"state", "units"}:
        errors.append(
            f"execution plan contains unsupported fields {sorted(set(raw_plan) - {'state', 'units'})}"
        )
    if str(raw_plan.get("state") or "dispatchable") != "dispatchable":
        errors.append("model response state must be 'dispatchable'")
    units = raw_plan.get("units")
    if not isinstance(units, list) or not units:
        errors.append("execution plan units must be a non-empty list")
        units = []

    agents_by_id = {str(agent["agent_id"]): agent for agent in agents}
    inventory = _initial_execution_inventory(agents)
    expected = set(effective["task_order"])
    seen: set[str] = set()
    finished: set[str] = set()
    assigned_agent_by_task: dict[str, str] = {}
    normalized_units: list[dict[str, Any]] = []
    for step, unit in enumerate(units, start=1):
        if not isinstance(unit, dict):
            errors.append(f"unit {step} must be a JSON object")
            continue
        if set(unit) - {"time_step", "assignments"}:
            errors.append(f"unit {step} contains unsupported fields")
        if unit.get("time_step") != step:
            errors.append(f"unit {step} time_step must equal {step}")
        assignments = unit.get("assignments")
        if not isinstance(assignments, list) or not assignments:
            errors.append(f"unit {step} assignments must be non-empty")
            continue
        used_agents: set[str] = set()
        used_sources: set[str] = set()
        used_resources: set[str] = set()
        accepted: list[tuple[str, str]] = []
        normalized_assignments: list[dict[str, str]] = []
        for position, assignment in enumerate(assignments, start=1):
            if not isinstance(assignment, dict):
                errors.append(f"unit {step} assignment {position} must be a JSON object")
                continue
            if set(assignment) != {"task_id", "agent_id"}:
                errors.append(
                    f"unit {step} assignment {position} must contain only task_id and agent_id"
                )
            task_id = str(assignment.get("task_id") or "")
            agent_id = str(assignment.get("agent_id") or "")
            if task_id not in expected:
                errors.append(f"unknown task_id {task_id!r}")
                continue
            if task_id in seen:
                errors.append(f"task {task_id} is scheduled more than once")
                continue
            if agent_id not in agents_by_id:
                errors.append(f"task {task_id} uses unknown agent_id {agent_id!r}")
                continue
            if agent_id in used_agents:
                errors.append(f"agent {agent_id} has multiple tasks in unit {step}")
                continue
            pending = set(effective["dependencies"][task_id]) - finished
            if pending:
                errors.append(
                    f"task {task_id} runs before dependencies {sorted(pending)}"
                )
                continue
            source_key = _source_key(effective["tasks"][task_id])
            if source_key and source_key in used_sources:
                errors.append(f"source {source_key} is used twice in unit {step}")
                continue
            action = normalize_logical_action(effective["tasks"][task_id].get("action"))
            resource_key = _task_resource_key(effective["tasks"][task_id])
            if (
                resource_key
                and action in {"place", "open", "close"}
                and resource_key in used_resources
            ):
                errors.append(
                    f"resource {resource_key} is used concurrently in unit {step}"
                )
                continue
            reason = _execution_eligibility_error(
                task_id,
                agent_id,
                effective=effective,
                agents_by_id=agents_by_id,
                inventory=inventory,
                assigned_agent_by_task=assigned_agent_by_task,
            )
            if reason:
                errors.append(f"task {task_id}: {reason}")
                continue
            seen.add(task_id)
            used_agents.add(agent_id)
            if source_key:
                used_sources.add(source_key)
            if resource_key and action in {"place", "open", "close"}:
                used_resources.add(resource_key)
            assigned_agent_by_task[task_id] = agent_id
            accepted.append((task_id, agent_id))
            normalized_assignments.append(
                {"task_id": task_id, "agent_id": agent_id}
            )
        for task_id, agent_id in accepted:
            _apply_execution_transition(
                task_id,
                agent_id,
                effective=effective,
                inventory=inventory,
            )
        finished.update(task_id for task_id, _ in accepted)
        normalized_units.append(
            {"time_step": step, "assignments": normalized_assignments}
        )

    missing = sorted(expected - seen)
    if missing:
        errors.append(f"execution plan is missing remaining tasks {missing}")
    return {
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "units": normalized_units,
        "covered_task_ids": sorted(seen),
        "missing_task_ids": missing,
        "assigned_agent_by_task": assigned_agent_by_task,
        "final_inventory": inventory,
    }


def _deterministic_execution_units(
    effective: dict[str, Any],
    agents: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    agents_by_id = {str(agent["agent_id"]): agent for agent in agents}
    inventory = _initial_execution_inventory(agents)
    finished: set[str] = set()
    assigned_agent_by_task: dict[str, str] = {}
    units: list[dict[str, Any]] = []
    errors: list[str] = []
    while len(finished) < len(effective["task_order"]):
        ready = [
            task_id
            for task_id in effective["task_order"]
            if task_id not in finished
            and set(effective["dependencies"][task_id]) <= finished
        ]
        ready.sort(
            key=lambda task_id: (
                0
                if normalize_logical_action(effective["tasks"][task_id].get("action"))
                in {"place", "drop"}
                else 1,
                task_id,
            )
        )
        selected: list[tuple[str, str]] = []
        used_agents: set[str] = set()
        used_sources: set[str] = set()
        used_resources: set[str] = set()
        reasons: list[str] = []
        for task_id in ready:
            task = effective["tasks"][task_id]
            action = normalize_logical_action(task.get("action"))
            source_key = _source_key(task)
            resource_key = _task_resource_key(task)
            if source_key and source_key in used_sources:
                continue
            if (
                resource_key
                and action in {"place", "open", "close"}
                and resource_key in used_resources
            ):
                continue
            pair_id = effective["pick_for_place"].get(task_id)
            pair_agent = assigned_agent_by_task.get(pair_id or "")
            candidates = list(agents_by_id)
            if pair_agent in candidates:
                candidates.remove(pair_agent)
                candidates.insert(0, pair_agent)
            for agent_id in candidates:
                if agent_id in used_agents:
                    continue
                reason = _execution_eligibility_error(
                    task_id,
                    agent_id,
                    effective=effective,
                    agents_by_id=agents_by_id,
                    inventory=inventory,
                    assigned_agent_by_task=assigned_agent_by_task,
                )
                if reason is None:
                    selected.append((task_id, agent_id))
                    used_agents.add(agent_id)
                    if source_key:
                        used_sources.add(source_key)
                    if resource_key and action in {"place", "open", "close"}:
                        used_resources.add(resource_key)
                    break
                reasons.append(f"{task_id}/{agent_id}: {reason}")
        if not selected:
            errors.extend(reasons or ["no ready task has an eligible agent"])
            break
        for task_id, agent_id in selected:
            assigned_agent_by_task[task_id] = agent_id
        for task_id, agent_id in selected:
            _apply_execution_transition(
                task_id,
                agent_id,
                effective=effective,
                inventory=inventory,
            )
        finished.update(task_id for task_id, _ in selected)
        units.append(
            {
                "time_step": len(units) + 1,
                "assignments": [
                    {"task_id": task_id, "agent_id": agent_id}
                    for task_id, agent_id in selected
                ],
            }
        )
    return units, errors


def _execution_blocking_preflight(
    effective: dict[str, Any],
    agents: list[dict[str, Any]],
) -> tuple[str, list[str], list[Any]] | None:
    inventory = _initial_execution_inventory(agents)
    inventory_conflicts: list[dict[str, Any]] = []
    owners_by_object_id: dict[str, list[str]] = defaultdict(list)
    for agent in agents:
        agent_id = str(agent["agent_id"])
        held = inventory[agent_id]
        capacity = int(agent.get("inventory_capacity") or 1)
        if len(held) > capacity:
            inventory_conflicts.append(
                {
                    "agent_id": agent_id,
                    "code": "inventory_capacity_exceeded",
                    "capacity": capacity,
                    "held": deepcopy(held),
                }
            )
        for token in held:
            if token.startswith("id:"):
                owners_by_object_id[token].append(agent_id)
    for token, owner_ids in owners_by_object_id.items():
        if len(owner_ids) > 1:
            inventory_conflicts.append(
                {
                    "code": "object_has_multiple_holders",
                    "object_id": token.split(":", 1)[1],
                    "agent_ids": sorted(owner_ids),
                }
            )
    if inventory_conflicts:
        return (
            "invalid_agent_state",
            list(effective["task_order"]),
            inventory_conflicts,
        )
    if effective["failed_dependencies"]:
        conflicts = [
            {"task_id": task_id, "failed_dependencies": values}
            for task_id, values in effective["failed_dependencies"].items()
        ]
        return ("failed_dependency", list(effective["failed_dependencies"]), conflicts)
    cycle = _effective_cycle_task_ids(effective)
    if cycle:
        return ("dependency_cycle", cycle, [{"cycle_task_ids": cycle}])
    if not effective["task_order"]:
        return ("no_remaining_tasks", [], [])
    ready = [
        task_id
        for task_id in effective["task_order"]
        if not effective["dependencies"][task_id]
    ]
    agents_by_id = {str(agent["agent_id"]): agent for agent in agents}
    conflicts = []
    any_eligible = False
    for task_id in ready:
        reasons = {}
        for agent_id in agents_by_id:
            reason = _execution_eligibility_error(
                task_id,
                agent_id,
                effective=effective,
                agents_by_id=agents_by_id,
                inventory=inventory,
                assigned_agent_by_task={},
            )
            if reason is None:
                any_eligible = True
            else:
                reasons[agent_id] = reason
        if reasons:
            conflicts.append({"task_id": task_id, "reasons_by_agent": reasons})
    if ready and not any_eligible:
        return ("no_eligible_agent", ready, conflicts)
    if not ready:
        return ("dependency_deadlock", effective["task_order"], conflicts)
    return None


def _execution_blocked_result(
    *,
    task_graph: dict[str, Any],
    task_graph_version: int,
    fingerprint: str,
    agents: list[dict[str, Any]],
    effective: dict[str, Any],
    code: str,
    task_ids: Iterable[str],
    conflicts: list[Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "state": "blocked",
        "task": task_graph.get("task"),
        "task_graph_version": int(task_graph_version),
        "graph_fingerprint": fingerprint,
        "execution_policy": EXECUTION_POLICY,
        "units": [],
        "dependency_edges": deepcopy(effective["dependency_edges"]),
        "blocking": {
            "code": code,
            "task_ids": sorted({str(value) for value in task_ids}),
            "agent_inventories": {
                str(agent["agent_id"]): deepcopy(agent.get("held_objects") or [])
                for agent in agents
            },
            "conflicts": deepcopy(conflicts),
            "recommended_recovery": "replan_task_graph",
        },
        "diagnostics": diagnostics,
    }


def _materialize_execution_edges(
    effective: dict[str, Any],
    units: list[dict[str, Any]],
) -> list[dict[str, str]]:
    edges = [deepcopy(edge) for edge in effective["dependency_edges"]]
    ordered = [
        (int(unit["time_step"]), item["task_id"], item["agent_id"])
        for unit in units
        for item in unit["assignments"]
    ]
    by_agent: dict[str, list[tuple[int, str]]] = defaultdict(list)
    by_resource: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for step, task_id, agent_id in ordered:
        by_agent[agent_id].append((step, task_id))
        action = normalize_logical_action(effective["tasks"][task_id].get("action"))
        resource = _task_resource_key(effective["tasks"][task_id])
        if resource and action in {"place", "open", "close"}:
            by_resource[resource].append((step, task_id))
    for values, kind in [
        *[(value, "agent_sequence") for value in by_agent.values()],
        *[(value, "resource_mutex") for value in by_resource.values()],
    ]:
        values.sort()
        for (_, source_id), (_, target_id) in zip(values, values[1:]):
            if any(
                edge["from_task_id"] == source_id
                and edge["to_task_id"] == target_id
                for edge in edges
            ):
                continue
            edges.append(
                {
                    "from_task_id": source_id,
                    "to_task_id": target_id,
                    "kind": kind,
                }
            )
    if any(edge["kind"] not in EXECUTION_EDGE_KINDS for edge in edges):
        raise AssertionError("unsupported Execution Plan dependency kind")
    return edges


def _call_qwen_for_execution_plan(
    *,
    effective: dict[str, Any],
    agents: list[dict[str, Any]],
    scene_context: dict[str, Any] | None,
    qwen_chat: Any,
    max_new_tokens: int,
    correction: dict[str, Any] | None,
    diagnostics: dict[str, Any],
    raw_response_path: Path | None,
) -> Any | None:
    payload = {
        "request": (
            "build_complete_execution_plan"
            if correction is None
            else "correct_complete_execution_plan"
        ),
        "hard_constraints": [
            "Return JSON only and use exactly the output schema.",
            "Schedule every remaining task exactly once.",
            "Never invent, rename, omit, or duplicate task IDs.",
            "Every dependency must finish in an earlier time step.",
            "Each agent may run at most one task per time step.",
            "Respect skills, one-slot inventory, object ownership, and resource mutexes.",
            "A paired Pick and Place must use the same agent.",
        ],
        "remaining_tasks": [
            {
                "id": task_id,
                "name": effective["tasks"][task_id].get("name"),
                "description": effective["tasks"][task_id].get("description"),
                "action": effective["tasks"][task_id].get("action"),
                "grounding": effective["tasks"][task_id].get("grounding") or {},
                "depends_on": effective["dependencies"][task_id],
            }
            for task_id in effective["task_order"]
        ],
        "execution_dependency_edges": effective["dependency_edges"],
        "pick_for_place": effective["pick_for_place"],
        "agents": _summarize_execution_agents(
            agents,
            {
                "flat_tasks": [
                    effective["tasks"][task_id]
                    for task_id in effective["task_order"]
                ]
            },
        ),
        "scene_context": _summarize_execution_scene_context(scene_context),
        "output_schema": {
            "state": "dispatchable",
            "units": [
                {
                    "time_step": 1,
                    "assignments": [
                        {"task_id": "T1", "agent_id": "0"}
                    ],
                }
            ],
        },
    }
    if correction is not None:
        payload["attempt"] = correction["attempt"]
        payload["max_attempts"] = correction["max_attempts"]
        payload["previous_validation_errors"] = correction["validation_errors"]

    original_max_new_tokens: Any = None
    override_tokens = False
    try:
        if hasattr(qwen_chat, "max_new_tokens"):
            original_max_new_tokens = qwen_chat.max_new_tokens
            qwen_chat.max_new_tokens = max(int(max_new_tokens), 1)
            override_tokens = True
        if correction is None:
            if hasattr(qwen_chat, "reset"):
                qwen_chat.reset()
            if hasattr(qwen_chat, "messages"):
                qwen_chat.messages = [
                    {
                        "role": "system",
                        "content": (
                            "You create a complete resource-aware schedule for "
                            "the remaining semantic Task Graph. Return JSON only."
                        ),
                    }
                ]
        response = qwen_chat(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as exc:
        diagnostics.update(
            {
                "status": "failed",
                "stage": "model_call",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        raise TaskAllocationError(
            "Qwen execution planning call failed.",
            code="allocation_model_unavailable",
            diagnostics=diagnostics,
        ) from exc
    finally:
        if override_tokens:
            qwen_chat.max_new_tokens = original_max_new_tokens

    response_text = str(response)
    if raw_response_path is not None:
        raw_response_path.parent.mkdir(parents=True, exist_ok=True)
        raw_response_path.write_text(response_text, encoding="utf-8")
    parsed = _parse_json_from_text(response_text)
    diagnostics.update(
        {
            "status": "success" if parsed is not None else "failed",
            "stage": "complete" if parsed is not None else "json_parse",
            "json_parse_status": "success" if parsed is not None else "invalid_json",
            "raw_response_preview": response_text[:4000],
            "response_characters": len(response_text),
            "raw_response_path": str(raw_response_path) if raw_response_path else None,
        }
    )
    return parsed


def build_execution_plan(
    task_graph: dict[str, Any] | Path,
    *,
    agent_states: list[dict[str, Any]],
    scene_context: dict[str, Any] | None = None,
    progress: dict[str, Any] | None = None,
    task_graph_version: int = 1,
    use_qwen: bool = True,
    qwen_model_path: str | None = DEFAULT_PLANNING_MODEL_PATH,
    qwen_conv_mode: str = "v0_mmtag",
    qwen_num_gpus: int = 1,
    qwen_chat: Any | None = None,
    qwen_max_new_tokens: int = 2048,
    allocation_max_attempts: int = 3,
    diagnostics_output_dir: Path | None = None,
) -> dict[str, Any]:
    """Build the complete independent Execution Plan for remaining work."""

    semantic_graph = load_json(task_graph) if isinstance(task_graph, Path) else task_graph
    if not isinstance(semantic_graph, dict):
        raise TypeError("task_graph must be a dict or Path")
    agents = _execution_agents(agent_states)
    effective = build_effective_execution_graph(semantic_graph, progress)
    fingerprint = _execution_graph_fingerprint(
        semantic_graph,
        progress,
        task_graph_version,
    )
    diagnostics: dict[str, Any] = {
        "status": "not_started",
        "stage": "not_started",
        "selected_backend": None,
        "fallback_used": False,
        "execution_policy": EXECUTION_POLICY,
        "remaining_task_ids": deepcopy(effective["task_order"]),
        "effective_dependencies": deepcopy(effective["dependencies"]),
        "pairing": deepcopy(effective["pairing_events"]),
        "attempts": [],
    }

    preflight = _execution_blocking_preflight(effective, agents)
    if preflight is not None:
        code, task_ids, conflicts = preflight
        diagnostics.update({"status": "blocked", "stage": "preflight"})
        return _execution_blocked_result(
            task_graph=semantic_graph,
            task_graph_version=task_graph_version,
            fingerprint=fingerprint,
            agents=agents,
            effective=effective,
            code=code,
            task_ids=task_ids,
            conflicts=conflicts,
            diagnostics=diagnostics,
        )

    if not use_qwen:
        units, scheduling_errors = _deterministic_execution_units(effective, agents)
        validation = validate_execution_plan(
            effective,
            agents,
            {"state": "dispatchable", "units": units},
        )
        if scheduling_errors or validation["status"] != "valid":
            conflicts = [
                *scheduling_errors,
                *validation.get("errors", []),
            ]
            diagnostics.update(
                {
                    "status": "blocked",
                    "stage": "deterministic_schedule",
                    "selected_backend": "deterministic_explicit",
                    "validation": validation,
                }
            )
            return _execution_blocked_result(
                task_graph=semantic_graph,
                task_graph_version=task_graph_version,
                fingerprint=fingerprint,
                agents=agents,
                effective=effective,
                code="resource_deadlock",
                task_ids=(
                    set(effective["task_order"])
                    - set(validation.get("covered_task_ids", []))
                ),
                conflicts=conflicts,
                diagnostics=diagnostics,
            )
        diagnostics.update(
            {
                "status": "valid",
                "stage": "complete",
                "selected_backend": "deterministic_explicit",
                "validation": validation,
            }
        )
        return {
            "state": "dispatchable",
            "task": semantic_graph.get("task"),
            "task_graph_version": int(task_graph_version),
            "graph_fingerprint": fingerprint,
            "execution_policy": EXECUTION_POLICY,
            "units": validation["units"],
            "dependency_edges": _materialize_execution_edges(
                effective,
                validation["units"],
            ),
            "blocking": None,
            "diagnostics": diagnostics,
        }

    chat = qwen_chat
    owns_chat = chat is None
    close_vlm_chat = None
    if chat is None:
        try:
            from conceptgraph.vlm import build_vlm_chat, close_vlm_chat

            chat = build_vlm_chat(
                backend="qwen",
                model_path=qwen_model_path,
                conv_mode=qwen_conv_mode,
                num_gpus=qwen_num_gpus,
            )
        except Exception as exc:
            raise TaskAllocationError(
                "Qwen execution planning model initialization failed.",
                code="allocation_model_unavailable",
                diagnostics={
                    **diagnostics,
                    "status": "failed",
                    "stage": "model_initialization",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            ) from exc

    accepted: dict[str, Any] | None = None
    correction: dict[str, Any] | None = None
    max_attempts = max(int(allocation_max_attempts), 1)
    try:
        for attempt in range(1, max_attempts + 1):
            attempt_diagnostics: dict[str, Any] = {
                "attempt": attempt,
                "prompt_kind": "initial" if attempt == 1 else "corrective_retry",
            }
            raw_path = (
                diagnostics_output_dir / f"attempt_{attempt:03d}_raw.txt"
                if diagnostics_output_dir is not None
                else None
            )
            candidate = _call_qwen_for_execution_plan(
                effective=effective,
                agents=agents,
                scene_context=scene_context,
                qwen_chat=chat,
                max_new_tokens=qwen_max_new_tokens,
                correction=correction,
                diagnostics=attempt_diagnostics,
                raw_response_path=raw_path,
            )
            validation = validate_execution_plan(effective, agents, candidate)
            if candidate is None:
                validation = {
                    "status": "invalid",
                    "errors": ["execution planning response is not valid JSON"],
                    "units": [],
                    "covered_task_ids": [],
                    "missing_task_ids": deepcopy(effective["task_order"]),
                }
            attempt_diagnostics["validation"] = deepcopy(validation)
            diagnostics["attempts"].append(deepcopy(attempt_diagnostics))
            if validation["status"] == "valid":
                accepted = validation
                diagnostics.update(
                    {
                        "status": "valid",
                        "stage": "complete",
                        "selected_backend": "qwen_local",
                        "selected_attempt": attempt,
                        "validation": deepcopy(validation),
                    }
                )
                break
            correction = {
                "attempt": attempt + 1,
                "max_attempts": max_attempts,
                "validation_errors": validation.get("errors") or [],
            }
    finally:
        if owns_chat and close_vlm_chat is not None and chat is not None:
            close_vlm_chat(chat)

    if accepted is None:
        diagnostics.update(
            {
                "status": "blocked",
                "stage": "attempts_exhausted",
                "selected_backend": None,
            }
        )
        conflicts = (
            diagnostics["attempts"][-1]["validation"].get("errors", [])
            if diagnostics["attempts"]
            else []
        )
        return _execution_blocked_result(
            task_graph=semantic_graph,
            task_graph_version=task_graph_version,
            fingerprint=fingerprint,
            agents=agents,
            effective=effective,
            code="invalid_model_plan",
            task_ids=effective["task_order"],
            conflicts=conflicts,
            diagnostics=diagnostics,
        )

    return {
        "state": "dispatchable",
        "task": semantic_graph.get("task"),
        "task_graph_version": int(task_graph_version),
        "graph_fingerprint": fingerprint,
        "execution_policy": EXECUTION_POLICY,
        "units": accepted["units"],
        "dependency_edges": _materialize_execution_edges(
            effective,
            accepted["units"],
        ),
        "blocking": None,
        "diagnostics": diagnostics,
    }


def first_execution_unit(
    execution_plan: dict[str, Any],
    task_graph: dict[str, Any],
) -> list[dict[str, Any]]:
    """Resolve the first ID-only unit into adapter assignment objects."""

    if execution_plan.get("state") != "dispatchable":
        return []
    units = execution_plan.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("dispatchable Execution Plan has no first unit")
    tasks = _task_index(task_graph)
    assignments = []
    for item in units[0].get("assignments") or []:
        task_id = str(item.get("task_id") or "")
        if task_id not in tasks:
            raise ValueError(f"Execution Plan references unknown task {task_id!r}")
        assignments.append(
            {
                "subtask": deepcopy(tasks[task_id]),
                "agent_id": str(item.get("agent_id")),
            }
        )
    if not assignments:
        raise ValueError("dispatchable Execution Plan has an empty first unit")
    return assignments


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a complete resource-aware Execution Plan."
    )
    parser.add_argument("--task-graph", type=Path, required=True)
    parser.add_argument("--agent-states", type=Path, required=True)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--scene-context", type=Path)
    parser.add_argument("--task-graph-version", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--disable-qwen", action="store_true")
    parser.add_argument("--qwen-model-path", default=DEFAULT_PLANNING_MODEL_PATH)
    parser.add_argument("--qwen-conv-mode", default="v0_mmtag")
    parser.add_argument("--qwen-num-gpus", type=int, default=1)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=2048)
    parser.add_argument("--allocation-max-attempts", type=int, default=3)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    agent_payload = load_json(args.agent_states)
    agent_states = (
        agent_payload.get("agents")
        if isinstance(agent_payload, dict)
        else agent_payload
    )
    plan = build_execution_plan(
        task_graph=args.task_graph,
        agent_states=agent_states,
        scene_context=(
            load_json(args.scene_context) if args.scene_context is not None else None
        ),
        progress=load_json(args.progress) if args.progress is not None else None,
        task_graph_version=args.task_graph_version,
        use_qwen=not args.disable_qwen,
        qwen_model_path=args.qwen_model_path,
        qwen_conv_mode=args.qwen_conv_mode,
        qwen_num_gpus=args.qwen_num_gpus,
        qwen_max_new_tokens=args.qwen_max_new_tokens,
        allocation_max_attempts=args.allocation_max_attempts,
        diagnostics_output_dir=(
            args.output.parent / "allocation_planning"
            if args.output
            else None
        ),
    )
    output = args.output or args.task_graph.with_name("execution_plan.json")
    save_json(plan, output)

    print(f"Execution Plan state: {plan['state']}")
    print(f"Time units: {len(plan['units'])}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
