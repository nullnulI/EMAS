"""
Hybrid decentralized-centralized embodied multi-agent decision loop.

This script is the Hybrid runtime entry point around the existing EMAS
planning/datagen components. The loop follows one explicit planning boundary:

1. build/load the scene graph and immutable semantic Task Graph
2. pass the complete remaining semantic graph, agent states, and current scene
   context to the Execution Planner
3. validate a complete Execution Plan, then dispatch only its first unit
4. receive execution feedback and re-observe the world
5. update progress and rebuild an Execution Plan for the remaining semantic graph
6. terminate when every task is complete or a loop budget is reached

Three execution modes are provided:

- adapter: writes JSON messages to disk and optionally consumes a JSON result
  file. Its legacy object changes, scene observations, and relation deltas are
  merged through ``memory.graph_update``.
- ai2thor: directly calls the existing agents.skill_plan executor and updates
  the scene graph from AI2-THOR observations, similar to datagen/generation.py.
- task_service: sends each allocated subtask to agents/task_execution_server.py
  through POST /execute_task and maps the response back to an EMAS report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


EMAS_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = EMAS_ROOT.parent
CONCEPTGRAPH_ROOT = EMAS_ROOT / "memory" / "concept-graphs"
for path in (WORKSPACE_ROOT, EMAS_ROOT, CONCEPTGRAPH_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from planning.datagen import generation as gen
from action_contracts import ACTION_CONTRACTS
from placement_contracts import compatible_receptacles, placement_compatibility
from planning.extract_subgraph import parse_task_locally
from planning.task_graph import (
    TaskPlanningError,
    decompose_task_to_graph,
    infer_task_object_tags,
    rebuild_task_graph_views as rebuild_task_graph,
    save_task_graph,
    unique_preserve_order,
)
from planning.task_allocation import (
    TaskAllocationError,
    build_execution_plan,
    first_execution_unit,
)
from planning.utils.task_status import FAILURE, SUCCESS, WAIT_RETRY
from memory.graph_update import diff_objects, infer_relation_deltas_from_metadata, update_scenegraph_files


OBJECT_GROUNDING_ACTIONS = {*ACTION_CONTRACTS, "navigate"}
SEARCH_ACTIONS = {"find", "inspect"}
RELAY_RETRYABLE_GROUNDING_FAILURES = {
    "target_not_visible",
    "object_not_actionable",
    "missing_required_state",
    "semantic_target_unresolved",
}
RELAY_TERMINAL_PLACEMENT_FAILURES = {
    "incompatible_receptacle",
    "receptacle_closed",
    "receptacle_full",
    "no_valid_placement",
    "target_not_reachable",
    "unknown_put_failure",
    "put_transport_exhausted",
}
DEFAULT_AGENT_SKILLS = ["find", "inspect", "navigate", *ACTION_CONTRACTS]


def log(message: str) -> None:
    print(f"[hybrid_loop] {message}", flush=True)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_planning_failure(
    error: TaskPlanningError,
    run_dir: Path,
    args: argparse.Namespace,
) -> Path:
    """Persist a machine-readable planning failure before aborting execution."""

    path = run_dir / "planning_failure.json"
    payload = error.to_dict()
    payload.update(
        {
            "status": "failed",
            "task": str(getattr(args, "task", "")),
            "planning_model_path": str(getattr(args, "planning_model_path", "")),
            "planning_max_attempts": max(
                int(getattr(args, "planning_max_attempts", 3)), 1
            ),
        }
    )
    save_json(gen.json_safe(payload), path)
    log(f"planning failed; diagnostics saved to {path}")
    return path


def save_allocation_failure(
    error: TaskAllocationError,
    loop_dir: Path,
    args: argparse.Namespace,
) -> Path:
    """Persist allocation diagnostics before aborting the episode."""

    diagnostics = gen.json_safe(error.diagnostics)
    save_json(diagnostics, loop_dir / "allocation_diagnostics.json")
    path = loop_dir / "allocation_failure.json"
    payload = error.to_dict()
    payload.update({
        "status": "failed",
        "task": str(getattr(args, "task", "")),
        "planning_model_path": str(getattr(args, "planning_model_path", "")),
        "planning_max_new_tokens": max(
            int(getattr(args, "planning_max_new_tokens", 2048)),
            1,
        ),
        "allocation_max_attempts": max(
            int(getattr(args, "planning_max_attempts", 3)),
            1,
        ),
    })
    save_json(gen.json_safe(payload), path)
    log(f"allocation failed; diagnostics saved to {path}")
    return path


def should_save_intermediates(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "save_intermediates", True))


def save_intermediate_json(data: Any, path: Path, args: argparse.Namespace) -> None:
    if should_save_intermediates(args):
        save_json(data, path)


def parse_json_object_from_text(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first:last + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def explain_exhaustive_runtime_failure(
    planning_chat: Any,
    *,
    parent_task: str,
    task_status: dict[str, Any],
    agent_states: list[dict[str, Any]],
) -> dict[str, Any]:
    """Ask Planning to explain a proved terminal failure without changing the graph."""

    system_prompt = (
        "You explain why an embodied task is impossible after the execution coordinator "
        "has exhaustively checked all known robots. Do not propose or modify a task graph. "
        "Use only the supplied evidence. Return JSON only."
    )
    payload = {
        "request": "explain_terminal_execution_failure",
        "parent_task": parent_task,
        "failed_subtask": deepcopy(task_status.get("subtask") or {}),
        "failure": {
            key: deepcopy(task_status.get(key))
            for key in (
                "failure_code",
                "reason",
                "candidate_evidence",
                "resource_recovery_history",
            )
        },
        "agent_state_summary": [
            {
                "agent_id": state.get("agent_id", state.get("robot_id")),
                "inventory": deepcopy(state.get("inventory") or []),
            }
            for state in agent_states
            if isinstance(state, dict)
        ],
        "output_schema": {
            "status": "impossible",
            "reason_code": (
                "resource_unavailable|no_executable_agent|target_unreachable|"
                "task_state_conflict|other"
            ),
            "reason": "concise evidence-based explanation",
            "evidence_refs": ["references to supplied candidate evidence"],
        },
    }
    if hasattr(planning_chat, "reset"):
        planning_chat.reset()
    if hasattr(planning_chat, "messages"):
        planning_chat.messages = [{"role": "system", "content": system_prompt}]
    raw_response = planning_chat(json.dumps(payload, ensure_ascii=False, indent=2))
    parsed = parse_json_object_from_text(str(raw_response or ""))
    allowed_codes = {
        "resource_unavailable",
        "no_executable_agent",
        "target_unreachable",
        "task_state_conflict",
        "other",
    }
    if (
        not isinstance(parsed, dict)
        or parsed.get("status") != "impossible"
        or parsed.get("reason_code") not in allowed_codes
        or not isinstance(parsed.get("reason"), str)
        or not parsed["reason"].strip()
        or not isinstance(parsed.get("evidence_refs"), list)
    ):
        raise RuntimeError("Planning failure explanation did not match the required JSON protocol")
    return parsed


def intermediate_file_ref(args: argparse.Namespace, path: Path) -> str | None:
    return str(path) if should_save_intermediates(args) else None


def now_run_name(scene_name: str) -> str:
    safe_scene = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in scene_name)
    return f"{safe_scene}_hybrid_{int(time.time())}"


def compact_task_text(task: dict[str, Any]) -> str:
    grounding = task.get("grounding") or {}
    parts = [
        str(task.get("name") or ""),
        str(task.get("description") or ""),
        str(task.get("action") or ""),
        " ".join(str(x) for x in grounding.get("object_tags") or []),
        " ".join(str(x) for x in grounding.get("relation_texts") or []),
    ]
    return " ".join(part for part in parts if part).strip()


def task_index(task_graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(task["id"]): deepcopy(task) for task in task_graph.get("flat_tasks") or []}


def initial_completed_task_ids(task_graph: dict[str, Any]) -> set[str]:
    completed: set[str] = set()
    for task in task_graph.get("flat_tasks") or []:
        if not isinstance(task, dict):
            continue
        runtime = task.get("runtime")
        if isinstance(runtime, dict) and bool(runtime.get("already_satisfied")):
            task_id = str(task.get("id") or "").strip()
            if task_id:
                completed.add(task_id)
    return completed


def subgraph_has_match(subgraph: dict[str, Any]) -> bool:
    return bool(subgraph.get("seed_nodes") or subgraph.get("nodes"))


def task_needs_scene_grounding(task: dict[str, Any]) -> bool:
    action = str(task.get("action") or "").lower()
    if action in SEARCH_ACTIONS:
        return False
    return action in OBJECT_GROUNDING_ACTIONS


def _grounding_entity_key(value: Any) -> str:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(value or "").replace("_", " "))
    return "".join(re.findall(r"[a-z0-9]+", separated.lower()))


def subgraph_has_entity(subgraph: dict[str, Any], entity: str) -> bool:
    wanted = _grounding_entity_key(entity)
    if not wanted:
        return False
    for node in [*(subgraph.get("seed_nodes") or []), *(subgraph.get("nodes") or [])]:
        if not isinstance(node, dict):
            continue
        values: list[Any] = [node.get("object_tag")]
        values.extend(node.get("possible_tags") or [])
        if any(_grounding_entity_key(value) == wanted for value in values if value not in (None, "")):
            return True
    return False


def missing_grounding_role(
    task: dict[str, Any],
    subgraph: dict[str, Any],
) -> tuple[str, list[str]] | None:
    grounding = task.get("grounding") if isinstance(task.get("grounding"), dict) else {}
    if str(task.get("action") or "").lower() != "place":
        return None
    roles = (
        ("source", grounding.get("source_object_tags") or [], grounding.get("source_object_ids") or []),
        ("destination", grounding.get("destination_object_tags") or [], grounding.get("destination_object_ids") or []),
    )
    for role, raw_tags, object_ids in roles:
        tags = unique_preserve_order([str(tag) for tag in raw_tags if str(tag).strip()])
        if tags and not object_ids and not any(subgraph_has_entity(subgraph, tag) for tag in tags):
            return role, tags
    return None


def task_has_complete_object_id_grounding(task: dict[str, Any]) -> bool:
    """Return whether every object role needed by the task has a stable ID."""

    grounding = task.get("grounding") if isinstance(task.get("grounding"), dict) else {}
    action = str(task.get("action") or "").lower()
    if action == "place":
        roles = (
            (grounding.get("source_object_tags") or [], grounding.get("source_object_ids") or []),
            (
                grounding.get("destination_object_tags") or [],
                grounding.get("destination_object_ids") or [],
            ),
        )
        return all(
            bool(object_ids) and (not tags or len(object_ids) >= len(tags))
            for tags, object_ids in roles
        )

    object_ids = grounding.get("object_ids") or grounding.get("source_object_ids") or []
    object_tags = grounding.get("object_tags") or grounding.get("source_object_tags") or []
    return bool(object_ids) and (not object_tags or len(object_ids) >= len(object_tags))


def task_requires_runtime_find(task: dict[str, Any], subgraph: dict[str, Any]) -> bool:
    """Return whether an object task needs a search before another execution attempt.

    A ConceptGraph match is semantic evidence only.  Once the executor reports
    that the target is not currently actionable, that runtime evidence takes
    precedence over the cached semantic match.
    """

    if not task_needs_scene_grounding(task):
        return False
    if task_has_complete_object_id_grounding(task):
        return False
    if missing_grounding_role(task, subgraph) is not None:
        return True
    grounding = task.get("grounding") or {}
    if str(grounding.get("execution_status") or "").lower() == "unresolved":
        return True
    if str(grounding.get("execution_failure_code") or "") in RELAY_RETRYABLE_GROUNDING_FAILURES:
        return True
    return not subgraph_has_match(subgraph)


def find_insertions_for_task(task_graph: dict[str, Any], source_task_id: str) -> list[dict[str, Any]]:
    return [
        task
        for task in task_graph.get("flat_tasks") or []
        if str((task.get("runtime") or {}).get("source_task_id")) == source_task_id
        and str(task.get("action") or "").lower() in SEARCH_ACTIONS
    ]


def next_find_task_id(task_graph: dict[str, Any], source_task_id: str) -> str:
    existing_ids = {str(task.get("id")) for task in task_graph.get("flat_tasks") or []}
    clean_source = "".join(ch if ch.isalnum() else "_" for ch in source_task_id)
    for index in range(1, 1000):
        candidate = f"F_{clean_source}_{index:02d}"
        if candidate not in existing_ids:
            return candidate
    raise RuntimeError(f"could not allocate a find task id for {source_task_id}")


def infer_object_tags_for_task(task: dict[str, Any]) -> list[str]:
    grounding = task.get("grounding") or {}
    tags = [str(tag) for tag in grounding.get("object_tags") or [] if str(tag).strip()]
    if tags:
        return unique_preserve_order(tags)
    return infer_task_object_tags(compact_task_text(task), limit=6)


def infer_actionable_root_action(task_text: str) -> str | None:
    """Infer only task-level interaction actions that must not collapse to find-only."""

    text = f" {str(task_text or '').lower()} "
    if " close " in text or " closes " in text or " shut " in text:
        return "close"
    if " open " in text or " opens " in text:
        return "open"
    if " pick up " in text or " pickup " in text or " grab " in text or " take " in text:
        return "pick"
    if " put " in text or " place " in text or " set " in text:
        return "place"
    return None


def repair_initial_find_only_task_graph(task_graph: dict[str, Any], root_task: str) -> dict[str, Any]:
    """Restore an interaction task when initial planning returned only a find node.

    Some memory states lack the requested object. In that case the planner can
    legally create a search node, but the search result should be a prerequisite
    for the original interaction, not the whole task. This repair keeps the
    search/find behavior and appends the original root task behind it.
    """

    tasks = [deepcopy(task) for task in task_graph.get("flat_tasks") or [] if isinstance(task, dict)]
    if len(tasks) != 1:
        return task_graph

    find_task = tasks[0]
    if str(find_task.get("action") or "").lower() not in SEARCH_ACTIONS:
        return task_graph

    root_action = infer_actionable_root_action(root_task)
    if not root_action:
        return task_graph

    source_task_id = str(find_task.get("id") or "T1")
    execute_id = source_task_id
    find_id = source_task_id
    if not find_id.startswith("F_"):
        find_id = f"F_{source_task_id}_01"
    if execute_id.startswith("F_"):
        execute_id = "T1"

    object_tags = infer_object_tags_for_task(find_task)
    find_task["id"] = find_id
    find_task.setdefault("name", f"Find target for {execute_id}")
    find_task["action"] = "find"
    find_task.setdefault("description", f"Search/inspect the environment before executing {execute_id}: {root_task}")
    find_task["depends_on"] = unique_preserve_order([str(dep) for dep in find_task.get("depends_on") or []])
    find_grounding = find_task.setdefault("grounding", {})
    find_grounding.setdefault("object_tags", object_tags)
    find_grounding.setdefault("status", "unresolved")
    find_grounding.setdefault("recovery", "search_visible_scene_then_update_scene_graph")
    find_runtime = find_task.setdefault("runtime", {})
    find_runtime.setdefault("inserted", True)
    find_runtime.setdefault("source_task_id", execute_id)
    find_runtime.setdefault("insert_reason", "initial task graph collapsed interaction to find-only")

    execute_task = {
        "id": execute_id,
        "name": f"Execute task {execute_id}",
        "description": str(root_task or "").strip() or compact_task_text(find_task),
        "action": root_action,
        "grounding": {
            "node_ids": [],
            "object_tags": object_tags,
            "relation_texts": list(find_grounding.get("relation_texts") or []),
            "status": "unresolved",
            "missing_reason": "initial task graph contained only a search node",
            "recovery": "wait_for_initial_find_task",
        },
        "depends_on": [find_id],
        "termination_check": "The original user-requested interaction has been executed successfully.",
        "runtime": {
            "recovered_from_initial_find_only": True,
            "source_find_task_id": find_id,
        },
    }

    repaired = deepcopy(task_graph)
    repaired["reasoning_summary"] = (
        str(repaired.get("reasoning_summary") or "").strip()
        + " Initial find-only graph was repaired to preserve the original interaction task."
    ).strip()
    repaired["flat_tasks"] = [find_task, execute_task]
    return rebuild_task_graph(repaired)


def apply_scene_catalog_selectors(
    planner_task_graph: dict[str, Any],
    object_catalog: Any,
) -> dict[str, Any]:
    """Validate planner-authored selectors and bind scene instances generically."""

    from planning.scene_goal_compiler import expand_scene_catalog_task_graph

    expanded = expand_scene_catalog_task_graph(planner_task_graph, object_catalog)
    diagnostics = (expanded.get("planner_diagnostics") or {}).get("selector_expansion") or {}
    if diagnostics.get("status") == "expanded":
        log(
            "expanded validated planner selectors against the scene catalogue "
            f"({len(expanded.get('flat_tasks') or [])} tasks)"
        )
    elif diagnostics.get("status") == "rejected":
        reason = str(diagnostics.get("reason") or "selector expansion was rejected")
        raise TaskPlanningError(
            f"scene selector validation failed: {reason}",
            diagnostics={
                "status": "failed",
                "stage": "selector_expansion",
                "selector_expansion": deepcopy(diagnostics),
                "planner_diagnostics": deepcopy(
                    expanded.get("planner_diagnostics") or {}
                ),
            },
        )
    return expanded


def insert_find_before_task(
    task_graph: dict[str, Any],
    task: dict[str, Any],
    *,
    reason: str,
    max_find_insertions: int,
    completed: set[str] | None = None,
    grounding_role: str | None = None,
    object_tags_override: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Insert/reuse a find task before an ungrounded task and return dispatch task."""

    completed = completed or set()
    source_task_id = str(task["id"])
    existing = find_insertions_for_task(task_graph, source_task_id)
    if grounding_role is not None:
        existing = [
            item for item in existing
            if str((item.get("runtime") or {}).get("grounding_role") or "") == grounding_role
        ]
    unfinished_existing = [
        item
        for item in existing
        if str(item.get("id")) not in completed
        and str(item.get("status") or "").lower() not in {"success", "completed"}
    ]
    if unfinished_existing:
        return task_graph, deepcopy(unfinished_existing[-1]), False

    if len(existing) >= max_find_insertions:
        # Keep the original task in circulation after the search budget. The
        # downstream executor/status checker can decide whether to retry/fail.
        task_copy = deepcopy(task)
        grounding = task_copy.setdefault("grounding", {})
        grounding["status"] = "unresolved"
        grounding["missing_reason"] = reason
        grounding["recovery"] = "search_budget_exhausted_continue_with_agent_feedback"
        return task_graph, task_copy, False

    indexed = task_index(task_graph)
    original = indexed[source_task_id]
    original_deps = unique_preserve_order([str(dep) for dep in original.get("depends_on") or []])
    find_id = next_find_task_id(task_graph, source_task_id)
    original_grounding = original.setdefault("grounding", {})
    preferred_agent_id = original_grounding.get("execution_preferred_agent_id")
    recovery_object_tags = [
        str(tag)
        for tag in original_grounding.get("execution_recovery_object_tags") or []
        if str(tag).strip()
    ]
    object_tags = (
        unique_preserve_order([str(tag) for tag in object_tags_override if str(tag).strip()])
        if object_tags_override
        else unique_preserve_order(recovery_object_tags)
        if recovery_object_tags
        else infer_object_tags_for_task(original)
    )
    relation_texts = (
        []
        if recovery_object_tags or object_tags_override
        else list(original_grounding.get("relation_texts") or [])
    )
    find_label = " and ".join(object_tags) or source_task_id
    find_task = {
        "id": find_id,
        "name": f"Find {find_label}",
        "description": (
            f"Locate the {find_label} required for the {grounding_role or 'object'} role of {source_task_id}."
        ),
        "action": "find",
        "grounding": {
            "node_ids": [],
            "object_tags": object_tags,
            "source_object_tags": object_tags if grounding_role != "destination" else [],
            "destination_object_tags": object_tags if grounding_role == "destination" else [],
            "source_node_ids": [],
            "destination_node_ids": [],
            "relation_texts": relation_texts,
            "status": "unresolved",
            "missing_reason": reason,
            "recovery": "search_visible_scene_then_update_scene_graph",
        },
        "depends_on": original_deps,
        "termination_check": "A plausible target object is observed and can be added to the scene graph.",
        "runtime": {
            "inserted": True,
            "insert_reason": reason,
            "source_task_id": source_task_id,
            "grounding_role": grounding_role,
            "recovery_object_tags": object_tags if recovery_object_tags else [],
            "preferred_agent_id": (
                str(preferred_agent_id)
                if preferred_agent_id not in (None, "")
                else None
            ),
            "inserted_at": time.time(),
        },
    }

    original_grounding["status"] = "unresolved"
    original_grounding["missing_reason"] = reason
    original_grounding["recovery"] = "wait_for_inserted_find_task"
    original["depends_on"] = unique_preserve_order(original_deps + [find_id])

    new_flat_tasks = []
    inserted = False
    for item in task_graph.get("flat_tasks") or []:
        if str(item.get("id")) == source_task_id:
            new_flat_tasks.append(find_task)
            new_flat_tasks.append(original)
            inserted = True
        else:
            new_flat_tasks.append(deepcopy(item))
    if not inserted:
        new_flat_tasks.append(find_task)

    updated = deepcopy(task_graph)
    updated["flat_tasks"] = new_flat_tasks
    updated = rebuild_task_graph(updated)
    return updated, deepcopy(find_task), True


def extract_subgraph_for_task(
    scenegraph_info: dict[str, Any],
    task: dict[str, Any],
    output_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    query = compact_task_text(task) or str(task.get("id"))
    if should_save_intermediates(args):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return gen.extract_task_relevant_subgraph_to_file(
            scenegraph_info=scenegraph_info,
            task=query,
            output_path=output_path,
            args=args,
        )

    with tempfile.TemporaryDirectory(prefix="hybrid_subgraph_") as tmpdir:
        return gen.extract_task_relevant_subgraph_to_file(
            scenegraph_info=scenegraph_info,
            task=query,
            output_path=Path(tmpdir) / output_path.name,
            args=args,
        )


def merge_subgraphs_for_allocation(task_subgraphs: dict[str, dict[str, Any]], output_path: Path | None) -> dict[str, Any]:
    nodes_by_key: dict[str, dict[str, Any]] = {}
    edges_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    triples_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    seed_nodes = []

    for task_id, subgraph in task_subgraphs.items():
        for node in subgraph.get("seed_nodes") or []:
            if isinstance(node, dict):
                seed_nodes.append(deepcopy(node) | {"source_task_id": task_id})
        for node in subgraph.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            key = str(node.get("pruned_id", node.get("original_id", node.get("object_tag", len(nodes_by_key)))))
            nodes_by_key.setdefault(key, deepcopy(node))
        for edge in subgraph.get("edges") or []:
            if not isinstance(edge, dict):
                continue
            key = (str(edge.get("source")), str(edge.get("target")), str(edge.get("object_relation") or edge.get("normalized_relation")))
            edges_by_key.setdefault(key, deepcopy(edge))
        for triple in subgraph.get("triples") or []:
            if not isinstance(triple, dict):
                continue
            key = (str(triple.get("source")), str(triple.get("target")), str(triple.get("relation") or triple.get("text")))
            triples_by_key.setdefault(key, deepcopy(triple))

    merged = {
        "mode": "current_task_list",
        "task": "Merged task-relevant subgraph for current dispatch list.",
        "task_subgraph_ids": list(task_subgraphs),
        "seed_nodes": seed_nodes,
        "nodes": list(nodes_by_key.values()),
        "edges": list(edges_by_key.values()),
        "triples": list(triples_by_key.values()),
    }
    if output_path is not None:
        save_json(merged, output_path)
    return merged


def load_agent_skills(path: Path | None, agentnum: int) -> dict[str, list[str]]:
    if path is None:
        return {str(index): list(DEFAULT_AGENT_SKILLS) for index in range(agentnum)}
    payload = load_json(path)
    if isinstance(payload, list):
        return {str(index): [str(skill) for skill in payload] for index in range(agentnum)}
    if isinstance(payload, dict):
        return {
            str(agent_id): [str(skill) for skill in skills]
            for agent_id, skills in payload.items()
            if isinstance(skills, list)
        }
    raise ValueError(f"Unsupported agent skill JSON shape: {path}")


def attach_agent_skills(agent_states: list[dict[str, Any]], skills_by_agent: dict[str, list[str]]) -> list[dict[str, Any]]:
    enriched = []
    for index, state in enumerate(agent_states):
        item = deepcopy(state)
        agent_id = str(item.get("agent_id", index))
        item["agent_id"] = agent_id
        item.setdefault("skills", skills_by_agent.get(agent_id, list(DEFAULT_AGENT_SKILLS)))
        enriched.append(item)
    return enriched


def offline_agent_states(agentnum: int, skills_by_agent: dict[str, list[str]]) -> list[dict[str, Any]]:
    return [
        {
            "agent_id": str(index),
            "status": "idle",
            "skills": skills_by_agent.get(str(index), list(DEFAULT_AGENT_SKILLS)),
            "visible_objects": [],
            "held_objects": [],
        }
        for index in range(agentnum)
    ]


def allocation_blocking_statuses(
    execution_plan: dict[str, Any],
    task_graph: dict[str, Any],
) -> list[dict[str, Any]]:
    """Represent an Allocation block as Planning-compatible runtime facts."""

    blocking = (
        execution_plan.get("blocking")
        if isinstance(execution_plan.get("blocking"), dict)
        else {}
    )
    indexed = task_index(task_graph)
    task_ids = [
        str(task_id)
        for task_id in blocking.get("task_ids") or []
        if str(task_id) in indexed
    ]
    if not task_ids:
        task_ids = [
            str(task_id)
            for task_id in (execution_plan.get("diagnostics") or {}).get(
                "remaining_task_ids", []
            )
            if str(task_id) in indexed
        ]
    code = str(blocking.get("code") or "allocation_blocked")
    reason = f"Allocation could not build a dispatchable Execution Plan ({code})."
    return [
        {
            "subtask_id": task_id,
            "subtask": deepcopy(indexed[task_id]),
            "status": FAILURE,
            "failure_code": code,
            "reason": reason,
            "recommended_recovery": "replan_task_graph",
            "exhaustive": True,
            "trigger_stage": "allocation",
            "allocation_blocking": deepcopy(blocking),
        }
        for task_id in unique_preserve_order(task_ids)
    ]


def allocation_blocking_budget_key(execution_plan: dict[str, Any]) -> str:
    """Key the once-per-version budget by stable graph and blocking facts."""

    payload = {
        "task_graph_version": execution_plan.get("task_graph_version"),
        "graph_fingerprint": execution_plan.get("graph_fingerprint"),
        "blocking": execution_plan.get("blocking"),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"v{execution_plan.get('task_graph_version', 1)}:allocation:{digest}"


def assignments_to_agent_task_map(assignments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    agent_task_map: dict[str, dict[str, Any]] = {}
    for item in assignments:
        if item.get("agent_id") is None:
            continue
        agent_id = str(item["agent_id"])
        if agent_id in agent_task_map:
            raise ValueError(f"allocation unit assigns more than one task to agent {agent_id}")
        agent_task_map[agent_id] = deepcopy(item.get("subtask") or {})
    return agent_task_map




class CommunicationAdapter:
    def send_task_allocation(self, payload: dict[str, Any], loop_dir: Path) -> None:
        raise NotImplementedError

    def receive_execution_report(self, payload: dict[str, Any], loop_dir: Path) -> dict[str, Any]:
        raise NotImplementedError


class FileCommunicationAdapter(CommunicationAdapter):
    """JSON file adapter for a future external executor/agent group."""

    def __init__(
        self,
        *,
        result_file: Path | None = None,
        auto_success: bool = True,
    ) -> None:
        self.result_file = result_file
        self.auto_success = auto_success

    def send_task_allocation(self, payload: dict[str, Any], loop_dir: Path) -> None:
        save_json(payload, loop_dir / "communication_outbox.json")

    def receive_execution_report(self, payload: dict[str, Any], loop_dir: Path) -> dict[str, Any]:
        if self.result_file is not None and self.result_file.exists():
            report = load_json(self.result_file)
            save_json(report, loop_dir / "communication_inbox.json")
            return report

        if not self.auto_success:
            report = {
                "execution": {"completed_task_ids": [], "traces": []},
                "task_statuses": [
                    {
                        "subtask_id": task.get("id"),
                        "agent_id": agent_id,
                        "status": WAIT_RETRY,
                        "reason": "No external execution result was provided.",
                        "retry_count_before": 0,
                        "retry_count_after": 1,
                        "subtask": task,
                    }
                    for agent_id, task in (payload.get("agent_task_map") or {}).items()
                ],
                "agent_states": payload.get("agent_states") or [],
                "object_changes": [],
                "feedback": [],
            }
            save_json(report, loop_dir / "communication_inbox.json")
            return report

        traces = []
        statuses = []
        completed_task_ids = []
        for agent_id, task in (payload.get("agent_task_map") or {}).items():
            task_id = str(task.get("id"))
            completed_task_ids.append(task_id)
            traces.append(
                {
                    "subtask_id": task_id,
                    "subtask": task,
                    "agent_id": str(agent_id),
                    "executor": "file_adapter_auto_success",
                    "completed": True,
                    "success_count": 1,
                    "num_actions": 1,
                    "actions": [
                        {
                            "action": str(task.get("action") or "execute"),
                            "params": {},
                            "lastActionSuccess": True,
                            "errorMessage": "",
                        }
                    ],
                }
            )
            statuses.append(
                {
                    "subtask_id": task_id,
                    "agent_id": str(agent_id),
                    "status": SUCCESS,
                    "reason": "Auto-success adapter report.",
                    "retry_count_before": 0,
                    "retry_count_after": 0,
                    "subtask": task,
                }
            )

        report = {
            "execution": {
                "completed_task_ids": completed_task_ids,
                "traces": traces,
                "execution_time_seconds": 0.0,
                "macro_step_wall_time_seconds": 0.0,
            },
            "task_statuses": statuses,
            "agent_states": payload.get("agent_states") or [],
            "object_changes": [],
            "feedback": [{"type": "auto_success", "message": "No external executor configured."}],
        }
        save_json(report, loop_dir / "communication_inbox.json")
        return report


def _task_service_robot_id(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, bool):
            return default
        robot_id = int(value)
    except (TypeError, ValueError):
        return default
    return robot_id if robot_id >= 0 else default


def task_service_subtask_text(subtask: dict[str, Any], root_task: str | None) -> str:
    text = compact_task_text(subtask).strip() if isinstance(subtask, dict) else ""
    if text:
        return text
    return str(root_task or "").strip() or "execute assigned subtask"


def compact_task_service_scene_context(scene_context: dict[str, Any], limit: int = 20) -> dict[str, Any]:
    """Keep relay context useful while staying well below the HTTP request cap."""

    return {
        "nodes": [
            {
                key: node.get(key)
                for key in ("pruned_id", "object_tag", "caption", "possible_tags", "objectId", "ai2thor_object_id")
                if node.get(key) is not None
            }
            for node in (scene_context.get("nodes") or [])[:limit]
            if isinstance(node, dict)
        ],
        "triples": [
            {
                key: triple.get(key)
                for key in ("source", "target", "relation", "text")
                if triple.get(key) is not None
            }
            for triple in (scene_context.get("triples") or [])[:limit]
            if isinstance(triple, dict)
        ],
    }


def task_service_request_identifier(loop_index: Any, subtask_id: Any, ordinal: int) -> str:
    raw_id = str(subtask_id or f"task_{ordinal}").strip() or f"task_{ordinal}"
    safe_id = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in raw_id)
    return f"loop_{loop_index}_{safe_id}"


def namespaced_task_service_request_identifier(namespace: Any, loop_index: Any, subtask_id: Any, ordinal: int) -> str:
    identifier = task_service_request_identifier(loop_index, subtask_id, ordinal)
    raw_namespace = str(namespace or "").strip()
    if not raw_namespace:
        return identifier
    safe_namespace = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in raw_namespace)
    return f"{safe_namespace}_{identifier}"


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"task_service HTTP {exc.code}: {error_body[:1000]}") from exc
    except URLError as exc:
        raise RuntimeError(f"task_service unreachable: {exc.reason}") from exc
    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"task_service returned invalid JSON: {response_body[:1000]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("task_service returned non-object JSON")
    return parsed


def relay_closed_loop_result(response: dict[str, Any]) -> dict[str, Any]:
    result = response.get("result")
    if not isinstance(result, dict):
        return {}
    closed_loop = result.get("closed_loop_result")
    return closed_loop if isinstance(closed_loop, dict) else {}


def task_service_response_completed(response: dict[str, Any]) -> bool:
    closed_loop = relay_closed_loop_result(response)
    return response.get("status") == SUCCESS and closed_loop.get("status") == SUCCESS


def task_service_response_preserves_subtask_action(
    response: dict[str, Any], subtask: dict[str, Any]
) -> bool:
    """Reject successful relay responses that dropped an interaction action."""

    required_actions = {
        "open": "OpenObject",
        "close": "CloseObject",
        "shut": "CloseObject",
        "pick": "PickupObject",
        "pickup": "PickupObject",
        "place": "PutObject",
        "put": "PutObject",
        "toggle_on": "ToggleObjectOn",
        "turn_on": "ToggleObjectOn",
        "toggle_off": "ToggleObjectOff",
        "turn_off": "ToggleObjectOff",
        "clean": "CleanObject",
        "wash": "CleanObject",
        "slice": "SliceObject",
        "cut": "SliceObject",
        "drop": "DropHandObject",
        "push": "PushObject",
        "pull": "PullObject",
        "move_held": "MoveHeldObject",
        "break": "BreakObject",
        "cook": "CookObject",
        "fill": "FillObjectWithLiquid",
    }
    required = required_actions.get(str(subtask.get("action") or "").strip().lower())
    if required is None:
        return True
    normalization = response.get("task_normalization")
    # Older/external relay implementations may not return normalization
    # diagnostics. The managed service always does, so validate when present.
    if normalization is None:
        return True
    if not isinstance(normalization, dict):
        return False
    steps = normalization.get("intentSteps")
    return isinstance(steps, list) and any(
        isinstance(step, dict) and step.get("action") == required for step in steps
    )


def task_service_response_reason(response: dict[str, Any]) -> str:
    closed_loop = relay_closed_loop_result(response)
    pieces = []
    failure_code = closed_loop.get("failure_code") or response.get("failure_code")
    reason = closed_loop.get("reason") or response.get("reason") or response.get("error")
    if failure_code:
        pieces.append(str(failure_code))
    if reason:
        pieces.append(str(reason))
    if pieces:
        return ": ".join(pieces)
    status = response.get("status") or "unknown"
    return f"task_service returned status {status!r}"


def task_service_response_failure_code(response: dict[str, Any]) -> str | None:
    closed_loop = relay_closed_loop_result(response)
    value = closed_loop.get("failure_code") or response.get("failure_code")
    return str(value) if value not in (None, "") else None


def task_service_response_recovery_target_type(response: dict[str, Any]) -> str | None:
    """Return the object type that actually caused a relay execution failure."""

    closed_loop = relay_closed_loop_result(response)
    for container in (closed_loop, response):
        for key in ("recovery_target_type", "failed_object_type"):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    reason = closed_loop.get("reason") or response.get("reason")
    if not isinstance(reason, str):
        return None
    receptacle = re.search(
        r"target receptacle\s+['\"]([^'\"]{1,64})['\"]\s+is not visible",
        reason,
        flags=re.IGNORECASE,
    )
    if receptacle:
        return receptacle.group(1).strip()
    for container in (closed_loop, response):
        for key in ("requested_object_type", "object_type", "target_type"):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    # Relay execution failures consistently quote AI2-THOR object types, e.g.
    # "'Fridge' is not visible" or "no known robot is holding 'Bread'".
    for candidate in re.findall(r"['\"]([^'\"]{1,64})['\"]", reason):
        candidate = candidate.strip()
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9 _-]*", candidate):
            return candidate
    unquoted = re.search(
        r"\b([A-Z][A-Za-z0-9_-]{0,63})\s+is\s+"
        r"(?:not\s+visible|visible|not\s+actionable)\b",
        reason,
    )
    if unquoted:
        return unquoted.group(1)
    return None


def task_service_response_recovery_agent_id(response: dict[str, Any]) -> str | None:
    """Return the robot that should retain ownership during recovery."""

    closed_loop = relay_closed_loop_result(response)
    for container in (closed_loop, response):
        for key in ("recovery_robot_id", "recovery_agent_id", "executor_robot_id"):
            value = container.get(key)
            if isinstance(value, int) and value >= 0:
                return str(value)
            if isinstance(value, str) and value.strip().isdigit():
                return str(int(value.strip()))

    reason = closed_loop.get("reason") or response.get("reason")
    if not isinstance(reason, str):
        return None
    holder_needing_target = re.search(
        r"robot\s+(\d+)\s*:\s*target receptacle\s+['\"][^'\"]+['\"]\s+is not visible",
        reason,
        flags=re.IGNORECASE,
    )
    return holder_needing_target.group(1) if holder_needing_target else None


def blocking_agent_pair(reason: Any) -> tuple[str, str] | None:
    """Return (blocking_agent_id, blocked_agent_id) from AI2-THOR failures."""

    if not isinstance(reason, str):
        return None
    match = re.search(
        r"\bAgent\s+(\d+)\s+is\s+blocking\s+Agent\s+(\d+)\b",
        reason,
        flags=re.IGNORECASE,
    )
    return (match.group(1), match.group(2)) if match else None


def _object_type_from_change(change: dict[str, Any], object_id: Any) -> str | None:
    for key in ("objectType", "object_type", "type"):
        value = change.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(object_id, str) and "|" in object_id:
        return object_id.split("|", 1)[0]
    return None


def _relay_execution_object_change(change: dict[str, Any]) -> dict[str, Any] | None:
    object_id = change.get("objectId") or change.get("object_id") or change.get("id")
    if not isinstance(object_id, str) or not object_id:
        return None
    object_type = _object_type_from_change(change, object_id)
    if not object_type:
        return None

    converted: dict[str, Any] = {
        "objectId": object_id,
        "objectType": object_type,
        "source": "task_execution_service_execute_action",
        "last_reported_state": deepcopy(change),
    }
    position = change.get("position")
    after = change.get("after")
    if position is None and isinstance(after, dict):
        position = after.get("position")
    if position is not None:
        converted["position"] = deepcopy(position)

    for key in (
        "source_action",
        "action_index",
        "robot_id",
        "state_changed",
        "before",
        "after",
        "isOpen",
        "isPickedUp",
        "inInventory",
        "parentReceptacles",
        "receptacleObjectIds",
        "visible",
    ):
        if key in change:
            converted[key] = deepcopy(change[key])
    return converted


def task_service_response_object_changes(
    response: dict[str, Any],
    *,
    source_task_id: str | None = None,
    observer_robot_id: int | str | None = None,
) -> list[dict[str, Any]]:
    result = response.get("result") if isinstance(response.get("result"), dict) else {}
    changes: list[dict[str, Any]] = []
    result_robot_id = result.get("primary_robot_id")
    observer_id = observer_robot_id if observer_robot_id not in (None, "") else result_robot_id

    goto_summary = result.get("goto_result_summary") if isinstance(result.get("goto_result_summary"), dict) else {}
    target = goto_summary.get("target") if isinstance(goto_summary.get("target"), dict) else {}
    object_id = target.get("object_id") or target.get("objectId")
    object_type = target.get("object_type") or target.get("objectType")
    if object_id and object_type:
        goto_change: dict[str, Any] = {
            "objectId": object_id,
            "objectType": object_type,
            "source": "task_execution_service_goto",
        }
        position = target.get("position") or goto_summary.get("target_position")
        if position is not None:
            goto_change["position"] = deepcopy(position)
        goal_position = goto_summary.get("goal_position")
        if goal_position is not None:
            goto_change["observer_goal_position"] = deepcopy(goal_position)
        if source_task_id:
            goto_change["source_task_id"] = source_task_id
        if observer_id not in (None, ""):
            goto_change["observer_robot_id"] = observer_id
        changes.append(goto_change)

    execution_changes = result.get("execution_state_changes") if isinstance(result.get("execution_state_changes"), dict) else {}
    for item in execution_changes.get("object_changes") or []:
        if not isinstance(item, dict):
            continue
        converted = _relay_execution_object_change(item)
        if converted is not None:
            if source_task_id:
                converted["source_task_id"] = source_task_id
            item_observer = converted.get("robot_id", observer_id)
            if item_observer not in (None, ""):
                converted["observer_robot_id"] = item_observer
            changes.append(converted)
    for item in result.get("discovered_objects") or []:
        if not isinstance(item, dict):
            continue
        object_id = item.get("objectId") or item.get("object_id")
        object_type = _object_type_from_change(item, object_id)
        if not object_id or not object_type:
            continue
        discovered = {
            "objectId": str(object_id),
            "objectType": object_type,
            "source": "relay_find_discovery",
            "visible": bool(item.get("visible", True)),
        }
        for key in ("position", "distance", "visible_by_agent_ids", "robot_id"):
            if key in item:
                discovered[key] = deepcopy(item[key])
        if source_task_id:
            discovered["source_task_id"] = source_task_id
        discovered_observer = discovered.get("robot_id", observer_id)
        if discovered_observer not in (None, ""):
            discovered["observer_robot_id"] = discovered_observer
        changes.append(discovered)
    return changes


def task_service_response_summary(response: dict[str, Any]) -> dict[str, Any]:
    closed_loop = relay_closed_loop_result(response)
    result = response.get("result") if isinstance(response.get("result"), dict) else {}
    summary: dict[str, Any] = {
        "status": response.get("status"),
        "task_id": response.get("task_id"),
    }
    if response.get("dry_run") is not None:
        summary["dry_run"] = response.get("dry_run")
    if isinstance(closed_loop, dict) and closed_loop:
        summary["closed_loop_result"] = {
            key: closed_loop.get(key)
            for key in (
                "status", "strategy", "failure_code", "reason", "failed_step_index",
                "recovery_strategy", "source_object_id", "destination_object_id",
                "excluded_destination_object_ids", "recovery_history",
                "exhaustive", "candidate_evidence", "resource_recovery_history",
            )
            if key in closed_loop
        }
    if isinstance(response.get("task_normalization"), dict):
        summary["task_normalization"] = deepcopy(response["task_normalization"])
    for key in (
        "task_intent",
        "intent_steps",
        "task_intent_source",
        "task_intent_tool_call",
        "task_intent_tool_call_validation",
    ):
        if isinstance(result, dict) and result.get(key) is not None:
            summary[key] = deepcopy(result.get(key))
    if isinstance(result, dict) and isinstance(result.get("closed_loop_trace"), list):
        recovery_traces = []
        for trace in result.get("closed_loop_trace") or []:
            if not isinstance(trace, dict):
                continue
            relay_result = trace.get("relay_result") if isinstance(trace.get("relay_result"), dict) else {}
            if relay_result.get("strategy") == "put_holder_goto_receptacle" or trace.get("goto_result_summary"):
                recovery_traces.append(deepcopy(trace))
        if recovery_traces:
            summary["recovery_traces"] = recovery_traces
    if result.get("image_path") is not None:
        summary["image_path"] = result.get("image_path")
    if response.get("failure_code") is not None:
        summary["failure_code"] = response.get("failure_code")
    if response.get("reason") is not None:
        summary["reason"] = response.get("reason")
    if response.get("error") is not None:
        summary["error"] = response.get("error")
    if response.get("runtime_log") is not None:
        summary["runtime_log"] = response.get("runtime_log")
    if isinstance(response.get("post_agent_states"), list):
        summary["post_agent_states"] = deepcopy(response["post_agent_states"])
    if isinstance(response.get("post_state_errors"), list):
        summary["post_state_errors"] = deepcopy(response["post_state_errors"])
    return summary


def task_service_response_completion_agent_id(response: dict[str, Any], fallback: int | str) -> str:
    explicit_completion_agent_id = response.get("completion_agent_id")
    if explicit_completion_agent_id not in (None, ""):
        return str(explicit_completion_agent_id)
    result = response.get("result") if isinstance(response.get("result"), dict) else {}
    closed_loop = result.get("closed_loop_result")
    if isinstance(closed_loop, dict):
        completion_agent_id = closed_loop.get("completion_agent_id")
        if completion_agent_id not in (None, ""):
            return str(completion_agent_id)
    for trace in reversed(result.get("closed_loop_trace") or []):
        if not isinstance(trace, dict):
            continue
        executor = trace.get("executor_robot_id", trace.get("executor_agent_id"))
        if executor not in (None, ""):
            return str(executor)
        relay_result = trace.get("relay_result") if isinstance(trace.get("relay_result"), dict) else {}
        executor = relay_result.get("executor_robot_id", relay_result.get("executor_agent_id"))
        if executor not in (None, ""):
            return str(executor)
    primary = result.get("primary_robot_id")
    return str(primary if primary not in (None, "") else fallback)


class TaskExecutionServiceAdapter(CommunicationAdapter):
    """HTTP adapter from EMAS allocation units to agents/task_execution_server.py."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def send_task_allocation(self, payload: dict[str, Any], loop_dir: Path) -> None:
        save_json(payload, loop_dir / "communication_outbox.json")

    def receive_execution_report(self, payload: dict[str, Any], loop_dir: Path) -> dict[str, Any]:
        assignments = [item for item in payload.get("assignments") or [] if isinstance(item, dict)]
        started = time.perf_counter()
        completed_task_ids: list[str] = []
        traces: list[dict[str, Any]] = []
        statuses: list[dict[str, Any]] = []
        feedback: list[dict[str, Any]] = []
        object_changes: list[dict[str, Any]] = []
        latest_post_states_by_id: dict[str, dict[str, Any]] = {}

        for ordinal, assignment in enumerate(assignments, start=1):
            subtask = assignment.get("subtask") if isinstance(assignment.get("subtask"), dict) else {}
            subtask_id = str(subtask.get("id") or f"task_{ordinal}")
            agent_id = _task_service_robot_id(assignment.get("agent_id", 0), default=0)
            task_service_request = {
                "task_id": namespaced_task_service_request_identifier(getattr(self.args, "_task_service_namespace", ""), payload.get("loop_index", 0), subtask_id, ordinal),
                "task": task_service_subtask_text(subtask, str(payload.get("task") or "")),
                "parent_task": str(payload.get("task") or ""),
                "subtask": deepcopy(subtask),
                "scene_context": compact_task_service_scene_context(
                    payload.get("scene_context") if isinstance(payload.get("scene_context"), dict) else {}
                ),
                "primary_robot_id": agent_id,
                "dry_run": bool(getattr(self.args, "task_service_dry_run", False)),
                "relay_strategy": getattr(self.args, "task_service_relay_strategy", "agent"),
                "max_replan_steps": int(getattr(self.args, "task_service_max_replan_steps", 10)),
                "relay_agent_max_turns": int(getattr(self.args, "task_service_relay_agent_max_turns", 8)),
                "max_actions": int(getattr(self.args, "task_service_max_actions", 8)),
            }
            action_logger = getattr(self, "action_logger", None)
            if action_logger is not None and hasattr(action_logger, "bind"):
                action_logger.bind(
                    macro_step=int(payload.get("loop_index", 0)) + 1,
                    subtask_id=subtask_id,
                    agent_id=agent_id,
                )
            try:
                task_service_response = post_json(
                    str(getattr(self.args, "task_service_url")),
                    task_service_request,
                    float(getattr(self.args, "task_service_timeout", 300.0)),
                )
                completed = task_service_response_completed(task_service_response)
                action_preserved = task_service_response_preserves_subtask_action(
                    task_service_response, subtask
                )
                if completed and not action_preserved:
                    completed = False
                    task_service_response["failure_code"] = "semantic_action_dropped"
                    task_service_response["reason"] = (
                        "relay reported success after dropping the original structured action"
                    )
                reason = "Relay closed-loop task completed." if completed else task_service_response_reason(task_service_response)
                failure_code = None if completed else task_service_response_failure_code(task_service_response)
                status = (
                    SUCCESS
                    if completed
                    else FAILURE
                    if (
                        failure_code in RELAY_TERMINAL_PLACEMENT_FAILURES
                        or bool(relay_closed_loop_result(task_service_response).get("exhaustive"))
                    )
                    else WAIT_RETRY
                )
                recovery_target_type = (
                    None if completed else task_service_response_recovery_target_type(task_service_response)
                )
                recovery_agent_id = (
                    None if completed else task_service_response_recovery_agent_id(task_service_response)
                )
            except RuntimeError as exc:
                task_service_response = {"status": "failed", "error": str(exc)}
                completed = False
                status = WAIT_RETRY
                reason = str(exc)
                failure_code = "task_service_transport_error"
                recovery_target_type = None
                recovery_agent_id = None

            response_post_states = [
                item for item in task_service_response.get("post_agent_states") or [] if isinstance(item, dict)
            ]
            if response_post_states:
                for state in response_post_states:
                    state_id = state.get("agent_id", state.get("robot_id"))
                    if state_id not in (None, ""):
                        latest_post_states_by_id[str(state_id)] = state

            blocked_pair = blocking_agent_pair(reason)
            completion_agent_id = task_service_response_completion_agent_id(task_service_response, agent_id)
            closed_loop_failure = relay_closed_loop_result(task_service_response)
            excluded_destination_object_ids = [
                str(object_id)
                for object_id in closed_loop_failure.get("excluded_destination_object_ids") or []
                if str(object_id).strip()
            ]
            recovery_history = [
                deepcopy(item)
                for item in closed_loop_failure.get("recovery_history") or []
                if isinstance(item, dict)
            ]
            if completed:
                completed_task_ids.append(subtask_id)
                response_changes = task_service_response_object_changes(
                    task_service_response,
                    source_task_id=subtask_id,
                    observer_robot_id=completion_agent_id,
                )
                post_state = next(
                    (
                        item for item in response_post_states
                        if str(item.get("agent_id", item.get("robot_id"))) == completion_agent_id
                    ),
                    None,
                )
                if isinstance(post_state, dict):
                    post_agent = post_state.get("agent") if isinstance(post_state.get("agent"), dict) else {}
                    observer_position = post_agent.get("position") or post_state.get("position")
                    for change in response_changes:
                        if observer_position is not None:
                            change.setdefault("observer_position", deepcopy(observer_position))
                object_changes.extend(response_changes)
            else:
                feedback.append({"type": "task_service_not_completed", "subtask_id": subtask_id, "message": reason})
            statuses.append(
                {
                    "subtask_id": subtask_id,
                    "agent_id": str(agent_id),
                    "completion_agent_id": completion_agent_id if completed else None,
                    "status": status,
                    "reason": reason,
                    "failure_code": failure_code,
                    "exhaustive": bool(closed_loop_failure.get("exhaustive")),
                    "candidate_evidence": deepcopy(
                        closed_loop_failure.get("candidate_evidence") or []
                    ),
                    "resource_recovery_history": deepcopy(
                        closed_loop_failure.get("resource_recovery_history") or []
                    ),
                    "recoverable": (
                        failure_code in RELAY_RETRYABLE_GROUNDING_FAILURES
                        or failure_code == "task_service_transport_error"
                        or blocked_pair is not None
                    ),
                    "recommended_recovery": (
                        "replan_task_graph"
                        if failure_code in RELAY_TERMINAL_PLACEMENT_FAILURES
                        else "retry_original_task"
                        if failure_code in RELAY_RETRYABLE_GROUNDING_FAILURES
                        else "switch_agent"
                        if blocked_pair is not None
                        else None
                    ),
                    "recovery_target_type": recovery_target_type,
                    "excluded_destination_object_ids": excluded_destination_object_ids,
                    "recovery_history": recovery_history,
                    "recovery_agent_id": recovery_agent_id,
                    "semantic_resolution": deepcopy(
                        (task_service_response.get("task_normalization") or {}).get("semantic_resolution")
                    ) if isinstance(task_service_response.get("task_normalization"), dict) else None,
                    "blocked_by_agent_id": blocked_pair[0] if blocked_pair else None,
                    "blocked_agent_id": blocked_pair[1] if blocked_pair else None,
                    "subtask": deepcopy(subtask),
                }
            )
            traces.append(
                {
                    "subtask_id": subtask_id,
                    "agent_id": str(agent_id),
                    "completion_agent_id": completion_agent_id if completed else None,
                    "subtask": deepcopy(subtask),
                    "executor": "task_execution_service",
                    "completed": completed,
                    "task_service_request": task_service_request,
                    "task_service_response": task_service_response_summary(task_service_response),
                }
            )

        latest_post_agent_states = list(latest_post_states_by_id.values())
        report = {
            "execution": {
                "completed_task_ids": completed_task_ids,
                "traces": traces,
                "execution_time_seconds": time.perf_counter() - started,
                "macro_step_wall_time_seconds": time.perf_counter() - started,
            },
            "task_statuses": statuses,
            "agent_states": latest_post_agent_states or payload.get("agent_states") or [],
            "post_agent_states": latest_post_agent_states,
            "object_changes": object_changes,
            "feedback": feedback,
        }
        save_json(report, loop_dir / "communication_inbox.json")
        return report


class AI2ThorDirectAdapter(CommunicationAdapter):
    def __init__(self, controller: Any, args: argparse.Namespace) -> None:
        self.controller = controller
        self.args = args

    def send_task_allocation(self, payload: dict[str, Any], loop_dir: Path) -> None:
        save_json(payload, loop_dir / "communication_outbox.json")

    def receive_execution_report(self, payload: dict[str, Any], loop_dir: Path) -> dict[str, Any]:
        from agents.skill_plan import execute_allocated_skills

        assignments = payload.get("assignments") or []
        execution = execute_allocated_skills(
            self.controller,
            assignments,
            use_qwen=not self.args.disable_qwen and not self.args.disable_skill_qwen,
            qwen_model_path=self.args.planning_model_path,
            qwen_conv_mode=self.args.qwen_conv_mode,
            qwen_num_gpus=self.args.qwen_num_gpus,
            max_steps=self.args.skill_max_steps,
            complete_on_execute=self.args.skill_complete_on_execute,
        )
        report = {"execution": execution}
        save_json(report, loop_dir / "communication_inbox.json")
        return report


def statuses_from_adapter_report(report: dict[str, Any], assignments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(report.get("task_statuses"), list):
        return [deepcopy(item) for item in report["task_statuses"] if isinstance(item, dict)]

    completed_ids = {str(task_id) for task_id in (report.get("execution") or {}).get("completed_task_ids") or []}
    statuses = []
    for assignment in assignments:
        subtask = deepcopy(assignment.get("subtask") or {})
        task_id = str(subtask.get("id"))
        status = SUCCESS if task_id in completed_ids else WAIT_RETRY
        statuses.append(
            {
                "subtask_id": task_id,
                "agent_id": str(assignment.get("agent_id", 0)),
                "status": status,
                "reason": "Derived from adapter execution.completed_task_ids.",
                "subtask": subtask,
            }
        )
    return statuses


def apply_progress(
    task_statuses: list[dict[str, Any]],
    completed: set[str],
    failed: set[str],
    retry_counts: dict[str, int],
    max_task_retries: int | None = None,
) -> dict[str, Any]:
    completed_this_loop = []
    failed_this_loop = []
    wait_retry_this_loop = []

    for status in task_statuses:
        task_id = str(status.get("subtask_id"))
        state = str(status.get("status") or "").lower()
        if state in {SUCCESS, "completed", "complete", "success"}:
            completed.add(task_id)
            failed.discard(task_id)
            retry_counts.pop(task_id, None)
            completed_this_loop.append(task_id)
        elif state in {FAILURE, "failed", "failure"}:
            failed.add(task_id)
            retry_counts.pop(task_id, None)
            failed_this_loop.append(task_id)
        else:
            next_retry_count = retry_counts.get(task_id, 0) + 1
            retry_counts[task_id] = next_retry_count
            if max_task_retries is not None and next_retry_count >= max_task_retries:
                failed.add(task_id)
                retry_counts.pop(task_id, None)
                failed_this_loop.append(task_id)
                status["status"] = FAILURE
                status["reason"] = (
                    f"{status.get('reason') or 'task did not complete'}; "
                    f"retry budget exhausted ({next_retry_count}/{max_task_retries})"
                )
            else:
                wait_retry_this_loop.append(task_id)

    return {
        "completed_this_loop": completed_this_loop,
        "failed_this_loop": failed_this_loop,
        "wait_retry_this_loop": wait_retry_this_loop,
    }


def propagate_semantic_destination_binding(
    task_graph: dict[str, Any],
    resolution: Any,
) -> bool:
    if not isinstance(resolution, dict) or resolution.get("status") != "resolved":
        return False
    semantic_input = str(resolution.get("semantic_input") or "")
    chosen_type = str(resolution.get("chosen_type") or "")
    chosen_id = str(resolution.get("chosen_object_id") or "")
    if not semantic_input or not chosen_type or not chosen_id:
        return False
    changed = False
    for task in task_graph.get("flat_tasks") or []:
        if not isinstance(task, dict):
            continue
        grounding = task.get("grounding") if isinstance(task.get("grounding"), dict) else {}
        semantic_tags = grounding.get("semantic_destination_tags") or grounding.get("destination_object_tags") or []
        if not any(_grounding_entity_key(tag) == _grounding_entity_key(semantic_input) for tag in semantic_tags):
            continue
        bound_ids = [
            str(object_id)
            for object_id in grounding.get("destination_object_ids") or []
            if str(object_id).strip()
        ]
        if bound_ids and chosen_id not in bound_ids:
            continue
        source_tags = [str(tag) for tag in grounding.get("source_object_tags") or [] if str(tag).strip()]
        grounding["semantic_destination_tags"] = [semantic_input]
        grounding["destination_object_tags"] = [chosen_type]
        grounding["destination_object_ids"] = [chosen_id]
        grounding["object_tags"] = unique_preserve_order([*source_tags, chosen_type])
        task["grounding"] = grounding
        changed = True
    return changed


def _selector_matches_object_type(
    grounding: dict[str, Any],
    role: str,
    object_type: str,
    *,
    require_one: bool = False,
) -> bool:
    selector = grounding.get(f"{role}_selector")
    if not isinstance(selector, dict):
        return False
    if require_one and str(selector.get("quantifier") or "").lower() != "one":
        return False
    wanted = _grounding_entity_key(object_type)
    return bool(wanted) and any(
        _grounding_entity_key(candidate) == wanted
        for candidate in selector.get("object_types") or []
    )


def _selector_object_ids(grounding: dict[str, Any], role: str) -> list[str]:
    return [
        str(object_id)
        for object_id in grounding.get(f"{role}_object_ids") or []
        if str(object_id).strip()
    ]


def _record_execution_selector_binding(
    task: dict[str, Any],
    *,
    source_task_id: str,
    role: str,
    object_id: str,
    object_type: str,
) -> None:
    runtime = task.setdefault("runtime", {})
    runtime["selector_binding"] = {
        "source_task_id": source_task_id,
        "source_action": "OpenObject",
        "object_id": object_id,
        "object_type": object_type,
        "role": role,
        "source": "execution_feedback",
    }


def _bind_selector_role_from_execution(
    task: dict[str, Any],
    *,
    source_task_id: str,
    role: str,
    object_id: str,
    object_type: str,
) -> tuple[bool, bool]:
    """Return (compatible, changed) for a quantifier=one selector role."""

    grounding = task.get("grounding") if isinstance(task.get("grounding"), dict) else {}
    if not _selector_matches_object_type(grounding, role, object_type, require_one=True):
        return False, False
    bound_ids = _selector_object_ids(grounding, role)
    if bound_ids:
        return object_id in bound_ids, False
    grounding[f"{role}_object_ids"] = [object_id]
    task["grounding"] = grounding
    _record_execution_selector_binding(
        task,
        source_task_id=source_task_id,
        role=role,
        object_id=object_id,
        object_type=object_type,
    )
    return True, True


def propagate_open_instance_binding(
    task_graph: dict[str, Any],
    *,
    source_task_id: str,
    object_id: str,
    object_type: str,
) -> bool:
    """Bind one opened receptacle through its dependent Place/Close chain."""

    tasks = [task for task in task_graph.get("flat_tasks") or [] if isinstance(task, dict)]
    indexed = {str(task.get("id")): task for task in tasks}
    source_task = indexed.get(source_task_id)
    if source_task is None:
        return False
    compatible, changed = _bind_selector_role_from_execution(
        source_task,
        source_task_id=source_task_id,
        role="source",
        object_id=object_id,
        object_type=object_type,
    )
    if not compatible:
        return False

    successors: dict[str, list[str]] = {}
    for task in tasks:
        task_id = str(task.get("id"))
        for dependency in task.get("depends_on") or []:
            successors.setdefault(str(dependency), []).append(task_id)

    queue = list(successors.get(source_task_id, []))
    visited: set[str] = set()
    while queue:
        task_id = queue.pop(0)
        if task_id in visited:
            continue
        visited.add(task_id)
        task = indexed.get(task_id)
        if task is None:
            continue
        grounding = task.get("grounding") if isinstance(task.get("grounding"), dict) else {}
        action = str(task.get("action") or "").lower()

        if action == "open" and _selector_matches_object_type(grounding, "source", object_type):
            continue

        matching_roles = [
            role
            for role in ("source", "destination")
            if _selector_matches_object_type(grounding, role, object_type)
        ]
        if any(
            _selector_object_ids(grounding, role)
            and object_id not in _selector_object_ids(grounding, role)
            for role in matching_roles
        ):
            continue

        role = None
        if action in {"place", "put"}:
            role = "destination"
        elif action == "close":
            role = "source"
        if role is not None and _selector_matches_object_type(grounding, role, object_type):
            compatible, role_changed = _bind_selector_role_from_execution(
                task,
                source_task_id=source_task_id,
                role=role,
                object_id=object_id,
                object_type=object_type,
            )
            if not compatible:
                continue
            changed = changed or role_changed

        queue.extend(successors.get(task_id, []))

    return changed


def successful_open_change_for_task(
    task_id: str,
    object_changes: list[dict[str, Any]],
) -> tuple[str, str] | None:
    for change in reversed(object_changes):
        if not isinstance(change, dict):
            continue
        if str(change.get("source_task_id") or "") != task_id:
            continue
        if str(change.get("source_action") or "").lower() != "openobject":
            continue
        if change.get("isOpen") is not True:
            continue
        object_id = change.get("objectId") or change.get("object_id")
        object_type = _object_type_from_change(change, object_id)
        if object_id and object_type:
            return str(object_id), str(object_type)
    return None


def apply_execution_feedback_to_task_graph(
    task_graph: dict[str, Any],
    task_statuses: list[dict[str, Any]],
    object_changes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Feed relay execution grounding back into the runtime task graph."""

    updated = deepcopy(task_graph)
    indexed = {
        str(task.get("id")): task
        for task in updated.get("flat_tasks") or []
        if isinstance(task, dict)
    }
    discovered_ids = [
        str(change.get("objectId") or change.get("object_id"))
        for change in (object_changes or [])
        if isinstance(change, dict) and (change.get("objectId") or change.get("object_id"))
    ]
    changed = False

    for status in task_statuses:
        task_id = str(status.get("subtask_id"))
        task = indexed.get(task_id)
        if task is None:
            continue
        state = str(status.get("status") or "").lower()
        action = str(task.get("action") or "").lower()
        runtime = task.get("runtime") if isinstance(task.get("runtime"), dict) else {}

        if propagate_semantic_destination_binding(updated, status.get("semantic_resolution")):
            changed = True

        if state in {SUCCESS, "completed", "complete"} and action == "open":
            opened = successful_open_change_for_task(task_id, object_changes or [])
            if opened is not None:
                opened_object_id, opened_object_type = opened
                if propagate_open_instance_binding(
                    updated,
                    source_task_id=task_id,
                    object_id=opened_object_id,
                    object_type=opened_object_type,
                ):
                    changed = True

        if state in {SUCCESS, "completed", "complete"} and action in SEARCH_ACTIONS:
            source_task_id = str(runtime.get("source_task_id") or "")
            source_task = indexed.get(source_task_id)
            if source_task is not None:
                grounding = source_task.setdefault("grounding", {})
                grounding["execution_status"] = "discovered"
                grounding.pop("execution_failure_code", None)
                grounding.pop("execution_recovery_object_tags", None)
                grounding.pop("missing_reason", None)
                grounding["recovery"] = "none"
                if discovered_ids:
                    grounding["object_ids"] = list(dict.fromkeys(discovered_ids))
                changed = True
            continue

        failure_code = str(status.get("failure_code") or "")
        blocked_pair = blocking_agent_pair(status.get("reason"))
        if state not in {SUCCESS, "completed", "complete"} and blocked_pair is not None:
            runtime = task.setdefault("runtime", {})
            blocked_agent_id = str(
                status.get("blocked_agent_id")
                or blocked_pair[1]
                or status.get("agent_id")
                or ""
            )
            avoided = unique_preserve_order([
                *[str(agent_id) for agent_id in runtime.get("avoid_agent_ids") or []],
                blocked_agent_id,
            ])
            runtime["avoid_agent_ids"] = avoided
            runtime["last_blocked_by_agent_id"] = str(
                status.get("blocked_by_agent_id") or blocked_pair[0]
            )
            runtime["agent_switch_reason"] = str(status.get("reason") or "")
            changed = True

        if action == "place" and failure_code in RELAY_TERMINAL_PLACEMENT_FAILURES:
            runtime = task.setdefault("runtime", {})
            runtime["last_execution_failure_code"] = failure_code
            runtime["last_execution_failure_reason"] = str(status.get("reason") or failure_code)
            runtime["retry_strategy"] = "replan_task_graph"
            runtime["excluded_destination_object_ids"] = unique_preserve_order([
                *[str(object_id) for object_id in runtime.get("excluded_destination_object_ids") or []],
                *[str(object_id) for object_id in status.get("excluded_destination_object_ids") or []],
            ])
            runtime["placement_recovery_history"] = [
                deepcopy(item)
                for item in status.get("recovery_history") or []
                if isinstance(item, dict)
            ]
            changed = True

        if action not in SEARCH_ACTIONS and failure_code in RELAY_RETRYABLE_GROUNDING_FAILURES:
            runtime = task.setdefault("runtime", {})
            runtime["last_execution_failure_code"] = failure_code
            runtime["last_execution_failure_reason"] = str(
                status.get("reason") or failure_code
            )
            runtime["retry_strategy"] = "retry_original_task"
            recovery_target_type = status.get("recovery_target_type")
            if isinstance(recovery_target_type, str) and recovery_target_type.strip():
                runtime["last_recovery_target_type"] = recovery_target_type.strip()
            recovery_agent_id = status.get("recovery_agent_id")
            if recovery_agent_id in (None, ""):
                recovery_agent_id = status.get("agent_id")
            if recovery_agent_id not in (None, ""):
                grounding = task.setdefault("grounding", {})
                grounding["execution_preferred_agent_id"] = str(recovery_agent_id)
            changed = True

    return rebuild_task_graph(updated) if changed else task_graph


def execution_status_requires_task_graph_replan(status: dict[str, Any]) -> bool:
    """Return whether runtime recovery (execution or allocation) needs Planning."""

    state = str(status.get("status") or "").lower()
    return (
        state in {FAILURE, "failed", "failure"}
        and str(status.get("recommended_recovery") or "") == "replan_task_graph"
    )


def compact_replanning_task(task: Any) -> dict[str, Any] | None:
    if not isinstance(task, dict):
        return None
    return {
        key: deepcopy(task.get(key))
        for key in (
            "id", "name", "description", "action", "action_args", "grounding",
            "depends_on", "termination_check", "runtime",
        )
        if task.get(key) not in (None, "", [], {})
    }


def compact_replanning_agent_states(agent_states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for index, state in enumerate(agent_states):
        if not isinstance(state, dict):
            continue
        nested_agent = state.get("agent") if isinstance(state.get("agent"), dict) else {}
        compact.append({
            "agent_id": str(state.get("agent_id", state.get("robot_id", index))),
            "robot_id": state.get("robot_id"),
            "position": deepcopy(state.get("position") or nested_agent.get("position") or {}),
            "rotation": deepcopy(state.get("rotation") or nested_agent.get("rotation") or {}),
            "inventory": deepcopy(state.get("inventoryObjects") or state.get("inventory") or []),
            "visible_objects": deepcopy((state.get("visible_objects") or [])[:24]),
            "skills": list(state.get("skills") or []),
        })
    return compact


def build_replanning_execution_context(
    *,
    root_task: str,
    report: dict[str, Any],
    task_statuses: list[dict[str, Any]],
    task_graph: dict[str, Any],
    completed: set[str],
    failed: set[str],
    agent_states: list[dict[str, Any]],
) -> dict[str, Any]:
    """Project a complete adapter report into bounded Planning input."""

    indexed = task_index(task_graph)
    replanning_statuses = [
        status for status in task_statuses
        if execution_status_requires_task_graph_replan(status)
    ]
    successful_this_loop = [
        str(status.get("subtask_id"))
        for status in task_statuses
        if str(status.get("status") or "").lower()
        in {SUCCESS, "completed", "complete", "success"}
    ]
    completed_ids = sorted(completed | set(successful_this_loop))
    return {
        "mode": "runtime_replanning",
        "root_task": root_task,
        "execution_report": deepcopy(report),
        "failed_subtasks": [
            {
                "subtask": compact_replanning_task(
                    status.get("subtask") or indexed.get(str(status.get("subtask_id")))
                ),
                **{
                    key: deepcopy(status.get(key))
                    for key in (
                        "subtask_id", "agent_id", "completion_agent_id", "status",
                        "failure_code", "reason", "recommended_recovery", "exhaustive",
                        "excluded_destination_object_ids", "recovery_history",
                        "candidate_evidence", "resource_recovery_history",
                        "semantic_resolution", "recovery_target_type", "recovery_agent_id",
                        "trigger_stage", "allocation_blocking",
                    )
                    if status.get(key) not in (None, "", [], {})
                },
            }
            for status in replanning_statuses
        ],
        "successful_subtask_ids_this_loop": successful_this_loop,
        "completed_tasks": [
            compact_replanning_task(indexed[task_id])
            for task_id in completed_ids
            if task_id in indexed
        ],
        "previously_failed_task_ids": sorted(failed),
        "remaining_tasks_before_replan": [
            compact_replanning_task(task)
            for task in task_graph.get("flat_tasks") or []
            if str(task.get("id")) not in set(completed_ids) | failed
        ],
        "agent_states": compact_replanning_agent_states(agent_states),
        "object_changes": deepcopy([
            item for item in report.get("object_changes") or [] if isinstance(item, dict)
        ]),
        "execution_summary": {
            "completed_task_ids": deepcopy(
                (report.get("execution") or {}).get("completed_task_ids") or []
            ),
            "trace_summaries": [
                {
                    key: deepcopy(trace.get(key))
                    for key in (
                        "subtask_id", "agent_id", "completion_agent_id", "executor",
                        "completed", "task_service_response",
                    )
                    if trace.get(key) not in (None, "", [], {})
                }
                for trace in (report.get("execution") or {}).get("traces") or []
                if isinstance(trace, dict)
            ],
        },
        "feedback": deepcopy([
            item for item in report.get("feedback") or [] if isinstance(item, dict)
        ]),
    }


def merge_runtime_scene_catalog(
    base_catalog: Any,
    agent_states: list[dict[str, Any]],
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    """Refresh the episode catalogue with authoritative post-execution facts."""

    by_id: dict[str, dict[str, Any]] = {}

    def merge_object(value: Any) -> None:
        if not isinstance(value, dict):
            return
        object_id = value.get("objectId") or value.get("object_id") or value.get("id")
        object_type = value.get("objectType") or value.get("object_type") or value.get("type")
        if object_id in (None, "") or object_type in (None, ""):
            return
        key = str(object_id)
        current = by_id.setdefault(key, {"objectId": key, "objectType": str(object_type)})
        current["objectType"] = str(object_type)
        for field in (
            "position", "pickupable", "moveable", "openable", "toggleable",
            "receptacle", "sliceable", "dirtyable", "breakable", "cookable",
            "canFillWithLiquid", "parentReceptacles", "receptacleObjectIds",
            "isOpen", "isPickedUp", "inInventory", "visible",
        ):
            if field in value:
                current[field] = deepcopy(value[field])
        after = value.get("after") if isinstance(value.get("after"), dict) else {}
        for field, field_value in after.items():
            if field in {
                "position", "parentReceptacles", "receptacleObjectIds", "isOpen",
                "isPickedUp", "inInventory", "visible",
            }:
                current[field] = deepcopy(field_value)

    for item in base_catalog or []:
        merge_object(item)
    for state in agent_states:
        if not isinstance(state, dict):
            continue
        for key in (
            "scene_object_catalog", "visible_objects", "objects",
            "inventoryObjects", "inventory",
        ):
            for item in state.get(key) or []:
                merge_object(item)
    for item in report.get("object_changes") or []:
        merge_object(item)
    return sorted(by_id.values(), key=lambda item: (item["objectType"], item["objectId"]))


def build_runtime_replanning_constraints(
    execution_context: dict[str, Any],
    scene_catalog: list[dict[str, Any]],
) -> dict[str, Any]:
    excluded_ids = unique_preserve_order([
        str(object_id)
        for failure in execution_context.get("failed_subtasks") or []
        for object_id in failure.get("excluded_destination_object_ids") or []
        if str(object_id).strip()
    ])
    held_objects = [
        {
            "agent_id": str(agent.get("agent_id")),
            "objectId": str(item.get("objectId") or item.get("object_id") or item.get("id") or ""),
            "objectType": str(item.get("objectType") or item.get("object_type") or item.get("type") or ""),
        }
        for agent in execution_context.get("agent_states") or []
        for item in agent.get("inventory") or []
        if isinstance(item, dict)
        and (item.get("objectType") or item.get("object_type") or item.get("type"))
    ]
    placement_sources: dict[str, list[str]] = {}
    for failure in execution_context.get("failed_subtasks") or []:
        subtask = failure.get("subtask") if isinstance(failure.get("subtask"), dict) else {}
        if str(subtask.get("action") or "").lower() not in {"place", "put"}:
            continue
        grounding = subtask.get("grounding") if isinstance(subtask.get("grounding"), dict) else {}
        for source_type in (
            grounding.get("source_object_tags")
            or grounding.get("object_tags")
            or []
        ):
            source_text = str(source_type).strip()
            allowed = compatible_receptacles(source_text) if source_text else None
            if source_text and allowed is not None:
                placement_sources[source_text] = list(allowed)
    return {
        "excluded_object_ids": excluded_ids,
        "held_objects": held_objects,
        "placement_compatibility": placement_sources,
        "completed_task_ids": [
            str(task.get("id"))
            for task in execution_context.get("completed_tasks") or []
            if isinstance(task, dict) and task.get("id") not in (None, "")
        ],
        "available_scene_object_ids": [
            str(item.get("objectId"))
            for item in scene_catalog
            if item.get("objectId") not in (None, "")
        ],
        "rules": {
            "preserve_original_goal": True,
            "plan_only_remaining_work": True,
            "do_not_repeat_completed_effects": True,
            "preserve_held_object_ownership": True,
            "exclude_failed_object_ids": True,
            "enforce_action_contracts": True,
            "enforce_placement_contracts": True,
        },
    }


def validate_replanned_graph_runtime_constraints(
    task_graph: dict[str, Any],
    constraints: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate constraints that cannot be expressed by the initial planner schema."""

    excluded = {str(value) for value in constraints.get("excluded_object_ids") or []}
    held_types = {
        _grounding_entity_key(item.get("objectType"))
        for item in constraints.get("held_objects") or []
        if isinstance(item, dict) and item.get("objectType")
    }
    diagnostics = task_graph.get("planner_diagnostics")
    root_validation = (
        diagnostics.get("root_intent_validation")
        if isinstance(diagnostics, dict)
        else {}
    )
    root_intent = (
        root_validation.get("root_intent")
        if isinstance(root_validation, dict)
        else {}
    )
    roles = root_intent.get("roles") if isinstance(root_intent, dict) else {}
    authoritative_destination_keys = {
        _grounding_entity_key(value)
        for value in (roles.get("destination") if isinstance(roles, dict) else []) or []
        if str(value).strip()
    }
    violations: list[dict[str, Any]] = []
    for task in task_graph.get("flat_tasks") or []:
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("id") or "")
        action = str(task.get("action") or "").lower()
        grounding = task.get("grounding") if isinstance(task.get("grounding"), dict) else {}
        grounded_ids = {
            str(value)
            for key in ("object_ids", "source_object_ids", "destination_object_ids")
            for value in grounding.get(key) or []
        }
        repeated_excluded = sorted(grounded_ids & excluded)
        if repeated_excluded:
            violations.append({
                "task_id": task_id,
                "code": "excluded_runtime_object",
                "field": "grounding",
                "message": f"replanned task reuses excluded object IDs {repeated_excluded}",
                "invalid_values": repeated_excluded,
            })
        source_types = {
            _grounding_entity_key(value)
            for value in (
                grounding.get("source_object_tags")
                or grounding.get("object_tags")
                or []
            )
        }
        if action in {"pick", "pickup"} and source_types & held_types:
            violations.append({
                "task_id": task_id,
                "code": "duplicate_pick_of_held_object",
                "field": "action",
                "message": "replanned task picks an object type that is already held",
                "invalid_values": sorted(source_types & held_types),
            })
        if action in {"place", "put"}:
            destination_types = {
                str(value)
                for value in (
                    (
                        grounding.get("destination_selector")
                        if isinstance(grounding.get("destination_selector"), dict)
                        else {}
                    ).get("object_types")
                    or grounding.get("destination_object_tags")
                    or []
                )
                if str(value).strip()
            }
            raw_source_types = {
                str(value)
                for value in (
                    (
                        grounding.get("source_selector")
                        if isinstance(grounding.get("source_selector"), dict)
                        else {}
                    ).get("object_types")
                    or grounding.get("source_object_tags")
                    or []
                )
                if str(value).strip()
            }
            incompatible = sorted(
                (source_type, destination_type)
                for source_type in raw_source_types
                for destination_type in destination_types
                if placement_compatibility(source_type, destination_type) is False
                and _grounding_entity_key(destination_type)
                not in authoritative_destination_keys
            )
            if incompatible:
                violations.append({
                    "task_id": task_id,
                    "code": "incompatible_receptacle",
                    "field": "grounding.destination_selector",
                    "message": f"replanned placement pairs are incompatible: {incompatible}",
                    "invalid_values": [
                        {
                            "source_type": source_type,
                            "destination_type": destination_type,
                            "compatible_receptacles": list(
                                compatible_receptacles(source_type) or ()
                            ),
                        }
                        for source_type, destination_type in incompatible
                    ],
                })
    return violations


def bind_replanned_destination_instances(
    task_graph: dict[str, Any],
    scene_catalog: list[dict[str, Any]],
    excluded_object_ids: set[str],
) -> dict[str, Any]:
    """Bind an allowed destination instance so execution cannot reuse exclusions."""

    updated = deepcopy(task_graph)
    for task in updated.get("flat_tasks") or []:
        if not isinstance(task, dict) or str(task.get("action") or "").lower() not in {"place", "put"}:
            continue
        grounding = task.get("grounding") if isinstance(task.get("grounding"), dict) else {}
        bound = [
            str(value) for value in grounding.get("destination_object_ids") or []
            if str(value).strip() and str(value) not in excluded_object_ids
        ]
        if bound:
            grounding["destination_object_ids"] = bound
            grounding["excluded_destination_object_ids"] = sorted(excluded_object_ids)
            task["grounding"] = grounding
            continue
        selector = grounding.get("destination_selector") if isinstance(grounding.get("destination_selector"), dict) else {}
        wanted_types = selector.get("object_types") or grounding.get("destination_object_tags") or []
        if isinstance(wanted_types, str):
            wanted_types = [wanted_types]
        wanted_keys = {_grounding_entity_key(value) for value in wanted_types}
        candidates = [
            item for item in scene_catalog
            if isinstance(item, dict)
            and bool(item.get("receptacle"))
            and str(item.get("objectId") or "") not in excluded_object_ids
            and _grounding_entity_key(item.get("objectType")) in wanted_keys
        ]
        candidates.sort(key=lambda item: (
            not bool(item.get("visible")),
            str(item.get("objectId") or ""),
        ))
        if candidates:
            selected = candidates[0]
            grounding["destination_object_ids"] = [str(selected["objectId"])]
            grounding["destination_object_tags"] = [str(selected["objectType"])]
            grounding["object_tags"] = unique_preserve_order([
                *[str(value) for value in grounding.get("source_object_tags") or []],
                str(selected["objectType"]),
            ])
        grounding["excluded_destination_object_ids"] = sorted(excluded_object_ids)
        task["grounding"] = grounding
    return rebuild_task_graph(updated)


def replan_from_execution_report(
    *,
    root_task: str,
    current_task_graph: dict[str, Any],
    report: dict[str, Any],
    task_statuses: list[dict[str, Any]],
    completed: set[str],
    failed: set[str],
    current_scenegraph: dict[str, Any],
    agent_states: list[dict[str, Any]],
    planning_chat: Any,
    loop_index: int,
    replan_index: int,
    loop_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Regenerate the complete remaining Task Graph from current execution facts."""

    replan_dir = loop_dir / "replanning" / f"replan_{replan_index:03d}"
    replan_dir.mkdir(parents=True, exist_ok=True)
    save_json(gen.json_safe(report), replan_dir / "original_execution_report.json")
    save_json(current_task_graph, replan_dir / "old_task_graph.json")

    context = build_replanning_execution_context(
        root_task=root_task,
        report=report,
        task_statuses=task_statuses,
        task_graph=current_task_graph,
        completed=completed,
        failed=failed,
        agent_states=agent_states,
    )
    runtime_catalog = merge_runtime_scene_catalog(
        getattr(args, "_scene_object_catalog", None), agent_states, report
    )
    constraints = build_runtime_replanning_constraints(context, runtime_catalog)
    excluded_ids = set(constraints["excluded_object_ids"])
    planner_catalog = [
        item for item in runtime_catalog
        if str(item.get("objectId")) not in excluded_ids
    ]
    save_json(gen.json_safe(context), replan_dir / "execution_context.json")
    save_json(gen.json_safe(runtime_catalog), replan_dir / "current_scene_catalog.json")
    save_json(gen.json_safe(constraints), replan_dir / "runtime_constraints.json")

    try:
        subgraph = gen.extract_task_relevant_subgraph_to_file(
            scenegraph_info=current_scenegraph,
            task=root_task,
            output_path=replan_dir / "task_relevant_subgraph.json",
            args=args,
        )
        compact_agents = compact_replanning_agent_states(agent_states)
        agent_context = {
            "agent_count": len(compact_agents) or max(int(getattr(args, "agentnum", 1)), 1),
            "agents": compact_agents,
            "inventory_capacity_per_agent": 1,
        }
        replanned = decompose_task_to_graph(
            task=root_task,
            subgraph=subgraph,
            use_qwen=True,
            qwen_model_path=args.planning_model_path,
            qwen_conv_mode=args.qwen_conv_mode,
            qwen_num_gpus=args.qwen_num_gpus,
            qwen_chat=planning_chat,
            qwen_max_new_tokens=args.planning_max_new_tokens,
            agent_context=agent_context,
            scene_catalog=planner_catalog,
            planning_max_attempts=max(
                int(getattr(args, "task_graph_replan_max_attempts", 3)), 1
            ),
            planning_mode="runtime_replan",
            execution_context=context,
            runtime_constraints=constraints,
        )
        replanned = apply_scene_catalog_selectors(replanned, planner_catalog)
        replanned = rebuild_task_graph(replanned)
        replanned = bind_replanned_destination_instances(
            replanned, planner_catalog, excluded_ids
        )
        violations = validate_replanned_graph_runtime_constraints(replanned, constraints)
        if violations:
            raise TaskPlanningError(
                "runtime replanning violated execution constraints",
                diagnostics={
                    "status": "failed",
                    "stage": "runtime_constraint_validation",
                    "violations": violations,
                },
            )
        replanned.setdefault("runtime_history", {})["replanning"] = {
            "replan_index": replan_index,
            "loop_index": loop_index,
            "source_failure_task_ids": [
                str(item.get("subtask_id")) for item in context["failed_subtasks"]
            ],
            "previous_task_graph": str(replan_dir / "old_task_graph.json"),
            "execution_context": str(replan_dir / "execution_context.json"),
            "constraints": str(replan_dir / "runtime_constraints.json"),
        }
        save_json(replanned, replan_dir / "new_task_graph.json")
        args._scene_object_catalog = deepcopy(runtime_catalog)
        result = {
            "status": "success",
            "replan_index": replan_index,
            "loop_index": loop_index,
            "task_graph": replanned,
            "task_graph_path": str(replan_dir / "new_task_graph.json"),
            "source_failure_task_ids": [
                str(item.get("subtask_id")) for item in context["failed_subtasks"]
            ],
            "replan_dir": str(replan_dir),
        }
    except Exception as exc:
        diagnostics = exc.to_dict() if isinstance(exc, TaskPlanningError) else {
            "error_type": type(exc).__name__, "error": str(exc)
        }
        result = {
            "status": "failed",
            "replan_index": replan_index,
            "loop_index": loop_index,
            "source_failure_task_ids": [
                str(item.get("subtask_id")) for item in context["failed_subtasks"]
            ],
            "diagnostics": diagnostics,
            "replan_dir": str(replan_dir),
        }
    save_json(gen.json_safe({key: value for key, value in result.items() if key != "task_graph"}), replan_dir / "replan_result.json")
    return result


def vector_from_change_position(value: Any) -> list[float] | None:
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


def adapter_object_changes_to_observed_nodes(object_changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert legacy adapter changes into graph_update observation nodes."""

    observed_nodes = []
    for index, change in enumerate(object_changes):
        object_id = change.get("objectId") or change.get("ai2thor_object_id") or change.get("id")
        if object_id in (None, ""):
            continue
        object_type = str(change.get("objectType") or change.get("object_type") or "object")
        position = vector_from_change_position(change.get("new_position") or change.get("position"))
        node: dict[str, Any] = {
            "id": change.get("id", index),
            "original_id": change.get("original_id", change.get("id", index)),
            "objectId": str(object_id),
            "ai2thor_object_id": str(object_id),
            "object_tag": object_type,
            "caption": f"Adapter observed {object_type}",
            "possible_tags": [object_type],
            "last_reported_state": deepcopy(change),
            "merge_source": "adapter_observation",
        }
        if position is not None:
            node["bbox_center"] = position
            node["last_reported_position"] = position
        observer_id = change.get("observer_robot_id")
        if observer_id not in (None, ""):
            node["last_observer_agent_id"] = str(observer_id)
        observer_position = change.get("observer_goal_position") or change.get("observer_position")
        observer_vector = vector_from_change_position(observer_position)
        if observer_vector is not None:
            node["last_observer_position"] = observer_vector
        if change.get("source_task_id") not in (None, ""):
            node["source_task_id"] = str(change["source_task_id"])
        observed_nodes.append(node)
    return observed_nodes


def report_observed_scenegraph_path(report: dict[str, Any], output_dir: Path) -> Path | None:
    """Return a report-provided observation graph, or materialize legacy changes."""

    for key in ("observed_scenegraph", "observed_scenegraph_info", "scenegraph_info"):
        value = report.get(key)
        if isinstance(value, str) and Path(value).exists():
            return Path(value)
        if isinstance(value, dict):
            scene_graph = value.get("scene_graph")
            if scene_graph and Path(scene_graph).exists():
                return Path(scene_graph)

    object_changes = [change for change in report.get("object_changes") or [] if isinstance(change, dict)]
    observed_nodes = adapter_object_changes_to_observed_nodes(object_changes)
    if not observed_nodes:
        return None

    observed_path = output_dir / "adapter_observed_scene_graph.json"
    save_json(observed_nodes, observed_path)
    return observed_path


def report_relation_deltas(report: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect explicit and metadata-derived relation deltas from an adapter report."""

    explicit_deltas = [delta for delta in report.get("relation_deltas") or [] if isinstance(delta, dict)]
    pre_metadata = report.get("pre_metadata", report.get("pre_object_metadata"))
    post_metadata = report.get("post_metadata", report.get("post_object_metadata"))
    execution_deltas = []
    if pre_metadata is not None and post_metadata is not None:
        execution_deltas = [
            delta.to_dict()
            for delta in infer_relation_deltas_from_metadata(
                pre_metadata,
                post_metadata,
                execution=report.get("execution"),
            )
        ]
    for change in report.get("object_changes") or []:
        if not isinstance(change, dict):
            continue
        object_id = change.get("objectId") or change.get("object_id") or change.get("id")
        before = change.get("before")
        after = change.get("after")
        if object_id in (None, "") or not isinstance(before, dict) or not isinstance(after, dict):
            continue
        before_object = {**before, "objectId": str(object_id)}
        after_object = {**after, "objectId": str(object_id)}
        object_type = change.get("objectType") or change.get("object_type")
        if object_type:
            before_object.setdefault("objectType", object_type)
            after_object.setdefault("objectType", object_type)
        execution_deltas.extend(
            delta.to_dict()
            for delta in infer_relation_deltas_from_metadata(
                [before_object],
                [after_object],
            )
        )
    return execution_deltas, explicit_deltas


OBJECT_STATE_FIELDS = (
    "position",
    "rotation",
    "axisAlignedBoundingBox",
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
    "inInventory",
    "visible",
)


def report_object_deltas(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize adapter state changes to the same deltas used by AI2-THOR."""

    pre_metadata = report.get("pre_metadata", report.get("pre_object_metadata"))
    post_metadata = report.get("post_metadata", report.get("post_object_metadata"))
    deltas = []
    if pre_metadata is not None and post_metadata is not None:
        deltas.extend(delta.to_dict() for delta in diff_objects(pre_metadata, post_metadata))

    for change in report.get("object_changes") or []:
        if not isinstance(change, dict):
            continue
        object_id = change.get("objectId") or change.get("object_id") or change.get("id")
        if object_id in (None, ""):
            continue
        object_type = change.get("objectType") or change.get("object_type")
        before = change.get("before") if isinstance(change.get("before"), dict) else {}
        after = change.get("after") if isinstance(change.get("after"), dict) else {}
        changed_fields: dict[str, dict[str, Any]] = {}
        for field_name in OBJECT_STATE_FIELDS:
            if field_name in after:
                changed_fields[field_name] = {
                    "before": deepcopy(before.get(field_name)),
                    "after": deepcopy(after[field_name]),
                }
            elif field_name in change:
                changed_fields[field_name] = {
                    "before": deepcopy(before.get(field_name)),
                    "after": deepcopy(change[field_name]),
                }
        explicit_fields = change.get("changed_fields")
        if isinstance(explicit_fields, dict):
            changed_fields.update(deepcopy(explicit_fields))
        if changed_fields:
            deltas.append(
                {
                    "object_id": str(object_id),
                    "object_type": str(object_type) if object_type else None,
                    "changed_fields": changed_fields,
                }
            )
    return deltas


def update_scenegraph_from_report(
    current_scenegraph: dict[str, Any],
    report: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Apply adapter observations and execution deltas with memory.graph_update."""

    scene_graph_path_raw = current_scenegraph.get("scene_graph")
    # ConceptGraphs exposes both binary graph edges (`relations`) and normalized
    # JSON relation records (`object_relations`). graph_update consumes the JSON
    # records, matching the direct AI2-THOR incremental update path.
    relations_path_raw = current_scenegraph.get("object_relations") or current_scenegraph.get("relations")
    if (
        not scene_graph_path_raw
        or not relations_path_raw
        or not Path(scene_graph_path_raw).exists()
        or not Path(relations_path_raw).exists()
    ):
        return current_scenegraph

    output_dir.mkdir(parents=True, exist_ok=True)
    observed_scenegraph_path = report_observed_scenegraph_path(report, output_dir)
    execution_deltas, explicit_deltas = report_relation_deltas(report)
    object_deltas = report_object_deltas(report)
    relation_deltas = [*execution_deltas, *explicit_deltas]
    relation_delta_path = output_dir / "relation_deltas.json"
    object_delta_path = output_dir / "object_deltas.json"
    save_json(
        {
            "execution_deltas": execution_deltas,
            "explicit_deltas": explicit_deltas,
            "all_deltas": relation_deltas,
        },
        relation_delta_path,
    )
    save_json(object_deltas, object_delta_path)
    updated = update_scenegraph_files(
        stored_scenegraph_path=Path(scene_graph_path_raw),
        stored_relations_path=Path(relations_path_raw),
        observed_scenegraph_path=observed_scenegraph_path,
        relation_deltas=relation_deltas,
        object_deltas=object_deltas,
        output_dir=output_dir,
        append_observed_only=True,
    )
    updated["update_backend"] = "adapter_graph_update"
    updated["relation_deltas"] = str(relation_delta_path)
    updated["object_deltas"] = str(object_delta_path)
    return updated


def initialize_scenegraph_and_agents(
    args: argparse.Namespace,
    run_dir: Path,
    skills_by_agent: dict[str, list[str]],
) -> tuple[Any | None, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Build/load the initial scene graph and return controller if needed."""

    if args.scenegraph_info:
        scenegraph_info = load_json(args.scenegraph_info)
        if args.initial_agent_states and args.initial_agent_states.exists():
            agent_states = load_json(args.initial_agent_states)
        else:
            agent_states = offline_agent_states(args.agentnum, skills_by_agent)
        return None, scenegraph_info, attach_agent_skills(agent_states, skills_by_agent), {
            "source": "scenegraph_info",
            "scenegraph_info": str(args.scenegraph_info),
            "initial_agent_states": str(args.initial_agent_states) if args.initial_agent_states else None,
        }

    gen.start_xserver_if_needed(args)
    gen.validate_cloud_rendering_environment(args)
    controller = gen.make_controller(args)
    gen.reset_scene(controller, args)

    runtime_dir = run_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    initial_record: dict[str, Any] = {"source": "ai2thor"}

    if args.global_scenegraph:
        log("building global initial scene graph")
        global_record = gen.build_global_scenegraph(controller, runtime_dir, args)
        if global_record is None:
            raise RuntimeError("global scene graph construction returned None")
        scenegraph_info = global_record["scenegraph"]
        initial_record["global_mapping"] = global_record
        gen.reset_scene(controller, args)
        initial_stage = gen.collect_stage(controller, runtime_dir, "initial_after_global_mapping", args)
    else:
        log("building initial local scene graph")
        initial_stage = gen.collect_stage(controller, runtime_dir, "initial", args)
        scenegraph_info = gen.build_stage_scenegraph(initial_stage, args)

    agent_states = attach_agent_skills(initial_stage["agent_states"], skills_by_agent)
    initial_record["initial_stage"] = {key: value for key, value in initial_stage.items() if key != "agent_states"}
    initial_record["scenegraph"] = scenegraph_info
    return controller, scenegraph_info, agent_states, initial_record


def build_initial_task_graph(
    scenegraph_info: dict[str, Any],
    task: str,
    output_dir: Path,
    args: argparse.Namespace,
    agent_states: list[dict[str, Any]] | None = None,
    qwen_chat: Any | None = None,
    use_qwen: bool | None = None,
) -> dict[str, Any]:
    save_intermediates = should_save_intermediates(args)
    compact_agents = []
    for index, state in enumerate(agent_states or []):
        if not isinstance(state, dict):
            continue
        nested_agent = state.get("agent") if isinstance(state.get("agent"), dict) else {}
        inventory = state.get("inventoryObjects") or state.get("inventory") or []
        compact_agents.append(
            {
                "agent_id": str(state.get("agent_id", index)),
                "position": state.get("position") or nested_agent.get("position"),
                "rotation": state.get("rotation") or nested_agent.get("rotation"),
                "inventory": inventory,
                "skills": list(state.get("skills") or []),
            }
        )
    agent_context = {
        "agent_count": len(compact_agents) or max(int(getattr(args, "agentnum", 1)), 1),
        "agents": compact_agents,
        "inventory_capacity_per_agent": 1,
    }

    def build_with_subgraph_path(subgraph_path: Path) -> dict[str, Any]:
        subgraph = gen.extract_task_relevant_subgraph_to_file(
            scenegraph_info=scenegraph_info,
            task=task,
            output_path=subgraph_path,
            args=args,
        )
        task_graph = decompose_task_to_graph(
            task=task,
            subgraph=subgraph,
            use_qwen=not args.disable_qwen if use_qwen is None else use_qwen,
            qwen_model_path=args.planning_model_path,
            qwen_conv_mode=args.qwen_conv_mode,
            qwen_num_gpus=args.qwen_num_gpus,
            agent_context=agent_context,
            qwen_max_new_tokens=args.planning_max_new_tokens,
            qwen_chat=qwen_chat,
            scene_catalog=getattr(args, "_scene_object_catalog", None),
            planning_max_attempts=max(
                int(getattr(args, "planning_max_attempts", 3)), 1
            ),
        )
        return {"subgraph": subgraph, "task_graph": task_graph}

    if save_intermediates:
        output_dir.mkdir(parents=True, exist_ok=True)
        subgraph_path = output_dir / "task_relevant_subgraph.json"
        task_graph_path = output_dir / "task_graph.json"
        result = build_with_subgraph_path(subgraph_path)
        task_graph = result["task_graph"]
        save_task_graph(task_graph, task_graph_path)
        subgraph_path_ref = str(subgraph_path)
        task_graph_path_ref = str(task_graph_path)
    else:
        with tempfile.TemporaryDirectory(prefix="hybrid_initial_planning_") as tmpdir:
            result = build_with_subgraph_path(Path(tmpdir) / "task_relevant_subgraph.json")
        subgraph_path_ref = None
        task_graph_path_ref = None

    return {
        "subgraph": result["subgraph"],
        "subgraph_path": subgraph_path_ref,
        "task_graph": result["task_graph"],
        "task_graph_path": task_graph_path_ref,
    }


def run_hybrid_loop(args: argparse.Namespace) -> dict[str, Any]:
    run_name = args.run_name or now_run_name(args.scene_name)
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    public_args = {
        key: value
        for key, value in vars(args).items()
        if not str(key).startswith("_")
    }
    save_json(
        gen.json_safe(public_args | {"output_dir": str(args.output_dir)}),
        run_dir / "args.json",
    )

    skills_by_agent = load_agent_skills(args.agent_skills_json, args.agentnum)
    controller, current_scenegraph, agent_states, initial_scene_record = initialize_scenegraph_and_agents(
        args,
        run_dir,
        skills_by_agent,
    )
    save_json(initial_scene_record, run_dir / "initial_scenegraph_record.json")
    save_json(agent_states, run_dir / "initial_agent_states.json")

    planning_chat = getattr(args, "_shared_planning_chat", None)
    owns_planning_chat = planning_chat is None
    use_planning_qwen = not args.disable_qwen
    try:
        if use_planning_qwen and planning_chat is None:
            try:
                from conceptgraph.vlm import build_vlm_chat

                planning_chat = build_vlm_chat(
                    backend="qwen",
                    model_path=args.planning_model_path,
                    conv_mode=args.qwen_conv_mode,
                    num_gpus=args.qwen_num_gpus,
                )
            except Exception as exc:
                raise TaskPlanningError(
                    f"planning Qwen initialization failed: {exc!r}",
                    diagnostics={
                        "status": "failed",
                        "stage": "model_initialization",
                        "max_attempts": max(
                            int(getattr(args, "planning_max_attempts", 3)), 1
                        ),
                        "attempts": [],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                ) from exc

        initial_planning = build_initial_task_graph(
            current_scenegraph,
            args.task,
            run_dir / "initial_planning",
            args,
            agent_states=agent_states,
            qwen_chat=planning_chat,
            use_qwen=use_planning_qwen,
        )
        task_graph = apply_scene_catalog_selectors(
            initial_planning["task_graph"],
            getattr(args, "_scene_object_catalog", None),
        )
        task_graph = rebuild_task_graph(task_graph)
        save_intermediate_json(task_graph, run_dir / "runtime_task_graph_initial.json", args)
    except TaskPlanningError as exc:
        save_planning_failure(exc, run_dir, args)
        if planning_chat is not None and owns_planning_chat:
            from conceptgraph.vlm import close_vlm_chat

            close_vlm_chat(planning_chat)
        if controller is not None:
            controller.stop()
        raise
    except Exception:
        if planning_chat is not None and owns_planning_chat:
            from conceptgraph.vlm import close_vlm_chat

            close_vlm_chat(planning_chat)
        if controller is not None:
            controller.stop()
        raise

    if args.execution_mode == "ai2thor":
        if controller is None:
            gen.start_xserver_if_needed(args)
            gen.validate_cloud_rendering_environment(args)
            controller = gen.make_controller(args)
            gen.reset_scene(controller, args)
        adapter: CommunicationAdapter = AI2ThorDirectAdapter(controller, args)
    elif args.execution_mode == "task_service":
        adapter = TaskExecutionServiceAdapter(args)
    else:
        adapter = FileCommunicationAdapter(
            result_file=args.adapter_result_file,
            auto_success=args.adapter_auto_success,
        )

    completed: set[str] = initial_completed_task_ids(task_graph)
    if completed:
        log(f"initially satisfied tasks: {sorted(completed)}")
    failed: set[str] = set()
    retry_counts: dict[str, int] = {}
    loop_records = []
    task_graph_replan_count = 0
    task_graph_version = 1
    task_graph_replan_events: list[dict[str, Any]] = []
    task_graph_replan_counts_by_source: dict[str, int] = {}
    historical_completed_task_ids: set[str] = set(completed)

    try:
        for loop_index in range(args.max_task_loops):
            all_task_ids = {str(task["id"]) for task in task_graph.get("flat_tasks") or []}
            if completed >= all_task_ids:
                log(f"all tasks completed at loop {loop_index}")
                break

            loop_dir = run_dir / "loops" / f"loop_{loop_index:03d}"
            loop_dir.mkdir(parents=True, exist_ok=True)
            save_intermediate_json(task_graph, loop_dir / "task_graph_before_dispatch.json", args)
            save_intermediate_json({"completed": sorted(completed), "failed": sorted(failed)}, loop_dir / "progress_before_dispatch.json", args)

            if args.execution_mode == "ai2thor" and controller is not None:
                pre_stage = gen.collect_stage(controller, run_dir / "runtime", f"loop_{loop_index:03d}_pre", args)
                agent_states = attach_agent_skills(pre_stage["agent_states"], skills_by_agent)
                save_intermediate_json(agent_states, loop_dir / "pre_agent_states.json", args)
            else:
                pre_stage = None

            remaining_tasks = [
                deepcopy(task)
                for task in task_graph.get("flat_tasks") or []
                if str(task.get("id")) not in completed | failed
            ]
            task_subgraphs: dict[str, dict[str, Any]] = {}
            for task in remaining_tasks:
                task_id = str(task["id"])
                subgraph_path = loop_dir / "task_subgraphs" / task_id / "task_relevant_subgraph.json"
                subgraph = extract_subgraph_for_task(current_scenegraph, task, subgraph_path, args)
                task_subgraphs[task_id] = subgraph

            task_graph = rebuild_task_graph(task_graph)
            save_intermediate_json(task_graph, loop_dir / "semantic_task_graph.json", args)
            merged_subgraph = merge_subgraphs_for_allocation(
                task_subgraphs,
                loop_dir / "remaining_scene_context.json" if should_save_intermediates(args) else None,
            )

            try:
                execution_plan = build_execution_plan(
                    task_graph=task_graph,
                    agent_states=agent_states,
                    scene_context=merged_subgraph,
                    progress={
                        "completed_task_ids": sorted(completed),
                        "failed_task_ids": sorted(failed),
                    },
                    task_graph_version=task_graph_version,
                    use_qwen=use_planning_qwen,
                    qwen_model_path=args.planning_model_path,
                    qwen_conv_mode=args.qwen_conv_mode,
                    qwen_num_gpus=args.qwen_num_gpus,
                    qwen_chat=planning_chat,
                    qwen_max_new_tokens=args.planning_max_new_tokens,
                    allocation_max_attempts=args.planning_max_attempts,
                    diagnostics_output_dir=loop_dir / "allocation_planning",
                )
                save_intermediate_json(
                    execution_plan,
                    loop_dir / "execution_plan.json",
                    args,
                )
                save_intermediate_json(
                    execution_plan.get("diagnostics") or {},
                    loop_dir / "allocation_diagnostics.json",
                    args,
                )

                if execution_plan.get("state") == "blocked":
                    blocking_statuses = allocation_blocking_statuses(
                        execution_plan,
                        task_graph,
                    )
                    blocking_report = {
                        "trigger": "allocation_blocked",
                        "execution_plan": deepcopy(execution_plan),
                        "task_statuses": deepcopy(blocking_statuses),
                        "execution": {"completed_task_ids": [], "traces": []},
                        "object_changes": [],
                        "feedback": [],
                    }
                    source_key = allocation_blocking_budget_key(execution_plan)
                    within_episode_budget = task_graph_replan_count < max(
                        int(getattr(args, "max_task_graph_replans", 3)), 0
                    )
                    within_source_budget = (
                        task_graph_replan_counts_by_source.get(source_key, 0)
                        < max(int(getattr(args, "max_task_replans_per_source", 1)), 0)
                    )
                    replan_enabled = bool(
                        getattr(args, "enable_task_graph_replan", True)
                    )
                    if (
                        replan_enabled
                        and within_episode_budget
                        and within_source_budget
                        and planning_chat is not None
                    ):
                        replan_result = replan_from_execution_report(
                            root_task=args.task,
                            current_task_graph=task_graph,
                            report=blocking_report,
                            task_statuses=blocking_statuses,
                            completed=completed,
                            failed=failed,
                            current_scenegraph=current_scenegraph,
                            agent_states=agent_states,
                            planning_chat=planning_chat,
                            loop_index=loop_index,
                            replan_index=task_graph_replan_count + 1,
                            loop_dir=loop_dir,
                            args=args,
                        )
                        task_graph_replan_count += 1
                        task_graph_replan_counts_by_source[source_key] = (
                            task_graph_replan_counts_by_source.get(source_key, 0) + 1
                        )
                    else:
                        reason = (
                            "task_graph_replan_disabled"
                            if not replan_enabled
                            else "episode_replan_budget_exhausted"
                            if not within_episode_budget
                            else "source_replan_budget_exhausted"
                            if not within_source_budget
                            else "planning_model_unavailable"
                        )
                        replan_result = {
                            "status": "skipped",
                            "loop_index": loop_index,
                            "source_failure_task_ids": [
                                str(item.get("subtask_id"))
                                for item in blocking_statuses
                            ],
                            "reason": reason,
                        }

                    event = {
                        key: deepcopy(value)
                        for key, value in replan_result.items()
                        if key != "task_graph"
                    }
                    event["trigger"] = "allocation_blocked"
                    event["blocking_budget_key"] = source_key
                    task_graph_replan_events.append(event)
                    save_intermediate_json(
                        event,
                        loop_dir / "task_graph_replan_result.json",
                        args,
                    )

                    if replan_result.get("status") == "success":
                        historical_completed_task_ids.update(completed)
                        task_graph = replan_result["task_graph"]
                        task_graph_version += 1
                        completed = initial_completed_task_ids(task_graph)
                        failed = set()
                        retry_counts = {}
                        replacement_ids = {
                            str(task["id"])
                            for task in task_graph.get("flat_tasks") or []
                        }
                        completion = {
                            "completed_this_loop": [],
                            "failed_this_loop": [],
                            "wait_retry_this_loop": [],
                            "completed_all": sorted(completed),
                            "failed_all": [],
                            "remaining": sorted(replacement_ids - completed),
                            "retry_counts": {},
                            "all_done": completed >= replacement_ids,
                            "task_graph_replanned": True,
                            "replan_trigger": "allocation_blocked",
                            "task_graph_version": task_graph_version,
                            "historical_completed_task_ids": sorted(
                                historical_completed_task_ids
                            ),
                        }
                        save_intermediate_json(
                            task_graph,
                            loop_dir / "task_graph_after_replanning.json",
                            args,
                        )
                        save_intermediate_json(
                            completion,
                            loop_dir / "task_completion.json",
                            args,
                        )
                        loop_record = {
                            "loop_index": loop_index,
                            "task_graph_before_dispatch": intermediate_file_ref(
                                args, loop_dir / "task_graph_before_dispatch.json"
                            ),
                            "remaining_scene_context": intermediate_file_ref(
                                args, loop_dir / "remaining_scene_context.json"
                            ),
                            "execution_plan": intermediate_file_ref(
                                args, loop_dir / "execution_plan.json"
                            ),
                            "task_graph_replanning": intermediate_file_ref(
                                args, loop_dir / "task_graph_replan_result.json"
                            ),
                            "runtime_task_graph_after_replanning": intermediate_file_ref(
                                args, loop_dir / "task_graph_after_replanning.json"
                            ),
                            "task_completion": intermediate_file_ref(
                                args, loop_dir / "task_completion.json"
                            ),
                            "scenegraph_after_update": current_scenegraph.get(
                                "scene_graph"
                            ),
                        }
                        save_intermediate_json(
                            loop_record,
                            loop_dir / "loop_record.json",
                            args,
                        )
                        loop_records.append(loop_record)
                        log(
                            f"loop {loop_index}: replaced runtime task graph after "
                            "Allocation reported blocked"
                        )
                        continue

                    blocked_code = str(
                        (execution_plan.get("blocking") or {}).get("code")
                        or "allocation_blocked"
                    )
                    termination_code = (
                        "allocation_replan_failed"
                        if replan_result.get("status") == "failed"
                        else "allocation_replan_budget_exhausted"
                        if str(replan_result.get("reason") or "").endswith(
                            "budget_exhausted"
                        )
                        else "allocation_replan_unavailable"
                    )
                    raise TaskAllocationError(
                        f"Execution Plan blocked ({blocked_code}); Planning did not recover.",
                        code=termination_code,
                        diagnostics={
                            "status": "blocked",
                            "stage": "allocation_replanning",
                            "execution_plan": deepcopy(execution_plan),
                            "replan_result": deepcopy(event),
                        },
                    )
                assignments = first_execution_unit(execution_plan, task_graph)
            except TaskAllocationError as exc:
                save_allocation_failure(exc, loop_dir, args)
                raise

            agent_task_map = assignments_to_agent_task_map(assignments)
            communication_payload = {
                "loop_index": loop_index,
                "task": args.task,
                "agent_task_map": agent_task_map,
                "assignments": assignments,
                "agent_states": agent_states,
                "task_graph": task_graph,
                "execution_plan": execution_plan,
                "scene_context": merged_subgraph,
                "protocol": {
                    "expected_report_fields": [
                        "task_statuses",
                        "execution",
                        "agent_states",
                        "object_changes",
                        "feedback",
                        "scenegraph_info",
                    ],
                    "task_status_values": [SUCCESS, WAIT_RETRY, FAILURE],
                },
            }
            save_intermediate_json(agent_task_map, loop_dir / "agent_task_map.json", args)
            adapter.send_task_allocation(communication_payload, loop_dir)

            report = adapter.receive_execution_report(communication_payload, loop_dir)
            if args.execution_mode == "ai2thor" and controller is not None:
                post_stage = gen.collect_stage(controller, run_dir / "runtime", f"loop_{loop_index:03d}_post", args)
                post_agent_states = attach_agent_skills(post_stage["agent_states"], skills_by_agent)
                task_statuses = gen.evaluate_task_statuses(
                    assignments=assignments,
                    execution=report.get("execution") or {},
                    pre_agent_states=agent_states,
                    post_agent_states=post_agent_states,
                    retry_counts=retry_counts,
                    max_task_retries=args.max_task_retries,
                )
                observed_scenegraph, merged_scenegraph = gen.update_stage_scenegraph_incrementally(
                    pre_stage=pre_stage,
                    post_stage=post_stage,
                    current_scenegraph=current_scenegraph,
                    execution=report.get("execution") or {},
                    output_dir=loop_dir / "merged_scenegraph",
                    args=args,
                )
                save_intermediate_json({key: value for key, value in post_stage.items() if key != "agent_states"}, loop_dir / "post_stage_info.json", args)
                save_intermediate_json(post_agent_states, loop_dir / "post_agent_states.json", args)
                save_intermediate_json(observed_scenegraph, loop_dir / "observed_scenegraph_info.json", args)
                current_scenegraph = merged_scenegraph
                agent_states = post_agent_states
            else:
                task_statuses = statuses_from_adapter_report(report, assignments)
                if isinstance(report.get("agent_states"), list):
                    agent_states = attach_agent_skills(report["agent_states"], skills_by_agent)
                current_scenegraph = update_scenegraph_from_report(
                    current_scenegraph,
                    report,
                    loop_dir / "merged_scenegraph",
                )

            runtime_planning_explanations: list[dict[str, Any]] = []
            for status in task_statuses:
                if not status.get("exhaustive"):
                    continue
                # Structural placement failures must reach the Task Graph replan
                # trigger below. Explain-only is reserved for terminal failures
                # that Planning cannot repair with a new executable graph.
                if execution_status_requires_task_graph_replan(status):
                    continue
                explanation_record: dict[str, Any] = {
                    "subtask_id": str(status.get("subtask_id")),
                    "request_status": deepcopy(status),
                }
                try:
                    if planning_chat is None:
                        raise RuntimeError("Planning Qwen is unavailable")
                    explanation = explain_exhaustive_runtime_failure(
                        planning_chat,
                        parent_task=args.task,
                        task_status=status,
                        agent_states=agent_states,
                    )
                    explanation_record.update({
                        "status": "success",
                        "explanation": explanation,
                    })
                    status["runtime_planning_explanation"] = explanation
                    status["reason"] = explanation["reason"]
                except Exception as exc:
                    explanation_record.update({
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
                    status["runtime_planning_explanation_error"] = str(exc)
                status["status"] = FAILURE
                status["recoverable"] = False
                status["recommended_recovery"] = None
                runtime_planning_explanations.append(explanation_record)
            if runtime_planning_explanations:
                save_intermediate_json(
                    {
                        "status": "complete",
                        "mode": "explain_only",
                        "task_graph_modified": False,
                        "explanations": runtime_planning_explanations,
                    },
                    loop_dir / "runtime_planning_diagnostics.json",
                    args,
                )

            task_graph = apply_execution_feedback_to_task_graph(
                task_graph,
                task_statuses,
                report.get("object_changes") if isinstance(report.get("object_changes"), list) else [],
            )
            replan_candidates = [
                status for status in task_statuses
                if execution_status_requires_task_graph_replan(status)
            ]
            replan_result: dict[str, Any] | None = None
            if replan_candidates and bool(getattr(args, "enable_task_graph_replan", True)):
                source_ids = unique_preserve_order([
                    str(status.get("subtask_id")) for status in replan_candidates
                ])
                source_budget_keys = [
                    f"v{task_graph_version}:{source_id}" for source_id in source_ids
                ]
                within_episode_budget = task_graph_replan_count < max(
                    int(getattr(args, "max_task_graph_replans", 3)), 0
                )
                within_source_budget = all(
                    task_graph_replan_counts_by_source.get(source_key, 0)
                    < max(int(getattr(args, "max_task_replans_per_source", 1)), 0)
                    for source_key in source_budget_keys
                )
                if within_episode_budget and within_source_budget and planning_chat is not None:
                    next_replan_index = task_graph_replan_count + 1
                    replan_result = replan_from_execution_report(
                        root_task=args.task,
                        current_task_graph=task_graph,
                        report=report,
                        task_statuses=task_statuses,
                        completed=completed,
                        failed=failed,
                        current_scenegraph=current_scenegraph,
                        agent_states=agent_states,
                        planning_chat=planning_chat,
                        loop_index=loop_index,
                        replan_index=next_replan_index,
                        loop_dir=loop_dir,
                        args=args,
                    )
                    task_graph_replan_count += 1
                    for source_key in source_budget_keys:
                        task_graph_replan_counts_by_source[source_key] = (
                            task_graph_replan_counts_by_source.get(source_key, 0) + 1
                        )
                else:
                    reason = (
                        "episode_replan_budget_exhausted" if not within_episode_budget
                        else "source_replan_budget_exhausted" if not within_source_budget
                        else "planning_model_unavailable"
                    )
                    replan_result = {
                        "status": "skipped",
                        "loop_index": loop_index,
                        "source_failure_task_ids": source_ids,
                        "reason": reason,
                    }

                event = {
                    key: deepcopy(value)
                    for key, value in (replan_result or {}).items()
                    if key != "task_graph"
                }
                task_graph_replan_events.append(event)
                save_intermediate_json(event, loop_dir / "task_graph_replan_result.json", args)

            if replan_result is not None and replan_result.get("status") == "success":
                historical_completed_task_ids.update(completed)
                historical_completed_task_ids.update(
                    str(status.get("subtask_id"))
                    for status in task_statuses
                    if str(status.get("status") or "").lower()
                    in {SUCCESS, "completed", "complete", "success"}
                )
                task_graph = replan_result["task_graph"]
                task_graph_version += 1
                completed = initial_completed_task_ids(task_graph)
                failed = set()
                retry_counts = {}
                all_task_ids = {
                    str(task["id"]) for task in task_graph.get("flat_tasks") or []
                }
                completion = {
                    "completed_this_loop": [],
                    "failed_this_loop": [],
                    "wait_retry_this_loop": [],
                    "completed_all": sorted(completed),
                    "failed_all": [],
                    "remaining": sorted(all_task_ids - completed),
                    "retry_counts": {},
                    "all_done": completed >= all_task_ids,
                    "task_graph_replanned": True,
                    "task_graph_version": task_graph_version,
                    "historical_completed_task_ids": sorted(historical_completed_task_ids),
                }
                save_intermediate_json(report.get("execution") or {}, loop_dir / "execution_trace.json", args)
                save_intermediate_json(task_statuses, loop_dir / "task_statuses.json", args)
                save_intermediate_json(current_scenegraph, loop_dir / "current_scenegraph_after_update.json", args)
                save_intermediate_json(task_graph, loop_dir / "task_graph_after_replanning.json", args)
                save_intermediate_json(completion, loop_dir / "task_completion.json", args)
                loop_record = {
                    "loop_index": loop_index,
                    "task_graph_before_dispatch": intermediate_file_ref(args, loop_dir / "task_graph_before_dispatch.json"),
                    "remaining_scene_context": intermediate_file_ref(args, loop_dir / "remaining_scene_context.json"),
                    "execution_plan": intermediate_file_ref(args, loop_dir / "execution_plan.json"),
                    "agent_task_map": intermediate_file_ref(args, loop_dir / "agent_task_map.json"),
                    "communication_outbox": str(loop_dir / "communication_outbox.json"),
                    "communication_inbox": str(loop_dir / "communication_inbox.json"),
                    "task_statuses": intermediate_file_ref(args, loop_dir / "task_statuses.json"),
                    "task_completion": intermediate_file_ref(args, loop_dir / "task_completion.json"),
                    "task_graph_replanning": str(loop_dir / "task_graph_replan_result.json"),
                    "runtime_task_graph_after_replanning": intermediate_file_ref(args, loop_dir / "task_graph_after_replanning.json"),
                    "scenegraph_after_update": current_scenegraph.get("scene_graph"),
                }
                save_intermediate_json(loop_record, loop_dir / "loop_record.json", args)
                loop_records.append(loop_record)
                log(
                    f"loop {loop_index}: replaced runtime task graph after execution failure "
                    f"(replan {replan_result.get('replan_index')})"
                )
                continue

            progress_delta = apply_progress(
                task_statuses,
                completed,
                failed,
                retry_counts,
                max_task_retries=args.max_task_retries,
            )
            all_task_ids = {str(task["id"]) for task in task_graph.get("flat_tasks") or []}
            completion = {
                **progress_delta,
                "completed_all": sorted(completed),
                "failed_all": sorted(failed),
                "remaining": sorted(all_task_ids - completed - failed),
                "retry_counts": retry_counts,
                "all_done": completed >= all_task_ids,
            }
            save_intermediate_json(report.get("execution") or {}, loop_dir / "execution_trace.json", args)
            save_intermediate_json(task_statuses, loop_dir / "task_statuses.json", args)
            save_intermediate_json(current_scenegraph, loop_dir / "current_scenegraph_after_update.json", args)
            save_intermediate_json(completion, loop_dir / "task_completion.json", args)

            loop_record = {
                "loop_index": loop_index,
                "task_graph_before_dispatch": intermediate_file_ref(args, loop_dir / "task_graph_before_dispatch.json"),
                "semantic_task_graph": intermediate_file_ref(args, loop_dir / "semantic_task_graph.json"),
                "remaining_scene_context": intermediate_file_ref(args, loop_dir / "remaining_scene_context.json"),
                "execution_plan": intermediate_file_ref(args, loop_dir / "execution_plan.json"),
                "allocation_diagnostics": intermediate_file_ref(args, loop_dir / "allocation_diagnostics.json"),
                "agent_task_map": intermediate_file_ref(args, loop_dir / "agent_task_map.json"),
                "communication_outbox": str(loop_dir / "communication_outbox.json"),
                "communication_inbox": str(loop_dir / "communication_inbox.json"),
                "task_statuses": intermediate_file_ref(args, loop_dir / "task_statuses.json"),
                "task_completion": intermediate_file_ref(args, loop_dir / "task_completion.json"),
                "scenegraph_after_update": current_scenegraph.get("scene_graph"),
            }
            save_intermediate_json(loop_record, loop_dir / "loop_record.json", args)
            loop_records.append(loop_record)

            if completion["all_done"]:
                historical_completed_task_ids.update(completed)
                log(f"all tasks completed after loop {loop_index}")
                break

    finally:
        if planning_chat is not None and owns_planning_chat:
            from conceptgraph.vlm import close_vlm_chat

            close_vlm_chat(planning_chat)
        if controller is not None:
            controller.stop()

    all_task_ids = {str(task["id"]) for task in task_graph.get("flat_tasks") or []}
    final_record = {
        "run_dir": str(run_dir),
        "scene_name": args.scene_name,
        "task": args.task,
        "execution_mode": args.execution_mode,
        "initial_scenegraph_record": str(run_dir / "initial_scenegraph_record.json"),
        "initial_task_graph": initial_planning["task_graph_path"],
        "runtime_task_graph_final": str(run_dir / "runtime_task_graph_final.json"),
        "final_scenegraph": current_scenegraph.get("scene_graph"),
        "completed_tasks": sorted(completed),
        "failed_tasks": sorted(failed),
        "remaining_tasks": sorted(all_task_ids - completed - failed),
        "all_done": completed >= all_task_ids,
        "historical_completed_task_ids": sorted(historical_completed_task_ids | completed),
        "task_graph_replan_count": task_graph_replan_count,
        "task_graph_version": task_graph_version,
        "task_graph_replan_events": task_graph_replan_events,
        "loops": loop_records,
    }
    save_json(task_graph, run_dir / "runtime_task_graph_final.json")
    save_json(final_record, run_dir / "run_record.json")
    log(f"saved hybrid decision run: {run_dir}")
    return final_record


def build_parser() -> argparse.ArgumentParser:
    parser = gen.build_parser()
    parser.description = "Run the hybrid decentralized-centralized EMAS decision loop."
    parser.set_defaults(output_dir=Path("/home/kinova-1/EMAS/225010231/mwl/EMAS/scripts/runs"))
    parser.add_argument(
        "--execution-mode",
        choices=["adapter", "ai2thor", "task_service"],
        default="adapter",
        help=(
            "adapter writes/reads JSON communication files; ai2thor directly executes existing skill plans; "
            "task_service sends allocated subtasks to agents/task_execution_server.py /execute_task."
        ),
    )
    parser.add_argument(
        "--scenegraph-info",
        type=Path,
        default=None,
        help="Optional JSON scenegraph info with scene_graph/relations paths. If omitted, an initial scene graph is built.",
    )
    parser.add_argument(
        "--initial-agent-states",
        type=Path,
        default=None,
        help="Optional JSON agent state list for adapter mode when --scenegraph-info is used.",
    )
    parser.add_argument(
        "--agent-skills-json",
        type=Path,
        default=None,
        help="Optional skills JSON: either a list applied to all agents or {agent_id: [skills...]} mapping.",
    )
    parser.add_argument(
        "--adapter-result-file",
        type=Path,
        default=None,
        help="Optional external result JSON consumed by the file communication adapter.",
    )
    parser.add_argument(
        "--adapter-auto-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When no adapter result file is present, synthesize success reports for dry-run/debugging.",
    )
    parser.add_argument(
        "--task-service-url",
        default="http://127.0.0.1:18080/execute_task",
        help="HTTP POST /execute_task endpoint exposed by agents/task_execution_server.py.",
    )
    parser.add_argument(
        "--task-service-timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds for each task_service /execute_task request.",
    )
    parser.add_argument(
        "--task-service-dry-run",
        action="store_true",
        help="Forward dry_run=true to task_execution_server.py without mutating the simulator.",
    )
    parser.add_argument(
        "--task-service-relay-strategy",
        choices=["agent", "rules"],
        default="agent",
        help="Relay strategy forwarded to task_execution_server.py.",
    )
    parser.add_argument("--task-service-max-replan-steps", type=int, default=10)
    parser.add_argument("--task-service-relay-agent-max-turns", type=int, default=8)
    parser.add_argument("--task-service-max-actions", type=int, default=8)
    parser.add_argument(
        "--task-service-autostart",
        action="store_true",
        help="Benchmark mode: start a managed local receiver and task_execution_server for task_service execution.",
    )
    parser.add_argument(
        "--task-service-python",
        default=None,
        help="Python interpreter used to launch agents/task_execution_server.py when --task-service-autostart is enabled.",
    )
    parser.add_argument(
        "--task-service-model-path",
        default="/225010231/mwl/Linhao/models/Qwen3.5-4B",
        help="Qwen model path used by managed task_execution_server.py.",
    )
    parser.add_argument("--task-service-device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--task-service-device-map", default="auto")
    parser.add_argument("--task-service-dtype", choices=("auto", "bfloat16", "float16", "float32"), default="float16")
    parser.add_argument(
        "--task-service-cuda-visible-devices",
        default=os.environ.get("EMAS_AGENTS_CUDA_VISIBLE_DEVICES", "1"),
        help=(
            "Physical GPU IDs exposed only to the managed agents relay-model "
            "subprocess (default: 1). Inside that subprocess the selected GPU "
            "is addressed as cuda:0."
        ),
    )
    parser.add_argument("--task-service-port-base", type=int, default=18090)
    parser.add_argument("--receiver-port-base", type=int, default=19010)
    parser.add_argument(
        "--max-find-insertions",
        type=int,
        default=3,
        help="Maximum number of runtime find/search nodes inserted before one original ungrounded task.",
    )
    parser.add_argument(
        "--save-intermediates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save subgraphs and loop debugging artifacts under the run directory. "
            "Use --no-save-intermediates to keep only final records, required scene state, "
            "and adapter communication files."
        ),
    )
    parser.add_argument(
        "--enable-task-graph-replan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Regenerate the remaining Task Graph when execution requests upstream replanning.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    record = run_hybrid_loop(args)
    print(json.dumps({
        "run_dir": record["run_dir"],
        "all_done": record["all_done"],
        "completed_tasks": record["completed_tasks"],
        "remaining_tasks": record["remaining_tasks"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
