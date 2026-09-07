from __future__ import annotations

import json
from typing import Any

from planning.task_graph import (
    decompose_task_to_graph,
    sanitize_subtasks,
    semantic_violations_challenge_resolved_intent,
    validate_planner_output,
)


TASK = "Put the apple and tomato in the bowl."


def _subgraph() -> dict[str, Any]:
    return {
        "task": TASK,
        "task_spec": {
            "raw_task": TASK,
            "task_type": "place",
            "actions": ["place"],
            "target_objects": ["Apple", "Tomato"],
            "destination_receptacles": ["Bowl"],
            "spatial_relations": ["in"],
        },
        "seed_nodes": [],
        "nodes": [
            {
                "pruned_id": 1,
                "original_id": 101,
                "object_tag": "Apple",
                "caption": "a red apple",
                "possible_tags": ["Apple", "fruit"],
            },
            {
                "pruned_id": 2,
                "original_id": 102,
                "object_tag": "Tomato",
                "caption": "a tomato",
                "possible_tags": ["Tomato", "fruit"],
            },
            {
                "pruned_id": 9,
                "original_id": 109,
                "object_tag": "Bowl",
                "caption": "a bowl",
                "possible_tags": ["Bowl", "receptacle"],
            },
        ],
        "triples": [],
    }


def _place_subtask(task_id: str, source: str, source_node_id: int) -> dict[str, Any]:
    return {
        "id": task_id,
        "name": f"Place {source}",
        "description": f"Put the {source} in the Bowl.",
        "action": "place",
        "grounding": {
            "node_ids": [source_node_id, 9],
            "source_node_ids": [source_node_id],
            "destination_node_ids": [9],
            "object_tags": [source, "Bowl"],
            "source_object_tags": [source],
            "destination_object_tags": ["Bowl"],
            "relation_texts": [f"{source} is in Bowl"],
            "status": "grounded",
            "missing_reason": "",
            "recovery": "none",
        },
        "depends_on": [],
        "termination_check": f"The {source} is in the Bowl.",
    }


def _plan() -> dict[str, Any]:
    return {
        "reasoning_summary": "Place both objects in the bowl.",
        "subtasks": [
            _place_subtask("T1", "Apple", 1),
            _place_subtask("T2", "Tomato", 2),
        ],
    }


def _root_intent() -> dict[str, Any]:
    return {
        "action": "place",
        "explicit_actions": ["place"],
        "quantifier": "one",
        "roles": {
            "source": ["Apple", "Tomato"],
            "destination": ["Bowl"],
        },
        "role_quantifiers": {"source": "one", "destination": "one"},
        "role_statuses": {"source": "specified", "destination": "specified"},
        "role_nodes": {"source": [1, 2], "destination": [9]},
        "role_tags": {
            "source": ["Apple", "Tomato"],
            "destination": ["Bowl"],
        },
        "grounding_mode": "subgraph",
        "semantic_args": {},
        "catalogue_constrained": False,
        "semantic_goal_resolved": True,
        "intent_resolution_source": "qwen_two_stage_intent",
    }


class _NodeModeChat:
    def reset(self) -> None:
        return None

    def __call__(self, prompt: str) -> str:
        payload = json.loads(prompt)
        if payload.get("request") == "resolve_task_intent":
            if payload.get("stage") == "action":
                return json.dumps({
                    "status": "resolved",
                    "action": "place",
                    "summary": "The instruction requests placement.",
                })
            selected_ids = {"source": {1, 2}, "destination": {9}}
            roles = {}
            for role, rows in payload["role_tables"].items():
                roles[role] = {
                    "status": "specified",
                    "quantifier": "one",
                    "classifications": [
                        "included"
                        if row["node_id"] in selected_ids[role]
                        else "excluded"
                        for row in rows
                    ],
                }
            return json.dumps({
                "status": "resolved",
                "roles": roles,
                "semantic_args": {},
                "summary": "Resolved both sources and their shared destination.",
            })
        if payload.get("request") == "validate_candidate_plan":
            return json.dumps({"status": "valid", "errors": []})
        return json.dumps(_plan())


def test_sanitize_preserves_explicit_subgraph_role_node_ids() -> None:
    normalized = sanitize_subtasks(_plan(), _subgraph(), TASK)

    assert [
        task["grounding"]["source_node_ids"]
        for task in normalized["subtasks"]
    ] == [[1], [2]]
    assert [
        task["grounding"]["destination_node_ids"]
        for task in normalized["subtasks"]
    ] == [[9], [9]]


def test_decompose_preserves_role_nodes_and_allows_shared_destination_node() -> None:
    graph = decompose_task_to_graph(
        TASK,
        _subgraph(),
        qwen_chat=_NodeModeChat(),
        scene_catalog=None,
        planning_max_attempts=1,
    )

    assert [
        task["grounding"]["source_node_ids"]
        for task in graph["flat_tasks"]
    ] == [[1], [2]]
    assert [
        task["grounding"]["destination_node_ids"]
        for task in graph["flat_tasks"]
    ] == [[9], [9]]
    assert graph["planner_diagnostics"]["root_intent_validation"]["status"] == "valid"


def test_subgraph_mode_rejects_nonempty_scene_selectors() -> None:
    candidate = _plan()
    for subtask in candidate["subtasks"]:
        source = subtask["grounding"]["source_object_tags"][0]
        subtask["grounding"]["source_selector"] = {
            "quantifier": "one",
            "object_types": [source],
        }
        subtask["grounding"]["destination_selector"] = {
            "quantifier": "one",
            "object_types": ["Bowl"],
        }

    validation = validate_planner_output(
        candidate,
        task=TASK,
        subgraph=_subgraph(),
        scene_catalog_summary=[],
        root_intent_override=_root_intent(),
    )

    assert validation["status"] == "invalid"
    for role in ("source", "destination"):
        assert any(
            f"{role}_selector" in error and "subgraph" in error.lower()
            for error in validation["errors"]
        )


def test_only_explicit_resolver_level_errors_request_intent_reresolution() -> None:
    ordinary_critic_errors = [
        {
            "code": "wrong_destination",
            "field": "grounding.destination_selector",
            "message": "The requested destination is unsuitable.",
            "required_fix": "Choose another destination.",
        },
        {
            "code": "wrong_action",
            "field": "action",
            "message": "The candidate uses the wrong action.",
            "required_fix": "Correct the planner subtask.",
        },
        {
            "code": "wrong_quantifier",
            "field": "grounding.source_selector.quantifier",
            "message": "The planner emitted an invalid quantifier.",
            "required_fix": "Correct the planner subtask.",
        },
    ]

    for violation in ordinary_critic_errors:
        assert not semantic_violations_challenge_resolved_intent([violation])

    assert semantic_violations_challenge_resolved_intent([{
        "code": "destination_suitability_mismatch",
        "field": "grounding.destination_selector",
        "message": "The locked resolver destination is unsuitable.",
        "required_fix": "Resolve the destination again.",
    }])

    assert semantic_violations_challenge_resolved_intent([{
        "code": "resolved_intent_mismatch",
        "field": "resolved_semantic_goal.destination_nodes",
        "message": "The resolved intent misunderstood the destination.",
        "required_fix": "Resolve the instruction again.",
    }])
