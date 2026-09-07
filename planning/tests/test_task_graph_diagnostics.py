from __future__ import annotations

import json

import pytest

from planning.task_graph import (
    TaskPlanningError,
    audit_plan_entity_coverage,
    decompose_task_to_graph,
    ensure_open_receptacles_closed,
)


SUBGRAPH = {
    "task": "Put the bread, lettuce, and tomato in the fridge",
    "task_spec": {
        "target_objects": ["bread", "lettuce", "tomato", "fridge"],
        "source_receptacles": [],
        "destination_receptacles": [],
        "landmarks": [],
    },
    "seed_nodes": [{"pruned_id": 0, "object_tag": "loaf of bread"}],
    "nodes": [
        {
            "pruned_id": 0,
            "object_tag": "loaf of bread",
            "caption": "a loaf of bread",
            "possible_tags": ["bread", "loaf"],
        }
    ],
    "triples": [],
}


def critic_response(prompt: str) -> str | None:
    if '"request": "validate_candidate_plan"' not in prompt:
        return None
    return json.dumps({"status": "valid", "errors": []})


def intent_response(
    prompt: str,
    *,
    action: str,
    source_types: set[str],
    destination_types: set[str] | None = None,
    source_quantifier: str = "one",
) -> str | None:
    """Return a protocol-valid response for either intent-resolution stage."""
    payload = json.loads(prompt)
    if payload.get("request") != "resolve_task_intent":
        return None
    if payload.get("stage") == "action":
        return json.dumps({
            "status": "resolved",
            "action": action,
            "summary": "The requested interaction action is selected.",
        })
    if payload.get("stage") != "roles":
        return None

    destination_types = destination_types or set()

    def selected(row: dict, expected: set[str]) -> bool:
        label = str(row.get("object_type") or row.get("object_tag") or "").casefold()
        return any(value.casefold() in label or label in value.casefold() for value in expected)

    roles = {}
    for role, rows in payload.get("role_tables", {}).items():
        expected = source_types if role == "source" else destination_types
        roles[role] = {
            "status": "specified" if role == "source" or expected else "unspecified",
            "quantifier": source_quantifier if role == "source" else "one",
            "classifications": [
                "included" if selected(row, expected) else "excluded"
                for row in rows
            ],
        }
    return json.dumps({
        "status": "resolved",
        "roles": roles,
        "semantic_args": {},
        "summary": "All action roles are classified.",
    })


class InvalidJsonChat:
    def __call__(self, prompt: str) -> str:
        response = intent_response(
            prompt,
            action="place",
            source_types={"bread"},
        )
        if response is not None:
            return response
        return "I cannot return the requested JSON."


class RaisingChat:
    def __call__(self, prompt: str) -> str:
        response = intent_response(
            prompt,
            action="place",
            source_types={"bread"},
        )
        if response is not None:
            return response
        raise RuntimeError("synthetic planner failure")


class ValidChat:
    def __call__(self, prompt: str) -> str:
        response = intent_response(
            prompt,
            action="pick",
            source_types={"bread"},
        )
        if response is not None:
            return response
        critic = critic_response(prompt)
        if critic is not None:
            return critic
        return """
        {
          "reasoning_summary": "Pick up bread.",
          "subtasks": [
            {
              "id": "T1",
              "name": "Pick up bread",
              "description": "Pick up the bread.",
              "action": "pick",
              "grounding": {
                "node_ids": [0],
                "object_tags": ["loaf of bread"],
                "source_node_ids": [0],
                "source_object_tags": ["loaf of bread"],
                "relation_texts": []
              },
              "depends_on": [],
              "termination_check": "Bread is held."
            }
          ]
        }
        """


class TokenAwareChat(ValidChat):
    def __init__(self):
        self.max_new_tokens = 256
        self.observed_max_new_tokens = None

    def __call__(self, prompt: str) -> str:
        if critic_response(prompt) is None:
            self.observed_max_new_tokens = self.max_new_tokens
        return super().__call__(prompt)


class MissingCloseChat:
    def __call__(self, prompt: str) -> str:
        response = intent_response(
            prompt,
            action="place",
            source_types={"Bread"},
            destination_types={"Fridge"},
        )
        if response is not None:
            return response
        critic = critic_response(prompt)
        if critic is not None:
            return critic
        return """
        {
          "reasoning_summary": "Open, place, then finish.",
          "subtasks": [
            {
              "id": "T1",
              "name": "Open fridge",
              "description": "Open the fridge.",
              "action": "open",
              "grounding": {"node_ids": [], "object_tags": ["Fridge"], "relation_texts": [],
                "source_selector": {"object_types": ["Fridge"], "quantifier": "one"}},
              "depends_on": [],
              "termination_check": "fridge is open."
            },
            {
              "id": "T2",
              "name": "Place bread in fridge",
              "description": "Put the bread inside the fridge.",
              "action": "place",
              "grounding": {"node_ids": [0], "object_tags": ["Bread", "Fridge"], "relation_texts": ["bread in fridge"],
                "source_selector": {"object_types": ["Bread"], "quantifier": "one"},
                "destination_selector": {"object_types": ["Fridge"], "quantifier": "one"}},
              "depends_on": ["T1"],
              "termination_check": "bread is in the fridge."
            }
          ]
        }
        """


class SliceWithToolDiscoveryChat:
    def __call__(self, prompt: str) -> str:
        response = intent_response(
            prompt,
            action="slice",
            source_types={"Tomato", "Egg"},
        )
        if response is not None:
            return response
        critic = critic_response(prompt)
        if critic is not None:
            return critic
        return """
        {
          "reasoning_summary": "Find a knife, then slice each food.",
          "subtasks": [
            {
              "id": "T1", "name": "Locate the knife", "description": "Find the knife.",
              "action": "find",
              "grounding": {"node_ids": [], "object_tags": ["knife"], "relation_texts": []},
              "depends_on": [], "termination_check": "Knife is located."
            },
            {
              "id": "T2", "name": "Slice the tomato", "description": "Cut the tomato.",
              "action": "slice",
              "grounding": {"node_ids": [], "object_tags": ["Tomato"], "relation_texts": [],
                "source_selector": {"object_types": ["Tomato"], "quantifier": "one"}},
              "depends_on": ["T1"], "termination_check": "Tomato is sliced."
            },
            {
              "id": "T3", "name": "Slice the egg", "description": "Slice the egg.",
              "action": "slice",
              "grounding": {"node_ids": [], "object_tags": ["Egg"], "relation_texts": [],
                "source_selector": {"object_types": ["Egg"], "quantifier": "one"}},
              "depends_on": ["T1"], "termination_check": "Egg is sliced."
            }
          ]
        }
        """


def test_invalid_json_exhausts_attempts_without_heuristic_fallback():
    with pytest.raises(TaskPlanningError) as raised:
        decompose_task_to_graph(
            SUBGRAPH["task"],
            SUBGRAPH,
            qwen_chat=InvalidJsonChat(),
        )

    qwen = raised.value.diagnostics["qwen"]
    assert qwen["status"] == "failed"
    assert qwen["stage"] == "attempts_exhausted"
    assert qwen["attempt_count"] == 3
    assert qwen["json_parse_status"] == "invalid_json"
    assert "cannot return" in qwen["raw_response_preview"]


def test_model_exception_records_type_and_message_then_fails_closed():
    with pytest.raises(TaskPlanningError) as raised:
        decompose_task_to_graph(
            SUBGRAPH["task"],
            SUBGRAPH,
            qwen_chat=RaisingChat(),
        )

    qwen = raised.value.diagnostics["qwen"]
    assert qwen["stage"] == "attempts_exhausted"
    assert qwen["error_type"] == "RuntimeError"
    assert qwen["error_message"] == "synthetic planner failure"


def test_entity_coverage_is_audit_only_and_reports_missing_entities():
    candidate = {
        "flat_tasks": [{
            "name": "Navigate to bread",
            "description": "Move to the bread.",
            "grounding": {"object_tags": ["bread"]},
        }]
    }
    coverage = audit_plan_entity_coverage(SUBGRAPH["task"], SUBGRAPH, candidate)

    assert coverage["covered_entities"] == ["bread"]
    assert coverage["missing_entities"] == ["lettuce", "tomato", "fridge"]
    assert coverage["complete"] is False
    assert coverage["enforced"] is False


def test_disabled_qwen_fails_closed_without_attempting_model_call():
    with pytest.raises(TaskPlanningError) as raised:
        decompose_task_to_graph(
            "go to bread",
            SUBGRAPH,
            use_qwen=False,
        )

    qwen = raised.value.diagnostics["qwen"]
    assert raised.value.code == "planning_disabled"
    assert qwen["attempted"] is False
    assert qwen["status"] == "disabled"


def test_planning_temporarily_raises_generation_limit_and_restores_shared_chat():
    chat = TokenAwareChat()

    token_subgraph = {
        "task": "Pick up bread",
        "task_spec": {"target_objects": ["bread"]},
        "nodes": SUBGRAPH["nodes"],
        "triples": [],
    }
    graph = decompose_task_to_graph(
        token_subgraph["task"],
        token_subgraph,
        qwen_chat=chat,
        qwen_max_new_tokens=1536,
    )

    assert graph["planner_backend"] == "qwen_local"
    assert chat.observed_max_new_tokens == 1536
    assert chat.max_new_tokens == 256
    assert graph["planner_diagnostics"]["qwen"]["max_new_tokens"] == 1536


def receptacle_plan(*, include_close: bool = False) -> dict:
    subtasks = [
        {
            "id": "T1",
            "name": "Open fridge",
            "description": "Open the fridge.",
            "action": "open",
            "grounding": {
                "node_ids": [],
                "object_tags": ["fridge"],
                "relation_texts": [],
                "status": "unresolved",
                "missing_reason": "fridge is not currently grounded",
                "recovery": "search_visible_scene",
            },
            "depends_on": [],
            "termination_check": "fridge is open.",
        },
        {
            "id": "T2",
            "name": "Place bread in fridge",
            "description": "Put the bread inside the fridge.",
            "action": "place",
            "grounding": {
                "node_ids": [],
                "object_tags": ["bread"],
                "relation_texts": ["bread in fridge"],
                "status": "unresolved",
                "missing_reason": "bread is not currently grounded",
                "recovery": "search_visible_scene",
            },
            "depends_on": ["T1"],
            "termination_check": "bread is in the fridge.",
        },
        {
            "id": "T3",
            "name": "Place tomato in fridge",
            "description": "Put the tomato inside the fridge.",
            "action": "place",
            "grounding": {
                "node_ids": [],
                "object_tags": ["tomato"],
                "relation_texts": ["tomato in fridge"],
                "status": "unresolved",
                "missing_reason": "tomato is not currently grounded",
                "recovery": "search_visible_scene",
            },
            "depends_on": ["T1"],
            "termination_check": "tomato is in the fridge.",
        },
    ]
    if include_close:
        subtasks.append(
            {
                "id": "T4",
                "name": "Close fridge",
                "description": "Close the fridge.",
                "action": "close",
                "grounding": {
                    "node_ids": [],
                    "object_tags": ["fridge"],
                    "relation_texts": [],
                    "status": "unresolved",
                    "missing_reason": "fridge is not currently grounded",
                    "recovery": "search_visible_scene",
                },
                "depends_on": ["T2", "T3"],
                "termination_check": "fridge is closed.",
            }
        )
    return {
        "reasoning_summary": "Store food.",
        "planner_backend": "qwen_local",
        "subtasks": subtasks,
    }


def test_completion_repair_appends_close_after_all_related_placements():
    repaired, repairs = ensure_open_receptacles_closed(
        "Put the bread and tomato in the fridge",
        receptacle_plan(),
    )

    close_tasks = [task for task in repaired["subtasks"] if task["action"] == "close"]
    assert len(close_tasks) == 1
    assert close_tasks[0]["id"] == "T4"
    assert close_tasks[0]["grounding"]["object_tags"] == ["fridge"]
    assert close_tasks[0]["depends_on"] == ["T2", "T3"]
    assert repairs == [
        {
            "kind": "append_missing_close",
            "task_id": "T4",
            "target": "fridge",
            "depends_on": ["T2", "T3"],
            "source_open_task_id": "T1",
        }
    ]


def test_completion_repair_is_idempotent_when_close_already_exists():
    original = receptacle_plan(include_close=True)

    repaired, repairs = ensure_open_receptacles_closed(
        "Put the bread and tomato in the fridge",
        original,
    )

    assert repaired == original
    assert repairs == []
    assert len([task for task in repaired["subtasks"] if task["action"] == "close"]) == 1


def test_completion_repair_preserves_explicit_leave_open_instruction():
    original = receptacle_plan()

    repaired, repairs = ensure_open_receptacles_closed(
        "Put the bread and tomato in the fridge and leave the fridge open",
        original,
    )

    assert repaired == original
    assert repairs == []


def test_completion_repair_does_not_close_an_open_only_task():
    original = receptacle_plan()
    original["subtasks"] = original["subtasks"][:1]

    repaired, repairs = ensure_open_receptacles_closed("Open the fridge", original)

    assert repaired == original
    assert repairs == []


def test_decomposition_records_and_assembles_missing_close_repair():
    graph = decompose_task_to_graph(
        "Put the bread in the fridge",
        SUBGRAPH,
        qwen_chat=MissingCloseChat(),
        scene_catalog=[
            {"objectType": "Bread", "objectId": "Bread|1", "pickupable": True},
            {"objectType": "Fridge", "objectId": "Fridge|1", "openable": True, "receptacle": True},
        ],
    )

    close_task = next(task for task in graph["flat_tasks"] if task["action"] == "close")
    assert close_task["depends_on"] == ["T2"]
    assert {"from": "T2", "to": close_task["id"]} in graph["dependency_edges"]
    assert graph["planner_diagnostics"]["completion_repairs"][0]["target"] == "Fridge"


def test_slice_decomposition_removes_redundant_tool_discovery_and_keeps_atomic_goals():
    slice_subgraph = {
        "task": "Slice the tomato and egg",
        "task_spec": {"target_objects": ["tomato", "egg"]},
        "nodes": [],
        "triples": [],
    }

    graph = decompose_task_to_graph(
        slice_subgraph["task"],
        slice_subgraph,
        qwen_chat=SliceWithToolDiscoveryChat(),
        scene_catalog=[
            {"objectType": "Tomato", "objectId": "Tomato|1", "sliceable": True},
            {"objectType": "Egg", "objectId": "Egg|1", "sliceable": True},
            {"objectType": "Knife", "objectId": "Knife|1", "pickupable": True},
        ],
    )

    assert [(task["action"], task["grounding"]["object_tags"]) for task in graph["flat_tasks"]] == [
        ("slice", ["Tomato"]),
        ("slice", ["Egg"]),
    ]
    assert all(task["depends_on"] == [] for task in graph["flat_tasks"])
    assert graph["planner_diagnostics"]["slice_tool_repairs"][0]["task_id"] == "T1"
