from __future__ import annotations

import pytest

from action_contracts import validate_action_args
from planning.scene_goal_compiler import expand_scene_catalog_task_graph, summarize_scene_catalogue
from planning.task_graph import validate_planner_output
from planning.task_allocation import build_effective_execution_graph, validate_execution_plan


def selector_task(action: str, object_type: str, *, action_args=None, quantifier: str = "one") -> dict:
    return {
        "id": "T1",
        "name": f"{action} {object_type}",
        "description": f"{action} the {object_type}",
        "action": action,
        "action_args": action_args or {},
        "grounding": {
            "node_ids": [],
            "object_tags": [object_type],
            "source_object_tags": [object_type],
            "source_selector": {"quantifier": quantifier, "object_types": [object_type]},
            "relation_texts": [],
        },
        "depends_on": [],
        "termination_check": "action succeeds",
    }


@pytest.mark.parametrize(
    ("action", "arguments", "expected"),
    [
        ("push", {"moveMagnitude": 200}, {"moveMagnitude": 200.0}),
        ("pull", {"moveMagnitude": 0.25}, {"moveMagnitude": 0.25}),
        ("move_held", {"right": 0.1}, {"right": 0.1, "up": 0.0, "ahead": 0.0}),
        ("fill", {"fillLiquid": "WATER"}, {"fillLiquid": "water"}),
    ],
)
def test_safe_action_args_are_normalized(action, arguments, expected) -> None:
    normalized, errors = validate_action_args(action, arguments)
    assert errors == []
    assert normalized == expected


@pytest.mark.parametrize(
    ("action", "arguments", "error_code"),
    [
        ("push", {}, "missing_action_arg"),
        ("pull", {"moveMagnitude": 1001}, "action_arg_out_of_range"),
        ("move_held", {}, "no_op_action_args"),
        ("move_held", {"x": 0.1}, "unsupported_action_args"),
        ("fill", {"fillLiquid": "soap"}, "invalid_action_arg_value"),
        ("break", {"forceAction": True}, "unsupported_action_args"),
    ],
)
def test_safe_action_args_reject_invalid_or_privileged_fields(action, arguments, error_code) -> None:
    _, errors = validate_action_args(action, arguments)
    assert error_code in {item["code"] for item in errors}


@pytest.mark.parametrize(
    ("action", "object_type", "affordance", "action_args"),
    [
        ("push", "Chair", "moveable", {"moveMagnitude": 200}),
        ("pull", "Apple", "pickupable", {"moveMagnitude": 200}),
        ("break", "Mug", "breakable", {}),
        ("cook", "Potato", "cookable", {}),
        ("fill", "Cup", "canFillWithLiquid", {"fillLiquid": "water"}),
    ],
)
def test_selector_compiler_accepts_new_action_affordances(action, object_type, affordance, action_args) -> None:
    task = selector_task(action, object_type, action_args=action_args)
    graph = {"task": task["description"], "flat_tasks": [task]}
    compiled = expand_scene_catalog_task_graph(
        graph,
        [{"objectType": object_type, "objectId": f"{object_type}|1", affordance: True}],
    )
    assert compiled["planner_diagnostics"]["selector_expansion"]["status"] == "expanded"
    assert compiled["flat_tasks"][0]["grounding"]["source_object_ids"] == [f"{object_type}|1"]


def test_push_rejects_object_that_is_neither_moveable_nor_pickupable() -> None:
    task = selector_task("push", "CounterTop", action_args={"moveMagnitude": 200})
    compiled = expand_scene_catalog_task_graph(
        {"task": "Push the countertop", "flat_tasks": [task]},
        [{"objectType": "CounterTop", "objectId": "CounterTop|1"}],
    )
    diagnostics = compiled["planner_diagnostics"]["selector_expansion"]
    assert diagnostics["status"] == "rejected"
    assert "any of ['moveable', 'pickupable']" in diagnostics["reason"]


def test_held_action_requires_one_selector_and_matching_pick_dependency() -> None:
    pick = selector_task("pick", "Mug")
    move = selector_task("move_held", "Mug", action_args={"right": 0.1})
    move["id"] = "T2"
    move["depends_on"] = ["T1"]
    plan = {"reasoning_summary": "Pick and reposition the mug.", "subtasks": [pick, move]}
    catalogue = summarize_scene_catalogue([
        {"objectType": "Mug", "objectId": "Mug|1", "pickupable": True},
    ])
    valid = validate_planner_output(
        plan,
        task="Reposition the held mug",
        subgraph={"task": "Reposition the held mug", "task_spec": {}},
        scene_catalog_summary=catalogue,
    )
    assert valid["status"] == "valid"

    move["depends_on"] = []
    invalid = validate_planner_output(
        {"reasoning_summary": "Reposition the mug.", "subtasks": [move]},
        task="Reposition the held mug",
        subgraph={"task": "Reposition the held mug", "task_spec": {}},
        scene_catalog_summary=catalogue,
    )
    assert any("requires an already-held matching source" in error for error in invalid["errors"])


def test_privileged_native_action_is_not_a_planner_action() -> None:
    task = selector_task("Teleport", "Mug")
    validation = validate_planner_output(
        {"subtasks": [task]},
        task="Move to the mug",
        subgraph={"task": "Move to the mug", "task_spec": {}},
        scene_catalog_summary=[],
    )
    assert validation["status"] == "invalid"
    assert any("invalid action" in error for error in validation["errors"])


def test_allocation_keeps_pick_move_and_drop_on_same_agent() -> None:
    pick = selector_task("pick", "Mug")
    move = selector_task("move_held", "Mug", action_args={"right": 0.1})
    move.update({"id": "T2", "depends_on": ["T1"]})
    drop = selector_task("drop", "Mug")
    drop.update({"id": "T3", "depends_on": ["T2"]})
    graph = {"flat_tasks": [pick, move, drop]}
    agents = [
        {"agent_id": "0", "skills": ["pick", "move_held", "drop"], "held_objects": []},
        {"agent_id": "1", "skills": ["pick", "move_held", "drop"], "held_objects": []},
    ]
    same_agent = {
        "state": "dispatchable",
        "units": [
            {"time_step": 1, "assignments": [{"task_id": "T1", "agent_id": "0"}]},
            {"time_step": 2, "assignments": [{"task_id": "T2", "agent_id": "0"}]},
            {"time_step": 3, "assignments": [{"task_id": "T3", "agent_id": "0"}]},
        ]
    }
    effective = build_effective_execution_graph(graph)
    assert validate_execution_plan(effective, agents, same_agent)["status"] == "valid"

    different_agent = {
        "state": "dispatchable",
        "units": [
            {"time_step": 1, "assignments": [{"task_id": "T1", "agent_id": "0"}]},
            {"time_step": 2, "assignments": [{"task_id": "T2", "agent_id": "1"}]},
            {"time_step": 3, "assignments": [{"task_id": "T3", "agent_id": "1"}]},
        ],
    }
    validation = validate_execution_plan(effective, agents, different_agent)
    assert validation["status"] == "invalid"
    assert any("does not hold source" in error for error in validation["errors"])
