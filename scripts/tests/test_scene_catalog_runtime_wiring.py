from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import pytest

from planning.task_graph import TaskPlanningError
from scripts import benchmark_hybrid_decision_loop as benchmark
from scripts import hybrid_decision_loop as hybrid


def _object(object_type: str, index: int, **state: object) -> dict[str, object]:
    return {
        "objectType": object_type,
        "objectId": f"{object_type}|{index}",
        **state,
    }


def test_episode_catalog_is_copied_into_private_args_state() -> None:
    args = argparse.Namespace(task="open all drawers")
    objects = [_object("Drawer", 1, openable=True)]

    benchmark._install_scene_object_catalog(args, objects)
    objects[0]["objectType"] = "Cabinet"

    assert args._scene_object_catalog == [
        _object("Drawer", 1, openable=True)
    ]
    assert "_scene_object_catalog" not in {
        key: value for key, value in vars(args).items() if not key.startswith("_")
    }


def test_scene_catalog_selectors_expand_planner_graph() -> None:
    planner_graph = {
        "task": "public instruction",
        "planner_backend": "qwen_local",
        "flat_tasks": [{
            "id": "T1", "action": "open", "depends_on": [],
            "grounding": {
                "source_selector": {"quantifier": "all", "object_types": ["HatchPanel"]},
                "source_object_tags": ["HatchPanel"],
            },
        }],
    }
    catalogue = [_object("HatchPanel", index, openable=True) for index in range(1, 10)]

    graph = hybrid.apply_scene_catalog_selectors(planner_graph, catalogue)
    graph = hybrid.rebuild_task_graph(graph)

    assert graph["planner_backend"] == "qwen_local"
    assert len(graph["flat_tasks"]) == 9
    assert {
        task["grounding"]["source_object_ids"][0]
        for task in graph["flat_tasks"]
    } == {f"HatchPanel|{index}" for index in range(1, 10)}

    completed = {graph["flat_tasks"][0]["id"]}
    remaining = [
        task for task in graph["flat_tasks"] if task["id"] not in completed
    ]
    assert len(remaining) == 8


def test_already_satisfied_place_tasks_seed_completed_progress() -> None:
    planner_graph = {
        "task": "public instruction",
        "planner_backend": "qwen_local",
        "flat_tasks": [{
            "id": "T1", "action": "place", "depends_on": [],
            "grounding": {
                "source_selector": {"quantifier": "all", "object_types": ["CargoItem"]},
                "destination_selector": {"quantifier": "one", "object_types": ["StorageBin"]},
                "source_object_tags": ["CargoItem"],
                "destination_object_tags": ["StorageBin"],
            },
        }],
    }
    catalogue = [
        _object("CargoItem", 1, pickupable=True, parentReceptacles=["StorageBin|1"]),
        _object("CargoItem", 2, pickupable=True),
        _object("StorageBin", 1, receptacle=True, openable=True),
    ]

    graph = hybrid.apply_scene_catalog_selectors(planner_graph, catalogue)
    graph = hybrid.rebuild_task_graph(graph)

    completed = hybrid.initial_completed_task_ids(graph)
    remaining = [
        task for task in graph["flat_tasks"] if task["id"] not in completed
    ]

    assert completed == {"T1"}
    assert [task["id"] for task in remaining] == ["T1__instance_002"]
    assert remaining[0]["grounding"]["source_object_ids"] == ["CargoItem|2"]


def test_missing_selector_or_catalog_preserves_planner_tasks() -> None:
    planner_graph = {
        "task": "find a useful object",
        "planner_backend": "qwen_local",
        "flat_tasks": [{"id": "T1", "action": "find", "grounding": {}, "depends_on": []}],
    }
    catalogue = [_object("Widget", 1, openable=True)]

    assert hybrid.apply_scene_catalog_selectors(
        planner_graph, catalogue
    )["flat_tasks"] == planner_graph["flat_tasks"]
    assert hybrid.apply_scene_catalog_selectors(
        planner_graph, None
    )["flat_tasks"] == planner_graph["flat_tasks"]


def test_rejected_interaction_selector_is_a_hard_planning_failure() -> None:
    planner_graph = {
        "task": "open all drawers",
        "planner_backend": "qwen_local",
        "flat_tasks": [{
            "id": "T1", "action": "open", "depends_on": [],
            "grounding": {
                "source_selector": {
                    "quantifier": "all", "object_types": ["Drawer"],
                },
            },
        }],
    }

    with pytest.raises(TaskPlanningError) as captured:
        hybrid.apply_scene_catalog_selectors(planner_graph, None)

    assert captured.value.diagnostics["stage"] == "selector_expansion"
    assert captured.value.diagnostics["selector_expansion"]["status"] == "rejected"
    assert "catalogue is unavailable" in str(captured.value)


def test_planning_failure_file_preserves_diagnostics(tmp_path: Path) -> None:
    error = TaskPlanningError(
        "planner returned an empty task graph",
        diagnostics={
            "status": "failed", "stage": "planning_validation",
            "max_attempts": 3, "attempts": [{"attempt": 1}],
        },
    )
    args = argparse.Namespace(
        task="open all drawers",
        planning_model_path="/models/planner",
        planning_max_attempts=3,
    )

    path = hybrid.save_planning_failure(error, tmp_path, args)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path.name == "planning_failure.json"
    assert payload["code"] == "task_planning_failed"
    assert payload["diagnostics"]["stage"] == "planning_validation"
    assert payload["task"] == "open all drawers"


def test_initial_decomposition_receives_configured_attempt_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        hybrid.gen,
        "extract_task_relevant_subgraph_to_file",
        lambda **kwargs: {"nodes": [], "seed_nodes": []},
    )

    def fake_decompose(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"task": kwargs["task"], "flat_tasks": []}

    monkeypatch.setattr(hybrid, "decompose_task_to_graph", fake_decompose)
    args = argparse.Namespace(
        save_intermediates=False,
        agentnum=2,
        disable_qwen=False,
        planning_model_path="/models/planner",
        qwen_conv_mode="v0_mmtag",
        qwen_num_gpus=1,
        planning_max_new_tokens=512,
        planning_max_attempts=5,
    )

    hybrid.build_initial_task_graph(
        {}, "open all drawers", Path("unused"), args, agent_states=[]
    )

    assert captured["planning_max_attempts"] == 5


def test_complete_object_ids_skip_redundant_runtime_find() -> None:
    empty_subgraph = {"seed_nodes": [], "nodes": []}
    bound_place = {
        "action": "place",
        "grounding": {
            "object_tags": ["Vase", "CounterTop"],
            "source_object_tags": ["Vase"],
            "source_object_ids": ["Vase|1"],
            "destination_object_tags": ["CounterTop"],
            "destination_object_ids": ["CounterTop|1"],
        },
    }
    bound_open = {
        "action": "open",
        "grounding": {
            "object_tags": ["Drawer"],
            "source_object_tags": ["Drawer"],
            "source_object_ids": ["Drawer|1"],
        },
    }

    assert not hybrid.task_requires_runtime_find(bound_place, empty_subgraph)
    assert not hybrid.task_requires_runtime_find(bound_open, empty_subgraph)

    incomplete = deepcopy(bound_place)
    incomplete["grounding"]["destination_object_ids"] = []
    assert hybrid.task_requires_runtime_find(incomplete, empty_subgraph)


def test_new_interaction_actions_are_grounded_skills_and_preserved() -> None:
    expected = {
        "toggle_on": "ToggleObjectOn",
        "toggle_off": "ToggleObjectOff",
        "clean": "CleanObject",
        "slice": "SliceObject",
    }
    for action, relay_action in expected.items():
        assert action in hybrid.OBJECT_GROUNDING_ACTIONS
        assert action in hybrid.DEFAULT_AGENT_SKILLS
        response = {
            "task_normalization": {
                "intentSteps": [{"order": 1, "action": relay_action}]
            }
        }
        assert hybrid.task_service_response_preserves_subtask_action(
            response, {"action": action}
        )
        assert not hybrid.task_service_response_preserves_subtask_action(
            {"task_normalization": {"intentSteps": []}}, {"action": action}
        )
