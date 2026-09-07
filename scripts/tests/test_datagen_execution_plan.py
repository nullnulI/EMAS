from __future__ import annotations

import json
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from planning.datagen import generation


def _args() -> Namespace:
    return Namespace(
        scene_name="FloorPlan2",
        task="Put all silverware in the drawer",
        max_task_loops=1,
        disable_qwen=True,
        planning_model_path="unused",
        qwen_conv_mode="unused",
        qwen_num_gpus=1,
        planning_max_new_tokens=128,
        planning_max_attempts=3,
        disable_skill_qwen=True,
        skill_max_steps=1,
        skill_complete_on_execute=True,
        max_task_retries=3,
        agentnum=2,
    )


def _graph() -> dict:
    return {
        "task": "Put all silverware in the drawer",
        "task_graph_version": 7,
        "flat_tasks": [
            {"id": "pick_1", "action": "pick", "depends_on": []},
            {"id": "place_1", "action": "place", "depends_on": ["pick_1"]},
        ],
    }


def _install_sample_stubs(monkeypatch, tmp_path: Path, graph: dict) -> list[list[str]]:
    class Recorder:
        def __init__(self, *_args, **_kwargs):
            pass

        def capture(self, *_args, **_kwargs):
            pass

        def close(self):
            return None

    agent_states = [
        {"agent_id": "0", "skills": ["pick", "place"], "inventoryObjects": []},
        {"agent_id": "1", "skills": ["pick", "place"], "inventoryObjects": []},
    ]
    subgraph_task_ids: list[list[str]] = []

    monkeypatch.setattr(generation, "SampleVideoRecorder", Recorder)
    monkeypatch.setattr(generation, "reset_scene", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(generation, "build_global_scenegraph", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        generation,
        "collect_stage",
        lambda _controller, sample_dir, stage_name, _args: {
            "stage_name": stage_name,
            "stage_dir": str(sample_dir / "stages" / stage_name),
            "agent_states": deepcopy(agent_states),
        },
    )
    monkeypatch.setattr(
        generation,
        "build_stage_scenegraph",
        lambda *_args, **_kwargs: {"scene_graph": None, "relations": None},
    )
    monkeypatch.setattr(
        generation,
        "build_subgraph_and_task_graph",
        lambda *_args, **_kwargs: {
            "subgraph_path": str(tmp_path / "initial_subgraph.json"),
            "task_graph_path": str(tmp_path / "task_graph.json"),
            "subgraph": {},
            "task_graph": deepcopy(graph),
        },
    )

    def fake_loop_subgraph(*, tasks, output_path, **_kwargs):
        task_ids = [str(task["id"]) for task in tasks]
        subgraph_task_ids.append(task_ids)
        return {
            "task_ids": task_ids,
            "subgraph_path": str(output_path),
            "num_nodes": 0,
            "num_edges": 0,
            "subgraph": {"requested_task_ids": task_ids},
        }

    monkeypatch.setattr(generation, "build_loop_task_subgraph", fake_loop_subgraph)
    return subgraph_task_ids


def test_remaining_tasks_preserves_full_semantic_order_and_dependencies():
    graph = _graph()

    tasks = generation.remaining_tasks(graph, completed=set(), failed=set())

    assert [task["id"] for task in tasks] == ["pick_1", "place_1"]
    assert tasks[1]["depends_on"] == ["pick_1"]
    tasks[1]["depends_on"].clear()
    assert graph["flat_tasks"][1]["depends_on"] == ["pick_1"]


def test_datagen_blocked_plan_is_recorded_without_fabricated_dispatch(monkeypatch, tmp_path):
    graph = _graph()
    subgraph_task_ids = _install_sample_stubs(monkeypatch, tmp_path, graph)
    captured_request = {}
    captured_replan = {}
    execution_called = False

    def fake_build_execution_plan(**kwargs):
        captured_request.update(deepcopy(kwargs))
        return {
            "state": "blocked",
            "task_graph_version": 7,
            "graph_fingerprint": "sha256:block",
            "execution_policy": "first_unit_then_replan",
            "units": [],
            "dependency_edges": [],
            "blocking": {
                "code": "resource_deadlock",
                "task_ids": ["pick_1", "place_1"],
                "agent_inventories": {"0": [], "1": []},
                "conflicts": ["fixture conflict"],
                "recommended_recovery": "replan_task_graph",
            },
            "diagnostics": {},
        }

    def fail_if_executed(*_args, **_kwargs):
        nonlocal execution_called
        execution_called = True
        raise AssertionError("blocked plans must not be dispatched")

    monkeypatch.setattr(generation, "build_execution_plan", fake_build_execution_plan)
    monkeypatch.setattr(generation, "execute_allocated_skills", fail_if_executed)
    monkeypatch.setattr(
        generation,
        "replan_datagen_task_graph",
        lambda **kwargs: captured_replan.update(deepcopy(kwargs)) or {
            "status": "failed",
            "trigger": "allocation_blocked",
            "reason": "planning_failed",
            "diagnostics": {"fixture": True},
        },
    )

    record = generation.run_sample(
        SimpleNamespace(last_event=object()),
        0,
        tmp_path,
        _args(),
    )

    assert captured_request["task_graph"] == graph
    assert captured_request["progress"] == {
        "completed_task_ids": [],
        "failed_task_ids": [],
    }
    assert captured_request["task_graph_version"] == 7
    assert captured_request["scene_context"]["requested_task_ids"] == [
        "pick_1",
        "place_1",
    ]
    assert subgraph_task_ids == [["pick_1", "place_1"]]
    assert execution_called is False
    assert record["termination"]["state"] == "blocked"
    assert record["termination"]["stage"] == "allocation_replanning"
    assert record["termination"]["reason_code"] == "allocation_replan_failed"
    assert record["termination"]["blocking"]["recommended_recovery"] == "replan_task_graph"
    assert record["loops"][0]["state"] == "blocked"
    assert captured_replan["trigger"] == "allocation_blocked"
    assert captured_replan["current_task_graph"] == graph

    unit_path = Path(record["loops"][0]["executed_allocation_unit"])
    assert json.loads(unit_path.read_text(encoding="utf-8"))["assignments"] == []


def test_datagen_replaces_whole_graph_after_allocation_block(monkeypatch, tmp_path):
    graph = _graph()
    _install_sample_stubs(monkeypatch, tmp_path, graph)
    monkeypatch.setattr(
        generation,
        "build_execution_plan",
        lambda **_kwargs: {
            "state": "blocked",
            "task_graph_version": 7,
            "graph_fingerprint": "sha256:block",
            "execution_policy": "first_unit_then_replan",
            "units": [],
            "dependency_edges": [],
            "blocking": {
                "code": "invalid_model_plan",
                "task_ids": ["pick_1", "place_1"],
                "agent_inventories": {"0": [], "1": []},
                "conflicts": ["three invalid outputs"],
                "recommended_recovery": "replan_task_graph",
            },
            "diagnostics": {},
        },
    )
    replacement = {
        "task": graph["task"],
        "task_graph_version": 8,
        "flat_tasks": [{"id": "R1", "action": "place", "depends_on": []}],
    }
    monkeypatch.setattr(
        generation,
        "replan_datagen_task_graph",
        lambda **_kwargs: {
            "status": "success",
            "trigger": "allocation_blocked",
            "replan_index": 1,
            "task_graph": deepcopy(replacement),
            "task_graph_path": str(tmp_path / "new_task_graph.json"),
        },
    )

    record = generation.run_sample(
        SimpleNamespace(last_event=object()),
        0,
        tmp_path,
        _args(),
    )

    assert "termination" not in record
    assert record["loops"][0]["state"] == "replanned"
    assert record["loops"][0]["trigger"] == "allocation_blocked"
    assert record["final"]["task_graph_version"] == 8
    assert record["final"]["task_graph_replan_count"] == 1
    assert record["final"]["remaining_tasks"] == ["R1"]


def test_datagen_dispatches_only_validated_first_execution_unit(monkeypatch, tmp_path):
    graph = _graph()
    subgraph_task_ids = _install_sample_stubs(monkeypatch, tmp_path, graph)
    dispatched = []

    monkeypatch.setattr(
        generation,
        "build_execution_plan",
        lambda **_kwargs: {
            "state": "dispatchable",
            "task_graph_version": 7,
            "graph_fingerprint": "sha256:dispatchable",
            "execution_policy": "first_unit_then_replan",
            "units": [
                {
                    "time_step": 1,
                    "assignments": [{"task_id": "pick_1", "agent_id": "0"}],
                },
                {
                    "time_step": 2,
                    "assignments": [{"task_id": "place_1", "agent_id": "0"}],
                },
            ],
            "dependency_edges": [
                {
                    "from_task_id": "pick_1",
                    "to_task_id": "place_1",
                    "kind": "object_handoff",
                }
            ],
            "blocking": None,
            "diagnostics": {},
        },
    )

    def fake_execute(_controller, assignments, **_kwargs):
        dispatched.extend(deepcopy(assignments))
        return {
            "traces": [],
            "execution_time_seconds": 0.0,
            "macro_step_wall_time_seconds": 0.0,
            "simulation_time_seconds": 0.0,
            "collided": False,
            "collision_count": 0,
            "collided_objects": [],
        }

    monkeypatch.setattr(generation, "execute_allocated_skills", fake_execute)
    monkeypatch.setattr(
        generation,
        "evaluate_task_statuses",
        lambda **_kwargs: [
            {
                "subtask_id": "pick_1",
                "subtask": deepcopy(graph["flat_tasks"][0]),
                "status": generation.SUCCESS,
            }
        ],
    )
    merged = {
        "scene_graph": str(tmp_path / "merged_scene_graph.json"),
        "relations": str(tmp_path / "merged_relations.json"),
    }
    monkeypatch.setattr(
        generation,
        "update_stage_scenegraph_incrementally",
        lambda **_kwargs: ({"skipped": True}, deepcopy(merged)),
    )

    record = generation.run_sample(
        SimpleNamespace(last_event=object()),
        0,
        tmp_path,
        _args(),
    )

    assert [item["subtask"]["id"] for item in dispatched] == ["pick_1"]
    assert subgraph_task_ids == [
        ["pick_1", "place_1"],
        ["pick_1", "place_1"],
    ]
    assert record["final"]["completed_tasks"] == ["pick_1"]
    assert record["final"]["remaining_tasks"] == ["place_1"]


def test_datagen_rebuilds_execution_plan_after_each_first_unit(monkeypatch, tmp_path):
    graph = {
        "task": "Perform two dependent interactions",
        "task_graph_version": 7,
        "flat_tasks": [
            {"id": "T1", "action": "open", "depends_on": []},
            {"id": "T2", "action": "toggle_on", "depends_on": ["T1"]},
        ],
    }
    _install_sample_stubs(monkeypatch, tmp_path, graph)
    args = _args()
    args.max_task_loops = 2

    initial_agent_states = [
        {
            "agent_id": "0",
            "skills": ["open", "toggle_on"],
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "inventoryObjects": [],
        },
        {
            "agent_id": "1",
            "skills": ["open", "toggle_on"],
            "position": {"x": 10.0, "y": 0.0, "z": 0.0},
            "inventoryObjects": [],
        },
    ]
    updated_agent_states = [
        {
            "agent_id": "0",
            "skills": ["open", "toggle_on"],
            "position": {"x": 10.0, "y": 0.0, "z": 0.0},
            "inventoryObjects": [],
        },
        {
            "agent_id": "1",
            "skills": ["open", "toggle_on"],
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "inventoryObjects": [],
        },
    ]

    def rolling_collect_stage(_controller, sample_dir, stage_name, _args):
        states = (
            initial_agent_states
            if stage_name in {"initial", "loop_000_pre"}
            else updated_agent_states
        )
        return {
            "stage_name": stage_name,
            "stage_dir": str(sample_dir / "stages" / stage_name),
            "agent_states": deepcopy(states),
        }

    monkeypatch.setattr(generation, "collect_stage", rolling_collect_stage)

    allocator_requests = []
    first_projection = {
        "state": "dispatchable",
        "task_graph_version": 7,
        "graph_fingerprint": "sha256:first",
        "execution_policy": "first_unit_then_replan",
        "units": [
            {
                "time_step": 1,
                "assignments": [{"task_id": "T1", "agent_id": "0"}],
            },
            {
                "time_step": 2,
                "assignments": [{"task_id": "T2", "agent_id": "0"}],
            },
        ],
        "dependency_edges": [
            {
                "from_task_id": "T1",
                "to_task_id": "T2",
                "kind": "semantic",
            }
        ],
        "blocking": None,
        "diagnostics": {},
    }
    second_projection = {
        "state": "dispatchable",
        "task_graph_version": 7,
        "graph_fingerprint": "sha256:second",
        "execution_policy": "first_unit_then_replan",
        "units": [
            {
                "time_step": 1,
                "assignments": [{"task_id": "T2", "agent_id": "1"}],
            }
        ],
        "dependency_edges": [],
        "blocking": None,
        "diagnostics": {},
    }

    def rolling_allocator(**kwargs):
        allocator_requests.append(deepcopy(kwargs))
        projections = [first_projection, second_projection]
        return deepcopy(projections[len(allocator_requests) - 1])

    monkeypatch.setattr(generation, "build_execution_plan", rolling_allocator)

    dispatched = []

    def fake_execute(_controller, assignments, **_kwargs):
        dispatched.extend(
            (item["subtask"]["id"], item["agent_id"])
            for item in deepcopy(assignments)
        )
        return {
            "traces": [],
            "execution_time_seconds": 0.0,
            "macro_step_wall_time_seconds": 0.0,
            "simulation_time_seconds": 0.0,
            "collided": False,
            "collision_count": 0,
            "collided_objects": [],
        }

    monkeypatch.setattr(generation, "execute_allocated_skills", fake_execute)

    def successful_statuses(*, assignments, **_kwargs):
        return [
            {
                "subtask_id": item["subtask"]["id"],
                "subtask": deepcopy(item["subtask"]),
                "agent_id": item["agent_id"],
                "status": generation.SUCCESS,
            }
            for item in assignments
        ]

    monkeypatch.setattr(generation, "evaluate_task_statuses", successful_statuses)
    merged = {
        "scene_graph": str(tmp_path / "merged_scene_graph.json"),
        "relations": str(tmp_path / "merged_relations.json"),
    }
    monkeypatch.setattr(
        generation,
        "update_stage_scenegraph_incrementally",
        lambda **_kwargs: ({"skipped": True}, deepcopy(merged)),
    )

    record = generation.run_sample(
        SimpleNamespace(last_event=object()),
        0,
        tmp_path,
        args,
    )

    assert first_projection["units"][1]["assignments"] == [
        {"task_id": "T2", "agent_id": "0"}
    ]
    assert dispatched == [("T1", "0"), ("T2", "1")]
    assert len(allocator_requests) == 2
    second_request = allocator_requests[1]
    assert second_request["progress"] == {
        "completed_task_ids": ["T1"],
        "failed_task_ids": [],
    }
    assert second_request["agent_states"] == updated_agent_states
    assert second_request["task_graph"] == graph
    assert [task["id"] for task in second_request["task_graph"]["flat_tasks"]] == [
        "T1",
        "T2",
    ]
    assert second_request["task_graph"]["flat_tasks"][1]["depends_on"] == ["T1"]
    assert second_request["scene_context"]["requested_task_ids"] == ["T2"]
    assert record["final"]["completed_tasks"] == ["T1", "T2"]
    assert record["final"]["remaining_tasks"] == []


def test_datagen_terminal_execution_failure_triggers_planning(monkeypatch, tmp_path):
    graph = _graph()
    _install_sample_stubs(monkeypatch, tmp_path, graph)
    monkeypatch.setattr(
        generation,
        "build_execution_plan",
        lambda **_kwargs: {
            "state": "dispatchable",
            "task_graph_version": 7,
            "graph_fingerprint": "sha256:dispatch",
            "execution_policy": "first_unit_then_replan",
            "units": [{
                "time_step": 1,
                "assignments": [{"task_id": "pick_1", "agent_id": "0"}],
            }, {
                "time_step": 2,
                "assignments": [{"task_id": "place_1", "agent_id": "0"}],
            }],
            "dependency_edges": [],
            "blocking": None,
            "diagnostics": {},
        },
    )
    monkeypatch.setattr(
        generation,
        "execute_allocated_skills",
        lambda *_args, **_kwargs: {
            "traces": [],
            "execution_time_seconds": 0.0,
            "macro_step_wall_time_seconds": 0.0,
        },
    )
    monkeypatch.setattr(
        generation,
        "evaluate_task_statuses",
        lambda **_kwargs: [{
            "subtask_id": "pick_1",
            "subtask": deepcopy(graph["flat_tasks"][0]),
            "status": generation.FAILURE,
            "reason": "retry budget exceeded",
        }],
    )
    merged = {
        "scene_graph": str(tmp_path / "merged_scene_graph.json"),
        "relations": str(tmp_path / "merged_relations.json"),
    }
    monkeypatch.setattr(
        generation,
        "update_stage_scenegraph_incrementally",
        lambda **_kwargs: ({"skipped": True}, deepcopy(merged)),
    )
    captured = {}
    replacement = {
        "task": graph["task"],
        "task_graph_version": 8,
        "flat_tasks": [{"id": "R1", "action": "place", "depends_on": []}],
    }
    monkeypatch.setattr(
        generation,
        "replan_datagen_task_graph",
        lambda **kwargs: captured.update(deepcopy(kwargs)) or {
            "status": "success",
            "trigger": "execution_failure",
            "replan_index": 1,
            "task_graph": deepcopy(replacement),
            "task_graph_path": str(tmp_path / "replanned.json"),
        },
    )

    record = generation.run_sample(
        SimpleNamespace(last_event=object()), 0, tmp_path, _args()
    )

    assert captured["trigger"] == "execution_failure"
    assert captured["failed"] == {"pick_1"}
    assert captured["trigger_evidence"]["failed_task_ids"] == ["pick_1"]
    assert record["loops"][0]["state"] == "replanned"
    assert record["loops"][0]["trigger"] == "execution_failure"
    assert record["final"]["task_graph_version"] == 8
