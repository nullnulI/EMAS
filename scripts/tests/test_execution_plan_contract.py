from __future__ import annotations

from copy import deepcopy
import json

import pytest

from planning.task_allocation import build_execution_plan, first_execution_unit


def _agent(
    agent_id: str,
    *,
    skills: tuple[str, ...] = ("pick", "place", "navigate"),
    held: tuple[tuple[str, str], ...] = (),
) -> dict:
    return {
        "agent_id": agent_id,
        "skills": list(skills),
        "held_objects": [
            {"objectType": object_type, "objectId": object_id}
            for object_type, object_id in held
        ],
        "visible_objects": [],
        "inventory_capacity": 1,
    }


def _task(
    task_id: str,
    action: str,
    source_type: str,
    source_id: str,
    *,
    depends_on: tuple[str, ...] = (),
    destination_id: str | None = None,
    expansion: dict | None = None,
) -> dict:
    grounding = {
        "source_object_tags": [source_type],
        "source_object_ids": [source_id],
        "source_selector": {
            "quantifier": "all",
            "object_types": [source_type],
        },
    }
    if destination_id is not None:
        grounding.update(
            {
                "destination_object_tags": ["Drawer"],
                "destination_object_ids": [destination_id],
                "destination_selector": {
                    "quantifier": "one",
                    "object_types": ["Drawer"],
                },
            }
        )
    task = {
        "id": task_id,
        "name": f"{action} {source_type}",
        "description": f"{action} {source_type}",
        "action": action,
        "grounding": grounding,
        "depends_on": list(depends_on),
    }
    if expansion is not None:
        task["runtime"] = {"selector_expansion": deepcopy(expansion)}
    return task


def _graph(*tasks: dict) -> dict:
    return {
        "task": "Put all silverware in the drawer",
        "flat_tasks": list(tasks),
        "dependency_edges": [
            {"from": dependency, "to": task["id"]}
            for task in tasks
            for dependency in task.get("depends_on") or []
        ],
        "chains": [],
    }


def _paired_graph() -> dict:
    pick = _task(
        "P1",
        "pick",
        "Fork",
        "Fork|1",
        expansion={
            "origin_task_id": "PICK_ALL",
            "instance_index": 1,
            "bound_source_object_id": "Fork|1",
        },
    )
    place = _task(
        "L1",
        "place",
        "Fork",
        "Fork|1",
        depends_on=("P1",),
        destination_id="Drawer|1",
        expansion={
            "origin_task_id": "PLACE_ALL",
            "instance_index": 1,
            "bound_source_object_id": "Fork|1",
            "expanded_dependency_origins": ["PICK_ALL"],
        },
    )
    return _graph(pick, place)


def _task19_style_graph() -> dict:
    object_specs = [
        ("ButterKnife", "ButterKnife|1"),
        ("Fork", "Fork|1"),
        ("Knife", "Knife|1"),
        ("Ladle", "Ladle|1"),
        ("Spatula", "Spatula|1"),
        ("Spoon", "Spoon|1"),
    ]
    picks = [
        _task(
            f"P{index}",
            "pick",
            object_type,
            object_id,
            expansion={
                "origin_task_id": "PICK_ALL",
                "instance_index": index,
                "bound_source_object_id": object_id,
            },
        )
        for index, (object_type, object_id) in enumerate(object_specs, start=1)
    ]
    all_pick_ids = tuple(task["id"] for task in picks)
    places = []
    previous_place: str | None = None
    for index, (object_type, object_id) in enumerate(object_specs, start=1):
        place_id = f"L{index}"
        dependencies = all_pick_ids + ((previous_place,) if previous_place else ())
        places.append(
            _task(
                place_id,
                "place",
                object_type,
                object_id,
                depends_on=dependencies,
                destination_id="Drawer|1",
                expansion={
                    "origin_task_id": "PLACE_ALL",
                    "instance_index": index,
                    "bound_source_object_id": object_id,
                    "expanded_dependency_origins": ["PICK_ALL"],
                },
            )
        )
        previous_place = place_id
    return _graph(*picks, *places)


def _semantic_graph_depth(graph: dict) -> int:
    tasks = {str(task["id"]): task for task in graph["flat_tasks"]}
    depths: dict[str, int] = {}

    def task_depth(task_id: str) -> int:
        if task_id not in depths:
            dependencies = [
                str(value)
                for value in tasks[task_id].get("depends_on") or []
                if str(value) in tasks
            ]
            depths[task_id] = 1 + max(
                (task_depth(dependency) for dependency in dependencies),
                default=0,
            )
        return depths[task_id]

    return max((task_depth(task_id) for task_id in tasks), default=0)


class JsonChat:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = [json.dumps(value) for value in responses]
        self.prompts: list[dict] = []
        self.max_new_tokens = 128

    def __call__(self, prompt: str) -> str:
        self.prompts.append(json.loads(prompt))
        return self.responses.pop(0)


def _model_result(
    raw_plan: dict,
    *,
    graph: dict | None = None,
    agents: list[dict] | None = None,
) -> dict:
    return build_execution_plan(
        graph or _paired_graph(),
        agent_states=agents or [_agent("0"), _agent("1")],
        qwen_chat=JsonChat([raw_plan]),
        allocation_max_attempts=1,
    )


def test_deterministic_plan_breaks_selector_expansion_barrier_without_mutation() -> None:
    graph = _task19_style_graph()
    original = deepcopy(graph)

    result = build_execution_plan(
        graph,
        agent_states=[_agent("0")],
        use_qwen=False,
        task_graph_version=7,
    )

    assert graph == original
    assert result["state"] == "dispatchable"
    assert result["task_graph_version"] == 7
    assert result["execution_policy"] == "first_unit_then_replan"
    assert result["graph_fingerprint"].startswith("sha256:")
    assert [
        unit["assignments"][0]["task_id"] for unit in result["units"]
    ] == [
        "P1", "L1",
        "P2", "L2",
        "P3", "L3",
        "P4", "L4",
        "P5", "L5",
        "P6", "L6",
    ]
    assert all(
        assignment["agent_id"] == "0"
        for unit in result["units"]
        for assignment in unit["assignments"]
    )
    edges = {
        (edge["from_task_id"], edge["to_task_id"], edge["kind"])
        for edge in result["dependency_edges"]
    }
    assert {
        ("P1", "L1", "object_handoff"),
        ("P2", "L2", "object_handoff"),
        ("P3", "L3", "object_handoff"),
        ("P4", "L4", "object_handoff"),
        ("P5", "L5", "object_handoff"),
        ("P6", "L6", "object_handoff"),
        ("L1", "L2", "resource_mutex"),
        ("L2", "L3", "resource_mutex"),
        ("L3", "L4", "resource_mutex"),
        ("L4", "L5", "resource_mutex"),
        ("L5", "L6", "resource_mutex"),
    } <= edges


def test_task19_rolls_first_units_with_two_single_slot_agents_until_complete() -> None:
    graph = _task19_style_graph()
    original_graph = deepcopy(graph)
    original_depth = _semantic_graph_depth(graph)
    agents = [_agent("0"), _agent("1")]
    completed: set[str] = set()
    dispatched: list[str] = []
    pick_agent_by_object: dict[str, str] = {}
    initial_edges: list[dict] | None = None

    while len(completed) < 12:
        plan = build_execution_plan(
            graph,
            agent_states=agents,
            progress={
                "completed_task_ids": sorted(completed),
                "failed_task_ids": [],
            },
            task_graph_version=7,
            use_qwen=False,
        )

        assert graph == original_graph
        assert _semantic_graph_depth(graph) == original_depth
        assert plan["state"] == "dispatchable"
        assert plan["units"]
        assert plan["units"][0]["assignments"]
        if initial_edges is None:
            initial_edges = deepcopy(plan["dependency_edges"])

        first_unit = first_execution_unit(plan, graph)
        assert first_unit
        for assignment in first_unit:
            task = assignment["subtask"]
            task_id = str(task["id"])
            agent_id = str(assignment["agent_id"])
            agent = next(item for item in agents if item["agent_id"] == agent_id)
            source_id = str(task["grounding"]["source_object_ids"][0])
            source_type = str(task["grounding"]["source_object_tags"][0])
            if task["action"] == "pick":
                assert agent["held_objects"] == []
                agent["held_objects"].append(
                    {"objectType": source_type, "objectId": source_id}
                )
                pick_agent_by_object[source_id] = agent_id
            else:
                assert task["action"] == "place"
                assert pick_agent_by_object[source_id] == agent_id
                assert agent["held_objects"] == [
                    {"objectType": source_type, "objectId": source_id}
                ]
                agent["held_objects"].clear()
            completed.add(task_id)
            dispatched.append(task_id)

        assert all(len(agent["held_objects"]) <= 1 for agent in agents)

    assert graph == original_graph
    assert _semantic_graph_depth(graph) == original_depth == 7
    assert len(dispatched) == len(set(dispatched)) == 12
    assert set(dispatched) == {*(f"P{index}" for index in range(1, 7)), *(f"L{index}" for index in range(1, 7))}
    assert all(agent["held_objects"] == [] for agent in agents)
    assert initial_edges is not None
    assert sum(edge["kind"] == "object_handoff" for edge in initial_edges) == 6
    assert [
        (edge["from_task_id"], edge["to_task_id"])
        for edge in initial_edges
        if edge["kind"] == "resource_mutex"
    ] == [(f"L{index}", f"L{index + 1}") for index in range(1, 6)]


def test_no_eligible_ready_task_returns_structured_blocked_result() -> None:
    graph = _graph(_task("P1", "pick", "Fork", "Fork|1"))

    result = build_execution_plan(
        graph,
        agent_states=[
            _agent("0", held=(("Bread", "Bread|1"),)),
        ],
        use_qwen=False,
    )

    assert result["state"] == "blocked"
    assert result["units"] == []
    assert result["blocking"]["code"] == "no_eligible_agent"
    assert result["blocking"]["task_ids"] == ["P1"]
    assert result["blocking"]["agent_inventories"]["0"] == [
        {"objectType": "Bread", "objectId": "Bread|1"}
    ]
    assert "inventory is full" in json.dumps(result["blocking"]["conflicts"])


@pytest.mark.parametrize(
    ("raw_plan", "error_fragment"),
    [
        (
            {
                "state": "dispatchable",
                "units": [
                    {
                        "time_step": 1,
                        "assignments": [{"task_id": "UNKNOWN", "agent_id": "0"}],
                    }
                ],
            },
            "unknown task_id",
        ),
        (
            {
                "state": "dispatchable",
                "units": [
                    {
                        "time_step": 1,
                        "assignments": [{"task_id": "P1", "agent_id": "0"}],
                    },
                    {
                        "time_step": 2,
                        "assignments": [{"task_id": "P1", "agent_id": "0"}],
                    },
                    {
                        "time_step": 3,
                        "assignments": [{"task_id": "L1", "agent_id": "0"}],
                    },
                ],
            },
            "scheduled more than once",
        ),
        (
            {
                "state": "dispatchable",
                "units": [
                    {
                        "time_step": 1,
                        "assignments": [{"task_id": "P1", "agent_id": "0"}],
                    }
                ],
            },
            "missing remaining tasks",
        ),
        (
            {
                "state": "dispatchable",
                "units": [
                    {
                        "time_step": 1,
                        "assignments": [{"task_id": "L1", "agent_id": "0"}],
                    },
                    {
                        "time_step": 2,
                        "assignments": [{"task_id": "P1", "agent_id": "0"}],
                    },
                ],
            },
            "runs before dependencies",
        ),
        (
            {
                "state": "dispatchable",
                "units": [
                    {
                        "time_step": 1,
                        "assignments": [{"task_id": "P1", "agent_id": "0"}],
                    },
                    {
                        "time_step": 2,
                        "assignments": [{"task_id": "L1", "agent_id": "1"}],
                    },
                ],
            },
            "must use the agent 0 assigned to pick P1",
        ),
    ],
    ids=["unknown", "duplicate", "omitted", "dependency_order", "handoff"],
)
def test_qwen_units_are_full_plan_validated(raw_plan: dict, error_fragment: str) -> None:
    result = _model_result(raw_plan)

    assert result["state"] == "blocked"
    assert result["blocking"]["code"] == "invalid_model_plan"
    assert error_fragment in " ".join(result["blocking"]["conflicts"])


def test_validator_enforces_one_task_per_agent_per_unit() -> None:
    graph = _graph(
        _task("N1", "navigate", "Fork", "Fork|1"),
        _task("N2", "navigate", "Spoon", "Spoon|1"),
    )
    result = _model_result(
        {
            "state": "dispatchable",
            "units": [
                {
                    "time_step": 1,
                    "assignments": [
                        {"task_id": "N1", "agent_id": "0"},
                        {"task_id": "N2", "agent_id": "0"},
                    ],
                }
            ],
        },
        graph=graph,
    )

    assert result["state"] == "blocked"
    assert "multiple tasks in unit" in " ".join(result["blocking"]["conflicts"])


def test_validator_enforces_skills_and_single_slot_inventory() -> None:
    skill_result = _model_result(
        {
            "state": "dispatchable",
            "units": [
                {
                    "time_step": 1,
                    "assignments": [{"task_id": "P1", "agent_id": "0"}],
                },
                {
                    "time_step": 2,
                    "assignments": [{"task_id": "L1", "agent_id": "0"}],
                },
            ],
        },
        agents=[_agent("0", skills=("place",))],
    )
    assert "does not support action" in json.dumps(
        skill_result["blocking"]["conflicts"]
    )

    inventory_graph = _graph(
        _task("P1", "pick", "Fork", "Fork|1"),
        _task("P2", "pick", "Spoon", "Spoon|1"),
    )
    inventory_result = _model_result(
        {
            "state": "dispatchable",
            "units": [
                {
                    "time_step": 1,
                    "assignments": [{"task_id": "P1", "agent_id": "0"}],
                },
                {
                    "time_step": 2,
                    "assignments": [{"task_id": "P2", "agent_id": "0"}],
                },
            ],
        },
        graph=inventory_graph,
        agents=[_agent("0")],
    )
    assert "inventory is full" in " ".join(
        inventory_result["blocking"]["conflicts"]
    )


def test_validator_enforces_receptacle_mutex() -> None:
    graph = _graph(
        _task("L1", "place", "Fork", "Fork|1", destination_id="Drawer|1"),
        _task("L2", "place", "Spoon", "Spoon|1", destination_id="Drawer|1"),
    )
    result = _model_result(
        {
            "state": "dispatchable",
            "units": [
                {
                    "time_step": 1,
                    "assignments": [
                        {"task_id": "L1", "agent_id": "0"},
                        {"task_id": "L2", "agent_id": "1"},
                    ],
                }
            ],
        },
        graph=graph,
        agents=[
            _agent("0", held=(("Fork", "Fork|1"),)),
            _agent("1", held=(("Spoon", "Spoon|1"),)),
        ],
    )

    assert result["state"] == "blocked"
    assert "used concurrently" in " ".join(result["blocking"]["conflicts"])


def test_three_invalid_qwen_plans_block_without_deterministic_fallback() -> None:
    invalid = {"state": "dispatchable", "units": []}
    chat = JsonChat([invalid, invalid, invalid])

    result = build_execution_plan(
        _paired_graph(),
        agent_states=[_agent("0")],
        qwen_chat=chat,
        allocation_max_attempts=3,
    )

    assert len(chat.prompts) == 3
    assert chat.prompts[0]["request"] == "build_complete_execution_plan"
    assert [prompt["request"] for prompt in chat.prompts[1:]] == [
        "correct_complete_execution_plan",
        "correct_complete_execution_plan",
    ]
    assert result["state"] == "blocked"
    assert result["units"] == []
    assert result["blocking"]["code"] == "invalid_model_plan"
    assert result["diagnostics"]["fallback_used"] is False
    assert result["diagnostics"]["selected_backend"] is None
    assert len(result["diagnostics"]["attempts"]) == 3
