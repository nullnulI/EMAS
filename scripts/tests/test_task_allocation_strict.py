from __future__ import annotations

import json
from copy import deepcopy

import pytest

from planning.task_allocation import (
    TaskAllocationError,
    build_effective_execution_graph,
    build_execution_plan,
    first_execution_unit,
    validate_execution_plan,
)


AGENTS = [
    {
        "agent_id": "0",
        "position": {"x": 0, "y": 0, "z": 0},
        "visible_objects": [],
        "held_objects": [],
        "inventory_capacity": 1,
        "skills": ["navigate", "pick", "place", "open", "close"],
    },
    {
        "agent_id": "1",
        "position": {"x": 1, "y": 0, "z": 0},
        "visible_objects": [],
        "held_objects": [],
        "inventory_capacity": 1,
        "skills": ["navigate", "pick", "place", "open", "close"],
    },
]


def make_task(
    task_id: str,
    *,
    action: str = "navigate",
    depends_on: list[str] | None = None,
    source_id: str | None = None,
    source_type: str | None = None,
    destination_id: str | None = None,
    quantifier: str | None = None,
    runtime: dict | None = None,
) -> dict:
    grounding: dict = {}
    if source_id is not None:
        grounding["source_object_ids"] = [source_id]
    if source_type is not None:
        grounding["source_object_tags"] = [source_type]
    if quantifier is not None:
        grounding["source_selector"] = {
            "quantifier": quantifier,
            "object_types": [source_type or "Object"],
        }
    if destination_id is not None:
        grounding["destination_object_ids"] = [destination_id]
        grounding["destination_object_tags"] = [destination_id.split("|", 1)[0]]
    task = {
        "id": task_id,
        "name": f"{action} {task_id}",
        "description": f"Perform {action} for {task_id}.",
        "action": action,
        "grounding": grounding,
        "depends_on": list(depends_on or []),
    }
    if runtime is not None:
        task["runtime"] = deepcopy(runtime)
    return task


def make_graph(*tasks: dict) -> dict:
    return {
        "task": "Complete the fixture task",
        "flat_tasks": [deepcopy(task) for task in tasks],
        "dependency_edges": [
            {
                "from_task_id": dependency,
                "to_task_id": task["id"],
                "kind": "semantic",
            }
            for task in tasks
            for dependency in task.get("depends_on") or []
        ],
        "chains": [],
    }


def raw_plan(*steps: list[tuple[str, str]]) -> dict:
    return {
        "state": "dispatchable",
        "units": [
            {
                "time_step": index,
                "assignments": [
                    {"task_id": task_id, "agent_id": agent_id}
                    for task_id, agent_id in assignments
                ],
            }
            for index, assignments in enumerate(steps, start=1)
        ],
    }


def response(*steps: list[tuple[str, str]]) -> str:
    return json.dumps(raw_plan(*steps))


BASIC_GRAPH = make_graph(make_task("T1"), make_task("T2"))
BASIC_RESPONSE = response([("T1", "0"), ("T2", "1")])


class SequencedChat:
    def __init__(self, responses: list[str | BaseException]) -> None:
        self.responses = list(responses)
        self.max_new_tokens = 256
        self.seen_token_limits: list[int] = []
        self.prompts: list[dict] = []
        self.messages: list[dict] = []
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1
        self.messages = []

    def __call__(self, prompt: str) -> str:
        self.seen_token_limits.append(self.max_new_tokens)
        self.prompts.append(json.loads(prompt))
        self.messages.append({"role": "user", "content": prompt})
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        self.messages.append({"role": "assistant", "content": value})
        return value


def selector_expansion(
    origin_task_id: str,
    instance_index: int,
    object_id: str,
    *,
    dependency_origins: list[str] | None = None,
    serialized_after_task_id: str | None = None,
) -> dict:
    return {
        "selector_expansion": {
            "origin_task_id": origin_task_id,
            "instance_index": instance_index,
            "bound_source_object_id": object_id,
            "expanded_dependency_origins": list(dependency_origins or []),
            "serialized_after_task_id": serialized_after_task_id,
        }
    }


def provenance_pick_place_graph() -> dict:
    pick_1 = make_task(
        "pick_silverware",
        action="pick",
        source_id="Fork|1",
        source_type="Fork",
        quantifier="all",
        runtime=selector_expansion("pick_origin", 1, "Fork|1"),
    )
    pick_2 = make_task(
        "pick_silverware_copy",
        action="pick",
        source_id="Spoon|1",
        source_type="Spoon",
        quantifier="all",
        runtime=selector_expansion("pick_origin", 2, "Spoon|1"),
    )
    place_1 = make_task(
        "place_silverware",
        action="place",
        depends_on=[pick_1["id"], pick_2["id"]],
        source_id="Fork|1",
        source_type="Fork",
        destination_id="Drawer|1",
        quantifier="all",
        runtime=selector_expansion(
            "place_origin",
            1,
            "Fork|1",
            dependency_origins=["pick_origin"],
        ),
    )
    place_2 = make_task(
        "place_silverware_copy",
        action="place",
        depends_on=[pick_1["id"], pick_2["id"], place_1["id"]],
        source_id="Spoon|1",
        source_type="Spoon",
        destination_id="Drawer|1",
        quantifier="all",
        runtime=selector_expansion(
            "place_origin",
            2,
            "Spoon|1",
            dependency_origins=["pick_origin"],
            serialized_after_task_id=place_1["id"],
        ),
    )
    return make_graph(pick_1, pick_2, place_1, place_2)


def legacy_pick_place_graph(
    *,
    include_object_ids: bool = True,
    quantifier: str = "all",
) -> dict:
    def object_id(value: str) -> str | None:
        return value if include_object_ids else None

    pick_1 = make_task(
        "PICK__instance_001",
        action="pick",
        source_id=object_id("Fork|1"),
        source_type="Silverware",
        quantifier=quantifier,
    )
    pick_2 = make_task(
        "PICK__instance_002",
        action="pick",
        source_id=object_id("Spoon|1"),
        source_type="Silverware",
        quantifier=quantifier,
    )
    place_1 = make_task(
        "PLACE__instance_001",
        action="place",
        depends_on=[pick_1["id"], pick_2["id"]],
        source_id=object_id("Fork|1"),
        source_type="Silverware",
        destination_id="Drawer|1",
        quantifier=quantifier,
    )
    place_2 = make_task(
        "PLACE__instance_002",
        action="place",
        depends_on=[pick_1["id"], pick_2["id"], place_1["id"]],
        source_id=object_id("Spoon|1"),
        source_type="Silverware",
        destination_id="Drawer|1",
        quantifier=quantifier,
    )
    return make_graph(pick_1, pick_2, place_1, place_2)


def test_qwen_retry_resets_once_restores_tokens_and_writes_raw_files(tmp_path):
    chat = SequencedChat(["{", BASIC_RESPONSE])

    plan = build_execution_plan(
        BASIC_GRAPH,
        agent_states=AGENTS,
        qwen_chat=chat,
        qwen_max_new_tokens=2048,
        allocation_max_attempts=3,
        diagnostics_output_dir=tmp_path,
    )

    assert plan["state"] == "dispatchable"
    assert chat.seen_token_limits == [2048, 2048]
    assert chat.max_new_tokens == 256
    assert chat.reset_count == 1
    assert [prompt["request"] for prompt in chat.prompts] == [
        "build_complete_execution_plan",
        "correct_complete_execution_plan",
    ]
    assert chat.prompts[1]["attempt"] == 2
    assert chat.prompts[1]["previous_validation_errors"] == [
        "execution planning response is not valid JSON"
    ]
    assert [message["role"] for message in chat.messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert plan["diagnostics"]["selected_attempt"] == 2
    assert plan["diagnostics"]["fallback_used"] is False
    assert (tmp_path / "attempt_001_raw.txt").read_text(encoding="utf-8") == "{"
    assert json.loads(
        (tmp_path / "attempt_002_raw.txt").read_text(encoding="utf-8")
    )["units"]


def test_owned_qwen_chat_is_created_once_and_closed_after_retry(monkeypatch):
    from conceptgraph import vlm

    chat = SequencedChat(["{", BASIC_RESPONSE])
    build_calls = []
    close_calls = []
    monkeypatch.setattr(
        vlm,
        "build_vlm_chat",
        lambda **kwargs: build_calls.append(kwargs) or chat,
    )
    monkeypatch.setattr(vlm, "close_vlm_chat", lambda value: close_calls.append(value))

    plan = build_execution_plan(
        BASIC_GRAPH,
        agent_states=AGENTS,
        allocation_max_attempts=3,
    )

    assert plan["state"] == "dispatchable"
    assert len(build_calls) == 1
    assert len(chat.prompts) == 2
    assert chat.reset_count == 1
    assert close_calls == [chat]


def test_shared_qwen_chat_resets_between_execution_plan_sessions():
    chat = SequencedChat([BASIC_RESPONSE, BASIC_RESPONSE])

    build_execution_plan(BASIC_GRAPH, agent_states=AGENTS, qwen_chat=chat)
    build_execution_plan(BASIC_GRAPH, agent_states=AGENTS, qwen_chat=chat)

    assert chat.reset_count == 2
    assert [message["role"] for message in chat.messages] == [
        "system",
        "user",
        "assistant",
    ]


def test_three_invalid_qwen_plans_return_typed_blocked_without_fallback(tmp_path):
    chat = SequencedChat(["{", "[]", "{}"])

    plan = build_execution_plan(
        BASIC_GRAPH,
        agent_states=AGENTS,
        qwen_chat=chat,
        qwen_max_new_tokens=2048,
        allocation_max_attempts=3,
        diagnostics_output_dir=tmp_path,
    )

    assert plan["state"] == "blocked"
    assert plan["units"] == []
    assert plan["blocking"]["code"] == "invalid_model_plan"
    assert plan["blocking"]["recommended_recovery"] == "replan_task_graph"
    assert plan["diagnostics"]["stage"] == "attempts_exhausted"
    assert plan["diagnostics"]["fallback_used"] is False
    assert len(plan["diagnostics"]["attempts"]) == 3
    assert chat.max_new_tokens == 256
    assert chat.reset_count == 1
    assert all(
        (tmp_path / f"attempt_{attempt:03d}_raw.txt").is_file()
        for attempt in (1, 2, 3)
    )


def test_qwen_transport_failure_raises_and_restores_shared_token_limit():
    chat = SequencedChat([RuntimeError("transport unavailable")])

    with pytest.raises(TaskAllocationError) as captured:
        build_execution_plan(
            BASIC_GRAPH,
            agent_states=AGENTS,
            qwen_chat=chat,
            qwen_max_new_tokens=4096,
        )

    assert captured.value.code == "allocation_model_unavailable"
    assert captured.value.diagnostics["stage"] == "model_call"
    assert chat.seen_token_limits == [4096]
    assert chat.max_new_tokens == 256
    assert chat.reset_count == 1


def test_prompt_contains_complete_remaining_graph_not_only_ready_frontier():
    graph = make_graph(
        make_task("T1"),
        make_task("T2", depends_on=["T1"]),
        make_task("T3", depends_on=["T2"]),
    )
    chat = SequencedChat(
        [response([("T1", "0")], [("T2", "1")], [("T3", "0")])]
    )
    scene_context = {
        "nodes": [{"object_tag": "Drawer"}],
        "edges": [{"source": 0, "target": 1}],
    }

    plan = build_execution_plan(
        graph,
        agent_states=AGENTS,
        scene_context=scene_context,
        qwen_chat=chat,
    )

    prompt = chat.prompts[0]
    assert plan["state"] == "dispatchable"
    assert [task["id"] for task in prompt["remaining_tasks"]] == ["T1", "T2", "T3"]
    assert prompt["remaining_tasks"][1]["depends_on"] == ["T1"]
    assert prompt["remaining_tasks"][2]["depends_on"] == ["T2"]
    assert prompt["execution_dependency_edges"] == [
        {"from_task_id": "T1", "to_task_id": "T2", "kind": "semantic"},
        {"from_task_id": "T2", "to_task_id": "T3", "kind": "semantic"},
    ]
    assert prompt["scene_context"]["nodes"] == scene_context["nodes"]
    assert set(prompt["output_schema"]) == {"state", "units"}


@pytest.mark.parametrize(
    ("graph", "plan", "error_text"),
    [
        pytest.param(
            BASIC_GRAPH,
            raw_plan([("T1", "0")]),
            "missing remaining tasks",
            id="missing-task",
        ),
        pytest.param(
            BASIC_GRAPH,
            raw_plan([("T1", "0")], [("T1", "0"), ("T2", "1")]),
            "scheduled more than once",
            id="duplicate-task",
        ),
        pytest.param(
            BASIC_GRAPH,
            raw_plan([("T1", "0")], [("T2", "1"), ("unknown", "0")]),
            "unknown task_id",
            id="unknown-task",
        ),
        pytest.param(
            BASIC_GRAPH,
            raw_plan([("T1", "9"), ("T2", "1")]),
            "unknown agent_id",
            id="unknown-agent",
        ),
        pytest.param(
            BASIC_GRAPH,
            raw_plan([("T1", "0"), ("T2", "0")]),
            "multiple tasks in unit",
            id="same-agent-conflict",
        ),
        pytest.param(
            make_graph(make_task("T1"), make_task("T2", depends_on=["T1"])),
            raw_plan([("T2", "1")], [("T1", "0")]),
            "runs before dependencies",
            id="dependency-order",
        ),
        pytest.param(
            BASIC_GRAPH,
            {
                "state": "dispatchable",
                "units": [
                    {
                        "time_step": 2,
                        "assignments": [
                            {"task_id": "T1", "agent_id": "0", "reason": "extra"},
                            {"task_id": "T2", "agent_id": "1"},
                        ],
                    }
                ],
            },
            "must contain only task_id and agent_id",
            id="strict-schema",
        ),
    ],
)
def test_validator_rejects_invalid_complete_plan_contract(graph, plan, error_text):
    effective = build_effective_execution_graph(graph)

    validation = validate_execution_plan(effective, AGENTS, plan)

    assert validation["status"] == "invalid"
    assert error_text in " ".join(validation["errors"])


def test_validator_rejects_agent_without_required_skill():
    graph = make_graph(
        make_task("P1", action="pick", source_id="Apple|1", source_type="Apple")
    )
    agents = [{**AGENTS[0], "skills": ["navigate"]}]

    validation = validate_execution_plan(
        build_effective_execution_graph(graph),
        agents,
        raw_plan([("P1", "0")]),
    )

    assert validation["status"] == "invalid"
    assert "does not support action 'pick'" in " ".join(validation["errors"])


def test_validator_rejects_pick_when_default_single_slot_inventory_is_full():
    graph = make_graph(
        make_task("P1", action="pick", source_id="Apple|1", source_type="Apple")
    )
    agents = [
        {
            **AGENTS[0],
            "held_objects": [{"objectId": "Bread|1", "objectType": "Bread"}],
        }
    ]

    validation = validate_execution_plan(
        build_effective_execution_graph(graph),
        agents,
        raw_plan([("P1", "0")]),
    )

    assert validation["status"] == "invalid"
    assert "inventory is full" in " ".join(validation["errors"])


def test_atomic_place_without_explicit_pick_acquires_and_places_in_one_task():
    graph = make_graph(
        make_task(
            "A1",
            action="place",
            source_id="Apple|1",
            source_type="Apple",
            destination_id="Bowl|1",
        )
    )

    validation = validate_execution_plan(
        build_effective_execution_graph(graph),
        [AGENTS[0]],
        raw_plan([("A1", "0")]),
    )

    assert validation["status"] == "valid"
    assert validation["final_inventory"] == {"0": []}


def test_atomic_place_consumes_matching_object_already_held_by_agent():
    graph = make_graph(
        make_task(
            "A1",
            action="place",
            source_id="Apple|1",
            source_type="Apple",
            destination_id="Bowl|1",
        )
    )
    agents = [
        {
            **AGENTS[0],
            "held_objects": [{"objectId": "Apple|1", "objectType": "Apple"}],
        }
    ]

    validation = validate_execution_plan(
        build_effective_execution_graph(graph),
        agents,
        raw_plan([("A1", "0")]),
    )

    assert validation["status"] == "valid"
    assert validation["final_inventory"] == {"0": []}


def test_validator_rejects_concurrent_use_of_same_receptacle():
    graph = make_graph(
        make_task(
            "A1",
            action="place",
            source_id="Apple|1",
            source_type="Apple",
            destination_id="Drawer|1",
        ),
        make_task(
            "A2",
            action="place",
            source_id="Spoon|1",
            source_type="Spoon",
            destination_id="Drawer|1",
        ),
    )

    validation = validate_execution_plan(
        build_effective_execution_graph(graph),
        AGENTS,
        raw_plan([("A1", "0"), ("A2", "1")]),
    )

    assert validation["status"] == "invalid"
    assert "used concurrently" in " ".join(validation["errors"])


def test_paired_pick_and_place_same_agent_preserves_inventory_continuity():
    graph = provenance_pick_place_graph()
    effective = build_effective_execution_graph(graph)
    plan = raw_plan(
        [("pick_silverware", "0"), ("pick_silverware_copy", "1")],
        [("place_silverware", "0")],
        [("place_silverware_copy", "1")],
    )

    validation = validate_execution_plan(effective, AGENTS, plan)

    assert validation["status"] == "valid"
    assert validation["final_inventory"] == {"0": [], "1": []}
    assert validation["assigned_agent_by_task"]["pick_silverware"] == "0"
    assert validation["assigned_agent_by_task"]["place_silverware"] == "0"


def test_paired_place_on_different_agent_is_rejected():
    graph = provenance_pick_place_graph()
    plan = raw_plan(
        [("pick_silverware", "0"), ("pick_silverware_copy", "1")],
        [("place_silverware", "1")],
        [("place_silverware_copy", "1")],
    )

    validation = validate_execution_plan(
        build_effective_execution_graph(graph),
        AGENTS,
        plan,
    )

    assert validation["status"] == "invalid"
    assert "must use the agent 0 assigned to pick" in " ".join(validation["errors"])


@pytest.mark.parametrize(
    ("graph", "progress", "agents", "expected_code"),
    [
        pytest.param(
            make_graph(
                make_task("T1", depends_on=["T2"]),
                make_task("T2", depends_on=["T1"]),
            ),
            {},
            AGENTS,
            "dependency_cycle",
            id="cycle",
        ),
        pytest.param(
            make_graph(make_task("T1"), make_task("T2", depends_on=["T1"])),
            {"failed_task_ids": ["T1"]},
            AGENTS,
            "failed_dependency",
            id="failed-dependency",
        ),
        pytest.param(
            make_graph(make_task("T1")),
            {"completed_task_ids": ["T1"]},
            AGENTS,
            "no_remaining_tasks",
            id="nothing-remaining",
        ),
        pytest.param(
            make_graph(
                make_task("P1", action="pick", source_id="Apple|1", source_type="Apple")
            ),
            {},
            [{**AGENTS[0], "skills": ["navigate"]}],
            "no_eligible_agent",
            id="no-skilled-agent",
        ),
    ],
)
def test_preflight_blocks_are_typed_and_never_return_empty_dispatchable(
    graph,
    progress,
    agents,
    expected_code,
):
    plan = build_execution_plan(
        graph,
        agent_states=agents,
        progress=progress,
        use_qwen=False,
    )

    assert plan["state"] == "blocked"
    assert plan["units"] == []
    assert plan["blocking"]["code"] == expected_code
    assert "task_ids" in plan["blocking"]
    assert "agent_inventories" in plan["blocking"]
    assert "conflicts" in plan["blocking"]
    assert plan["blocking"]["recommended_recovery"] == "replan_task_graph"


def test_duplicate_object_ownership_blocks_invalid_agent_state():
    graph = make_graph(make_task("T1"))
    agents = [
        {
            **AGENTS[0],
            "held_objects": [{"objectId": "Apple|1", "objectType": "Apple"}],
        },
        {
            **AGENTS[1],
            "held_objects": [{"objectId": "Apple|1", "objectType": "Apple"}],
        },
    ]

    plan = build_execution_plan(graph, agent_states=agents, use_qwen=False)

    assert plan["state"] == "blocked"
    assert plan["blocking"]["code"] == "invalid_agent_state"
    assert plan["blocking"]["conflicts"] == [
        {
            "code": "object_has_multiple_holders",
            "object_id": "Apple|1",
            "agent_ids": ["0", "1"],
        }
    ]


def test_explicit_deterministic_mode_is_not_a_model_fallback():
    plan = build_execution_plan(
        BASIC_GRAPH,
        agent_states=AGENTS,
        use_qwen=False,
    )

    assert plan["state"] == "dispatchable"
    assert plan["diagnostics"]["selected_backend"] == "deterministic_explicit"
    assert plan["diagnostics"]["fallback_used"] is False
    assert plan["execution_policy"] == "first_unit_then_replan"
    assert {
        assignment["task_id"]
        for unit in plan["units"]
        for assignment in unit["assignments"]
    } == {"T1", "T2"}


def test_fingerprint_is_stable_and_includes_graph_version_and_progress():
    first = build_execution_plan(
        BASIC_GRAPH,
        agent_states=AGENTS,
        task_graph_version=4,
        use_qwen=False,
    )
    repeat = build_execution_plan(
        deepcopy(BASIC_GRAPH),
        agent_states=deepcopy(AGENTS),
        task_graph_version=4,
        use_qwen=False,
    )
    new_version = build_execution_plan(
        BASIC_GRAPH,
        agent_states=AGENTS,
        task_graph_version=5,
        use_qwen=False,
    )
    new_progress = build_execution_plan(
        BASIC_GRAPH,
        agent_states=AGENTS,
        progress={"completed_task_ids": ["T1"]},
        task_graph_version=4,
        use_qwen=False,
    )

    assert first["graph_fingerprint"] == repeat["graph_fingerprint"]
    assert first["graph_fingerprint"].startswith("sha256:")
    assert len(first["graph_fingerprint"].split(":", 1)[1]) == 64
    assert new_version["task_graph_version"] == 5
    assert new_version["graph_fingerprint"] != first["graph_fingerprint"]
    assert new_progress["graph_fingerprint"] != first["graph_fingerprint"]


def test_first_execution_unit_materializes_only_first_unit_and_copies_tasks():
    execution_plan = {
        "state": "dispatchable",
        "units": [
            {
                "time_step": 1,
                "assignments": [{"task_id": "T1", "agent_id": "0"}],
            },
            {
                "time_step": 2,
                "assignments": [{"task_id": "T2", "agent_id": "1"}],
            },
        ],
    }

    assignments = first_execution_unit(execution_plan, BASIC_GRAPH)

    assert [(item["subtask"]["id"], item["agent_id"]) for item in assignments] == [
        ("T1", "0")
    ]
    assignments[0]["subtask"]["name"] = "mutated"
    assert BASIC_GRAPH["flat_tasks"][0]["name"] != "mutated"


def test_first_execution_unit_returns_empty_only_for_blocked_state():
    assert first_execution_unit({"state": "blocked", "units": []}, BASIC_GRAPH) == []


def test_first_execution_unit_rejects_empty_dispatchable_plan():
    with pytest.raises(ValueError, match="no first unit"):
        first_execution_unit(
            {"state": "dispatchable", "units": []},
            BASIC_GRAPH,
        )


def test_execution_graph_and_plan_never_mutate_semantic_task_graph():
    graph = provenance_pick_place_graph()
    before = deepcopy(graph)

    effective = build_effective_execution_graph(graph)
    plan = build_execution_plan(graph, agent_states=AGENTS, use_qwen=False)

    assert graph == before
    assert plan["state"] == "dispatchable"
    assert effective["dependencies"]["place_silverware"] == ["pick_silverware"]
    assert graph["flat_tasks"][2]["depends_on"] == [
        "pick_silverware",
        "pick_silverware_copy",
    ]


def test_provenance_pairing_replaces_aggregate_barrier_with_exact_handoffs():
    effective = build_effective_execution_graph(provenance_pick_place_graph())

    assert effective["pick_for_place"] == {
        "place_silverware": "pick_silverware",
        "place_silverware_copy": "pick_silverware_copy",
    }
    assert effective["dependencies"]["place_silverware"] == ["pick_silverware"]
    assert effective["dependencies"]["place_silverware_copy"] == [
        "pick_silverware_copy",
        "place_silverware",
    ]
    assert {
        (edge["from_task_id"], edge["to_task_id"], edge["kind"])
        for edge in effective["dependency_edges"]
    } == {
        ("pick_silverware", "place_silverware", "object_handoff"),
        ("pick_silverware_copy", "place_silverware_copy", "object_handoff"),
        ("place_silverware", "place_silverware_copy", "resource_mutex"),
    }
    assert effective["pairing_events"][0]["mode"] == "provenance"


def test_legacy_pairing_requires_clone_all_barrier_and_exact_object_id_bijection():
    effective = build_effective_execution_graph(legacy_pick_place_graph())

    assert effective["pick_for_place"] == {
        "PLACE__instance_001": "PICK__instance_001",
        "PLACE__instance_002": "PICK__instance_002",
    }
    assert effective["dependencies"]["PLACE__instance_001"] == [
        "PICK__instance_001"
    ]
    assert effective["dependencies"]["PLACE__instance_002"] == [
        "PICK__instance_002",
        "PLACE__instance_001",
    ]
    assert effective["pairing_events"] == [
        {
            "status": "paired",
            "mode": "legacy_strict",
            "pick_origin": "PICK",
            "place_origin": "PLACE",
            "pairs": {
                "PLACE__instance_001": "PICK__instance_001",
                "PLACE__instance_002": "PICK__instance_002",
            },
        }
    ]


def test_legacy_pairing_refuses_type_only_guessing():
    graph = legacy_pick_place_graph(include_object_ids=False)

    effective = build_effective_execution_graph(graph)

    assert effective["pick_for_place"] == {}
    assert effective["dependencies"]["PLACE__instance_001"] == [
        "PICK__instance_001",
        "PICK__instance_002",
    ]
    assert effective["pairing_events"] == []


def test_legacy_pairing_refuses_non_all_selector_even_with_exact_ids():
    effective = build_effective_execution_graph(
        legacy_pick_place_graph(quantifier="one")
    )

    assert effective["pick_for_place"] == {}
    assert effective["pairing_events"] == []


def test_provenance_pairing_rejects_bound_id_that_disagrees_with_grounding():
    pick_a = make_task(
        "pick-a",
        action="pick",
        source_id="Fork|1",
        source_type="Fork",
        quantifier="all",
        runtime=selector_expansion("pick-origin", 1, "Knife|wrong"),
    )
    pick_b = make_task(
        "pick-b",
        action="pick",
        source_id="Spoon|1",
        source_type="Spoon",
        quantifier="all",
        runtime=selector_expansion("pick-origin", 2, "Spoon|1"),
    )
    place_a = make_task(
        "place-a",
        action="place",
        depends_on=["pick-a", "pick-b"],
        source_id="Fork|1",
        source_type="Fork",
        destination_id="Drawer|1",
        quantifier="all",
        runtime=selector_expansion(
            "place-origin",
            1,
            "Fork|1",
            dependency_origins=["pick-origin"],
        ),
    )
    place_b = make_task(
        "place-b",
        action="place",
        depends_on=["pick-a", "pick-b", "place-a"],
        source_id="Spoon|1",
        source_type="Spoon",
        destination_id="Drawer|1",
        quantifier="all",
        runtime=selector_expansion(
            "place-origin",
            2,
            "Spoon|1",
            dependency_origins=["pick-origin"],
        ),
    )

    effective = build_effective_execution_graph(
        make_graph(pick_a, pick_b, place_a, place_b)
    )

    assert effective["pick_for_place"] == {}
    assert effective["pairing_events"] == [
        {
            "status": "rejected",
            "mode": "provenance",
            "pick_origin": "pick-origin",
            "place_origin": "place-origin",
            "reason": "bound_source_object_id_does_not_match_grounding",
        }
    ]


def test_progress_removes_completed_tasks_but_preserves_failed_dependency_signal():
    graph = make_graph(
        make_task("T1"),
        make_task("T2", depends_on=["T1"]),
        make_task("T3", depends_on=["T2"]),
    )

    after_completion = build_effective_execution_graph(
        graph,
        {"completed_task_ids": ["T1"]},
    )
    after_failure = build_effective_execution_graph(
        graph,
        {"failed_task_ids": ["T1"]},
    )

    assert after_completion["task_order"] == ["T2", "T3"]
    assert after_completion["dependencies"]["T2"] == []
    assert after_failure["failed_dependencies"] == {"T2": ["T1"]}

