from __future__ import annotations

import json

import pytest

from planning.scene_goal_compiler import expand_scene_catalog_task_graph, summarize_scene_catalogue
from planning.task_graph import (
    TaskPlanningError,
    decompose_task_to_graph,
    derive_root_intent,
    enforce_semantic_critic_scope,
    role_intent_resolution_request,
    semantic_action_contracts_for_critic,
    semantic_critic_plan_projection,
    semantic_goal_resolution_request,
    validate_semantic_goal_resolution,
    validate_planner_output,
)


TASK = "Open all the drawers"
SUBGRAPH = {
    "task": TASK,
    "task_spec": {"raw_task": TASK, "parser": "disabled_scenegraph"},
    "seed_nodes": [],
    "nodes": [],
    "triples": [],
}
CATALOGUE = [
    {
        "objectType": "Drawer",
        "objectId": f"Drawer|{index:02d}",
        "openable": True,
    }
    for index in range(1, 10)
] + [
    {"objectType": "Cabinet", "objectId": "Cabinet|01", "openable": True},
]


def plan(action: str, *, quantifier: str = "all", object_type: str = "Drawer") -> str:
    return json.dumps({
        "reasoning_summary": "Open the selected scene instances.",
        "subtasks": [{
            "id": "T1",
            "name": "Open selected objects",
            "description": "Open every selected object.",
            "action": action,
            "grounding": {
                "node_ids": [],
                "object_tags": [object_type],
                "source_object_tags": [object_type],
                "source_selector": {
                    "quantifier": quantifier,
                    "object_types": [object_type],
                },
                "relation_texts": [],
            },
            "depends_on": [],
            "termination_check": "All selected objects are open.",
        }],
    })


class SequencedChat:
    def __init__(
        self,
        responses: list[str],
        critic_responses: list[str] | None = None,
        resolver_responses: list[str] | None = None,
    ) -> None:
        self.responses = list(responses)
        self.critic_responses = list(critic_responses or [])
        self.resolver_responses = list(resolver_responses or [])
        self.prompts: list[dict] = []
        self.critic_prompts: list[dict] = []
        self.resolver_prompts: list[dict] = []
        self.action_prompts: list[dict] = []
        self.role_prompts: list[dict] = []
        self.messages: list[dict] = []
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def _default_action(self, payload: dict) -> str:
        text = str(payload.get("instruction") or "").lower()
        patterns = (
            (("turn off", "switch off", "toggle off", "dark"), "toggle_off"),
            (("turn on", "switch on", "toggle on"), "toggle_on"),
            (("place", "put", "clear", "store", "move everything"), "place"),
            (("open",), "open"),
            (("close", "shut"), "close"),
            (("slice", "cut"), "slice"),
            (("clean", "wash"), "clean"),
            (("fill",), "fill"),
            (("cook", "heat"), "cook"),
            (("break", "smash"), "break"),
            (("push",), "push"),
            (("pull",), "pull"),
            (("drop",), "drop"),
            (("pick", "grab", "take"), "pick"),
        )
        for needles, action in patterns:
            if any(needle in text for needle in needles):
                return action
        allowed = {
            item.get("action") for item in payload.get("allowed_actions") or []
        }
        for response in reversed(self.responses):
            try:
                candidate = json.loads(response)
            except (TypeError, json.JSONDecodeError):
                continue
            for task in candidate.get("subtasks") or []:
                action = str(task.get("action") or "")
                if action in allowed:
                    return action
        return next(iter(allowed))

    @staticmethod
    def _words(value: str) -> set[str]:
        expanded = "".join(
            (" " + character.lower()) if character.isupper() else character
            for character in str(value)
        )
        cleaned = "".join(
            character if character.isalnum() else " "
            for character in expanded.lower()
        )
        return {
            (
                word[:-1]
                if len(word) > 3 and word.endswith("s") and not word.endswith("ss")
                else word
            )

            for word in cleaned.split()
            if word
        }

    def _planned_role_types(self, action: str) -> tuple[dict[str, set[str]], str]:
        for response in reversed(self.responses):
            try:
                candidate = json.loads(response)
            except (TypeError, json.JSONDecodeError):
                continue
            selected = {"source": set(), "destination": set()}
            quantifier = "one"
            for task in candidate.get("subtasks") or []:
                if not isinstance(task, dict):
                    continue
                if str(task.get("action") or "") != action:
                    continue
                grounding = task.get("grounding") or {}
                for role in selected:
                    selector = grounding.get(f"{role}_selector") or {}
                    values = selector.get("object_types") or []
                    if isinstance(values, str):
                        values = [values]
                    selected[role].update(str(value) for value in values)
                    if (
                        role == "source"
                        and selector.get("quantifier") in {"one", "all"}
                    ):
                        quantifier = selector["quantifier"]
            if selected["source"] or selected["destination"]:
                return selected, quantifier
        return {"source": set(), "destination": set()}, "one"

    def _default_roles(self, payload: dict) -> dict:
        action = str(payload.get("selected_action") or "")
        planned, quantifier = self._planned_role_types(action)
        instruction_words = self._words(str(payload.get("instruction") or ""))
        roles = {}
        for role, rows in (payload.get("role_tables") or {}).items():
            exact_rows = set()
            for index, row in enumerate(rows):
                label = row.get("object_type") or row.get("object_tag") or ""
                row_words = self._words(str(label))
                if row_words and (
                    row_words.issubset(instruction_words)
                    or any(
                        word in instruction_words
                        for word in row_words
                        if len(word) > 3
                    )
                ):
                    exact_rows.add(index)
                if (
                    (str(label) == "Laptop" and "computer" in instruction_words)
                    or (
                        str(label) == "RemoteControl"
                        and "remotecontrol" in instruction_words
                    )
                ):
                    exact_rows.add(index)
            included = exact_rows or {
                index for index, row in enumerate(rows)
                if str(row.get("object_type") or "") in planned.get(role, set())
            }
            if role == "source" and not included and rows:
                included = {0}
            if (
                role == "source"
                and {"all", "every", "each"}.intersection(instruction_words)
            ):
                quantifier = "all"
            roles[role] = {
                "status": (
                    "specified"
                    if role == "source" or included
                    else "unspecified"
                ),
                "quantifier": quantifier if role == "source" else "one",
                "classifications": [
                    "included" if index in included else "excluded"
                    for index in range(len(rows))
                ],
            }
        semantic_args = {}
        for name, rule in (payload.get("semantic_arg_schema") or {}).items():
            values = list(rule.get("values") or [])
            semantic_args[name] = next(
                (value for value in values if value in instruction_words),
                values[0] if values else None,
            )
        return {
            "status": "resolved",
            "roles": roles,
            "semantic_args": semantic_args,
            "summary": "Resolved from the instruction and supplied rows.",
        }

    def __call__(self, prompt: str) -> str:
        payload = json.loads(prompt)
        if payload.get("request") == "resolve_task_intent":
            self.resolver_prompts.append(payload)
            if payload.get("stage") == "action":
                self.action_prompts.append(payload)
                if self.resolver_responses:
                    try:
                        explicit = json.loads(self.resolver_responses[0])
                    except json.JSONDecodeError:
                        explicit = None
                    if isinstance(explicit, dict) and "action" in explicit:
                        return self.resolver_responses.pop(0)
                return json.dumps({
                    "status": "resolved",
                    "action": self._default_action(payload),
                    "summary": "Selected the requested interaction.",
                })
            self.role_prompts.append(payload)
            if self.resolver_responses:
                response = self.resolver_responses.pop(0)
                try:
                    legacy = json.loads(response)
                except json.JSONDecodeError:
                    return response
                if "included_types" in legacy:
                    included = set(legacy.get("included_types") or [])
                    converted = self._default_roles(payload)
                    converted["status"] = legacy.get("status", "resolved")
                    for role, rows in (payload.get("role_tables") or {}).items():
                        if converted["status"] == "no_match":
                            converted["roles"][role]["classifications"] = [
                                "excluded" for _ in rows
                            ]
                            if role == "destination":
                                converted["roles"][role]["status"] = "unspecified"
                        elif role == "source":
                            converted["roles"][role]["classifications"] = [
                                (
                                    "included"
                                    if row.get("object_type") in included
                                    else "excluded"
                                )
                                for row in rows
                            ]
                    return json.dumps(converted)
                return response
            return json.dumps(self._default_roles(payload))
        if payload.get("request") == "resolve_semantic_goal":
            self.resolver_prompts.append(payload)
            return self.resolver_responses.pop(0)
        if payload.get("request") == "validate_candidate_plan":
            self.critic_prompts.append(payload)
            if self.critic_responses:
                return self.critic_responses.pop(0)
            return json.dumps({"status": "valid", "errors": []})
        self.prompts.append(payload)
        return self.responses.pop(0)


class RaisingChat:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        raise RuntimeError("synthetic planning failure")


def test_runtime_replanning_passes_execution_report_and_constraints_to_planner_and_critic() -> None:
    chat = SequencedChat(
        [plan("open")],
        critic_responses=[json.dumps({"status": "valid", "errors": []})],
    )
    execution_context = {
        "mode": "runtime_replanning",
        "execution_report": {"marker": "complete-report"},
        "completed_tasks": [],
        "agent_states": [],
    }
    runtime_constraints = {
        "excluded_object_ids": ["Drawer|failed"],
        "held_objects": [],
    }

    graph = decompose_task_to_graph(
        TASK,
        SUBGRAPH,
        qwen_chat=chat,
        scene_catalog=CATALOGUE,
        planning_mode="runtime_replan",
        execution_context=execution_context,
        runtime_constraints=runtime_constraints,
    )

    assert chat.prompts[0]["planning_mode"] == "runtime_replan"
    assert chat.prompts[0]["execution_context"]["execution_report"]["marker"] == "complete-report"
    assert chat.prompts[0]["runtime_constraints"]["excluded_object_ids"] == ["Drawer|failed"]
    assert chat.critic_prompts[0]["planning_mode"] == "runtime_replan"
    assert chat.critic_prompts[0]["execution_context"] == execution_context
    assert graph["planner_diagnostics"]["planning_mode"] == "runtime_replan"


def test_runtime_validation_allows_completed_root_effect_to_be_omitted() -> None:
    candidate = json.loads(plan("close", quantifier="one"))
    validation = validate_planner_output(
        candidate,
        task="Open one Drawer",
        subgraph={
            "task": "Open one Drawer",
            "task_spec": {"raw_task": "Open one Drawer"},
            "nodes": [], "seed_nodes": [], "triples": [],
        },
        scene_catalog_summary=summarize_scene_catalogue(CATALOGUE),
        planning_mode="runtime_replan",
        execution_context={
            "completed_tasks": [{"id": "T1", "action": "open"}],
        },
    )

    assert not any(
        "root intent action 'open' is missing" in error
        for error in validation["errors"]
    )


def test_task10_retries_empty_and_wrong_action_then_accepts_strict_selector() -> None:
    chat = SequencedChat([
        json.dumps({"reasoning_summary": "No targets.", "subtasks": []}),
        plan("find"),
        plan("open"),
    ])

    graph = decompose_task_to_graph(
        TASK,
        SUBGRAPH,
        qwen_chat=chat,
        scene_catalog=CATALOGUE,
    )

    assert len(chat.prompts) == 3
    assert "correction" not in chat.prompts[0]
    assert chat.prompts[1]["correction"]["attempt"] == 2
    assert [item["message"] for item in chat.prompts[1]["correction"]["violations"]] == [
        "planner response must contain a non-empty subtasks list"
    ]
    assert "previous_response_preview" not in chat.prompts[1]["correction"]
    assert chat.prompts[2]["correction"]["attempt"] == 3
    assert any(
        "root intent action 'open'" in item["message"]
        for item in chat.prompts[2]["correction"]["violations"]
    )

    assert graph["planner_backend"] == "qwen_local"
    assert graph["planner_diagnostics"]["fallback_used"] is False
    assert graph["planner_diagnostics"]["qwen"]["selected_attempt"] == 3
    assert [entry["validation_status"] for entry in graph["planner_diagnostics"]["qwen"]["attempts"]] == [
        "invalid", "invalid", "valid",
    ]
    assert graph["flat_tasks"][0]["grounding"]["source_selector"] == {
        "quantifier": "all",
        "object_types": ["Drawer"],
    }

    expanded = expand_scene_catalog_task_graph(graph, CATALOGUE)
    assert len(expanded["flat_tasks"]) == 9
    assert {
        task["grounding"]["source_object_ids"][0]
        for task in expanded["flat_tasks"]
    } == {f"Drawer|{index:02d}" for index in range(1, 10)}


@pytest.mark.parametrize(
    ("response", "expected_error"),
    [
        (plan("open", quantifier="one"), "quantifier must be 'all'"),
        (plan("open", object_type="Cabinet"), "missing semantic goal types ['Drawer']"),
        (json.dumps({"subtasks": ["not-an-object"]}), "not JSON objects"),
    ],
)
def test_invalid_root_intent_exhausts_attempts_without_fallback(
    response: str,
    expected_error: str,
) -> None:
    chat = SequencedChat([response, response, response])

    with pytest.raises(TaskPlanningError) as captured:
        decompose_task_to_graph(
            TASK,
            SUBGRAPH,
            qwen_chat=chat,
            scene_catalog=CATALOGUE,
        )

    diagnostics = captured.value.diagnostics
    assert captured.value.code == "task_planning_failed"
    assert diagnostics["stage"] == "planning_attempts_exhausted"
    root_intent = diagnostics["root_intent"]
    assert root_intent["action"] == "open"
    assert root_intent["explicit_actions"] == ["open"]
    assert root_intent["quantifier"] == "all"
    assert root_intent["roles"] == {"source": ["Drawer"]}
    assert root_intent["catalogue_constrained"] is True
    assert root_intent["semantic_goal_resolved"] is True
    assert root_intent["intent_resolution_source"] == "qwen_two_stage_intent"
    attempts = diagnostics["qwen"]["attempts"]
    assert len(attempts) == 3
    assert all(expected_error in " ".join(item["validation_errors"]) for item in attempts)
    assert all(item["raw_response_preview"] for item in attempts)


def test_disable_qwen_fails_with_planning_disabled() -> None:
    with pytest.raises(TaskPlanningError) as captured:
        decompose_task_to_graph(TASK, SUBGRAPH, use_qwen=False, scene_catalog=CATALOGUE)

    assert captured.value.code == "planning_disabled"
    assert captured.value.diagnostics["qwen"]["attempted"] is False
    assert captured.value.diagnostics["qwen"]["attempts"] == []


def test_catalogue_backed_open_allows_non_root_find_when_selector_is_valid() -> None:
    response = json.loads(plan("open"))
    response["subtasks"].insert(0, {
        "id": "T0",
        "name": "Find drawers",
        "description": "Find the drawers before opening them.",
        "action": "find",
        "grounding": {"object_tags": ["Drawer"]},
        "depends_on": [],
    })

    graph = decompose_task_to_graph(
        TASK,
        SUBGRAPH,
        qwen_chat=SequencedChat([json.dumps(response)]),
        scene_catalog=CATALOGUE,
    )

    assert graph["planner_backend"] == "qwen_local"
    assert graph["planner_diagnostics"]["qwen"]["selected_attempt"] == 1
    assert [task["action"] for task in graph["flat_tasks"]] == ["find", "open"]


def test_invalid_json_and_model_errors_keep_complete_attempt_diagnostics() -> None:
    invalid_json = SequencedChat(["not json", "still not json", "final invalid response"])
    with pytest.raises(TaskPlanningError) as invalid_capture:
        decompose_task_to_graph(
            TASK,
            SUBGRAPH,
            qwen_chat=invalid_json,
            scene_catalog=CATALOGUE,
        )

    invalid_attempts = invalid_capture.value.diagnostics["qwen"]["attempts"]
    assert [attempt["raw_response_preview"] for attempt in invalid_attempts] == [
        "not json", "still not json", "final invalid response",
    ]
    assert all(attempt["json_parse_status"] == "invalid_json" for attempt in invalid_attempts)

    class PlannerRaisingChat(SequencedChat):
        def __call__(self, prompt: str) -> str:
            payload = json.loads(prompt)
            if payload.get("request") in {
                "resolve_task_intent",
                "validate_candidate_plan",
            }:
                return super().__call__(prompt)
            raise RuntimeError("synthetic planning failure")

    raising = PlannerRaisingChat([plan("open"), plan("open"), plan("open")])
    with pytest.raises(TaskPlanningError) as model_capture:
        decompose_task_to_graph(
            TASK, SUBGRAPH, qwen_chat=raising, scene_catalog=CATALOGUE
        )

    model_attempts = model_capture.value.diagnostics["qwen"]["attempts"]
    assert len(model_attempts) == 3
    assert all(attempt["stage"] == "model_call" for attempt in model_attempts)
    assert all(attempt["error_type"] == "RuntimeError" for attempt in model_attempts)
    assert all(attempt["error_message"] == "synthetic planning failure" for attempt in model_attempts)


def test_root_type_validation_prefers_exact_compound_types() -> None:
    summary = summarize_scene_catalogue([
        {"objectType": "Knife", "objectId": "Knife|1", "pickupable": True},
        {
            "objectType": "ButterKnife",
            "objectId": "ButterKnife|1",
            "pickupable": True,
        },
        {
            "objectType": "SaltShaker",
            "objectId": "SaltShaker|1",
            "pickupable": True,
        },
        {
            "objectType": "PepperShaker",
            "objectId": "PepperShaker|1",
            "pickupable": True,
        },
    ])
    empty_subgraph = {"task_spec": {}, "nodes": [], "triples": []}

    butter_knife = derive_root_intent("Pick up the butter knife", empty_subgraph, summary)
    knife = derive_root_intent("Pick up the knife", empty_subgraph, summary)
    shakers = derive_root_intent("Pick up all the shakers", empty_subgraph, summary)

    assert butter_knife["roles"]["source"] == ["ButterKnife"]
    assert knife["roles"]["source"] == ["Knife"]
    assert shakers["roles"]["source"] == []



def _place_plan_response(sources: list[str], destination: str) -> str:
    subtasks = []
    for index, source in enumerate(sources, start=1):
        subtasks.append({
            "id": f"T{index}",
            "name": f"Place {source}",
            "description": f"Place {source} on {destination}.",
            "action": "place",
            "grounding": {
                "node_ids": [],
                "object_tags": [source],
                "source_object_tags": [source],
                "destination_object_tags": [destination],
                "source_selector": {"quantifier": "one", "object_types": [source]},
                "destination_selector": {"quantifier": "one", "object_types": [destination]},
                "relation_texts": ["on"],
                "status": "grounded",
                "missing_reason": "",
                "recovery": "none",
            },
            "depends_on": [],
            "termination_check": f"{source} is on {destination}.",
        })
    return json.dumps({"reasoning_summary": "Place each requested object.", "subtasks": subtasks})


def _place_subgraph(task: str, sources: list[str], destination: str) -> dict:
    return {
        "task": task,
        "task_spec": {
            "raw_task": task,
            "task_type": "place",
            "actions": ["place"],
            "target_objects": sources,
            "destination_receptacles": [destination],
            "spatial_relations": ["on"],
        },
        "seed_nodes": [],
        "nodes": [],
        "triples": [],
    }


def _place_catalogue(sources: list[str], destination: str) -> list[dict]:
    return [
        {"objectType": source, "objectId": f"{source}|1", "pickupable": True}
        for source in sources
    ] + [{"objectType": destination, "objectId": f"{destination}|1", "receptacle": True}]


def test_multi_object_place_allows_parallel_root_place_tasks() -> None:
    task = "Put the butter knife, bowl, and mug on the countertop"
    sources = ["ButterKnife", "Bowl", "Mug"]
    destination = "CounterTop"

    graph = decompose_task_to_graph(
        task,
        _place_subgraph(task, sources, destination),
        qwen_chat=SequencedChat([_place_plan_response(sources, destination)]),
        scene_catalog=_place_catalogue(sources, destination),
    )

    validation = graph["planner_diagnostics"]["root_intent_validation"]
    assert validation["status"] == "valid"
    assert set(validation["root_intent"]["roles"]["source"]) == set(sources)
    assert graph["planner_diagnostics"]["qwen"]["attempts"][0]["validation_errors"] == []


def test_living_room_multi_object_place_allows_all_requested_sources() -> None:
    task = "Put the computer, book, and remotecontrol on the sofa"
    sources = ["Laptop", "Book", "RemoteControl"]
    destination = "Sofa"

    graph = decompose_task_to_graph(
        task,
        _place_subgraph(task, sources, destination),
        qwen_chat=SequencedChat([_place_plan_response(sources, destination)]),
        scene_catalog=_place_catalogue(sources, destination),
    )

    validation = graph["planner_diagnostics"]["root_intent_validation"]
    assert validation["status"] == "valid"
    assert set(validation["root_intent"]["roles"]["source"]) == set(sources)
    assert graph["planner_diagnostics"]["qwen"]["selected_attempt"] == 1


def test_source_selector_extra_catalogue_types_do_not_fail_multi_object_place() -> None:
    task = "Put the butter knife, bowl, and mug on the countertop"
    sources = ["ButterKnife", "Bowl", "Mug"]
    destination = "CounterTop"
    raw_plan = json.loads(_place_plan_response(sources, destination))
    subgraph = _place_subgraph(task, ["ButterKnife"], destination)

    validation = validate_planner_output(
        raw_plan,
        task=task,
        subgraph=subgraph,
        scene_catalog_summary=summarize_scene_catalogue(_place_catalogue(sources, destination)),
    )

    assert validation["status"] == "valid"
    assert validation["root_intent"]["roles"]["source"] == ["ButterKnife"]
    assert not any("unrelated catalogue types" in error for error in validation["errors"])

def test_atomic_place_without_pick_is_valid_with_empty_agent_inventories() -> None:
    task = "Put the bread, lettuce, and tomato in the fridge"
    sources = ["Bread", "Lettuce", "Tomato"]
    raw_plan = json.loads(_place_plan_response(sources, "Fridge"))

    validation = validate_planner_output(
        raw_plan,
        task=task,
        subgraph=_place_subgraph(task, sources, "Fridge"),
        scene_catalog_summary=summarize_scene_catalogue(_place_catalogue(sources, "Fridge")),
        agent_context={"agents": [{"agent_id": "0", "inventory": []}, {"agent_id": "1", "inventory": []}]},
    )

    assert validation["status"] == "valid"
    assert not any("matching ancestor pick" in error for error in validation["errors"])


def test_all_goal_allows_non_overlapping_selector_tasks_per_source_type() -> None:
    task = "Put all the tomatoes and potatoes in the fridge"
    sources = ["Tomato", "Potato"]
    raw_plan = json.loads(_place_plan_response(sources, "Fridge"))
    for subtask in raw_plan["subtasks"]:
        subtask["grounding"]["source_selector"]["quantifier"] = "all"

    validation = validate_planner_output(
        raw_plan,
        task=task,
        subgraph=_place_subgraph(task, sources, "Fridge"),
        scene_catalog_summary=summarize_scene_catalogue(_place_catalogue(sources, "Fridge")),
        agent_context={"agents": [{"agent_id": "0", "inventory": []}, {"agent_id": "1", "inventory": []}]},
    )

    assert validation["status"] == "valid"
    assert validation["errors"] == []


def test_place_into_openable_destination_requires_matching_open_dependency() -> None:
    task = "Put the bread in the fridge"
    raw_plan = {
        "reasoning_summary": "Pick, open, place.",
        "subtasks": [
            {
                "id": "T1",
                "name": "Pick bread",
                "description": "Pick bread.",
                "action": "pick",
                "grounding": {
                    "object_tags": ["Bread"],
                    "source_object_tags": ["Bread"],
                    "source_selector": {"quantifier": "one", "object_types": ["Bread"]},
                    "relation_texts": [],
                },
                "depends_on": [],
            },
            {
                "id": "T2",
                "name": "Open fridge",
                "description": "Open fridge.",
                "action": "open",
                "grounding": {
                    "object_tags": ["Fridge"],
                    "source_object_tags": ["Fridge"],
                    "source_selector": {"quantifier": "one", "object_types": ["Fridge"]},
                    "relation_texts": [],
                },
                "depends_on": [],
            },
            {
                "id": "T3",
                "name": "Place bread",
                "description": "Place bread in fridge.",
                "action": "place",
                "grounding": {
                    "object_tags": ["Bread", "Fridge"],
                    "source_object_tags": ["Bread"],
                    "destination_object_tags": ["Fridge"],
                    "source_selector": {"quantifier": "one", "object_types": ["Bread"]},
                    "destination_selector": {"quantifier": "one", "object_types": ["Fridge"]},
                    "relation_texts": ["in"],
                },
                "depends_on": ["T1"],
            },
        ],
    }

    validation = validate_planner_output(
        raw_plan,
        task=task,
        subgraph=_place_subgraph(task, ["Bread"], "Fridge"),
        scene_catalog_summary=summarize_scene_catalogue(_place_catalogue(["Bread"], "Fridge")),
    )

    assert validation["status"] == "invalid"
    assert any("matching open dependency" in error for error in validation["errors"])


def test_close_fridge_requires_all_fridge_placements_as_dependencies() -> None:
    task = "Put bread and lettuce in the fridge, then close it"
    raw_plan = {
        "reasoning_summary": "Close after placements.",
        "subtasks": [
            {
                "id": "T1",
                "name": "Place bread",
                "description": "Place bread in fridge.",
                "action": "place",
                "grounding": {
                    "object_tags": ["Bread", "Fridge"],
                    "source_object_tags": ["Bread"],
                    "destination_object_tags": ["Fridge"],
                    "source_selector": {"quantifier": "one", "object_types": ["Bread"]},
                    "destination_selector": {"quantifier": "one", "object_types": ["Fridge"]},
                    "relation_texts": ["in"],
                },
                "depends_on": [],
            },
            {
                "id": "T2",
                "name": "Place lettuce",
                "description": "Place lettuce in fridge.",
                "action": "place",
                "grounding": {
                    "object_tags": ["Lettuce", "Fridge"],
                    "source_object_tags": ["Lettuce"],
                    "destination_object_tags": ["Fridge"],
                    "source_selector": {"quantifier": "one", "object_types": ["Lettuce"]},
                    "destination_selector": {"quantifier": "one", "object_types": ["Fridge"]},
                    "relation_texts": ["in"],
                },
                "depends_on": [],
            },
            {
                "id": "T3",
                "name": "Close fridge",
                "description": "Close fridge.",
                "action": "close",
                "grounding": {
                    "object_tags": ["Fridge"],
                    "source_object_tags": ["Fridge"],
                    "source_selector": {"quantifier": "one", "object_types": ["Fridge"]},
                    "relation_texts": [],
                },
                "depends_on": ["T1"],
            },
        ],
    }

    validation = validate_planner_output(
        raw_plan,
        task=task,
        subgraph=_place_subgraph(task, ["Bread", "Lettuce"], "Fridge"),
        scene_catalog_summary=summarize_scene_catalogue(_place_catalogue(["Bread", "Lettuce"], "Fridge")),
    )

    assert validation["status"] == "invalid"
    assert any("must depend on all placement tasks" in error for error in validation["errors"])


def _all_silverware_plan_response(sources: list[str]) -> str:
    return json.dumps({
        "reasoning_summary": "Place all selected silverware in one drawer.",
        "subtasks": [{
            "id": "T1",
            "name": "Place silverware",
            "description": "Place all selected silverware in one drawer.",
            "action": "place",
            "grounding": {
                "node_ids": [],
                "object_tags": [*sources, "Drawer"],
                "source_object_tags": sources,
                "destination_object_tags": ["Drawer"],
                "source_selector": {"quantifier": "all", "object_types": sources},
                "destination_selector": {"quantifier": "one", "object_types": ["Drawer"]},
                "relation_texts": ["in"],
            },
            "depends_on": [],
            "termination_check": "All selected silverware is in the drawer.",
        }],
    })


def test_all_selector_does_not_infer_category_membership_from_type_suffixes() -> None:
    task = "Put all groceries in the fridge"
    selected_sources = ["Apple", "Bottle", "Bread", "Lettuce", "Potato", "Tomato"]
    catalogue_sources = [*selected_sources, "SoapBottle", "WineBottle"]
    raw_plan = json.loads(_all_silverware_plan_response(selected_sources))
    place = raw_plan["subtasks"][0]
    place["name"] = "Place groceries"
    place["description"] = "Place the selected groceries in the fridge."
    place["grounding"]["object_tags"][-1] = "Fridge"
    place["grounding"]["destination_object_tags"] = ["Fridge"]
    place["grounding"]["destination_selector"]["object_types"] = ["Fridge"]

    validation = validate_planner_output(
        raw_plan,
        task=task,
        subgraph=_place_subgraph(task, ["all groceries"], "Fridge"),
        scene_catalog_summary=summarize_scene_catalogue(
            _place_catalogue(catalogue_sources, "Fridge")
        ),
    )

    assert validation["status"] == "valid"
    assert not any("compound catalogue types" in error for error in validation["errors"])

def test_all_silverware_requires_every_semantic_category_type() -> None:
    task = "Put all silverware in any drawer"
    required_sources = ["ButterKnife", "Fork", "Knife", "Ladle", "Spatula", "Spoon"]
    raw_plan = json.loads(_all_silverware_plan_response([
        source for source in required_sources if source != "Spatula"
    ]))

    root_intent = derive_root_intent(
        task,
        _place_subgraph(task, ["all silverware"], "Drawer"),
        summarize_scene_catalogue(_place_catalogue(required_sources, "Drawer")),
    )
    root_intent["roles"]["source"] = required_sources
    root_intent["semantic_goal_resolved"] = True
    validation = validate_planner_output(
        raw_plan,
        task=task,
        subgraph=_place_subgraph(task, ["all silverware"], "Drawer"),
        scene_catalog_summary=summarize_scene_catalogue(
            _place_catalogue(required_sources, "Drawer")
        ),
        root_intent_override=root_intent,
    )

    assert validation["root_intent"]["roles"]["source"] == required_sources
    assert validation["status"] == "invalid"
    assert validation["errors"] == [
        "root source_selector is missing semantic goal types ['Spatula']; "
        "required exact types are ['ButterKnife', 'Fork', 'Knife', 'Ladle', "
        "'Spatula', 'Spoon']"
    ]


def test_resolved_semantic_goal_rejects_unexpected_source_types() -> None:
    task = "Put all silverware in any drawer"
    required_sources = ["Fork", "Spoon"]
    raw_plan = json.loads(
        _all_silverware_plan_response(["Fork", "Spoon", "Apple"])
    )
    summary = summarize_scene_catalogue(
        _place_catalogue(["Fork", "Spoon", "Apple"], "Drawer")
    )
    root_intent = derive_root_intent(
        task,
        _place_subgraph(task, ["all silverware"], "Drawer"),
        summary,
    )
    root_intent["roles"]["source"] = required_sources
    root_intent["semantic_goal_resolved"] = True

    validation = validate_planner_output(
        raw_plan,
        task=task,
        subgraph=_place_subgraph(task, ["all silverware"], "Drawer"),
        scene_catalog_summary=summary,
        root_intent_override=root_intent,
    )

    assert validation["status"] == "invalid"
    assert validation["errors"] == [
        "root source_selector contains unexpected semantic goal types ['Apple']; "
        "required exact types are ['Fork', 'Spoon']"
    ]


def _resolved_category_response(
    included: list[str],
    candidates: list[str],
) -> str:
    return json.dumps({
        "status": "resolved",
        "included_types": included,
        "excluded_types": [
            value for value in candidates if value not in set(included)
        ],
        "summary": "Resolved the public category against every candidate.",
    })


@pytest.mark.parametrize(
    ("raw_result", "error_fragment"),
    [
        (
            {
                "status": "resolved",
                "included_types": ["Fork"],
                "excluded_types": [],
            },
            "does not classify",
        ),
        (
            {
                "status": "resolved",
                "included_types": ["Fork", "Fork"],
                "excluded_types": ["Spoon"],
            },
            "contains duplicates",
        ),
        (
            {
                "status": "resolved",
                "included_types": ["Fork"],
                "excluded_types": ["Fork", "Spoon"],
            },
            "overlap",
        ),
        (
            {
                "status": "resolved",
                "included_types": ["Fork", "Knife"],
                "excluded_types": ["Spoon"],
            },
            "unknown candidate",
        ),
    ],
)
def test_semantic_goal_resolver_requires_an_exact_candidate_partition(
    raw_result: dict,
    error_fragment: str,
) -> None:
    validation = validate_semantic_goal_resolution(
        raw_result,
        ["Fork", "Spoon"],
    )

    assert validation["protocol_status"] == "invalid"
    assert any(error_fragment in error for error in validation["errors"])


def test_task14_exact_types_still_use_two_stage_intent_and_compact_catalogue() -> None:
    task = "Put all the tomatoes and potatoes in the fridge"
    raw_plan = json.loads(
        _all_silverware_plan_response(["Tomato", "Potato"])
    )
    grounding = raw_plan["subtasks"][0]["grounding"]
    grounding["object_tags"][-1] = "Fridge"
    grounding["destination_object_tags"] = ["Fridge"]
    grounding["destination_selector"]["object_types"] = ["Fridge"]
    chat = SequencedChat([json.dumps(raw_plan)])

    graph = decompose_task_to_graph(
        task,
        _place_subgraph(task, ["all the tomatoes", "potatoes"], "Fridge"),
        qwen_chat=chat,
        scene_catalog=_place_catalogue(
            ["Tomato", "Potato", "Apple", "Mug"],
            "Fridge",
        ),
    )

    diagnostics = graph["planner_diagnostics"]["semantic_goal_resolution"]
    assert diagnostics["status"] == "success"
    assert len(chat.action_prompts) == 1
    assert len(chat.role_prompts) == 1
    assert set(chat.prompts[0]["resolved_semantic_goal"][
        "required_source_types"
    ]) == {"Tomato", "Potato"}
    prompt_types = {
        entry["object_type"] for entry in chat.prompts[0]["scene_catalog"]
    }
    assert prompt_types == {"Tomato", "Potato", "Fridge"}
    assert "executable_selector_catalog" not in chat.prompts[0]
    assert "allowed_object_types" not in json.dumps(chat.prompts[0])


def test_task19_resolver_is_reused_across_three_corrective_attempts() -> None:
    task = "Put all silverware in any drawer"
    required_sources = [
        "ButterKnife", "Fork", "Knife", "Ladle", "Spatula", "Spoon",
    ]
    candidates = [*required_sources, "Apple"]
    chat = SequencedChat(
        [
            _all_silverware_plan_response([
                value for value in required_sources if value != "Spatula"
            ]),
            _all_silverware_plan_response([
                value for value in required_sources if value != "ButterKnife"
            ]),
            _all_silverware_plan_response(required_sources),
        ],
        resolver_responses=[
            _resolved_category_response(required_sources, candidates),
        ],
    )

    graph = decompose_task_to_graph(
        task,
        _place_subgraph(task, ["all silverware"], "Drawer"),
        qwen_chat=chat,
        scene_catalog=_place_catalogue(candidates, "Drawer"),
    )

    assert len(chat.resolver_prompts) == 2
    assert len(chat.prompts) == 3
    assert len(chat.critic_prompts) == 3
    assert all(
        prompt["resolved_semantic_goal"]["required_source_types"]
        == required_sources
        for prompt in chat.prompts
    )
    assert all(
        "Apple" not in {
            entry["object_type"] for entry in prompt["scene_catalog"]
        }
        for prompt in chat.prompts
    )
    assert all(
        prompt["resolved_semantic_goal"]["required_source_types"]
        == required_sources
        for prompt in chat.critic_prompts
    )
    assert {
        row["object_type"]
        for row in chat.role_prompts[0]["role_tables"]["source"]
    } == set(candidates)
    assert "candidate_scene_facts" not in chat.role_prompts[0]
    first_codes = {
        item["code"]
        for item in chat.prompts[1]["correction"]["violations"]
    }
    assert "missing_semantic_goal_types" in first_codes
    assert "Spatula" in chat.prompts[1]["correction"]["summary"]
    assert "previous_response_preview" not in chat.prompts[1]["correction"]
    diagnostics = graph["planner_diagnostics"]
    assert diagnostics["semantic_goal_resolution"]["attempt_count"] == 2
    assert diagnostics["semantic_goal_resolution"][
        "reused_across_planning_attempts"
    ] is True
    assert diagnostics["qwen"]["selected_attempt"] == 3


def test_semantic_goal_resolver_retries_protocol_once_without_old_output() -> None:
    task = "Put all shakers in the fridge"
    sources = ["PepperShaker", "SaltShaker"]
    raw_plan = json.loads(_all_silverware_plan_response(sources))
    grounding = raw_plan["subtasks"][0]["grounding"]
    grounding["object_tags"][-1] = "Fridge"
    grounding["destination_object_tags"] = ["Fridge"]
    grounding["destination_selector"]["object_types"] = ["Fridge"]
    chat = SequencedChat(
        [json.dumps(raw_plan)],
        resolver_responses=[
            "not json",
            _resolved_category_response(sources, sources),
        ],
    )

    graph = decompose_task_to_graph(
        task,
        _place_subgraph(task, ["all shakers"], "Fridge"),
        qwen_chat=chat,
        scene_catalog=_place_catalogue(sources, "Fridge"),
    )

    assert len(chat.resolver_prompts) == 3
    correction = chat.role_prompts[1]["protocol_correction"]
    assert correction["errors"] == ["resolver response is not valid JSON"]
    assert "not json" not in json.dumps(correction)
    assert graph["planner_diagnostics"]["semantic_goal_resolution"][
        "attempt_count"
    ] == 3


def test_semantic_goal_resolver_retries_one_model_exception() -> None:
    task = "Put all shakers in the fridge"
    source = "SaltShaker"
    raw_plan = json.loads(_all_silverware_plan_response([source]))
    grounding = raw_plan["subtasks"][0]["grounding"]
    grounding["object_tags"][-1] = "Fridge"
    grounding["destination_object_tags"] = ["Fridge"]
    grounding["destination_selector"]["object_types"] = ["Fridge"]

    class ResolverRaisesOnceChat(SequencedChat):
        def __init__(self) -> None:
            super().__init__(
                [json.dumps(raw_plan)],
                resolver_responses=[
                    _resolved_category_response([source], [source]),
                ],
            )
            self.raise_resolver_once = True

        def __call__(self, prompt: str) -> str:
            payload = json.loads(prompt)
            if (
                payload.get("request") == "resolve_task_intent"
                and payload.get("stage") == "roles"
                and self.raise_resolver_once
            ):
                self.raise_resolver_once = False
                self.resolver_prompts.append(payload)
                self.role_prompts.append(payload)
                raise RuntimeError("synthetic resolver transport error")
            return super().__call__(prompt)

    chat = ResolverRaisesOnceChat()
    graph = decompose_task_to_graph(
        task,
        _place_subgraph(task, ["all shakers"], "Fridge"),
        qwen_chat=chat,
        scene_catalog=_place_catalogue([source], "Fridge"),
    )

    assert len(chat.resolver_prompts) == 3
    correction = chat.role_prompts[1]["protocol_correction"]
    assert correction["errors"] == [
        "resolver model call failed: RuntimeError: "
        "synthetic resolver transport error"
    ]
    assert graph["planner_diagnostics"]["semantic_goal_resolution"][
        "attempt_count"
    ] == 3


def test_semantic_goal_resolver_fails_closed_after_two_protocol_errors() -> None:
    task = "Put all shakers in the fridge"
    chat = SequencedChat(
        [],
        resolver_responses=["not json", "still not json"],
    )

    with pytest.raises(TaskPlanningError) as capture:
        decompose_task_to_graph(
            task,
            _place_subgraph(task, ["all shakers"], "Fridge"),
            qwen_chat=chat,
            scene_catalog=_place_catalogue(["SaltShaker"], "Fridge"),
        )

    assert capture.value.code == "semantic_goal_resolution"
    diagnostics = capture.value.diagnostics["semantic_goal_resolution"]
    assert diagnostics["stage"] == "role_attempts_exhausted"
    assert diagnostics["attempt_count"] == 3
    assert chat.prompts == []


def test_semantic_goal_resolver_no_match_fails_closed_without_planning() -> None:
    task = "Put all shakers in the fridge"
    chat = SequencedChat(
        [],
        resolver_responses=[
            json.dumps({
                "status": "no_match",
                "included_types": [],
                "excluded_types": ["SaltShaker"],
                "summary": "No candidate matches.",
            }),
        ],
    )

    with pytest.raises(TaskPlanningError) as capture:
        decompose_task_to_graph(
            task,
            _place_subgraph(task, ["all shakers"], "Fridge"),
            qwen_chat=chat,
            scene_catalog=_place_catalogue(["SaltShaker"], "Fridge"),
        )

    assert capture.value.code == "semantic_goal_resolution"
    assert capture.value.diagnostics["semantic_goal_resolution"][
        "stage"
    ] == "role_no_match"
    assert chat.prompts == []


def test_spatial_resolver_projection_keeps_parent_labels_without_coordinates() -> None:
    task = "Place all items from the central countertop in appropriate positions"
    raw_catalogue = [
        {
            "objectType": "Apple",
            "objectId": "Apple|1",
            "pickupable": True,
            "parentReceptacles": ["CounterTop|central"],
        },
        {
            "objectType": "CounterTop",
            "objectId": "CounterTop|central",
            "receptacle": True,
            "position": {"x": 0.0, "y": 1.0, "z": 0.0},
        },
        {
            "objectType": "CounterTop",
            "objectId": "CounterTop|side",
            "receptacle": True,
            "position": {"x": 10.0, "y": 1.0, "z": 0.0},
        },
    ]
    summary = summarize_scene_catalogue(raw_catalogue)
    subgraph = _place_subgraph(task, ["all items"], "")
    root_intent = derive_root_intent(task, subgraph, summary)

    request = semantic_goal_resolution_request(
        task=task,
        subgraph=subgraph,
        root_intent=root_intent,
        scene_catalog_summary=summary,
    )

    assert request is not None
    facts = request["candidate_scene_facts"]
    apple = next(item for item in facts if item["object_type"] == "Apple")
    parent = apple["parent_locations"][0]
    assert parent == {
        "parent_type": "CounterTop",
        "instance_count": 1,
        "relative_location": "central",
    }
    serialized = json.dumps(request)
    assert "parent_position" not in serialized
    assert "objectId" not in serialized


def _interaction_plan(
    action: str,
    source: str,
    destination: str | None = None,
    *,
    empty_source: bool = False,
) -> str:
    source_types = [] if empty_source else [source]
    grounding = {
        "node_ids": [],
        "object_tags": [value for value in (source, destination) if value],
        "source_object_tags": source_types,
        "source_selector": {"quantifier": "one", "object_types": source_types},
        "relation_texts": [],
    }
    if destination is not None:
        grounding.update({
            "destination_object_tags": [destination],
            "destination_selector": {
                "quantifier": "one",
                "object_types": [destination],
            },
        })
    return json.dumps({
        "reasoning_summary": "Execute the requested interaction.",
        "subtasks": [{
            "id": "T1",
            "name": f"{action} {source}",
            "description": f"Execute {action} for {source}.",
            "action": action,
            "grounding": grounding,
            "depends_on": [],
            "termination_check": "The interaction is complete.",
        }],
    })


def _task23_subgraph(task: str) -> dict:
    return {
        "task": task,
        "task_spec": {"raw_task": task, "parser": "disabled_scenegraph"},
        "seed_nodes": [],
        "nodes": [],
        "triples": [],
    }


def test_task23_validation_chain_retries_deterministic_and_semantic_errors() -> None:
    task = "Clear the central countertop by placing items in their appropriate positions"
    catalogue = [
        {"objectType": "Apple", "objectId": "Apple|1", "pickupable": True},
        {"objectType": "CounterTop", "objectId": "CounterTop|1", "receptacle": True},
        {"objectType": "Fridge", "objectId": "Fridge|1", "receptacle": True},
    ]
    chat = SequencedChat(
        [
            _interaction_plan("clean", "Apple", "CounterTop", empty_source=True),
            _interaction_plan("place", "Apple", "CounterTop"),
            _interaction_plan("place", "Apple", "Fridge"),
        ],
        critic_responses=[
            json.dumps({
                "status": "invalid",
                "errors": [
                    {
                        "task_id": "T1",
                        "code": "wrong_action",
                        "field": "action",
                        "message": "clean does not express relocating countertop items",
                        "invalid_values": ["clean"],
                        "required_fix": "Generate explicit place tasks for the countertop items.",
                    },
                    {
                        "task_id": "T1",
                        "code": "wrong_destination",
                        "field": "grounding.destination_selector",
                        "message": "CounterTop is the surface being cleared, not the placement destination",
                        "invalid_values": ["CounterTop"],
                        "required_fix": "Choose a destination consistent with clearing the surface.",
                    },
                ],
            }),
            json.dumps({
                "status": "invalid",
                "errors": [{
                    "task_id": "T1",
                    "code": "wrong_destination",
                    "field": "grounding.destination_selector",
                    "message": "Placing Apple on CounterTop contradicts clearing the countertop.",
                    "invalid_values": ["CounterTop"],
                    "required_fix": "Place the Apple at a destination other than CounterTop.",
                }],
            }),
            json.dumps({"status": "valid", "errors": []}),
        ],
    )

    graph = decompose_task_to_graph(
        task,
        _task23_subgraph(task),
        qwen_chat=chat,
        scene_catalog=catalogue,
    )

    attempts = graph["planner_diagnostics"]["qwen"]["attempts"]
    assert len(chat.prompts) == 3
    assert len(chat.critic_prompts) == 3
    assert any(
        "incomplete subgraph omitted it" in constraint
        for constraint in chat.critic_prompts[0]["constraints"]
    )
    assert chat.critic_prompts[0]["evidence_policy"][
        "commonsense_category_suitability_must_be_checked"
    ] is True
    assert any(
        "one catch-all destination" in constraint
        for constraint in chat.critic_prompts[0]["constraints"]
    )
    serialized_critic_prompt = json.dumps(chat.critic_prompts[0], ensure_ascii=False)
    assert "Apple, Tomato, Fork, and ButterKnife" not in serialized_critic_prompt
    assert "Apple to Fridge" not in serialized_critic_prompt
    assert any(
        example["status"] == "invalid"
        and "heterogeneous food and utensils" in example["candidate_fact"]
        for example in chat.critic_prompts[0]["verdict_examples"]
    )
    place_sources = chat.prompts[0]["execution_action_contracts"]["place"]["roles"]["source"]
    assert place_sources == ["pickupable"]
    assert "allowed_object_types" not in serialized_critic_prompt
    assert any(
        "Never reject a candidate because grounding.node_ids" in constraint
        for constraint in chat.critic_prompts[0]["constraints"]
    )
    assert chat.reset_count == 8
    assert attempts[0]["selector_expansion"]["status"] == "rejected"
    assert any(
        "missing source_selector object_types" in error
        for error in attempts[0]["validation_errors"]
    )
    assert any(
        "do not copy any selector named by a violation" in constraint
        for constraint in chat.prompts[1]["constraints"]
    )
    assert any(
        "destination_selector is not supported for action 'clean'" in error
        for error in attempts[0]["validation_errors"]
    )
    assert any(
        "clean does not express relocating countertop items" in error
        for error in attempts[0]["validation_errors"]
    )
    assert any(
        "CounterTop is the surface being cleared" in error
        for error in attempts[0]["validation_errors"]
    )
    first_correction = chat.prompts[1]["correction"]
    empty_source = next(
        item for item in first_correction["violations"]
        if item["code"] == "empty_source_selector"
    )
    assert empty_source["task_id"] == "T1"
    assert empty_source["field"] == "grounding.source_selector.object_types"
    assert empty_source["required_fix"]
    semantic_destination = next(
        item for item in first_correction["violations"]
        if item["code"] == "wrong_destination"
    )
    assert semantic_destination["field"] == "grounding.destination_selector"
    assert semantic_destination["invalid_values"] == ["CounterTop"]
    assert "complete replacement JSON plan" in first_correction["summary"]
    assert attempts[1]["selector_expansion"]["status"] == "expanded"
    assert attempts[1]["semantic_validation"]["verdict"]["status"] == "invalid"
    assert [item["message"] for item in chat.prompts[2]["correction"]["violations"]] == [
        "root destination_selector is missing catalogue types ['Fridge']",
        "root destination_selector contains unrelated catalogue types ['CounterTop']",
        "semantic validation: Placing Apple on CounterTop contradicts clearing the countertop."
    ]
    assert attempts[2]["semantic_validation"]["verdict"] == {
        "protocol_status": "valid",
        "status": "valid",
        "errors": [],
    }
    assert graph["planner_diagnostics"]["qwen"]["selected_attempt"] == 3

    place = graph["flat_tasks"][0]
    assert place["grounding"]["destination_selector"]["object_types"] == ["Fridge"]
    assert not place["grounding"].get("source_object_ids")
    runtime_graph = expand_scene_catalog_task_graph(graph, catalogue)
    assert runtime_graph["planner_diagnostics"]["selector_expansion"]["status"] == "expanded"
    assert runtime_graph["flat_tasks"][0]["grounding"]["source_object_ids"] == ["Apple|1"]


def test_empty_subgraph_without_catalogue_fails_before_planner() -> None:
    task = "Pick up the apple."
    chat = SequencedChat([])

    with pytest.raises(TaskPlanningError) as captured:
        decompose_task_to_graph(
            task,
            _task23_subgraph(task),
            qwen_chat=chat,
            scene_catalog=None,
        )

    diagnostics = captured.value.diagnostics
    assert diagnostics["stage"] == "semantic_goal_resolution"
    resolution = diagnostics["semantic_goal_resolution"]
    assert resolution["stage"] == "candidate_filtering"
    assert len(chat.action_prompts) == 1
    assert chat.role_prompts == []
    assert chat.prompts == []


def test_subgraph_node_mode_resolves_and_validates_selected_nodes() -> None:
    task = "Pick up the apple."
    subgraph = _task23_subgraph(task)
    subgraph["nodes"] = [{
        "pruned_id": 7,
        "original_id": 70,
        "object_tag": "Apple",
        "caption": "a red apple",
        "possible_tags": ["Apple", "fruit"],
    }]
    candidate = json.loads(_interaction_plan("pick", "Apple"))
    grounding = candidate["subtasks"][0]["grounding"]
    grounding.pop("source_selector", None)
    grounding["node_ids"] = [7]
    grounding["source_node_ids"] = [7]
    chat = SequencedChat([json.dumps(candidate)])

    graph = decompose_task_to_graph(
        task,
        subgraph,
        qwen_chat=chat,
        scene_catalog=None,
    )

    role_prompt = chat.role_prompts[0]
    assert role_prompt["grounding_mode"] == "subgraph"
    assert role_prompt["role_tables"]["source"] == [{
        "row_id": 0,
        "node_id": 7,
        "object_tag": "Apple",
        "caption": "a red apple",
        "possible_tags": ["Apple", "fruit"],
        "capabilities_verified": False,
    }]
    root_intent = graph["planner_diagnostics"][
        "root_intent_validation"
    ]["root_intent"]
    assert root_intent["role_nodes"]["source"] == [7]
    assert graph["planner_diagnostics"]["qwen"]["attempts"][0][
        "selector_expansion"
    ]["status"] == "skipped"


@pytest.mark.parametrize("critic_response", ["not JSON", json.dumps({"status": "valid"})])
def test_semantic_critic_protocol_failure_fails_closed(critic_response: str) -> None:
    chat = SequencedChat(
        [_interaction_plan("pick", "Apple")],
        critic_responses=[critic_response],
    )

    with pytest.raises(TaskPlanningError) as captured:
        decompose_task_to_graph(
            "Pick up the apple.",
            _task23_subgraph("Pick up the apple."),
            qwen_chat=chat,
            scene_catalog=[
                {"objectType": "Apple", "objectId": "Apple|1", "pickupable": True},
            ],
        )

    assert captured.value.diagnostics["stage"] == "semantic_validation"
    assert captured.value.diagnostics["qwen"]["attempt_count"] == 1
    assert captured.value.diagnostics["qwen"]["selected_attempt"] is None


def test_semantic_critic_model_failure_fails_closed() -> None:
    class CriticRaisingChat(SequencedChat):
        def __call__(self, prompt: str) -> str:
            payload = json.loads(prompt)
            if payload.get("request") == "validate_candidate_plan":
                raise RuntimeError("synthetic critic failure")
            return super().__call__(prompt)

    chat = CriticRaisingChat([_interaction_plan("pick", "Apple")])
    with pytest.raises(TaskPlanningError) as captured:
        decompose_task_to_graph(
            "Pick up the apple.",
            _task23_subgraph("Pick up the apple."),
            qwen_chat=chat,
            scene_catalog=[
                {"objectType": "Apple", "objectId": "Apple|1", "pickupable": True},
            ],
        )

    semantic = captured.value.diagnostics["semantic_validation"]
    assert captured.value.diagnostics["stage"] == "semantic_validation"
    assert semantic["stage"] == "model_call"
    assert semantic["error_type"] == "RuntimeError"
    assert semantic["error_message"] == "synthetic critic failure"


def test_semantic_invalid_verdict_retries_then_exhausts() -> None:
    response = _interaction_plan("pick", "Apple")
    invalid = json.dumps({
        "status": "invalid",
        "errors": [{
            "task_id": None,
            "code": "missing_semantic_condition",
            "field": None,
            "message": "The candidate omits the requested semantic condition.",
            "invalid_values": [],
            "required_fix": "Add the omitted requested semantic condition.",
        }],
    })
    chat = SequencedChat(
        [response, response, response],
        critic_responses=[invalid, invalid, invalid],
    )

    with pytest.raises(TaskPlanningError) as captured:
        decompose_task_to_graph(
            "Pick up the apple.",
            _task23_subgraph("Pick up the apple."),
            qwen_chat=chat,
            scene_catalog=[
                {"objectType": "Apple", "objectId": "Apple|1", "pickupable": True},
            ],
        )

    diagnostics = captured.value.diagnostics
    assert diagnostics["stage"] == "planning_attempts_exhausted"
    assert len(chat.prompts) == 3
    assert len(chat.critic_prompts) == 3
    assert all(
        attempt["validation_errors"] == [
            "semantic validation: The candidate omits the requested semantic condition."
        ]
        for attempt in diagnostics["qwen"]["attempts"]
    )


def test_semantic_critic_ignores_instance_grounding_errors_for_slice_plan() -> None:
    task = "Slice the bread, lettuce, tomato, and egg"
    object_types = ["Bread", "Lettuce", "Tomato", "Egg"]
    plan_response = json.dumps({
        "reasoning_summary": "Slice every requested food.",
        "subtasks": [
            {
                "id": f"T{index}",
                "name": f"Slice {object_type}",
                "description": f"Slice the {object_type}.",
                "action": "slice",
                "action_args": {},
                "grounding": {
                    "node_ids": [],
                    "source_node_ids": [],
                    "source_object_ids": [],
                    "object_tags": [object_type],
                    "source_object_tags": [object_type],
                    "source_selector": {
                        "quantifier": "one",
                        "object_types": [object_type],
                    },
                    "relation_texts": [],
                    "status": "unresolved",
                    "missing_reason": "Runtime binding is pending.",
                    "recovery": "search_visible_scene",
                },
                "depends_on": [],
                "termination_check": f"{object_type} is sliced.",
            }
            for index, object_type in enumerate(object_types, start=1)
        ],
    })
    grounding_error = json.dumps({
        "status": "invalid",
        "errors": [{
            "task_id": "T1",
            "code": "missing_source_grounding",
            "field": "grounding.source_node_ids",
            "message": "Bread has no grounded scene node ID.",
            "invalid_values": ["[]"],
            "required_fix": "Bind Bread to a concrete node_id before execution.",
        }],
    })
    chat = SequencedChat([plan_response], critic_responses=[grounding_error])
    catalogue = [
        {
            "objectType": object_type,
            "objectId": f"{object_type}|1",
            "sliceable": True,
        }
        for object_type in object_types
    ] + [{"objectType": "ButterKnife", "objectId": "ButterKnife|1", "pickupable": True}]

    graph = decompose_task_to_graph(
        task,
        _task23_subgraph(task),
        qwen_chat=chat,
        scene_catalog=catalogue,
    )

    assert len(chat.prompts) == 1
    assert len(chat.critic_prompts) == 1
    projected_plan = json.dumps(chat.critic_prompts[0]["candidate_plan"])
    for forbidden_field in (
        "node_ids",
        "source_node_ids",
        "destination_node_ids",
        "source_object_ids",
        "destination_object_ids",
        "missing_reason",
        "recovery",
    ):
        assert forbidden_field not in projected_plan

    attempt = graph["planner_diagnostics"]["qwen"]["attempts"][0]
    semantic = attempt["semantic_validation"]
    assert semantic["verdict"]["status"] == "invalid"
    assert semantic["effective_verdict"] == {
        "protocol_status": "valid",
        "status": "valid",
        "errors": [],
    }
    assert semantic["discarded_out_of_scope_errors"][0]["code"] == (
        "out_of_scope_grounding_error"
    )
    assert semantic["discarded_out_of_scope_errors"][0]["original_code"] == (
        "missing_source_grounding"
    )
    assert attempt["validation_errors"] == []
    assert graph["planner_diagnostics"]["qwen"]["selected_attempt"] == 1
    assert all(
        not task_item["grounding"]["source_object_ids"]
        for task_item in graph["flat_tasks"]
    )


def test_semantic_critic_keeps_real_errors_when_discarding_grounding_errors() -> None:
    task = "Clear the central countertop by placing items in appropriate positions"
    grounding_error = {
        "task_id": "T1",
        "code": "missing_grounding",
        "field": "grounding.node_ids",
        "message": "Apple has no concrete scene node ID.",
        "invalid_values": ["[]"],
        "required_fix": "Add a valid node_id for Apple.",
    }
    destination_error = {
        "task_id": "T1",
        "code": "wrong_destination",
        "field": "grounding.destination_selector",
        "message": "CounterTop is the surface being cleared.",
        "invalid_values": ["CounterTop"],
        "required_fix": "Choose a destination other than CounterTop.",
    }
    chat = SequencedChat(
        [
            _interaction_plan("place", "Apple", "CounterTop"),
            _interaction_plan("place", "Apple", "Fridge"),
        ],
        critic_responses=[
            json.dumps({
                "status": "invalid",
                "errors": [grounding_error, destination_error],
            }),
            json.dumps({"status": "valid", "errors": []}),
        ],
    )
    catalogue = [
        {"objectType": "Apple", "objectId": "Apple|1", "pickupable": True},
        {"objectType": "CounterTop", "objectId": "CounterTop|1", "receptacle": True},
        {"objectType": "Fridge", "objectId": "Fridge|1", "receptacle": True},
    ]

    graph = decompose_task_to_graph(
        task,
        _task23_subgraph(task),
        qwen_chat=chat,
        scene_catalog=catalogue,
    )

    attempts = graph["planner_diagnostics"]["qwen"]["attempts"]
    first_semantic = attempts[0]["semantic_validation"]
    assert len(first_semantic["verdict"]["errors"]) == 2
    assert [item["code"] for item in first_semantic["effective_verdict"]["errors"]] == [
        "wrong_destination"
    ]
    assert len(first_semantic["discarded_out_of_scope_errors"]) == 1
    correction_codes = {
        item["code"] for item in chat.prompts[1]["correction"]["violations"]
    }
    assert "wrong_destination" in correction_codes
    serialized_correction = json.dumps(chat.prompts[1]["correction"])
    assert "node_ids" not in serialized_correction
    assert "missing_grounding" not in serialized_correction
    assert graph["planner_diagnostics"]["qwen"]["selected_attempt"] == 2

def _clean_plan_response(object_types: list[str]) -> str:
    return json.dumps({
        "reasoning_summary": "Clean every requested object.",
        "subtasks": [
            {
                "id": f"T{index}",
                "name": f"Clean {object_type}",
                "description": f"Clean the {object_type}.",
                "action": "clean",
                "action_args": {},
                "grounding": {
                    "node_ids": [],
                    "object_tags": [object_type],
                    "source_object_tags": [object_type],
                    "destination_object_tags": [],
                    "source_selector": {
                        "quantifier": "one",
                        "object_types": [object_type],
                    },
                    "destination_selector": {},
                    "relation_texts": [],
                    "status": "grounded",
                    "missing_reason": "",
                    "recovery": "none",
                },
                "depends_on": [],
                "termination_check": f"{object_type} is clean.",
            }
            for index, object_type in enumerate(object_types, start=1)
        ],
    })


def test_semantic_critic_intent_challenge_reparses_both_stages_once() -> None:
    task = "Put all shakers in the fridge"
    raw_plan = json.loads(_place_plan_response(["SaltShaker"], "Fridge"))
    raw_plan["subtasks"][0]["grounding"]["source_selector"]["quantifier"] = "all"
    critic_error = {
        "task_id": "T1",
        "code": "resolved_intent_mismatch",
        "field": "grounding.destination_selector",
        "message": "Re-check whether Fridge matches the public instruction.",
        "invalid_values": ["Fridge"],
        "required_fix": "Resolve the destination from the instruction again.",
    }
    role_response = json.dumps({
        "status": "resolved",
        "included_types": ["SaltShaker"],
        "excluded_types": [],
        "summary": "SaltShaker matches the requested category.",
    })
    chat = SequencedChat(
        [json.dumps(raw_plan), json.dumps(raw_plan)],
        critic_responses=[
            json.dumps({"status": "invalid", "errors": [critic_error]}),
            json.dumps({"status": "valid", "errors": []}),
        ],
        resolver_responses=[role_response, role_response],
    )

    graph = decompose_task_to_graph(
        task,
        _place_subgraph(task, ["all shakers"], "Fridge"),
        qwen_chat=chat,
        scene_catalog=_place_catalogue(["SaltShaker"], "Fridge"),
    )

    assert len(chat.prompts) == 2
    assert len(chat.action_prompts) == 2
    assert len(chat.role_prompts) == 2
    assert "semantic_correction" in chat.action_prompts[1]
    assert "semantic_correction" in chat.role_prompts[1]
    recovery = graph["planner_diagnostics"]["intent_recovery"]
    assert recovery["status"] == "recovered"
    assert recovery["critic_errors"][0]["code"] == "resolved_intent_mismatch"
    assert graph["flat_tasks"][0]["grounding"][
        "destination_selector"
    ]["object_types"] == ["Fridge"]


def test_semantic_critic_cannot_require_sink_for_source_only_clean_tasks() -> None:
    task = "Wash the bowl, mug, pot, and pan"
    object_types = ["Bowl", "Mug", "Pot", "Pan"]
    critic_errors = [
        {
            "task_id": f"T{index}",
            "code": "missing_destination",
            "field": "grounding.destination_selector",
            "message": f"{object_type} must be cleaned at a sink.",
            "invalid_values": [],
            "required_fix": "Add a SinkBasin destination_selector.",
        }
        for index, object_type in enumerate(object_types, start=1)
    ]
    chat = SequencedChat(
        [_clean_plan_response(object_types)],
        critic_responses=[
            json.dumps({"status": "invalid", "errors": critic_errors}),
        ],
    )
    catalogue = [
        {
            "objectType": object_type,
            "objectId": f"{object_type}|1",
            "dirtyable": True,
        }
        for object_type in object_types
    ]

    graph = decompose_task_to_graph(
        task,
        _task23_subgraph(task),
        qwen_chat=chat,
        scene_catalog=catalogue,
    )

    assert len(chat.prompts) == 1
    assert len(chat.critic_prompts) == 1
    assert "execution_action_contracts" not in chat.critic_prompts[0]
    clean_contract = chat.critic_prompts[0]["semantic_action_contracts"]["clean"]
    assert clean_contract == {"roles": ["source"]}
    assert any(
        "wash/clean never requires a sink" in rule
        for rule in chat.critic_prompts[0]["mandatory_decision_rules"]
    )

    attempt = graph["planner_diagnostics"]["qwen"]["attempts"][0]
    semantic = attempt["semantic_validation"]
    assert semantic["effective_verdict"] == {
        "protocol_status": "valid",
        "status": "valid",
        "errors": [],
    }
    discarded = semantic["discarded_out_of_scope_errors"]
    assert len(discarded) == 4
    assert {item["original_code"] for item in discarded} == {"missing_destination"}
    assert {item["code"] for item in discarded} == {
        "out_of_scope_action_contract_error"
    }
    assert {item["contract_action"] for item in discarded} == {"clean"}
    assert {item["unsupported_role"] for item in discarded} == {"destination"}
    assert attempt["validation_errors"] == []
    assert graph["planner_diagnostics"]["qwen"]["selected_attempt"] == 1
    assert all(
        not task_item["grounding"]["destination_selector"].get("object_types")
        for task_item in graph["flat_tasks"]
    )


@pytest.mark.parametrize(
    ("action", "destination", "violation_task_id"),
    [
        ("place", None, "T1"),
        ("clean", "SinkBasin", "T1"),
        ("clean", None, "unknown"),
        ("clean", None, None),
    ],
)
def test_semantic_contract_scope_keeps_ambiguous_or_real_role_errors(
    action: str,
    destination: str | None,
    violation_task_id: str | None,
) -> None:
    grounding = {
        "source_object_tags": ["Bowl"],
        "source_selector": {"quantifier": "one", "object_types": ["Bowl"]},
    }
    if destination is not None:
        grounding.update({
            "destination_object_tags": [destination],
            "destination_selector": {
                "quantifier": "one",
                "object_types": [destination],
            },
        })
    candidate_plan = {
        "subtasks": [{
            "id": "T1",
            "action": action,
            "grounding": grounding,
        }],
    }
    semantic_result = {
        "protocol_status": "valid",
        "status": "invalid",
        "errors": [{
            "task_id": violation_task_id,
            "code": "missing_destination",
            "field": "grounding.destination_selector",
            "message": "A destination is required.",
            "invalid_values": [],
            "required_fix": "Add a destination_selector.",
        }],
    }

    effective, discarded = enforce_semantic_critic_scope(
        semantic_result,
        candidate_plan=candidate_plan,
    )

    assert effective == semantic_result
    assert discarded == []


def test_deterministic_validator_still_rejects_clean_destination_selector() -> None:
    task = "Wash the bowl"
    raw_plan = json.loads(_interaction_plan("clean", "Bowl", "SinkBasin"))
    catalogue = [
        {"objectType": "Bowl", "objectId": "Bowl|1", "dirtyable": True},
        {"objectType": "SinkBasin", "objectId": "SinkBasin|1", "receptacle": True},
    ]

    validation = validate_planner_output(
        raw_plan,
        task=task,
        subgraph=_task23_subgraph(task),
        scene_catalog_summary=summarize_scene_catalogue(catalogue),
    )

    assert validation["status"] == "invalid"
    assert any(
        "destination_selector is not supported for action 'clean'" in error
        for error in validation["errors"]
    )


def test_task10_exact_drawer_with_parser_noise_uses_two_stage_intent() -> None:
    noisy_subgraph = {
        **SUBGRAPH,
        "task_spec": {
            "raw_task": TASK,
            "target_objects": ["open", "all", "drawers"],
        },
    }
    chat = SequencedChat([plan("open")])

    graph = decompose_task_to_graph(
        TASK,
        noisy_subgraph,
        qwen_chat=chat,
        scene_catalog=CATALOGUE,
    )

    resolution = graph["planner_diagnostics"]["semantic_goal_resolution"]
    assert resolution["status"] == "success"
    assert resolution["result"]["required_source_types"] == ["Drawer"]
    assert len(chat.action_prompts) == 1
    assert len(chat.role_prompts) == 1
    assert chat.prompts[0]["resolved_semantic_goal"][
        "required_source_types"
    ] == ["Drawer"]


def test_task8_two_stage_intent_classifies_complete_toggleable_table() -> None:
    task = "Turn off the faucet and light if either is on"
    subgraph = {
        "task": task,
        "task_spec": {
            "raw_task": task,
            "target_objects": [
                "turn", "off", "faucet", "light", "if", "either",
            ],
        },
        "nodes": [],
        "triples": [],
    }
    catalogue = [
        {
            "objectType": "Faucet",
            "objectId": "Faucet|1",
            "toggleable": True,
        },
        {
            "objectType": "LightSwitch",
            "objectId": "LightSwitch|1",
            "toggleable": True,
        },
        {
            "objectType": "StoveKnob",
            "objectId": "StoveKnob|1",
            "toggleable": True,
        },
    ]
    raw_plan = json.dumps({
        "reasoning_summary": "Turn off both requested fixtures.",
        "subtasks": [
            {
                "id": "T1",
                "name": "Turn off faucet",
                "description": "Turn off the faucet.",
                "action": "toggle_off",
                "action_args": {},
                "grounding": {
                    "node_ids": [],
                    "object_tags": ["Faucet"],
                    "source_object_tags": ["Faucet"],
                    "source_selector": {
                        "quantifier": "one",
                        "object_types": ["Faucet"],
                    },
                    "relation_texts": [],
                },
                "depends_on": [],
                "termination_check": "Faucet is off.",
            },
            {
                "id": "T2",
                "name": "Turn off light",
                "description": "Turn off the light switch.",
                "action": "toggle_off",
                "action_args": {},
                "grounding": {
                    "node_ids": [],
                    "object_tags": ["LightSwitch"],
                    "source_object_tags": ["LightSwitch"],
                    "source_selector": {
                        "quantifier": "one",
                        "object_types": ["LightSwitch"],
                    },
                    "relation_texts": [],
                },
                "depends_on": [],
                "termination_check": "Light switch is off.",
            },
        ],
    })
    chat = SequencedChat([raw_plan])

    graph = decompose_task_to_graph(
        task,
        subgraph,
        qwen_chat=chat,
        scene_catalog=catalogue,
    )

    assert len(chat.action_prompts) == 1
    assert len(chat.role_prompts) == 1
    assert chat.action_prompts[0]["stage"] == "action"
    role_prompt = chat.role_prompts[0]
    source_rows = role_prompt["role_tables"]["source"]
    assert [row["object_type"] for row in source_rows] == [
        "Faucet", "LightSwitch", "StoveKnob",
    ]
    diagnostics = graph["planner_diagnostics"]["semantic_goal_resolution"]
    included_ids = diagnostics["role_result"]["roles"]["source"][
        "included_row_ids"
    ]
    included_types = {source_rows[index]["object_type"] for index in included_ids}
    assert included_types == {"Faucet", "LightSwitch"}
    resolved_goal = chat.prompts[0]["resolved_semantic_goal"]
    assert set(resolved_goal["required_source_types"]) == {
        "Faucet", "LightSwitch",
    }
    assert resolved_goal["quantifier"] == "one"
    assert diagnostics["status"] == "success"
    assert diagnostics["reason"] == "mandatory two-stage intent resolution"


def test_role_request_exposes_complete_action_specific_tables() -> None:
    task = "Put all tomatoes and groceries in the fridge"
    subgraph = _place_subgraph(
        task,
        ["all tomatoes and groceries", "Fridge"],
        "Fridge",
    )
    summary = summarize_scene_catalogue(
        _place_catalogue(["Tomato", "Apple", "Bread"], "Fridge")
    )
    request = role_intent_resolution_request(
        task=task,
        subgraph=subgraph,
        action="place",
        scene_catalog_summary=summary,
    )

    assert request["stage"] == "roles"
    source_rows = request["role_tables"]["source"]
    destination_rows = request["role_tables"]["destination"]
    assert [row["row_id"] for row in source_rows] == [0, 1, 2]
    assert [row["object_type"] for row in source_rows] == [
        "Apple", "Bread", "Tomato",
    ]
    assert [row["row_id"] for row in destination_rows] == [0]
    assert [row["object_type"] for row in destination_rows] == ["Fridge"]
    assert "objectId" not in json.dumps(request)


def test_critic_contract_and_plan_projection_hide_execution_details() -> None:
    contracts = semantic_action_contracts_for_critic()

    assert contracts["slice"] == {"roles": ["source"]}
    assert contracts["drop"] == {"roles": ["source"]}
    assert contracts["push"] == {"roles": ["source"]}
    assert contracts["fill"] == {
        "roles": ["source"],
        "semantic_args": {
            "fillLiquid": {"values": ["water", "coffee", "wine"]},
        },
    }
    projection = semantic_critic_plan_projection({
        "subtasks": [
            {"id": "T1", "action": "push", "action_args": {"moveMagnitude": 200}},
            {"id": "T2", "action": "fill", "action_args": {"fillLiquid": "water"}},
        ],
    })
    assert projection["subtasks"][0]["action_args"] == {}
    assert projection["subtasks"][1]["action_args"] == {"fillLiquid": "water"}


@pytest.mark.parametrize(
    ("action", "code", "field", "message"),
    [
        ("slice", "missing_tool", "grounding", "A held tool must be acquired first."),
        ("drop", "missing_pick", "grounding", "The source is not held and needs an ancestor pick."),
        ("move_held", "missing_inventory", "grounding", "No matching held inventory object exists."),
        ("push", "action_arg_out_of_range", "action_args.moveMagnitude", "Magnitude is too large."),
        ("open", "invalid_affordance", "grounding.source_selector", "Drawer must prove its openable affordance."),
    ],
)
def test_critic_discards_execution_contract_errors_for_all_actions(
    action: str,
    code: str,
    field: str,
    message: str,
) -> None:
    semantic_result = {
        "protocol_status": "valid",
        "status": "invalid",
        "errors": [{
            "task_id": "T1",
            "code": code,
            "field": field,
            "message": message,
            "invalid_values": [],
            "required_fix": "Change an execution detail.",
        }],
    }
    candidate = {"subtasks": [{"id": "T1", "action": action, "grounding": {}}]}

    effective, discarded = enforce_semantic_critic_scope(
        semantic_result,
        candidate_plan=candidate,
    )

    assert effective["status"] == "valid"
    assert effective["errors"] == []
    assert discarded[0]["code"] == "out_of_scope_execution_error"
    assert discarded[0]["contract_action"] == action


@pytest.mark.parametrize("field", ["action_args.fillLiquid", "action_args"])
def test_critic_retains_semantic_fill_liquid_mismatch(field: str) -> None:
    semantic_result = {
        "protocol_status": "valid",
        "status": "invalid",
        "errors": [{
            "task_id": "T1",
            "code": "wrong_liquid",
            "field": field,
            "message": "The instruction requests water, not coffee.",
            "invalid_values": ["coffee"],
            "required_fix": "Use water.",
        }],
    }
    candidate = {"subtasks": [{"id": "T1", "action": "fill", "grounding": {}}]}

    effective, discarded = enforce_semantic_critic_scope(
        semantic_result,
        candidate_plan=candidate,
    )

    assert effective == semantic_result
    assert discarded == []


def test_task7_missing_tool_critic_error_is_discarded_without_retry() -> None:
    task = "Slice the bread, lettuce, tomato, and egg"
    object_types = ["Bread", "Lettuce", "Tomato", "Egg"]
    raw_plan = {
        "reasoning_summary": "Slice every requested object.",
        "subtasks": [
            {
                "id": f"T{index}",
                "name": f"Slice {object_type}",
                "description": f"Slice the {object_type}.",
                "action": "slice",
                "grounding": {
                    "object_tags": [object_type],
                    "source_object_tags": [object_type],
                    "source_selector": {
                        "quantifier": "one",
                        "object_types": [object_type],
                    },
                },
                "depends_on": [],
                "termination_check": f"{object_type} is sliced.",
            }
            for index, object_type in enumerate(object_types, start=1)
        ],
    }
    critic_error = {
        "task_id": "T1",
        "code": "missing_tool",
        "field": "grounding",
        "message": "Slice requires a held tool and an earlier pickup.",
        "invalid_values": [],
        "required_fix": "Pick up a cutting tool first.",
    }
    chat = SequencedChat(
        [json.dumps(raw_plan)],
        critic_responses=[json.dumps({"status": "invalid", "errors": [critic_error]})],
    )
    catalogue = [
        {"objectType": value, "objectId": f"{value}|1", "sliceable": True}
        for value in object_types
    ] + [{"objectType": "Knife", "objectId": "Knife|1", "pickupable": True}]

    graph = decompose_task_to_graph(
        task,
        {"task": task, "task_spec": {"raw_task": task}, "nodes": [], "triples": []},
        qwen_chat=chat,
        scene_catalog=catalogue,
    )

    assert len(chat.prompts) == 1
    assert [item["action"] for item in graph["flat_tasks"]] == ["slice"] * 4
    attempt = graph["planner_diagnostics"]["qwen"]["attempts"][0]
    assert attempt["validation_errors"] == []
    assert attempt["semantic_validation"]["discarded_out_of_scope_errors"][0][
        "code"
    ] == "out_of_scope_execution_error"
