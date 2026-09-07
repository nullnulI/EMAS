from __future__ import annotations

import json

import pytest

import planning.extract_subgraph as extract_subgraph
import planning.task_graph as task_graph


EMPTY_SUBGRAPH = {
    "task": "",
    "task_spec": {},
    "seed_nodes": [],
    "nodes": [],
    "triples": [],
}


def catalogue_entry(object_type: str, **affordances: bool) -> dict:
    return {
        "object_type": object_type,
        "count": 1,
        "affordances": affordances,
        "parent_locations": [],
    }


def place_catalogue() -> list[dict]:
    return [
        catalogue_entry("Apple", pickupable=True),
        catalogue_entry("Bowl", receptacle=True),
        catalogue_entry("Plate", receptacle=True),
    ]


class QueueChat:
    def __init__(self, *responses: dict) -> None:
        self.responses = [json.dumps(response) for response in responses]
        self.requests: list[dict] = []

    def reset(self) -> None:
        pass

    def __call__(self, prompt: str) -> str:
        self.requests.append(json.loads(prompt))
        return self.responses.pop(0)


def fail_if_called(*_args, **_kwargs):
    raise AssertionError("the active Qwen path must not parse task text")


def test_role_request_has_no_deterministic_extraction_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(extract_subgraph, "parse_task_locally", fail_if_called)

    request = task_graph.role_intent_resolution_request(
        task="Pick up the butter knife from the drawer",
        subgraph={
            **EMPTY_SUBGRAPH,
            "task_spec": {"target_objects": ["ButterKnife"]},
        },
        action="pick",
        scene_catalog_summary=[
            catalogue_entry("ButterKnife", pickupable=True),
            catalogue_entry("Drawer", receptacle=True),
        ],
    )

    assert request["instruction"] == "Pick up the butter knife from the drawer"
    for forbidden in (
        "task_spec",
        "role_mentions",
        "explicit_role_constraints",
        "source_scope_constraint",
    ):
        assert forbidden not in request


def test_full_two_stage_resolver_does_not_call_local_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(extract_subgraph, "parse_task_locally", fail_if_called)
    chat = QueueChat(
        {"status": "resolved", "action": "place", "summary": "place apple"},
        {
            "status": "resolved",
            "roles": {
                "source": {
                    "status": "specified",
                    "quantifier": "one",
                    "reviewed_row_count": 1,
                    "included_row_ids": [0],
                },
                "destination": {
                    "status": "specified",
                    "quantifier": "one",
                    "reviewed_row_count": 2,
                    "included_row_ids": [0],
                },
            },
            "semantic_args": {},
            "summary": "apple to bowl",
        },
    )

    root_intent, _, diagnostics, error = (
        task_graph.resolve_semantic_goal_for_planning(
            task="Put the apple in the bowl",
            subgraph=dict(EMPTY_SUBGRAPH),
            scene_catalog_summary=place_catalogue(),
            model_path=None,
            conv_mode="v0_mmtag",
            num_gpus=1,
            qwen_chat=chat,
            planning_max_new_tokens=2048,
        )
    )

    assert error is None
    assert diagnostics["status"] == "success"
    assert root_intent["roles"] == {
        "source": ["Apple"],
        "destination": ["Bowl"],
    }
    assert [request["stage"] for request in chat.requests] == ["action", "roles"]


def test_active_qwen_sanitizer_does_not_infer_missing_semantics() -> None:
    subgraph = {
        "task": "Slice the apple and put it in the bowl",
        "task_spec": {
            "target_objects": ["Apple"],
            "destination_receptacles": ["Bowl"],
        },
        "seed_nodes": [],
        "nodes": [
            {"pruned_id": 1, "object_tag": "Apple", "caption": "an apple"},
            {"pruned_id": 2, "object_tag": "Bowl", "caption": "a bowl"},
        ],
        "triples": [
            {"source": 1, "target": 2, "relation": "near", "text": "Apple near Bowl"},
        ],
    }
    raw_plan = {
        "reasoning_summary": "model output with deliberately missing fields",
        "subtasks": [
            {
                "id": "T1",
                "name": "Slice the apple",
                "description": "Use a knife to slice the apple",
                "grounding": {},
                "depends_on": [],
            },
            {
                "id": "T2",
                "name": "Put the apple in the bowl",
                "description": "Place it into the destination",
                "action": "place",
                "grounding": {
                    "node_ids": [1, 2],
                    "object_tags": ["Apple", "Bowl"],
                },
                "depends_on": ["T1"],
            },
        ],
    }

    graph, _ = task_graph._prepare_planner_candidate(
        raw_plan,
        task="Slice the apple and put it in the bowl",
        subgraph=subgraph,
    )
    by_id = {item["id"]: item for item in graph["flat_tasks"]}

    assert by_id["T1"]["action"] == "other"
    assert by_id["T1"]["grounding"]["object_tags"] == []
    assert by_id["T1"]["grounding"]["source_object_tags"] == []
    assert by_id["T1"]["grounding"]["destination_object_tags"] == []
    assert by_id["T2"]["grounding"]["object_tags"] == ["Apple", "Bowl"]
    assert by_id["T2"]["grounding"]["source_object_tags"] == []
    assert by_id["T2"]["grounding"]["destination_object_tags"] == []
    assert by_id["T2"]["grounding"]["relation_texts"] == []


def test_no_inference_mode_preserves_explicit_model_roles() -> None:
    normalized = task_graph.sanitize_subtasks(
        {
            "subtasks": [
                {
                    "id": "T1",
                    "name": "Place apple",
                    "action": "place",
                    "grounding": {
                        "object_tags": ["Apple", "Bowl", "Apple"],
                        "source_object_tags": ["Apple"],
                        "destination_object_tags": ["Bowl"],
                        "relation_texts": ["Apple in Bowl"],
                    },
                },
            ],
        },
        EMPTY_SUBGRAPH,
        "Put the apple in the bowl",
        allow_semantic_inference=False,
    )
    grounding = normalized["subtasks"][0]["grounding"]

    assert normalized["subtasks"][0]["action"] == "place"
    assert grounding["object_tags"] == ["Apple", "Bowl"]
    assert grounding["source_object_tags"] == ["Apple"]
    assert grounding["destination_object_tags"] == ["Bowl"]
    assert grounding["relation_texts"] == ["Apple in Bowl"]
