from __future__ import annotations

from copy import deepcopy
import json

import pytest
import planning.task_graph as task_graph_module

from action_contracts import ACTION_CONTRACTS
from planning.task_graph import (
    action_intent_resolution_request,
    resolve_semantic_goal_for_planning,
    role_intent_resolution_request,
    validate_action_intent_resolution,
    validate_semantic_goal_resolution,
)


EMPTY_SUBGRAPH = {
    "task": "",
    "task_spec": {},
    "seed_nodes": [],
    "nodes": [],
    "triples": [],
}


def catalogue_entry(
    object_type: str,
    *,
    count: int = 1,
    parent_locations: list[dict] | None = None,
    **affordances: bool,
) -> dict:
    return {
        "object_type": object_type,
        "count": count,
        "affordances": affordances,
        "parent_locations": list(parent_locations or []),
        # A summary carrying an accidental instance identifier must still not
        # leak it into either role table.
        "objectId": f"{object_type}|instance-to-drop",
    }


class QueueChat:
    def __init__(self, *responses: dict) -> None:
        self.responses = [json.dumps(response) for response in responses]
        self.requests: list[dict] = []
        self.messages: list[dict] = []

    def reset(self) -> None:
        pass

    def __call__(self, prompt: str) -> str:
        self.requests.append(json.loads(prompt))
        return self.responses.pop(0)


def resolve_with_chat(*, task: str, catalogue: list[dict], chat: QueueChat):
    return resolve_semantic_goal_for_planning(
        task=task,
        subgraph=deepcopy(EMPTY_SUBGRAPH),
        scene_catalog_summary=catalogue,
        model_path=None,
        conv_mode="v0_mmtag",
        num_gpus=1,
        qwen_chat=chat,
        planning_max_new_tokens=2048,
    )


def test_action_request_contains_only_contract_whitelist() -> None:
    request = action_intent_resolution_request(task="Open the drawer")

    allowed = [row["action"] for row in request["allowed_actions"]]
    assert allowed == list(ACTION_CONTRACTS)
    assert set(allowed).isdisjoint({"navigate", "find", "inspect", "other"})
    for action in allowed:
        validated = validate_action_intent_resolution({
            "status": "resolved",
            "action": action,
            "summary": "selected",
        })
        assert validated["protocol_status"] == "valid"


@pytest.mark.parametrize("action", ["navigate", "unknown"])
def test_action_validator_rejects_non_whitelisted_actions(action: str) -> None:
    validated = validate_action_intent_resolution({
        "status": "resolved",
        "action": action,
        "summary": "invalid choice",
    })

    assert validated["protocol_status"] == "invalid"
    assert any("exact allowed interaction action" in error for error in validated["errors"])


def test_role_tables_are_stable_position_keyed_and_drop_instance_ids() -> None:
    catalogue = [
        catalogue_entry("Mug", count=2, pickupable=True, receptacle=True),
        catalogue_entry("Cabinet", openable=True),
        catalogue_entry("CounterTop", receptacle=True),
        catalogue_entry(
            "Bowl",
            pickupable=True,
            receptacle=True,
            parent_locations=[{
                "parent_type": "CounterTop",
                "instance_count": 1,
                "parent_position": {"x": 1.0, "z": 2.0},
            }],
        ),
        catalogue_entry("Apple", pickupable=True),
    ]

    request = role_intent_resolution_request(
        task="Put the apple in the bowl",
        subgraph=deepcopy(EMPTY_SUBGRAPH),
        action="place",
        scene_catalog_summary=catalogue,
    )
    reversed_request = role_intent_resolution_request(
        task="Put the apple in the bowl",
        subgraph=deepcopy(EMPTY_SUBGRAPH),
        action="place",
        scene_catalog_summary=list(reversed(catalogue)),
    )

    assert request["role_tables"] == reversed_request["role_tables"]
    assert [row["row_id"] for row in request["role_tables"]["source"]] == [0, 1, 2]
    assert [row["object_type"] for row in request["role_tables"]["source"]] == [
        "Apple",
        "Bowl",
        "Mug",
    ]
    assert [row["row_id"] for row in request["role_tables"]["destination"]] == [0, 1, 2]
    assert [row["object_type"] for row in request["role_tables"]["destination"]] == [
        "Bowl",
        "CounterTop",
        "Mug",
    ]
    encoded_tables = json.dumps(request["role_tables"])
    assert "objectId" not in encoded_tables
    assert "instance-to-drop" not in encoded_tables
    assert "parent_position" not in encoded_tables
    assert request["selection_encoding"] == "included_row_ids_v2"
    for role, rows in request["role_tables"].items():
        role_schema = request["output_schema"]["roles"][role]
        expected_status = (
            "specified" if role == "source" else "specified or unspecified"
        )
        assert role_schema["status"] == expected_status
        assert role_schema["reviewed_row_count"] == len(rows)
        assert role_schema["included_row_ids"] == []
        assert "classifications" not in role_schema


def place_request() -> dict:
    return role_intent_resolution_request(
        task="Put the apple in the bowl",
        subgraph=deepcopy(EMPTY_SUBGRAPH),
        action="place",
        scene_catalog_summary=[
            catalogue_entry("Apple", pickupable=True),
            catalogue_entry("Bowl", receptacle=True),
            catalogue_entry("Plate", receptacle=True),
        ],
    )


def valid_place_response() -> dict:
    return {
        "status": "resolved",
        "roles": {
            "source": {
                "status": "specified",
                "quantifier": "one",
                "classifications": ["included"],
            },
            "destination": {
                "status": "specified",
                "quantifier": "one",
                "classifications": ["included", "excluded"],
            },
        },
        "semantic_args": {},
        "summary": "apple to bowl",
    }



def valid_sparse_place_response() -> dict:
    return {
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
    }


def test_sparse_row_id_selection_is_normalized_to_legacy_diagnostics() -> None:
    validated = validate_semantic_goal_resolution(
        valid_sparse_place_response(),
        place_request(),
    )

    assert validated["protocol_status"] == "valid"
    assert validated["roles"]["source"]["included_row_ids"] == [0]
    assert validated["roles"]["source"]["classifications"] == ["included"]
    assert validated["roles"]["destination"]["included_row_ids"] == [0]
    assert validated["roles"]["destination"]["classifications"] == [
        "included",
        "excluded",
    ]


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    [
        (
            lambda result: result["roles"]["source"].pop("reviewed_row_count"),
            "roles.source.reviewed_row_count must be 1",
        ),
        (
            lambda result: result["roles"]["source"].update(
                included_row_ids=["0"]
            ),
            "roles.source.included_row_ids must contain only integer row ids",
        ),
        (
            lambda result: result["roles"]["source"].update(
                included_row_ids=[True]
            ),
            "roles.source.included_row_ids must contain only integer row ids",
        ),
        (
            lambda result: result["roles"]["destination"].update(
                included_row_ids=[0, 0]
            ),
            "roles.destination.included_row_ids must not contain duplicates",
        ),
        (
            lambda result: result["roles"]["destination"].update(
                included_row_ids=[2]
            ),
            "roles.destination.included_row_ids contains unknown row ids",
        ),
        (
            lambda result: result["roles"]["source"].update(
                classifications=["included"]
            ),
            "roles.source must use included_row_ids or classifications, not both",
        ),
    ],
    ids=[
        "review-count",
        "string-id",
        "bool-id",
        "duplicate-id",
        "unknown-id",
        "mixed-encodings",
    ],
)
def test_role_validator_rejects_malformed_sparse_selection(
    mutate,
    expected_error: str,
) -> None:
    result = valid_sparse_place_response()
    mutate(result)

    validated = validate_semantic_goal_resolution(result, place_request())

    assert validated["protocol_status"] == "invalid"
    assert any(expected_error in error for error in validated["errors"])


def test_sparse_selection_scales_and_normalizes_row_order() -> None:
    request = role_intent_resolution_request(
        task="Put the selected objects in storage",
        subgraph=deepcopy(EMPTY_SUBGRAPH),
        action="place",
        scene_catalog_summary=[
            *[
                catalogue_entry(f"Object{index:02d}", pickupable=True)
                for index in range(29)
            ],
            *[
                catalogue_entry(f"Bin{index:02d}", receptacle=True)
                for index in range(19)
            ],
        ],
    )
    response = {
        "status": "resolved",
        "roles": {
            "source": {
                "status": "specified",
                "quantifier": "all",
                "reviewed_row_count": 29,
                "included_row_ids": [28, 0, 14],
            },
            "destination": {
                "status": "specified",
                "quantifier": "one",
                "reviewed_row_count": 19,
                "included_row_ids": [7],
            },
        },
        "semantic_args": {},
        "summary": "sparse selection over long tables",
    }

    validated = validate_semantic_goal_resolution(response, request)

    assert validated["protocol_status"] == "valid"
    assert validated["roles"]["source"]["included_row_ids"] == [0, 14, 28]
    assert len(validated["roles"]["source"]["classifications"]) == 29
    assert validated["roles"]["destination"]["included_row_ids"] == [7]
    assert len(validated["roles"]["destination"]["classifications"]) == 19


def test_sparse_selection_uses_row_ids_not_list_positions() -> None:
    request = place_request()
    request["role_tables"]["source"][0]["row_id"] = 11
    request["role_tables"]["destination"][0]["row_id"] = 7
    request["role_tables"]["destination"][1]["row_id"] = 9
    response = valid_sparse_place_response()
    response["roles"]["source"]["included_row_ids"] = [11]
    response["roles"]["destination"]["included_row_ids"] = [9]

    validated = validate_semantic_goal_resolution(response, request)

    assert validated["protocol_status"] == "valid"
    assert validated["roles"]["source"]["included_row_ids"] == [11]
    assert validated["roles"]["destination"]["included_row_ids"] == [9]
    assert validated["roles"]["destination"]["classifications"] == [
        "excluded",
        "included",
    ]


def test_resolver_maps_noncontiguous_row_ids_to_correct_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = place_request()
    request["role_tables"]["source"][0]["row_id"] = 11
    request["role_tables"]["destination"][0]["row_id"] = 7
    request["role_tables"]["destination"][1]["row_id"] = 9
    monkeypatch.setattr(
        task_graph_module,
        "role_intent_resolution_request",
        lambda **_kwargs: deepcopy(request),
    )
    chat = QueueChat(
        {"status": "resolved", "action": "place", "summary": "place apple"},
        {
            "status": "resolved",
            "roles": {
                "source": {
                    "status": "specified",
                    "quantifier": "one",
                    "reviewed_row_count": 1,
                    "included_row_ids": [11],
                },
                "destination": {
                    "status": "specified",
                    "quantifier": "one",
                    "reviewed_row_count": 2,
                    "included_row_ids": [9],
                },
            },
            "semantic_args": {},
            "summary": "apple to plate",
        },
    )

    root_intent, semantic_goal, diagnostics, error = resolve_with_chat(
        task="Put the apple on the plate",
        catalogue=[
            catalogue_entry("Apple", pickupable=True),
            catalogue_entry("Bowl", receptacle=True),
            catalogue_entry("Plate", receptacle=True),
        ],
        chat=chat,
    )

    assert error is None
    assert diagnostics["status"] == "success"
    assert root_intent["roles"]["source"] == ["Apple"]
    assert root_intent["roles"]["destination"] == ["Plate"]
    assert semantic_goal["destination_types"] == ["Plate"]


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    [
        (
            lambda result: result["roles"]["source"].update(classifications=[]),
            "roles.source.classifications length must be 1, got 0",
        ),
        (
            lambda result: result["roles"]["destination"].update(
                classifications=["included", "maybe"]
            ),
            "roles.destination.classifications contains invalid values",
        ),
        (
            lambda result: result["roles"]["destination"].update(
                classifications=["excluded", "excluded"]
            ),
            "specified destination requires exactly one included row",
        ),
        (
            lambda result: result["roles"]["destination"].update(
                classifications=["included", "included"]
            ),
            "specified destination requires exactly one included row",
        ),
    ],
    ids=["length", "enum", "specified-zero", "specified-many"],
)
def test_role_validator_rejects_malformed_complete_classification(
    mutate,
    expected_error: str,
) -> None:
    result = valid_place_response()
    mutate(result)

    validated = validate_semantic_goal_resolution(result, place_request())

    assert validated["protocol_status"] == "invalid"
    assert any(expected_error in error for error in validated["errors"])


def test_same_type_can_be_source_and_destination_when_quantifier_is_one() -> None:
    request = role_intent_resolution_request(
        task="Put one bowl into another bowl",
        subgraph=deepcopy(EMPTY_SUBGRAPH),
        action="place",
        scene_catalog_summary=[
            catalogue_entry("Bowl", count=2, pickupable=True, receptacle=True),
        ],
    )
    response = {
        "status": "resolved",
        "roles": {
            "source": {
                "status": "specified",
                "quantifier": "one",
                "classifications": ["included"],
            },
            "destination": {
                "status": "specified",
                "quantifier": "one",
                "classifications": ["included"],
            },
        },
        "semantic_args": {},
        "summary": "one bowl into another",
    }

    assert request["role_tables"]["source"][0]["object_type"] == "Bowl"
    assert request["role_tables"]["destination"][0]["object_type"] == "Bowl"
    validated = validate_semantic_goal_resolution(response, request)
    assert validated["protocol_status"] == "valid"
    assert validated["roles"]["source"]["included_row_ids"] == [0]
    assert validated["roles"]["destination"]["included_row_ids"] == [0]


def test_no_match_may_preserve_one_specified_destination() -> None:
    request = place_request()
    response = {
        "status": "no_match",
        "roles": {
            "source": {
                "status": "specified",
                "quantifier": "one",
                "classifications": ["excluded"],
            },
            "destination": {
                "status": "specified",
                "quantifier": "one",
                "classifications": ["included", "excluded"],
            },
        },
        "semantic_args": {},
        "summary": "destination resolved but no source matched",
    }

    validated = validate_semantic_goal_resolution(response, request)

    assert validated["protocol_status"] == "valid"
    assert validated["status"] == "no_match"
    assert validated["roles"]["source"]["included_row_ids"] == []
    assert validated["roles"]["destination"]["included_row_ids"] == [0]


def test_resolve_fails_closed_for_all_source_overlapping_destination() -> None:
    catalogue = [
        catalogue_entry("Bowl", count=2, pickupable=True, receptacle=True),
    ]
    chat = QueueChat(
        {"status": "resolved", "action": "place", "summary": "place bowls"},
        {
            "status": "resolved",
            "roles": {
                "source": {
                    "status": "specified",
                    "quantifier": "all",
                    "classifications": ["included"],
                },
                "destination": {
                    "status": "specified",
                    "quantifier": "one",
                    "classifications": ["included"],
                },
            },
            "semantic_args": {},
            "summary": "all bowls into a bowl",
        },
    )

    root_intent, semantic_goal, diagnostics, error = resolve_with_chat(
        task="Put all bowls into a bowl",
        catalogue=catalogue,
        chat=chat,
    )

    assert error is not None
    assert "source quantifier all overlaps the selected destination" in error
    assert diagnostics["stage"] == "overlapping_all_roles"
    assert root_intent["semantic_goal_resolved"] is False
    assert semantic_goal["required_source_types"] == []


def test_subgraph_resolve_fails_closed_when_one_node_has_both_roles() -> None:
    subgraph = {
        **deepcopy(EMPTY_SUBGRAPH),
        "nodes": [{
            "pruned_id": 7,
            "object_tag": "bowl",
            "caption": "a bowl",
            "possible_tags": ["Bowl"],
        }],
    }
    chat = QueueChat(
        {"status": "resolved", "action": "place", "summary": "place a bowl"},
        {
            "status": "resolved",
            "roles": {
                "source": {
                    "status": "specified",
                    "quantifier": "one",
                    "classifications": ["included"],
                },
                "destination": {
                    "status": "specified",
                    "quantifier": "one",
                    "classifications": ["included"],
                },
            },
            "semantic_args": {},
            "summary": "same node selected for both roles",
        },
    )

    root_intent, semantic_goal, diagnostics, error = resolve_semantic_goal_for_planning(
        task="Put one bowl into another bowl",
        subgraph=subgraph,
        scene_catalog_summary=[],
        model_path=None,
        conv_mode="v0_mmtag",
        num_gpus=1,
        qwen_chat=chat,
        planning_max_new_tokens=2048,
    )

    assert error is not None
    assert "overlap" in error.lower()
    assert diagnostics["status"] == "failed"
    assert root_intent["semantic_goal_resolved"] is False
    assert semantic_goal["required_source_nodes"] == []


def test_fill_semantic_argument_is_validated_and_assembled() -> None:
    catalogue = [catalogue_entry("Mug", canFillWithLiquid=True)]
    chat = QueueChat(
        {"status": "resolved", "action": "fill", "summary": "fill a mug"},
        {
            "status": "resolved",
            "roles": {
                "source": {
                    "status": "specified",
                    "quantifier": "one",
                    "classifications": ["included"],
                },
            },
            "semantic_args": {"fillLiquid": "COFFEE"},
            "summary": "fill the mug with coffee",
        },
    )

    root_intent, semantic_goal, diagnostics, error = resolve_with_chat(
        task="Fill the mug with coffee",
        catalogue=catalogue,
        chat=chat,
    )

    assert error is None
    assert diagnostics["status"] == "success"
    assert root_intent["semantic_args"] == {"fillLiquid": "coffee"}
    assert semantic_goal["semantic_args"] == {"fillLiquid": "coffee"}
    assert [request["stage"] for request in chat.requests] == ["action", "roles"]
