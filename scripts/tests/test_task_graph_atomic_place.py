from __future__ import annotations

import json

from planning.extract_subgraph import parse_task_locally
from planning.task_graph import (
    decompose_task_to_graph,
    rebuild_task_graph_views,
    restore_compound_object_tags,
    semantic_entity_catalog,
    split_multi_entity_find_subtasks,
)


TASK = "Put the remotecontrol, keys, and watch in the box"
EMPTY_SUBGRAPH = {
    "task": TASK,
    "task_spec": {
        "target_objects": ["remotecontrol", "keys", "watch", "box"],
        "source_receptacles": [],
        "destination_receptacles": ["box"],
        "landmarks": [],
    },
    "seed_nodes": [],
    "nodes": [],
    "triples": [],
}

def critic_response(prompt: str) -> str | None:
    payload = json.loads(prompt)
    if payload.get("request") == "validate_candidate_plan":
        return json.dumps({"status": "valid", "errors": []})
    return None


def intent_response(
    payload: dict,
    *,
    action: str,
    source_types: set[str],
    destination_types: set[str] | None = None,
    source_quantifier: str = "one",
) -> str | None:
    """Return a protocol-valid response for either intent-resolution stage."""
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
    roles = {}
    for role, rows in payload.get("role_tables", {}).items():
        selected = source_types if role == "source" else destination_types
        roles[role] = {
            "status": (
                "specified"
                if role == "source" or selected
                else "unspecified"
            ),
            "quantifier": source_quantifier if role == "source" else "one",
            "classifications": [
                "included"
                if str(row.get("object_type") or row.get("object_tag")) in selected
                else "excluded"
                for row in rows
            ],
        }
    return json.dumps({
        "status": "resolved",
        "roles": roles,
        "semantic_args": {},
        "summary": "All action roles are classified.",
    })


class AggregatedPlaceChat:
    def __init__(self) -> None:
        self.prompt: dict | None = None

    def __call__(self, prompt: str) -> str:
        payload = json.loads(prompt)
        response = intent_response(
            payload,
            action="place",
            source_types={"Parcel"},
            destination_types={"StoragePod"},
        )
        if response is not None:
            return response
        if payload.get("request") == "validate_candidate_plan":
            return json.dumps({"status": "valid", "errors": []})
        self.prompt = payload
        subtasks = []
        for index, tag in enumerate(("remotecontrol", "keys", "watch", "box"), start=1):
            subtasks.append(
                {
                    "id": f"T{index}",
                    "name": f"Find {tag}",
                    "description": f"Locate the {tag}.",
                    "action": "find",
                    "grounding": {
                        "node_ids": [],
                        "object_tags": [tag],
                        "relation_texts": [],
                    },
                    "depends_on": [],
                    "termination_check": f"{tag} is located.",
                }
            )
        subtasks.append(
            {
                "id": "T5",
                "name": "Place items in box",
                "description": "Pick up the remotecontrol, keys, and watch and place them inside the box.",
                "action": "place",
                "grounding": {
                    "node_ids": [],
                    "object_tags": ["remotecontrol", "keys", "watch", "box"],
                    "relation_texts": [
                        "remotecontrol is in box",
                        "keys is in box",
                        "watch is in box",
                    ],
                    "source_selector": {"quantifier": "one", "object_types": ["Parcel"]},
                    "destination_selector": {"quantifier": "one", "object_types": ["StoragePod"]},
                },
                "depends_on": ["T1", "T2", "T3", "T4"],
                "termination_check": "All three items are inside the box.",
            }
        )
        return json.dumps({"reasoning_summary": "Find items, then place them.", "subtasks": subtasks})


def test_initial_planner_receives_agent_state_and_splits_aggregate_place() -> None:
    chat = AggregatedPlaceChat()
    agent_context = {
        "agent_count": 2,
        "agents": [
            {"agent_id": "0", "inventory": [], "skills": ["find", "place"]},
            {"agent_id": "1", "inventory": [], "skills": ["find", "place"]},
        ],
        "inventory_capacity_per_agent": 1,
    }

    scene_catalog = [
        {"objectType": "Parcel", "objectId": "Parcel|1", "pickupable": True},
        {"objectType": "StoragePod", "objectId": "StoragePod|1", "receptacle": True},
    ]
    graph = decompose_task_to_graph(
        TASK,
        EMPTY_SUBGRAPH,
        qwen_chat=chat,
        agent_context=agent_context,
        scene_catalog=scene_catalog,
    )

    assert chat.prompt is not None
    assert chat.prompt["agent_context"] == agent_context
    assert chat.prompt["scene_catalog"] == [
        {
            "object_type": "Parcel", "count": 1,
            "affordances": {"pickupable": True},
        },
        {
            "object_type": "StoragePod", "count": 1,
            "affordances": {"receptacle": True},
        },
    ]
    grounding_schema = chat.prompt["output_schema"]["subtasks"][0]["grounding"]
    assert grounding_schema["source_selector"]["quantifier"] == "one/all"
    assert "exact object_type" in grounding_schema["source_selector"]["object_types"][0]
    assert any(
        "quantifier=all selector" in constraint
        for constraint in chat.prompt["constraints"]
    )
    assert any(
        "runtime binds and navigates" in constraint
        for constraint in chat.prompt["constraints"]
    )
    assert any("end-to-end executor interaction" in constraint for constraint in chat.prompt["constraints"])
    assert not any("add a find/inspect/search subtask before object interaction" in constraint
                   for constraint in chat.prompt["constraints"])

    place_tasks = [task for task in graph["flat_tasks"] if task["action"] == "place"]
    assert [task["grounding"]["object_tags"] for task in place_tasks] == [
        ["remotecontrol", "box"],
        ["keys", "box"],
        ["watch", "box"],
    ]
    assert {task["name"] for task in place_tasks} == {
        "Place remotecontrol in box",
        "Place keys in box",
        "Place watch in box",
    }
    assert {
        task["grounding"]["object_tags"][0]: task["depends_on"]
        for task in place_tasks
    } == {
        "remotecontrol": ["T1", "T4"],
        "keys": ["T2", "T4"],
        "watch": ["T3", "T4"],
    }

    diagnostics = graph["planner_diagnostics"]
    assert diagnostics["agent_context_provided"] is True
    assert diagnostics["atomic_place_repairs"] == [
        {
            "kind": "split_multi_object_place",
            "source_task_id": "T5",
            "task_ids": ["T5", "T6", "T7"],
            "placements": [
                {"source": "remotecontrol", "destination": "box"},
                {"source": "keys", "destination": "box"},
                {"source": "watch", "destination": "box"},
            ],
        }
    ]


def test_atomic_single_object_place_is_not_rewritten() -> None:
    class AtomicPlaceChat:
        def __call__(self, prompt: str) -> str:
            response = intent_response(
                json.loads(prompt),
                action="place",
                source_types={"Watch"},
                destination_types={"Box"},
            )
            if response is not None:
                return response
            response = critic_response(prompt)
            if response is not None:
                return response
            return json.dumps(
                {
                    "reasoning_summary": "Place one object.",
                    "subtasks": [
                        {
                            "id": "T1",
                            "name": "Place watch in box",
                            "description": "Pick up the watch and place it inside the box.",
                            "action": "place",
                            "grounding": {
                                "node_ids": [],
                                "source_selector": {"quantifier": "one", "object_types": ["Watch"]},
                                "destination_selector": {"quantifier": "one", "object_types": ["Box"]},
                                "object_tags": ["watch", "box"],
                                "relation_texts": ["watch is in box"],
                            },
                            "depends_on": [],
                            "termination_check": "watch is in box.",
                        }
                    ],
                }
            )

    task = "Put the watch in the box"
    subgraph = {
        **EMPTY_SUBGRAPH,
        "task": task,
        "task_spec": {
            **EMPTY_SUBGRAPH["task_spec"],
            "target_objects": ["watch", "box"],
        },
    }
    graph = decompose_task_to_graph(
        task,
        subgraph,
        qwen_chat=AtomicPlaceChat(),
        scene_catalog=[
            {"objectType": "Watch", "objectId": "Watch|1", "pickupable": True},
            {"objectType": "Box", "objectId": "Box|1", "receptacle": True},
        ],
    )

    assert len(graph["flat_tasks"]) == 1
    assert graph["flat_tasks"][0]["grounding"]["object_tags"] == ["watch", "box"]
    assert graph["planner_diagnostics"]["atomic_place_repairs"] == []



def test_all_selector_is_preserved_for_runtime_instance_expansion() -> None:
    class SelectorChat:
        def __call__(self, prompt: str) -> str:
            payload = json.loads(prompt)
            response = intent_response(
                payload,
                action="place",
                source_types={"Parcel", "Token"},
                destination_types={"StoragePod"},
                source_quantifier="all",
            )
            if response is not None:
                return response
            response = critic_response(prompt)
            if response is not None:
                return response
            return json.dumps({
                "reasoning_summary": "Use a quantified scene selector.",
                "subtasks": [{
                    "id": "T1", "name": "Place selected objects",
                    "description": "Place every selected source in the destination.",
                    "action": "place",
                    "grounding": {
                        "node_ids": [],
                        "object_tags": ["Parcel", "Token", "StoragePod"],
                        "source_object_tags": ["Parcel", "Token"],
                        "destination_object_tags": ["StoragePod"],
                        "source_selector": {
                            "quantifier": "all", "object_types": ["Parcel", "Token"],
                        },
                        "destination_selector": {
                            "quantifier": "one", "object_types": ["StoragePod"],
                        },
                        "relation_texts": [],
                    },
                    "depends_on": [], "termination_check": "All selected sources are placed.",
                }],
            })

    result = decompose_task_to_graph(
        "place every selected source", EMPTY_SUBGRAPH, qwen_chat=SelectorChat(),
        scene_catalog=[
            {"objectType": "Parcel", "objectId": "Parcel|1", "pickupable": True},
            {"objectType": "Token", "objectId": "Token|1", "pickupable": True},
            {"objectType": "StoragePod", "objectId": "StoragePod|1", "receptacle": True},
        ],
    )

    place_tasks = [task for task in result["flat_tasks"] if task["action"] == "place"]
    assert len(place_tasks) == 1
    assert place_tasks[0]["grounding"]["source_selector"] == {
        "quantifier": "all", "object_types": ["Parcel", "Token"],
    }
    assert result["planner_diagnostics"]["atomic_place_repairs"] == []


def test_selector_find_is_not_split_into_semantic_tokens() -> None:
    plan = {"subtasks": [{
        "id": "T0", "action": "find", "depends_on": [],
        "grounding": {
            "source_object_tags": ["Parcel", "Token"],
            "source_selector": {
                "quantifier": "all", "object_types": ["Parcel", "Token"],
            },
        },
    }]}

    repaired, diagnostics = split_multi_entity_find_subtasks(plan)

    assert repaired == plan
    assert diagnostics == []


def test_task6_compounds_and_aggregate_search_are_repaired_generically() -> None:
    task = "Put the vase, tissue box, and remote control on the table"
    spec = parse_task_locally(task)
    assert spec.target_objects == ["vase", "tissue box", "remote control"]
    assert spec.destination_receptacles == ["table"]
    subgraph = {"task_spec": spec.__dict__, "nodes": [], "triples": []}
    catalog = semantic_entity_catalog(subgraph)
    assert restore_compound_object_tags(["tissue", "box", "remote", "control"], catalog) == ["tissue box", "remote control"]
    assert restore_compound_object_tags(["invalid", "words"], catalog) == ["invalid", "words"]

    plan = {"subtasks": [
        {"id": "T0", "action": "find", "grounding": {"object_tags": ["tissue box", "remote control"]}, "depends_on": []},
        {"id": "T1", "action": "place", "grounding": {"source_object_tags": ["tissue box"], "destination_object_tags": ["table"]}, "depends_on": ["T0"]},
        {"id": "T2", "action": "place", "grounding": {"source_object_tags": ["remote control"], "destination_object_tags": ["table"]}, "depends_on": ["T0"]},
    ]}
    repaired, diagnostics = split_multi_entity_find_subtasks(plan)
    finds = [item for item in repaired["subtasks"] if item["action"] == "find"]
    assert [item["grounding"]["source_object_tags"][0] for item in finds] == ["tissue box", "remote control"]
    find_ids = {item["grounding"]["source_object_tags"][0]: item["id"] for item in finds}
    places = [item for item in repaired["subtasks"] if item["action"] == "place"]
    assert {item["grounding"]["source_object_tags"][0]: item["depends_on"] for item in places} == {key: [value] for key, value in find_ids.items()}


def test_rebuild_task_graph_views_uses_flat_tasks_as_canonical_source() -> None:
    provenance = {
        "origin_task_id": "T2",
        "instance_index": 1,
        "bound_source_object_id": "Parcel|01",
        "expanded_dependency_origins": ["T1"],
        "serialized_after_task_id": None,
    }
    graph = {
        "task": "place parcel",
        "planner_backend": "qwen_local",
        "flat_tasks": [
            {
                "id": "T2", "action": "place", "depends_on": ["T1"],
                "runtime": {"selector_expansion": provenance},
            },
            {"id": "T1", "action": "find", "depends_on": []},
        ],
        "dependency_edges": [{"from": "stale", "to": "T1"}],
        "root_task_ids": ["stale"],
        "chains": [{"chain_id": "stale", "task_ids": ["stale"]}],
        "planner_diagnostics": {"marker": "preserved"},
    }

    rebuilt = rebuild_task_graph_views(graph)

    assert [task["id"] for task in rebuilt["flat_tasks"]] == ["T1", "T2"]
    assert rebuilt["dependency_edges"] == [{"from": "T1", "to": "T2"}]
    assert rebuilt["root_task_ids"] == ["T1"]
    assert len(rebuilt["chains"]) == 1
    assert rebuilt["chains"][0]["task_ids"] == ["T1", "T2"]
    assert rebuilt["chains"][0]["linked_list"]["id"] == "T1"
    assert rebuilt["chains"][0]["linked_list"]["chain_id"] == "C1"
    assert rebuilt["chains"][0]["linked_list"]["next"]["id"] == "T2"
    assert rebuilt["chains"][0]["linked_list"]["next"]["chain_id"] == "C1"
    assert rebuilt["flat_tasks"][1]["runtime"]["selector_expansion"] == provenance
    assert rebuilt["planner_diagnostics"] == {"marker": "preserved"}
    assert graph["dependency_edges"] == [{"from": "stale", "to": "T1"}]
