from __future__ import annotations

from copy import deepcopy

from planning.scene_goal_compiler import (
    expand_scene_catalog_task_graph,
    summarize_scene_catalogue,
)


def obj(object_type: str, index: int, **affordances: object) -> dict[str, object]:
    return {"objectType": object_type, "objectId": f"{object_type}|{index:02d}", **affordances}


def graph(
    action: str,
    source_types: list[str],
    *,
    source_quantifier: str = "all",
    destination_types: list[str] | None = None,
) -> dict[str, object]:
    grounding: dict[str, object] = {
        "source_selector": {"quantifier": source_quantifier, "object_types": source_types},
        "source_object_tags": source_types,
        "destination_object_tags": destination_types or [],
        "source_object_ids": [],
        "destination_object_ids": [],
    }
    if destination_types is not None:
        grounding["destination_selector"] = {
            "quantifier": "one", "object_types": destination_types,
        }
    return {
        "task": "public instruction",
        "planner_backend": "qwen_local",
        "flat_tasks": [{
            "id": "T1", "name": "planned task", "description": "planned task",
            "action": action, "grounding": grounding, "depends_on": [],
        }],
        "planner_diagnostics": {"qwen": {"status": "success"}},
    }


def test_scene_catalogue_summary_includes_parent_type_and_position_without_ids() -> None:
    catalogue = [
        obj(
            "Apple", 1, pickupable=True,
            parentReceptacles=["CounterTop|01"],
        ),
        obj(
            "CounterTop", 1, receptacle=True,
            position={"x": -0.08, "y": 1.15, "z": 0.0},
            receptacleObjectIds=["Apple|01"],
        ),
        obj(
            "CounterTop", 2, receptacle=True,
            position={"x": -2.0, "y": 0.95, "z": -1.2},
        ),
        obj(
            "CounterTop", 3, receptacle=True,
            position={"x": 1.0, "y": 0.95, "z": -2.5},
        ),
    ]

    summary = summarize_scene_catalogue(catalogue)

    apple = next(item for item in summary if item["object_type"] == "Apple")
    assert apple["parent_locations"] == [{
        "parent_type": "CounterTop",
        "parent_position": {"x": -0.08, "y": 1.15, "z": 0.0},
        "instance_count": 1,
        "relative_location": "central",
    }]
    assert "objectId" not in apple


def test_any_openable_type_expands_all_instances_without_type_rules() -> None:
    planner = graph("open", ["HatchPanel"])
    catalogue = [obj("HatchPanel", index, openable=True) for index in range(1, 4)]

    expanded = expand_scene_catalog_task_graph(planner, catalogue)

    assert expanded["planner_backend"] == "qwen_local"
    assert expanded["planner_diagnostics"]["selector_expansion"]["status"] == "expanded"
    assert [
        task["grounding"]["source_object_ids"][0]
        for task in expanded["flat_tasks"]
    ] == ["HatchPanel|01", "HatchPanel|02", "HatchPanel|03"]


def test_any_toggleable_type_expands_all_instances() -> None:
    planner = graph("toggle_off", ["SignalBeacon"])
    catalogue = [obj("SignalBeacon", index, toggleable=True) for index in range(1, 3)]

    expanded = expand_scene_catalog_task_graph(planner, catalogue)

    assert len(expanded["flat_tasks"]) == 2
    assert all(task["action"] == "toggle_off" for task in expanded["flat_tasks"])


def test_model_selected_type_union_expands_without_semantic_group_table() -> None:
    planner = graph("place", ["Parcel", "Token"], destination_types=["StoragePod"])
    catalogue = [
        obj("Parcel", 1, pickupable=True),
        obj("Parcel", 2, pickupable=True),
        obj("Token", 1, pickupable=True),
        obj("StoragePod", 1, receptacle=True),
    ]

    expanded = expand_scene_catalog_task_graph(planner, catalogue)

    tasks = expanded["flat_tasks"]
    assert len(tasks) == 3
    assert {task["grounding"]["source_object_ids"][0] for task in tasks} == {
        "Parcel|01", "Parcel|02", "Token|01",
    }
    assert {task["grounding"]["destination_object_ids"][0] for task in tasks} == {
        "StoragePod|01"
    }


def test_place_rejects_multiple_destination_object_types() -> None:
    planner = graph(
        "place", ["Parcel"],
        source_quantifier="one",
        destination_types=["StoragePod", "StorageShelf"],
    )
    catalogue = [
        obj("Parcel", 1, pickupable=True),
        obj("StoragePod", 1, receptacle=True),
        obj("StorageShelf", 1, receptacle=True),
    ]

    expanded = expand_scene_catalog_task_graph(planner, catalogue)

    diagnostics = expanded["planner_diagnostics"]["selector_expansion"]
    assert diagnostics["status"] == "rejected"
    assert "exactly one object_type" in diagnostics["reason"]


def test_place_instances_already_in_destination_are_marked_satisfied() -> None:
    planner = graph("place", ["CargoItem"], destination_types=["StorageBin"])
    catalogue = [
        obj("CargoItem", 1, pickupable=True, parentReceptacles=["StorageBin|01"]),
        obj("CargoItem", 2, pickupable=True),
        obj("CargoItem", 3, pickupable=True, parentReceptacles=["OtherBin|01"]),
        obj("StorageBin", 1, receptacle=True, receptacleObjectIds=["CargoItem|02"]),
    ]

    expanded = expand_scene_catalog_task_graph(planner, catalogue)

    tasks = expanded["flat_tasks"]
    assert len(tasks) == 3
    satisfied = [
        task for task in tasks
        if (task.get("runtime") or {}).get("already_satisfied")
    ]
    assert [task["grounding"]["source_object_ids"][0] for task in satisfied] == [
        "CargoItem|01", "CargoItem|02",
    ]
    assert all(task["grounding"]["status"] == "satisfied" for task in satisfied)
    assert not (tasks[2].get("runtime") or {}).get("already_satisfied")
    assert expanded["planner_diagnostics"]["selector_expansion"]["events"][0][
        "already_satisfied_task_ids"
    ] == ["T1", "T1__instance_002"]


def test_non_place_actions_ignore_containment_state() -> None:
    planner = graph("open", ["StorageBin"])
    catalogue = [
        obj("CargoItem", 1, pickupable=True, parentReceptacles=["StorageBin|01"]),
        obj("StorageBin", 1, openable=True, receptacleObjectIds=["CargoItem|01"]),
    ]

    expanded = expand_scene_catalog_task_graph(planner, catalogue)

    task = expanded["flat_tasks"][0]
    assert task["grounding"]["status"] == "grounded"
    assert not (task.get("runtime") or {}).get("already_satisfied")


def test_openable_destination_serializes_expanded_place_tasks() -> None:
    planner = graph("place", ["Parcel"], destination_types=["StoragePod"])
    planner["flat_tasks"].insert(0, {
        "id": "T0", "action": "find", "grounding": {}, "depends_on": [],
    })
    planner["flat_tasks"][1]["depends_on"] = ["T0"]
    catalogue = [
        obj("Parcel", 1, pickupable=True),
        obj("Parcel", 2, pickupable=True),
        obj("StoragePod", 1, receptacle=True, openable=True),
    ]

    tasks = [
        task
        for task in expand_scene_catalog_task_graph(planner, catalogue)["flat_tasks"]
        if task["action"] == "place"
    ]

    assert tasks[0]["depends_on"] == ["T0"]
    assert tasks[1]["depends_on"] == [tasks[0]["id"]]
    assert tasks[0]["runtime"]["selector_expansion"] == {
        "origin_task_id": "T1",
        "instance_index": 1,
        "bound_source_object_id": "Parcel|01",
        "expanded_dependency_origins": ["T0"],
        "serialized_after_task_id": None,
    }
    assert tasks[1]["runtime"]["selector_expansion"] == {
        "origin_task_id": "T1",
        "instance_index": 2,
        "bound_source_object_id": "Parcel|02",
        "expanded_dependency_origins": ["T0"],
        "serialized_after_task_id": tasks[0]["id"],
    }


def test_one_quantifier_only_binds_a_unique_instance() -> None:
    planner = graph("open", ["HatchPanel"], source_quantifier="one")
    unique = expand_scene_catalog_task_graph(
        planner, [obj("HatchPanel", 1, openable=True)]
    )
    ambiguous = expand_scene_catalog_task_graph(
        planner, [obj("HatchPanel", 1, openable=True), obj("HatchPanel", 2, openable=True)]
    )

    assert unique["flat_tasks"][0]["grounding"]["source_object_ids"] == ["HatchPanel|01"]
    assert ambiguous["flat_tasks"][0]["grounding"]["source_object_ids"] == []
    assert ambiguous["flat_tasks"][0]["grounding"]["status"] == "partial"


def test_invalid_type_or_affordance_preserves_original_graph_atomically() -> None:
    planner = graph("toggle_on", ["SignalBeacon"])
    for catalogue in (
        [obj("OtherType", 1, toggleable=True)],
        [obj("SignalBeacon", 1, toggleable=False)],
    ):
        original = deepcopy(planner)
        output = expand_scene_catalog_task_graph(planner, catalogue)
        assert output["flat_tasks"] == original["flat_tasks"]
        assert output["planner_diagnostics"]["selector_expansion"]["status"] == "rejected"


def test_required_interaction_selector_rejects_missing_catalogue() -> None:
    planner = graph("open", ["HatchPanel"])

    output = expand_scene_catalog_task_graph(planner, None)

    diagnostics = output["planner_diagnostics"]["selector_expansion"]
    assert diagnostics["status"] == "rejected"
    assert diagnostics["reason"] == "scene catalogue is unavailable"


def test_missing_selector_is_a_diagnostic_noop() -> None:
    planner = {
        "planner_backend": "qwen_local",
        "flat_tasks": [{"id": "T1", "action": "find", "grounding": {}, "depends_on": []}],
    }

    output = expand_scene_catalog_task_graph(planner, [obj("Widget", 1)])

    assert output["flat_tasks"] == planner["flat_tasks"]
    assert output["planner_diagnostics"]["selector_expansion"]["status"] == "not_applicable"


def test_selector_bound_find_is_elided_without_blocking_interaction_expansion() -> None:
    planner = {
        "planner_backend": "qwen_local",
        "flat_tasks": [
            {
                "id": "T0",
                "action": "find",
                "grounding": {
                    "source_selector": {
                        "quantifier": "all",
                        "object_types": ["HatchPanel"],
                    },
                    "source_object_tags": ["HatchPanel"],
                },
                "depends_on": [],
            },
            {
                "id": "T1",
                "action": "open",
                "grounding": {
                    "source_selector": {
                        "quantifier": "all",
                        "object_types": ["HatchPanel"],
                    },
                    "source_object_tags": ["HatchPanel"],
                },
                "depends_on": ["T0"],
            },
        ],
    }
    catalogue = [obj("HatchPanel", index, openable=True) for index in range(1, 3)]

    output = expand_scene_catalog_task_graph(planner, catalogue)

    assert output["planner_diagnostics"]["selector_expansion"]["status"] == "expanded"
    assert [task["id"] for task in output["flat_tasks"]] == [
        "T1", "T1__instance_002",
    ]
    assert all(task["depends_on"] == [] for task in output["flat_tasks"])
    assert output["planner_diagnostics"]["selector_expansion"]["events"][0][
        "status"
    ] == "elided_selector_bound_discovery"


def test_unrelated_selector_find_is_preserved() -> None:
    planner = {
        "planner_backend": "qwen_local",
        "flat_tasks": [
            {"id": "T0", "action": "find", "depends_on": [], "grounding": {
                "source_selector": {"quantifier": "one", "object_types": ["Locator"]},
                "source_object_tags": ["Locator"],
            }},
            {"id": "T1", "action": "open", "depends_on": [], "grounding": {
                "source_selector": {"quantifier": "all", "object_types": ["HatchPanel"]},
                "source_object_tags": ["HatchPanel"],
            }},
        ],
    }
    catalogue = [obj("Locator", 1), obj("HatchPanel", 1, openable=True)]

    output = expand_scene_catalog_task_graph(planner, catalogue)

    assert [task["id"] for task in output["flat_tasks"]] == ["T0", "T1"]
    assert output["planner_diagnostics"]["selector_expansion"]["events"][0]["status"] == "ignored_non_interaction_selector"


def test_catalogue_summary_exposes_counts_and_affordances_without_ids() -> None:
    summary = summarize_scene_catalogue([
        obj("HatchPanel", 1, openable=True),
        obj("HatchPanel", 2, openable=True),
    ])

    assert summary == [{
        "object_type": "HatchPanel",
        "count": 2,
            "affordances": {
                "pickupable": False, "moveable": False, "openable": True,
                "toggleable": False, "receptacle": False, "sliceable": False,
                "dirtyable": False, "breakable": False, "cookable": False,
                "canFillWithLiquid": False,
            },
    }]
    assert "objectId" not in str(summary)
