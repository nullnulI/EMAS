from __future__ import annotations

import json
from pathlib import Path

from placement_contracts import (
    CONTRACT_PATH,
    compatible_receptacles,
    placement_compatibility,
)
from planning.scene_goal_compiler import expand_scene_catalog_task_graph
from scripts.generate_ai2thor_placement_contract import generated_payload


ROOT = Path(__file__).resolve().parents[2]
UNITY_SOURCE = ROOT / "ai2thor" / "unity" / "Assets" / "Scripts" / "SimObjType.cs"


def _place_graph(source_type: str, destination_type: str) -> dict:
    return {
        "task": "place an object",
        "flat_tasks": [{
            "id": "T1",
            "action": "place",
            "depends_on": [],
            "grounding": {
                "source_selector": {
                    "quantifier": "all",
                    "object_types": [source_type],
                },
                "destination_selector": {
                    "quantifier": "one",
                    "object_types": [destination_type],
                },
            },
        }],
    }


def _catalogue(source_type: str, destination_type: str) -> list[dict]:
    return [
        {
            "objectType": source_type,
            "objectId": f"{source_type}|1",
            "pickupable": True,
        },
        {
            "objectType": destination_type,
            "objectId": f"{destination_type}|1",
            "receptacle": True,
        },
    ]


def test_checked_in_contract_matches_bundled_unity_source() -> None:
    checked_in = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert checked_in == generated_payload(UNITY_SOURCE)


def test_known_ai2thor_placement_pairs_include_mug_cabinet_and_apple_fridge() -> None:
    assert placement_compatibility("Mug", "Cabinet") is True
    assert placement_compatibility("Apple", "Fridge") is True
    assert "Cabinet" in (compatible_receptacles("Mug") or ())


def test_selector_compiler_rejects_incompatible_known_pair_structurally() -> None:
    compiled = expand_scene_catalog_task_graph(
        _place_graph("Apple", "Drawer"),
        _catalogue("Apple", "Drawer"),
    )
    diagnostics = compiled["planner_diagnostics"]["selector_expansion"]
    assert diagnostics["status"] == "rejected"
    assert diagnostics["violations"][0]["code"] == "incompatible_receptacle"
    assert diagnostics["violations"][0]["invalid_values"][0]["source_type"] == "Apple"


def test_selector_compiler_preserves_authoritative_explicit_destination() -> None:
    graph = _place_graph("SaltShaker", "Fridge")
    graph["planner_diagnostics"] = {
        "root_intent_validation": {
            "root_intent": {
                "action": "place",
                "roles": {"source": ["SaltShaker"], "destination": ["Fridge"]},
            },
        },
    }

    compiled = expand_scene_catalog_task_graph(
        graph,
        _catalogue("SaltShaker", "Fridge"),
    )

    diagnostics = compiled["planner_diagnostics"]["selector_expansion"]
    assert diagnostics["status"] == "expanded"
    assert any(
        event.get("status") == "authoritative_destination_override"
        for event in diagnostics["events"]
    )


def test_selector_compiler_accepts_known_compatible_pairs() -> None:
    for source_type, destination_type in (("Mug", "Cabinet"), ("Apple", "Fridge")):
        compiled = expand_scene_catalog_task_graph(
            _place_graph(source_type, destination_type),
            _catalogue(source_type, destination_type),
        )
        assert compiled["planner_diagnostics"]["selector_expansion"]["status"] == "expanded"


def test_unknown_custom_types_remain_extensible() -> None:
    assert placement_compatibility("Parcel", "StoragePod") is None
    compiled = expand_scene_catalog_task_graph(
        _place_graph("Parcel", "StoragePod"),
        _catalogue("Parcel", "StoragePod"),
    )
    assert compiled["planner_diagnostics"]["selector_expansion"]["status"] == "expanded"

