from __future__ import annotations

from agents.ai2thor_receiver_server import NativeControllerThorServer


def test_receiver_preserves_generic_affordance_and_state_fields() -> None:
    metadata = {
        "objects": [{
            "objectId": "SignalBeacon|1", "objectType": "SignalBeacon",
            "visible": True, "distance": 0.5,
            "toggleable": True, "isToggled": True,
            "sliceable": True, "isSliced": False,
            "dirtyable": True, "isDirty": True,
            "breakable": True, "isBroken": False,
            "cookable": True, "isCooked": False,
            "canFillWithLiquid": True, "isFilledWithLiquid": False,
            "fillLiquid": None,
            "name": "SignalBeacon_1",
            "rotation": {"x": 0.0, "y": 90.0, "z": 0.0},
            "axisAlignedBoundingBox": {"center": {"x": 1.0, "y": 1.0, "z": 1.0}},
            "isPickedUp": False,
            "parentReceptacles": ["Drawer|1"],
            "receptacleObjectIds": None,
            "openness": 0.0,
            "isMoving": False,
        }]
    }

    objects = NativeControllerThorServer._objects_from_metadata(metadata)

    assert len(objects) == 1
    item = objects[0]
    for key, value in metadata["objects"][0].items():
        if key not in {"objectId", "objectType"}:
            assert item[key] == value
    assert item["objectId"] == "SignalBeacon|1"
    assert item["objectType"] == "SignalBeacon"
