from __future__ import annotations

import json
from pathlib import Path

from memory.graph_update import update_scenegraph_files


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_incremental_merge_appends_new_nodes_and_patches_existing_properties(tmp_path: Path) -> None:
    stored_nodes = tmp_path / "stored_nodes.json"
    stored_relations = tmp_path / "stored_relations.json"
    observed_nodes = tmp_path / "observed_nodes.json"
    write_json(
        stored_nodes,
        [
            {
                "id": 0,
                "original_id": 0,
                "pruned_id": 0,
                "objectId": "Cabinet|1",
                "object_tag": "Cabinet",
                "caption": "visual caption that must survive",
                "isOpen": False,
            }
        ],
    )
    write_json(stored_relations, [])
    write_json(
        observed_nodes,
        [
            {"id": 0, "objectId": "Cabinet|1", "object_tag": "Cabinet"},
            {"id": 1, "objectId": "Mug|1", "object_tag": "Mug"},
        ],
    )

    result = update_scenegraph_files(
        stored_scenegraph_path=stored_nodes,
        stored_relations_path=stored_relations,
        observed_scenegraph_path=observed_nodes,
        object_deltas=[
            {
                "object_id": "Cabinet|1",
                "object_type": "Cabinet",
                "changed_fields": {"isOpen": {"before": False, "after": True}},
            },
            {
                "object_id": "Mug|1",
                "object_type": "Mug",
                "changed_fields": {
                    "position": {"before": None, "after": {"x": 1, "y": 2, "z": 3}}
                },
            },
        ],
        relation_deltas=[
            {
                "op": "add",
                "object1_id": "Mug|1",
                "object2_id": "Cabinet|1",
                "object_relation": "a in b",
            }
        ],
        output_dir=tmp_path / "updated",
        append_observed_only=True,
    )

    nodes = json.loads(Path(result["scene_graph"]).read_text(encoding="utf-8"))
    relations = json.loads(Path(result["relations"]).read_text(encoding="utf-8"))
    cabinet = next(node for node in nodes if node["objectId"] == "Cabinet|1")
    mug = next(node for node in nodes if node["objectId"] == "Mug|1")

    assert len(nodes) == 2
    assert cabinet["caption"] == "visual caption that must survive"
    assert cabinet["isOpen"] is True
    assert mug["bbox_center"] == [1.0, 2.0, 3.0]
    assert relations == [
        {
            "object1": {"id": mug["original_id"]},
            "object2": {"id": cabinet["original_id"]},
            "object_relation": "a in b",
        }
    ]
