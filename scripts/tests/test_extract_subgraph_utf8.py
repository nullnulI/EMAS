from __future__ import annotations

import builtins
import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture
def extract_module() -> ModuleType:
    return importlib.import_module("planning.extract_subgraph")


def write_utf8_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def force_ascii_default_for(
    monkeypatch: pytest.MonkeyPatch,
    paths: set[Path],
) -> None:
    real_open = builtins.open
    resolved_paths = {path.resolve() for path in paths}

    def ascii_default_open(
        file: Any,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
        closefd: bool = True,
        opener: Any = None,
    ) -> Any:
        try:
            is_target = Path(file).resolve() in resolved_paths
        except (TypeError, ValueError):
            is_target = False
        if is_target and "b" not in mode and encoding is None:
            encoding = "ascii"
        return real_open(
            file,
            mode,
            buffering,
            encoding,
            errors,
            newline,
            closefd,
            opener,
        )

    monkeypatch.setattr(builtins, "open", ascii_default_open)


def test_json_readers_use_utf8_when_default_encoding_is_ascii(
    extract_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene_graph_path = tmp_path / "scene_graph.json"
    response_path = tmp_path / "responses" / "0.json"
    relations_path = tmp_path / "relations.json"

    write_utf8_json(
        scene_graph_path,
        [
            {
                "id": 36,
                "pruned_id": 0,
                "original_id": 36,
                "object_tag": "espresso machine",
                "caption": "machine in a café",
                "possible_tags": ["coffee machine", "café"],
            }
        ],
    )
    write_utf8_json(
        response_path,
        {
            "id": 36,
            "response": {
                "object_tag": "espresso machine",
                "summary": "machine in a café",
                "possible_tags": ["café"],
            },
        },
    )
    write_utf8_json(
        relations_path,
        [
            {
                "object1": {"id": 36},
                "object2": {"id": 37},
                "object_relation": "a in b",
                "description": "café counter",
            }
        ],
    )

    force_ascii_default_for(
        monkeypatch,
        {scene_graph_path, response_path, relations_path},
    )

    scene_nodes = extract_module.load_nodes_from_scene_graph_json(scene_graph_path)
    response_nodes = extract_module.load_nodes_from_response_dir(response_path.parent)
    relation_nodes = {
        0: {"pruned_id": 0, "original_id": 36, "id": 0},
        1: {"pruned_id": 1, "original_id": 37, "id": 1},
    }
    edges = extract_module.load_edges_from_relations_json(
        relations_path,
        relation_nodes,
    )

    assert scene_nodes[0]["possible_tags"] == ["coffee machine", "café"]
    assert response_nodes[0]["caption"] == "machine in a café"
    assert edges[0]["description"] == "café counter"
    assert edges[0]["normalized_relation"] == "in"


def test_scene_graph_reader_rejects_invalid_utf8(
    extract_module: ModuleType,
    tmp_path: Path,
) -> None:
    scene_graph_path = tmp_path / "invalid_scene_graph.json"
    scene_graph_path.write_bytes(b'[{"object_tag": "caf\xe9"}]')

    with pytest.raises(UnicodeDecodeError):
        extract_module.load_nodes_from_scene_graph_json(scene_graph_path)
