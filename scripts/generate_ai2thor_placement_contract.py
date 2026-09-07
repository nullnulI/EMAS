#!/usr/bin/env python3
"""Generate the Python placement contract snapshot from bundled Unity source."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "ai2thor" / "unity" / "Assets" / "Scripts" / "SimObjType.cs"


def parse_placement_restrictions(source: str) -> dict[str, list[str]]:
    marker = "PlacementRestrictions = new Dictionary"
    start = source.find(marker)
    if start < 0:
        raise ValueError("PlacementRestrictions dictionary was not found")
    section = source[start:]
    entry_pattern = re.compile(
        r"\{\s*SimObjType\.(\w+)\s*,\s*new\s+List<SimObjType>\(\)\s*"
        r"\{(.*?)\}\s*\}\s*,?",
        flags=re.DOTALL,
    )
    restrictions: dict[str, list[str]] = {}
    for source_type, body in entry_pattern.findall(section):
        destinations = re.findall(r"SimObjType\.(\w+)", body)
        restrictions[source_type] = list(dict.fromkeys(destinations))
    if not restrictions:
        raise ValueError("PlacementRestrictions dictionary contained no entries")
    return restrictions


def generated_payload(source_path: Path) -> dict[str, object]:
    restrictions = parse_placement_restrictions(source_path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "source": "ai2thor/unity/Assets/Scripts/SimObjType.cs::ReceptacleRestrictions.PlacementRestrictions",
        "placement_restrictions": restrictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    print(json.dumps(generated_payload(args.source), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
