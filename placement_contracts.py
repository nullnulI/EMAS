"""AI2-THOR object-to-receptacle placement execution contracts.

The checked-in JSON is generated from the exact Unity source bundled with this
repository.  These rules describe controller executability only; they do not
encode which destination is semantically appropriate for a benchmark task.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


CONTRACT_PATH = Path(__file__).resolve().parent / "planning" / "data" / "ai2thor_placement_restrictions.json"


def normalize_object_type(value: Any) -> str:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(value or "").replace("_", " "))
    return "".join(re.findall(r"[a-z0-9]+", separated.lower()))


@lru_cache(maxsize=1)
def placement_contract() -> dict[str, tuple[str, ...]]:
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    restrictions = payload.get("placement_restrictions")
    if not isinstance(restrictions, dict):
        raise ValueError(f"invalid placement contract at {CONTRACT_PATH}")
    return {
        str(source): tuple(str(destination) for destination in destinations)
        for source, destinations in restrictions.items()
        if isinstance(destinations, list)
    }


@lru_cache(maxsize=1)
def _normalized_contract() -> dict[str, frozenset[str]]:
    return {
        normalize_object_type(source): frozenset(normalize_object_type(item) for item in destinations)
        for source, destinations in placement_contract().items()
    }


def compatible_receptacles(source_type: Any) -> tuple[str, ...] | None:
    """Return known compatible receptacle types, or ``None`` for an unknown type."""

    wanted = normalize_object_type(source_type)
    for source, destinations in placement_contract().items():
        if normalize_object_type(source) == wanted:
            return destinations
    return None


def placement_compatibility(source_type: Any, destination_type: Any) -> bool | None:
    """Return True/False for known source types and None for unknown source types."""

    allowed = _normalized_contract().get(normalize_object_type(source_type))
    if allowed is None:
        return None
    return normalize_object_type(destination_type) in allowed

