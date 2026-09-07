"""Small shared schemas for online scene graph updates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

RelationOp = Literal["add", "remove"]


@dataclass(slots=True)
class ObjectDelta:
    """A compact pre/post change for one AI2-THOR object."""

    object_id: str
    object_type: str | None = None
    changed_fields: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RelationDelta:
    """A spatial relation change inferred from execution metadata."""

    op: RelationOp
    object1_id: str
    object2_id: str
    object_relation: str
    reason: str | None = None
    confidence: float = 1.0
    source: str = "execution_metadata"

    def to_record(self) -> dict[str, Any]:
        record = {
            "object1": {"id": self.object1_id},
            "object2": {"id": self.object2_id},
            "object_relation": self.object_relation,
        }
        if self.reason:
            record["reason"] = self.reason
        return record

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
