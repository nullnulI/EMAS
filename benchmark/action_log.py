from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


def _events(event: Any) -> list[Any]:
    child_events = getattr(event, "events", None)
    return list(child_events) if child_events else [event]


def _agent_metadata(event: Any, agent_id: int) -> dict[str, Any]:
    events = _events(event)
    if not events:
        return {}
    index = min(max(int(agent_id), 0), len(events) - 1)
    return dict(getattr(events[index], "metadata", {}) or {})


class ObjectNameIndex:
    """Stable raw AI2-THOR object id -> MAP-THOR readable name mapping."""

    def __init__(self, objects: list[dict[str, Any]]) -> None:
        counters: dict[str, int] = defaultdict(int)
        self.raw_to_readable: dict[str, str] = {}
        for obj in objects:
            object_id = str(obj.get("objectId") or "")
            object_type = str(obj.get("objectType") or object_id.split("|", 1)[0])
            if not object_id or not object_type:
                continue
            counters[object_type] += 1
            self.raw_to_readable[object_id] = f"{object_type}_{counters[object_type]}"

    def readable(self, object_id: str | None) -> str:
        if not object_id:
            return "nothing"
        raw = str(object_id)
        return self.raw_to_readable.get(raw, f"{raw.split('|', 1)[0]}_1")


ACTION_NAMES = {
    "MoveAhead": "Move(Ahead)",
    "MoveBack": "Move(Back)",
    "MoveLeft": "Move(Left)",
    "MoveRight": "Move(Right)",
    "RotateLeft": "Rotate(Left)",
    "RotateRight": "Rotate(Right)",
}


class BenchmarkActionLogger:
    def __init__(
        self,
        path: Path,
        checker: Any,
        initial_objects: list[dict[str, Any]],
        event_observer: Callable[[Any, dict[str, Any] | None], None] | None = None,
    ) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self.checker = checker
        self.names = ObjectNameIndex(initial_objects)
        self.records: list[dict[str, Any]] = []
        self.inventory_by_agent: dict[int, str] = defaultdict(lambda: "nothing")
        self.progress_by_agent: dict[int, int] = defaultdict(int)
        self.current_macro_step = 0
        self.current_subtask_id: str | None = None
        self.current_agent_id = 0
        self.event_observer = event_observer

    @staticmethod
    def _completed_count(checker: Any) -> int:
        completed = getattr(checker, "subtasks_completed_numerated", None)
        if completed is None:
            completed = getattr(checker, "subtasks_completed", [])
        return len(completed or [])

    def bind(self, *, macro_step: int, subtask_id: str, agent_id: int) -> None:
        self.current_macro_step = int(macro_step)
        self.current_subtask_id = str(subtask_id)
        self.current_agent_id = int(agent_id)

    def _official_pickup_action(self, readable_object: str) -> str:
        required = [str(item) for item in getattr(self.checker, "subtasks", []) or []]
        pickup_action = f"PickupObject({readable_object})"
        pick_up_action = f"PickUpObject({readable_object})"
        pick_action = f"PickObject({readable_object})"

        # Prefer an object-specific checker declaration. Some upstream MAP-THOR
        # checkers use PickupObject, PickUpObject, or PickObject for the same
        # AI2-THOR action.
        if pickup_action in required:
            return pickup_action
        if pick_up_action in required:
            return pick_up_action
        if pick_action in required:
            return pick_action
        if any(item.startswith("PickupObject(") for item in required):
            return pickup_action
        if any(item.startswith("PickUpObject(") for item in required):
            return pick_up_action
        if any(item.startswith("PickObject(") for item in required):
            return pick_action
        return pick_action

    def _official_action(self, action: str, params: dict[str, Any]) -> str:
        object_id = params.get("objectId")
        if action == "PickupObject" and object_id:
            return self._official_pickup_action(self.names.readable(str(object_id)))
        if action in ACTION_NAMES:
            return ACTION_NAMES[action]
        if object_id:
            return f"{action}({self.names.readable(str(object_id))})"
        if action in {"LookUp", "LookDown"}:
            return f"{action}({params.get('degrees', 30)})"
        return action if "(" in action else f"{action}()"

    def __call__(self, event: Any, result: dict[str, Any]) -> None:
        if self.event_observer is not None:
            try:
                self.event_observer(event, result)
            except Exception:
                pass
        raw_agent_id = result.get("agent_id", result.get("robot_id", self.current_agent_id))
        try:
            agent_id = int(raw_agent_id)
        except (TypeError, ValueError):
            agent_id = self.current_agent_id
        action = str(result.get("action") or "")
        params = dict(result.get("params") or {})
        success = bool(result.get("lastActionSuccess"))
        official_action = self._official_action(action, params)
        metric_inventory = self.inventory_by_agent[agent_id]
        metadata = _agent_metadata(event, agent_id)

        if success and action == "PickupObject":
            inventory = metadata.get("inventoryObjects") or []
            if inventory:
                metric_inventory = self.names.readable(inventory[0].get("objectId"))
                self.inventory_by_agent[agent_id] = metric_inventory

        completed_before = self._completed_count(self.checker)
        metric_error = None
        try:
            self.checker.perform_metric_check(official_action, success, metric_inventory)
        except Exception as exc:  # Preserve the episode when an upstream checker is malformed.
            metric_error = repr(exc)
        completed_after = self._completed_count(self.checker)
        progress_credit = max(0, completed_after - completed_before)
        self.progress_by_agent[agent_id] += progress_credit

        record = {
            "macro_step": self.current_macro_step,
            "low_level_step": len(self.records),
            "subtask_id": self.current_subtask_id,
            "agent_id": agent_id,
            "action": action,
            "official_action": official_action,
            "params": params,
            "success": success,
            "error_message": result.get("errorMessage"),
            "inventory_for_metric": metric_inventory,
            "inventory_after": [
                self.names.readable(item.get("objectId"))
                for item in metadata.get("inventoryObjects") or []
            ],
            "progress_credit": progress_credit,
            "collided": bool(result.get("collided")),
            "collided_objects": list(result.get("collided_objects") or []),
            "wall_time_seconds": float(result.get("wall_time_seconds") or 0.0),
            "timestamp": time.time(),
            "metric_error": metric_error,
        }
        self.records.append(record)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        if success and action in {"PutObject", "DropHandObject", "ThrowObject"}:
            self.inventory_by_agent[agent_id] = "nothing"

    def balance(self) -> float:
        contributing = [count for count in self.progress_by_agent.values() if count > 0]
        if not contributing:
            return 0.0
        if len(contributing) == 1:
            return 1.0
        return min(contributing) / max(contributing)
