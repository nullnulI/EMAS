from __future__ import annotations

from typing import Any


SUCCESS = "success"
WAIT_RETRY = "wait_retry"
FAILURE = "failure"


RECOVERABLE_ERROR_HINTS = (
    "not visible",
    "too far",
    "too close",
    "obstructed",
    "blocked",
    "another object",
    "cannot see",
    "failed to open",
    "failed to close",
)

NON_RECOVERABLE_ERROR_HINTS = (
    "does not exist",
    "invalid action",
    "not pickupable",
    "not openable",
    "not receptacle",
    "not toggleable",
)


def _objects_from_agent_state(agent_state: dict[str, Any]) -> list[dict[str, Any]]:
    return list(agent_state.get("visible_objects") or [])


def _inventory_ids(agent_state: dict[str, Any]) -> set[str]:
    items = agent_state.get("inventoryObjects") or agent_state.get("inventory") or []
    ids = set()
    for item in items:
        if isinstance(item, dict) and item.get("objectId"):
            ids.add(str(item["objectId"]))
        elif item:
            ids.add(str(item))
    return ids


def _match_object_by_grounding(
    subtask: dict[str, Any],
    objects: list[dict[str, Any]],
) -> dict[str, Any] | None:
    grounding = subtask.get("grounding") or {}
    node_tags = {str(x).lower() for x in grounding.get("object_tags") or []}
    text_terms = " ".join(
        [
            str(subtask.get("name") or ""),
            str(subtask.get("description") or ""),
            " ".join(str(x) for x in grounding.get("relation_texts") or []),
        ]
    ).lower()

    best_obj = None
    best_score = -1
    for obj in objects:
        text = " ".join(
            [
                str(obj.get("objectId") or ""),
                str(obj.get("objectType") or ""),
                str(obj.get("name") or ""),
            ]
        ).lower()
        score = 0
        for tag in node_tags:
            if tag and tag in text:
                score += 2
        for token in text_terms.split():
            if len(token) > 2 and token in text:
                score += 1
        if score > best_score:
            best_score = score
            best_obj = obj
    return best_obj if best_score > 0 else None


def _grounding_is_unresolved(subtask: dict[str, Any]) -> bool:
    grounding = subtask.get("grounding") or {}
    return str(grounding.get("status") or "").lower() == "unresolved"


def _verify_success(
    subtask: dict[str, Any],
    pre_agent_state: dict[str, Any],
    post_agent_state: dict[str, Any],
) -> tuple[bool, str]:
    action = str(subtask.get("action") or "").lower()
    visible_post = _objects_from_agent_state(post_agent_state)
    target_obj = _match_object_by_grounding(subtask, visible_post)
    pre_inv = _inventory_ids(pre_agent_state)
    post_inv = _inventory_ids(post_agent_state)

    if action == "pick":
        if target_obj and str(target_obj.get("objectId")) in post_inv:
            return True, "target object is in inventory"
        new_items = post_inv - pre_inv
        if new_items:
            return True, "inventory gained a new object"
        return False, "target object not in inventory"

    if action == "open":
        if target_obj and bool(target_obj.get("isOpen")):
            return True, "target object is open"
        return False, "target object is not open"

    if action == "close":
        if target_obj and target_obj.get("isOpen") is False:
            return True, "target object is closed"
        return False, "target object is not closed"

    if action in {"find", "inspect"}:
        if target_obj is not None:
            return True, "target object is visible"
        return False, "target object is still not visible"

    if action == "navigate":
        if target_obj and isinstance(target_obj.get("distance"), (int, float)) and target_obj["distance"] < 1.5:
            return True, "agent is close to target"
        return False, "agent is not close enough to target"

    if action == "place":
        if len(post_inv) < len(pre_inv):
            return True, "inventory item was placed"
        return False, "inventory did not change as expected"

    if action in {"toggle_on", "turn_on", "switch_on"}:
        if target_obj and bool(target_obj.get("isToggled")):
            return True, "target object is toggled on"
        return False, "target object is not toggled on"

    if action in {"toggle_off", "turn_off", "switch_off"}:
        if target_obj and target_obj.get("isToggled") is False:
            return True, "target object is toggled off"
        return False, "target object is not toggled off"

    return False, f"no verifier implemented for action={action}"


def _trace_failure_type(subtask: dict[str, Any], trace: dict[str, Any]) -> tuple[bool, str]:
    actions = list(trace.get("actions") or [])
    if not actions:
        if _grounding_is_unresolved(subtask):
            return True, "grounding unresolved; no executable object action generated yet"
        return True, "no executable actions generated"

    errors = " ".join(str(x.get("errorMessage") or "") for x in actions).lower()
    if _grounding_is_unresolved(subtask) and "does not exist" in errors:
        return True, f"stale or unresolved object grounding: {errors}"
    if any(hint in errors for hint in NON_RECOVERABLE_ERROR_HINTS):
        return False, f"non-recoverable error: {errors}"
    if any(hint in errors for hint in RECOVERABLE_ERROR_HINTS):
        return True, f"recoverable error: {errors}"

    success_count = int(trace.get("success_count") or 0)
    num_actions = int(trace.get("num_actions") or 0)
    if num_actions > 0 and success_count == 0:
        return True, "all actions failed but failure looks recoverable"
    if _grounding_is_unresolved(subtask):
        return True, "grounding unresolved; continue search and scene-graph update"
    return True, "task not satisfied yet"


def judge_task_status(
    subtask: dict[str, Any],
    trace: dict[str, Any],
    pre_agent_state: dict[str, Any],
    post_agent_state: dict[str, Any],
    retry_count: int,
    max_retry: int = 3,
) -> dict[str, Any]:
    next_retry_count = retry_count + 1
    ok, reason = _verify_success(subtask, pre_agent_state, post_agent_state)
    if ok:
        return {
            "subtask_id": str(subtask.get("id")),
            "status": SUCCESS,
            "reason": reason,
            "retry_count_before": retry_count,
            "retry_count_after": 0,
        }

    recoverable, fail_reason = _trace_failure_type(subtask, trace)
    if recoverable and retry_count < max_retry:
        return {
            "subtask_id": str(subtask.get("id")),
            "status": WAIT_RETRY,
            "reason": fail_reason,
            "retry_count_before": retry_count,
            "retry_count_after": next_retry_count,
        }

    return {
        "subtask_id": str(subtask.get("id")),
        "status": FAILURE,
        "reason": fail_reason if not recoverable else "retry budget exceeded",
        "retry_count_before": retry_count,
        "retry_count_after": next_retry_count if recoverable else retry_count,
    }
