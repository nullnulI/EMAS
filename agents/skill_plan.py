"""
Low-level skill planning and execution for allocated EMAS subtasks.

The entry point is ``execute_allocated_skills``. It accepts allocation units from
``planning.task_allocation`` and runs one low-level action sequence per
``{"subtask": ..., "agent_id": ...}`` assignment.
"""

from __future__ import annotations

import json
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable


EMAS_ROOT = Path(__file__).resolve().parents[1]
AI2THOR_ROOT = EMAS_ROOT / "ai2thor"
CONCEPTGRAPH_ROOT = EMAS_ROOT / "memory" / "concept-graphs"
for path in (EMAS_ROOT, AI2THOR_ROOT, CONCEPTGRAPH_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


DEFAULT_ACTION_DESCRIPTIONS = {
    "MoveAhead": "Move the agent forward by one navigation step.",
    "MoveBack": "Move the agent backward by one navigation step.",
    "MoveLeft": "Move the agent left by one navigation step.",
    "MoveRight": "Move the agent right by one navigation step.",
    "RotateLeft": "Rotate the agent left.",
    "RotateRight": "Rotate the agent right.",
    "LookUp": "Tilt camera upward.",
    "LookDown": "Tilt camera downward.",
}

OBJECT_ACTION_SPECS: list[dict[str, Any]] = [
    {
        "action": "MoveAgent",
        "description": "Move the agent by relative ahead/right offsets.",
        "params": {"ahead": "optional number", "right": "optional number"},
        "uses_object": False,
    },
    {
        "action": "RotateAgent",
        "description": "Rotate the agent by a custom number of degrees.",
        "params": {"degrees": "required number"},
        "uses_object": False,
    },
    {
        "action": "PickupObject",
        "description": "Pick up a visible pickupable object.",
        "params": {"objectId": "required visible pickupable object id", "forceAction": "optional bool"},
        "uses_object": True,
    },
    {
        "action": "PutObject",
        "description": "Place the held object into or onto a visible receptacle.",
        "params": {"objectId": "required visible receptacle object id", "forceAction": "optional bool"},
        "uses_object": True,
    },
    {
        "action": "DropHandObject",
        "description": "Drop the currently held object.",
        "params": {"forceAction": "optional bool"},
        "uses_object": False,
    },
    {
        "action": "ThrowObject",
        "description": "Throw the currently held object forward.",
        "params": {"moveMagnitude": "optional number", "forceAction": "optional bool"},
        "uses_object": False,
    },
    {
        "action": "OpenObject",
        "description": "Open a visible openable object.",
        "params": {"objectId": "required visible openable object id", "forceAction": "optional bool"},
        "uses_object": True,
    },
    {
        "action": "CloseObject",
        "description": "Close a visible openable object.",
        "params": {"objectId": "required visible openable object id", "forceAction": "optional bool"},
        "uses_object": True,
    },
    {
        "action": "ToggleObjectOn",
        "description": "Turn on a visible toggleable object.",
        "params": {"objectId": "required visible toggleable object id", "forceAction": "optional bool"},
        "uses_object": True,
    },
    {
        "action": "ToggleObjectOff",
        "description": "Turn off a visible toggleable object.",
        "params": {"objectId": "required visible toggleable object id", "forceAction": "optional bool"},
        "uses_object": True,
    },
    {
        "action": "SliceObject",
        "description": "Slice a visible sliceable object.",
        "params": {"objectId": "required visible sliceable object id", "forceAction": "optional bool"},
        "uses_object": True,
    },
    {
        "action": "BreakObject",
        "description": "Break a visible breakable object.",
        "params": {"objectId": "required visible breakable object id", "forceAction": "optional bool"},
        "uses_object": True,
    },
    {
        "action": "DirtyObject",
        "description": "Make a visible dirtyable object dirty.",
        "params": {"objectId": "required visible dirtyable object id", "forceAction": "optional bool"},
        "uses_object": True,
    },
    {
        "action": "CleanObject",
        "description": "Clean a visible dirtyable object.",
        "params": {"objectId": "required visible dirtyable object id", "forceAction": "optional bool"},
        "uses_object": True,
    },
    {
        "action": "FillObjectWithLiquid",
        "description": "Fill a visible fillable object with liquid.",
        "params": {"objectId": "required visible fillable object id", "fillLiquid": "water, coffee, or wine", "forceAction": "optional bool"},
        "uses_object": True,
    },
    {
        "action": "EmptyLiquidFromObject",
        "description": "Empty liquid from a visible fillable object.",
        "params": {"objectId": "required visible fillable object id", "forceAction": "optional bool"},
        "uses_object": True,
    },
    {
        "action": "UseUpObject",
        "description": "Use up a visible consumable object.",
        "params": {"objectId": "required visible object id", "forceAction": "optional bool"},
        "uses_object": True,
    },
    {
        "action": "CookObject",
        "description": "Cook a visible cookable object.",
        "params": {"objectId": "required visible cookable object id", "forceAction": "optional bool"},
        "uses_object": True,
    },
    {
        "action": "MoveHeldObject",
        "description": "Move the held object relative to the agent hand.",
        "params": {"ahead": "optional number", "right": "optional number", "up": "optional number", "forceVisible": "optional bool"},
        "uses_object": False,
    },
    {
        "action": "MoveHeldObjectAhead",
        "description": "Move the held object forward relative to the agent hand.",
        "params": {"moveMagnitude": "required number", "forceVisible": "optional bool"},
        "uses_object": False,
    },
    {
        "action": "MoveHeldObjectBack",
        "description": "Move the held object backward relative to the agent hand.",
        "params": {"moveMagnitude": "required number", "forceVisible": "optional bool"},
        "uses_object": False,
    },
    {
        "action": "MoveHeldObjectLeft",
        "description": "Move the held object left relative to the agent hand.",
        "params": {"moveMagnitude": "required number", "forceVisible": "optional bool"},
        "uses_object": False,
    },
    {
        "action": "MoveHeldObjectRight",
        "description": "Move the held object right relative to the agent hand.",
        "params": {"moveMagnitude": "required number", "forceVisible": "optional bool"},
        "uses_object": False,
    },
    {
        "action": "MoveHeldObjectUp",
        "description": "Move the held object upward relative to the agent hand.",
        "params": {"moveMagnitude": "required number", "forceVisible": "optional bool"},
        "uses_object": False,
    },
    {
        "action": "MoveHeldObjectDown",
        "description": "Move the held object downward relative to the agent hand.",
        "params": {"moveMagnitude": "required number", "forceVisible": "optional bool"},
        "uses_object": False,
    },
    {
        "action": "RotateHeldObject",
        "description": "Rotate the held object.",
        "params": {"pitch": "optional degrees", "yaw": "optional degrees", "roll": "optional degrees"},
        "uses_object": False,
    },
    {
        "action": "Pass",
        "description": "No-op action that advances one simulator step.",
        "params": {},
        "uses_object": False,
    },
]


def build_ai2thor_action_library() -> list[dict[str, Any]]:
    try:
        from ai2thor.ai2thor.interact import DefaultActions

        default_action_names = [action.name for action in DefaultActions]
    except Exception:
        default_action_names = list(DEFAULT_ACTION_DESCRIPTIONS)

    library: list[dict[str, Any]] = []
    seen = set()
    for action_name in default_action_names:
        library.append(
            {
                "action": action_name,
                "description": DEFAULT_ACTION_DESCRIPTIONS.get(action_name, f"AI2-THOR default action {action_name}."),
                "params": {"moveMagnitude": "optional number"} if action_name.startswith("Move") else {"degrees": "optional number"} if action_name.startswith("Rotate") else {},
                "uses_object": False,
                "source": "ai2thor.interact.DefaultActions",
            }
        )
        seen.add(action_name)

    for spec in OBJECT_ACTION_SPECS:
        if spec["action"] in seen:
            continue
        item = dict(spec)
        item["source"] = "ai2thor object interaction action"
        library.append(item)
        seen.add(item["action"])
    return library


ACTION_LIBRARY = build_ai2thor_action_library()
ALLOWED_ACTIONS = {item["action"] for item in ACTION_LIBRARY}
OBJECT_ACTIONS = {item["action"] for item in ACTION_LIBRARY if item.get("uses_object")}
ACTION_OBJECT_REQUIREMENTS = {
    "PickupObject": "pickupable",
    "PutObject": "receptacle",
    "OpenObject": "openable",
    "CloseObject": "openable",
    "ToggleObjectOn": "toggleable",
    "ToggleObjectOff": "toggleable",
    "SliceObject": "sliceable",
    "BreakObject": "breakable",
    "DirtyObject": "dirtyable",
    "CleanObject": "dirtyable",
    "FillObjectWithLiquid": "canFillWithLiquid",
    "EmptyLiquidFromObject": "canFillWithLiquid",
    "UseUpObject": "canBeUsedUp",
    "CookObject": "cookable",
}
ALLOWED_PARAM_KEYS = {
    "objectId",
    "degrees",
    "moveMagnitude",
    "forceAction",
    "forceVisible",
    "placeStationary",
    "randomSeed",
    "ahead",
    "right",
    "up",
    "pitch",
    "yaw",
    "roll",
    "fillLiquid",
}

MOTION_ACTION_PREFIXES = ("Move", "Rotate", "Teleport")
COLLISION_ERROR_HINTS = ("collid", "blocking agent", "blocked by", "obstructed by")


def agent_event_metadata(event: Any, agent_id: int) -> dict[str, Any]:
    events = event_list(event)
    if not events:
        return {}
    index = max(0, min(agent_id, len(events) - 1))
    return dict(getattr(events[index], "metadata", {}) or {})


def metadata_time(metadata: dict[str, Any]) -> float | None:
    value = metadata.get("currentTime")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def collision_from_metadata(metadata: dict[str, Any], action: str) -> tuple[bool, list[str], str]:
    collided_objects = [str(item) for item in metadata.get("collidedObjects") or [] if item]
    if bool(metadata.get("collided")) or collided_objects:
        return True, collided_objects, "ai2thor_metadata"

    error_message = str(metadata.get("errorMessage") or "")
    normalized_error = error_message.lower()
    explicit_collision = "collid" in normalized_error
    blocked_motion = action.startswith(MOTION_ACTION_PREFIXES) and any(
        hint in normalized_error for hint in COLLISION_ERROR_HINTS[1:]
    )
    if explicit_collision or blocked_motion:
        match = re.search(r"collided with:\s*([^.;]+)|([^.;]+?)\s+is blocking agent", error_message, re.I)
        inferred_objects = []
        if match:
            inferred_objects = [str(match.group(1) or match.group(2)).strip()]
        return True, inferred_objects, "error_message_fallback"
    return False, [], "none"


def json_safe(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:
        np = None

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)


def parse_json_from_text(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def event_list(event: Any) -> list[Any]:
    events = getattr(event, "events", None)
    if events is not None:
        return list(events)
    return [event]


def visible_objects_from_metadata(metadata: dict[str, Any], limit: int = 24) -> list[dict[str, Any]]:
    objects = []
    for obj in metadata.get("objects") or []:
        if not isinstance(obj, dict) or not obj.get("visible"):
            continue
        objects.append(
            {
                "objectId": obj.get("objectId"),
                "objectType": obj.get("objectType"),
                "name": obj.get("name"),
                "distance": obj.get("distance"),
                "position": obj.get("position"),
                "pickupable": obj.get("pickupable"),
                "openable": obj.get("openable"),
                "receptacle": obj.get("receptacle"),
                "toggleable": obj.get("toggleable"),
                "sliceable": obj.get("sliceable"),
                "breakable": obj.get("breakable"),
                "dirtyable": obj.get("dirtyable"),
                "canFillWithLiquid": obj.get("canFillWithLiquid"),
                "canBeUsedUp": obj.get("canBeUsedUp"),
                "cookable": obj.get("cookable"),
                "isOpen": obj.get("isOpen"),
                "isToggled": obj.get("isToggled"),
                "isBroken": obj.get("isBroken"),
                "isDirty": obj.get("isDirty"),
                "isFilledWithLiquid": obj.get("isFilledWithLiquid"),
                "fillLiquid": obj.get("fillLiquid"),
                "isUsedUp": obj.get("isUsedUp"),
                "isCooked": obj.get("isCooked"),
            }
        )
        if len(objects) >= limit:
            break
    return json_safe(objects)


def current_agent_metadata(controller: Any, agent_id: int) -> dict[str, Any]:
    events = event_list(controller.last_event)
    if not events:
        return {}
    index = max(0, min(agent_id, len(events) - 1))
    return dict(getattr(events[index], "metadata", {}) or {})


def summarize_agent_state(controller: Any, agent_id: int) -> dict[str, Any]:
    metadata = current_agent_metadata(controller, agent_id)
    agent = metadata.get("agent") or {}
    return json_safe(
        {
            "agent_id": str(agent_id),
            "agent": agent,
            "lastAction": metadata.get("lastAction"),
            "lastActionSuccess": metadata.get("lastActionSuccess"),
            "errorMessage": metadata.get("errorMessage"),
            "inventoryObjects": metadata.get("inventoryObjects") or metadata.get("inventory") or [],
            "visible_objects": visible_objects_from_metadata(metadata),
        }
    )


def grounding_terms(subtask: dict[str, Any]) -> list[str]:
    grounding = subtask.get("grounding") or {}
    terms = []
    terms.extend(str(item) for item in grounding.get("object_tags") or [])
    terms.extend(str(item) for item in grounding.get("relation_texts") or [])
    terms.append(str(subtask.get("name") or ""))
    terms.append(str(subtask.get("description") or ""))
    tokens = re.findall(r"[a-zA-Z0-9_-]+", " ".join(terms).lower())
    stop = {"the", "a", "an", "to", "of", "and", "near", "in", "on", "at", "with"}
    return [token for token in tokens if token not in stop and len(token) > 1]


def grounding_is_unresolved(subtask: dict[str, Any]) -> bool:
    grounding = subtask.get("grounding") or {}
    return str(grounding.get("status") or "").lower() == "unresolved"


def score_object_for_subtask(obj: dict[str, Any], subtask: dict[str, Any], action: str) -> float:
    text = " ".join(
        str(obj.get(key) or "")
        for key in ("objectId", "objectType", "name")
    ).lower()
    score = 0.0
    for term in grounding_terms(subtask):
        if term in text:
            score += 1.0
    if action == "PickupObject" and obj.get("pickupable"):
        score += 2.0
    if action in {"OpenObject", "CloseObject"} and obj.get("openable"):
        score += 2.0
    if action == "PutObject" and obj.get("receptacle"):
        score += 2.0
    requirement = ACTION_OBJECT_REQUIREMENTS.get(action)
    if requirement and obj.get(requirement):
        score += 2.0
    distance = obj.get("distance")
    if isinstance(distance, (int, float)):
        score += max(0.0, 1.5 - float(distance)) * 0.1
    return score


def semantic_object_match_score(obj: dict[str, Any], subtask: dict[str, Any]) -> float:
    text = " ".join(
        str(obj.get(key) or "")
        for key in ("objectId", "objectType", "name")
    ).lower()
    return float(sum(1 for term in grounding_terms(subtask) if term in text))


def resolve_object_id(subtask: dict[str, Any], agent_state: dict[str, Any], action: str) -> str | None:
    candidates = [obj for obj in agent_state.get("visible_objects") or [] if isinstance(obj, dict)]
    if not candidates:
        return None
    terms = grounding_terms(subtask)
    if grounding_is_unresolved(subtask) or terms:
        semantic_scored = [
            (semantic_object_match_score(obj, subtask), score_object_for_subtask(obj, subtask, action), obj)
            for obj in candidates
            if obj.get("objectId")
        ]
        semantic_scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if semantic_scored and semantic_scored[0][0] > 0:
            return str(semantic_scored[0][2]["objectId"])
        return None

    scored = [
        (score_object_for_subtask(obj, subtask, action), obj)
        for obj in candidates
        if obj.get("objectId")
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored and scored[0][0] > 0:
        return str(scored[0][1]["objectId"])
    return str(scored[0][1]["objectId"]) if scored else None


def action_for_subtask_type(subtask: dict[str, Any]) -> str:
    action = str(subtask.get("action") or "").lower()
    if action == "pick":
        return "PickupObject"
    if action == "open":
        return "OpenObject"
    if action == "close":
        return "CloseObject"
    if action == "place":
        return "PutObject"
    if action in {"toggle_on", "turn_on", "switch_on"}:
        return "ToggleObjectOn"
    if action in {"toggle_off", "turn_off", "switch_off"}:
        return "ToggleObjectOff"
    if action == "slice":
        return "SliceObject"
    if action == "break":
        return "BreakObject"
    if action == "dirty":
        return "DirtyObject"
    if action == "clean":
        return "CleanObject"
    if action == "fill":
        return "FillObjectWithLiquid"
    if action == "empty":
        return "EmptyLiquidFromObject"
    if action in {"use", "use_up"}:
        return "UseUpObject"
    if action == "cook":
        return "CookObject"
    if action in {"find", "inspect"}:
        return "RotateRight"
    if action == "navigate":
        return "MoveAhead"
    return "MoveAhead"


def heuristic_skill_plan(subtask: dict[str, Any], agent_state: dict[str, Any], max_steps: int) -> dict[str, Any]:
    subtask_action = str(subtask.get("action") or "").lower()
    target_action = action_for_subtask_type(subtask)
    actions: list[dict[str, Any]] = []

    if subtask_action in {"find", "inspect"}:
        actions = [
            {"action": "RotateRight", "params": {"degrees": 45}, "reason": "scan visible scene"},
            {"action": "LookDown", "params": {}, "reason": "inspect nearby objects"},
            {"action": "LookUp", "params": {}, "reason": "inspect higher shelves and counters"},
            {"action": "RotateRight", "params": {"degrees": 45}, "reason": "continue search sweep"},
            {"action": "MoveAhead", "params": {}, "reason": "move to reveal previously unseen objects"},
        ]
    elif subtask_action == "navigate":
        actions = [
            {"action": "MoveAhead", "params": {}, "reason": "approach the target region"},
            {"action": "RotateRight", "params": {"degrees": 30}, "reason": "adjust heading"},
            {"action": "MoveAhead", "params": {}, "reason": "continue approach"},
        ]
    elif target_action in OBJECT_ACTIONS:
        object_id = resolve_object_id(subtask, agent_state, target_action)
        if object_id:
            params: dict[str, Any] = {"objectId": object_id}
            if target_action in OBJECT_ACTIONS:
                params["forceAction"] = True
            if target_action == "FillObjectWithLiquid":
                params["fillLiquid"] = "water"
            actions = [
                {"action": target_action, "params": params, "reason": "execute grounded object interaction"}
            ]
        else:
            actions = [
                {"action": "RotateRight", "params": {"degrees": 45}, "reason": "search for target object"},
                {"action": "LookDown", "params": {}, "reason": "look for nearby target object"},
                {"action": "LookUp", "params": {}, "reason": "look for target object on elevated surfaces"},
                {"action": "RotateRight", "params": {"degrees": 45}, "reason": "continue active grounding search"},
                {"action": "MoveAhead", "params": {}, "reason": "explore to update visible objects"},
            ]
    else:
        actions = [{"action": "MoveAhead", "params": {}, "reason": "default exploratory action"}]

    return {
        "planner_backend": "heuristic_fallback",
        "reasoning_summary": (
            "Heuristic skill sequence from subtask action type; unresolved grounding triggers search actions."
        ),
        "grounding_recovery": grounding_is_unresolved(subtask),
        "actions": actions[: max(1, max_steps)],
    }


def call_local_qwen_for_skill_plan(
    subtask: dict[str, Any],
    agent_state: dict[str, Any],
    action_library: list[dict[str, Any]],
    max_steps: int,
    model_path: str | None,
    conv_mode: str,
    num_gpus: int,
    qwen_chat: Any | None = None,
) -> dict | None:
    system_prompt = (
        "You are a low-level AI2-THOR skill planner. Convert one assigned robot "
        "subtask into a short sequence of executable AI2-THOR actions. Return only "
        "valid JSON. Use only actions from the provided action library."
    )
    user_payload = {
        "assigned_subtask": subtask,
        "agent_state": agent_state,
        "action_library": action_library,
        "max_steps": max_steps,
        "output_schema": {
            "reasoning_summary": "short explanation",
            "actions": [
                {
                    "action": "one action name from action_library",
                    "params": {"objectId": "only when required"},
                    "reason": "why this action is useful",
                }
            ],
        },
        "constraints": [
            "Return JSON only.",
            "Use only action names from action_library.",
            "Do not include agentId in params.",
            "Use visible objectId values from agent_state when an object interaction is needed.",
            "If the needed object is not visible or assigned_subtask.grounding.status is unresolved, do not declare failure and do not invent objectId values; produce search/inspect/navigation actions such as RotateRight, LookUp, LookDown, and MoveAhead.",
            "Keep the sequence short and executable.",
        ],
    }

    chat = qwen_chat
    owns_chat = chat is None
    close_vlm_chat = None
    if owns_chat:
        try:
            from conceptgraph.vlm import build_vlm_chat, close_vlm_chat
        except Exception:
            return None
    try:
        if chat is None:
            chat = build_vlm_chat(
                backend="qwen",
                model_path=model_path,
                conv_mode=conv_mode,
                num_gpus=num_gpus,
            )
        if hasattr(chat, "reset"):
            chat.reset()
        if hasattr(chat, "messages"):
            chat.messages = [{"role": "system", "content": system_prompt}]
        response_text = chat(json.dumps(user_payload, ensure_ascii=False, indent=2))
    except Exception:
        return None
    finally:
        if owns_chat and close_vlm_chat is not None:
            close_vlm_chat(chat)

    plan = parse_json_from_text(response_text)
    if isinstance(plan, dict):
        plan["planner_backend"] = "qwen_local"
    return plan


def clean_action_params(params: Any) -> dict[str, Any]:
    if not isinstance(params, dict):
        return {}
    cleaned: dict[str, Any] = {}
    for key, value in params.items():
        if key == "agentId":
            continue
        if key in ALLOWED_PARAM_KEYS:
            cleaned[key] = value
    return cleaned


def sanitize_skill_plan(
    raw_plan: dict | None,
    subtask: dict[str, Any],
    agent_state: dict[str, Any],
    max_steps: int,
) -> dict[str, Any]:
    if not isinstance(raw_plan, dict) or not isinstance(raw_plan.get("actions"), list):
        return heuristic_skill_plan(subtask, agent_state, max_steps)

    actions = []
    for item in raw_plan.get("actions") or []:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or "").strip()
        if action not in ALLOWED_ACTIONS:
            continue
        params = clean_action_params(item.get("params") or {})
        if action in OBJECT_ACTIONS and not params.get("objectId"):
            object_id = resolve_object_id(subtask, agent_state, action)
            if object_id:
                params["objectId"] = object_id
            else:
                continue
        actions.append(
            {
                "action": action,
                "params": params,
                "reason": str(item.get("reason") or ""),
            }
        )
        if len(actions) >= max(1, max_steps):
            break

    if not actions:
        return heuristic_skill_plan(subtask, agent_state, max_steps)

    return {
        "planner_backend": str(raw_plan.get("planner_backend") or "qwen_local"),
        "reasoning_summary": str(raw_plan.get("reasoning_summary") or ""),
        "grounding_recovery": grounding_is_unresolved(subtask),
        "actions": actions,
    }


def plan_skill_for_assignment(
    controller: Any,
    assignment: dict[str, Any],
    *,
    use_qwen: bool,
    qwen_model_path: str | None,
    qwen_conv_mode: str,
    qwen_num_gpus: int,
    max_steps: int,
    qwen_chat: Any | None = None,
) -> dict[str, Any]:
    subtask = assignment.get("subtask") or {}
    agent_id = int(assignment.get("agent_id", 0))
    agent_state = summarize_agent_state(controller, agent_id)

    raw_plan = None
    if use_qwen:
        raw_plan = call_local_qwen_for_skill_plan(
            subtask=subtask,
            agent_state=agent_state,
            action_library=ACTION_LIBRARY,
            max_steps=max_steps,
            model_path=qwen_model_path,
            conv_mode=qwen_conv_mode,
            num_gpus=qwen_num_gpus,
            qwen_chat=qwen_chat,
        )

    plan = sanitize_skill_plan(raw_plan, subtask, agent_state, max_steps)
    plan["agent_state_before_plan"] = agent_state
    plan["subtask"] = deepcopy(subtask)
    plan["agent_id"] = str(agent_id)
    return json_safe(plan)


def execute_action(
    controller: Any,
    agent_id: int,
    action_item: dict[str, Any],
    *,
    step_callback: Callable[[Any, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    action = str(action_item["action"])
    params = dict(action_item.get("params") or {})
    before_metadata = agent_event_metadata(getattr(controller, "last_event", None), agent_id)
    simulation_time_before = metadata_time(before_metadata)
    wall_time_start = time.perf_counter()
    try:
        event = controller.step(action=action, agentId=agent_id, **params)
        wall_time_seconds = time.perf_counter() - wall_time_start
        metadata = agent_event_metadata(event, agent_id)
        simulation_time_after = metadata_time(metadata)
        simulation_time_seconds = None
        if simulation_time_before is not None and simulation_time_after is not None:
            simulation_time_seconds = max(0.0, simulation_time_after - simulation_time_before)
        collided, collided_objects, collision_source = collision_from_metadata(metadata, action)
        result = json_safe(
            {
                "action": action,
                "params": params,
                "reason": action_item.get("reason"),
                "lastActionSuccess": bool(metadata.get("lastActionSuccess")),
                "errorMessage": metadata.get("errorMessage"),
                "wall_time_seconds": wall_time_seconds,
                "simulation_time_seconds": simulation_time_seconds,
                "collided": collided,
                "collided_objects": collided_objects,
                "collision_source": collision_source,
            }
        )
        if step_callback is not None:
            step_callback(event, result)
        return result
    except Exception as exc:
        wall_time_seconds = time.perf_counter() - wall_time_start
        error_message = repr(exc)
        collided, collided_objects, collision_source = collision_from_metadata(
            {"errorMessage": error_message}, action
        )
        return json_safe(
            {
                "action": action,
                "params": params,
                "reason": action_item.get("reason"),
                "lastActionSuccess": False,
                "errorMessage": error_message,
                "wall_time_seconds": wall_time_seconds,
                "simulation_time_seconds": None,
                "collided": collided,
                "collided_objects": collided_objects,
                "collision_source": collision_source,
            }
        )


def execute_allocated_skills(
    controller: Any,
    assignments: list[dict[str, Any]],
    *,
    use_qwen: bool = True,
    qwen_model_path: str | None = None,
    qwen_conv_mode: str = "v0_mmtag",
    qwen_num_gpus: int = 1,
    max_steps: int = 6,
    complete_on_execute: bool = True,
    step_callback: Callable[[Any, dict[str, Any]], None] | None = None,
    qwen_chat: Any | None = None,
) -> dict[str, Any]:
    macro_step_start = time.perf_counter()
    traces = []
    completed_task_ids = []

    for assignment in assignments:
        subtask = assignment.get("subtask") or {}
        task_id = str(subtask.get("id"))
        agent_id = int(assignment.get("agent_id", 0))
        plan = plan_skill_for_assignment(
            controller,
            assignment,
            use_qwen=use_qwen,
            qwen_model_path=qwen_model_path,
            qwen_conv_mode=qwen_conv_mode,
            qwen_num_gpus=qwen_num_gpus,
            max_steps=max_steps,
            qwen_chat=qwen_chat,
        )

        action_results = [
            execute_action(controller, agent_id, action_item, step_callback=step_callback)
            for action_item in plan.get("actions") or []
        ]
        success_count = sum(1 for item in action_results if item.get("lastActionSuccess"))
        execution_time_seconds = sum(float(item.get("wall_time_seconds") or 0.0) for item in action_results)
        simulation_time_values = [
            float(item["simulation_time_seconds"])
            for item in action_results
            if isinstance(item.get("simulation_time_seconds"), (int, float))
        ]
        collision_actions = [item for item in action_results if item.get("collided")]
        collided_objects = sorted(
            {
                str(object_id)
                for item in collision_actions
                for object_id in item.get("collided_objects") or []
            }
        )
        completed = bool(action_results) if complete_on_execute else success_count == len(action_results)
        if completed:
            completed_task_ids.append(task_id)

        traces.append(
            json_safe(
                {
                    "subtask_id": task_id,
                    "subtask": subtask,
                    "agent_id": str(agent_id),
                    "executor": "qwen_skill_plan" if plan.get("planner_backend") == "qwen_local" else "heuristic_skill_plan",
                    "planner_backend": plan.get("planner_backend"),
                    "completed": completed,
                    "success_count": success_count,
                    "num_actions": len(action_results),
                    "execution_time_seconds": execution_time_seconds,
                    "simulation_time_seconds": sum(simulation_time_values) if simulation_time_values else None,
                    "collided": bool(collision_actions),
                    "collision_count": len(collision_actions),
                    "collided_objects": collided_objects,
                    "skill_plan": plan,
                    "actions": action_results,
                }
            )
        )

    macro_step_wall_time_seconds = time.perf_counter() - macro_step_start
    collision_traces = [trace for trace in traces if trace.get("collided")]
    simulation_time_values = [
        float(trace["simulation_time_seconds"])
        for trace in traces
        if isinstance(trace.get("simulation_time_seconds"), (int, float))
    ]
    return {
        "completed_task_ids": completed_task_ids,
        "traces": traces,
        "execution_time_seconds": sum(float(trace.get("execution_time_seconds") or 0.0) for trace in traces),
        "macro_step_wall_time_seconds": macro_step_wall_time_seconds,
        "simulation_time_seconds": sum(simulation_time_values) if simulation_time_values else None,
        "collided": bool(collision_traces),
        "collision_count": sum(int(trace.get("collision_count") or 0) for trace in traces),
        "collided_objects": sorted(
            {
                str(object_id)
                for trace in collision_traces
                for object_id in trace.get("collided_objects") or []
            }
        ),
    }
