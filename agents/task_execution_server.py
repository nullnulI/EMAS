#!/usr/bin/env python3
"""HTTP wrapper for EmbodiedGPT's multi-robot relay closed-loop runtime."""

from __future__ import annotations

from copy import deepcopy
import argparse
import contextlib
import io
import json
import re
import sys
import threading
import time
import traceback
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse


MAX_REQUEST_BYTES = 128 * 1024
SERVICE_NAME = "task_execution_server"
SERVICE_REVISION = "2026-08-11-placement-recovery-v12"
REPO_ROOT = Path(__file__).resolve().parent
EMBODIED_ROOT = REPO_ROOT / "EmbodiedGPT_Pytorch"
EMAS_ROOT = REPO_ROOT.parent
if str(EMAS_ROOT) not in sys.path:
    sys.path.insert(0, str(EMAS_ROOT))

from action_contracts import ACTION_CONTRACTS, validate_action_args

SAFE_NATIVE_ACTIONS = {
    str(contract["native_action"])
    for contract in ACTION_CONTRACTS.values()
}
SUPPORTED_NORMALIZED_ACTIONS = {
    "GotoObject",
    *SAFE_NATIVE_ACTIONS,
    "Done",
    "MoveAhead",
    "MoveBack",
    "MoveLeft",
    "MoveRight",
    "RotateLeft",
    "RotateRight",
    "LookUp",
    "LookDown",
}
NO_ARG_NORMALIZED_ACTIONS = {
    "MoveAhead",
    "MoveBack",
    "MoveLeft",
    "MoveRight",
    "RotateLeft",
    "RotateRight",
    "LookUp",
    "LookDown",
    "Done",
}
ACTION_INTENT_PATTERNS = (
    (("turn off", "switch off", "toggle off"), "ToggleObjectOff"),
    (("turn on", "switch on", "toggle on"), "ToggleObjectOn"),
    (("turn right", "rotate right"), "RotateRight"),
    (("turn left", "rotate left"), "RotateLeft"),
    (("move right", "strafe right", "step right"), "MoveRight"),
    (("move left", "strafe left", "step left"), "MoveLeft"),
    (("move forward", "go forward", "move ahead"), "MoveAhead"),
    (("move back", "go back", "back up"), "MoveBack"),
    (("look up",), "LookUp"),
    (("look down",), "LookDown"),
    (("go to", "navigate to", "walk to", "find", "search", "inspect"), "GotoObject"),
    (("pick up", "pickup", "grab", "take"), "PickupObject"),
    (("open",), "OpenObject"),
    (("close", "shut"), "CloseObject"),
    (("put", "place"), "PutObject"),
    (("slice", "cut"), "SliceObject"),
    (("clean", "wash"), "CleanObject"),
    (("drop", "release"), "DropHandObject"),
    (("push", "shove"), "PushObject"),
    (("pull", "drag"), "PullObject"),
    (("reposition held", "move held", "adjust held"), "MoveHeldObject"),
    (("break", "smash"), "BreakObject"),
    (("cook", "heat"), "CookObject"),
    (("fill",), "FillObjectWithLiquid"),
)
OPENABLE_RECEPTACLE_TYPE_NAMES = {
    "box",
    "cabinet",
    "drawer",
    "fridge",
    "garbagecan",
    "microwave",
    "safe",
}
MAPTHOR_OBJECT_TYPE_ALIASES = {
    "computer": ("Laptop",),
    "laptopcomputer": ("Laptop",),
    "keys": ("KeyChain",),
    "couch": ("Sofa",),
    "refrigerator": ("Fridge",),
    "counter": ("CounterTop",),
    "kitchencounter": ("CounterTop",),
    "table": ("Table", "DiningTable", "CoffeeTable", "SideTable"),
}
TASK_NORMALIZER_TOOL_NAME = "normalize_incoming_task"
TASK_NORMALIZER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": TASK_NORMALIZER_TOOL_NAME,
        "description": (
            "Normalize an upstream planning subtask into one concise robot task and ordered intent steps "
            "that the agents runtime can execute. Use only supported actions and object types from the "
            "provided AI2-THOR object types."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "normalized_task": {"type": "string"},
                "intentSteps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "order": {"type": "integer"},
                            "action": {"type": "string", "enum": sorted(SUPPORTED_NORMALIZED_ACTIONS)},
                            "objectType": {"type": ["string", "null"]},
                            "targetType": {"type": ["string", "null"]},
                        },
                        "required": ["order", "action", "objectType", "targetType"],
                    },
                },
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "reason": {"type": "string"},
            },
            "required": ["normalized_task", "intentSteps", "confidence", "reason"],
        },
    },
}


def log(message: str) -> None:
    print(f"[RELAY TASK] {message}", flush=True)



def _json_values_from_text(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    values: list[Any] = []
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        values.append(value)
    return values


def _tool_call_from_value(value: Any) -> dict[str, Any] | None:
    if isinstance(value, list):
        for item in value:
            tool_call = _tool_call_from_value(item)
            if tool_call is not None:
                return tool_call
        return None
    if not isinstance(value, dict):
        return None
    if isinstance(value.get("tool_calls"), list):
        for item in value["tool_calls"]:
            tool_call = _tool_call_from_value(item)
            if tool_call is not None:
                return tool_call
    function_value = value.get("function")
    if isinstance(function_value, dict):
        if "parameters" in function_value and "arguments" not in function_value and "arguments" not in value:
            return None
        name = function_value.get("name") or value.get("name")
        arguments = function_value.get("arguments", value.get("arguments", {}))
    else:
        name = value.get("name") or value.get("tool_name")
        arguments = value.get("arguments", value.get("parameters", {}))
    if not isinstance(name, str) or not name.strip():
        return None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"raw": arguments}
    if not isinstance(arguments, dict):
        arguments = {}
    return {"name": name.strip(), "arguments": arguments}


def _parse_qwen_parameter_tool_call(output: str) -> dict[str, Any] | None:
    function_match = re.search(r"<function=([^>]+)>", output, flags=re.IGNORECASE)
    if function_match is None:
        return None
    name = function_match.group(1).strip()
    arguments: dict[str, Any] = {}
    parameter_matches = list(re.finditer(r"<parameter=([^>]+)>\s*", output, flags=re.IGNORECASE))
    for index, match in enumerate(parameter_matches):
        key = match.group(1).strip()
        start = match.end()
        end = parameter_matches[index + 1].start() if index + 1 < len(parameter_matches) else len(output)
        raw_value = output[start:end]
        raw_value = re.split(r"</parameter>|</function>|</tool_call>", raw_value, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if key in {"intentSteps", "intent_steps"}:
            try:
                arguments[key] = json.loads(raw_value)
            except json.JSONDecodeError:
                arguments[key] = raw_value
        elif raw_value.lower() == "null":
            arguments[key] = None
        else:
            arguments[key] = raw_value
    return {"name": name, "arguments": arguments}


def parse_task_normalizer_tool_call(output: str) -> dict[str, Any]:
    tool_blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", output, flags=re.DOTALL | re.IGNORECASE)
    for block in tool_blocks:
        parameter_tool_call = _parse_qwen_parameter_tool_call(block)
        if parameter_tool_call is not None:
            return parameter_tool_call
        for value in _json_values_from_text(block):
            tool_call = _tool_call_from_value(value)
            if tool_call is not None:
                return tool_call

    output_without_tool_schemas = re.sub(r"<tools>.*?</tools>", "", output, flags=re.DOTALL | re.IGNORECASE)
    parameter_tool_call = _parse_qwen_parameter_tool_call(output_without_tool_schemas)
    if parameter_tool_call is not None:
        return parameter_tool_call
    for value in _json_values_from_text(output_without_tool_schemas):
        tool_call = _tool_call_from_value(value)
        if tool_call is not None:
            return tool_call
    raise ValueError("Qwen output did not contain a valid task normalization tool call")


def _object_type_from_object(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None
    for key in ("objectType", "object_type", "type"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    object_id = item.get("objectId") or item.get("object_id")
    if isinstance(object_id, str) and "|" in object_id:
        head = object_id.split("|", 1)[0].strip()
        return head or None
    return None


def extract_state_object_types(state: dict[str, Any]) -> list[str]:
    object_types: list[str] = []
    def add(value: str | None) -> None:
        if value and value not in object_types:
            object_types.append(value)
    for item in state.get("objects") or []:
        add(_object_type_from_object(item))
    for robot in state.get("robots") or []:
        if not isinstance(robot, dict):
            continue
        for key in ("visible_objects", "objects", "inventory"):
            for item in robot.get(key) or []:
                add(_object_type_from_object(item))
    return sorted(object_types)


def fetch_receiver_state(receiver_url: str, timeout: float) -> tuple[dict[str, Any] | None, str | None]:
    state_url = f"{receiver_url.rstrip('/')}/state"
    try:
        with urlopen(state_url, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
        parsed = json.loads(body)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return None, f"could not read receiver /state: {type(exc).__name__}: {exc}"
    if not isinstance(parsed, dict):
        return None, "receiver /state did not return a JSON object"
    return parsed, None


def post_receiver_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def fetch_receiver_state_object_types(receiver_url: str, timeout: float) -> tuple[list[str], str | None]:
    state_url = f"{receiver_url.rstrip('/')}/state"
    try:
        with urlopen(state_url, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
        parsed = json.loads(body)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return [], f"could not read receiver /state for task normalization: {type(exc).__name__}: {exc}"
    if not isinstance(parsed, dict):
        return [], "receiver /state did not return a JSON object"
    return extract_state_object_types(parsed), None


def _post_state_visible_object(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict) or not bool(item.get("visible", False)):
        return None
    object_id = item.get("objectId") or item.get("object_id") or item.get("id")
    object_type = _object_type_from_object(item)
    if not object_id or not object_type:
        return None
    summary = {"objectId": str(object_id), "objectType": object_type}
    for key in (
        "name", "distance", "position", "pickupable", "openable", "isOpen",
        "toggleable", "isToggled", "sliceable", "isSliced", "dirtyable", "isDirty",
        "breakable", "isBroken", "cookable", "isCooked", "canFillWithLiquid",
        "isFilledWithLiquid", "fillLiquid", "receptacle", "parentReceptacles",
        "receptacleObjectIds",
    ):
        if key in item:
            summary[key] = item[key]
    return summary


def _post_state_catalog_object(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    object_id = item.get("objectId") or item.get("object_id") or item.get("id")
    object_type = _object_type_from_object(item)
    if not object_id or not object_type:
        return None
    summary = {"objectId": str(object_id), "objectType": object_type}
    for key in (
        "name", "distance", "position", "visible", "pickupable", "moveable",
        "openable", "isOpen", "toggleable", "isToggled", "sliceable", "isSliced",
        "dirtyable", "isDirty", "breakable", "isBroken", "cookable", "isCooked",
        "canFillWithLiquid", "isFilledWithLiquid", "fillLiquid", "receptacle",
        "parentReceptacles", "receptacleObjectIds", "isPickedUp",
    ):
        if key in item:
            summary[key] = item[key]
    return summary


def normalize_receiver_agent_state(state: dict[str, Any], robot_id: int) -> dict[str, Any]:
    """Convert receiver /state into the shape consumed by hybrid planning."""

    robots = [item for item in state.get("robots") or [] if isinstance(item, dict)]
    robot = next((item for item in robots if item.get("robot_id") == robot_id), {})
    agent = state.get("agent") if isinstance(state.get("agent"), dict) else {}
    position = agent.get("position") or robot.get("position") or {}
    rotation = agent.get("rotation") or robot.get("rotation") or {}
    horizon = agent.get("cameraHorizon", agent.get("horizon", robot.get("horizon", 0.0)))
    inventory = state.get("inventory")
    if not isinstance(inventory, list):
        inventory = robot.get("inventory") if isinstance(robot.get("inventory"), list) else []
    visible_objects = []
    scene_object_catalog = []
    for item in state.get("objects") or []:
        summary = _post_state_visible_object(item)
        if summary is not None:
            visible_objects.append(summary)
        catalog_item = _post_state_catalog_object(item)
        if catalog_item is not None:
            scene_object_catalog.append(catalog_item)

    return {
        "agent_id": str(robot_id),
        "robot_id": robot_id,
        "agent": {
            "name": robot.get("name", f"robot_{robot_id}"),
            "position": position,
            "rotation": rotation,
            "cameraHorizon": horizon,
        },
        "position": position,
        "rotation": rotation,
        "horizon": horizon,
        "inventoryObjects": inventory,
        "inventory": inventory,
        "visible_objects": visible_objects,
        "scene_object_catalog": scene_object_catalog,
        "lastAction": robot.get("last_action"),
        "lastActionSuccess": robot.get("last_success"),
        "errorMessage": robot.get("last_error") or "",
        "state_step": state.get("step"),
        "state_timestamp": time.time(),
        "sceneName": state.get("sceneName"),
    }


def fetch_receiver_agent_states(
    receiver_url: str,
    robot_ids: list[int],
    timeout: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch authoritative post-execution state for every known robot."""

    states: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for robot_id in robot_ids:
        query = urlencode({"robot_id": robot_id, "render_image": "false"})
        state_url = f"{receiver_url.rstrip('/')}/state?{query}"
        try:
            with urlopen(state_url, timeout=timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
            if not isinstance(parsed, dict):
                raise ValueError("receiver /state did not return a JSON object")
            normalized = normalize_receiver_agent_state(parsed, robot_id)
            # The full catalogue is shared across robot views; include it once
            # in the execution report instead of duplicating the large payload.
            if states:
                normalized.pop("scene_object_catalog", None)
            states.append(normalized)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError) as exc:
            errors.append({"robot_id": robot_id, "error": f"{type(exc).__name__}: {exc}"})
    return states, errors


def fetch_receiver_health(receiver_url: str, timeout: float) -> dict[str, Any]:
    health_url = f"{receiver_url.rstrip('/')}/health"
    try:
        with urlopen(health_url, timeout=min(float(timeout), 5.0)) as response:
            body = response.read().decode("utf-8", errors="replace")
        parsed = json.loads(body)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return {
            "reachable": False,
            "controller_ready": False,
            "robot_ids": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(parsed, dict):
        return {
            "reachable": True,
            "controller_ready": False,
            "robot_ids": [],
            "error": "receiver /health did not return a JSON object",
        }
    robot_count = parsed.get("robots")
    robot_count = robot_count if isinstance(robot_count, int) and not isinstance(robot_count, bool) else 0
    return {
        "reachable": parsed.get("status") == "ok",
        "controller_ready": parsed.get("status") == "ok" and robot_count > 0,
        "robot_ids": list(range(robot_count)),
        "service": parsed.get("service"),
    }


def coordinator_robot_ids_for_post_state(
    receiver_url: str,
    timeout: float,
    primary_robot_id: int,
    runtime_result: dict[str, Any],
) -> tuple[list[int], dict[str, Any]]:
    """Refresh the Coordinator-owned robot registry from authoritative runtime state."""

    robot_ids: list[int] = []
    sources: list[str] = []

    def extend(values: Any) -> None:
        if not isinstance(values, list):
            return
        for value in values:
            if (
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
                and value not in robot_ids
            ):
                robot_ids.append(value)

    receiver_health = fetch_receiver_health(receiver_url, timeout)
    extend(receiver_health.get("robot_ids"))
    if robot_ids:
        sources.append("receiver_health")

    runtime_containers = [runtime_result]
    for key in ("relay_result", "closed_loop_result"):
        nested = runtime_result.get(key)
        if isinstance(nested, dict):
            runtime_containers.append(nested)
    for trace in runtime_result.get("closed_loop_trace") or []:
        if not isinstance(trace, dict):
            continue
        runtime_containers.append(trace)
        nested = trace.get("relay_result")
        if isinstance(nested, dict):
            runtime_containers.append(nested)
    before_runtime = len(robot_ids)
    for container in runtime_containers:
        extend(container.get("known_robot_ids"))
    if len(robot_ids) > before_runtime:
        sources.append("coordinator_runtime_discovery")

    if primary_robot_id not in robot_ids:
        robot_ids.append(primary_robot_id)
        sources.append("primary_fallback")

    return sorted(robot_ids), {
        "source": "+".join(sources) or "primary_fallback",
        "receiver_health": receiver_health,
    }


def task_normalizer_messages(
    task: str,
    object_types: list[str],
    feedback: list[str] | None = None,
    *,
    original_task: str | None = None,
    subtask_context: dict[str, Any] | None = None,
    action_coverage_text: str | None = None,
) -> list[dict[str, Any]]:
    supported = ", ".join(sorted(SUPPORTED_NORMALIZED_ACTIONS))
    objects = ", ".join(object_types) if object_types else "(none)"
    primary_task = task.strip()
    coverage_task = (
        action_coverage_text.strip()
        if isinstance(action_coverage_text, str) and action_coverage_text.strip()
        else primary_task
    )
    reference_steps = compact_reference_intent_steps(recognized_intent_steps_for_task(coverage_task, object_types))
    reference_text = ""
    if reference_steps:
        reference_actions = ", ".join(str(step.get("action")) for step in reference_steps)
        reference_text = (
            "\nHard action constraint inferred from action_coverage_text: "
            f"intentSteps must contain these requested action names in this order: {reference_actions}. "
            "Use the objectType/targetType implied by each action, and keep no-argument actions null. "
            "Extra helper actions are allowed only when they directly support a PickupObject task for the same object or a PutObject/Place task for the same object/receptacle. "
            "If action_coverage_text contains both find/locate/search and pick up, keep both GotoObject and PickupObject.\n"
        )
    feedback_text = ""
    if feedback:
        feedback_text = (
            "\nPrevious normalize_incoming_task output was rejected by hard validation for these reason(s): "
            + "; ".join(str(item) for item in feedback)
            + "\nReturn a corrected normalize_incoming_task tool call. Do not repeat rejected actions or invalid object types.\n"
        )
    context_text = ""
    if isinstance(subtask_context, dict) and subtask_context:
        context_text = (
            "\nStructured subtask context, for disambiguation only; do not let action='other' or repeated object tags "
            "override primary_task_text:\n"
            + json.dumps(subtask_context, ensure_ascii=False, sort_keys=True)
            + "\n"
        )
    coverage_text = ""
    if coverage_task and coverage_task != primary_task:
        coverage_text = (
            "\nAction coverage text combines subtask name and description. Preserve actions found here, "
            "but keep primary_task_text as the main wording:\n"
            f"{coverage_task}\n"
        )
    original_text = ""
    if isinstance(original_task, str) and original_task.strip() and original_task.strip() != primary_task:
        original_text = (
            "\nOriginal relay task text may include duplicated hybrid context. Use it only as secondary context:\n"
            f"{original_task.strip()}\n"
        )
    prompt = (
        "You are the agents module boundary normalizer. Convert an upstream planning subtask into one concise "
        "natural-language robot task and ordered intent steps for the agents runtime.\n"
        "Rules:\n"
        "- Call normalize_incoming_task exactly once.\n"
        "- Fill only normalized_task, intentSteps, confidence, and reason.\n"
        "- primary_task_text is the highest-priority wording. action_coverage_text may require additional actions from the description.\n"
        "- Use only these actions: " + supported + ".\n"
        "- objectType and targetType must be null or exactly one of the current AI2-THOR object types.\n"
        "- For LookDown, LookUp, movement, and rotation commands, preserve the exact no-argument action with objectType=null and targetType=null.\n"
        "- Never replace a requested action with a different action. If you cannot represent a requested action, set confidence low.\n"
        "- Do not replace requested actions or add unrelated actions. PickupObject tasks may add GotoObject for the same object. PutObject/Place tasks are compound: you may add GotoObject for the object/receptacle, PickupObject for the object, and OpenObject/CloseObject for an openable receptacle when those steps are needed to complete the same PutObject.\n"
        "- Destination placement order is: GotoObject(item), PickupObject(item), GotoObject(target receptacle), OpenObject(target if openable), PutObject(item,target), CloseObject(target if openable). Map-THOR placement tasks usually require closing openable receptacles after the put.\n"
        "- Do not navigate to or open the destination receptacle before acquiring the item, unless the task explicitly says the item is already held.\n"
        "- Fill intentSteps with every required high-level step in order. Multi-step tasks must not be collapsed.\n"
        "- Use GotoObject when primary_task_text or action_coverage_text asks to go/navigate/move to/find/search/inspect a target object.\n"
        "- Do not invent object types. If no suitable object type exists, use confidence low and keep the task close to the original.\n\n"
        f"Current AI2-THOR object types: {objects}\n"
        "Use object type spelling exactly as listed above; for example, use CounterTop rather than counter.\n"
        f"{reference_text}"
        f"{feedback_text}"
        f"{context_text}"
        f"{coverage_text}"
        f"{original_text}\n"
        f"primary_task_text: {primary_task}"
    )
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]




def _preview_text(value: Any, limit: int = 1000) -> str:
    text = str(value).strip()
    return text if len(text) <= limit else text[:limit] + "..."


def _first_present(arguments: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in arguments:
            return arguments.get(key)
    return None


def _normalize_confidence(value: Any) -> tuple[str, str | None]:
    if isinstance(value, str):
        confidence = value.strip().lower()
        if confidence in {"high", "medium", "low"}:
            return confidence, None
        try:
            value = float(confidence)
        except ValueError:
            return "low", f"normalizer returned invalid confidence {value!r}"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        score = float(value)
        if score >= 0.75:
            return "high", None
        if score >= 0.4:
            return "medium", None
        return "low", None
    return "low", f"normalizer returned invalid confidence {value!r}"


def _canonical_task_from_normalized(action: Any, object_type: Any, target_type: Any) -> str | None:
    if action == "Done":
        return "done."
    if not isinstance(object_type, str) or not object_type:
        return None
    if action == "GotoObject":
        return f"go to the {object_type}."
    if action == "PickupObject":
        return f"pick up the {object_type}."
    if action == "OpenObject":
        return f"open the {object_type}."
    if action == "CloseObject":
        return f"close the {object_type}."
    if action == "ToggleObjectOn":
        return f"turn on the {object_type}."
    if action == "ToggleObjectOff":
        return f"turn off the {object_type}."
    if action == "CleanObject":
        return f"clean the {object_type}."
    if action == "SliceObject":
        return f"slice the {object_type}."
    if action == "DropHandObject":
        return f"drop the held {object_type}."
    if action == "PushObject":
        return f"push the {object_type}."
    if action == "PullObject":
        return f"pull the {object_type}."
    if action == "MoveHeldObject":
        return f"move the held {object_type}."
    if action == "BreakObject":
        return f"break the {object_type}."
    if action == "CookObject":
        return f"cook the {object_type}."
    if action == "FillObjectWithLiquid":
        return f"fill the {object_type}."
    if action == "PutObject" and isinstance(target_type, str) and target_type:
        return f"put the {object_type} on the {target_type}."
    return None


def _normalize_type_name(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower()) if isinstance(value, str) else ""


def _normalizer_target_may_be_openable(object_type: Any) -> bool:
    return _normalize_type_name(object_type) in OPENABLE_RECEPTACLE_TYPE_NAMES


def _object_type_lookup(object_types: list[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for object_type in object_types:
        key = _normalize_type_name(object_type)
        if key:
            lookup[key] = object_type
    return lookup


def _type_name_words(value: str) -> list[str]:
    separated = re.sub(r"[_-]+", " ", value)
    return [
        word.lower()
        for word in re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|[0-9]+", separated)
    ]


def _pluralize_type_word(word: str) -> str:
    if word.endswith("fe") and len(word) > 2:
        return f"{word[:-2]}ves"
    if word.endswith("f") and len(word) > 1:
        return f"{word[:-1]}ves"
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return f"{word[:-1]}ies"
    if word.endswith(("s", "x", "z", "ch", "sh", "o")):
        return f"{word}es"
    return f"{word}s"


def _plural_type_key(object_type: str) -> str:
    words = _type_name_words(object_type)
    if not words:
        return ""
    words[-1] = _pluralize_type_word(words[-1])
    return _normalize_type_name(" ".join(words))


def resolve_object_type(value: Any, object_types: list[str]) -> dict[str, Any]:
    """Resolve a task-facing name against the receiver's current object ontology."""

    resolution: dict[str, Any] = {
        "input": value if isinstance(value, str) else None,
        "canonical": None,
        "method": "unresolved",
        "candidates": [],
    }
    if not isinstance(value, str) or not value.strip():
        return resolution

    key = _normalize_type_name(value)
    exact_candidates = list(dict.fromkeys(
        object_type for object_type in object_types if _normalize_type_name(object_type) == key
    ))
    if len(exact_candidates) == 1:
        resolution.update(canonical=exact_candidates[0], method="exact", candidates=exact_candidates)
        return resolution
    if len(exact_candidates) > 1:
        resolution.update(method="ambiguous", candidates=exact_candidates)
        return resolution

    plural_candidates = list(dict.fromkeys(
        object_type for object_type in object_types if _plural_type_key(object_type) == key
    ))
    if len(plural_candidates) == 1:
        resolution.update(canonical=plural_candidates[0], method="plural", candidates=plural_candidates)
        return resolution
    if len(plural_candidates) > 1:
        resolution.update(method="ambiguous", candidates=plural_candidates)
        return resolution

    available_lookup = _object_type_lookup(object_types)
    alias_candidates = list(dict.fromkeys(
        available_lookup[target_key]
        for target in MAPTHOR_OBJECT_TYPE_ALIASES.get(key, ())
        if (target_key := _normalize_type_name(target)) in available_lookup
    ))
    if len(alias_candidates) == 1:
        resolution.update(canonical=alias_candidates[0], method="alias", candidates=alias_candidates)
    elif len(alias_candidates) > 1:
        resolution.update(method="ambiguous", candidates=alias_candidates)
    return resolution


def _semantic_receptacle_candidates(value: str, state: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    objects = [item for item in state.get("objects") or [] if isinstance(item, dict)]
    object_types = extract_state_object_types(state)
    resolution = resolve_object_type(value, object_types)
    accepted_types = {
        str(candidate)
        for candidate in ([resolution["canonical"]] if resolution.get("canonical") else resolution.get("candidates") or [])
        if candidate
    }
    candidates: list[dict[str, Any]] = []
    for item in objects:
        if not bool(item.get("receptacle")):
            continue
        object_type = _object_type_from_object(item)
        object_id = item.get("objectId") or item.get("object_id") or item.get("id")
        if not object_type or not object_id:
            continue
        if object_type not in accepted_types:
            continue
        candidates.append({
            "objectId": str(object_id),
            "objectType": object_type,
            "visible": bool(item.get("visible")),
            "distance": item.get("distance"),
            "position": item.get("position"),
        })
    return resolution, candidates


def resolve_semantic_receptacle(
    value: str,
    state: dict[str, Any],
    *,
    route_planner: Any | None = None,
) -> dict[str, Any]:
    resolution, candidates = _semantic_receptacle_candidates(value, state)
    evaluated: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        if route_planner is not None:
            try:
                route = route_planner(candidate["objectId"])
            except Exception as exc:
                route = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            status = str(route.get("status") or "failed") if isinstance(route, dict) else "failed"
            actions = route.get("actions") if isinstance(route, dict) else []
            item.update(route_status=status, route_action_count=len(actions or []), route_error=route.get("error") if isinstance(route, dict) else None)
        else:
            item.update(route_status="not_checked", route_action_count=0, route_error=None)
        evaluated.append(item)
    reachable = [item for item in evaluated if item["route_status"] in {"success", "not_checked"}]
    if not reachable:
        return {
            "semantic_input": value,
            "status": "unresolved",
            "type_resolution": resolution,
            "candidates": evaluated,
            "chosen_type": None,
            "chosen_object_id": None,
            "selection_reason": "no_reachable_receptacle_candidate",
        }
    def rank(item: dict[str, Any]) -> tuple[int, float, str]:
        action_count = int(item.get("route_action_count") or 0)
        try:
            distance = float(item.get("distance"))
        except (TypeError, ValueError):
            distance = float("inf")
        return (action_count, distance, str(item.get("objectId") or ""))
    selected = min(reachable, key=rank)
    return {
        "semantic_input": value,
        "status": "resolved",
        "type_resolution": resolution,
        "candidates": evaluated,
        "chosen_type": selected["objectType"],
        "chosen_object_id": selected["objectId"],
        "selection_reason": "reachable_min_actions_then_distance_then_object_id",
    }


def _record_type_resolution(
    type_resolutions: list[dict[str, Any]] | None,
    resolution: dict[str, Any],
) -> None:
    if type_resolutions is not None and resolution not in type_resolutions:
        type_resolutions.append(resolution)


def _extract_requested_action(text: str) -> str | None:
    lowered = text.lower()
    for phrases, action_name in ACTION_INTENT_PATTERNS:
        if any(phrase in lowered for phrase in phrases):
            return action_name
    return None


def _extract_requested_object_type(
    text: str,
    object_types: list[str],
    type_resolutions: list[dict[str, Any]] | None = None,
) -> str | None:
    words = re.findall(r"[a-z0-9]+", text.lower())
    for start in range(len(words)):
        for end in range(min(len(words), start + 3), start, -1):
            resolution = resolve_object_type(" ".join(words[start:end]), object_types)
            if resolution["canonical"] is not None:
                _record_type_resolution(type_resolutions, resolution)
                return str(resolution["canonical"])
            if resolution["method"] == "ambiguous":
                _record_type_resolution(type_resolutions, resolution)
    return None


def _split_task_clauses(task: str) -> list[str]:
    normalized = re.sub(r"\b(?:and then|then)\b", ",", task, flags=re.IGNORECASE)
    normalized = re.sub(r"\band\b", ",", normalized, flags=re.IGNORECASE)
    return [part.strip(" .") for part in re.split(r"[,;]", normalized) if part.strip(" .")]


def recognized_intent_steps_for_task(
    task: str,
    object_types: list[str],
    type_resolutions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    last_object_type: str | None = None
    for clause in _split_task_clauses(task):
        action = _extract_requested_action(clause)
        if action is None:
            continue
        target_type = None
        if action in NO_ARG_NORMALIZED_ACTIONS:
            object_type = None
        elif action == "PutObject":
            target_match = re.search(r"\b(?:on|onto|in|into|inside|to)\b\s+(.+)$", clause, flags=re.IGNORECASE)
            if target_match:
                target_type = _extract_requested_object_type(
                    target_match.group(1), object_types, type_resolutions
                )
            pronoun_object = re.search(
                r"\bput\s+(?:it|them|this|that|the\s+held\s+object|held\s+object)\b",
                clause,
                flags=re.IGNORECASE,
            )
            object_type = last_object_type if pronoun_object else _extract_requested_object_type(
                clause, object_types, type_resolutions
            )
            if object_type == target_type and last_object_type:
                object_type = last_object_type
        else:
            object_type = _extract_requested_object_type(clause, object_types, type_resolutions)
        steps.append({"order": len(steps) + 1, "action": action, "objectType": object_type, "targetType": target_type})
        if action in {"PickupObject", "PutObject"} and isinstance(object_type, str) and object_type:
            last_object_type = object_type
    return steps



def compact_reference_intent_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not steps:
        return []

    def step_key(step: dict[str, Any]) -> tuple[Any, Any, Any]:
        return (step.get("action"), step.get("objectType"), step.get("targetType"))

    keys = [step_key(step) for step in steps]
    for size in range(1, len(keys) // 2 + 1):
        if len(keys) % size == 0 and keys == keys[:size] * (len(keys) // size):
            return [dict(step, order=index + 1) for index, step in enumerate(steps[:size])]

    compacted: list[dict[str, Any]] = []
    last_key: tuple[Any, Any, Any] | None = None
    for step in steps:
        current_key = step_key(step)
        if current_key == last_key:
            continue
        compacted.append(step)
        last_key = current_key
    return [dict(step, order=index + 1) for index, step in enumerate(compacted)]


def _normalized_step(
    action: str,
    object_type: str | None,
    target_type: str | None = None,
    *,
    object_id: str | None = None,
    target_object_id: str | None = None,
    action_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    step = {"order": 0, "action": action, "objectType": object_type, "targetType": target_type}
    if object_id:
        step["objectId"] = object_id
    if target_object_id:
        step["targetObjectId"] = target_object_id
    if action_args:
        step["actionArgs"] = deepcopy(action_args)
    return step


def _is_source_placement_task(task: str) -> bool:
    return bool(re.search(r"\b(?:from|out\s+of|off(?:\s+of)?)\b", task, flags=re.IGNORECASE))


def _is_same_object_type(left: Any, right: Any) -> bool:
    return bool(_normalize_type_name(left) and _normalize_type_name(left) == _normalize_type_name(right))


def _put_helper_steps_are_repairable(
    steps: list[dict[str, Any]],
    put_object: str,
    put_target: str,
) -> bool:
    target_is_openable = _normalizer_target_may_be_openable(put_target)
    for step in steps:
        action = step.get("action")
        object_type = step.get("objectType")
        target_type = step.get("targetType")
        if action == "PutObject":
            if not (_is_same_object_type(object_type, put_object) and _is_same_object_type(target_type, put_target)):
                return False
            continue
        if action == "GotoObject":
            if not (_is_same_object_type(object_type, put_object) or _is_same_object_type(object_type, put_target)):
                return False
            if target_type is None or _is_same_object_type(target_type, put_target):
                continue
            return False
        if action == "PickupObject" and target_type is None and _is_same_object_type(object_type, put_object):
            continue
        if action in {"OpenObject", "CloseObject"} and target_type is None:
            if target_is_openable and _is_same_object_type(object_type, put_target):
                continue
            return False
        return False
    return True


def _contains_step(steps: list[dict[str, Any]], action: str, object_type: str | None, target_type: str | None = None) -> bool:
    return any(
        step.get("action") == action
        and _is_same_object_type(step.get("objectType"), object_type)
        and (target_type is None or _is_same_object_type(step.get("targetType"), target_type))
        for step in steps
    )


def repair_destination_put_intent_order(
    original_task: str,
    object_types: list[str],
    normalized: dict[str, Any],
) -> list[str]:
    reference_steps = compact_reference_intent_steps(recognized_intent_steps_for_task(original_task, object_types))
    if len(reference_steps) != 1 or reference_steps[0].get("action") != "PutObject":
        return []
    if _is_source_placement_task(original_task):
        return []

    put_object = reference_steps[0].get("objectType")
    put_target = reference_steps[0].get("targetType")
    if not isinstance(put_object, str) or not put_object.strip():
        return []
    if not isinstance(put_target, str) or not put_target.strip():
        return []

    steps = normalized.get("intentSteps") if isinstance(normalized.get("intentSteps"), list) else []
    if not steps or not all(isinstance(step, dict) for step in steps):
        return []
    if sum(1 for step in steps if step.get("action") == "PutObject") != 1:
        return []
    if not _put_helper_steps_are_repairable(steps, put_object, put_target):
        return []

    target_is_openable = _normalizer_target_may_be_openable(put_target)
    canonical_steps = [
        _normalized_step("GotoObject", put_object),
        _normalized_step("PickupObject", put_object),
        _normalized_step("GotoObject", put_target),
    ]
    if target_is_openable:
        canonical_steps.append(_normalized_step("OpenObject", put_target))
    canonical_steps.append(_normalized_step("PutObject", put_object, put_target))
    if target_is_openable:
        canonical_steps.append(_normalized_step("CloseObject", put_target))

    for order, step in enumerate(canonical_steps, start=1):
        step["order"] = order

    if steps == canonical_steps:
        return []

    repairs: list[str] = []
    if not _contains_step(steps, "GotoObject", put_object):
        repairs.append(f"inserted GotoObject({put_object}) before PickupObject for destination placement")
    if not _contains_step(steps, "PickupObject", put_object):
        repairs.append(f"inserted PickupObject({put_object}) before PutObject for destination placement")
    if not _contains_step(steps, "GotoObject", put_target):
        repairs.append(f"inserted GotoObject({put_target}) before PutObject for destination placement")
    if target_is_openable and not _contains_step(steps, "OpenObject", put_target):
        repairs.append(f"inserted OpenObject({put_target}) before PutObject for openable target")
    if target_is_openable and not _contains_step(steps, "CloseObject", put_target):
        repairs.append(f"inserted CloseObject({put_target}) after PutObject for Map-THOR placement completion")
    if not repairs:
        repairs.append(f"reordered PutObject helpers into destination placement order for {put_object}->{put_target}")

    normalized["intentSteps"] = canonical_steps
    normalized["requestedAction"] = canonical_steps[0]["action"]
    normalized["requestedObjectType"] = put_object
    normalized["requestedTargetType"] = put_target
    normalized["action"] = canonical_steps[0]["action"]
    normalized["object_type"] = put_object
    normalized["target_type"] = put_target
    task_intent = normalized.get("task_intent")
    if isinstance(task_intent, dict):
        task_intent["requestedAction"] = canonical_steps[0]["action"]
        task_intent["requestedObjectType"] = put_object
        task_intent["requestedTargetType"] = put_target
        task_intent["intentSteps"] = canonical_steps
    normalized["repairs"] = [*normalized.get("repairs", []), *repairs] if isinstance(normalized.get("repairs"), list) else repairs
    return repairs


def _normalizer_extra_step_is_allowed(
    extra_step: dict[str, Any],
    reference_steps: list[dict[str, Any]],
    original_task: str,
) -> bool:
    action = extra_step.get("action")
    object_type = extra_step.get("objectType")
    target_type = extra_step.get("targetType")
    for reference_step in reference_steps:
        if reference_step.get("action") == "PickupObject":
            pickup_object = reference_step.get("objectType")
            if action == "GotoObject" and object_type == pickup_object and target_type is None:
                return True
            continue
        if reference_step.get("action") != "PutObject":
            continue
        put_object = reference_step.get("objectType")
        put_target = reference_step.get("targetType")
        if action == "GotoObject" and object_type in {put_object, put_target} and target_type is None:
            return True
        if action == "PickupObject" and object_type == put_object and target_type is None:
            return True
        if (
            action == "OpenObject"
            and object_type == put_target
            and target_type is None
            and _normalizer_target_may_be_openable(put_target)
        ):
            return True
        if (
            action == "CloseObject"
            and object_type == put_target
            and target_type is None
            and _normalizer_target_may_be_openable(put_target)
        ):
            return True
    return False


def action_sequence_conflict_warnings(original_task: str, object_types: list[str], normalized: dict[str, Any]) -> list[str]:
    reference_steps = compact_reference_intent_steps(recognized_intent_steps_for_task(original_task, object_types))
    if not reference_steps:
        return []
    normalized_steps = normalized.get("intentSteps") if isinstance(normalized.get("intentSteps"), list) else []
    warnings: list[str] = []
    reference_actions = [step.get("action") for step in reference_steps]
    normalized_actions = [step.get("action") for step in normalized_steps if isinstance(step, dict)]
    search_start = 0
    matched_indexes: list[int] = []
    for expected in reference_actions:
        try:
            match_index = normalized_actions.index(expected, search_start)
        except ValueError:
            replacement = normalized_actions[search_start] if search_start < len(normalized_actions) else None
            warnings.append(f"normalizer changed recognized action {expected} to {replacement}")
            return warnings
        matched_indexes.append(match_index)
        search_start = match_index + 1

    matched_index_set = set(matched_indexes)
    extra_steps = [
        step
        for index, step in enumerate(normalized_steps)
        if isinstance(step, dict) and index not in matched_index_set
    ]
    rejected_extras = [
        step.get("action")
        for step in extra_steps
        if not _normalizer_extra_step_is_allowed(step, reference_steps, original_task)
    ]
    if rejected_extras:
        warnings.append(f"normalizer added unrequested action(s) {rejected_extras}")
        return warnings

    for reference_step, match_index in zip(reference_steps, matched_indexes):
        if match_index >= len(normalized_steps) or not isinstance(normalized_steps[match_index], dict):
            continue
        expected_object = reference_step.get("objectType")
        actual_object = normalized_steps[match_index].get("objectType")
        if expected_object is not None and actual_object != expected_object:
            warnings.append(f"normalizer changed recognized object {expected_object} to {actual_object}")
        expected_target = reference_step.get("targetType")
        actual_target = normalized_steps[match_index].get("targetType")
        if expected_target is not None and actual_target != expected_target:
            warnings.append(f"normalizer changed recognized target {expected_target} to {actual_target}")
    return warnings

def _canonical_object_type_value(
    value: Any,
    object_types: list[str],
    type_resolutions: list[dict[str, Any]] | None = None,
) -> Any:
    if not isinstance(value, str):
        return value
    resolution = resolve_object_type(value, object_types)
    _record_type_resolution(type_resolutions, resolution)

    return resolution["canonical"] if resolution["canonical"] is not None else value

def _object_type_is_valid(value: Any, object_types: list[str]) -> bool:
    return value is None or (isinstance(value, str) and value in set(object_types))


def _normalize_intent_steps(
    arguments: dict[str, Any],
    action: Any,
    object_type: Any,
    target_type: Any,
    object_types: list[str],
    type_resolutions: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    raw_steps = _first_present(arguments, "intentSteps", "intent_steps")
    steps: list[dict[str, Any]] = []
    if isinstance(raw_steps, list):
        for index, raw_step in enumerate(raw_steps, start=1):
            if not isinstance(raw_step, dict):
                warnings.append(f"intentSteps[{index - 1}] must be an object")
                continue
            step_action = _first_present(raw_step, "action", "requestedAction")
            step_object = _canonical_object_type_value(
                _first_present(raw_step, "objectType", "object_type", "requestedObjectType"),
                object_types,
                type_resolutions,
            )
            step_target = _canonical_object_type_value(
                _first_present(raw_step, "targetType", "target_type", "requestedTargetType"),
                object_types,
                type_resolutions,
            )
            order = raw_step.get("order", index)
            if not isinstance(order, int) or isinstance(order, bool) or order <= 0:
                warnings.append(f"intentSteps[{index - 1}] returned invalid order {order!r}; using {index}")
                order = index
            if step_action not in SUPPORTED_NORMALIZED_ACTIONS:
                warnings.append(f"intentSteps[{index - 1}] returned unsupported action {step_action!r}")
            for field, value in (("objectType", step_object), ("targetType", step_target)):
                if not _object_type_is_valid(value, object_types):
                    warnings.append(f"intentSteps[{index - 1}] returned {field} not present in receiver state: {value!r}")
            steps.append({"order": order, "action": step_action, "objectType": step_object, "targetType": step_target})
    elif raw_steps is not None:
        warnings.append("intentSteps must be a list")

    if not steps and action in SUPPORTED_NORMALIZED_ACTIONS:
        steps.append({"order": 1, "action": action, "objectType": object_type, "targetType": target_type})
    if not steps:
        warnings.append("normalizer omitted usable intentSteps")
    return steps, warnings


def validate_task_normalization(arguments: dict[str, Any], object_types: list[str]) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    type_resolutions: list[dict[str, Any]] = []
    normalized_task = _first_present(arguments, "normalized_task", "normalizedTask", "task")
    requested_action = _first_present(arguments, "requestedAction", "requested_action", "action")
    requested_object = _canonical_object_type_value(
        _first_present(arguments, "requestedObjectType", "requested_object_type", "object_type", "objectType"),
        object_types,
        type_resolutions,
    )
    requested_target = _canonical_object_type_value(
        _first_present(arguments, "requestedTargetType", "requested_target_type", "target_type", "targetType"),
        object_types,
        type_resolutions,
    )
    confidence_value = arguments.get("confidence")
    reason = arguments.get("reason")
    if not isinstance(normalized_task, str) or not normalized_task.strip():
        warnings.append("normalizer omitted non-empty normalized_task")
        normalized_task = ""
    confidence, confidence_warning = _normalize_confidence(confidence_value)
    if confidence_warning:
        warnings.append(confidence_warning)

    steps, step_warnings = _normalize_intent_steps(
        arguments,
        requested_action,
        requested_object,
        requested_target,
        object_types,
        type_resolutions,
    )
    warnings.extend(step_warnings)
    primary_step = next((step for step in steps if step.get("action") in SUPPORTED_NORMALIZED_ACTIONS), None)
    target_step = next((step for step in steps if step.get("targetType") is not None), None)
    if primary_step is not None:
        if requested_action not in SUPPORTED_NORMALIZED_ACTIONS:
            requested_action = primary_step.get("action")
        if requested_object is None:
            requested_object = primary_step.get("objectType")
        if requested_target is None:
            requested_target = primary_step.get("targetType")
    if requested_target is None and target_step is not None:
        requested_target = target_step.get("targetType")
    if requested_action not in SUPPORTED_NORMALIZED_ACTIONS:
        warnings.append(f"normalizer returned unsupported action {requested_action!r}")
    for field, value in (("requestedObjectType", requested_object), ("requestedTargetType", requested_target)):
        if not _object_type_is_valid(value, object_types):
            warnings.append(f"normalizer returned {field} not present in receiver state: {value!r}")

    for step in steps:
        if step.get("action") in NO_ARG_NORMALIZED_ACTIONS:
            if step.get("objectType") is not None:
                warnings.append(f"no-argument action {step.get('action')} must not include objectType {step.get('objectType')!r}")
            if step.get("targetType") is not None:
                warnings.append(f"no-argument action {step.get('action')} must not include targetType {step.get('targetType')!r}")
    canonical_task = _canonical_task_from_normalized(requested_action, requested_object, requested_target)
    if canonical_task and (not normalized_task or normalized_task.strip() == requested_action):
        normalized_task = canonical_task
    task_intent = {
        "requestedAction": requested_action,
        "requestedObjectType": requested_object,
        "requestedTargetType": requested_target,
        "intentSteps": steps,
    }
    normalized = {
        "normalized_task": normalized_task.strip() if isinstance(normalized_task, str) else "",
        "action": requested_action,
        "object_type": requested_object,
        "target_type": requested_target,
        "requestedAction": requested_action,
        "requestedObjectType": requested_object,
        "requestedTargetType": requested_target,
        "intentSteps": steps,
        "task_intent": task_intent,
        "confidence": confidence,
        "reason": reason if isinstance(reason, str) else "",
        "type_resolutions": type_resolutions,
    }
    return normalized, warnings

def _generate_normalizer_tool_call(
    backend: Any,
    task: str,
    object_types: list[str],
    feedback: list[str] | None = None,
    *,
    original_task: str | None = None,
    subtask_context: dict[str, Any] | None = None,
    action_coverage_text: str | None = None,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    config = getattr(backend, "config", None)
    original_max_new_tokens = getattr(config, "max_new_tokens", None)
    if isinstance(original_max_new_tokens, int) and original_max_new_tokens < 512:
        setattr(config, "max_new_tokens", 512)
    try:
        output = backend.generate_with_tools(
            task_normalizer_messages(
                task,
                object_types,
                feedback,
                original_task=original_task,
                subtask_context=subtask_context,
                action_coverage_text=action_coverage_text,
            ),
            [TASK_NORMALIZER_TOOL_SCHEMA],
        )
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"
    finally:
        if isinstance(original_max_new_tokens, int):
            setattr(config, "max_new_tokens", original_max_new_tokens)
    preview = _preview_text(output)
    try:
        tool_call = parse_task_normalizer_tool_call(str(output).strip())
    except Exception as exc:
        return None, preview, f"{type(exc).__name__}: {exc}"
    if tool_call.get("name") != TASK_NORMALIZER_TOOL_NAME:
        return tool_call, preview, f"normalizer called unexpected tool {tool_call.get('name')!r}"
    return tool_call, preview, None


def _normalization_failure_reason(task_normalization: dict[str, Any]) -> str:
    warnings = task_normalization.get("warnings")
    if isinstance(warnings, list) and warnings:
        return "; ".join(str(warning) for warning in warnings)
    confidence = task_normalization.get("confidence")
    reason = task_normalization.get("reason")
    pieces: list[str] = []
    if confidence not in {"high", "medium"}:
        pieces.append(f"normalizer confidence is {confidence!r}")
    if isinstance(reason, str) and reason.strip():
        pieces.append(reason.strip())
    if pieces:
        return "; ".join(pieces)
    return "task normalizer did not produce a usable structured intent"


def task_normalization_failure_response(task_id: str, dry_run: bool, task_normalization: dict[str, Any]) -> dict[str, Any]:
    reason = _normalization_failure_reason(task_normalization)
    return {
        "status": "needs_upstream_planning",
        "task_id": task_id,
        "dry_run": dry_run,
        "failure_code": "task_normalization_failed",
        "reason": reason,
        "result": {
            "task_id": task_id,
            "closed_loop_result": {
                "status": "needs_upstream_planning",
                "failure_code": "task_normalization_failed",
                "reason": reason,
            },
        },
        "task_normalization": task_normalization,
    }


def semantic_target_failure_response(task_id: str, dry_run: bool, resolution: dict[str, Any]) -> dict[str, Any]:
    reason = f"semantic target {resolution.get('semantic_input')!r} has no reachable receptacle candidate"
    return {
        "status": "needs_upstream_planning",
        "task_id": task_id,
        "dry_run": dry_run,
        "failure_code": "semantic_target_unresolved",
        "reason": reason,
        "result": {
            "task_id": task_id,
            "closed_loop_result": {
                "status": "needs_upstream_planning",
                "failure_code": "semantic_target_unresolved",
                "reason": reason,
            },
        },
        "task_normalization": {"semantic_resolution": resolution, "warnings": [reason]},
    }


def normalize_incoming_task_with_backend(
    backend: Any,
    task: str,
    object_types: list[str],
    *,
    primary_task_text: str | None = None,
    subtask_context: dict[str, Any] | None = None,
    used_subtask_name: bool = False,
    action_coverage_text: str | None = None,
) -> dict[str, Any]:
    primary_task = primary_task_text.strip() if isinstance(primary_task_text, str) and primary_task_text.strip() else task
    coverage_task = (
        action_coverage_text.strip()
        if isinstance(action_coverage_text, str) and action_coverage_text.strip()
        else primary_task
    )
    context = subtask_context if isinstance(subtask_context, dict) else {}
    record: dict[str, Any] = {
        "original_task": task,
        "primary_task_text": primary_task,
        "used_subtask_name": bool(used_subtask_name),
        "action_coverage_text": coverage_task,
        "subtask_context": context,
        "normalized_task": task,
        "used": False,
        "available_object_types": object_types,
        "warnings": [],
    }
    if not hasattr(backend, "generate_with_tools"):
        record["warnings"].append("backend does not support tool-based task normalization")
        return record

    tool_call, preview, error = _generate_normalizer_tool_call(
        backend,
        primary_task,
        object_types,
        original_task=task,
        subtask_context=context,
        action_coverage_text=coverage_task,
    )
    if preview:
        record["raw_tool_output_preview"] = preview

    normalized: dict[str, Any] | None = None
    warnings: list[str] = []
    source = "qwen_tool_call"
    if error:
        warnings.append(f"task normalization failed: {error}")
    elif tool_call is None:
        warnings.append("task normalization failed: model returned no tool call")
    else:
        arguments = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), dict) else {}
        normalized, warnings = validate_task_normalization(arguments, object_types)
        repair_destination_put_intent_order(coverage_task, object_types, normalized)
        warnings.extend(action_sequence_conflict_warnings(coverage_task, object_types, normalized))

    if warnings:
        retry_call, retry_preview, retry_error = _generate_normalizer_tool_call(
            backend,
            primary_task,
            object_types,
            warnings,
            original_task=task,
            subtask_context=context,
            action_coverage_text=coverage_task,
        )
        if retry_preview:
            record["raw_retry_tool_output_preview"] = retry_preview
        if retry_error:
            warnings.append(f"tool retry failed: {retry_error}")
        elif retry_call is None:
            warnings.append("tool retry failed: model returned no tool call")
        else:
            retry_arguments = retry_call.get("arguments") if isinstance(retry_call.get("arguments"), dict) else {}
            retry_normalized, retry_warnings = validate_task_normalization(retry_arguments, object_types)
            repair_destination_put_intent_order(coverage_task, object_types, retry_normalized)
            retry_warnings.extend(action_sequence_conflict_warnings(coverage_task, object_types, retry_normalized))
            if not retry_warnings:
                normalized, warnings = retry_normalized, []
                source = "qwen_tool_call_retry"
            else:
                warnings.extend(f"tool retry: {warning}" for warning in retry_warnings)
    if normalized is None:
        normalized, default_warnings = validate_task_normalization({}, object_types)
        warnings.extend(default_warnings)
    record.update(normalized)
    record["source"] = source
    record["warnings"].extend(warnings)
    if not warnings and normalized.get("confidence") in {"high", "medium"}:
        record["normalized_task"] = normalized["normalized_task"]
        record["used"] = normalized["normalized_task"] != task
    else:
        record["candidate_normalized_task"] = normalized.get("normalized_task") or ""
        record["normalized_task"] = task
        record["used"] = False
    return record


def normalizer_subtask_inputs(subtask: Any, task: str) -> tuple[str, bool, dict[str, Any], str]:
    """Return the primary text and structured context used by the task normalizer."""

    if not isinstance(subtask, dict):
        return task, False, {}, task
    name = subtask.get("name")
    description = subtask.get("description")
    primary_task = name.strip() if isinstance(name, str) and name.strip() else task
    coverage_parts = [primary_task]
    if isinstance(description, str) and description.strip() and description.strip() not in coverage_parts:
        coverage_parts.append(description.strip())
    action_coverage_text = " ".join(coverage_parts).strip() or primary_task
    grounding = subtask.get("grounding") if isinstance(subtask.get("grounding"), dict) else {}
    context: dict[str, Any] = {}
    for key in ("id", "description", "action", "termination_check"):
        value = subtask.get(key)
        if value not in (None, ""):
            context[key] = value
    if grounding:
        context["grounding"] = {
            key: grounding.get(key)
            for key in ("object_tags", "source_object_tags", "destination_object_tags",
                        "source_object_ids", "destination_object_ids", "relation_texts",
                        "status", "missing_reason", "recovery")
            if grounding.get(key) not in (None, "", [])
        }
    return primary_task, primary_task != task, context, action_coverage_text


def prepare_semantic_destination(
    subtask: Any,
    state: dict[str, Any] | None,
    *,
    route_planner: Any | None = None,
) -> tuple[Any, dict[str, Any] | None]:
    if not isinstance(subtask, dict) or not isinstance(state, dict):
        return subtask, None
    augmented = deepcopy(subtask)
    grounding = augmented.get("grounding") if isinstance(augmented.get("grounding"), dict) else {}
    destination_tags = [str(tag) for tag in grounding.get("destination_object_tags") or [] if str(tag).strip()]
    if not destination_tags:
        return augmented, None
    bound_ids = [str(value) for value in grounding.get("destination_object_ids") or [] if str(value).strip()]
    bound_object = next(
        (item for item in state.get("objects") or [] if isinstance(item, dict)
         and str(item.get("objectId") or item.get("id") or "") == (bound_ids[0] if bound_ids else "")
         and bool(item.get("receptacle"))),
        None,
    )
    resolution = {
        "semantic_input": destination_tags[0],
        "status": "resolved",
        "type_resolution": resolve_object_type(destination_tags[0], extract_state_object_types(state)),
        "candidates": [],
        "chosen_type": _object_type_from_object(bound_object),
        "chosen_object_id": bound_ids[0],
        "selection_reason": "existing_episode_binding",
    } if bound_object is not None else resolve_semantic_receptacle(destination_tags[0], state, route_planner=route_planner)
    if resolution.get("status") != "resolved":
        return augmented, resolution
    source_tags = [str(tag) for tag in grounding.get("source_object_tags") or [] if str(tag).strip()]
    grounding["semantic_destination_tags"] = destination_tags
    grounding["destination_object_tags"] = [str(resolution["chosen_type"])]
    grounding["destination_object_ids"] = [str(resolution["chosen_object_id"])]
    grounding["object_tags"] = list(dict.fromkeys([*source_tags, str(resolution["chosen_type"])]))
    augmented["grounding"] = grounding
    return augmented, resolution


def should_prepare_semantic_destination(subtask: dict[str, Any] | None) -> bool:
    if not isinstance(subtask, dict):
        return False
    action = str(subtask.get("action") or "").strip().lower()
    return action in {"place", "put"}


def normalize_structured_subtask(
    subtask: Any,
    task: str,
    object_types: list[str],
    *,
    primary_task_text: str | None = None,
    subtask_context: dict[str, Any] | None = None,
    used_subtask_name: bool = False,
    action_coverage_text: str | None = None,
) -> dict[str, Any] | None:
    """Deterministically normalize supported upstream structured task intents."""

    if not isinstance(subtask, dict):
        return None
    action = str(subtask.get("action") or "").strip().lower()
    primary_task = primary_task_text.strip() if isinstance(primary_task_text, str) and primary_task_text.strip() else task
    coverage_task = (
        action_coverage_text.strip()
        if isinstance(action_coverage_text, str) and action_coverage_text.strip()
        else primary_task
    )
    type_resolutions: list[dict[str, Any]] = []
    reference_steps = compact_reference_intent_steps(
        recognized_intent_steps_for_task(coverage_task, object_types, type_resolutions)
    )
    action_kind = None
    if action in {"find", "inspect", "search", "navigate"}:
        action_kind = "search"
    elif action in {"pick", "pickup", "grab", "take"}:
        action_kind = "pick"
    elif action in {"place", "put"}:
        action_kind = "place"
    elif action in {"open"}:
        action_kind = "open"
    elif action in {"close", "shut"}:
        action_kind = "close"
    elif action in {"slice", "cut"}:
        action_kind = "slice"
    elif action in {"toggle_on", "turn_on", "switch_on", "toggleobjecton"}:
        action_kind = "toggle_on"
    elif action in {"toggle_off", "turn_off", "switch_off", "toggleobjectoff"}:
        action_kind = "toggle_off"
    elif action in {"clean", "wash", "cleanobject"}:
        action_kind = "clean"
    elif action in {"drop", "drophandobject"}:
        action_kind = "drop"
    elif action in {"push", "pushobject"}:
        action_kind = "push"
    elif action in {"pull", "pullobject"}:
        action_kind = "pull"
    elif action in {"move_held", "moveheldobject"}:
        action_kind = "move_held"
    elif action in {"break", "breakobject"}:
        action_kind = "break"
    elif action in {"cook", "cookobject"}:
        action_kind = "cook"
    elif action in {"fill", "fillobjectwithliquid"}:
        action_kind = "fill"
    elif action in {"", "other"} and reference_steps:
        reference_actions = {step.get("action") for step in reference_steps}
        if (
            "PutObject" in reference_actions
            and reference_actions
            <= {"GotoObject", "PickupObject", "OpenObject", "PutObject", "CloseObject"}
        ):
            action_kind = "place"
        elif "PickupObject" in reference_actions and reference_actions <= {"GotoObject", "PickupObject"}:
            action_kind = "pick"
        elif reference_actions and reference_actions <= {"GotoObject"}:
            action_kind = "search"
        elif reference_actions and reference_actions <= {"GotoObject", "OpenObject"}:
            action_kind = "open"
        elif reference_actions and reference_actions <= {"GotoObject", "CloseObject"}:
            action_kind = "close"
        elif "SliceObject" in reference_actions and reference_actions <= {"GotoObject", "SliceObject"}:
            action_kind = "slice"
        elif "ToggleObjectOn" in reference_actions and reference_actions <= {"GotoObject", "ToggleObjectOn"}:
            action_kind = "toggle_on"
        elif "ToggleObjectOff" in reference_actions and reference_actions <= {"GotoObject", "ToggleObjectOff"}:
            action_kind = "toggle_off"
        elif "CleanObject" in reference_actions and reference_actions <= {"GotoObject", "CleanObject"}:
            action_kind = "clean"
    if action_kind is None:
        return None
    action_args, action_arg_errors = validate_action_args(
        action_kind, subtask.get("action_args")
    )
    if action_arg_errors:
        return None

    grounding = subtask.get("grounding") if isinstance(subtask.get("grounding"), dict) else {}
    source_ids = [str(value) for value in grounding.get("source_object_ids") or [] if str(value).strip()]
    destination_ids = [str(value) for value in grounding.get("destination_object_ids") or [] if str(value).strip()]
    source_object_id = source_ids[0] if source_ids else None
    target_object_id = destination_ids[0] if destination_ids else None
    if action_kind == "search" and source_object_id is None and target_object_id is not None:
        source_object_id = target_object_id
    tags = [
        str(tag).strip()
        for tag in grounding.get("object_tags") or []
        if isinstance(tag, str) and tag.strip()
    ]
    resolved_tags: list[str] = []
    for tag in tags:
        resolution = resolve_object_type(tag, object_types)
        _record_type_resolution(type_resolutions, resolution)
        resolved = resolution["canonical"]
        if resolved is not None and resolved not in resolved_tags:
            resolved_tags.append(str(resolved))

    resolved_source_tags: list[str] = []
    resolved_destination_tags: list[str] = []
    for role_values, role_output in (
        (grounding.get("source_object_tags") or [], resolved_source_tags),
        (grounding.get("destination_object_tags") or [], resolved_destination_tags),
    ):
        for tag in role_values:
            resolution = resolve_object_type(tag, object_types)
            _record_type_resolution(type_resolutions, resolution)
            resolved = resolution["canonical"]
            if resolved is not None and resolved not in role_output:
                role_output.append(str(resolved))

    referenced_types: list[str] = []
    for step in reference_steps:
        for key in ("objectType", "targetType"):
            value = step.get(key)
            if isinstance(value, str) and value not in referenced_types:
                referenced_types.append(value)

    source_type = next(iter(resolved_source_tags or resolved_tags or referenced_types), None)
    target_type = None
    if action_kind == "place":
        put_step = next((step for step in reference_steps if step.get("action") == "PutObject"), None)
        if isinstance(put_step, dict):
            if not resolved_source_tags:
                source_type = put_step.get("objectType") or source_type
            parsed_target = put_step.get("targetType")
            if parsed_target and not _is_same_object_type(parsed_target, source_type):
                target_type = parsed_target
        if resolved_destination_tags:
            target_type = resolved_destination_tags[0]
        if target_type is None:
            target_type = next(
                (value for value in [*resolved_tags, *referenced_types] if not _is_same_object_type(value, source_type)),
                None,
            )

    if source_type is None or (action_kind == "place" and target_type is None):
        return None

    steps = [] if action_kind in {"drop", "move_held"} else [
        _normalized_step("GotoObject", source_type, object_id=source_object_id)
    ]
    if action_kind in {"pick", "place"}:
        steps.append(_normalized_step("PickupObject", source_type, object_id=source_object_id))
    if action_kind == "open":
        steps.append(_normalized_step("OpenObject", source_type, object_id=source_object_id))
    if action_kind == "close":
        steps.append(_normalized_step("CloseObject", source_type, object_id=source_object_id))
    if action_kind == "slice":
        steps.append(_normalized_step("SliceObject", source_type, object_id=source_object_id))
    if action_kind == "toggle_on":
        steps.append(_normalized_step("ToggleObjectOn", source_type, object_id=source_object_id))
    if action_kind == "toggle_off":
        steps.append(_normalized_step("ToggleObjectOff", source_type, object_id=source_object_id))
    if action_kind == "clean":
        steps.append(_normalized_step("CleanObject", source_type, object_id=source_object_id))
    if action_kind in {"drop", "push", "pull", "move_held", "break", "cook", "fill"}:
        native_action = str(ACTION_CONTRACTS[action_kind]["native_action"])
        steps.append(_normalized_step(
            native_action,
            source_type,
            object_id=(None if action_kind in {"drop", "move_held"} else source_object_id),
            action_args=action_args,
        ))
    if action_kind == "place":
        steps.append(_normalized_step("GotoObject", target_type, object_id=target_object_id))
        if _normalizer_target_may_be_openable(target_type):
            steps.append(_normalized_step("OpenObject", target_type, object_id=target_object_id))
        steps.append(_normalized_step("PutObject", source_type, target_type, object_id=source_object_id, target_object_id=target_object_id))
        if _normalizer_target_may_be_openable(target_type):
            steps.append(_normalized_step("CloseObject", target_type, object_id=target_object_id))
    for order, step in enumerate(steps, start=1):
        step["order"] = order

    if action_kind == "search":
        normalized_task = f"go to the {source_type}."
    elif action_kind == "pick":
        normalized_task = f"go to the {source_type} and pick it up."
    elif action_kind == "open":
        normalized_task = f"go to the {source_type} and open it."
    elif action_kind == "close":
        normalized_task = f"go to the {source_type} and close it."
    elif action_kind == "slice":
        normalized_task = f"go to the {source_type} and slice it."
    elif action_kind == "toggle_on":
        normalized_task = f"go to the {source_type} and turn it on."
    elif action_kind == "toggle_off":
        normalized_task = f"go to the {source_type} and turn it off."
    elif action_kind == "clean":
        normalized_task = f"go to the {source_type} and clean it."
    elif action_kind == "drop":
        normalized_task = f"drop the held {source_type}."
    elif action_kind == "push":
        normalized_task = f"go to the {source_type} and push it."
    elif action_kind == "pull":
        normalized_task = f"go to the {source_type} and pull it."
    elif action_kind == "move_held":
        normalized_task = f"move the held {source_type}."
    elif action_kind == "break":
        normalized_task = f"go to the {source_type} and break it."
    elif action_kind == "cook":
        normalized_task = f"go to the {source_type} and cook it."
    elif action_kind == "fill":
        normalized_task = f"go to the {source_type} and fill it with {action_args['fillLiquid']}."
    else:
        normalized_task = f"put the {source_type} in the {target_type}."
    first_step = steps[0]
    task_intent = {
        "requestedAction": first_step["action"],
        "requestedObjectType": source_type,
        "requestedTargetType": target_type,
        "intentSteps": steps,
    }
    return {
        "original_task": task,
        "primary_task_text": primary_task,
        "used_subtask_name": bool(used_subtask_name),
        "action_coverage_text": coverage_task,
        "subtask_context": subtask_context if isinstance(subtask_context, dict) else {},
        "normalized_task": normalized_task,
        "used": normalized_task != task,
        "available_object_types": object_types,
        "warnings": [],
        "source": "upstream_structured_task",
        "action": first_step["action"],
        "object_type": source_type,
        "target_type": target_type,
        "requestedAction": first_step["action"],
        "requestedObjectType": source_type,
        "requestedTargetType": target_type,
        "intentSteps": steps,
        "task_intent": task_intent,
        "confidence": "high",
        "reason": (
            f"deterministically mapped upstream {action} subtask to "
            + " -> ".join(
                f"{step['action']}({step.get('objectType') or ''}"
                f"{', ' + step['targetType'] if step.get('targetType') else ''})"
                for step in steps
            )
        ),
        "type_resolutions": type_resolutions,
    }


def normalize_structured_search_subtask(*args: Any, **kwargs: Any) -> dict[str, Any] | None:
    """Backward-compatible alias for callers importing the previous helper name."""

    return normalize_structured_subtask(*args, **kwargs)


def missing_model_shards(model_path: str) -> list[str]:
    path = Path(model_path).expanduser()
    if not path.is_dir():
        return []
    index_path = path / "model.safetensors.index.json"
    if not index_path.exists():
        return []
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        return []
    missing = []
    for shard in sorted({str(value) for value in weight_map.values() if isinstance(value, str)}):
        if not (path / shard).exists():
            missing.append(shard)
    return missing


def model_shard_error(model_path: str) -> str | None:
    missing = missing_model_shards(model_path)
    if not missing:
        return None
    return (
        f"model path {model_path!r} is incomplete; missing safetensors shard(s): {', '.join(missing)}. "
        "Use a complete Qwen3.5-4B directory, e.g. /225010231/mwl/Linhao/models/Qwen3.5-4B."
    )


def _last_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append((index, index + end, value))
    if not candidates:
        raise RuntimeError("relay runtime did not produce a JSON result")
    return max(candidates, key=lambda item: (item[1] - item[0], item[1]))[2]


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class TaskExecutionRuntimeConfig:
    receiver_url: str
    model_path: str
    device: str
    device_map: str
    dtype: str
    max_new_tokens: int
    temperature: float
    send_timeout: float
    output_dir: Path
    max_replan_steps: int
    relay_agent_max_turns: int
    max_actions: int


class TaskExecutionService:
    """Runs one task at a time because one Qwen backend is shared."""

    def __init__(self, engine: Any, backend: Any, config: TaskExecutionRuntimeConfig):
        self.engine = engine
        self.backend = backend
        self.config = config
        self.lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        relay_agent_module = sys.modules.get("demo.relay_agent")
        receiver = fetch_receiver_health(self.config.receiver_url, self.config.send_timeout)
        backend_ready = self.backend is not None
        ready = backend_ready and bool(receiver.get("controller_ready"))
        return {
            "status": "ready" if ready else "not_ready",
            "service": SERVICE_NAME,
            "service_revision": SERVICE_REVISION,
            "receiver_url": self.config.receiver_url,
            "receiver": receiver,
            "model_path": self.config.model_path,
            "device": self.config.device,
            "relay_mode": True,
            "closed_loop_replan": True,
            "backend_ready": backend_ready,
            "model_loaded": bool(getattr(self.backend, "model", None)),
            "relay_agent_module": str(getattr(relay_agent_module, "__file__", "not-loaded")),
        }

    def execute_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "known_robot_ids" in payload:
            raise ValueError(
                "known_robot_ids is Coordinator-owned and must not be supplied by callers"
            )
        task = payload.get("task", payload.get("instruction", payload.get("prompt")))
        if not isinstance(task, str) or not task.strip():
            raise ValueError("missing non-empty task (or instruction/prompt)")

        task_id = str(payload.get("task_id") or uuid.uuid4())
        dry_run = bool(payload.get("dry_run", False))
        if hasattr(self.backend, "generate_with_tools"):
            state_object_types, state_warning = fetch_receiver_state_object_types(
                self.config.receiver_url,
                self.config.send_timeout,
            )
        else:
            state_object_types, state_warning = [], None
        upstream_subtask = payload.get("subtask") if isinstance(payload.get("subtask"), dict) else None
        semantic_resolution = None
        upstream_grounding = upstream_subtask.get("grounding") if isinstance(upstream_subtask, dict) and isinstance(upstream_subtask.get("grounding"), dict) else {}
        if (
            upstream_grounding.get("destination_object_tags")
            and should_prepare_semantic_destination(upstream_subtask)
        ):
            receiver_state, receiver_state_warning = fetch_receiver_state(self.config.receiver_url, self.config.send_timeout)
            if receiver_state_warning:
                state_warning = "; ".join(value for value in (state_warning, receiver_state_warning) if value)
            if receiver_state is not None:
                semantic_robot_id = payload.get("primary_robot_id", payload.get("robot_id", 0))
                if not isinstance(semantic_robot_id, int) or isinstance(semantic_robot_id, bool) or semantic_robot_id < 0:
                    raise ValueError("primary_robot_id must be a non-negative integer")
                def route_planner(object_id: str) -> dict[str, Any]:
                    return post_receiver_json(
                        f"{self.config.receiver_url.rstrip('/')}/goto",
                        {"task_id": f"{task_id}_semantic_target", "robot_id": semantic_robot_id, "object_id": object_id, "execute": False},
                        self.config.send_timeout,
                    )
                upstream_subtask, semantic_resolution = prepare_semantic_destination(
                    upstream_subtask, receiver_state, route_planner=route_planner,
                )
                if semantic_resolution is not None and semantic_resolution.get("status") != "resolved":
                    return semantic_target_failure_response(task_id, dry_run, semantic_resolution)
        primary_task_text, used_subtask_name, subtask_context, action_coverage_text = normalizer_subtask_inputs(upstream_subtask, task.strip())
        task_normalization = normalize_structured_subtask(
            upstream_subtask,
            task.strip(),
            state_object_types,
            primary_task_text=primary_task_text,
            subtask_context=subtask_context,
            used_subtask_name=used_subtask_name,
            action_coverage_text=action_coverage_text,
        )
        if task_normalization is None:
            task_normalization = normalize_incoming_task_with_backend(
                self.backend,
                task.strip(),
                state_object_types,
                primary_task_text=primary_task_text,
                subtask_context=subtask_context,
                used_subtask_name=used_subtask_name,
                action_coverage_text=action_coverage_text,
            )
        if semantic_resolution is not None:
            task_normalization["semantic_resolution"] = semantic_resolution
        if state_warning:
            task_normalization.setdefault("warnings", []).append(state_warning)

        if not (
            isinstance(task_normalization.get("task_intent"), dict)
            and not task_normalization.get("warnings")
            and task_normalization.get("confidence") in {"high", "medium"}
        ):
            return task_normalization_failure_response(task_id, dry_run, task_normalization)

        executable_task = str(task_normalization.get("normalized_task") or task).strip()
        task_intent_source = (
            "upstream_structured_task"
            if task_normalization.get("source") == "upstream_structured_task"
            else "qwen_normalizer_tool_call"
        )
        task_intent_json = json.dumps(
            {
                "task_intent": task_normalization["task_intent"],
                "task_intent_source": task_intent_source,
                "task_normalization": task_normalization,
                "upstream_subtask": upstream_subtask,
                "parent_task": payload.get("parent_task") if isinstance(payload.get("parent_task"), str) else None,
            },
            ensure_ascii=False,
        )
        primary_robot_id = payload.get("primary_robot_id", payload.get("robot_id", 0))
        if not isinstance(primary_robot_id, int) or isinstance(primary_robot_id, bool) or primary_robot_id < 0:
            raise ValueError("primary_robot_id must be a non-negative integer")
        max_replan_steps = _positive_int(
            payload.get("max_replan_steps", self.config.max_replan_steps), "max_replan_steps"
        )
        relay_agent_max_turns = _positive_int(
            payload.get("relay_agent_max_turns", self.config.relay_agent_max_turns),
            "relay_agent_max_turns",
        )
        max_actions = _nonnegative_int(payload.get("max_actions", self.config.max_actions), "max_actions")
        relay_strategy = payload.get("relay_strategy", "agent")
        if relay_strategy not in {"agent", "rules"}:
            raise ValueError("relay_strategy must be 'agent' or 'rules'")

        execute_actions_url = f"{self.config.receiver_url.rstrip('/')}/execute_actions"
        argv = [
            "--execute-actions-url", execute_actions_url,
            "--task", executable_task,
            "--task-id", task_id,
            "--output-dir", str(self.config.output_dir),
            "--send-timeout", str(self.config.send_timeout),
            "--qwen-model", self.config.model_path,
            "--qwen-device-map", self.config.device_map,
            "--qwen-dtype", self.config.dtype,
            "--device", self.config.device,
            "--max-new-tokens", str(self.config.max_new_tokens),
            "--temperature", str(self.config.temperature),
            "--max-actions", str(max_actions),
            "--save-raw-output",
            "--primary-robot-id", str(primary_robot_id),
            "--relay-mode",
            "--relay-strategy", relay_strategy,
            "--relay-agent-max-turns", str(relay_agent_max_turns),
            "--closed-loop-replan",
            "--max-replan-steps", str(max_replan_steps),
        ]
        if task_intent_json is not None:
            argv.extend(["--task-intent-json", task_intent_json])
        if dry_run:
            argv.append("--dry-run")

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        args = self.engine.parse_args(argv)
        setattr(args, "_qwen_backend", self.backend)
        stdout = io.StringIO()
        stderr = io.StringIO()
        with self.lock, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = self.engine.run(args)
        runtime_stdout = stdout.getvalue().strip()
        runtime_stderr = stderr.getvalue().strip()
        try:
            result = _last_json_object(runtime_stdout)
        except RuntimeError as exc:
            detail = runtime_stderr or runtime_stdout or "no diagnostic output"
            raise RuntimeError(
                f"relay runtime exited with code {exit_code} before producing a JSON result: {detail}"
            ) from exc
        closed_loop = result.get("closed_loop_result")
        closed_loop_status = closed_loop.get("status") if isinstance(closed_loop, dict) else None
        status = "success" if exit_code == 0 and closed_loop_status == "success" else "needs_upstream_planning"
        response: dict[str, Any] = {
            "status": status,
            "task_id": task_id,
            "dry_run": dry_run,
            "result": result,
            "task_normalization": task_normalization,
        }
        if status == "success":
            for trace in reversed(result.get("closed_loop_trace") or []):
                if not isinstance(trace, dict):
                    continue
                completion_agent_id = trace.get(
                    "executor_robot_id",
                    trace.get("executor_agent_id"),
                )
                if completion_agent_id not in (None, ""):
                    response["completion_agent_id"] = completion_agent_id
                    break
        if not dry_run:
            post_robot_ids, registry_diagnostics = coordinator_robot_ids_for_post_state(
                self.config.receiver_url,
                self.config.send_timeout,
                primary_robot_id,
                result,
            )
            post_agent_states, post_state_errors = fetch_receiver_agent_states(
                self.config.receiver_url,
                post_robot_ids,
                self.config.send_timeout,
            )
            response["post_agent_states"] = post_agent_states
            response["coordinator_agent_registry"] = {
                "robot_ids": post_robot_ids,
                "source": registry_diagnostics["source"],
                "refreshed_from": registry_diagnostics["receiver_health"],
            }
            if post_state_errors:
                response["post_state_errors"] = post_state_errors
        if status != "success" and isinstance(closed_loop, dict):
            failure_code = closed_loop.get("failure_code")
            reason = closed_loop.get("reason")
            if failure_code:
                response["failure_code"] = failure_code
                response["recoverable"] = failure_code in {
                    "target_not_visible",
                    "object_not_actionable",
                    "missing_required_state",
                    "navigation_failed",
                }
                if failure_code in {
                    "target_not_visible",
                    "object_not_actionable",
                    "missing_required_state",
                }:
                    response["recommended_recovery"] = "insert_find"
            if reason:
                response["reason"] = reason
        if runtime_stderr:
            response["runtime_log"] = runtime_stderr
        return response


task_service: TaskExecutionService | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "TaskExecutionServer/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        log(f"HTTP {self.address_string()} - {fmt % args}")

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status_code)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length > MAX_REQUEST_BYTES:
            raise ValueError(f"request is too large (max {MAX_REQUEST_BYTES} bytes)")
        body = self.rfile.read(length)
        if not body:
            return {}
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("request JSON must be an object")
        return payload

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        if urlparse(self.path).path in {"/", "/health"}:
            self._send_json(200, task_service.health())
        else:
            self._send_json(404, {"status": "failed", "error": "not_found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/execute_task":
            self._send_json(404, {"status": "failed", "error": "not_found"})
            return
        try:
            payload = self._read_json()
            log(f"POST /execute_task task_id={payload.get('task_id', 'generated')}")
            self._send_json(200, task_service.execute_task(payload))
        except ValueError as exc:
            self._send_json(400, {"status": "failed", "error": str(exc)})
        except Exception as exc:
            log(f"/execute_task failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            self._send_json(502, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    global task_service
    parser = argparse.ArgumentParser(description="Task execution service backed by EmbodiedGPT relay agent for AI2-THOR")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--receiver-url", default="http://127.0.0.1:19000")
    parser.add_argument("--model-path", default="models/Qwen3.5-4B")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--send-timeout", type=float, default=60.0)
    parser.add_argument("--output", type=Path, default=Path("output/task_execution"))
    parser.add_argument("--max-replan-steps", type=int, default=10)
    parser.add_argument("--relay-agent-max-turns", type=int, default=8)
    parser.add_argument("--max-actions", type=int, default=8)
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.temperature <= 0 or args.send_timeout <= 0:
        parser.error("--max-new-tokens, --temperature, and --send-timeout must be positive")
    if args.max_replan_steps <= 0 or args.relay_agent_max_turns <= 0 or args.max_actions < 0:
        parser.error("invalid relay/action limits")
    if not EMBODIED_ROOT.is_dir():
        parser.error(f"EmbodiedGPT runtime is missing: {EMBODIED_ROOT}")
    shard_error = model_shard_error(args.model_path)
    if shard_error is not None:
        parser.error(shard_error)

    sys.path.insert(0, str(EMBODIED_ROOT))
    from demo import auto_scene_actions
    from demo.qwen35_backend import Qwen35Backend, Qwen35Config

    config = TaskExecutionRuntimeConfig(
        receiver_url=args.receiver_url,
        model_path=args.model_path,
        device=args.device,
        device_map=args.device_map,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        send_timeout=args.send_timeout,
        output_dir=args.output,
        max_replan_steps=args.max_replan_steps,
        relay_agent_max_turns=args.relay_agent_max_turns,
        max_actions=args.max_actions,
    )
    backend = Qwen35Backend(
        Qwen35Config(
            model_name=config.model_path,
            device=config.device,
            device_map=config.device_map,
            torch_dtype=config.dtype,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
        )
    )
    task_service = TaskExecutionService(auto_scene_actions, backend, config)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    log(f"listening at http://{args.host}:{args.port}")
    log(f"revision: {SERVICE_REVISION}; relay agent: {sys.modules['demo.relay_agent'].__file__}")
    log(f"receiver: {config.receiver_url}; endpoint: POST /execute_task")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("KeyboardInterrupt: shutting down")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
