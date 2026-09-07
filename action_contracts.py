"""Shared action contracts for planning, relay normalization, and execution.

The registry separates safe planner-visible semantic actions from controller-only
actions.  It intentionally contains execution contracts only; task-language
semantics remain the responsibility of the planning model and semantic critic.
"""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any


AFFORDANCE_FIELDS = (
    "pickupable",
    "moveable",
    "openable",
    "toggleable",
    "receptacle",
    "sliceable",
    "dirtyable",
    "breakable",
    "cookable",
    "canFillWithLiquid",
)


# ``affordances`` is an any-of list.  A one-item tuple is the usual contract.
ACTION_CONTRACTS: dict[str, dict[str, Any]] = {
    "pick": {"native_action": "PickupObject", "roles": {"source": ("pickupable",)}},
    "place": {
        "native_action": "PutObject",
        "roles": {"source": ("pickupable",), "destination": ("receptacle",)},
    },
    "open": {"native_action": "OpenObject", "roles": {"source": ("openable",)}},
    "close": {"native_action": "CloseObject", "roles": {"source": ("openable",)}},
    "toggle_on": {"native_action": "ToggleObjectOn", "roles": {"source": ("toggleable",)}},
    "toggle_off": {"native_action": "ToggleObjectOff", "roles": {"source": ("toggleable",)}},
    "clean": {"native_action": "CleanObject", "roles": {"source": ("dirtyable",)}},
    "slice": {
        "native_action": "SliceObject",
        "roles": {"source": ("sliceable",)},
        "required_tool": {
            "mode": "held",
            "any_of_types": ("Knife", "ButterKnife"),
        },
    },
    "drop": {
        "native_action": "DropHandObject",
        "roles": {"source": ("pickupable",)},
        "source_quantifier": "one",
        "requires_held_source": True,
    },
    "push": {
        "native_action": "PushObject",
        "roles": {"source": ("moveable", "pickupable")},
        "args": {"moveMagnitude": {"type": "number", "minimum_exclusive": 0.0, "maximum": 1000.0}},
    },
    "pull": {
        "native_action": "PullObject",
        "roles": {"source": ("moveable", "pickupable")},
        "args": {"moveMagnitude": {"type": "number", "minimum_exclusive": 0.0, "maximum": 1000.0}},
    },
    "move_held": {
        "native_action": "MoveHeldObject",
        "roles": {"source": ("pickupable",)},
        "source_quantifier": "one",
        "requires_held_source": True,
        "args": {
            "right": {"type": "number", "minimum": -0.5, "maximum": 0.5, "default": 0.0},
            "up": {"type": "number", "minimum": -0.5, "maximum": 0.5, "default": 0.0},
            "ahead": {"type": "number", "minimum": -0.5, "maximum": 0.5, "default": 0.0},
        },
        "at_least_one_nonzero": ("right", "up", "ahead"),
    },
    "break": {"native_action": "BreakObject", "roles": {"source": ("breakable",)}},
    "cook": {"native_action": "CookObject", "roles": {"source": ("cookable",)}},
    "fill": {
        "native_action": "FillObjectWithLiquid",
        "roles": {"source": ("canFillWithLiquid",)},
        "args": {
            "fillLiquid": {
                "type": "enum",
                "values": ("water", "coffee", "wine"),
                # Unlike numeric execution parameters, the requested liquid
                # changes the public task semantics and may be reviewed by the
                # semantic critic.
                "semantic": True,
            },
        },
    },
}

ACTION_ALIASES = {
    "pickup": "pick",
    "put": "place",
    "turn_on": "toggle_on",
    "switch_on": "toggle_on",
    "toggleobjecton": "toggle_on",
    "turn_off": "toggle_off",
    "switch_off": "toggle_off",
    "toggleobjectoff": "toggle_off",
    "wash": "clean",
    "cut": "slice",
    "drophandobject": "drop",
    "moveheldobject": "move_held",
    "breakobject": "break",
    "cookobject": "cook",
    "fillobjectwithliquid": "fill",
}

PLANNER_ACTIONS = frozenset({
    *ACTION_CONTRACTS,
    "navigate",
    "find",
    "inspect",
    "other",
})

INTERNAL_NATIVE_ACTIONS = frozenset({
    "MoveAhead", "MoveBack", "MoveLeft", "MoveRight",
    "RotateLeft", "RotateRight", "LookUp", "LookDown",
    "Teleport", "TeleportFull", "GetReachablePositions",
    "Pass", "Done", "SetObjectStates",
})


def normalize_logical_action(value: Any) -> str:
    action = str(value or "other").strip().lower()
    if action == "search":
        return "find"
    return ACTION_ALIASES.get(action, action)


def action_contract(value: Any) -> dict[str, Any] | None:
    contract = ACTION_CONTRACTS.get(normalize_logical_action(value))
    return deepcopy(contract) if contract is not None else None


def action_role_affordances() -> dict[str, dict[str, tuple[str, ...]]]:
    result: dict[str, dict[str, tuple[str, ...]]] = {}
    for name, contract in ACTION_CONTRACTS.items():
        result[name] = {
            role: tuple(values)
            for role, values in (contract.get("roles") or {}).items()
        }
    for alias, canonical in ACTION_ALIASES.items():
        if canonical in result:
            result[alias] = deepcopy(result[canonical])
    return result


def validate_action_args(action: Any, value: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate and normalize planner-controlled arguments for one logical action."""

    logical = normalize_logical_action(action)
    contract = ACTION_CONTRACTS.get(logical) or {}
    rules = contract.get("args") or {}
    raw = {} if value is None else value
    errors: list[dict[str, Any]] = []
    if not isinstance(raw, dict):
        return {}, [{
            "code": "invalid_action_args",
            "field": "action_args",
            "message": "action_args must be a JSON object",
            "invalid_values": [raw],
            "required_fix": "Return action_args as an object containing only supported fields.",
        }]

    unknown = sorted(str(key) for key in raw if key not in rules)
    if unknown:
        errors.append({
            "code": "unsupported_action_args",
            "field": "action_args",
            "message": f"unsupported action_args fields for {logical!r}: {unknown}",
            "invalid_values": unknown,
            "required_fix": "Remove every unsupported argument field.",
        })

    normalized: dict[str, Any] = {}
    for key, rule in rules.items():
        if key not in raw:
            if "default" in rule:
                normalized[key] = rule["default"]
                continue
            errors.append({
                "code": "missing_action_arg",
                "field": f"action_args.{key}",
                "message": f"missing required action argument {key!r}",
                "invalid_values": [],
                "required_fix": f"Provide action_args.{key} using the declared action contract.",
            })
            continue
        item = raw.get(key)
        if rule.get("type") == "number":
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not isfinite(float(item)):
                errors.append({
                    "code": "invalid_action_arg_type",
                    "field": f"action_args.{key}",
                    "message": f"{key!r} must be a finite number",
                    "invalid_values": [item],
                    "required_fix": f"Provide a finite numeric value for {key}.",
                })
                continue
            number = float(item)
            if "minimum_exclusive" in rule and number <= float(rule["minimum_exclusive"]):
                errors.append({
                    "code": "action_arg_out_of_range", "field": f"action_args.{key}",
                    "message": f"{key!r} must be greater than {rule['minimum_exclusive']}",
                    "invalid_values": [item], "required_fix": f"Increase {key} into the supported range.",
                })
                continue
            if "minimum" in rule and number < float(rule["minimum"]):
                errors.append({
                    "code": "action_arg_out_of_range", "field": f"action_args.{key}",
                    "message": f"{key!r} must be at least {rule['minimum']}",
                    "invalid_values": [item], "required_fix": f"Move {key} into the supported range.",
                })
                continue
            if "maximum" in rule and number > float(rule["maximum"]):
                errors.append({
                    "code": "action_arg_out_of_range", "field": f"action_args.{key}",
                    "message": f"{key!r} must be at most {rule['maximum']}",
                    "invalid_values": [item], "required_fix": f"Reduce {key} into the supported range.",
                })
                continue
            normalized[key] = number
        elif rule.get("type") == "enum":
            text = str(item or "").strip().lower()
            if text not in set(rule.get("values") or ()):
                errors.append({
                    "code": "invalid_action_arg_value", "field": f"action_args.{key}",
                    "message": f"{key!r} must be one of {list(rule.get('values') or ())}",
                    "invalid_values": [item], "required_fix": f"Use a supported {key} value.",
                })
                continue
            normalized[key] = text

    nonzero_fields = contract.get("at_least_one_nonzero") or ()
    if nonzero_fields and not errors and not any(abs(float(normalized.get(key, 0.0))) > 0 for key in nonzero_fields):
        errors.append({
            "code": "no_op_action_args",
            "field": "action_args",
            "message": f"at least one of {list(nonzero_fields)} must be non-zero",
            "invalid_values": [deepcopy(normalized)],
            "required_fix": "Provide a non-zero relative displacement.",
        })
    return normalized, errors
