from __future__ import annotations

from typing import Any

from .action_log import BenchmarkActionLogger


def _safe_metric(checker: Any, method: str, default: float) -> float:
    try:
        return float(getattr(checker, method)())
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return default


def _completed(checker: Any) -> list[str]:
    values = getattr(checker, "subtasks_completed_numerated", None)
    if values is None:
        values = getattr(checker, "subtasks_completed", [])
    return [str(item) for item in values or []]


def _required(checker: Any) -> list[str]:
    return [str(item) for item in getattr(checker, "subtasks", []) or []]


def evaluate_official_checker(
    checker: Any,
    action_logger: BenchmarkActionLogger,
    *,
    internal_all_done: bool,
    macro_steps: int,
    timeout: int,
    runtime_error: str | None = None,
) -> dict[str, Any]:
    required = _required(checker)
    completed = _completed(checker)
    try:
        success = bool(checker.check_success())
    except Exception:
        success = bool(required) and len(completed) == len(required)

    if runtime_error:
        termination_reason = "runtime_error"
    elif success:
        termination_reason = "benchmark_success"
    elif internal_all_done:
        termination_reason = "planner_exhausted_but_goal_unsatisfied"
    elif macro_steps >= timeout:
        termination_reason = "timeout"
    else:
        termination_reason = "hybrid_stopped"

    collisions = sum(1 for item in action_logger.records if item.get("collided"))
    successful_interactions = sum(
        1
        for item in action_logger.records
        if item.get("success")
        and item.get("action")
        not in {
            "MoveAhead",
            "MoveBack",
            "MoveLeft",
            "MoveRight",
            "RotateLeft",
            "RotateRight",
            "LookUp",
            "LookDown",
            "Pass",
        }
    )
    return {
        "success": int(success),
        "success_rate": float(success),
        "transport_rate": _safe_metric(checker, "get_transport_rate", 0.0),
        "coverage": _safe_metric(checker, "get_coverage", 0.0),
        "balance": action_logger.balance(),
        "steps": min(int(macro_steps), int(timeout)),
        "timeout": int(timeout),
        "low_level_actions": len(action_logger.records),
        "successful_interactions": successful_interactions,
        "collisions": collisions,
        "required_subtasks": required,
        "completed_subtasks": completed,
        "internal_all_done": bool(internal_all_done),
        "internal_benchmark_agree": bool(internal_all_done) == bool(success),
        "termination_reason": termination_reason,
        "progress_actions_by_agent": {
            str(key): value for key, value in sorted(action_logger.progress_by_agent.items())
        },
        "metric_errors": [
            item["metric_error"] for item in action_logger.records if item.get("metric_error")
        ],
        "runtime_error": runtime_error,
    }
