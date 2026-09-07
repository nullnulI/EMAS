from __future__ import annotations

import math
import statistics
from typing import Any


METRIC_KEYS = ("success_rate", "transport_rate", "coverage", "balance", "steps")


def _mean_ci(values: list[float]) -> dict[str, float | int | str]:
    if not values:
        return {"n": 0, "mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "method": "none"}
    mean = statistics.fmean(values)
    if len(values) == 1:
        return {
            "n": 1,
            "mean": mean,
            "ci95_low": mean,
            "ci95_high": mean,
            "method": "single_observation",
        }
    stderr = statistics.stdev(values) / math.sqrt(len(values))
    margin = 1.96 * stderr
    return {
        "n": len(values),
        "mean": mean,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "method": "normal_approximation",
    }


def _success_ci(successes: list[int]) -> dict[str, float | int | str]:
    if not successes:
        return {"n": 0, "mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "method": "none"}
    count = sum(successes)
    total = len(successes)
    try:
        from scipy.stats import beta

        low = 0.0 if count == 0 else float(beta.ppf(0.025, count, total - count + 1))
        high = 1.0 if count == total else float(beta.ppf(0.975, count + 1, total - count))
        method = "clopper_pearson"
    except ImportError:
        p = count / total
        z = 1.96
        denom = 1 + z * z / total
        center = (p + z * z / (2 * total)) / denom
        margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
        low, high = max(0.0, center - margin), min(1.0, center + margin)
        method = "wilson_fallback"
    return {
        "n": total,
        "mean": count / total,
        "ci95_low": low,
        "ci95_high": high,
        "method": method,
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [item for item in results if item.get("evaluation")]
    metrics = {}
    for key in METRIC_KEYS:
        values = [float(item["evaluation"].get(key, 0.0)) for item in valid]
        metrics[key] = (
            _success_ci([int(value > 0.5) for value in values])
            if key == "success_rate"
            else _mean_ci(values)
        )
    return {
        "episodes_total": len(results),
        "episodes_evaluated": len(valid),
        "runtime_errors": sum(
            1 for item in results if (item.get("evaluation") or {}).get("runtime_error")
        ),
        "metrics": metrics,
    }
