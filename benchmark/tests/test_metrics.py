from benchmark.metrics import aggregate_results


def test_aggregate_results():
    summary = aggregate_results(
        [
            {"evaluation": {"success_rate": 1.0, "transport_rate": 1.0, "coverage": 0.5, "balance": 1.0, "steps": 10}},
            {"evaluation": {"success_rate": 0.0, "transport_rate": 0.5, "coverage": 1.0, "balance": 0.5, "steps": 30}},
        ]
    )
    assert summary["episodes_evaluated"] == 2
    assert summary["metrics"]["success_rate"]["mean"] == 0.5
    assert summary["metrics"]["steps"]["mean"] == 20.0
