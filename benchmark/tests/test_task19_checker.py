from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MAPTHOR_ROOT = Path(__file__).resolve().parents[1] / "mapthor_assets"
CHECKER_PATH = (
    MAPTHOR_ROOT
    / "AI2Thor"
    / "Tasks"
    / "3_put_all_silverware_drawer"
    / "checker.py"
)
SILVERWARE = ("ButterKnife", "Knife", "Spatula", "Spoon", "Fork", "Ladle")


def _checker():
    root = str(MAPTHOR_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location("task19_checker_test", CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checker = module.Checker()
    checker.all_objects(
        [*(f"{kind}|0|0|0" for kind in SILVERWARE), "Drawer|0|0|0", "Drawer|1|0|0"],
        "FloorPlan2",
    )
    return checker


def _complete_object(checker, object_type: str) -> None:
    inventory = f"{object_type}_1"
    checker.perform_metric_check(f"PickUpObject({inventory})", True, inventory)
    checker.perform_metric_check("PutObject(Drawer_1)", True, inventory)


def test_task19_requires_all_six_silverware_and_one_drawer_open_close() -> None:
    checker = _checker()

    assert len(checker.subtasks) == 26
    assert "PickUpObject(Ladle_1)" in checker.subtasks
    assert "PutObject(Drawer, Ladle)" in checker.subtasks
    assert all("Butterknife" not in subtask for subtask in checker.subtasks)
    assert checker.subtasks.count("OpenObject(Drawer)") == 1
    assert checker.subtasks.count("CloseObject(Drawer)") == 1


def test_task19_full_execution_is_exactly_complete_without_duplicate_credit() -> None:
    checker = _checker()

    checker.perform_metric_check("OpenObject(Drawer_1)", True, "nothing")
    checker.perform_metric_check("OpenObject(Drawer_1)", True, "nothing")
    for object_type in SILVERWARE:
        _complete_object(checker, object_type)
    checker.perform_metric_check("CloseObject(Drawer_1)", True, "nothing")

    assert len(checker.subtasks_completed_numerated) == 26
    assert checker.get_transport_rate() == 1.0
    assert checker.check_success()


def test_task19_missing_ladle_is_not_successful() -> None:
    checker = _checker()

    checker.perform_metric_check("OpenObject(Drawer_2)", True, "nothing")
    for object_type in SILVERWARE[:-1]:
        _complete_object(checker, object_type)
    checker.perform_metric_check("CloseObject(Drawer_2)", True, "nothing")

    assert checker.get_transport_rate() < 1.0
    assert not checker.check_success()
