from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MAPTHOR_ROOT = Path(__file__).resolve().parents[1] / "mapthor_assets"
if str(MAPTHOR_ROOT) not in sys.path:
    sys.path.insert(0, str(MAPTHOR_ROOT))

from AI2Thor.baselines.utils.checker import BaseChecker


def _load_checker(relative_path: str):
    path = MAPTHOR_ROOT / "AI2Thor" / "Tasks" / relative_path / "checker.py"
    module_name = "checker_contract_" + relative_path.replace("/", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Checker()


def test_inventory_enrichment_cannot_manufacture_unrequired_conditional_credit() -> None:
    checker = BaseChecker(
        subtasks=["PutObject(Fridge, Apple)"],
        conditional_subtasks=["PutObject(Fridge, Apple)"],
        independent_subtasks=[],
        coverage=[],
        interact_objects=["Apple", "Mug"],
        interact_receptacles=["Fridge", "Drawer"],
    )

    checker.perform_metric_check("PutObject(Drawer_1)", True, "Mug_1")
    assert checker.subtasks_completed == []
    assert checker.subtasks_completed_numerated == []

    checker.perform_metric_check("PutObject(Fridge_1)", True, "Apple_1")
    assert checker.subtasks_completed == ["PutObject(Fridge, Apple)"]


def test_task14_checker_tolerates_low_level_actions_and_credits_compound_place() -> None:
    checker = _load_checker("2_put_all_tomatoes_potatoes_fridge")

    checker.perform_metric_check("MoveAhead()", True, "nothing")
    checker.perform_metric_check("OpenObject(Fridge_1)", True, "nothing")
    for object_type in ("Tomato", "Potato"):
        inventory = f"{object_type}_1"
        checker.perform_metric_check(f"PickupObject({inventory})", True, inventory)
        checker.perform_metric_check("PutObject(Fridge_1)", True, inventory)
    checker.perform_metric_check("CloseObject(Fridge_1)", True, "nothing")

    assert checker.get_transport_rate() == 1.0
    assert checker.check_success()


def test_task23_drawer_actions_keep_held_object_in_normalized_credit() -> None:
    checker = _load_checker("4_clear_countertop_kitchen")
    assert checker.interact_receptacles == ["Fridge", "Drawer"]
    assert all("Butterknife" not in action for action in checker.subtasks)

    checker.perform_metric_check("PutObject(Drawer_2)", True, "ButterKnife_1")
    assert "PutObject(Drawer, ButterKnife)" in checker.subtasks_completed
    assert "PutObject(Drawer_2, ButterKnife_1)" in checker.subtasks_completed_numerated


def test_task23_does_not_credit_wrong_object_destination_pair() -> None:
    checker = _load_checker("4_clear_countertop_kitchen")
    checker.perform_metric_check("PutObject(Drawer_1)", True, "Tomato_1")
    assert "PutObject(Drawer, Tomato)" not in checker.subtasks_completed


def test_audited_checker_names_and_receptacles_are_canonical() -> None:
    clear_table = _load_checker("4_clear_table_kitchen")
    storage = _load_checker("4_put_appropriate_storage")
    couch = _load_checker("4_clear_couch_livingroom")
    school = _load_checker("3_put_all_school_supplies_sofa")
    shakers = _load_checker("3_put_all_shakers_tomato/3_put_all_groceries_fridge")
    sofa_table = _load_checker("3_clear_table_to_sofa")

    serialized = "\n".join(
        action
        for checker in (clear_table, storage, couch, school, shakers, sofa_table)
        for action in checker.subtasks
    )
    assert "Butterknife" not in serialized
    assert "Keychain" not in serialized
    assert "Cellphone" not in serialized
    assert "NavigateTo(PepperShaker" not in serialized.replace(
        "NavigateTo(PepperShaker)", ""
    )
    assert "Drawer" in clear_table.interact_receptacles
    assert shakers.interact_receptacles == ["Tomato", "CounterTop"]
    assert "Plate" in sofa_table.coverage

