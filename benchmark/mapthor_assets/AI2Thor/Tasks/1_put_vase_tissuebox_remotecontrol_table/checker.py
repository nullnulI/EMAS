"""
To check completion of subtasks in 1_put_vase_tissuebox_remotecontrol_table task.
Also checks coverage of the task and success of the task.
Subtasks:
    • NavigateTo(Vase)
    • PickupObject(Vase)
    • NavigateTo(Table) [with vase in inventory]
    • PutObject(Table) [with vase in inventory]
    • NavigateTo(TissueBox)
    • PickupObject(TissueBox)
    • NavigateTo(Table) [with tissuebox in inventory]
    • PutObject(Table) [with tissuebox in inventory]
    • NavigateTo(RemoteControl)
    • PickupObject(RemoteControl)
    • NavigateTo(Table) [with remotecontrol in inventory]
    • PutObject(Table) [with remotecontrol in inventory]
Coverage:
    • Vase
    • Table
    • Tissuebox
    • Remotecontrol
"""

from AI2Thor.baselines.utils.checker import BaseChecker


class Checker(BaseChecker):
    def __init__(self) -> None:
        subtasks = [
            "NavigateTo(Vase)",
            "PickupObject(Vase)",
            "NavigateTo(Table, Vase)",
            "PutObject(Table, Vase)",
            "NavigateTo(TissueBox)",
            "PickupObject(TissueBox)",
            "NavigateTo(Table, TissueBox)",
            "PutObject(Table, TissueBox)",
            "NavigateTo(RemoteControl)",
            "PickupObject(RemoteControl)",
            "NavigateTo(Table, RemoteControl)",
            "PutObject(Table, RemoteControl)",
        ]
        conditional_subtasks = [
            "NavigateTo(Table, Vase)",
            "PutObject(Table, Vase)",
            "NavigateTo(Table, TissueBox)",
            "PutObject(Table, TissueBox)",
            "NavigateTo(Table, RemoteControl)",
            "PutObject(Table, RemoteControl)",
        ]
        independent_subtasks = [
            "NavigateTo(Vase)",
            "PickupObject(Vase)",
            "NavigateTo(TissueBox)",
            "PickupObject(TissueBox)",
            "NavigateTo(RemoteControl)",
            "PickupObject(RemoteControl)",
        ]
        coverage = ["Vase", "Table", "TissueBox", "RemoteControl"]
        interact_objects = ["Vase", "TissueBox", "RemoteControl"]
        interact_receptacles = ["Table"]

        super().__init__(
            subtasks,
            conditional_subtasks,
            independent_subtasks,
            coverage,
            interact_objects,
            interact_receptacles,
        )

    # slight modification since there weren't enough DiningTables or Vases in the 30 FloorPlans
    def denumerate_object(self, object: str) -> str:
        """
        Remove object number from object_id
        Example: Bread_1 -> Bread
        """
        if "_" in object:
            object_name, object_id = object.split("_")
        else:
            object_name = object

        if "Table" in object_name:
            return "Table"  # DiningTable or SideTable -> Table
        if "WateringCan" in object_name:
            return "Vase"  # vase or watering can -> vase
        return object_name
