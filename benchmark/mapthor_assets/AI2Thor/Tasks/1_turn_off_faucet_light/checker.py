"""
To check completion of subtasks in 1_turn_off_faucet_light task.
Also checks coverage of the task and success of the task.
Subtasks:
    • NavigateTo(Faucet)
    • ToggleObjectOff(Faucet)
    • NavigateTo(LightSwitch)
    • ToggleObjectOff(LightSwitch)
Coverage:
    • Faucet
    • LightSwitch
"""

from AI2Thor.baselines.utils.checker import BaseChecker


class Checker(BaseChecker):
    def __init__(self) -> None:
        subtasks = [
            "NavigateTo(Faucet)",
            "ToggleObjectOff(Faucet)",
            "NavigateTo(LightSwitch)",
            "ToggleObjectOff(LightSwitch)",
        ]
        conditional_subtasks = []
        independent_subtasks = [
            "NavigateTo(Faucet)",
            "ToggleObjectOff(Faucet)",
            "NavigateTo(LightSwitch)",
            "ToggleObjectOff(LightSwitch)",
        ]
        coverage = ["Faucet", "LightSwitch"]
        interact_objects = ["Faucet", "LightSwitch"]
        interact_receptacles = []

        super().__init__(
            subtasks,
            conditional_subtasks,
            independent_subtasks,
            coverage,
            interact_objects,
            interact_receptacles,
        )
