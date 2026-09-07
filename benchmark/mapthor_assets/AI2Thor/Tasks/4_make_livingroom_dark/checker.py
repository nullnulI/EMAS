"""
To check completion of subtasks
Also checks coverage of the task and success of the task.
Subtasks:
    • NavigateTo(LightSwitch)
    • ToggleObjectOff(LightSwitch)

Coverage:
    • LightSwitch
"""


from AI2Thor.baselines.utils.checker import BaseChecker

class Checker(BaseChecker):
    def __init__(self) -> None:

        subtasks = [
        'NavigateTo(LightSwitch)', 'ToggleObjectOff(LightSwitch)'
        ]

        conditional_subtasks = []

        independent_subtasks = subtasks

        coverage = ["LightSwitch"]
        interact_objects = coverage
        interact_receptacles = []

        super().__init__(
            subtasks,
            conditional_subtasks,
            independent_subtasks,
            coverage,
            interact_objects,
            interact_receptacles,
        )
