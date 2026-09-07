"""
To check completion of subtasks in 2_open_all_drawers
Also checks coverage of the task and success of the task.
Subtasks:
    • NavigateTo(Cabinet)
    • OpenObject(Cabinet) [no need for objects in inventory]
Coverage:
    • Cabinet
"""

from AI2Thor.baselines.utils.checker import BaseChecker

class Checker(BaseChecker):
    def __init__(self) -> None:
        subtasks = [
            "NavigateTo(Cabinet)",
            "OpenObject(Cabinet)",
        ]
        conditional_subtasks = [
        ]

        independent_subtasks = [
            "NavigateTo(Cabinet)",
            "OpenObject(Cabinet)",
        ]
        coverage = ["Cabinet"]
        interact_objects = ["Cabinet"]
        interact_receptacles = []

        super().__init__(
            subtasks,
            conditional_subtasks,
            independent_subtasks,
            coverage,
            interact_objects,
            interact_receptacles,
        )
