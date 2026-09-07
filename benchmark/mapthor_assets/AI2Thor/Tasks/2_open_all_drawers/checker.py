"""
To check completion of subtasks in 2_open_all_drawers
Also checks coverage of the task and success of the task.
Subtasks:
    • NavigateTo(Drawer)
    • OpenObject(Drawer) [no need for objects in inventory]
Coverage:
    • Drawer
"""

from AI2Thor.baselines.utils.checker import BaseChecker

class Checker(BaseChecker):
    def __init__(self) -> None:
        subtasks = [
            "NavigateTo(Drawer)",
            "OpenObject(Drawer)",
        ]
        conditional_subtasks = [
        ]

        independent_subtasks = [
            "NavigateTo(Drawer)",
            "OpenObject(Drawer)",
        ]
        coverage = ["Drawer"]
        interact_objects = ["Drawer"]
        interact_receptacles = []

        super().__init__(
            subtasks,
            conditional_subtasks,
            independent_subtasks,
            coverage,
            interact_objects,
            interact_receptacles,
        )
