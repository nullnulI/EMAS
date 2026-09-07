"""
To check completion of subtasks in 1_wash_bowl_mug_pot_pan task.
Also checks coverage of the task and success of the task.
Subtasks:
• NavigateTo(Bowl)
• CleanObject(Bowl)
• NavigateTo(Mug)
• CleanObject(Mug)
• NavigateTo(Pan)
• CleanObject(Pan)
• NavigateTo(Pot)
• CleanObject(Pot)
Coverage:
    • Pot
    • Pan
    • Bowl
    • Mug
"""

from AI2Thor.baselines.utils.checker import BaseChecker


class Checker(BaseChecker):
    def __init__(self) -> None:
        subtasks = [
            "NavigateTo(Bowl)",
            "CleanObject(Bowl)",
            "NavigateTo(Mug)",
            "CleanObject(Mug)",
            "NavigateTo(Pan)",
            "CleanObject(Pan)",
            "NavigateTo(Pot)",
            "CleanObject(Pot)",
        ]
        conditional_subtasks = []
        independent_subtasks = [
            "NavigateTo(Bowl)",
            "CleanObject(Bowl)",
            "NavigateTo(Mug)",
            "CleanObject(Mug)",
            "NavigateTo(Pan)",
            "CleanObject(Pan)",
            "NavigateTo(Pot)",
            "CleanObject(Pot)",
        ]
        coverage = ["Bowl", "Mug", "Pan", "Pot"]
        interact_objects = ["Bowl", "Mug", "Pan", "Pot"]
        interact_receptacles = []

        super().__init__(
            subtasks,
            conditional_subtasks,
            independent_subtasks,
            coverage,
            interact_objects,
            interact_receptacles,
        )
