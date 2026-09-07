"""
Subtasks:
    • NavigateTo(StoveKnob)
    • ToggleObjectOn(StoveKnob) 
Coverage:
    • StoveKnob 

Turn all the stoveknobs off before the pre-init
"""

from AI2Thor.baselines.utils.checker import BaseChecker

class Checker(BaseChecker):
    def __init__(self) -> None:
        subtasks = [
            "NavigateTo(StoveKnob)",
            "ToggleObjectOn(StoveKnob)",
        ]
        conditional_subtasks = []
        independent_subtasks = [
            "NavigateTo(StoveKnob)",
            "ToggleObjectOn(StoveKnob)",
        ]
        coverage = ["StoveKnob"] 
        interact_objects = ["StoveKnob"] 
        interact_receptacles = []

        super().__init__(
            subtasks,
            conditional_subtasks,
            independent_subtasks,
            coverage,
            interact_objects,
            interact_receptacles,
        )
