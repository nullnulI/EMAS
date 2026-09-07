"""
To check completion of subtasks in 1_put_knife_bowl_mug_countertop
Also checks coverage of the task and success of the task.
Subtasks:
    • NavigateTo(ButterKnife)
    • PickupObject(ButterKnife)
    • NavigateTo(CounterTop) [with butterknife in inventory]
    • PutObject(CounterTop) [with butterknife in inventory]
    • NavigateTo(Bowl)
    • PickupObject(Bowl)
    • NavigateTo(CounterTop) [with bowl in inventory]
    • PutObject(CounterTop) [with bowl in inventory]
    • NavigateTo(Mug)
    • PickupObject(Mug)
    • NavigateTo(CounterTop) [with mug in inventory]
    • PutObject(CounterTop) [with mug in inventory]
Coverage:
    • ButterKnife 
    • Mug 
    • Bowl 
    • CounterTop 
"""

from AI2Thor.baselines.utils.checker import BaseChecker


class Checker(BaseChecker):
    def __init__(self) -> None:
        subtasks = [
            "NavigateTo(ButterKnife)",
            "PickupObject(ButterKnife)",
            "NavigateTo(CounterTop, ButterKnife)",
            "PutObject(CounterTop, ButterKnife)",
            "NavigateTo(Bowl)",
            "PickupObject(Bowl)",
            "NavigateTo(CounterTop, Bowl)",
            "PutObject(CounterTop, Bowl)",
            "NavigateTo(Mug)",
            "PickupObject(Mug)",
            "NavigateTo(CounterTop, Mug)",
            "PutObject(CounterTop, Mug)",
        ]
        conditional_subtasks = [
            "NavigateTo(CounterTop, ButterKnife)",
            "PutObject(CounterTop, ButterKnife)",
            "NavigateTo(CounterTop, Bowl)",
            "PutObject(CounterTop, Bowl)",
            "NavigateTo(CounterTop, Mug)",
            "PutObject(CounterTop, Mug)",
        ]
        independent_subtasks = [
            "NavigateTo(ButterKnife)",
            "PickupObject(ButterKnife)",
            "NavigateTo(Bowl)",
            "PickupObject(Bowl)",
            "NavigateTo(Mug)",
            "PickupObject(Mug)",
        ]
        coverage = ["ButterKnife", "CounterTop", "Bowl", "Mug"]
        interact_objects = ["ButterKnife", "Bowl", "Mug"]
        interact_receptacles = ["CounterTop"]

        super().__init__(
            subtasks,
            conditional_subtasks,
            independent_subtasks,
            coverage,
            interact_objects,
            interact_receptacles,
        )
