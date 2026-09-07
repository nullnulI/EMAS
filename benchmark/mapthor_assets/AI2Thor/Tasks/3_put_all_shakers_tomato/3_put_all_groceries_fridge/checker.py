"""
To check completion of subtasks in 3_put_all_groceries_fridge task.
Also checks coverage of the task and success of the task.
Subtasks:
    • NavigateTo(SaltShaker)
    • PickupObject(SaltShaker)
    • NavigateTo(Tomato) [with SaltShaker in inventory]
    • PutObject(CounterTop) [with SaltShaker in inventory]
    • NavigateTo(PepperShaker)
    • PickupObject(PepperShaker)
    • NavigateTo(Tomato) [with PepperShaker in inventory]
    • PutObject(CounterTop) [with PepperShaker in inventory]
Coverage:
    • SaltShaker
    • PepperShaker
    • Tomato
    • CounterTop
"""

from AI2Thor.baselines.utils.checker import BaseChecker


class Checker(BaseChecker):
    def __init__(self) -> None:
        subtasks = [
            "NavigateTo(SaltShaker)",
            "PickupObject(SaltShaker)",
            "NavigateTo(Tomato, SaltShaker)",
            "PutObject(CounterTop, SaltShaker)",
            "NavigateTo(PepperShaker)",
            "PickupObject(PepperShaker)",
            "NavigateTo(Tomato, PepperShaker)",
            "PutObject(CounterTop, PepperShaker)",
        ]
        conditional_subtasks = [
            "NavigateTo(Tomato, SaltShaker)",
            "PutObject(CounterTop, SaltShaker)",
            "NavigateTo(Tomato, PepperShaker)",
            "PutObject(CounterTop, PepperShaker)",
        ]
        independent_subtasks = [
            "NavigateTo(SaltShaker)",
            "PickupObject(SaltShaker)",
            "NavigateTo(PepperShaker)",
            "PickupObject(PepperShaker)",
        ]
        coverage = ["PepperShaker", "SaltShaker", "Tomato", "CounterTop"]
        interact_objects = ["PepperShaker", "SaltShaker", "Tomato"]
        interact_receptacles = ["Tomato", "CounterTop"]

        super().__init__(
            subtasks,
            conditional_subtasks,
            independent_subtasks,
            coverage,
            interact_objects,
            interact_receptacles,
        )
