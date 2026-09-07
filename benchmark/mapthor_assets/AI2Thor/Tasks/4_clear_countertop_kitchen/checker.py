"""
To check completion of subtasks in 4_clear_countertop_kitchen task.
Also checks coverage of the task and success of the task.
Subtasks:
    • NavigateTo(Tomato)
    • PickUpObject(Tomato)
    • NavigateTo(Fridge) [with Tomato in inventory]
    • PutObject(Fridge) [with Tomato in inventory]
    • NavigateTo(Apple)
    • PickUpObject(Apple)
    • NavigateTo(Fridge) [with Apple in inventory]
    • PutObject(Fridge) [with Apple in inventory]
    • NavigateTo(ButterKnife)
    • PickUpObject(ButterKnife)
    • NavigateTo(Drawer) [with ButterKnife in inventory]
    • OpenObject(Drawer) [with ButterKnife in inventory]
    • PutObject(Drawer) [with ButterKnife in inventory]
    • NavigateTo(Fork)
    • PickUpObject(Fork)
    • NavigateTo(Drawer) [with Fork in inventory]
    • OpenObject(Drawer) [with Fork in inventory]
    • PutObject(Drawer) [with Fork in inventory]
    • OpenObject(Fridge)
    • CloseObject(Fridge)

Coverage:
    • Apple
    • Tomato
    • Fork
    • ButterKnife
    • Fridge 
"""


from AI2Thor.baselines.utils.checker import BaseChecker

class Checker(BaseChecker):
    def __init__(self) -> None:
        subtasks = [
        'NavigateTo(Tomato)',
         'PickUpObject(Tomato)',
         'NavigateTo(Fridge, Tomato)',
         'PutObject(Fridge, Tomato)',
         'NavigateTo(Apple)',
         'PickUpObject(Apple)',
         'NavigateTo(Fridge, Apple)',
         'PutObject(Fridge, Apple)',
         'NavigateTo(ButterKnife)',
         'PickUpObject(ButterKnife)',
         'NavigateTo(Drawer, ButterKnife)',
         'OpenObject(Drawer, ButterKnife)',
         'PutObject(Drawer, ButterKnife)',
         'NavigateTo(Fork)',
         'PickUpObject(Fork)',
         'NavigateTo(Drawer, Fork)',
         'OpenObject(Drawer, Fork)',
         'PutObject(Drawer, Fork)',
         'OpenObject(Fridge)',
         'CloseObject(Fridge)'
        ]


        conditional_subtasks = [
        'NavigateTo(Fridge, Tomato)',
         'PutObject(Fridge, Tomato)',
         'NavigateTo(Fridge, Apple)',
         'PutObject(Fridge, Apple)',
         'NavigateTo(Drawer, ButterKnife)',
         'OpenObject(Drawer, ButterKnife)',
         'PutObject(Drawer, ButterKnife)',
         'NavigateTo(Drawer, Fork)',
         'OpenObject(Drawer, Fork)',
         'PutObject(Drawer, Fork)',
        ]

        independent_subtasks = [
        'NavigateTo(Tomato)',
         'PickUpObject(Tomato)',
         'NavigateTo(Apple)',
         'PickUpObject(Apple)',
         'NavigateTo(ButterKnife)',
         'PickUpObject(ButterKnife)',
         'NavigateTo(Fork)',
         'PickUpObject(Fork)',
         'OpenObject(Fridge)',
         'CloseObject(Fridge)'
        ]

        coverage = ["Fridge", "Drawer", "Apple", "Tomato", "Fork", "ButterKnife"]
        interact_objects = coverage
        interact_receptacles = ["Fridge", "Drawer"]

        super().__init__(
            subtasks,
            conditional_subtasks,
            independent_subtasks,
            coverage,
            interact_objects,
            interact_receptacles,
        )
