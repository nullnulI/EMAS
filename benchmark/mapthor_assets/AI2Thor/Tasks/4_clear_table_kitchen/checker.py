"""
To check completion of subtasks in 2_put_all_tomatoes_potatoes_fridge task.
Also checks coverage of the task and success of the task.
Subtasks:
    • NavigateTo(Tomato)
    • PickUpObject(Tomato)
    • NavigateTo(Fridge) [with Tomato in inventory]
    • PutObject(Fridge) [with Tomato in inventory]
    • NavigateTo(Lettuce)
    • PickUpObject(Lettuce)
    • NavigateTo(Fridge) [with Lettuce in inventory]
    • PutObject(Fridge) [with Lettuce in inventory]
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
    • NavigateTo(Spoon)
    • PickUpObject(Spoon)
    • NavigateTo(Drawer) [with Spoon in inventory]
    • OpenObject(Drawer) [with Spoon in inventory]
    • PutObject(Drawer) [with Spoon in inventory]
    • OpenObject(Fridge)
    • CloseObject(Fridge)

Coverage:
    • ButterKnife
    • Spoon
    • Fork
    • Tomato
    • Lettuce
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
         'NavigateTo(Lettuce)',
         'PickUpObject(Lettuce)',
         'NavigateTo(Fridge, Lettuce)',
         'PutObject(Fridge, Lettuce)',
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
         'NavigateTo(Spoon)',
         'PickUpObject(Spoon)',
         'NavigateTo(Drawer, Spoon)',
         'OpenObject(Drawer, Spoon)',
         'PutObject(Drawer, Spoon)',
         'OpenObject(Fridge)',
         'CloseObject(Fridge)'
        ]



        conditional_subtasks = [
        'NavigateTo(Fridge, Tomato)',
         'PutObject(Fridge, Tomato)',
         'NavigateTo(Fridge, Lettuce)',
         'PutObject(Fridge, Lettuce)',
         'NavigateTo(Drawer, ButterKnife)',
         'OpenObject(Drawer, ButterKnife)',
         'PutObject(Drawer, ButterKnife)',
         'NavigateTo(Drawer, Fork)',
         'OpenObject(Drawer, Fork)',
         'PutObject(Drawer, Fork)',
         'NavigateTo(Drawer, Spoon)',
         'OpenObject(Drawer, Spoon)',
         'PutObject(Drawer, Spoon)'
        ]

        independent_subtasks = [
        'NavigateTo(Tomato)',
         'PickUpObject(Tomato)',
         'NavigateTo(Lettuce)',
         'PickUpObject(Lettuce)',
         'NavigateTo(ButterKnife)',
         'PickUpObject(ButterKnife)',
         'NavigateTo(Fork)',
         'PickUpObject(Fork)',
         'NavigateTo(Spoon)',
         'PickUpObject(Spoon)',
         'OpenObject(Fridge)',
         'CloseObject(Fridge)'
        ]


        # ButterKnife, spoon, fork, tomato, lettuce, fridge
        coverage = ["ButterKnife", "Spoon", "Fork", "Tomato", "Lettuce", "Fridge", "Drawer"]
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
