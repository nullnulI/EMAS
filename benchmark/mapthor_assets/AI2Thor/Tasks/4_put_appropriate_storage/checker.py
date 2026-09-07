"""
To check completion of subtasks
Also checks coverage of the task and success of the task.
Subtasks:
• NavigateTo(ButterKnife)
• PickUpObject(ButterKnife)
• NavigateTo(Drawer) [with ButterKnife in inventory]
• OpenObject(Drawer) [with ButterKnife in inventory]
• PutObject(Drawer) [with ButterKnife in inventory]
• NavigateTo(Knife)
• PickUpObject(Knife)
• NavigateTo(Drawer) [with Knife in inventory]
• OpenObject(Drawer) [with Knife in inventory]
• PutObject(Drawer) [with Knife in inventory]
• NavigateTo(Spatula)
• PickUpObject(Spatula)
• NavigateTo(Drawer) [with Spatula in inventory]
• OpenObject(Drawer) [with Spatula in inventory]
• PutObject(Drawer) [with Spatula in inventory]
• NavigateTo(Spoon)
• PickUpObject(Spoon)
• NavigateTo(Drawer) [with Spoon in inventory]
• OpenObject(Drawer) [with Spoon in inventory]
• PutObject(Drawer) [with Spoon in inventory]
• NavigateTo(Fork)
• PickUpObject(Fork)
• NavigateTo(Drawer) [with Fork in inventory]
• OpenObject(Drawer) [with Fork in inventory]
• PutObject(Drawer) [with Fork in inventory]

Coverage:
    • ButterKnife 
    • Knife
    • Spatula
    • Spoon
    • Fork
    • Ladle
"""


from AI2Thor.baselines.utils.checker import BaseChecker

class Checker(BaseChecker):
    def __init__(self) -> None:
        subtasks = [
         'NavigateTo(ButterKnife)',
         'PickUpObject(ButterKnife)',
         'NavigateTo(Drawer, ButterKnife)',
         'OpenObject(Drawer, ButterKnife)',
         'PutObject(Drawer, ButterKnife)',
         'NavigateTo(Knife)',
         'PickUpObject(Knife)',
         'NavigateTo(Drawer, Knife)',
         'OpenObject(Drawer, Knife)',
         'PutObject(Drawer, Knife)',
         'NavigateTo(Spatula)',
         'PickUpObject(Spatula)',
         'NavigateTo(Drawer, Spatula)',
         'OpenObject(Drawer, Spatula)',
         'PutObject(Drawer, Spatula)',
         'NavigateTo(Spoon)',
         'PickUpObject(Spoon)',
         'NavigateTo(Drawer, Spoon)',
         'OpenObject(Drawer, Spoon)',
         'PutObject(Drawer, Spoon)',
         'NavigateTo(Fork)',
         'PickUpObject(Fork)',
         'NavigateTo(Drawer, Fork)',
         'OpenObject(Drawer, Fork)',
         'PutObject(Drawer, Fork)'
         ]



        conditional_subtasks = [
        'NavigateTo(Drawer, ButterKnife)',
         'OpenObject(Drawer, ButterKnife)',
         'PutObject(Drawer, ButterKnife)',
         'NavigateTo(Drawer, Knife)',
         'OpenObject(Drawer, Knife)',
         'PutObject(Drawer, Knife)',
         'NavigateTo(Drawer, Spatula)',
         'OpenObject(Drawer, Spatula)',
         'PutObject(Drawer, Spatula)',
         'NavigateTo(Drawer, Spoon)',
         'OpenObject(Drawer, Spoon)',
         'PutObject(Drawer, Spoon)',
         'NavigateTo(Drawer, Fork)',
         'OpenObject(Drawer, Fork)',
         'PutObject(Drawer, Fork)'
        ]

        independent_subtasks = [
        'NavigateTo(ButterKnife)',
         'PickUpObject(ButterKnife)',
         'NavigateTo(Knife)',
         'PickUpObject(Knife)',
         'NavigateTo(Spatula)',
         'PickUpObject(Spatula)',
         'NavigateTo(Spoon)',
         'PickUpObject(Spoon)',
         'NavigateTo(Fork)',
         'PickUpObject(Fork)'
        ]

        coverage = ["ButterKnife", "Knife", "Spatula", "Spoon", "Fork", "Ladle", "Drawer"]
        interact_objects = coverage
        interact_receptacles = ["Drawer"]

        super().__init__(
            subtasks,
            conditional_subtasks,
            independent_subtasks,
            coverage,
            interact_objects,
            interact_receptacles,
        )
