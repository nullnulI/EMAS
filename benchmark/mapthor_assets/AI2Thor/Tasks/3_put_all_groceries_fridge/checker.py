"""
To check completion of subtasks
Also checks coverage of the task and success of the task.
Subtasks:
    • NavigateTo(Bread)
    • PickUpObject(Bread)
    • NavigateTo(Fridge) [with bread in inventory]
    • OpenObject(Fridge)
    • PutObject(Fridge) [with bread in inventory]
    • NavigateTo(Tomato)
    • PickUpObject(Tomato)
    • NavigateTo(Fridge) [with tomato in inventory]
    • OpenObject(Fridge) [no need for objects in inventory]
    • PutObject(Fridge) [with tomato in inventory]
    • NavigateTo(Lettuce)
    • PickUpObject(Lettuce)
    • NavigateTo(Fridge) [with lettuce in inventory]
    • OpenObject(Fridge)
    • PutObject(Fridge) [with lettuce in inventory]
    • NavigateTo(Potato)
    • PickUpObject(Potato)
    • NavigateTo(Fridge) [with potato in inventory]
    • OpenObject(Fridge)
    • PutObject(Fridge) [with potato in inventory]
    • NavigateTo(Apple)
    • PickUpObject(Apple)
    • NavigateTo(Fridge) [with apple in inventory]
    • OpenObject(Fridge)
    • PutObject(Fridge) [with apple in inventory]
    • CloseObject(Fridge)

Coverage:
    • Bread 
    • Lettuce 
    • Tomato 
    • Potato 
    • Apple 
    • Fridge 
"""


from AI2Thor.baselines.utils.checker import BaseChecker

class Checker(BaseChecker):
    def __init__(self) -> None:
        subtasks = [
         'NavigateTo(Bread)',
         'PickUpObject(Bread)',
         'NavigateTo(Fridge, Bread)',
         'PutObject(Fridge, Bread)',
         'NavigateTo(Tomato)',
         'PickUpObject(Tomato)',
         'NavigateTo(Fridge, Tomato)',
         'PutObject(Fridge, Tomato)',
         'NavigateTo(Lettuce)',
         'PickUpObject(Lettuce)',
         'NavigateTo(Fridge, Lettuce)',
         'PutObject(Fridge, Lettuce)',
         'NavigateTo(Potato)',
         'PickUpObject(Potato)',
         'NavigateTo(Fridge, Potato)',
         'PutObject(Fridge, Potato)',
         'NavigateTo(Apple)',
         'PickUpObject(Apple)',
         'NavigateTo(Fridge, Apple)',
         'PutObject(Fridge, Apple)',

         'OpenObject(Fridge)',
         'CloseObject(Fridge)'
         ]


        conditional_subtasks = [
             'NavigateTo(Fridge, Bread)',
             'PutObject(Fridge, Bread)',
             'NavigateTo(Fridge, Tomato)',
             'PutObject(Fridge, Tomato)',
             'NavigateTo(Fridge, Lettuce)',
             'PutObject(Fridge, Lettuce)',
             'NavigateTo(Fridge, Potato)',
             'PutObject(Fridge, Potato)',
             'NavigateTo(Fridge, Apple)',
             'PutObject(Fridge, Apple)'
         ]

        independent_subtasks = [
             'NavigateTo(Bread)',
             'PickUpObject(Bread)',
             'NavigateTo(Tomato)',
             'PickUpObject(Tomato)',
             'NavigateTo(Lettuce)',
             'PickUpObject(Lettuce)',
             'NavigateTo(Potato)',
             'PickUpObject(Potato)',
             'NavigateTo(Apple)',
             'PickUpObject(Apple)',

             'OpenObject(Fridge)',
             'CloseObject(Fridge)'
         ]

        coverage = ["Bread", "Lettuce", "Tomato", "Potato", "Apple", "Fridge"]
        interact_objects = coverage
        interact_receptacles = ["Fridge"]

        super().__init__(
            subtasks,
            conditional_subtasks,
            independent_subtasks,
            coverage,
            interact_objects,
            interact_receptacles,
        )
