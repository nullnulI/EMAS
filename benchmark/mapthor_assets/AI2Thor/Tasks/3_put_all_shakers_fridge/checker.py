"""
To check completion of subtasks in 2_put_all_tomatoes_potatoes_fridge task.
Also checks coverage of the task and success of the task.
Subtasks:
• NavigateTo(SaltShaker)
• PickUpObject(SaltShaker)
• NavigateTo(Fridge) [with SaltShaker in inventory]
• PutObject(Fridge) [with SaltShaker in inventory]
• NavigateTo(PepperShaker)
• PickUpObject(PepperShaker)
• NavigateTo(Fridge) [with PepperShaker in inventory]
• PutObject(Fridge) [with PepperShaker in inventory]
• OpenObject(Fridge) [no need for objects in inventory, need to check only once]
• CloseObject(Fridge)

Coverage:
    • SaltShaker
    • PepperShaker
"""


from AI2Thor.baselines.utils.checker import BaseChecker

class Checker(BaseChecker):
    def __init__(self) -> None:
        subtasks = [
         'NavigateTo(SaltShaker)',
         'PickUpObject(SaltShaker)',
         'NavigateTo(Fridge, SaltShaker)',
         'PutObject(Fridge, SaltShaker)',
         'NavigateTo(PepperShaker)',
         'PickUpObject(PepperShaker)',
         'NavigateTo(Fridge, PepperShaker)',
         'PutObject(Fridge, PepperShaker)',
         'OpenObject(Fridge)',
         'CloseObject(Fridge)']

        conditional_subtasks = [
         'NavigateTo(Fridge, SaltShaker)',
         'PutObject(Fridge, SaltShaker)',
         'NavigateTo(Fridge, PepperShaker)',
         'PutObject(Fridge, PepperShaker)']

        independent_subtasks = [
         'NavigateTo(SaltShaker)',
         'PickUpObject(SaltShaker)',
         'NavigateTo(PepperShaker)',
         'PickUpObject(PepperShaker)',
         'OpenObject(Fridge)',
         'CloseObject(Fridge)']

        coverage = ["Fridge", "SaltShaker", "PepperShaker"]
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
