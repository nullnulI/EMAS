"""
To check completion of subtasks in 3_put_all_groceries_fridge task.
Also checks coverage of the task and success of the task.
Subtasks:
    • NavigateTo(Bread)
    • PickupObject(Bread)
    • NavigateTo(Fridge, Bread) [with bread in inventory]
    • PutObject(Fridge, Bread) [with bread in inventory]
    • NavigateTo(Tomato)
    • PickupObject(Tomato)
    • NavigateTo(Fridge, Tomato) [with tomato in inventory]
    • PutObject(Fridge, Tomato) [with tomato in inventory]
    • NavigateTo(Lettuce)
    • PickupObject(Lettuce)
    • NavigateTo(Fridge, Lettuce) [with lettuce in inventory]
    • PutObject(Fridge, Lettuce) [with lettuce in inventory]
    • NavigateTo(Apple)
    • PickupObject(Apple)
    • NavigateTo(Fridge, Apple) [with apple in inventory]
    • PutObject(Fridge, Apple)  [with apple in inventory]
    • NavigateTo(Potato)
    • PickupObject(Potato)
    • NavigateTo(Fridge, Potato) [with potato in inventory]
    • PutObject(Fridge, Potato) [with potato in inventory]
    • OpenObject(Fridge) [no need for objects in inventory]
    • CloseObject(Fridge)
Coverage:
    • Bread
    • Fridge
    • Tomato
    • Lettuce
    • Apple
    • Potato
"""


class Checker:
    def __init__(self) -> None:
        self.subtasks = [
            "NavigateTo(Bread)",  # independent
            "PickupObject(Bread)",  # independent
            "NavigateTo(Fridge, Bread)",  # with bread in inventory
            "PutObject(Fridge, Bread)",  # with bread in inventory
            "NavigateTo(Tomato)",  # independent
            "PickupObject(Tomato)",  # independent
            "NavigateTo(Fridge, Tomato)",  # with tomato in inventory
            "PutObject(Fridge, Tomato)",  # with tomato in inventory
            "NavigateTo(Lettuce)",  # independent
            "PickupObject(Lettuce)",  # independent
            "NavigateTo(Fridge, Lettuce)",  # with lettuce in inventory
            "PutObject(Fridge, Lettuce)",  # with lettuce in inventory
            "NavigateTo(Apple)",  # independent
            "PickupObject(Apple)",  # independent
            "NavigateTo(Fridge, Apple)",  # with lettuce in inventory
            "PutObject(Fridge, Apple)",  # with lettuce in inventory
            "NavigateTo(Potato)",  # independent
            "PickupObject(Potato)",  # independent
            "NavigateTo(Fridge, Potato)",  # with lettuce in inventory
            "PutObject(Fridge, Potato)",  # with lettuce in inventory
            "OpenObject(Fridge)",  # independent
            "CloseObject(Fridge)",  # independent
        ]
        # subtasks that are conditional on some other subtasks/state
        self.conditional_subtasks = [
            "NavigateTo(Fridge, Bread)",
            "NavigateTo(Fridge, Tomato)",
            "NavigateTo(Fridge, Lettuce)",
            "NavigateTo(Fridge, Apple)",
            "NavigateTo(Fridge, Potato)",
            "PutObject(Fridge, Bread)",
            "PutObject(Fridge, Tomato)",
            "PutObject(Fridge, Lettuce)",
            "PutObject(Fridge, Apple)",
            "PutObject(Fridge, Potato)",
        ]
        # subtasks that are not conditional on any other subtasks/state
        # basically depend on the object in the inventory (either before or after the action)
        self.independent_subtasks = [
            "OpenObject(Fridge)",
            "CloseObject(Fridge)",
            "NavigateTo(Bread)",
            "NavigateTo(Tomato)",
            "NavigateTo(Lettuce)",
            "NavigateTo(Apple)",
            "NavigateTo(Potato)",
            "PickupObject(Bread)",
            "PickupObject(Tomato)",
            "PickupObject(Lettuce)",
            "PickupObject(Apple)",
            "PickupObject(Potato)",
        ]
        self.coverage = ["Bread", "Fridge", "Tomato", "Lettuce", "Apple", "Potato"]
        self.subtasks_completed = []
        self.coverage_completed = []

    def split_action(self, action: str):
        """Split action string into action and object"""
        act = action.split("(")[0]
        object = action.split("(")[1].split(")")[0]
        return act, object

    def denumerate_action(self, action: str):
        """
        Remove object number from object_id
        Example: NavigateTo(Bread_1) -> NavigateTo(Bread)
        """
        act, object = self.split_action(action)
        object_name = self.denumerate_object(object)
        # object_name, object_id = object.split("_")
        return act + "(" + object_name + ")"

    def denumerate_object(self, object: str) -> str:
        """
        Remove object number from object_id
        Example: Bread_1 -> Bread
        """
        if "_" in object:
            object_name, object_id = object.split("_")
            return object_name
        else:
            return object

    def check_subtask(
        self,
        action,
        success,
        inventory_object,  # object in inventory after the action
    ):
        if action in ["Done", "Idle"]:
            return None
        denum_act = self.denumerate_action(action)
        # first check for indpendent subtasks
        # if the action is in independent subtasks and not in subtasks_completed
        # and the action was successful, then add it to subtasks_completed
        if (
            denum_act in self.independent_subtasks
            and denum_act not in self.subtasks_completed
            and success
        ):
            self.subtasks_completed.append(self.denumerate_action(action))
        # elif we need to check for conditional subtasks
        # first check for NavigateTo(Fridge, object_in_inventory)
        elif (
            denum_act in ["NavigateTo(Fridge)"]
            and success
            and self.denumerate_object(inventory_object)
            in [
                "Bread",
                "Tomato",
                "Lettuce",
                "Apple",
                "Potato",
            ]
        ):
            subtask_attempted = (
                "NavigateTo(Fridge, " + self.denumerate_object(inventory_object) + ")"
            )
            if subtask_attempted not in self.subtasks_completed:
                self.subtasks_completed.append(subtask_attempted)
        # TODO: @nsidn98 check this reasoning:
        # since we are checking after put object is executed,
        # we need to check in previous inventory
        elif (
            denum_act in ["PutObject(Fridge)"]
            and success
            and self.denumerate_object(inventory_object)
            in [
                "Bread",
                "Tomato",
                "Lettuce",
                "Apple",
                "Potato",
            ]
        ):
            subtask_attempted = (
                "PutObject(Fridge, " + self.denumerate_object(inventory_object) + ")"
            )
            if subtask_attempted not in self.subtasks_completed:
                self.subtasks_completed.append(subtask_attempted)

    def get_transport_rate(self):
        return len(self.subtasks_completed) / len(self.subtasks)

    def check_coverage(self, action: str):
        for object in self.coverage:
            if object in action and object not in self.coverage_completed:
                self.coverage_completed.append(object)

    def perform_metric_check(self, action, success, inventory_object):
        self.check_subtask(action, success, inventory_object)
        self.check_coverage(action)

    def get_coverage(self):
        return len(self.coverage_completed) / len(self.coverage)

    def check_success(self):
        return len(self.subtasks_completed) == len(self.subtasks)
