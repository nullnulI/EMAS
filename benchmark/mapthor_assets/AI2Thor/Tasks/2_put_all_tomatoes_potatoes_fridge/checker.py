"""
To check completion of subtasks in 2_put_all_tomatoes_potatoes_fridge task.
Also checks coverage of the task and success of the task.
Subtasks:
    • NavigateTo(Potato)
    • PickupObject(Potato)
    • NavigateTo(Fridge) [with Potato in inventory]
    • OpenObject(Fridge) [no need for objects in inventory, need to check only once]
    • PutObject(Fridge) [with Potato in inventory]
    • NavigateTo(Tomato)
    • PickupObject(Tomato)
    • NavigateTo(Fridge) [with tomato in inventory]
    • OpenObject(Fridge) [no need for objects in inventory]
    • PutObject(Fridge) [with tomato in inventory]
Coverage:
    • Potato
    • Fridge
    • Tomato
"""


class Checker:
    def __init__(self) -> None:
        self.subtasks = [
            "NavigateTo(Potato)",  # independent
            "PickupObject(Potato)",  # independent
            "NavigateTo(Fridge, Potato)",  # with Potato in inventory
            "PutObject(Fridge, Potato)",  # with Potato in inventory
            "NavigateTo(Tomato)",  # independent
            "PickupObject(Tomato)",  # independent
            "NavigateTo(Fridge, Tomato)",  # with tomato in inventory
            "PutObject(Fridge, Tomato)",  # with tomato in inventory
            "OpenObject(Fridge)",  # independent
            "CloseObject(Fridge)",  # independent
        ]
        # subtasks that are conditional on some other subtasks/state
        self.conditional_subtasks = [
            "NavigateTo(Fridge, Potato)",
            "NavigateTo(Fridge, Tomato)",
            "PutObject(Fridge, Potato)",
            "PutObject(Fridge, Tomato)",
        ]
        # subtasks that are not conditional on any other subtasks/state
        # basically depend on the object in the inventory (either before or after the action)
        self.independent_subtasks = [
            "OpenObject(Fridge)",
            "CloseObject(Fridge)",
            "NavigateTo(Potato)",
            "NavigateTo(Tomato)",
            "PickupObject(Potato)",
            "PickupObject(Tomato)",
        ]
        self.coverage = ["Potato", "Fridge", "Tomato"]
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
        Example: NavigateTo(Potato_1) -> NavigateTo(Potato)
        """
        act, object = self.split_action(action)
        object_name = self.denumerate_object(object)
        return act + "(" + object_name + ")"

    def denumerate_object(self, object: str) -> str:
        """
        Remove object number from object_id
        Example: Potato_1 -> Potato
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
            if denum_act.startswith("PickupObject("):
                object_type = denum_act.removeprefix("PickupObject(").removesuffix(")")
                navigate = f"NavigateTo({object_type})"
                if navigate not in self.subtasks_completed:
                    self.subtasks_completed.append(navigate)
        # elif we need to check for conditional subtasks
        # first check for NavigateTo(Fridge, object_in_inventory)
        elif (
            denum_act in ["NavigateTo(Fridge)"]
            and success
            and self.denumerate_object(inventory_object)
            in [
                "Potato",
                "Tomato",
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
                "Potato",
                "Tomato",
            ]
        ):
            subtask_attempted = (
                "PutObject(Fridge, " + self.denumerate_object(inventory_object) + ")"
            )
            if subtask_attempted not in self.subtasks_completed:
                self.subtasks_completed.append(subtask_attempted)
            navigate = (
                "NavigateTo(Fridge, "
                + self.denumerate_object(inventory_object)
                + ")"
            )
            if navigate not in self.subtasks_completed:
                self.subtasks_completed.append(navigate)

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
