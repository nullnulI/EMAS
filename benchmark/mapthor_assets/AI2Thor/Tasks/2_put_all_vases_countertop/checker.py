"""
To check completion of subtasks in 2_put_all_vases_CounterTop task.
Also checks coverage of the task and success of the task.
Subtasks:
    • NavigateTo(Vase_1)
    • PickupObject(Vase_1)
    • NavigateTo(CounterTop)
    • PutObject(CounterTop)
    • NavigateTo(Vase_2)
    • PickupObject(Vase_2)
    • NavigateTo(CounterTop)
    • PutObject(CounterTop)
    • NavigateTo(Vase_3) # vase 3 only exists in FloorPlan5
    • PickupObject(Vase_3)
    • NavigateTo(CounterTop)
    • PutObject(CounterTop)
Coverage:
    • Vase 1, 2, 3 # vase 3 only exists in FloorPlan5
    • CounterTop 1/2/3
"""


class Checker:
    def __init__(self) -> None:
        self.subtasks = [
            "NavigateTo(Vase)",  # independent
            "PickupObject(Vase)",  # independent
            "NavigateTo(CounterTop, Vase)",  # with vase in inventory
            "PutObject(CounterTop, Vase)",  # with vase in inventory
        ]
        # subtasks that are conditional on some other subtasks/state
        self.conditional_subtasks = [
            "NavigateTo(CounterTop, Vase)",
            "PutObject(CounterTop, Vase)",
        ]
        # subtasks that are not conditional on any other subtasks/state
        # basically depend on the object in the inventory (either before or after the action)
        self.independent_subtasks = [
            "NavigateTo(Vase)",  # independent
            "PickupObject(Vase)",  # independent
        ]
        self.coverage = ["Vase_1", "Vase_2", "CounterTop"]
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
        object_name, object_id = object.split("_")
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
        print(denum_act)
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
            denum_act in ["NavigateTo(CounterTop)"]
            and success
            and self.denumerate_object(inventory_object)
            in [
                "Vase",
            ]
        ):
            subtask_attempted = (
                "NavigateTo(CounterTop, "
                + self.denumerate_object(inventory_object)
                + ")"
            )
            if subtask_attempted not in self.subtasks_completed:
                self.subtasks_completed.append(subtask_attempted)
        # TODO: @nsidn98 check this reasoning:
        # since we are checking after put object is executed,
        # we need to check in previous inventory
        elif (
            denum_act in ["PutObject(CounterTop)"]
            and success
            and self.denumerate_object(inventory_object)
            in [
                "Vase",
            ]
        ):
            subtask_attempted = (
                "PutObject(Countertop, "
                + self.denumerate_object(inventory_object)
                + ")"
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
