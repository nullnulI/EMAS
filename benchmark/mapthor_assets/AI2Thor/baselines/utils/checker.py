"""
    General checker to check completion of subtasks.
    Also checks coverage of the task and success of the task.
"""
from collections import Counter
from functools import reduce

class BaseChecker:
    def __init__(
            self,
            subtasks,
            conditional_subtasks,
            independent_subtasks,
            coverage,
            interact_objects,
            interact_receptacles
            ) -> None:

        self.subtasks = subtasks

        # subtasks that are conditional on some other subtasks/state
        self.conditional_subtasks = conditional_subtasks

        # subtasks that are not conditional on any other subtasks/state
        # basically depend on the object in the inventory (either before or after the action)
        self.independent_subtasks = independent_subtasks

        # objects or receptacles
        self.coverage = coverage
        self.coverage_completed = [] 

        # objects to be interacted with
        self.interact_objects = interact_objects

        # receptacles to be used
        self.interact_receptacles = interact_receptacles

        # all actions in conditional subtask
        self.init_conditional_acts()
        self.subtasks_completed = []

        # to store the subtasks completed with object number (@coela)
        self.subtasks_completed_numerated = []

    def split_action(self, action: str):
        """Split action string into action and object"""
        act = action.split("(")[0]
        obj = action.split("(")[1].split(")")[0]
        if "," in obj:
            obj=obj.split(", ") # obj can be list
        return act, obj 

    def denumerate_action(self, action: str):
        """
        Remove object number from object_id
        Example: NavigateTo(Bread_1) -> NavigateTo(Bread)
        """
        act, object = self.split_action(action)
        object_name = self.denumerate_object(object)
        return f"{act}({object_name})"

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

    def init_conditional_acts(self):
        conditional_acts = set()
        for action in self.conditional_subtasks:
            act = action.split("(")[0]
            conditional_acts.add(act)
        self.conditional_acts = list(conditional_acts)

    def give_credit_for_navigate(self, action, success, inventory_object):
        act, object = self.split_action(action)
        if act not in ["NavigateTo"]:
            navigate_action = f"NavigateTo({object})"
            self.check_subtask(navigate_action, success, inventory_object)

    def check_subtask(
        self,
        action,
        success,
        inventory_object,  # object in inventory after the action
    ):
        if action in ["Done", "Idle"]:
            return None

        if "SendMessage" in action:
            return None

        denum_act = self.denumerate_action(action)

        # first check for independent subtasks
        # if the action is in independent subtasks and not in subtasks_completed
        # and the action was successful, then add it to subtasks_completed
        # check numerated action in independent_subtasks (since it'll be numerated in list)

        if (
            action in self.independent_subtasks
            and action not in self.subtasks_completed_numerated
            and success
        ):
            self.subtasks_completed.append(denum_act)
            self.subtasks_completed_numerated.append(action)

            # Give credit for NavigateTo(object) if action(object) is successful
            # Since this means the agent just happened to already be close enough
            self.give_credit_for_navigate(action, success, inventory_object)

        elif self.conditional_subtasks is not None:
            # elif we need to check for conditional subtasks
            # first check for NavigateTo(Fridge, object_in_inventory)
            # TODO: @nsidn98 check this reasoning:
            # since we are checking after put object is executed,
            # we need to check in previous inventory
            for receptacle in self.interact_receptacles:
                for act in self.conditional_acts:
                    if (
                        denum_act in [f"{act}({receptacle})"]
                        and success
                        and self.denumerate_object(inventory_object) in self.interact_objects
                    ):
                        subtask_attempted = (
                            f"{act}({receptacle}, {self.denumerate_object(inventory_object)})"
                        )
                        # Inventory-based enrichment is only normalization.  It must
                        # never manufacture credit for an action the task did not require.
                        if subtask_attempted not in self.conditional_subtasks:
                            continue
                        if subtask_attempted not in self.subtasks_completed:
                            self.subtasks_completed.append(subtask_attempted)
                            act_receptacle = action.split(")")[0]
                            action_object = f"{act_receptacle}, {inventory_object})"
                            self.subtasks_completed_numerated.append(action_object)

                            # Give credit for NavigateTo(interact_receptacle, object_in_inventory) if action(interact_receptacle, object_in_inventory) is successful
                            # Since this means the agent just happened to already be close enough
                            self.give_credit_for_navigate(action, success, inventory_object)

    def get_transport_rate(self):
        return len(self.subtasks_completed_numerated) / len(self.subtasks)

    def all_objects(self, obj_ids, scene):
        # run function when reseting the environment in the env (each env instance has checker member)
        # functionality so that objects with more than one instance (e.g. drawers) are accounted for in the metrics (coverage, subtasks, etc)

        nm_ids = [oid.split('|')[0] for oid in obj_ids]
        # if exists self.manual_override_nms, then filter nm_ids by those existing there
        # NOTE: This allows to restrict to not only existing objects in scene, but subset of them
        if hasattr(self, "manual_override_nms"):
            nm_ids=self.manual_override_nms[scene]

        self.obj_counter = Counter(nm_ids)

        # if exists self.manual_override_numerate, then replace it by that
        # NOTE: This allows to only add n objects of a certain type instead of all of them
        if hasattr(self, "manual_override_numerate"):
            for k,v in self.manual_override_numerate.items():
                if k in self.obj_counter.keys():    self.obj_counter[k]=v

        # return list as numerated version of list l
        def numerate(l):
            ll = [[f'{nm}_{i+1}' for i in range(self.obj_counter[nm])] for nm in l]
            numerated_objs = reduce(lambda x,y:x+y, ll)
            return numerated_objs

        # numerate objects in coverage
        self.coverage = numerate(self.coverage)

        # ASSUMPTION:
        # Since there is ambiguity for conditional subtasks w/ using receptacles w/ more than 1 copy (navigateto(cabinet) w/ potato),
        # we just assume it's complete if it goes to any such receptacle  (no specifit one)

        # Thus, we will only modify independent subtasks (numerate them)
        nind_tsks = []
        for act in self.independent_subtasks:
            a, o = self.split_action(act)
            nlst = numerate([o])
            nitsk = [f"{a}({no})" for no in nlst]
            nind_tsks += nitsk

        self.independent_subtasks = nind_tsks

        # modify subtasks list - loses any order
        self.subtasks = self.independent_subtasks + self.conditional_subtasks

        # NOTE: Filter out all the subtasks that don't apply to this current floorplan
        # TODO: check on this, @here
        def exists(s):
            return any([(nm in s) for nm in nm_ids])

        def all_exist(action):
            a,objs=self.split_action(action)
            if not isinstance(objs, list): objs=[objs]
            for o in objs:
                if not exists(o):
                    return False
            return True

        self.coverage=list(filter(exists, self.coverage))
        self.interact_objects=list(filter(exists, self.interact_objects))
        self.interact_receptacles=list(filter(exists, self.interact_receptacles))
        self.independent_subtasks=list(filter(all_exist, self.independent_subtasks))
        self.conditional_subtasks=list(filter(all_exist, self.conditional_subtasks))

        self.subtasks=self.independent_subtasks + self.conditional_subtasks

    def check_coverage(self, action: str):
        # when we get here, self.coverage includes the numeration (i.e. "Drawer_2") for accurate measurement of coverage
        for object in self.coverage:
            if object in action and object not in self.coverage_completed:
                self.coverage_completed.append(object)

    def perform_metric_check(self, action, success, inventory_object):
        self.check_subtask(action, success, inventory_object)
        self.check_coverage(action)

    def get_coverage(self):
        return len(self.coverage_completed) / len(self.coverage)

    def check_success(self):
        return len(self.subtasks_completed_numerated) == len(self.subtasks)
