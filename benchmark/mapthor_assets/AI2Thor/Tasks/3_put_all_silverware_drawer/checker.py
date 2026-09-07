"""Official metrics for putting every silverware object in one drawer."""

from AI2Thor.baselines.utils.checker import BaseChecker


SILVERWARE = ("ButterKnife", "Knife", "Spatula", "Spoon", "Fork", "Ladle")
DRAWER_STATE_SUBTASKS = ("OpenObject(Drawer)", "CloseObject(Drawer)")


def _object_subtasks(object_type):
    return [
        f"NavigateTo({object_type})",
        f"PickUpObject({object_type})",
        f"NavigateTo(Drawer, {object_type})",
        f"PutObject(Drawer, {object_type})",
    ]


def _conditional_subtasks(object_type):
    return [
        f"NavigateTo(Drawer, {object_type})",
        f"PutObject(Drawer, {object_type})",
    ]


def _independent_subtasks(object_type):
    return [
        f"NavigateTo({object_type})",
        f"PickUpObject({object_type})",
    ]


class Checker(BaseChecker):
    def __init__(self) -> None:
        subtasks = [item for kind in SILVERWARE for item in _object_subtasks(kind)]
        subtasks.extend(DRAWER_STATE_SUBTASKS)
        conditional_subtasks = [
            item for kind in SILVERWARE for item in _conditional_subtasks(kind)
        ]
        independent_subtasks = [
            item for kind in SILVERWARE for item in _independent_subtasks(kind)
        ]
        coverage = [*SILVERWARE, "Drawer"]

        super().__init__(
            subtasks,
            conditional_subtasks,
            independent_subtasks,
            coverage,
            list(SILVERWARE),
            ["Drawer"],
        )

    def all_objects(self, obj_ids, scene):
        super().all_objects(obj_ids, scene)
        # BaseChecker rebuilds subtasks and would otherwise either omit
        # Open/Close or require them for every drawer instance in the scene.
        if "Drawer" in self.interact_receptacles:
            self.subtasks.extend(DRAWER_STATE_SUBTASKS)

    def check_subtask(self, action, success, inventory_object):
        if success:
            denumerated = self.denumerate_action(action)
            if denumerated in DRAWER_STATE_SUBTASKS:
                if denumerated not in self.subtasks_completed:
                    self.subtasks_completed.append(denumerated)
                    self.subtasks_completed_numerated.append(denumerated)
                return
        super().check_subtask(action, success, inventory_object)
