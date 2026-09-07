"""
    To check completion of subtasks in 1_put_remotecontrol_keys_watch_box.
    Also checks coverage of the task and success of the task.
    Subtasks:
    • NavigateTo(remotecontrol)
    • PickupObject(remotecontrol)
    • NavigateTo(Box) [with remotecontrol in inventory]
    • PutObject(Box) [with remotecontrol in inventory]
    • NavigateTo(KeyChain)
    • PickupObject(KeyChain)
    • NavigateTo(Box) [with KeyChain in inventory]
    • PutObject(Box) [with KeyChain in inventory]
    • NavigateTo(Watch)
    • PickupObject(Watch)
    • NavigateTo(Box) [with Watch in inventory]
    • PutObject(Box) [with Watch in inventory]
    Coverage:
        • Watch
        • RemoteControl
        • KeyChain
        • Box
"""

from AI2Thor.baselines.utils.checker import BaseChecker


class Checker(BaseChecker):
    def __init__(self) -> None:
        subtasks = [
            "NavigateTo(RemoteControl)",
            "PickupObject(RemoteControl)",
            "NavigateTo(Box, RemoteControl)",
            "PutObject(Box, RemoteControl)",
            "NavigateTo(KeyChain)",
            "PickupObject(KeyChain)",
            "NavigateTo(Box, KeyChain)",
            "PutObject(Box, KeyChain)",
            "NavigateTo(Watch)",
            "PickupObject(Watch)",
            "NavigateTo(Box, Watch)",
            "PutObject(Box, Watch)",
        ]
        conditional_subtasks = [
            "NavigateTo(Box, RemoteControl)",
            "PutObject(Box, RemoteControl)",
            "NavigateTo(Box, KeyChain)",
            "PutObject(Box, KeyChain)",
            "NavigateTo(Box, Watch)",
            "PutObject(Box, Watch)",
        ]
        independent_subtasks = [
            "NavigateTo(RemoteControl)",
            "PickupObject(RemoteControl)",
            "NavigateTo(KeyChain)",
            "PickupObject(KeyChain)",
            "NavigateTo(Watch)",
            "PickupObject(Watch)",
        ]
        coverage = ["RemoteControl", "Box", "KeyChain", "Watch"]
        interact_objects = ["RemoteControl", "KeyChain", "Watch"]
        interact_receptacles = ["Box"]

        super().__init__(
            subtasks,
            conditional_subtasks,
            independent_subtasks,
            coverage,
            interact_objects,
            interact_receptacles,
        )
