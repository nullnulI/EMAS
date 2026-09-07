"""
Pre-initalization for FloorPlan201 task.
FloorPlan201 does not need any modifications for the task of putting all the credit cards and remote controls in the box
"""


class SceneInitializer:
    def __init__(self) -> None:
        pass

    def preinit(self, event, controller):
        """Pre-initialize the environment for the task.

        Args:
            event: env.event object
            controller: ai2thor.controller object

        Returns:
            event: env.event object
        """
        return event
