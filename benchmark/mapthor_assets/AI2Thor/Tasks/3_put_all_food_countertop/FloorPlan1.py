"""
Pre-initalization for FloorPlan1 task.
FloorPlan1 does not need any modifications for the task
of transporting a groceries to the fridge
"""


class SceneInitializer:
    def __init__(self) -> None:
        pass

    def preinit(event, controller):
        """Pre-initialize the environment for the task.

        Args:
            event: env.event object
            controller: ai2thor.controller object

        Returns:
            event: env.event object
        """

        return event
