"""
Pre-initalization for FloorPlan1 task.
FloorPlan3 does not need any modifications for the task of putting all the potatoes and tomatoes in the fridge
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
