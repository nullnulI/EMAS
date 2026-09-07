"""
Pre-initalization for FloorPlan2 task.
FloorPlan3 needs to move the tomato out of the fridge
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

        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Tomato|+00.92|+01.91|+01.61',
        position={'x': -1.437359094619751, 'y': 2.4899539589881896, 'z': 0.906745195388794}
        )

        return event
