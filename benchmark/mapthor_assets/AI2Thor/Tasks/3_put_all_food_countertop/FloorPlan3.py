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
        objectId='Bread|-01.51|+01.38|+00.66',
        position={'x': -1.5136306285858154, 'y': 1.3809161186218262, 'z': 0.6621981263160706}
        )

        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Lettuce|-01.89|+01.40|-01.07',
        position={'x': -1.892227053642273, 'y': 1.399842381477356, 'z': -1.0701444149017334}
        )

        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Tomato|+00.92|+01.91|+01.61',
        position={'x': -1.437359094619751, 'y': 2.4899539589881896, 'z': 0.906745195388794}
        )

        return event
