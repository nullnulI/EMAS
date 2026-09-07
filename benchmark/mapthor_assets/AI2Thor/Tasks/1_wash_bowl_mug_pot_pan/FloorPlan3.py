"""
Pre-initialization for FloorPlan3 task.
FloorPlan3 bowl, mug, pot, and pan were dirtied, pan was moved out to visible position
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

        event=controller.step(
            action='DirtyObject',
            objectId="Bowl|-01.82|+01.31|+00.27",
            forceAction=True,
        )

        event=controller.step(
            action='DirtyObject',
            objectId='Mug|-01.97|+01.04|-00.43',
            forceAction=True,
        )

        event=controller.step(
            action='DirtyObject',
            objectId='Pan|-01.88|+00.33|-02.04',
            forceAction=True,
        )

        event=controller.step(
            action='DirtyObject',
            objectId='Pot|-01.58|+01.31|-01.58',
            forceAction=True,
        )

        event = controller.step(
            action='PlaceObjectAtPoint',
            objectId='Pan|-01.88|+00.33|-02.04',
            position={'x': -1.8811466693878174, 'y': 1.36003920316696167, 'z': -2.035330057144165},
        )

        return event
