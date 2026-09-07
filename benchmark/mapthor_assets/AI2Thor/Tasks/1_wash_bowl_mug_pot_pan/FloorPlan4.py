"""
Pre-initialization for FloorPlan4 task.
FloorPlan4 bowl, mug, pot, and pan were dirtied
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
            objectId="Bowl|-00.50|+01.04|+02.22",
            forceAction=True,
        )

        event=controller.step(
            action='DirtyObject',
            objectId='Mug|-00.38|+01.09|+03.19',
            forceAction=True,
        )

        event=controller.step(
            action='DirtyObject',
            objectId='Pan|-00.27|+01.04|+02.56',
            forceAction=True,
        )

        event=controller.step(
            action='DirtyObject',
            objectId='Pot|-03.00|+01.14|+00.48',
            forceAction=True,
        )

        return event
