"""
Pre-initialization for FloorPlan1 task.
FloorPlan1 bowl, mug, pot, and pan were dirtied
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
            objectId="Bowl|+00.27|+01.10|-00.75",
            forceAction=True,
        )

        event=controller.step(
            action='DirtyObject',
            objectId='Mug|-01.76|+00.90|-00.62',
            forceAction=True,
        )

        event=controller.step(
            action='DirtyObject',
            objectId='Pot|-01.22|+00.90|-02.36',
            forceAction=True,
        )

        event=controller.step(
            action='DirtyObject',
            objectId='Pan|+00.72|+00.90|-02.42',
            forceAction=True,
        )

        return event
