"""
Pre-initialization for FloorPlan2 task.
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
            objectId="Bowl|+00.28|+00.92|+01.09",
            forceAction=True,
        )

        event=controller.step(
            action='DirtyObject',
            objectId='Mug|-00.24|+00.92|-00.26',
            forceAction=True,
        )

        event=controller.step(
            action='DirtyObject',
            objectId='Pot|+00.89|+00.90|-01.41',
            forceAction=True,
        )

        event=controller.step(
            action='DirtyObject',
            objectId='Pan|-01.29|+00.90|-01.35',
            forceAction=True,
        )

        return event
