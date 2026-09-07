"""
Put all Apple and Tomato on the floor
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
        # set the apple position on the floor
        apple_position = {'x': 0.5408684015274048, 'y': 0, 'z': 2.2864272594451904}
        # set the tomato position on the floor
        tomato_position = {'x': 0.5408684015274048, 'y': 0, 'z': -1.3342022895812988}

        # Teleport the apple and tomato to the given positions on the floor
        event = controller.step(
            action="PlaceObjectAtPoint",
            objectId='Apple|-01.75|+01.37|-01.16',
            position=apple_position,
        )
        event = controller.step(
            action="PlaceObjectAtPoint",
            objectId='Tomato|+00.92|+01.91|+01.61',
            position=tomato_position,
        )

        return event