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
        apple_position = {'x': 1.4710159301757812, 'y': 0, 'z': 2.538677215576172}
        # set the tomato position on the floor
        tomato_position = {'x': 1.4710159301757812, 'y': 0, 'z': 1.660527229309082}

        # Teleport the apple and tomato to the given positions on the floor
        event = controller.step(
            action="PlaceObjectAtPoint",
            objectId='Apple|-01.65|+00.81|+00.07',
            position=apple_position,
        )
        event = controller.step(
            action="PlaceObjectAtPoint",
            objectId='Tomato|+00.17|+00.97|-00.28',
            position=tomato_position,
        )

        return event