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
        apple_position = {'x': 2.76284122467041, 'y': 0, 'z': 0.4431222677230835}
        # set the tomato position on the floor
        tomato_position = {'x': 2.6983327865600586, 'y': 0, 'z': 1.8128788471221924}

        # Teleport the apple and tomato to the given positions on the floor
        event = controller.step(
            action="PlaceObjectAtPoint",
            objectId= 'Apple|+01.83|+00.78|-00.65',
            position=apple_position,
        )
        event = controller.step(
            action="PlaceObjectAtPoint",
            objectId= 'Tomato|-01.15|+00.96|-01.66',
            position=tomato_position,
        )

        return event