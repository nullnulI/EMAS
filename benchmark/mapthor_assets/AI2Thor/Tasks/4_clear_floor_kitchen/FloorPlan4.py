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
        apple_position = {'x': -0.0033719539642333984,'y':0,'z': 3.494899034500122}
        # set the tomato position on the floor
        tomato_position = {'x': -0.20256537199020386, 'y':0, 'z': 3.085575580596924}

        # Teleport the apple and tomato to the given positions on the floor
        event = controller.step(
            action="PlaceObjectAtPoint",
            objectId= 'Apple|-02.48|+01.18|+00.48',
            position=apple_position,
        )
        event = controller.step(
            action="PlaceObjectAtPoint",
            objectId= 'Tomato|-00.75|+01.08|+02.40',
            position=tomato_position,
        )

        return event