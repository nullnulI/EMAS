"""
Pre-initalization for FloorPlan1 task.
This will place the apple near the stove on the floor 
and tomato near the stove and island on the floor.
# NOTE: @UROPs Add docstrings like this at the top to explain 
what kind of changes does the function make to the environment
This is just an example file to show how to preinit the environment for the task.
PLease modify this task too as per the overleaf document.
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
        apple_position = {"x": 1.1306087523698807, "y": 0, "z": -1.6201415956020355}
        # set the tomato position on the floor
        tomato_position = {
            "x": 0.8448016047477722,
            "y": 0,
            "z": -1.3311898310979209,
        }

        # Teleport the apple and tomato to the given positions on the floor
        event = controller.step(
            action="PlaceObjectAtPoint",
            objectId="Apple|-00.47|+01.15|+00.48",
            position=apple_position,
        )
        event = controller.step(
            action="PlaceObjectAtPoint",
            objectId="Tomato|-00.39|+01.14|-00.81",
            position=tomato_position,
        )

        return event
