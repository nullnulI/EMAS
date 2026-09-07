"""
Pre-initalization for FloorPlan1 task.
FloorPlan1 does not need any modifications for the task 
of transporting a groceries to the fridge
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

        return event
