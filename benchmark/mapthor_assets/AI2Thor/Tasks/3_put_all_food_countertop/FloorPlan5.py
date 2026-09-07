"""
Pre-initialization for FloorPlan4 task.
FloorPlan4 needs to move the tomato out of the fridge
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
        action='PlaceObjectAtPoint',
        objectId='Apple|+01.83|+00.78|-00.65',
        position={'x': -0.2872421264648439, 'y': 1.3173237442970276, 'z': -0.6516057252883911}
        )

        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Bread|-01.15|+00.98|-00.19',
        position={'x': -1.1480002403259277, 'y': 0.9816430807113647, 'z': -0.19300001859664917}
        )

        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Tomato|-01.15|+00.96|-01.66',
        position={'x': -1.1477510929107666, 'y': 0.9576261043548584, 'z': -1.6583662033081055}
        )

        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Lettuce|-00.25|+00.81|-01.92',
        position={'x': -1.0475484549999237, 'y': 1.1003166496753694, 'z': -0.4931554317474365}
        )



        return event
