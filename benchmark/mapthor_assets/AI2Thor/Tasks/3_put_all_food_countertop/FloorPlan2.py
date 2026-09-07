"""
Pre-initalization for FloorPlan2 task.
FloorPlan2 needs to move the lettuce and apple out of the fridge
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

        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Apple|-01.65|+00.81|+00.07',
        position={'x': -0.11696076393127419, 'y': 1.8335982859134674, 'z': 0.46363327503204355}
        )

        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Lettuce|-01.71|+00.82|-00.14',
        position={'x': 1.623041033744812, 'y': 1.2524701476097106, 'z': -0.7363682210445406}
        )

        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Tomato|+00.17|+00.97|-00.28',
        position={'x': 0.16834744811058044, 'y': 0.9745080471038818, 'z': -0.28154730796813965}
        )

        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Bread|+00.26|+00.99|-00.08',
        position={'x': 0.25613734126091003, 'y': 0.987483024597168, 'z': -0.07600243389606476}
        )

        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Potato|+00.33|+00.94|+01.31',
        position={'x': 0.3274373412132263, 'y': 0.9400352835655212, 'z': 1.3115087747573853}
        )

        return event
