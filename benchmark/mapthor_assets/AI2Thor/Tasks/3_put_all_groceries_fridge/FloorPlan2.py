
"""Move bread, lettuce, tomato, apple, and potato visibly outside in the pre-initialization."""
    
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
        objectId='Lettuce|-01.71|+00.82|-00.14',
        position={'x': 0.028041750192642212, 'y': 0.9821591973304749, 'z': 0.46363192796707153}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Potato|+00.33|+00.94|+01.31',
        position={'x': 1.4874423742294312, 'y': 0.9250437021255493, 'z': 1.3115118741989136}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Egg|+00.13|+00.95|-00.17',
        position={'x': -1.6054714918136597, 'y': 1.3845981359481812, 'z': -0.16909930109977722}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Bread|+00.26|+00.99|-00.08',
        position={'x': -0.3238624036312103, 'y': 0.987483024597168, 'z': -0.07600240409374237}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Apple|-01.65|+00.81|+00.07',
        position={'x': 0.09092368185520172, 'y': 0.9697577357292175, 'z': 1.272389531135559}
        )
        
        return event
        