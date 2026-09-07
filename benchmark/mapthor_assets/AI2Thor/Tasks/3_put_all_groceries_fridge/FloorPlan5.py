
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
        objectId='Lettuce|-00.25|+00.81|-01.92',
        position={'x': 1.3356760740280151, 'y': 0.9552526473999023, 'z': -1.9169596433639526}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Apple|+01.83|+00.78|-00.65',
        position={'x': 0.24276965856552124, 'y': 1.1693910360336304, 'z': 0.535920262336731}
        )
        
        return event
        