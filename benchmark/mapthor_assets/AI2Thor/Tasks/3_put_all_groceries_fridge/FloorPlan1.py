
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
        objectId='Bread|-00.52|+01.17|-00.03',
        position={'x': 0.9710586667060852, 'y': 0.970058798789978, 'z': -2.225581169128418}
        )
        
        return event
        