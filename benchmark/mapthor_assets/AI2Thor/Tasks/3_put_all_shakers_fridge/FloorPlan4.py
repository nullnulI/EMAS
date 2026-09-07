
"""Move salt shaker and pepper shaker apart at initialization"""
    
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
        objectId='PepperShaker|-03.76|+01.11|+00.39',
        position={'x': -0.7843236923217773, 'y': 1.0383224487304688, 'z': 2.951219081878662}
        )
        
        return event
        