
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
        objectId='PepperShaker|+00.47|+01.31|-03.00',
        position={'x': 0.7026405334472656, 'y': 1.3104068040847778, 'z': -1.955133080482483}
        )
        
        return event
        