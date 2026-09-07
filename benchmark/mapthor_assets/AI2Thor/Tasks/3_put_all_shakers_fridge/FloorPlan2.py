
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
        objectId='Apple|-01.65|+00.81|+00.07',
        position={'x': -1.650647759437561, 'y': 0.8054452538490295, 'z': 0.07250117510557175}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='PepperShaker|+01.89|+00.90|+00.03',
        position={'x': -0.14100012183189392, 'y': 0.9150844812393188, 'z': 0.3349994421005249}
        )
        
        return event
        