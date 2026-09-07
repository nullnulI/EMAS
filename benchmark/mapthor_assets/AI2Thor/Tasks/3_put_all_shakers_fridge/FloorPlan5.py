
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
        objectId='SaltShaker|+01.19|+00.90|-02.18',
        position={'x': -0.3982231020927429, 'y': 0.9000833630561829, 'z': 0.1997799277305603}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Apple|+01.83|+00.78|-00.65',
        position={'x': 1.8327804803848267, 'y': 0.7751184105873108, 'z': -0.6516042351722717}
        )
        
        return event
        