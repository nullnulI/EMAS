
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
        objectId='PepperShaker|+00.30|+00.90|-02.47',
        position={'x': -0.19740967452526093, 'y': 1.100084900856018, 'z': -0.2675131559371948}
        )
        
        return event
        