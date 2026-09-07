
"""Close all cabinets for pre-initialization"""
    
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
        action='OpenObject',
        objectId='Cabinet|+00.68|+00.50|-02.20',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-01.18|+00.50|-02.20',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-01.55|+00.50|+00.38',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+00.72|+02.02|-02.46',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-01.85|+02.02|+00.38',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+00.68|+02.02|-02.46',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-01.55|+00.50|-01.97',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-01.69|+02.02|-02.46',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-00.73|+02.02|-02.46',
        forceAction=True
        )
        
        return event
        