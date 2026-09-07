
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
        objectId='Cabinet|-01.67|+02.01|-01.54',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-00.57|+00.37|-01.24',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+00.38|+00.37|-01.24',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+00.78|+00.37|-01.24',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+01.78|+00.37|-01.24',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-01.48|+00.37|-01.24',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+01.78|+02.01|-01.54',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+00.52|+02.01|-01.54',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-02.22|+00.37|-01.87',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+01.45|+02.26|-01.54',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-02.17|+02.01|-01.54',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+00.85|+02.26|-01.54',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-00.71|+02.01|-01.54',
        forceAction=True
        )
        
        return event
        