
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
        objectId='Cabinet|-02.15|+00.40|-00.24',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-02.45|+02.15|-01.28',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-02.45|+01.95|+01.64',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-02.45|+01.95|+01.69',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-02.45|+02.15|-00.29',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-02.45|+01.95|+00.36',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-02.45|+01.95|+00.41',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+00.15|+02.01|-01.60',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-02.15|+00.40|+01.58',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-02.15|+00.40|+00.64',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-02.45|+01.95|+02.93',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-02.15|+00.40|+00.70',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-02.29|+01.97|-01.33',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+01.57|+02.01|-00.78',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+01.57|+02.01|+00.47',
        forceAction=True
        )
        
        return event
        