
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
        objectId='Cabinet|-00.87|+02.01|-01.69',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-00.20|+00.40|-01.39',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+00.83|+00.40|-01.39',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-01.61|+02.01|-01.69',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+01.16|+02.06|-01.02',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-01.67|+00.40|-01.39',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-02.24|+00.40|-01.39',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-00.19|+02.06|-01.69',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-00.82|+00.40|-01.39',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-00.24|+00.40|-01.39',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+00.86|+00.40|-01.37',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+00.86|+00.40|-00.55',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+00.86|+00.40|+00.67',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-00.82|+02.06|-01.69',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-01.66|+02.06|-01.68',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+00.49|+02.06|-01.69',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+01.16|+02.06|-00.34',
        forceAction=True
        )
        
        return event
        