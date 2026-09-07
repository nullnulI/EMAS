
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
        objectId='Cabinet|+02.35|+00.40|-00.30',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+00.07|+02.26|+01.31',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+01.72|+00.40|+01.00',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+02.35|+02.01|-00.61',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+01.68|+02.01|-00.61',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+01.67|+02.01|+01.31',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+02.37|+02.01|+01.31',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+02.33|+00.40|+01.00',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+00.02|+02.01|+01.31',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+01.03|+02.01|+01.31',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-00.27|+00.40|+01.00',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+02.33|+02.01|+01.31',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+01.67|+00.40|-00.30',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+00.33|+02.01|-00.61',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+00.36|+02.01|-00.61',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+01.68|+00.40|+01.00',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|-00.28|+02.01|-00.61',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+01.73|+02.01|+01.31',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+01.72|+00.40|-00.30',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+00.98|+02.01|-00.61',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+00.97|+02.26|+01.31',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+01.02|+00.40|-00.30',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+01.03|+00.40|+01.00',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+02.37|+00.40|+01.00',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+01.02|+02.01|-00.61',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+02.99|+02.01|+01.31',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+01.72|+02.01|-00.61',
        forceAction=True
        )
        
        event=controller.step(
        action='OpenObject',
        objectId='Cabinet|+02.98|+00.40|+01.00',
        forceAction=True
        )
        
        return event
        