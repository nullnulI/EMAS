
"""Turn lights on at initialization"""
    
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
        action='ToggleObjectOn',
        objectId='LightSwitch|-00.13|+01.41|+04.35',
        forceAction=True
        )
        
        return event
        