
"""Close all drawers in the pre-initialization."""
    
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
        action='ToggleObjectOff',
        objectId='Drawer|-00.82|+00.75|-01.69',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|-00.45|+00.75|-00.01',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|-00.07|+00.75|-00.01',
        forceAction=True
        )
        
        return event
        