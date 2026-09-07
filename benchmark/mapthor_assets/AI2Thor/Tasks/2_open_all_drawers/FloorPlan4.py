
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
        objectId='Drawer|-02.50|+00.61|+00.59',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|-02.50|+00.22|+00.59',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|-02.04|+00.61|+00.59',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|-02.04|+00.22|+00.59',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|-02.04|+00.94|+00.60',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|-02.51|+00.94|+00.60',
        forceAction=True
        )
        
        return event
        