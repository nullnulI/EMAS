
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
        objectId='Drawer|+02.51|+00.79|+01.00',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|+01.86|+00.79|+01.00',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|-00.13|+00.79|+01.00',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|+02.19|+00.79|+01.00',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|+01.53|+00.79|+01.00',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|+01.17|+00.79|-00.30',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|+01.52|+00.79|-00.30',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|+02.84|+00.79|+01.00',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|+01.87|+00.79|-00.30',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|+02.21|+00.79|-00.30',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|+01.18|+00.79|+01.00',
        forceAction=True
        )
        
        return event
        