
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
        objectId='Drawer|-00.69|+00.75|-01.39',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|-02.11|+00.75|-01.39',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|+00.59|+00.75|-01.39',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|-00.38|+00.75|-01.39',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|+00.86|+00.75|-00.70',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|+00.87|+00.75|-01.14',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|-01.80|+00.75|-01.39',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|+00.86|+00.75|+00.43',
        forceAction=True
        )
        
        return event
        