
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
        objectId='Drawer|+00.81|+00.48|-01.16',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|+01.50|+00.20|-00.02',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|+01.50|+00.63|-00.02',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|+01.50|+00.14|+00.60',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|+01.50|+00.60|-00.02',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|+01.50|+00.20|+01.22',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|+01.50|+00.78|+01.22',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|+01.50|+00.31|+00.60',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|+01.50|+00.54|+00.60',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|+01.50|+00.63|+00.61',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|+01.50|+00.43|-00.02',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|-00.70|+00.48|-01.16',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='Drawer|+01.50|+00.52|+01.22',
        forceAction=True
        )
        
        return event
        