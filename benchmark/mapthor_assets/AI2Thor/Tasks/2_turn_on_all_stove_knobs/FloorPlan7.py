
"""Toggle all stoveknobs off in the pre-initialization."""
    
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
        objectId='StoveKnob|+01.34|+00.91|-01.24',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='StoveKnob|+00.96|+00.91|-01.24',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='StoveKnob|+01.21|+00.91|-01.24',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='StoveKnob|+01.09|+00.91|-01.24',
        forceAction=True
        )
        
        return event
        