
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
        objectId='StoveKnob|+01.60|+00.92|+00.46',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='StoveKnob|+01.60|+00.92|+00.68',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='StoveKnob|+01.60|+00.92|+00.74',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='StoveKnob|+01.60|+00.92|+00.63',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='StoveKnob|+01.60|+00.92|+00.57',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='StoveKnob|+01.60|+00.92|+00.52',
        forceAction=True
        )
        
        return event
        