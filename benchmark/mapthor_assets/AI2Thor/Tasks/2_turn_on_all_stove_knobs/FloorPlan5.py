
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
        objectId='StoveKnob|-00.87|+00.90|-00.83',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='StoveKnob|-00.87|+00.90|-00.68',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='StoveKnob|-00.87|+00.90|-00.97',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='StoveKnob|-00.87|+00.90|-01.12',
        forceAction=True
        )
        
        return event
        