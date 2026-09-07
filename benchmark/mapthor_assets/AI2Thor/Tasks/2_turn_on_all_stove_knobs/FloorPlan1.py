
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
        objectId='StoveKnob|-00.48|+00.88|-02.19',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='StoveKnob|-00.02|+00.88|-02.19',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='StoveKnob|-00.33|+00.88|-02.19',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='StoveKnob|-00.18|+00.88|-02.19',
        forceAction=True
        )
        
        return event
        