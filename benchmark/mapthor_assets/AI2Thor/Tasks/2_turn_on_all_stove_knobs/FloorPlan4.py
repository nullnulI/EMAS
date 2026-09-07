
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
        objectId='StoveKnob|-03.16|+01.05|+00.71',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='StoveKnob|-03.49|+01.05|+00.71',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='StoveKnob|-02.99|+01.05|+00.71',
        forceAction=True
        )
        
        event=controller.step(
        action='ToggleObjectOff',
        objectId='StoveKnob|-03.32|+01.05|+00.71',
        forceAction=True
        )
        
        return event
        