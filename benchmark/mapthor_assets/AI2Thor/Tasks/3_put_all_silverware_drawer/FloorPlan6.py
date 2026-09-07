
"""Move butterknife, knife, spatula, spoon, and fork around for initialization"""
    
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
        action='PlaceObjectAtPoint',
        objectId='Fork|+01.44|+00.90|+00.38',
        position={'x': -0.17513976991176605, 'y': 0.9016438722610474, 'z': 1.4646979570388794}
        )
        
        return event
        