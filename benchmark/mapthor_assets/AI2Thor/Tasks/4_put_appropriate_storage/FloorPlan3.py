
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
        objectId='Spoon|+00.61|+01.32|-02.36',
        position={'x': -1.5106533765792847, 'y': 1.3106834888458252, 'z': -2.0139286518096924}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Knife|+00.72|+01.32|-02.20',
        position={'x': 0.715499758720398, 'y': 1.3173491954803467, 'z': -1.1554244756698608}
        )
        
        return event
        