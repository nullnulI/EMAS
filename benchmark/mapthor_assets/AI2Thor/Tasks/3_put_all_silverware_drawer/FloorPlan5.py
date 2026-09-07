
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
        objectId='Knife|+00.06|+00.91|+00.18',
        position={'x': 1.1185201406478882, 'y': 0.907116174697876, 'z': -1.9585427045822144}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Fork|-00.75|+00.90|+00.12',
        position={'x': 1.3695459365844727, 'y': 0.9017117023468018, 'z': -1.7839741706848145}
        )
        
        return event
        