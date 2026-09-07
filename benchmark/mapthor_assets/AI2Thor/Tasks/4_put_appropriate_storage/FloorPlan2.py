
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
        objectId='ButterKnife|+01.67|+00.69|-00.11',
        position={'x': 0.21802067756652832, 'y': 0.9203400611877441, 'z': 0.18601493537425995}
        )
        
        return event
        