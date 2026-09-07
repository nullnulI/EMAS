
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
        objectId='Fork|-00.46|+01.04|+01.78',
        position={'x': -0.9567781686782837, 'y': 1.0398825407028198, 'z': 2.709099054336548}
        )
        
        return event
        