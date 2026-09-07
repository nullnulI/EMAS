
"""Move bread, lettuce, tomato, apple, and potato visibly outside in the pre-initialization."""
    
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
        objectId='Apple|-02.48|+01.18|+00.48',
        position={'x': -2.2328367233276367, 'y': 1.183811068534851, 'z': 0.4808482825756073}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Lettuce|-03.31|+00.97|+03.04',
        position={'x': -0.8297501802444458, 'y': 1.1071399450302124, 'z': 3.044142723083496}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Potato|-02.66|+01.15|+00.25',
        position={'x': -3.155620813369751, 'y': 1.1427940130233765, 'z': 0.4863404929637909}
        )
        
        return event
        