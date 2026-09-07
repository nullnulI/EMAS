
"""Move pen, pencil, computer, book, cellphone around for initialization"""
    
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
        objectId='Pencil|-02.83|+00.68|+01.51',
        position={'x': -2.1178371906280518, 'y': 0.6001876592636108, 'z': 4.741385459899902}
        )
        
        return event
        