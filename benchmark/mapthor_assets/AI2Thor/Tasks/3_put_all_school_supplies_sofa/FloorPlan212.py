
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
        objectId='Pencil|+03.89|+00.87|+01.18',
        position={'x': 2.53610897064209, 'y': 0.32378891110420227, 'z': 1.644966721534729}
        )
        
        return event
        