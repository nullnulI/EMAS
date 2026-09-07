
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
        objectId='Pen|-02.83|+00.70|-04.96',
        position={'x': -2.8290140628814697, 'y': 0.7048178315162659, 'z': -4.963045597076416}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Book|-02.53|+00.70|-05.12',
        position={'x': -3.366100549697876, 'y': 0.6998944878578186, 'z': -5.1249680519104}
        )
        
        return event
        