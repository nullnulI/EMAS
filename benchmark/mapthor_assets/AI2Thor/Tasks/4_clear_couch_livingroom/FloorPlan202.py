
"""Move keychain, pencil, pen, book, watch, drawer for preinitialization"""
    
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
        objectId='KeyChain|-01.01|+00.70|+03.53',
        position={'x': -0.7243834137916565, 'y': 0.5341503024101257, 'z': 0.8830432295799255}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Book|-01.94|+00.70|+03.61',
        position={'x': -1.9425570964813232, 'y': 0.5331102609634399, 'z': 0.6194378733634949}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Watch|-01.22|+00.70|+03.47',
        position={'x': -1.3601510524749756, 'y': 0.5334572196006775, 'z': 0.7081215977668762}
        )
        
        return event
        