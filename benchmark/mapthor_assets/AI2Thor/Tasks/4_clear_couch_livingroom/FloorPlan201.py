
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
        objectId='KeyChain|-00.27|+00.70|+03.13',
        position={'x': -2.5889039039611816, 'y': 0.5507990717887878, 'z': 3.5884695053100586}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Pen|-03.00|+00.69|+01.46',
        position={'x': -2.1066579818725586, 'y': 0.5554601550102234, 'z': 3.5402324199676514}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Book|-01.87|+00.68|+01.17',
        position={'x': -1.6935069561004639, 'y': 0.5490167737007141, 'z': 3.4750213623046875}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Pencil|-02.83|+00.68|+01.51',
        position={'x': -2.2922263145446777, 'y': 0.5545456409454346, 'z': 3.5876312255859375}
        )
        
        return event
        