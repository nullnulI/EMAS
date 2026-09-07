
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
        objectId='Pillow|+00.65|+00.39|+01.71',
        position={'x': -1.1589105129241943, 'y': 0.06653190404176712, 'z': 1.360148310661316}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='RemoteControl|+01.88|+00.33|+01.73',
        position={'x': 3.6926870346069336, 'y': 0.0007512643933296204, 'z': 1.7323088645935059}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Pencil|+03.89|+00.87|+01.18',
        position={'x': 1.178609013557434, 'y': 0.33182233572006226, 'z': 1.6449646949768066}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='KeyChain|+01.50|+00.47|+00.53',
        position={'x': 1.9540479183197021, 'y': 0.32807642221450806, 'z': 1.6930464506149292}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Pen|+03.93|+00.87|+01.04',
        position={'x': 1.668033242225647, 'y': 0.3327394723892212, 'z': 1.6211962699890137}
        )
        
        return event
        