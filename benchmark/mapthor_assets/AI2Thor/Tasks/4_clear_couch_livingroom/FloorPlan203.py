
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
        objectId='Watch|+00.24|+00.55|+02.52',
        position={'x': -1.0674285888671875, 'y': 0.4441039562225342, 'z': 2.764545440673828}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Pencil|-04.49|+00.76|-00.19',
        position={'x': -0.9854170083999634, 'y': 0.44856181740760803, 'z': 2.0293655395507812}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Book|-03.72|+00.76|-00.64',
        position={'x': -1.3095077276229858, 'y': 0.4438406527042389, 'z': 3.2924513816833496}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='RemoteControl|-01.27|+00.45|+03.29',
        position={'x': -1.2716068029403687, 'y': 0.007753010839223862, 'z': 6.985918045043945}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='KeyChain|+00.11|+00.45|+04.86',
        position={'x': -0.7632629871368408, 'y': 0.4448205232620239, 'z': 2.3945634365081787}
        )
        
        return event
        