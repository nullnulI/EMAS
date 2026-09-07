
"""Move butterknife, spoon, fork, tomato, lettuce, fridge for preinit"""
    
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
        objectId='Lettuce|+01.25|+00.96|-00.15',
        position={'x': 1.0722064971923828, 'y': 1.0216376781463623, 'z': 2.3423094749450684}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Spoon|+01.33|+00.97|+02.76',
        position={'x': 1.4174798727035522, 'y': 0.9696021676063538, 'z': 2.3674557209014893}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Tomato|-00.51|+00.95|-00.49',
        position={'x': 0.7463608980178833, 'y': 0.9973886013031006, 'z': 2.7569162845611572}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Fork|+01.16|+00.96|+02.52',
        position={'x': 0.7988155484199524, 'y': 0.9509323239326477, 'z': 2.515821695327759}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Bread|+00.97|+01.02|+02.70',
        position={'x': -1.1918928623199463, 'y': 0.8193426132202148, 'z': 2.6993236541748047}
        )
        
        return event
        