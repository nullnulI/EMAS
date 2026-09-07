
"""Move apple, egg, lettuce, tomato, spoon, fork, butterknife for pre-init"""
    
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
        objectId='Bowl|+00.27|+01.10|-00.75',
        position={'x': -0.2242959588766098, 'y': 0.9318560361862183, 'z': -2.5391762256622314}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Book|+00.15|+01.10|+00.62',
        position={'x': 2.0203912258148193, 'y': -0.0007973164319992065, 'z': 1.7159671783447266}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='CreditCard|-00.46|+01.10|+00.87',
        position={'x': -0.46366456151008606, 'y': 0.00035418616607785225, 'z': 2.23897647857666}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Fork|+00.95|+00.77|-02.37',
        position={'x': 0.2030719518661499, 'y': 1.1016438007354736, 'z': -0.7147209644317627}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Bread|-00.52|+01.17|-00.03',
        position={'x': -1.642450213432312, 'y': 0.9700067043304443, 'z': -1.402009129524231}
        )
        
        return event
        