
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
        objectId='Tomato|+00.92|+01.91|+01.61',
        position={'x': -1.4379316568374634, 'y': 1.348083257675171, 'z': 1.954134464263916}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Egg|-01.57|+01.36|-01.05',
        position={'x': 1.032253623008728, 'y': 1.7255436182022095, 'z': 2.061701774597168}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Potato|-01.61|+01.36|-01.24',
        position={'x': 0.5166969895362854, 'y': 1.3515303134918213, 'z': -1.2407563924789429}
        )
        
        return event
        