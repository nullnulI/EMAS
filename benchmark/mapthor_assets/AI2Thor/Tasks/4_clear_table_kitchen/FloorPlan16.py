
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
        objectId='Bowl|+02.19|+00.83|+03.51',
        position={'x': 2.4386329650878906, 'y': 1.0158066749572754, 'z': 1.361649513244629}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='ButterKnife|+02.27|+01.02|+00.89',
        position={'x': 2.023603677749634, 'y': 0.8289129137992859, 'z': 3.581190586090088}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Fork|+02.38|+01.02|+00.97',
        position={'x': 1.7658246755599976, 'y': 0.829293966293335, 'z': 4.023651599884033}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Tomato|+01.97|+01.07|+01.37',
        position={'x': 2.2180631160736084, 'y': 0.8817158937454224, 'z': 3.3467812538146973}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Lettuce|+02.88|+01.07|-00.43',
        position={'x': 1.899806261062622, 'y': 0.8816666603088379, 'z': 3.516369581222534}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Apple|+01.95|+00.87|+03.77',
        position={'x': 2.6954543590545654, 'y': 1.0621665716171265, 'z': 0.5448036193847656}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Plate|+02.03|+00.83|+04.18',
        position={'x': 2.771172285079956, 'y': 1.0156333446502686, 'z': 1.4916856288909912}
        )
        
        return event
        