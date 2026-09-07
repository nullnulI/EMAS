
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
        objectId='Bowl|-00.50|+01.04|+02.22',
        position={'x': -2.1063849925994873, 'y': 1.1733312606811523, 'z': 0.7011251449584961}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Pan|-00.27|+01.04|+02.56',
        position={'x': -1.7537232637405396, 'y': 1.1448051929473877, 'z': 0.4624919891357422}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Lettuce|-03.31|+00.97|+03.04',
        position={'x': -0.8295101523399353, 'y': 1.1071478128433228, 'z': 3.0430819988250732}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Bread|-00.18|+01.12|+01.89',
        position={'x': -0.1839999258518219, 'y': 0.08092722296714783, 'z': 1.6588339805603027}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Fork|-00.46|+01.04|+01.78',
        position={'x': -0.9567239284515381, 'y': 1.0398824214935303, 'z': 1.89305579662323}
        )
        
        return event
        