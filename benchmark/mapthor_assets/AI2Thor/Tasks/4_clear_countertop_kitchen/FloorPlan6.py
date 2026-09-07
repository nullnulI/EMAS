
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
        objectId='Knife|-00.64|+00.91|+01.62',
        position={'x': -2.2610397338867188, 'y': 0.9067869186401367, 'z': 1.4871816635131836}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Fork|+01.44|+00.90|+00.38',
        position={'x': -0.5791712999343872, 'y': 0.9016438126564026, 'z': 0.9246930480003357}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Tomato|+01.30|+00.96|-01.08',
        position={'x': 0.02001042291522026, 'y': 0.9764814972877502, 'z': 1.544486165046692}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Bread|-00.71|+00.98|+00.43',
        position={'x': 1.482456922531128, 'y': 0.9807197451591492, 'z': -1.0083768367767334}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Bowl|-00.65|+00.90|+01.26',
        position={'x': -2.263826608657837, 'y': 0.9005077481269836, 'z': 0.8549989461898804}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Pan|+00.00|+00.90|+00.95',
        position={'x': 1.7508095502853394, 'y': 1.9402243196964264e-05, 'z': 1.6230014562606812}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Cup|-00.65|+00.90|+00.74',
        position={'x': -2.2621123790740967, 'y': 0.9930315613746643, 'z': 0.468029260635376}
        )
        
        return event
        
