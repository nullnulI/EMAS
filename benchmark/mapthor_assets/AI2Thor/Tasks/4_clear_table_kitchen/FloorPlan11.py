
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
        objectId='Spoon|+00.38|+00.77|-01.57',
        position={'x': -2.227555751800537, 'y': 0.9425931572914124, 'z': 0.3501785099506378}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Fork|+00.40|+00.77|-01.72',
        position={'x': -2.592850923538208, 'y': 0.9435455203056335, 'z': 0.20568370819091797}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Pot|-02.38|+00.94|+00.58',
        position={'x': 1.7767935991287231, 'y': 0.9012487530708313, 'z': -1.416894793510437}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Mug|-01.90|+00.94|+00.38',
        position={'x': 0.4367360770702362, 'y': 0.7667778730392456, 'z': -1.5418280363082886}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Bowl|-02.48|+00.95|+00.23',
        position={'x': 0.7740212082862854, 'y': 0.9134719371795654, 'z': -1.4468461275100708}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Lettuce|+00.85|+00.99|-01.55',
        position={'x': -1.910172700881958, 'y': 1.0227789878845215, 'z': 0.6657306551933289}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Tomato|+00.72|+00.96|-01.48',
        position={'x': -1.880627155303955, 'y': 0.9948645830154419, 'z': 0.2922794222831726}
        )
        
        return event
        