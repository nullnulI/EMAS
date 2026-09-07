
"""Move butterknife, spoon, fork, tomato, lettuce for preinit"""
    
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
        objectId='Mug|-00.24|+00.92|-00.26',
        position={'x': -0.09001296758651733, 'y': 0.10500861704349518, 'z': -1.3060003519058228}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Potato|+00.33|+00.94|+01.31',
        position={'x': 1.632442593574524, 'y': 0.9250426888465881, 'z': 1.4615103006362915}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Bread|+00.26|+00.99|-00.08',
        position={'x': 1.561137080192566, 'y': 0.9724830985069275, 'z': -0.8260024189949036}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='ButterKnife|+01.67|+00.69|-00.11',
        position={'x': -0.2641521990299225, 'y': 0.9152843356132507, 'z': -0.09703084081411362}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Apple|-01.65|+00.81|+00.07',
        position={'x': -1.6506528854370117, 'y': 1.3972967863082886, 'z': -0.07749859988689423}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Fork|+01.69|+00.90|+01.45',
        position={'x': -0.33628538250923157, 'y': 0.9166437983512878, 'z': 0.6980903744697571}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Bowl|+00.28|+00.92|+01.09',
        position={'x': 1.581531286239624, 'y': 0.9011380076408386, 'z': 1.2411946058273315}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='CellPhone|-00.31|+00.92|+01.29',
        position={'x': 2.011401891708374, 'y': 0.9007588624954224, 'z': 1.2878098487854004}
        )
        
        return event
        
