
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
        objectId='Tomato|-02.30|+00.96|+03.80',
        position={'x': -0.9313076138496399, 'y': 0.9331694841384888, 'z': 0.4850468337535858}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Lettuce|-01.67|+00.96|+00.10',
        position={'x': -1.787859320640564, 'y': 0.9561364650726318, 'z': -0.03037651628255844}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Fork|-03.12|+00.74|+02.93',
        position={'x': -0.9404678344726562, 'y': 0.8765342235565186, 'z': 0.30756425857543945}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='PepperShaker|-01.72|+00.87|+00.40',
        position={'x': -3.22237491607666, 'y': 0.9724594354629517, 'z': 2.3348512649536133}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='ButterKnife|-01.84|+00.78|+03.68',
        position={'x': -2.111360788345337, 'y': 0.9039754271507263, 'z': 3.678042411804199}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='SaltShaker|-01.66|+00.87|+00.53',
        position={'x': -3.301236391067505, 'y': 0.9030385613441467, 'z': 2.7362852096557617}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Bread|-01.52|+00.96|+00.28',
        position={'x': -3.0234293937683105, 'y': 0.9882468581199646, 'z': 1.2429991960525513}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Spoon|-03.12|+00.74|+02.83',
        position={'x': -1.8885072469711304, 'y': 0.8751829862594604, 'z': 0.4792134761810303}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Potato|-01.80|+00.91|+00.52',
        position={'x': -3.3053972721099854, 'y': 0.9736559391021729, 'z': 1.7609015703201294}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Knife|-01.36|+00.88|+00.50',
        position={'x': -1.9035930633544922, 'y': 0.7866649031639099, 'z': 3.6743361949920654}
        )
        
        return event
        