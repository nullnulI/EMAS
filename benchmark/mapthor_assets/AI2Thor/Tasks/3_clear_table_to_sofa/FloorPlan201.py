
"""Move bowl, laptop, book, statue, box, remote around at initialization"""
    
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
        objectId='Laptop|-01.70|+00.68|+01.66',
        position={'x': -2.0556342601776123, 'y': 0.5956512093544006, 'z': 5.122322082519531}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='CreditCard|-01.94|+00.68|+01.80',
        position={'x': -1.9360617399215698, 'y': 0.733784019947052, 'z': -0.04747398942708969}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Plate|-02.40|+00.68|+01.68',
        position={'x': -2.401923179626465, 'y': 0.6809349656105042, 'z': 1.4451384544372559}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Book|-01.87|+00.68|+01.17',
        position={'x': -2.0505142211914062, 'y': 0.5949640870094299, 'z': 4.629279136657715}
        )
        
        return event
        