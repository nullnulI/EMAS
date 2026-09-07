
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
        objectId='ButterKnife|-01.33|+00.92|-00.88',
        position={'x': -0.5883023738861084, 'y': 0.9249398708343506, 'z': -1.4846999645233154}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Fork|-01.29|+00.92|-01.29',
        position={'x': -0.9956160187721252, 'y': 0.9207424521446228, 'z': -1.5871869325637817}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='CreditCard|-01.40|+00.92|-00.22',
        position={'x': -1.3961955308914185, 'y': 0.00035418313927948475, 'z': 2.4804463386535645}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Pot|-01.22|+00.92|-00.49',
        position={'x': 0.40150195360183716, 'y': 0.000287763774394989, 'z': 2.8099987506866455}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Bread|-00.78|+01.00|+00.21',
        position={'x': 0.6960005164146423, 'y': 0.9771426916122437, 'z': 0.06000032275915146}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Statue|-01.33|+00.92|+00.22',
        position={'x': -1.3319753408432007, 'y': 1.1699843406677246, 'z': 2.7723612785339355}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Apple|-01.09|+00.96|-00.02',
        position={'x': -1.3900885581970215, 'y': 0.963293194770813, 'z': 0.2798955738544464}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='ButterKnife|-01.33|+00.92|-00.88',
        position={'x': 0.6755719780921936, 'y': 0.9002934098243713, 'z': -0.6094114780426025}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Tomato|-00.87|+00.96|-00.10',
        position={'x': -0.5750590562820435, 'y': 0.962647020816803, 'z': 0.20400404930114746}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Potato|-00.69|+00.95|-00.05',
        position={'x': 0.780447781085968, 'y': 0.9324489235877991, 'z': -0.503066897392273}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Plate|-00.56|+00.92|-01.52',
        position={'x': 0.9162791967391968, 'y': 1.333861231803894, 'z': -1.9661157131195068}
        )
        
        return event
        
