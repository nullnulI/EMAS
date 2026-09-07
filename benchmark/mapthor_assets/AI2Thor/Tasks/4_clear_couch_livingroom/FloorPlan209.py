
"""Move keychain, pencil, pen, book, watch, drawer for preinitialization"""
    
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
        objectId='KeyChain|-04.21|+00.32|-03.65',
        position={'x': -4.004233360290527, 'y': 0.3297809660434723, 'z': -2.7201120853424072}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='RemoteControl|-04.29|+00.33|-02.66',
        position={'x': -3.4522864818573, 'y': 0.7007746696472168, 'z': -5.15516996383667}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Newspaper|-04.24|+00.34|-02.16',
        position={'x': -6.341192245483398, 'y': 0.0065051838755607605, 'z': -0.6072340607643127}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Book|-02.53|+00.70|-05.12',
        position={'x': -3.9965062141418457, 'y': 0.32934918999671936, 'z': -2.3224453926086426}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Watch|-02.79|+00.58|-02.07',
        position={'x': -4.050622463226318, 'y': 0.33011940121650696, 'z': -1.9149408340454102}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Pen|-02.83|+00.70|-04.96',
        position={'x': -4.090189456939697, 'y': 0.3344421088695526, 'z': -1.5378044843673706}
        )
        
        return event
        