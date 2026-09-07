"""
Pre-initalization for FloorPlan201 task (living room).
FloorPlan201 does not need any modifications for the task 
of transporting a vase, tissue box, and remote control to the dining table.
Initial locations:
- vase (side table)
- tissue box (tv stand)
- remote control (other side table)
"""

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
    
            # initialization function
    

                    
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Vase|-00.26|+00.70|+03.41',
        position={'x': -0.2572877109050751, 'y': 0.7002500295639038, 'z': 3.4148831367492676}
        )

        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='TissueBox|-02.74|+00.72|+06.13',
        position={'x': -2.7439849376678467, 'y': 0.7183737754821777, 'z': 6.131121635437012}
        )

        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='RemoteControl|-02.58|+00.74|-00.15',
        position={'x': -2.5830140113830566, 'y': 0.7350274920463562, 'z': -0.14601318538188934}
        )
                    
        return event
            