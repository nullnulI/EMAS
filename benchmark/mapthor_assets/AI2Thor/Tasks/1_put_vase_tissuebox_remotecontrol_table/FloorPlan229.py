"""
Pre-initalization for FloorPlan229 task (living room).
FloorPlan229 does not need any modifications for the task 
of transporting a watering can, tissue box, and remote control to the coffee table.
Initial locations:
- watering can (floor)
- tissue box (desk)
- remote control (sofa)
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
        objectId='WateringCan|-05.24|+00.02|+00.14',
        position={'x': -5.238763809204102, 'y': 0.02019282430410385, 'z': 0.13643841445446014}
        )

        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='TissueBox|-00.31|+00.77|+01.07',
        position={'x': -0.31250590085983276, 'y': 0.7699369192123413, 'z': 1.0714961290359497}
        )

        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='RemoteControl|-02.52|+00.43|+00.58',
        position={'x': -2.519028663635254, 'y': 0.4268532693386078, 'z': 0.5830515623092651}
        )

        
                    
        return event
            