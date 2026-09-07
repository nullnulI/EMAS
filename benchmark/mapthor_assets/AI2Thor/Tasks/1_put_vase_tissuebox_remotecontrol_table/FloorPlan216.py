"""
Pre-initalization for FloorPlan216 task (living room).
FloorPlan216 does not need any modifications for the task 
of transporting a vase, tissue box, and remote control to the dining table.
Initial locations:
- vase [tall] (under the shelf)
- tissue box (side table)
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
        objectId='Vase|-00.03|+00.11|+01.19',
        position={'x': -0.025592807680368423, 'y': 0.1110864132642746, 'z': 1.1932799816131592}
        )

        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='TissueBox|+00.92|+00.65|+00.96',
        position={'x': 0.9229203462600708, 'y': 0.6517916917800903, 'z': 0.9576342701911926}
        )

        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='RemoteControl|-00.64|+00.53|-01.24',
        position={'x': -0.6360008716583252, 'y': 0.5279459357261658, 'z': -1.2369985580444336}
        )
                    
        return event
            