"""
Pre-initalization for FloorPlan219 task (living room).
FloorPlan219 requires a modification of the vase and remote control locations for the task 
of transporting a vase, tissue box, and remote control to the side table.
Initial locations:
- vase (shelf, initially was at a less visible spot)
- tissue box (dresser)
- remote control (sofa, initially was side table)
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
        objectId='Vase|-00.18|+00.56|+00.88',
        position={'x': -0.38392430543899536, 'y': 0.49358272552490234, 'z': 1.2484190464019775}
        )

        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='TissueBox|-04.67|+01.11|+04.72',
        position={'x': -4.668089866638184, 'y': 1.108041763305664, 'z': 4.721411228179932}
        )

        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='RemoteControl|-04.87|+01.05|+00.30',
        position={'x': -0.949939489364624, 'y': 1.0701113939285278, 'z': 2.2469985485076904}
        )
                    
        return event
            