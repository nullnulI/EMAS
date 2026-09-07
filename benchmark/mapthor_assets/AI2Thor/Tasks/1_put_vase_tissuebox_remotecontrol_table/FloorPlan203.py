"""
Pre-initalization for FloorPlan203 task (living room).
FloorPlan203 requires a modification of the vase location for the task 
of transporting a vase, tissue box, and remote control to the dining table.
Initial locations:
- vase (drawer table, initially was dining table)
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
        objectId='Vase|-04.27|+00.76|-00.44',
        position={'x': 0.33705905079841614, 'y': 0.6720947623252869, 'z': 2.2441368103027344}
        )

        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='TissueBox|-00.99|+00.88|+00.24',
        position={'x': -0.9940035343170166, 'y': 0.8751899600028992, 'z': 0.23653732240200043}
        )

        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='RemoteControl|-01.27|+00.45|+03.29',
        position={'x': -1.2715975046157837, 'y': 0.44544124603271484, 'z': 3.294555187225342}
        )
                    
        return event
            