
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
        objectId='Statue|-04.23|+01.27|+01.02',
        position={'x': -4.051924705505371, 'y': 0.007187321782112122, 'z': 1.3810529708862305}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='CellPhone|-02.22|+00.44|+03.81',
        position={'x': -0.5700804591178894, 'y': 0.42341890931129456, 'z': 3.443169593811035}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Laptop|-00.60|+00.74|+01.31',
        position={'x': -1.5101807117462158, 'y': 0.0013239234685897827, 'z': 4.6004533767700195}
        )
        
        return event
        