
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
        objectId='Statue|-03.03|+00.78|-03.75',
        position={'x': -0.23501159250736237, 'y': 0.4170750081539154, 'z': 0.9230156540870667}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Laptop|-02.91|+00.78|-03.09',
        position={'x': -0.9151050448417664, 'y': 0.4182480573654175, 'z': 1.1601171493530273}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='RemoteControl|-00.33|+00.42|+01.08',
        position={'x': -0.3316395580768585, 'y': 0.4189847707748413, 'z': 1.296773076057434}
        )
        
        return event
        