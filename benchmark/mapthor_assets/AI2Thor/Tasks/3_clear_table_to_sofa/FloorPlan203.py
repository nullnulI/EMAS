
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
        objectId='RemoteControl|-01.27|+00.45|+03.29',
        position={'x': -1.2716063261032104, 'y': 0.0077530695125460625, 'z': 6.739828109741211}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Book|-03.72|+00.76|-00.64',
        position={'x': 0.44289106130599976, 'y': 0.5504924654960632, 'z': 2.3081116676330566}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Plate|-04.04|+00.76|-00.08',
        position={'x': 0.11712534725666046, 'y': 0.5515668988227844, 'z': 2.8745787143707275}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Watch|+00.24|+00.55|+02.52',
        position={'x': 0.2394097000360489, 'y': 0.006279114633798599, 'z': 6.946480751037598}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Laptop|+00.38|+00.55|+02.83',
        position={'x': 0.5981862545013428, 'y': 0.550445556640625, 'z': 3.08017897605896}
        )
        
        return event
        