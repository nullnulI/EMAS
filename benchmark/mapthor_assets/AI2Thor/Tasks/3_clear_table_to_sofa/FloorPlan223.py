
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
        objectId='Laptop|-02.02|+00.45|-00.12',
        position={'x': -2.2291338443756104, 'y': 0.44744396209716797, 'z': -0.4008593261241913}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='CreditCard|-01.83|+00.44|+00.28',
        position={'x': -1.8339998722076416, 'y': 0.42745640873908997, 'z': -1.4316697120666504}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Statue|-03.59|+00.60|+01.43',
        position={'x': -2.095688581466675, 'y': 0.4431034028530121, 'z': 0.14861541986465454}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Statue|-02.04|+00.24|+02.75',
        position={'x': -1.1891067028045654, 'y': 0.005894094705581665, 'z': 2.748652219772339}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='KeyChain|-02.22|+00.44|-00.34',
        position={'x': -3.292119264602661, 'y': 0.0009808018803596497, 'z': 2.649573802947998}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Plate|+01.36|+00.78|-00.62',
        position={'x': -1.846252679824829, 'y': 0.0005869250744581223, 'z': -0.33921459317207336}
        )
        
        return event
        