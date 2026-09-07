
"""Move butterknife, spoon, fork, tomato, lettuce, fridge for preinit"""
    
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
        objectId='CellPhone|+01.61|+00.90|+00.83',
        position={'x': 3.3284084796905518, 'y': 0.026973476633429527, 'z': 1.1551697254180908}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Spatula|+01.38|+00.91|+00.17',
        position={'x': 3.095881938934326, 'y': 0.8965924382209778, 'z': -0.26656848192214966}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Knife|+01.37|+00.91|+00.31',
        position={'x': -0.04700018838047981, 'y': 0.880096435546875, 'z': -1.3546944856643677}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Bread|+01.53|+00.97|+00.49',
        position={'x': 2.0623512268066406, 'y': 0.09959318488836288, 'z': -1.5704861879348755}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Pot|+00.42|+00.90|+00.87',
        position={'x': 2.5358388423919678, 'y': 0.026462316513061523, 'z': -1.4045460224151611}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='ButterKnife|+01.59|+00.90|+00.03',
        position={'x': 1.7199410200119019, 'y': 0.9014365077018738, 'z': 0.027499116957187653}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Ladle|+00.85|+00.92|+00.81',
        position={'x': -0.9999111890792847, 'y': 0.9030390977859497, 'z': 0.8064401149749756}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Bowl|+01.14|+00.90|+00.87',
        position={'x': -0.9774191379547119, 'y': 1.271459698677063, 'z': 1.1909656524658203}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Tomato|+00.95|+00.97|-01.52',
        position={'x': 0.9446504712104797, 'y': 0.9427269101142883, 'z': 0.8576599359512329}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Mug|+01.10|+00.90|+00.11',
        position={'x': -1.1440354585647583, 'y': 0.8731666803359985, 'z': 0.00011984159937128425}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Fork|+03.12|+00.74|+00.18',
        position={'x': 0.4821437895298004, 'y': 0.9020131826400757, 'z': 0.18152427673339844}
        )
        
        return event
        