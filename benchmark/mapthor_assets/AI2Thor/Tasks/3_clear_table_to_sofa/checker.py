"""
To check completion of subtasks in 2_put_all_tomatoes_potatoes_fridge task.
Also checks coverage of the task and success of the task.
Subtasks:
    • NavigateTo(Bowl)
    • PickUpObject(Bowl)
    • NavigateTo(Sofa) [with Bowl in inventory]
    • PutObject(Sofa) [with Bowl in inventory]
    • NavigateTo(Laptop)
    • PickUpObject(Laptop)
    • NavigateTo(Sofa) [with Laptop in inventory]
    • PutObject(Sofa) [with Laptop in inventory]
    • NavigateTo(Book)
    • PickUpObject(Book)
    • NavigateTo(Sofa) [with Book in inventory]
    • PutObject(Sofa) [with Book in inventory]
    • NavigateTo(Statue)
    • PickUpObject(Statue)
    • NavigateTo(Sofa) [with Statue in inventory]
    • PutObject(Sofa) [with Statue in inventory]
    • NavigateTo(Box)
    • PickUpObject(Box)
    • NavigateTo(Sofa) [with Box in inventory]
    • PutObject(Sofa) [with Box in inventory]
    • NavigateTo(RemoteControl)
    • PickUpObject(RemoteControl)
    • NavigateTo(Sofa) [with RemoteControl in inventory]
    • PutObject(Sofa) [with RemoteControl in inventory]

Coverage:
    • Bowl
    • Laptop 
    • Statue
    • Box
    • RemoteControl
    • Sofa
    • DiningTable
"""


from AI2Thor.baselines.utils.checker import BaseChecker

class Checker(BaseChecker):
    def __init__(self) -> None:
        subtasks = [
            'NavigateTo(DiningTable_1)',
            'NavigateTo(Bowl)',
             'PickUpObject(Bowl)',
             'NavigateTo(Sofa, Bowl)',
             'PutObject(Sofa, Bowl)',
            'NavigateTo(Plate)',
             'PickUpObject(Plate)',
             'NavigateTo(Sofa, Plate)',
             'PutObject(Sofa, Plate)',
             'NavigateTo(Laptop)',
             'PickUpObject(Laptop)',
             'NavigateTo(Sofa, Laptop)',
             'PutObject(Sofa, Laptop)',
             'NavigateTo(Book)',
             'PickUpObject(Book)',
             'NavigateTo(Sofa, Book)',
             'PutObject(Sofa, Book)',
             'NavigateTo(Statue)',
             'PickUpObject(Statue)',
             'NavigateTo(Sofa, Statue)',
             'PutObject(Sofa, Statue)',
             'NavigateTo(Box)',
             'PickUpObject(Box)',
             'NavigateTo(Sofa, Box)',
             'PutObject(Sofa, Box)',
             'NavigateTo(RemoteControl)',
             'PickUpObject(RemoteControl)',
             'NavigateTo(Sofa, RemoteControl)',
             'PutObject(Sofa, RemoteControl)'
            ]

        conditional_subtasks = [
            'NavigateTo(Sofa, Bowl)',
             'PutObject(Sofa, Bowl)',
             'NavigateTo(Sofa, Plate)',
             'PutObject(Sofa, Plate)',
             'NavigateTo(Sofa, Laptop)',
             'PutObject(Sofa, Laptop)',
             'NavigateTo(Sofa, Book)',
             'PutObject(Sofa, Book)',
             'NavigateTo(Sofa, Statue)',
             'PutObject(Sofa, Statue)',
             'NavigateTo(Sofa, Box)',
             'PutObject(Sofa, Box)',
             'NavigateTo(Sofa, RemoteControl)',
             'PutObject(Sofa, RemoteControl)'
            ]

        independent_subtasks = [
        'NavigateTo(DiningTable_1)',
        'NavigateTo(Bowl)',
        'PickUpObject(Bowl)',
        'NavigateTo(Plate)',
        'PickUpObject(Plate)',
        'NavigateTo(Laptop)',
        'PickUpObject(Laptop)',
        'NavigateTo(Book)',
        'PickUpObject(Book)',
        'NavigateTo(Statue)',
        'PickUpObject(Statue)',
        'NavigateTo(Box)',
        'PickUpObject(Box)',
        'NavigateTo(RemoteControl)',
        'PickUpObject(RemoteControl)'
        ]

        # bowl, laptop, book, statue, box, remote, sofa, diningtable
        coverage = ["Bowl", "Plate", "Laptop", "Book", "Statue", "Box", "RemoteControl", "Sofa", "DiningTable"]
        interact_objects = coverage
        interact_receptacles = ["Sofa"]

        # have so that not all objects w/ that name are added if there are multiple, specify number here
        self.manual_override_numerate={"Statue" : 1} # only add the first one
        # have so that we can further restrict to subsets for each floorplan, even if there exists a certain object

        # 201,203,204,208,223
        # Diningtable + 
        # 201 - (bowl, laptop, book)
        # 203 - (book, plate, laptop)
        # 204 - (statue, box, laptop)
        # 208 - (remote, statue, laptop)
        # 223 - (laptop, plate, statue)


        self.manual_override_nms={
                "FloorPlan201" : ["Bowl", "Laptop", "Book"],
                "FloorPlan203" : ["Book", "Plate", "Laptop"],
                "FloorPlan204" : ["Statue", "Box", "Laptop"],
                "FloorPlan208" : ["RemoteControl", "Statue", "Laptop"],
                "FloorPlan223" : ["Plate", "Statue", "Laptop"],
                }
        for k in self.manual_override_nms.keys():
            self.manual_override_nms[k].append("DiningTable")
            self.manual_override_nms[k].append("Sofa")

        super().__init__(
            subtasks,
            conditional_subtasks,
            independent_subtasks,
            coverage,
            interact_objects,
            interact_receptacles,
        )
