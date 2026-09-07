"""
To check completion of subtasks in 2_put_all_tomatoes_potatoes_fridge task.
Also checks coverage of the task and success of the task.
Subtasks:
• NavigateTo(Pen)
• PickUpObject(Pen)
• NavigateTo(Sofa) [with Pen in inventory]
• PutObject(Sofa) [with Pen in inventory]
• NavigateTo(Pencil)
• PickUpObject(Pencil)
• NavigateTo(Sofa) [with Pencil in inventory]
• PutObject(Sofa) [with Pencil in inventory]
• NavigateTo(Laptop)
• PickUpObject(Laptop)
• NavigateTo(Sofa) [with Laptop in inventory]
• PutObject(Sofa) [with Laptop in inventory]
• NavigateTo(Book)
• PickUpObject(Book)
• NavigateTo(Sofa) [with Book in inventory]
• PutObject(Sofa) [with Book in inventory]
• NavigateTo(CellPhone)
• PickUpObject(CellPhone)
• NavigateTo(Sofa) [with CellPhone in inventory]
• PutObject(Sofa) [with CellPhone in inventory]

Coverage:
    • Pen
    • Pencil
    • Laptop
    • Book
    • CellPhone
    • Sofa
"""


from AI2Thor.baselines.utils.checker import BaseChecker

class Checker(BaseChecker):
    def __init__(self) -> None:
        subtasks = [
            'NavigateTo(Pen)',
             'PickUpObject(Pen)',
             'NavigateTo(Sofa, Pen)',
             'PutObject(Sofa, Pen)',
             'NavigateTo(Pencil)',
             'PickUpObject(Pencil)',
             'NavigateTo(Sofa, Pencil)',
             'PutObject(Sofa, Pencil)',
             'NavigateTo(Laptop)',
             'PickUpObject(Laptop)',
             'NavigateTo(Sofa, Laptop)',
             'PutObject(Sofa, Laptop)',
             'NavigateTo(Book)',
             'PickUpObject(Book)',
             'NavigateTo(Sofa, Book)',
             'PutObject(Sofa, Book)',
             'NavigateTo(CellPhone)',
             'PickUpObject(CellPhone)',
             'NavigateTo(Sofa, CellPhone)',
             'PutObject(Sofa, CellPhone)'
            ]

        conditional_subtasks = [
        'NavigateTo(Sofa, Pen)',
         'PutObject(Sofa, Pen)',
         'NavigateTo(Sofa, Pencil)',
         'PutObject(Sofa, Pencil)',
         'NavigateTo(Sofa, Laptop)',
         'PutObject(Sofa, Laptop)',
         'NavigateTo(Sofa, Book)',
         'PutObject(Sofa, Book)',
         'NavigateTo(Sofa, CellPhone)',
         'PutObject(Sofa, CellPhone)'
        ]


        independent_subtasks = [
            'NavigateTo(Pen)',
             'PickUpObject(Pen)',
             'NavigateTo(Pencil)',
             'PickUpObject(Pencil)',
             'NavigateTo(Laptop)',
             'PickUpObject(Laptop)',
             'NavigateTo(Book)',
             'PickUpObject(Book)',
             'NavigateTo(CellPhone)',
             'PickUpObject(CellPhone)'
            ]

        coverage = ["Pen", "Pencil", "Laptop", "Book", "CellPhone", "Sofa"]
        interact_objects = coverage
        interact_receptacles = ["Sofa"]

        super().__init__(
            subtasks,
            conditional_subtasks,
            independent_subtasks,
            coverage,
            interact_objects,
            interact_receptacles,
        )
