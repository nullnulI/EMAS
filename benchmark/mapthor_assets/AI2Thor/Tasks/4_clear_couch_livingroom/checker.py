"""
To check completion of subtasks in 2_put_all_tomatoes_potatoes_fridge task.
Also checks coverage of the task and success of the task.
Subtasks:
    • NavigateTo(KeyChain)
    • PickUpObject(KeyChain)
    • NavigateTo(Drawer) [with KeyChain in inventory]
    • OpenObject(Drawer) [with KeyChain in inventory]
    • PutObject(Drawer) [with KeyChain in inventory]
    • NavigateTo(Pencil)
    • PickUpObject(Pencil)
    • NavigateTo(Drawer) [with Pencil in inventory]
    • OpenObject(Drawer) [with Pencil in inventory]
    • PutObject(Drawer) [with Pencil in inventory]
    • NavigateTo(Pen)
    • PickUpObject(Pen)
    • NavigateTo(Drawer) [with Pen in inventory]
    • OpenObject(Drawer) [with Pen in inventory]
    • PutObject(Drawer) [with Pen in inventory]
    • NavigateTo(Book)
    • PickUpObject(Book)
    • NavigateTo(Drawer) [with Book in inventory]
    • OpenObject(Drawer) [with Book in inventory]
    • PutObject(Drawer) [with Book in inventory]
    • NavigateTo(Watch)
    • PickUpObject(Watch)
    • NavigateTo(Drawer) [with Watch in inventory]
    • OpenObject(Drawer) [with Watch in inventory]
    • PutObject(Drawer) [with Watch in inventory]

Coverage:
    • KeyChain
    • Pencil
    • Pen
    • Book
    • Watch
    • Drawer
"""


from AI2Thor.baselines.utils.checker import BaseChecker

class Checker(BaseChecker):
    def __init__(self) -> None:
        subtasks = [
        'NavigateTo(KeyChain)',
         'PickUpObject(KeyChain)',
         'NavigateTo(Drawer, KeyChain)',
         'OpenObject(Drawer, KeyChain)',
         'PutObject(Drawer, KeyChain)',
         'NavigateTo(Pencil)',
         'PickUpObject(Pencil)',
         'NavigateTo(Drawer, Pencil)',
         'OpenObject(Drawer, Pencil)',
         'PutObject(Drawer, Pencil)',
         'NavigateTo(Pen)',
         'PickUpObject(Pen)',
         'NavigateTo(Drawer, Pen)',
         'OpenObject(Drawer, Pen)',
         'PutObject(Drawer, Pen)',
         'NavigateTo(Book)',
         'PickUpObject(Book)',
         'NavigateTo(Drawer, Book)',
         'OpenObject(Drawer, Book)',
         'PutObject(Drawer, Book)',
         'NavigateTo(Watch)',
         'PickUpObject(Watch)',
         'NavigateTo(Drawer, Watch)',
         'OpenObject(Drawer, Watch)',
         'PutObject(Drawer, Watch)'
        ]

        conditional_subtasks = [
            'NavigateTo(Drawer, KeyChain)',
             'OpenObject(Drawer, KeyChain)',
             'PutObject(Drawer, KeyChain)',
             'NavigateTo(Drawer, Pencil)',
             'OpenObject(Drawer, Pencil)',
             'PutObject(Drawer, Pencil)',
             'NavigateTo(Drawer, Pen)',
             'OpenObject(Drawer, Pen)',
             'PutObject(Drawer, Pen)',
             'NavigateTo(Drawer, Book)',
             'OpenObject(Drawer, Book)',
             'PutObject(Drawer, Book)',
             'NavigateTo(Drawer, Watch)',
             'OpenObject(Drawer, Watch)',
             'PutObject(Drawer, Watch)'
            ]

        independent_subtasks = [
            'NavigateTo(KeyChain)',
             'PickUpObject(KeyChain)',
             'NavigateTo(Pencil)',
             'PickUpObject(Pencil)',
             'NavigateTo(Pen)',
             'PickUpObject(Pen)',
             'NavigateTo(Book)',
             'PickUpObject(Book)',
             'NavigateTo(Watch)',
             'PickUpObject(Watch)'
            ]

        # keychain, pencil, pen, book, watch + (drawer)
        coverage = ["KeyChain", "Pencil", "Pen", "Book", "Watch", "Drawer"]
        interact_objects = coverage
        interact_receptacles = ["Drawer"]

        super().__init__(
            subtasks,
            conditional_subtasks,
            independent_subtasks,
            coverage,
            interact_objects,
            interact_receptacles,
        )
