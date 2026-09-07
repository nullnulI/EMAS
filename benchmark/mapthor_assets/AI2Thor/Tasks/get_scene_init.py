"""
Map tasks to the file to import the right preinit function for the task.
"""

import os
import importlib.util


def import_module_from_file(file_path: str, module_name: str):
    """Import module from file path.
    Example:
    func = import_module_from_file('Tasks/4_clean_floor_kitchen/FloorPlan1.py','preinit')
    func(event, controller)
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_scene_initializer(task: str, floor_plan: str, ignore_applicable : bool = False):
    """Map the task to the file to import the right preinit function for the task.

    Args:
        task: str, task name

    Returns:
        str: file name to import the preinit function for the task
    # TODO @nsidn98 add semantic mapper for task to file name
    """
    file_path = os.path.dirname(os.path.realpath(__file__))
    file_path += "/" + task + "/" + floor_plan + ".py"

    if not ignore_applicable:
        # assertion error for tasks that do not exist
        assert os.path.exists(file_path), f'File {file_path}. Likely cause: {floor_plan} is not applicable to {task}.'

        checker_file_path = (
            os.path.dirname(os.path.abspath(__file__)) + "/" + task + "/checker.py"
        )
        # file_path += floor_plan + ".py"
        scene_initializer = import_module_from_file(file_path, "SceneInitializer")
        checker = import_module_from_file(checker_file_path, "Checker")

        return scene_initializer, checker
    
    return None, None


# if __name__ == "__main__":
#     import os

#     print(__file__)
#     print(os.path.dirname(os.path.realpath(__file__)))
