## The tasks folder

## Naming convention:
Name the folder as: `<task_difficulty>_<task_description>`

Example: 1_transport_bread_lettuce_tomato

Nomenclature for `task_difficuly`:
* Explicit object type, quantity and target: 1
* Explicit object type and target: 2
* Explicit target, implicit object type: 3
* Impliciy target and object type: 4

In each task folder make one file for each FloorPlan (5 each)

Example of file structure:
```
├── Tasks
│   ├── 1_transport_bread_lettuce_tomato
│   │   ├── FloorPlan1.py
│   │   ├── FloorPlan2.py
│   │   ├── FloorPlan3.py
│   │   ├── FloorPlan4.py
│   │   ├── FloorPlan5.py
│   ├── 1_transport_knife_mug_bowl
│   │   ├── FloorPlan1.py
│   │   ├── FloorPlan2.py
│   │   ├── FloorPlan3.py
│   │   ├── FloorPlan4.py
│   │   ├── FloorPlan5.py
│   ├── 4_clean_floor_kitchen
│   │   ├── FloorPlan1.py
│   │   ├── FloorPlan2.py
│   │   ├── FloorPlan3.py
│   │   ├── FloorPlan4.py
│   │   ├── FloorPlan5.py
│   ├── README.md
│   ├── map_tasks.py
```