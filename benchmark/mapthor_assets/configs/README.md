## Configs

We define configs for the different tasks from MAP-THOR based on their types. Note that the tasks are classified based on the ambiguity in the instruction. (Refer to Appendix C in the paper)

- Type 1: Explicit object type, explicit quantity and target
- Type 2: Explicit object type and explicit target: Here we explicitly describe the object type but keep the quantity of the objects ambiguous. E.g. Put all the apples in the fridge. For this, the
  agents have to explore the environment to ensure that they find all of them.
- Type 3: Explicit target, implicit object types: The object types are implicitly defined whereas the target is explicitly defined. E.g. Put all groceries in the fridge. This tests whether the model
  can identify objects of certain categories.
- Type 4: Implicit target and object types: Here both the object type and the target are implicitly defined. E.g. Clear the floor by placing the items at their appropriate positions. Here the model is expected to keep items like pens, book, laptop on the study table, litter in the trash can, etc.

## Config Structure

Each of the config files has a list of tasks that fall under that category. Each task has the following attributes:

- `task_name`: used to instantiate the episode with the task instruction. This is just the code for the task
- `task_description`: This is the natural language description of the task. This is the input to the VLMs/LLMs.
- `task_id`: An ID for the task.
- `task_type`: What type of task it is. Will be either `transport` or `manipulation`
- `task_timeout`: Timeout for number of high-level decisions. In the paper we used 30 as timeout for all tasks since we were constrained by the costs associated with GPT-4. Ideally, a longer timeout should be used for the VLMs/LLMs for better success rates.
- `task_floorplans`: The floor plans that are most suitable for the given tasks. Note that not all floorplans in AI2THOR are compatible for all tasks.
- `task_checklist`: Minimal set of subtasks that are supposed to be executed for the completion of the task. Note that this is only used for the evaluation of the metrics at the end of the episode and not given as input to the VLMs/LLMs.

## Adding your own tasks:

You can create a similar config or append to existing configs to add more tasks. Refer to the `init_maker` folder for more information on how to initialize scenes based on your tasks.
