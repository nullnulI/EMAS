"""
Map tasks to strings of instructions
this will use a sentence transformer to map generic task instructions to s
pecific task instructions for preiniting the environment
Example: "I want a tomato, lettuce and bread in the fridge"  -> "Put a bread, lettuce and tomato in the fridge" -> "put_all_groceries_fridge"

NOTE: This is only for pre-intializing the environment with tasks and not as inputs to the model
"""

from typing import Tuple
import torch
from sentence_transformers import SentenceTransformer

task_mapper = {
    # type 1 tasks - ALL COMPLETED
    "Put the bread, lettuce, and tomato in the fridge": "1_put_bread_lettuce_tomato_fridge",
    "Put the computer, book, and remotecontrol on the sofa": "1_put_computer_book_remotecontrol_sofa",
    "Put the butter knife, bowl, and mug on the countertop": "1_put_knife_bowl_mug_countertop",
    "Put the plate, mug, and bowl in the fridge": "1_put_plate_mug_bowl_fridge",
    "Put the remotecontrol, keys, and watch in the box": "1_put_remotecontrol_keys_watch_box",
    "Put the vase, tissue box, and remote control on the table": "1_put_vase_tissuebox_remotecontrol_table",
    "Slice the bread, lettuce, tomato, and egg": "1_slice_bread_lettuce_tomato_egg",
    "Turn off the faucet and light if either is on": "1_turn_off_faucet_light",
    "Wash the bowl, mug, pot, and pan": "1_wash_bowl_mug_pot_pan",

    # Probably outdated:
    "Put the computer, book, and pen on the couch": "1_put_computer_book_pen_couch",
    "Put the bowl and tissue box on the table": "1_put_bowl_tissue_table",

    # type 2 tasks - ALL COMPLETED
    "Open all the drawers": "2_open_all_drawers",
    "Open all the cabinets": "2_open_all_cabinets",
    "Turn on all the stove knobs": "2_turn_on_all_stove_knobs",
    "Put all the vases on the countertop": "2_put_all_vases_countertop",
    "Put all the tomatoes and potatoes in the fridge": "2_put_all_tomatoes_potatoes_fridge",
    "Put all credit cards and remote controls in the box": "2_put_all_creditcards_remotecontrols_box",
    "Move all lamps next to the door": "2_move_all_lamps_door",

    # type 3 tasks
    "Put all groceries in the fridge": "3_put_all_groceries_fridge", # COMPLETED
    "Put all shakers in the fridge": "3_put_all_shakers_fridge", # COMPLETED
    "Put all silverware in any drawer": "3_put_all_silverware_drawer", # COMPLETED
    "Put all school supplies on the sofa": "3_put_all_school_supplies_sofa", # COMPLETED
    "Move everything on the table to the sofa": "3_clear_table_to_sofa", # COMPLETED

    "Put all kitchenware in the cardboard box": "3_put_all_kitchenware_box",

    # type 4 tasks
    "Clear the table by placing items at their appropriate positions": "4_clear_table_kitchen", # COMPLETED
    "Clear the kitchen central countertop by placing items in their appropriate positions": "4_clear_countertop_kitchen", # COMPLETED
    "Clear the couch by placing the items in other appropriate positions": "4_clear_couch_livingroom", # COMPLETED
    "Make the living room dark" : "4_make_livingroom_dark", # COMPLETED
    "Slice all sliceable objects" : "4_slice_all_sliceable", # COMPLETED
    "Put appropriate utensils in storage" : "4_put_appropriate_storage", # COMPLETED

}

all_orig_instructions = list(task_mapper.keys())

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
orig_instr_embeddings = torch.FloatTensor(model.encode(all_orig_instructions))


def get_closest_instruction(instruction: str) -> str:
    """To free-form instructions"""
    instr_embedding = torch.FloatTensor(model.encode([instruction]))
    scores = torch.cosine_similarity(orig_instr_embeddings, instr_embedding)
    max_score, max_idx = torch.max(scores, 0)
    return all_orig_instructions[max_idx]


def get_closest_task(instruction: str) -> Tuple[str, str]:
    """To convert instructions to tasks"""
    closest_orig_instr = get_closest_instruction(instruction)
    return task_mapper[closest_orig_instr], closest_orig_instr


if __name__ == "__main__":
    instruction = "I want a tomato, lettuce and bread in the fridge"
    print(get_closest_task(instruction))

    instruction = "Transport a tomato, lettuce and bread in the fridge"
    print(get_closest_task(instruction))

    instruction = "transport groceries to the fridge"
    print(get_closest_task(instruction))

    instruction = "I want to keep groceries in the fridge"
    print(get_closest_task(instruction))

    instruction = "can you clean the kitchen table"
    print(get_closest_task(instruction))

    instruction = "I want to clean the kitchen countertop"
    print(get_closest_task(instruction))
