"""
List objects in a ProcTHOR-10K house without starting AI2-THOR.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import prior


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
EMAS_ROOT = SCRIPT_DIR.parents[4]
AI2THOR_CONTROLLER_PATH = EMAS_ROOT / "ai2thor" / "ai2thor" / "controller.py"
AI2THOR_SIMOBJTYPE_PATH = EMAS_ROOT / "ai2thor" / "unity" / "Assets" / "Scripts" / "SimObjType.cs"


def parse_controller_receptacles(path: Path = AI2THOR_CONTROLLER_PATH) -> dict[str, set[str]]:
    if not path.exists():
        return {}

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "RECEPTACLE_OBJECTS" for target in node.targets):
            continue
        value = ast.literal_eval(node.value)
        return {str(key): {str(item) for item in items} for key, items in value.items()}
    return {}


def parse_placement_restrictions(path: Path = AI2THOR_SIMOBJTYPE_PATH) -> dict[str, set[str]]:
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8")
    start = text.find("PlacementRestrictions")
    if start < 0:
        return {}
    text = text[start:]

    restrictions: dict[str, set[str]] = {}
    pattern = re.compile(
        r"\{\s*SimObjType\.(?P<object_type>\w+)\s*,\s*new List\s*<\s*SimObjType\s*>\s*\(\)\s*\{(?P<body>.*?)\}\s*\}",
        re.DOTALL,
    )
    for match in pattern.finditer(text):
        obj_type = match.group("object_type")
        targets = set(re.findall(r"SimObjType\.(\w+)", match.group("body")))
        restrictions[obj_type] = targets
    return restrictions


def build_interaction_info() -> dict[str, dict[str, Any]]:
    receptacle_objects = parse_controller_receptacles()
    placement_restrictions = parse_placement_restrictions()

    all_types = set(receptacle_objects)
    all_types.update(placement_restrictions)
    for values in receptacle_objects.values():
        all_types.update(values)
    for values in placement_restrictions.values():
        all_types.update(values)

    info: dict[str, dict[str, Any]] = {}
    for obj_type in sorted(all_types):
        can_be_placed_in = set(placement_restrictions.get(obj_type, set()))
        can_contain = set(receptacle_objects.get(obj_type, set()))
        can_contain.update(
            placed_type
            for placed_type, targets in placement_restrictions.items()
            if obj_type in targets
        )

        info[obj_type] = {
            "pickupable": bool(can_be_placed_in),
            "receptacle": bool(can_contain),
            "can_be_placed_in": sorted(can_be_placed_in),
            "can_contain": sorted(can_contain),
        }
    return info


def object_type(obj: dict[str, Any]) -> str:
    obj_id = obj.get("id")
    if isinstance(obj_id, str) and "|" in obj_id:
        return obj_id.split("|", 1)[0]
    if isinstance(obj_id, str) and obj_id:
        return obj_id

    asset_id = obj.get("assetId")
    if isinstance(asset_id, str) and asset_id:
        return asset_id.split("_", 1)[0]

    return "Unknown"


def iter_objects(objects: Iterable[dict[str, Any]], *, parent_id: str | None = None):
    for obj in objects:
        item = dict(obj)
        item["_parentId"] = parent_id
        yield item

        children = obj.get("children") or []
        yield from iter_objects(children, parent_id=obj.get("id"))


def parse_scene_name(scene: str) -> tuple[str, int]:
    try:
        split, raw_index = scene.rsplit("_", 1)
        if split not in {"train", "val", "test"}:
            raise ValueError
        return split, int(raw_index)
    except ValueError as exc:
        raise ValueError(
            "Scene must look like train_0, val_12, or test_3. "
            "Alternatively pass --split and --index."
        ) from exc


def load_house(args: argparse.Namespace) -> tuple[str, int, dict[str, Any]]:
    split, index = (args.split, args.index)
    if args.scene is not None:
        split, index = parse_scene_name(args.scene)

    if split is None or index is None:
        raise ValueError("Pass --scene train_0 or pass both --split and --index.")

    kwargs = {"offline": args.offline}
    if args.revision is not None:
        kwargs["revision"] = args.revision
    dataset = prior.load_dataset("procthor-10k", **kwargs)
    return split, index, dataset[split][index]


def short_list(items: list[str], *, limit: int = 8) -> str:
    if not items:
        return "[]"
    shown = items[:limit]
    suffix = f", ... +{len(items) - limit}" if len(items) > limit else ""
    return "[" + ", ".join(shown) + suffix + "]"


def object_interaction(obj_type: str, interaction_info: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return interaction_info.get(
        obj_type,
        {
            "pickupable": False,
            "receptacle": False,
            "can_be_placed_in": [],
            "can_contain": [],
        },
    )


def enrich_objects(objects: list[dict[str, Any]], interaction_info: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for obj in objects:
        item = dict(obj)
        item["_objectType"] = object_type(obj)
        item["_interaction"] = object_interaction(item["_objectType"], interaction_info)
        enriched.append(item)
    return enriched


def format_text(
    split: str,
    index: int,
    house: dict[str, Any],
    objects: list[dict[str, Any]],
    interaction_info: dict[str, dict[str, Any]],
) -> str:
    house_id = house.get("id") or house.get("metadata", {}).get("id") or f"{split}_{index}"
    lines = [
        f"House: {house_id} ({split}_{index})",
        f"Rooms: {len(house.get('rooms') or [])}",
        f"Top-level objects: {len(house.get('objects') or [])}",
        f"Objects including children: {len(objects)}",
        "Interaction fields: type-level inference from local AI2-THOR tables; runtime visibility/distance is not checked.",
        "",
        "Object types:",
    ]

    for obj_type, count in sorted(Counter(object_type(obj) for obj in objects).items()):
        interaction = object_interaction(obj_type, interaction_info)
        lines.append(
            f"  {obj_type}: {count} "
            f"pickupable={interaction['pickupable']} "
            f"receptacle={interaction['receptacle']}"
        )

    lines.extend(["", "Objects:"])
    for obj in objects:
        obj_type = object_type(obj)
        interaction = object_interaction(obj_type, interaction_info)
        pos = obj.get("position") or {}
        parent = obj.get("_parentId")
        parent_text = f" parent={parent}" if parent else ""
        lines.append(
            f"{obj_type:<18} "
            f"id={obj.get('id', ''):<36} "
            f"asset={obj.get('assetId', ''):<36} "
            f"pos=({pos.get('x', 0):.2f}, {pos.get('y', 0):.2f}, {pos.get('z', 0):.2f})"
            f" pickupable={interaction['pickupable']}"
            f" receptacle={interaction['receptacle']}"
            f" can_be_placed_in={short_list(interaction['can_be_placed_in'])}"
            f" can_contain={short_list(interaction['can_contain'])}"
            f"{parent_text}"
        )
    return "\n".join(lines) + "\n"


def result_path(args: argparse.Namespace, split: str, index: int) -> Path:
    if args.output is not None:
        return args.output

    suffixes = []
    if args.parents_only:
        suffixes.append("parents_only")
    if args.json:
        suffixes.append("json")
    suffix = "_" + "_".join(suffixes) if suffixes else ""
    return DEFAULT_RESULTS_DIR / f"{split}_{index}_objects{suffix}.txt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", help="ProcTHOR scene name, e.g. train_0, val_12, test_3.")
    parser.add_argument("--split", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int)
    parser.add_argument("--parents-only", action="store_true", help="Only list top-level house objects.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output text file. Defaults to ./results/<scene>_objects.txt beside this script.",
    )
    parser.add_argument("--offline", action="store_true", help="Use only the local prior cache.")
    parser.add_argument(
        "--revision",
        default=None,
        help=(
            "Optional ProcTHOR dataset revision. For older AI2-THOR builds, "
            "ab3cacd0fc17754d4c080a3fd50b18395fae8647 is commonly used."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    split, index, house = load_house(args)
    objects = list(house.get("objects") or []) if args.parents_only else list(iter_objects(house.get("objects") or []))
    interaction_info = build_interaction_info()

    if args.json:
        output = json.dumps(enrich_objects(objects, interaction_info), indent=2) + "\n"
    else:
        output = format_text(split, index, house, objects, interaction_info)

    output_path = result_path(args, split, index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")
    print(f"Wrote {len(objects)} objects to {output_path}")


if __name__ == "__main__":
    main()
