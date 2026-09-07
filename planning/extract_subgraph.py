'''
Extract task-relevant subgraphs from ConceptGraphs outputs.

This script reads node metadata from either:
1. scene_graph.json
2. map/scene_map_cfslam_pruned.pkl.gz

And edge metadata from either:
1. cfslam_scenegraph_edges.pkl
2. cfslam_object_relations.json (fallback only)

The default task mode does:
task text -> task parsing -> seed node retrieval -> budgeted k-hop expansion
-> shortest path completion between seeds -> planner-friendly JSON.

Relation expressions such as "a in b" and "a on b" are normalized as:
container/support -> contained/supported.
'''

from __future__ import annotations

import argparse
import gzip
import heapq
import json
import math
import os
import pickle
import re
import sys
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

EMAS_ROOT = Path(__file__).resolve().parents[1]
CONCEPTGRAPH_REPO_ROOT = EMAS_ROOT / "memory" / "concept-graphs"
for path in (EMAS_ROOT, CONCEPTGRAPH_REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from planning.model_config import DEFAULT_PLANNING_MODEL_PATH


SUPPORTED_DIRECTED_RELATIONS = {
    "a on b",
    "b on a",
    "a in b",
    "b in a",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "bring",
    "by",
    "find",
    "for",
    "from",
    "get",
    "go",
    "grab",
    "in",
    "into",
    "is",
    "it",
    "me",
    "move",
    "near",
    "of",
    "on",
    "onto",
    "pick",
    "place",
    "put",
    "take",
    "the",
    "then",
    "to",
    "up",
    "with",
}

ACTION_KEYWORDS = {
    "pick": {"pick", "grab", "take", "get", "fetch", "collect"},
    "place": {"put", "place", "drop", "insert", "store", "set"},
    "navigate": {"go", "navigate", "move", "walk", "reach", "approach"},
    "find": {"find", "locate", "search", "look"},
    "inspect": {"inspect", "check", "observe", "look"},
}

BASE_RELATION_WEIGHTS = {
    "in": 1.0,
    "inside": 1.0,
    "contains": 1.0,
    "contain": 1.0,
    "on": 0.95,
    "support": 0.95,
    "supports": 0.95,
    "same room": 0.85,
    "same_room": 0.85,
    "room": 0.8,
    "near": 0.7,
    "next to": 0.65,
    "next_to": 0.65,
    "beside": 0.65,
    "close to": 0.65,
    "front": 0.45,
    "behind": 0.45,
    "left": 0.4,
    "right": 0.4,
}

TASK_RELATION_BOOSTS = {
    "pick": {"in": 0.25, "on": 0.25, "contains": 0.2, "support": 0.2, "near": 0.05},
    "place": {"in": 0.3, "on": 0.3, "contains": 0.25, "support": 0.25, "near": 0.1},
    "navigate": {"near": 0.25, "same room": 0.25, "same_room": 0.25, "room": 0.2},
    "find": {"in": 0.2, "on": 0.2, "near": 0.15, "same room": 0.15, "same_room": 0.15},
}


@dataclass
class TaskSpec:
    raw_task: str
    task_type: str = "generic"
    actions: list[str] = field(default_factory=list)
    target_objects: list[str] = field(default_factory=list)
    source_receptacles: list[str] = field(default_factory=list)
    destination_receptacles: list[str] = field(default_factory=list)
    landmarks: list[str] = field(default_factory=list)
    spatial_relations: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    entity_mentions: list[dict[str, Any]] = field(default_factory=list)
    parser: str = "heuristic"


def _normalize_node_record(node: dict, fallback_pruned_id: int) -> dict:
    normalized = dict(node)
    normalized.setdefault("pruned_id", fallback_pruned_id)
    normalized.setdefault("original_id", normalized.get("id", fallback_pruned_id))
    return normalized


def _to_plain_list(value: Any) -> list | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return value
    return None


def _round_vector(value: Any, digits: int = 3) -> list[float] | None:
    values = _to_plain_list(value)
    if values is None:
        return None
    try:
        return [round(float(item), digits) for item in values]
    except (TypeError, ValueError):
        return None


def _bbox_fields_from_node(node: dict) -> dict:
    if not isinstance(node, dict):
        return {}
    bbox = node.get("bbox")
    center = _round_vector(getattr(bbox, "center", None))
    extent = _round_vector(getattr(bbox, "extent", None))

    if center is None or extent is None:
        points = _to_plain_list(node.get("bbox_np"))
        if points:
            try:
                coordinates = [[float(coord) for coord in point[:3]] for point in points]
                mins = [min(values) for values in zip(*coordinates)]
                maxs = [max(values) for values in zip(*coordinates)]
                center = [round((lo + hi) / 2.0, 3) for lo, hi in zip(mins, maxs)]
                extent = [round(hi - lo, 3) for lo, hi in zip(mins, maxs)]
            except (TypeError, ValueError):
                center = None
                extent = None

    fields = {}
    if center is not None:
        fields["bbox_center"] = center
    if extent is not None:
        fields["bbox_extent"] = extent
    return fields


def load_nodes_from_scene_graph_json(scene_graph_path: Path) -> dict[int, dict]:
    with open(scene_graph_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    normalized_nodes = {}
    for fallback_pruned_id, node in enumerate(nodes):
        normalized = _normalize_node_record(node, fallback_pruned_id)
        normalized_nodes[int(normalized["pruned_id"])] = normalized

    return normalized_nodes


def load_nodes_from_pruned_scene_map(scene_map_path: Path) -> dict[int, dict]:
    with gzip.open(scene_map_path, "rb") as f:
        scene_map = pickle.load(f)

    normalized_nodes = {}
    for pruned_id, node in enumerate(scene_map):
        caption_dict = node.get("caption_dict", {}) if isinstance(node, dict) else {}
        response = caption_dict.get("response", {}) if isinstance(caption_dict, dict) else {}
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                response = {}
        normalized = {
            "pruned_id": pruned_id,
            "id": pruned_id,
            "original_id": caption_dict.get("id", pruned_id),
            "object_tag": response.get("object_tag", ""),
            "caption": response.get("summary", ""),
            "possible_tags": response.get("possible_tags", []),
        }
        normalized.update(_bbox_fields_from_node(node))
        class_name = node.get("class_name") if isinstance(node, dict) else None
        if class_name:
            normalized["class_name"] = class_name
        normalized_nodes[pruned_id] = normalized

    return normalized_nodes


def load_invalid_indices(invalid_indices_path: Path) -> set[int]:
    if not invalid_indices_path.exists():
        return set()
    with open(invalid_indices_path, "rb") as f:
        invalid_indices = pickle.load(f)
    return {int(index) for index in invalid_indices}


def load_nodes_from_response_dir(response_dir: Path) -> dict[int, dict]:
    invalid_indices = load_invalid_indices(response_dir.parent / "cfslam_scenegraph_invalid_indices.pkl")
    response_files = sorted(
        response_dir.glob("*.json"),
        key=lambda file_path: int(file_path.stem) if file_path.stem.isdigit() else file_path.stem,
    )

    normalized_nodes = {}
    pruned_id = 0
    for response_file in response_files:
        with open(response_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        original_id = payload.get("id")
        try:
            original_id_int = int(original_id)
        except (TypeError, ValueError):
            continue
        if original_id_int in invalid_indices:
            continue

        response = payload.get("response", {})
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                continue

        object_tag = str(response.get("object_tag", "")).strip().lower()
        if object_tag in {"invalid", "fail"}:
            continue

        normalized = {
            "pruned_id": pruned_id,
            "id": pruned_id,
            "original_id": original_id_int,
            "object_tag": response.get("object_tag", ""),
            "caption": response.get("summary", ""),
            "possible_tags": response.get("possible_tags", []),
        }
        if "bbox_center" in payload:
            normalized["bbox_center"] = payload["bbox_center"]
        if "bbox_extent" in payload:
            normalized["bbox_extent"] = payload["bbox_extent"]
        normalized_nodes[pruned_id] = normalized
        pruned_id += 1

    return normalized_nodes


def load_nodes(scene_graph_path: Path) -> dict[int, dict]:
    if scene_graph_path.is_dir():
        return load_nodes_from_response_dir(scene_graph_path)
    if scene_graph_path.suffix == ".json":
        return load_nodes_from_scene_graph_json(scene_graph_path)
    if scene_graph_path.name.endswith(".pkl.gz"):
        return load_nodes_from_pruned_scene_map(scene_graph_path)
    raise ValueError(f"Unsupported node metadata file: {scene_graph_path}")


def build_id_lookup(nodes_by_pruned_id: dict[int, dict]) -> dict[int, int]:
    id_lookup = {}
    for pruned_id, node in nodes_by_pruned_id.items():
        for key in ("pruned_id", "original_id", "id"):
            if key in node:
                try:
                    id_lookup[int(node[key])] = pruned_id
                except (TypeError, ValueError):
                    continue
    return id_lookup


def resolve_relation_node_id(raw_id: Any, id_lookup: dict[int, int]) -> int | None:
    try:
        numeric_id = int(raw_id)
    except (TypeError, ValueError):
        return None
    return id_lookup.get(numeric_id, numeric_id)


def normalize_relation_name(relation_name: str) -> str:
    relation_name = relation_name.strip().lower().replace("_", " ")
    relation_name = re.sub(r"\s+", " ", relation_name)
    if relation_name in {"a on b", "b on a"}:
        return "on"
    if relation_name in {"a in b", "b in a"}:
        return "in"
    if "next to" in relation_name:
        return "next to"
    if "same room" in relation_name:
        return "same room"
    for key in BASE_RELATION_WEIGHTS:
        if key in relation_name:
            return key
    return relation_name


def build_normalized_edge(
    object1_id_raw: Any,
    object2_id_raw: Any,
    relation_name: str,
    id_lookup: dict[int, int],
    metadata: dict | None = None,
) -> dict | None:
    relation_name = relation_name.strip().lower()
    if not relation_name or relation_name not in SUPPORTED_DIRECTED_RELATIONS:
        return None

    object1_id = resolve_relation_node_id(object1_id_raw, id_lookup)
    object2_id = resolve_relation_node_id(object2_id_raw, id_lookup)
    if object1_id is None or object2_id is None:
        return None

    if relation_name in {"a on b", "a in b"}:
        source_id = object2_id
        target_id = object1_id
    else:
        source_id = object1_id
        target_id = object2_id

    edge = dict(metadata or {})
    edge["source"] = source_id
    edge["target"] = target_id
    edge["object_relation"] = relation_name
    edge["normalized_relation"] = normalize_relation_name(relation_name)
    return edge


def normalize_relation_edge(relation: dict, id_lookup: dict[int, int]) -> dict | None:
    relation_name = relation.get("object_relation", "")
    object1 = relation.get("object1", {})
    object2 = relation.get("object2", {})
    return build_normalized_edge(
        object1.get("id"),
        object2.get("id"),
        relation_name,
        id_lookup,
        metadata=relation,
    )


def load_edges_from_relations_json(relations_path: Path, nodes_by_pruned_id: dict[int, dict]) -> list[dict]:
    with open(relations_path, "r", encoding="utf-8") as f:
        relations = json.load(f)

    id_lookup = build_id_lookup(nodes_by_pruned_id)
    edges = []
    for relation in relations:
        edge = normalize_relation_edge(relation, id_lookup)
        if edge is not None:
            edges.append(edge)

    return edges


def load_edges_from_scenegraph_edges(edges_path: Path, nodes_by_pruned_id: dict[int, dict]) -> list[dict]:
    with open(edges_path, "rb") as f:
        raw_edges = pickle.load(f)

    id_lookup = build_id_lookup(nodes_by_pruned_id)
    edges = []
    for item in raw_edges:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            continue
        object1_id, object2_id, relation_name = item
        edge = build_normalized_edge(
            object1_id,
            object2_id,
            str(relation_name),
            id_lookup,
            metadata={
                "object1": {"id": object1_id},
                "object2": {"id": object2_id},
            },
        )
        if edge is not None:
            edges.append(edge)

    return edges


def load_edges(relations_path: Path, nodes_by_pruned_id: dict[int, dict]) -> list[dict]:
    if relations_path.suffix == ".json":
        return load_edges_from_relations_json(relations_path, nodes_by_pruned_id)
    if relations_path.suffix == ".pkl":
        return load_edges_from_scenegraph_edges(relations_path, nodes_by_pruned_id)
    raise ValueError(f"Unsupported edge metadata file: {relations_path}")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", text.lower())


def unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        item = item.strip().lower()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def build_object_tag_text(node: dict) -> str:
    parts = [
        str(node.get("object_tag", "")),
        str(node.get("caption", "")),
        " ".join(str(tag) for tag in node.get("possible_tags", [])),
    ]
    return " ".join(parts).lower()


def node_label(node: dict) -> str:
    label = str(node.get("object_tag") or node.get("caption") or node.get("pruned_id"))
    return label.strip()


def compact_node(node: dict, score: float | None = None) -> dict:
    keep_keys = (
        "pruned_id",
        "original_id",
        "object_tag",
        "caption",
        "possible_tags",
        "bbox_center",
        "bbox_extent",
        "class_name",
    )
    compact = {key: node.get(key) for key in keep_keys if key in node}
    if score is not None:
        compact["retrieval_score"] = round(score, 4)
    return compact


def compact_edge(edge: dict) -> dict:
    return {
        "source": edge.get("source"),
        "target": edge.get("target"),
        "object_relation": edge.get("object_relation"),
        "normalized_relation": edge.get("normalized_relation"),
    }


def describe_node(node: dict) -> str:
    return (
        f"pruned_id={node['pruned_id']}, "
        f"original_id={node['original_id']}, "
        f"object_tag={node.get('object_tag', '')}"
    )


def score_object_tag_match(query: str, node: dict) -> tuple[float, list[str]]:
    query = query.strip().lower()
    object_tag = str(node.get("object_tag", "")).strip().lower()
    node_text = build_object_tag_text(node)
    query_tokens = set(tokenize(query))
    object_tag_tokens = set(tokenize(object_tag))
    node_tokens = set(tokenize(node_text))

    score = 0.0
    reasons = []

    if object_tag == query:
        score += 12.0
        reasons.append("exact_object_tag")
    elif query and query in object_tag:
        score += 7.0
        reasons.append("substring_object_tag")
    elif query and query in node_text:
        score += 4.0
        reasons.append("substring_node_text")

    if query_tokens:
        overlap_with_tag = query_tokens & object_tag_tokens
        overlap_with_node = query_tokens & node_tokens
        if overlap_with_tag:
            score += 2.5 * len(overlap_with_tag)
            reasons.append("token_overlap_tag:" + ",".join(sorted(overlap_with_tag)))
        elif overlap_with_node:
            score += 1.5 * len(overlap_with_node)
            reasons.append("token_overlap_node:" + ",".join(sorted(overlap_with_node)))

        if query_tokens and query_tokens <= object_tag_tokens:
            score += 2.0
            reasons.append("all_query_tokens_in_object_tag")

    score -= 0.01 * node.get("pruned_id", 0)
    return score, reasons


def rank_object_tag_matches(query: str, matches: list[dict]) -> list[dict]:
    ranked = []
    for node in matches:
        score, reasons = score_object_tag_match(query, node)
        ranked.append({
            "node": node,
            "score": score,
            "reasons": reasons,
        })

    ranked.sort(
        key=lambda item: (
            item["score"],
            -len(str(item["node"].get("object_tag", ""))),
            -item["node"].get("pruned_id", 0),
        ),
        reverse=True,
    )
    return ranked


def resolve_target_node(
    args: argparse.Namespace,
    nodes_by_pruned_id: dict[int, dict],
) -> tuple[dict, dict | None]:
    if args.pruned_id is not None:
        if args.pruned_id not in nodes_by_pruned_id:
            raise ValueError(f"pruned_id {args.pruned_id} not found in node metadata")
        return nodes_by_pruned_id[args.pruned_id], {
            "selection_method": "pruned_id",
            "query": args.pruned_id,
        }

    if args.original_id is not None:
        matches = [
            node for node in nodes_by_pruned_id.values() if node["original_id"] == args.original_id
        ]
        if not matches:
            raise ValueError(f"original_id {args.original_id} not found in node metadata")
        if len(matches) > 1:
            raise ValueError(
                "original_id matches multiple nodes: "
                + "; ".join(describe_node(node) for node in matches)
            )
        return matches[0], {
            "selection_method": "original_id",
            "query": args.original_id,
        }

    query = args.object_tag.lower()
    matches = [
        node for node in nodes_by_pruned_id.values() if query in build_object_tag_text(node)
    ]
    if not matches:
        raise ValueError(f"object_tag query '{args.object_tag}' did not match any node")

    ranked_matches = rank_object_tag_matches(query, matches)
    selected = ranked_matches[0]
    selection_metadata = {
        "selection_method": "object_tag_best_match",
        "query": args.object_tag,
        "num_matches": len(ranked_matches),
        "selected_score": round(selected["score"], 4),
        "selected_reasons": selected["reasons"],
        "candidates": [
            {
                "pruned_id": item["node"].get("pruned_id"),
                "original_id": item["node"].get("original_id"),
                "object_tag": item["node"].get("object_tag", ""),
                "score": round(item["score"], 4),
                "reasons": item["reasons"],
            }
            for item in ranked_matches[:10]
        ],
    }
    return selected["node"], selection_metadata


def build_adjacency(
    edges: list[dict],
) -> tuple[dict[int, list[dict]], dict[int, list[dict]], dict[int, list[tuple[int, dict]]]]:
    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    undirected = defaultdict(list)

    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        outgoing[source].append(edge)
        incoming[target].append(edge)
        undirected[source].append((target, edge))
        undirected[target].append((source, edge))

    return incoming, outgoing, undirected


def _strip_entity_determiner(value: str) -> str:
    return re.sub(r"^(?:the|a|an)\s+", "", value.strip(" \t\r\n,.;:!?"), flags=re.IGNORECASE)


def _split_entity_list(value: str) -> list[str]:
    normalized = re.sub(r"\s*,\s*(?:and\s+)?", ",", value.strip(), flags=re.IGNORECASE)
    normalized = re.sub(r"\s+\band\b\s+", ",", normalized, flags=re.IGNORECASE)
    return unique_preserve_order([
        entity
        for part in normalized.split(",")
        if (entity := _strip_entity_determiner(part))
    ])


def _entity_mention(task: str, entity: str, role: str) -> dict[str, Any]:
    words = [re.escape(word) for word in tokenize(entity)]
    pattern = r"\b" + r"[\s_-]+".join(words) + r"\b" if words else None
    match = re.search(pattern, task, flags=re.IGNORECASE) if pattern else None
    return {
        "text": entity,
        "role": role,
        "start": match.start() if match else None,
        "end": match.end() if match else None,
    }


def _parse_explicit_place_task(task: str, actions: list[str]) -> TaskSpec | None:
    match = re.search(
        r"\b(?:put|place|set)\b\s+(.+?)\s+\b(on|onto|in|into|inside)\b\s+(.+?)\s*[.!?]*$",
        task,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None

    sources = _split_entity_list(match.group(1))
    destination = _strip_entity_determiner(match.group(3))
    if not sources or not destination:
        return None

    raw_relation = match.group(2).lower()
    relation = "on" if raw_relation in {"on", "onto"} else "in"
    mentions = [
        *[_entity_mention(task, source, "target_object") for source in sources],
        _entity_mention(task, destination, "destination_receptacle"),
    ]
    phrases = unique_preserve_order([*sources, destination])
    return TaskSpec(
        raw_task=task,
        task_type="place",
        actions=unique_preserve_order([*actions, "place"]),
        target_objects=sources,
        destination_receptacles=[destination],
        spatial_relations=[relation],
        keywords=phrases,
        entity_mentions=mentions,
        parser="heuristic_explicit_place",
    )


def parse_task_locally(task: str) -> TaskSpec:
    tokens = tokenize(task)
    actions = []
    for action, keywords in ACTION_KEYWORDS.items():
        if any(token in keywords for token in tokens):
            actions.append(action)

    explicit_place = _parse_explicit_place_task(task, actions)
    if explicit_place is not None:
        return explicit_place

    task_type = actions[0] if actions else "generic"
    if "place" in actions:
        task_type = "place"
    elif "pick" in actions:
        task_type = "pick"

    keywords = [
        token
        for token in tokens
        if token not in STOPWORDS and len(token) > 1 and not token.isdigit()
    ]
    relation_words = [
        relation
        for relation in ("on", "in", "inside", "near", "next to", "left", "right", "front", "behind")
        if relation in task.lower()
    ]

    return TaskSpec(
        raw_task=task,
        task_type=task_type,
        actions=unique_preserve_order(actions),
        target_objects=unique_preserve_order(keywords),
        spatial_relations=unique_preserve_order(relation_words),
        keywords=unique_preserve_order(keywords),
    )


def parse_json_from_text(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def parse_task_with_qwen(
    task: str,
    model_path: str | None,
    conv_mode: str,
    num_gpus: int,
) -> TaskSpec | None:
    try:
        from conceptgraph.vlm import build_vlm_chat
    except Exception:
        return None

    system_prompt = (
        "Parse an embodied robot planning task into retrieval hints for a 3D scene graph. "
        "Return only valid JSON with the requested schema and no extra commentary."
    )
    user_prompt = {
        "task": task,
        "schema": {
            "task_type": "one of pick, place, navigate, find, inspect, generic",
            "actions": ["verb strings"],
            "target_objects": ["objects directly manipulated or searched"],
            "source_receptacles": ["containers/supports where targets may start"],
            "destination_receptacles": ["containers/supports/places where targets should end"],
            "landmarks": ["nearby landmarks or rooms"],
            "spatial_relations": ["on/in/near/left/right/etc mentioned or implied"],
            "keywords": ["short retrieval keywords"],
        },
    }

    try:
        chat = build_vlm_chat(
            backend="qwen",
            model_path=model_path,
            conv_mode=conv_mode,
            num_gpus=num_gpus,
        )
        if hasattr(chat, "reset"):
            chat.reset()
        if hasattr(chat, "messages"):
            chat.messages = [{"role": "system", "content": system_prompt}]
        response_text = chat(json.dumps(user_prompt, ensure_ascii=False, indent=2))
    except Exception:
        return None

    parsed = parse_json_from_text(response_text)
    if parsed is None:
        return None

    local_fallback = parse_task_locally(task)
    return TaskSpec(
        raw_task=task,
        task_type=str(parsed.get("task_type") or local_fallback.task_type),
        actions=unique_preserve_order(list(parsed.get("actions") or local_fallback.actions)),
        target_objects=unique_preserve_order(
            list(parsed.get("target_objects") or local_fallback.target_objects)
        ),
        source_receptacles=unique_preserve_order(list(parsed.get("source_receptacles") or [])),
        destination_receptacles=unique_preserve_order(
            list(parsed.get("destination_receptacles") or [])
        ),
        landmarks=unique_preserve_order(list(parsed.get("landmarks") or [])),
        spatial_relations=unique_preserve_order(
            list(parsed.get("spatial_relations") or local_fallback.spatial_relations)
        ),
        keywords=unique_preserve_order(list(parsed.get("keywords") or local_fallback.keywords)),
        parser="qwen_local",
    )


def parse_task(args: argparse.Namespace) -> TaskSpec:
    if args.use_qwen:
        qwen_spec = parse_task_with_qwen(
            task=args.task,
            model_path=args.qwen_model_path,
            conv_mode=args.qwen_conv_mode,
            num_gpus=args.qwen_num_gpus,
        )
        if qwen_spec is not None:
            return qwen_spec
    return parse_task_locally(args.task)


def task_phrases(task_spec: TaskSpec) -> list[str]:
    phrases = []
    phrases.extend(task_spec.target_objects)
    phrases.extend(task_spec.source_receptacles)
    phrases.extend(task_spec.destination_receptacles)
    phrases.extend(task_spec.landmarks)
    phrases.extend(task_spec.keywords)
    return unique_preserve_order(phrases)


def node_text_parts(node: dict) -> list[str]:
    parts = [str(node.get("object_tag", "")), str(node.get("caption", ""))]
    parts.extend(str(tag) for tag in node.get("possible_tags", []))
    return [part.lower() for part in parts if part]


def score_node_for_task(node: dict, task_spec: TaskSpec) -> tuple[float, list[str]]:
    node_parts = node_text_parts(node)
    node_text = " ".join(node_parts)
    node_tokens = set(tokenize(node_text))
    task_tokens = set(tokenize(task_spec.raw_task))
    phrases = task_phrases(task_spec)

    score = 0.0
    reasons = []

    for phrase in phrases:
        phrase_tokens = set(tokenize(phrase))
        if not phrase_tokens:
            continue
        if any(phrase == part for part in node_parts):
            score += 6.0
            reasons.append(f"exact:{phrase}")
        elif phrase in node_text:
            score += 3.0
            reasons.append(f"substring:{phrase}")
        else:
            overlap = phrase_tokens & node_tokens
            if overlap:
                overlap_weight = 1.2 if len(phrase_tokens) == 1 else 0.25
                score += overlap_weight * len(overlap) / math.sqrt(len(phrase_tokens))
                reasons.append("token_overlap:" + ",".join(sorted(overlap)))

    overlap = (task_tokens - STOPWORDS) & node_tokens
    if overlap:
        score += 0.4 * len(overlap)
        reasons.append("task_overlap:" + ",".join(sorted(overlap)))

    normalized_phrases = {
        " ".join(tokenize(phrase))
        for phrase in phrases
        if tokenize(phrase)
    }
    normalized_tag = " ".join(tokenize(str(node.get("object_tag", ""))))
    if normalized_tag and normalized_tag in normalized_phrases:
        score += 2.0
        reasons.append("tag_in_task")

    return score, unique_preserve_order(reasons)


def retrieve_seed_nodes(
    task_spec: TaskSpec,
    nodes_by_pruned_id: dict[int, dict],
    max_seed_nodes: int,
    min_seed_score: float,
) -> tuple[list[int], dict[int, float], dict[int, list[str]]]:
    scored_nodes = []
    node_scores = {}
    node_reasons = {}
    for node_id, node in nodes_by_pruned_id.items():
        score, reasons = score_node_for_task(node, task_spec)
        if score > 0:
            scored_nodes.append((score, node_id))
            node_scores[node_id] = score
            node_reasons[node_id] = reasons

    scored_nodes.sort(reverse=True)
    seed_ids = [
        node_id for score, node_id in scored_nodes[:max_seed_nodes] if score >= min_seed_score
    ]

    if not seed_ids and scored_nodes:
        seed_ids = [scored_nodes[0][1]]

    return seed_ids, node_scores, node_reasons


def relation_weight(edge: dict, task_spec: TaskSpec) -> float:
    relation = edge.get("normalized_relation") or edge.get("object_relation", "")
    relation = str(relation).lower()
    weight = 0.35
    for key, value in BASE_RELATION_WEIGHTS.items():
        if key in relation:
            weight = max(weight, value)

    boosts = TASK_RELATION_BOOSTS.get(task_spec.task_type, {})
    for key, value in boosts.items():
        if key in relation:
            weight += value
    for relation_hint in task_spec.spatial_relations:
        if relation_hint in relation:
            weight += 0.2

    return min(weight, 1.5)


def edge_key(edge: dict) -> tuple[int, int, str]:
    return (
        int(edge["source"]),
        int(edge["target"]),
        str(edge.get("object_relation", edge.get("normalized_relation", ""))),
    )


def find_shortest_path(
    start_id: int,
    goal_id: int,
    undirected_edges: dict[int, list[tuple[int, dict]]],
    max_depth: int,
) -> tuple[list[int], list[dict]] | None:
    queue = deque([(start_id, [start_id], [])])
    visited = {start_id}

    while queue:
        current_id, path_nodes, path_edges = queue.popleft()
        if len(path_edges) >= max_depth:
            continue
        for neighbor_id, edge in undirected_edges.get(current_id, []):
            if neighbor_id in visited:
                continue
            next_nodes = path_nodes + [neighbor_id]
            next_edges = path_edges + [edge]
            if neighbor_id == goal_id:
                return next_nodes, next_edges
            visited.add(neighbor_id)
            queue.append((neighbor_id, next_nodes, next_edges))
    return None


def shortest_paths_between_seeds(
    seed_ids: list[int],
    undirected_edges: dict[int, list[tuple[int, dict]]],
    max_depth: int,
    max_seed_pairs: int,
) -> list[dict]:
    paths = []
    pair_count = 0
    for index, source_id in enumerate(seed_ids):
        for target_id in seed_ids[index + 1 :]:
            if pair_count >= max_seed_pairs:
                return paths
            pair_count += 1
            path = find_shortest_path(source_id, target_id, undirected_edges, max_depth)
            if path is None:
                continue
            path_nodes, path_edges = path
            paths.append({"node_ids": path_nodes, "edges": path_edges})
    return paths


def budgeted_k_hop_expansion(
    seed_ids: list[int],
    task_spec: TaskSpec,
    node_relevance: dict[int, float],
    undirected_edges: dict[int, list[tuple[int, dict]]],
    max_hops: int,
) -> tuple[dict[int, float], dict[tuple[int, int, str], float], dict[tuple[int, int, str], dict]]:
    selected_node_scores = {seed_id: node_relevance.get(seed_id, 1.0) + 5.0 for seed_id in seed_ids}
    selected_edge_scores = {}
    selected_edges = {}
    queue = []

    for seed_id in seed_ids:
        heapq.heappush(queue, (-selected_node_scores[seed_id], seed_id, 0))

    best_seen = dict(selected_node_scores)
    while queue:
        negative_score, current_id, depth = heapq.heappop(queue)
        current_score = -negative_score
        if depth >= max_hops:
            continue

        for neighbor_id, edge in undirected_edges.get(current_id, []):
            rel_score = relation_weight(edge, task_spec)
            neighbor_relevance = node_relevance.get(neighbor_id, 0.0)
            next_score = current_score * 0.58 + rel_score + 0.35 * neighbor_relevance - 0.18 * depth
            key = edge_key(edge)

            if next_score > selected_edge_scores.get(key, -1.0):
                selected_edge_scores[key] = next_score
                selected_edges[key] = edge

            if next_score > best_seen.get(neighbor_id, -1.0):
                best_seen[neighbor_id] = next_score
                selected_node_scores[neighbor_id] = next_score
                heapq.heappush(queue, (-next_score, neighbor_id, depth + 1))

    return selected_node_scores, selected_edge_scores, selected_edges


def edge_to_triple(edge: dict, nodes_by_pruned_id: dict[int, dict]) -> dict:
    source = nodes_by_pruned_id.get(edge["source"], {"object_tag": edge["source"]})
    target = nodes_by_pruned_id.get(edge["target"], {"object_tag": edge["target"]})
    relation = edge.get("normalized_relation") or edge.get("object_relation", "")
    source_label = node_label(source)
    target_label = node_label(target)

    if relation in {"on", "in", "inside"}:
        text = f"{target_label} is {relation} {source_label}"
    elif relation in {"contains", "contain"}:
        text = f"{source_label} contains {target_label}"
    else:
        text = f"{source_label} {relation} {target_label}".strip()

    return {
        "source": edge["source"],
        "target": edge["target"],
        "relation": relation,
        "text": text,
    }


def extract_one_hop_subgraph(
    target_node: dict,
    nodes_by_pruned_id: dict[int, dict],
    incoming_edges: dict[int, list[dict]],
    outgoing_edges: dict[int, list[dict]],
) -> dict:
    target_pruned_id = target_node["pruned_id"]
    upstream_edges = list(incoming_edges.get(target_pruned_id, []))
    downstream_edges = list(outgoing_edges.get(target_pruned_id, []))

    selected_node_ids = {target_pruned_id}
    selected_node_ids.update(edge["source"] for edge in upstream_edges)
    selected_node_ids.update(edge["target"] for edge in downstream_edges)

    selected_nodes = [
        nodes_by_pruned_id[node_id]
        for node_id in sorted(selected_node_ids)
        if node_id in nodes_by_pruned_id
    ]

    return {
        "mode": "one_hop",
        "center_node": target_node,
        "upstream_edges": upstream_edges,
        "downstream_edges": downstream_edges,
        "upstream_nodes": [nodes_by_pruned_id[edge["source"]] for edge in upstream_edges],
        "downstream_nodes": [nodes_by_pruned_id[edge["target"]] for edge in downstream_edges],
        "nodes": selected_nodes,
        "edges": upstream_edges + downstream_edges,
        "triples": [edge_to_triple(edge, nodes_by_pruned_id) for edge in upstream_edges + downstream_edges],
    }


def extract_task_relevant_subgraph(
    task_spec: TaskSpec,
    nodes_by_pruned_id: dict[int, dict],
    edges: list[dict],
    max_hops: int = 2,
    max_nodes: int = 30,
    max_edges: int = 80,
    max_seed_nodes: int = 8,
    min_seed_score: float = 1.0,
    include_shortest_paths: bool = True,
) -> dict:
    _, _, undirected_edges = build_adjacency(edges)
    seed_ids, node_relevance, _ = retrieve_seed_nodes(
        task_spec=task_spec,
        nodes_by_pruned_id=nodes_by_pruned_id,
        max_seed_nodes=max_seed_nodes,
        min_seed_score=min_seed_score,
    )

    if not seed_ids:
        return {
            "mode": "task_relevant",
            "task": task_spec.raw_task,
            "task_spec": asdict(task_spec),
            "seed_nodes": [],
            "nodes": [],
            "edges": [],
            "triples": [],
            "warning": "No seed nodes matched the task.",
        }

    node_scores, edge_scores, selected_edges = budgeted_k_hop_expansion(
        seed_ids=seed_ids,
        task_spec=task_spec,
        node_relevance=node_relevance,
        undirected_edges=undirected_edges,
        max_hops=max_hops,
    )

    candidate_paths = []
    path_node_ids = set(seed_ids)
    if include_shortest_paths and len(seed_ids) > 1:
        candidate_paths = shortest_paths_between_seeds(
            seed_ids=seed_ids[: min(len(seed_ids), 5)],
            undirected_edges=undirected_edges,
            max_depth=max(max_hops + 2, 4),
            max_seed_pairs=8,
        )
        for path in candidate_paths:
            path_node_ids.update(path["node_ids"])
            for edge in path["edges"]:
                key = edge_key(edge)
                selected_edges[key] = edge
                edge_scores[key] = max(edge_scores.get(key, 0.0), 10.0)
            for node_id in path["node_ids"]:
                node_scores[node_id] = max(node_scores.get(node_id, 0.0), 8.0)

    ranked_node_ids = sorted(node_scores, key=lambda node_id: node_scores[node_id], reverse=True)
    pinned_node_ids = [node_id for node_id in ranked_node_ids if node_id in path_node_ids]
    remaining_node_ids = [node_id for node_id in ranked_node_ids if node_id not in path_node_ids]
    selected_node_ids = (pinned_node_ids + remaining_node_ids)[:max_nodes]
    selected_node_id_set = set(selected_node_ids)

    ranked_edge_keys = sorted(edge_scores, key=lambda key: edge_scores[key], reverse=True)
    selected_edge_list = []
    for key in ranked_edge_keys:
        edge = selected_edges[key]
        if edge["source"] not in selected_node_id_set or edge["target"] not in selected_node_id_set:
            continue
        selected_edge_list.append(edge)
        if len(selected_edge_list) >= max_edges:
            break

    selected_nodes = [
        compact_node(nodes_by_pruned_id[node_id], node_scores.get(node_id, 0.0))
        for node_id in selected_node_ids
        if node_id in nodes_by_pruned_id
    ]
    seed_nodes = [
        compact_node(nodes_by_pruned_id[node_id], node_relevance.get(node_id, 0.0))
        for node_id in seed_ids
        if node_id in nodes_by_pruned_id
    ]

    return {
        "mode": "task_relevant",
        "task": task_spec.raw_task,
        "task_spec": asdict(task_spec),
        "seed_nodes": seed_nodes,
        "nodes": selected_nodes,
        "edges": [compact_edge(edge) for edge in selected_edge_list],
        "triples": [edge_to_triple(edge, nodes_by_pruned_id) for edge in selected_edge_list],
    }


def resolve_input_file(path_or_dir: Path, *filenames: str) -> Path:
    if path_or_dir.is_dir():
        candidates = [path_or_dir / filename for filename in filenames]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]
    return path_or_dir


def extract_subgraph(
    scene_graph: Path,
    relations: Path,
    task: str | None = None,
    output: Path | None = None,
    max_hops: int = 2,
    max_nodes: int = 30,
    max_edges: int = 80,
) -> dict:
    scene_graph_path = resolve_input_file(
        scene_graph,
        "scene_graph.json",
        "cfslam_gpt-4_responses",
        "map/scene_map_cfslam_pruned.pkl.gz",
    )
    relations_path = resolve_input_file(
        relations,
        "cfslam_scenegraph_edges.pkl",
        "cfslam_object_relations.json",
    )

    if not scene_graph_path.exists():
        raise FileNotFoundError(f"node metadata file not found: {scene_graph_path}")
    if not relations_path.exists():
        raise FileNotFoundError(f"edge metadata file not found: {relations_path}")

    nodes_by_pruned_id = load_nodes(scene_graph_path)
    edges = load_edges(relations_path, nodes_by_pruned_id)

    if task is None:
        raise ValueError("task is required for extract_subgraph(); use extract_one_hop_subgraph manually for baseline mode")

    task_spec = parse_task_locally(task)
    subgraph = extract_task_relevant_subgraph(
        task_spec=task_spec,
        nodes_by_pruned_id=nodes_by_pruned_id,
        edges=edges,
        max_hops=max_hops,
        max_nodes=max_nodes,
        max_edges=max_edges,
    )

    if output is not None:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(subgraph, f, indent=4, ensure_ascii=False)
    return subgraph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract one-hop or task-relevant subgraphs from ConceptGraphs outputs."
    )
    parser.add_argument("--cachedir", type=Path, default=Path("."))
    parser.add_argument("--scene-graph", type=Path, default=None)
    parser.add_argument("--relations", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)

    parser.add_argument("--mode", choices=("task", "one_hop"), default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--pruned-id", type=int, default=None)
    parser.add_argument("--original-id", type=int, default=None)
    parser.add_argument("--object-tag", type=str, default=None)

    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=30)
    parser.add_argument("--max-edges", type=int, default=80)
    parser.add_argument("--max-seed-nodes", type=int, default=8)
    parser.add_argument("--min-seed-score", type=float, default=1.0)
    parser.add_argument("--no-shortest-paths", action="store_true")

    parser.add_argument("--use-qwen", action="store_true")
    parser.add_argument(
        "--qwen-model-path",
        default=DEFAULT_PLANNING_MODEL_PATH,
        help="Local planning model path (default: Qwen3.5-9B).",
    )
    parser.add_argument("--qwen-conv-mode", default="v0_mmtag")
    parser.add_argument("--qwen-num-gpus", type=int, default=1)

    args = parser.parse_args()
    if args.mode is None:
        args.mode = "task" if args.task else "one_hop"
    if args.mode == "task" and not args.task:
        parser.error("--task is required in task mode")
    if args.mode == "one_hop" and args.pruned_id is None and args.original_id is None and not args.object_tag:
        parser.error("one_hop mode requires --pruned-id, --original-id, or --object-tag")
    return args


def main() -> None:
    args = parse_args()

    scene_graph_path = args.scene_graph or resolve_input_file(
        args.cachedir,
        "scene_graph.json",
        "cfslam_gpt-4_responses",
        "map/scene_map_cfslam_pruned.pkl.gz",
    )
    relations_path = args.relations or resolve_input_file(
        args.cachedir,
        "cfslam_scenegraph_edges.pkl",
        "cfslam_object_relations.json",
    )

    if not scene_graph_path.exists():
        raise FileNotFoundError(f"node metadata file not found: {scene_graph_path}")
    if not relations_path.exists():
        raise FileNotFoundError(f"edge metadata file not found: {relations_path}")

    nodes_by_pruned_id = load_nodes(scene_graph_path)
    edges = load_edges(relations_path, nodes_by_pruned_id)
    incoming_edges, outgoing_edges, _ = build_adjacency(edges)

    if args.mode == "one_hop":
        target_node, selection_metadata = resolve_target_node(args, nodes_by_pruned_id)
        subgraph = extract_one_hop_subgraph(
            target_node=target_node,
            nodes_by_pruned_id=nodes_by_pruned_id,
            incoming_edges=incoming_edges,
            outgoing_edges=outgoing_edges,
        )
        if selection_metadata is not None:
            subgraph["selection"] = selection_metadata
        default_output_name = f"subgraph_pruned_{target_node['pruned_id']}.json"
    else:
        task_spec = parse_task(args)
        subgraph = extract_task_relevant_subgraph(
            task_spec=task_spec,
            nodes_by_pruned_id=nodes_by_pruned_id,
            edges=edges,
            max_hops=args.max_hops,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
            max_seed_nodes=args.max_seed_nodes,
            min_seed_score=args.min_seed_score,
            include_shortest_paths=not args.no_shortest_paths,
        )
        default_output_name = "task_relevant_subgraph.json"

    output_path = args.output or args.cachedir / default_output_name
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(subgraph, f, indent=4, ensure_ascii=False)

    print(f"Saved {subgraph['mode']} subgraph")
    if args.mode == "task":
        print(f"Task parser: {subgraph['task_spec']['parser']}")
        print(f"Seed nodes: {len(subgraph['seed_nodes'])}")
    else:
        selection = subgraph.get("selection", {})
        if selection.get("selection_method") == "object_tag_best_match":
            selected = subgraph.get("center_node", {})
            print(
                "Object-tag selection:",
                f"query='{selection.get('query')}',",
                f"selected pruned_id={selected.get('pruned_id')},",
                f"matches={selection.get('num_matches')}"
            )
    print(f"Nodes: {len(subgraph['nodes'])}")
    print(f"Edges: {len(subgraph['edges'])}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
