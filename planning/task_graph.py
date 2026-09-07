"""
Build a dependency-aware task graph from a task instruction and a grounded subgraph.

The output contains both:
1. a flat list of subtasks with dependency metadata
2. a chain-oriented view where each item in `chains` is a linked-list-like structure
   representing one mostly-serial branch of the task graph.

Qwen local planning is validated before normalization. Invalid output is retried
with corrective feedback and ultimately fails closed; task decomposition never
falls back to a heuristic plan.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict, deque
from copy import deepcopy
from pathlib import Path
from typing import Any

EMAS_ROOT = Path(__file__).resolve().parents[1]
CONCEPTGRAPH_REPO_ROOT = EMAS_ROOT / "memory" / "concept-graphs"
for path in (EMAS_ROOT, CONCEPTGRAPH_REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from planning.model_config import DEFAULT_PLANNING_MODEL_PATH
from action_contracts import (
    ACTION_CONTRACTS,
    PLANNER_ACTIONS,
    normalize_logical_action,
    validate_action_args,
)
from placement_contracts import compatible_receptacles
from planning.scene_goal_compiler import (
    ACTION_ROLE_AFFORDANCES,
    VALID_QUANTIFIERS,
    expand_scene_catalog_task_graph,
    summarize_scene_catalogue,
)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "get",
    "go", "in", "into", "is", "it", "move", "near", "of", "on", "onto", "pick",
    "out", "place", "put", "take", "the", "to", "up", "down", "with",
}

ACTION_KEYWORDS = {
    "pick": {"pick", "grab", "take", "fetch", "collect"},
    "place": {"place", "put", "insert", "store", "set"},
    "navigate": {"go", "navigate", "move", "walk", "reach", "approach"},
    "find": {"find", "locate", "search", "look"},
    "inspect": {"inspect", "check", "observe"},
    "open": {"open"},
    "close": {"close", "shut"},
    "slice": {"slice", "cut"},
    "clean": {"clean", "wash"},
    "drop": {"drop", "release"},
    "push": {"push", "shove"},
    "pull": {"pull", "drag"},
    "move_held": {"reposition", "adjust"},
    "break": {"break", "smash"},
    "cook": {"cook", "heat"},
    "fill": {"fill"},
}


class TaskPlanningError(RuntimeError):
    """Raised when planning cannot produce a validated, executable task graph."""

    code = "task_planning_failed"

    def __init__(
        self,
        message: str,
        *,
        diagnostics: dict[str, Any],
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code or type(self).code)
        self.diagnostics = deepcopy(diagnostics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "diagnostics": deepcopy(self.diagnostics),
        }


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", text.lower())


def unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


def _error_code(message: str) -> str:
    lowered = message.lower()
    for needle, code in (
        ('missing source_selector object_types', 'empty_source_selector'),
        ('missing destination_selector object_types', 'empty_destination_selector'),
        ('missing source_selector', 'missing_source_selector'),
        ('missing destination_selector', 'missing_destination_selector'),
        ('absent from the scene catalogue', 'scene_type_absent'),
        ('does not satisfy', 'affordance_mismatch'),
        ('quantifier', 'invalid_quantifier'),
        ('exactly one object_type', 'invalid_selector_cardinality'),
        ('not supported for action', 'unsupported_selector_role'),
        ('invalid action', 'invalid_action'),
        ('action argument', 'invalid_action_args'),
        ('action_args', 'invalid_action_args'),
        ('root intent action', 'root_action_mismatch'),
        ('missing semantic goal types', 'missing_semantic_goal_types'),
        ('unexpected semantic goal types', 'unexpected_semantic_goal_types'),
        ('selector compiler', 'selector_compiler_rejected'),
    ):
        if needle in lowered:
            return code
    return 'validation_error'


def make_validation_violation(
    message: Any,
    *,
    stage: str,
    task_id: str | None = None,
    code: str | None = None,
    field: str | None = None,
    invalid_values: list[Any] | None = None,
    required_fix: str | None = None,
) -> dict[str, Any]:
    text = str(message).strip()
    resolved_code = str(code or _error_code(text))
    inferred_task_id = task_id
    if inferred_task_id is None:
        match = re.match(r'^([A-Za-z][A-Za-z0-9_.-]*):\s*', text)
        if match:
            inferred_task_id = match.group(1)
        else:
            match = re.search(r'\btask\s+([A-Za-z][A-Za-z0-9_.-]*)\s*:', text)
            if match:
                inferred_task_id = match.group(1)
    inferred_field = field
    lowered = text.lower()
    if inferred_field is None:
        if 'source_selector' in lowered:
            inferred_field = (
                'grounding.source_selector.object_types'
                if 'object_types' in lowered or ' type ' in lowered
                else 'grounding.source_selector'
            )
        elif 'destination_selector' in lowered:
            inferred_field = (
                'grounding.destination_selector.object_types'
                if 'object_types' in lowered or ' type ' in lowered
                else 'grounding.destination_selector'
            )
        elif 'action argument' in lowered or 'action_args' in lowered:
            inferred_field = 'action_args'
        elif 'invalid action' in lowered or 'root intent action' in lowered:
            inferred_field = 'action'
        elif 'dependency' in lowered or 'depends_on' in lowered:
            inferred_field = 'depends_on'
    inferred_invalid_values = deepcopy(invalid_values or [])
    if not inferred_invalid_values:
        quoted_type = re.search(r"(?:selector type|invalid action)\s+['\"]([^'\"]+)['\"]", text)
        if quoted_type:
            inferred_invalid_values = [quoted_type.group(1)]
    fix_by_code = {
        'missing_source_selector': 'Add a complete grounding.source_selector using executable scene object types.',
        'empty_source_selector': 'Replace grounding.source_selector.object_types with one or more executable scene object types.',
        'missing_destination_selector': 'Add a complete grounding.destination_selector naming one receptacle type.',
        'empty_destination_selector': 'Replace grounding.destination_selector.object_types with exactly one receptacle scene type.',
        'scene_type_absent': 'Remove or replace every selector type absent from the scene catalogue.',
        'affordance_mismatch': 'Replace the action or selector type so the selected type satisfies the required affordance.',
        'invalid_quantifier': 'Use the quantifier required by this action contract.',
        'invalid_selector_cardinality': 'Replace the destination selector with exactly one receptacle object type.',
        'unsupported_selector_role': 'Remove the unsupported non-empty selector role from this action.',
        'invalid_action': 'Replace the action with a planner-visible logical action from the action contract.',
        'invalid_action_args': 'Replace action_args with exactly the fields and values declared by the action contract.',
        'root_action_mismatch': 'Return a complete replacement plan containing the requested root interaction action.',
        'missing_semantic_goal_types': 'Replace the source selector with the complete resolved semantic goal type set.',
        'unexpected_semantic_goal_types': 'Remove source types excluded by the resolved semantic goal.',
        'selector_compiler_rejected': 'Replace the rejected selector or action with an executable contract-compliant one.',
    }
    inferred_required_fix = required_fix or fix_by_code.get(resolved_code)
    if resolved_code == 'affordance_mismatch':
        mismatch = re.search(
            r"(source_selector|destination_selector) type ['\"]([^'\"]+)['\"] "
            r"does not satisfy (.+)$",
            text,
        )
        if mismatch:
            selector_name, invalid_type, requirement = mismatch.groups()
            inferred_required_fix = (
                f"Remove {invalid_type!r} from every {selector_name} used by actions "
                f"requiring {requirement}; do not repeat that invalid action/type pair. "
                f"Use only scene catalogue types that satisfy {requirement}, or omit the "
                "interaction when no executable type is justified."
            )
    elif resolved_code == 'scene_type_absent' and inferred_invalid_values:
        inferred_required_fix = (
            f"Remove {inferred_invalid_values!r} from every selector in the replacement "
            "plan; use only exact object_type values present in scene_catalog."
        )
    elif resolved_code in {
        'missing_semantic_goal_types',
        'unexpected_semantic_goal_types',
    }:
        exact_set = re.search(r"required exact types are (\[[^\n]+\])", text)
        if exact_set:
            inferred_required_fix = (
                'Replace the root grounding.source_selector.object_types with exactly '
                f'{exact_set.group(1)}.'
            )
    return {
        'stage': str(stage),
        'task_id': inferred_task_id,
        'code': resolved_code,
        'field': inferred_field,
        'message': text,
        'invalid_values': inferred_invalid_values,
        'required_fix': str(
            inferred_required_fix
            or 'Correct this error in the complete replacement plan.'
        ),
    }


def normalize_validation_violations(
    errors: list[Any], *, stage: str,
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None, str, str | None]] = set()
    for item in errors:
        if isinstance(item, dict):
            violation = make_validation_violation(
                item.get('message') or item,
                stage=str(item.get('stage') or stage),
                task_id=item.get('task_id'),
                code=item.get('code'),
                field=item.get('field'),
                invalid_values=item.get('invalid_values') if isinstance(item.get('invalid_values'), list) else [],
                required_fix=item.get('required_fix'),
            )
        else:
            violation = make_validation_violation(item, stage=stage)
        key = (
            violation['stage'], violation.get('task_id'),
            violation['message'], violation.get('field'),
        )
        if key not in seen:
            seen.add(key)
            violations.append(violation)
    return violations


def correction_summary(violations: list[dict[str, Any]]) -> str:
    lines: list[str] = [
        'FULL REPLACEMENT REQUIRED: fix every violation below; repeating any invalid '
        'action/type pair is another failure.'
    ]
    for violation in violations:
        label = str(violation.get('task_id') or violation.get('stage') or 'plan')
        lines.append(f"{label}: {violation.get('message')}")
        invalid_values = list(violation.get('invalid_values') or [])
        field = str(violation.get('field') or '').strip()
        if invalid_values and field:
            lines.append(
                f"FORBIDDEN REPEAT: do not return {invalid_values!r} in {field} "
                "for the rejected action/selector context."
            )
        required_fix = str(violation.get('required_fix') or '').strip()
        if required_fix:
            lines.append(f"Required fix: {required_fix}")
    lines.append('Return a complete replacement JSON plan; do not reproduce invalid selectors or actions.')
    return '\n'.join(lines)


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


def resolve_subgraph_path(path_or_dir: Path) -> Path:
    if path_or_dir.is_dir():
        candidate = path_or_dir / 'task_relevant_subgraph.json'
        if candidate.exists():
            return candidate
    return path_or_dir


def load_subgraph(subgraph_path: Path) -> dict:
    subgraph_path = resolve_subgraph_path(subgraph_path)
    with open(subgraph_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def bbox_summary(node: dict) -> dict:
    summary = {}
    if 'bbox_center' in node:
        summary['bbox_center'] = node.get('bbox_center')
    if 'bbox_extent' in node:
        summary['bbox_extent'] = node.get('bbox_extent')
    return summary


def summarize_subgraph(subgraph: dict, max_nodes: int = 20, max_triples: int = 24) -> dict:
    nodes = []
    for node in subgraph.get('nodes', [])[:max_nodes]:
        node_summary = {
            'pruned_id': node.get('pruned_id'),
            'original_id': node.get('original_id'),
            'object_tag': node.get('object_tag', ''),
            'caption': node.get('caption', ''),
            'possible_tags': node.get('possible_tags', [])[:8],
        }
        node_summary.update(bbox_summary(node))
        nodes.append(node_summary)

    triples = []
    for triple in subgraph.get('triples', [])[:max_triples]:
        triples.append({
            'source': triple.get('source'),
            'target': triple.get('target'),
            'relation': triple.get('relation'),
            'text': triple.get('text'),
        })

    return {
        'task': subgraph.get('task'),
        'task_spec': subgraph.get('task_spec', {}),
        'seed_nodes': [
            {
                'pruned_id': node.get('pruned_id'),
                'object_tag': node.get('object_tag', ''),
                'caption': node.get('caption', ''),
                **bbox_summary(node),
                'retrieval_score': node.get('retrieval_score'),
            }
            for node in subgraph.get('seed_nodes', [])[:10]
        ],
        'nodes': nodes,
        'triples': triples,
        'candidate_paths': subgraph.get('candidate_paths', [])[:8],
    }


def build_node_lookup(subgraph: dict) -> dict[int, dict]:
    lookup = {}
    for node in subgraph.get('nodes', []):
        pruned_id = node.get('pruned_id')
        if pruned_id is not None:
            lookup[int(pruned_id)] = node
    return lookup


def build_triple_lookup(subgraph: dict) -> list[dict]:
    return list(subgraph.get('triples', []))


def semantic_action_contracts_for_prompt() -> dict[str, dict[str, Any]]:
    """Return compact execution contracts without repeating catalogue type lists."""

    contracts: dict[str, dict[str, Any]] = {}
    for action, contract in ACTION_CONTRACTS.items():
        item: dict[str, Any] = {
            'native_action': str(contract.get('native_action') or ''),
            'roles': {
                role: list(affordances)
                for role, affordances in (contract.get('roles') or {}).items()
            },
        }
        if contract.get('args'):
            item['args'] = deepcopy(contract['args'])
        if contract.get('source_quantifier'):
            item['source_quantifier'] = str(contract['source_quantifier'])
        if contract.get('requires_held_source'):
            item['requires_held_source'] = True
        if contract.get('required_tool'):
            tool = deepcopy(contract['required_tool'])
            tool['any_of_types'] = list(tool.get('any_of_types') or [])
            item['required_tool'] = tool
        contracts[action] = item
    return contracts


def _required_affordances(value: str | tuple[str, ...]) -> tuple[str, ...]:
    return (value,) if isinstance(value, str) else tuple(value)


def _catalogue_types_for_affordances(
    scene_catalog_summary: list[dict[str, Any]],
    affordances: str | tuple[str, ...],
) -> list[str]:
    required = _required_affordances(affordances)
    return [
        str(entry.get('object_type'))
        for entry in scene_catalog_summary
        if isinstance(entry, dict)
        and str(entry.get('object_type') or '').strip()
        and any(
            bool((entry.get('affordances') or {}).get(field))
            for field in required
        )
    ]


def _task_uses_spatial_source(task: str, subgraph: dict[str, Any]) -> bool:
    task_spec = (
        subgraph.get('task_spec')
        if isinstance(subgraph.get('task_spec'), dict)
        else {}
    )
    if task_spec.get('source_receptacles'):
        return True
    return bool(re.search(
        r'\b(?:central|center|centre|countertop|counter\s+top|table|surface|'
        r'from|located|peripheral|leftmost|rightmost)\b',
        str(task or ''),
        flags=re.IGNORECASE,
    ))


def _compact_catalogue_entry(
    entry: dict[str, Any],
    *,
    include_parent_locations: bool,
) -> dict[str, Any]:
    compact: dict[str, Any] = {
        'object_type': str(entry.get('object_type') or ''),
        'count': int(entry.get('count') or 0),
        'affordances': {
            str(field): True
            for field, enabled in (entry.get('affordances') or {}).items()
            if bool(enabled)
        },
    }
    if include_parent_locations:
        parent_locations = []
        for location in entry.get('parent_locations') or []:
            if not isinstance(location, dict):
                continue
            parent = {
                'parent_type': str(location.get('parent_type') or ''),
                'instance_count': int(location.get('instance_count') or 0),
            }
            if location.get('relative_location'):
                parent['relative_location'] = str(location['relative_location'])
            if parent['parent_type']:
                parent_locations.append(parent)
        if parent_locations:
            compact['parent_locations'] = parent_locations
    return compact


def _selector_types_from_plan(candidate_plan: dict[str, Any] | None) -> list[str]:
    selected: list[str] = []
    if not isinstance(candidate_plan, dict):
        return selected
    tasks = candidate_plan.get('flat_tasks') or candidate_plan.get('subtasks') or []
    for task_item in tasks if isinstance(tasks, list) else []:
        if not isinstance(task_item, dict):
            continue
        grounding = (
            task_item.get('grounding')
            if isinstance(task_item.get('grounding'), dict)
            else {}
        )
        for role in ('source', 'destination'):
            selector = grounding.get(f'{role}_selector')
            values = selector.get('object_types') if isinstance(selector, dict) else []
            if isinstance(values, str):
                values = [values]
            if isinstance(values, list):
                selected.extend(str(value) for value in values if str(value).strip())
    return unique_preserve_order(selected)


def build_prompt_scene_catalogue(
    *,
    task: str,
    subgraph: dict[str, Any],
    root_intent: dict[str, Any],
    scene_catalog_summary: list[dict[str, Any]],
    candidate_plan: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Project the full compiler catalogue into task-relevant prompt facts."""

    by_type = {
        str(entry.get('object_type')): entry
        for entry in scene_catalog_summary
        if isinstance(entry, dict) and str(entry.get('object_type') or '').strip()
    }
    roles = (
        root_intent.get('roles')
        if isinstance(root_intent.get('roles'), dict)
        else {}
    )
    relevant = unique_preserve_order([
        *[str(value) for value in roles.get('source') or []],
        *[str(value) for value in roles.get('destination') or []],
        *_selector_types_from_plan(candidate_plan),
    ])
    action = str(root_intent.get('action') or '')
    requirements = ACTION_ROLE_AFFORDANCES.get(action) or {}
    if not roles.get('source') and requirements.get('source'):
        relevant.extend(
            _catalogue_types_for_affordances(
                scene_catalog_summary,
                requirements['source'],
            )
        )

    contract = ACTION_CONTRACTS.get(action) or {}
    required_tool = contract.get('required_tool') or {}
    relevant.extend(
        str(value)
        for value in required_tool.get('any_of_types') or []
        if str(value) in by_type
    )

    if action == 'place' and not roles.get('destination'):
        available_receptacles = set(
            _catalogue_types_for_affordances(scene_catalog_summary, 'receptacle')
        )
        sources = [str(value) for value in roles.get('source') or []]
        compatible: set[str] = set()
        for source_type in sources:
            allowed = compatible_receptacles(source_type)
            if allowed is None:
                compatible.update(available_receptacles)
            else:
                compatible.update(
                    destination
                    for destination in allowed
                    if destination in available_receptacles
                )
        relevant.extend(sorted(compatible or available_receptacles))

    relevant_set = set(unique_preserve_order(relevant))
    include_parents = _task_uses_spatial_source(task, subgraph)
    return [
        _compact_catalogue_entry(
            entry,
            include_parent_locations=include_parents,
        )
        for object_type, entry in by_type.items()
        if object_type in relevant_set
    ]


def placement_compatibility_for_prompt(
    root_intent: dict[str, Any],
    scene_catalog_summary: list[dict[str, Any]],
) -> dict[str, list[str]]:
    if root_intent.get('action') != 'place':
        return {}
    roles = root_intent.get('roles') if isinstance(root_intent.get('roles'), dict) else {}
    available = set(
        _catalogue_types_for_affordances(scene_catalog_summary, 'receptacle')
    )
    result: dict[str, list[str]] = {}
    for source_type in roles.get('source') or []:
        allowed = compatible_receptacles(source_type)
        result[str(source_type)] = sorted(
            available if allowed is None
            else available.intersection(set(allowed))
        )
    return result


def _semantic_source_entities(
    task: str,
    subgraph: dict[str, Any],
    *,
    action: str,
) -> list[str]:
    task_spec = (
        subgraph.get('task_spec')
        if isinstance(subgraph.get('task_spec'), dict)
        else {}
    )
    entities = [
        str(value).strip()
        for value in task_spec.get('target_objects') or []
        if str(value).strip()
    ]
    entities = [
        entity for entity in entities
        if _catalogue_type_match(task, entity) is not None
    ]
    if action == 'place':
        destination_keys = {
            _semantic_key(value)
            for value in task_spec.get('destination_receptacles') or []
            if str(value).strip()
        }
        entities = [
            entity for entity in entities
            if _semantic_key(entity) not in destination_keys
        ]
    return entities or [str(task or '').strip()]


def _is_semantic_entity_grammar_fragment(value: str, action: str) -> bool:
    """Ignore parser fragments that cannot denote an object category."""

    words = [_singular_word(word) for word in _type_words(value)]
    if not words:
        return True
    grammar_words = {
        *STOPWORDS,
        'all', 'any', 'each', 'every', 'either', 'both',
        'if', 'when', 'unless', 'otherwise', 'then',
    }
    action_words = {
        _singular_word(word)
        for keyword in ACTION_KEYWORDS.get(action, set()) | {action}
        for word in _type_words(keyword)
    }
    if action in {'toggle_on', 'toggle_off'}:
        action_words.update({'turn', 'switch', 'toggle'})
        action_words.add('on' if action == 'toggle_on' else 'off')
    return all(word in grammar_words or word in action_words for word in words)


def _unresolved_semantic_entity_text(
    value: str,
    *,
    action: str,
    resolved_types: list[str],
) -> str:
    """Return only entity words not covered by exact scene types or grammar."""

    words = _type_words(value)
    normalized_words = [_singular_word(word) for word in words]
    covered = [False] * len(words)
    for object_type in resolved_types:
        type_words = [_singular_word(word) for word in _type_words(object_type)]
        if not type_words:
            continue
        width = len(type_words)
        for index in range(len(words) - width + 1):
            if normalized_words[index:index + width] == type_words:
                covered[index:index + width] = [True] * width
        compact_type = ''.join(type_words)
        for index, word in enumerate(normalized_words):
            if word == compact_type:
                covered[index] = True
    unresolved_words = [
        word
        for index, word in enumerate(words)
        if not covered[index]
        and not _is_semantic_entity_grammar_fragment(word, action)
    ]
    return ' '.join(unresolved_words)


def _legacy_semantic_goal_resolution_request(
    *,
    task: str,
    subgraph: dict[str, Any],
    root_intent: dict[str, Any],
    scene_catalog_summary: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Build a resolver request for any unresolved executable source entity."""

    if not scene_catalog_summary:
        return None
    action = str(root_intent.get('action') or '')
    source_affordances = (ACTION_ROLE_AFFORDANCES.get(action) or {}).get('source')
    if not source_affordances:
        return None
    entities = _semantic_source_entities(task, subgraph, action=action)
    roles = (
        root_intent.get('roles')
        if isinstance(root_intent.get('roles'), dict)
        else {}
    )
    destination_types = [str(value) for value in roles.get('destination') or []]
    unresolved = []
    for entity in entities:
        matched_types = _match_catalogue_types(
            entity,
            source_affordances,
            scene_catalog_summary,
        )
        unresolved_text = _unresolved_semantic_entity_text(
            entity,
            action=action,
            resolved_types=[*matched_types, *destination_types],
        )
        if unresolved_text:
            unresolved.append(unresolved_text)
    if not unresolved:
        return None
    exact_types = list(
        roles.get('source') or []
    )
    candidates = [
        object_type
        for object_type in _catalogue_types_for_affordances(
            scene_catalog_summary,
            source_affordances,
        )
        if object_type not in set(exact_types)
    ]
    request: dict[str, Any] = {
        'request': 'resolve_semantic_goal',
        'instruction': task,
        'semantic_phrase': ' and '.join(unresolved),
        'quantifier': str(root_intent.get('quantifier') or 'one'),
        'action': action,
        'role': 'source',
        'required_any_affordance': list(source_affordances),
        'candidate_types': candidates,
        'already_resolved_types': exact_types,
    }
    if _task_uses_spatial_source(task, subgraph):
        candidate_set = set(candidates)
        request['candidate_scene_facts'] = [
            _compact_catalogue_entry(entry, include_parent_locations=True)
            for entry in scene_catalog_summary
            if str(entry.get('object_type') or '') in candidate_set
        ]
    return request


def _legacy_validate_semantic_goal_resolution(
    raw_result: Any,
    candidate_types: list[str],
) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(raw_result, dict):
        return {'protocol_status': 'invalid', 'errors': ['response is not a JSON object']}
    status = str(raw_result.get('status') or '').strip().lower()
    if status not in {'resolved', 'no_match'}:
        errors.append('status must be resolved or no_match')
    included = raw_result.get('included_types')
    excluded = raw_result.get('excluded_types')
    if not isinstance(included, list) or any(not isinstance(value, str) for value in included):
        errors.append('included_types must be a list of strings')
        included = []
    if not isinstance(excluded, list) or any(not isinstance(value, str) for value in excluded):
        errors.append('excluded_types must be a list of strings')
        excluded = []
    included = [str(value).strip() for value in included if str(value).strip()]
    excluded = [str(value).strip() for value in excluded if str(value).strip()]
    if len(included) != len(set(included)):
        errors.append('included_types contains duplicates')
    if len(excluded) != len(set(excluded)):
        errors.append('excluded_types contains duplicates')
    overlap = sorted(set(included).intersection(excluded))
    if overlap:
        errors.append(f'included_types and excluded_types overlap: {overlap}')
    candidates = set(candidate_types)
    unknown = sorted((set(included) | set(excluded)) - candidates)
    if unknown:
        errors.append(f'resolution contains unknown candidate types: {unknown}')
    missing = sorted(candidates - (set(included) | set(excluded)))
    if missing:
        errors.append(f'resolution does not classify candidate types: {missing}')
    if status == 'resolved' and not included:
        errors.append('resolved status requires at least one included type')
    if status == 'no_match' and included:
        errors.append('no_match status requires an empty included_types list')
    if errors:
        return {'protocol_status': 'invalid', 'errors': errors}
    return {
        'protocol_status': 'valid',
        'status': status,
        'included_types': included,
        'excluded_types': excluded,
        'summary': str(raw_result.get('summary') or '').strip(),
    }


def _legacy_call_local_qwen_for_semantic_goal_resolution(
    *,
    request: dict[str, Any],
    model_path: str | None,
    conv_mode: str,
    num_gpus: int,
    qwen_chat: Any,
    max_new_tokens: int,
    correction_errors: list[str] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Classify every executable scene candidate in an isolated Qwen session."""

    if diagnostics is not None:
        diagnostics.update({
            'attempted': True,
            'status': 'started',
            'stage': 'initialization',
            'error_type': None,
            'error_message': None,
            'raw_response_preview': None,
            'response_characters': 0,
            'json_parse_status': 'not_attempted',
            'max_new_tokens': max_new_tokens,
        })
    system_prompt = (
        'You are an independent semantic goal resolver for an embodied household '
        'robot. Classify every supplied candidate type against the public instruction '
        'and semantic phrase. Use broad ordinary household-task meaning, including '
        'food-preparation, serving, and eating implements when the category warrants '
        'it, but do not use benchmark answers or invent types. Return JSON only. '
        'Every candidate must appear exactly once in included_types or excluded_types.'
    )
    payload = deepcopy(request)
    payload['output_schema'] = {
        'status': 'resolved or no_match',
        'included_types': ['exact candidate type strings matching the semantic phrase'],
        'excluded_types': ['every remaining exact candidate type string'],
        'summary': 'one short sentence',
    }
    payload['constraints'] = [
        'Copy type strings exactly, including case.',
        'Do not omit, duplicate, or invent candidate types.',
        'included_types and excluded_types must be disjoint and partition candidate_types.',
        'Use no_match only when none of the candidate types satisfy the phrase.',
    ]
    if correction_errors:
        payload['protocol_correction'] = {
            'errors': list(correction_errors),
            'instruction': (
                'Return a fresh complete JSON classification that fixes every protocol '
                'error. Do not reproduce the prior response.'
            ),
        }

    chat = qwen_chat
    owns_chat = chat is None
    close_vlm_chat = None
    original_max_new_tokens: Any = None
    overrides_max_new_tokens = False
    if owns_chat:
        try:
            from conceptgraph.vlm import build_vlm_chat, close_vlm_chat
            chat = build_vlm_chat(
                backend='qwen',
                model_path=model_path,
                conv_mode=conv_mode,
                num_gpus=num_gpus,
            )
        except Exception as exc:
            if diagnostics is not None:
                diagnostics.update({
                    'status': 'failed',
                    'stage': 'initialization',
                    'error_type': type(exc).__name__,
                    'error_message': str(exc),
                })
            return None
    try:
        if hasattr(chat, 'max_new_tokens'):
            original_max_new_tokens = chat.max_new_tokens
            chat.max_new_tokens = max_new_tokens
            overrides_max_new_tokens = True
        if hasattr(chat, 'reset'):
            chat.reset()
        if hasattr(chat, 'messages'):
            chat.messages = [{'role': 'system', 'content': system_prompt}]
        response_text = chat(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as exc:
        if diagnostics is not None:
            diagnostics.update({
                'status': 'failed',
                'stage': 'model_call',
                'error_type': type(exc).__name__,
                'error_message': str(exc),
            })
        return None
    finally:
        if overrides_max_new_tokens:
            chat.max_new_tokens = original_max_new_tokens
        if owns_chat and close_vlm_chat is not None:
            close_vlm_chat(chat)
    response_string = str(response_text or '')
    if diagnostics is not None:
        diagnostics.update({
            'stage': 'json_parse',
            'raw_response_preview': (
                response_string
                if len(response_string) <= 8000
                else response_string[:8000] + '...'
            ),
            'response_characters': len(response_string),
        })
    parsed = parse_json_from_text(response_string)
    if diagnostics is not None:
        diagnostics.update({
            'status': 'success' if parsed is not None else 'failed',
            'stage': 'complete' if parsed is not None else 'json_parse',
            'json_parse_status': 'success' if parsed is not None else 'invalid_json',
        })
    return parsed


# Backward-compatible request builder for diagnostics/tests. The production
# planning path below always uses action_intent_resolution_request followed by
# role_intent_resolution_request.
semantic_goal_resolution_request = _legacy_semantic_goal_resolution_request
def action_intent_resolution_request(
    *,
    task: str,
    semantic_correction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the mandatory first-stage request that selects one root action."""

    allowed_actions = [
        {
            'action': action,
            'native_action': str(contract.get('native_action') or ''),
            'roles': {
                role: list(affordances)
                for role, affordances in (contract.get('roles') or {}).items()
            },
        }
        for action, contract in ACTION_CONTRACTS.items()
    ]
    request: dict[str, Any] = {
        'request': 'resolve_task_intent',
        'stage': 'action',
        'instruction': str(task or ''),
        'allowed_actions': allowed_actions,
        'output_schema': {
            'status': 'resolved or no_match',
            'action': 'exact action string from allowed_actions, or null for no_match',
            'summary': 'one short sentence',
        },
        'constraints': [
            'Choose exactly one user-level interaction action.',
            'Copy the action string exactly from allowed_actions.',
            'When the instruction implies multiple execution steps, choose the final user-requested goal interaction as the root action; the planner will add prerequisite helper steps.',
            'An instruction that puts, places, inserts, or sets an object in, into, on, or onto a destination selects place even when pickup, opening, or navigation helpers are implied.',
            'Navigation, finding, inspection, opening containers, and tool use are planner helpers unless explicitly requested as the goal.',
            'Use no_match only when no allowed interaction action expresses the instruction.',
            'Return JSON only.',
        ],
    }
    if semantic_correction:
        request['semantic_correction'] = deepcopy(semantic_correction)
    return request


def validate_action_intent_resolution(raw_result: Any) -> dict[str, Any]:
    """Validate the first-stage action choice without interpreting task text."""

    if not isinstance(raw_result, dict):
        return {'protocol_status': 'invalid', 'errors': ['response is not a JSON object']}
    errors: list[str] = []
    status = str(raw_result.get('status') or '').strip().lower()
    if status not in {'resolved', 'no_match'}:
        errors.append('status must be resolved or no_match')
    action = str(raw_result.get('action') or '').strip()
    if status == 'resolved' and action not in ACTION_CONTRACTS:
        errors.append(
            'resolved status requires action to be an exact allowed interaction action'
        )
    if status == 'no_match' and action:
        errors.append('no_match status requires action to be null or empty')
    if errors:
        return {'protocol_status': 'invalid', 'errors': errors}
    return {
        'protocol_status': 'valid',
        'status': status,
        'action': action or None,
        'summary': str(raw_result.get('summary') or '').strip(),
    }


def _catalogue_role_rows(
    scene_catalog_summary: list[dict[str, Any]],
    affordances: str | tuple[str, ...],
) -> list[dict[str, Any]]:
    required = _required_affordances(affordances)
    entries = sorted(
        (
            entry for entry in scene_catalog_summary
            if isinstance(entry, dict)
            and str(entry.get('object_type') or '').strip()
            and any(
                bool((entry.get('affordances') or {}).get(field))
                for field in required
            )
        ),
        key=lambda entry: str(entry.get('object_type') or '').casefold(),
    )
    rows = []
    for row_id, entry in enumerate(entries):
        compact = _compact_catalogue_entry(entry, include_parent_locations=True)
        compact['parent_locations'] = list(compact.get('parent_locations') or [])
        rows.append({'row_id': row_id, **compact})
    return rows


def _subgraph_role_rows(subgraph: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for node in subgraph.get('nodes') or []:
        if not isinstance(node, dict) or node.get('pruned_id') is None:
            continue
        possible_tags = node.get('possible_tags') or []
        if isinstance(possible_tags, str):
            possible_tags = [possible_tags]
        rows.append({
            'node_id': node.get('pruned_id'),
            'object_tag': str(node.get('object_tag') or ''),
            'caption': str(node.get('caption') or ''),
            'possible_tags': [
                str(value) for value in possible_tags[:8] if str(value).strip()
            ],
            'capabilities_verified': False,
        })
    rows.sort(key=lambda row: (str(row['node_id']), row['object_tag'].casefold()))
    return [{'row_id': row_id, **row} for row_id, row in enumerate(rows)]


def role_intent_resolution_request(
    *,
    task: str,
    subgraph: dict[str, Any],
    action: str,
    scene_catalog_summary: list[dict[str, Any]],
    semantic_correction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build action-specific, role-independent candidate tables."""

    contract = ACTION_CONTRACTS[action]
    requirements = contract.get('roles') or {}
    grounding_mode = 'catalogue' if scene_catalog_summary else 'subgraph'
    node_rows = _subgraph_role_rows(subgraph) if grounding_mode == 'subgraph' else []
    role_tables: dict[str, list[dict[str, Any]]] = {}
    for role, affordances in requirements.items():
        role_tables[role] = (
            _catalogue_role_rows(scene_catalog_summary, tuple(affordances))
            if grounding_mode == 'catalogue'
            else deepcopy(node_rows)
        )
    semantic_arg_schema = {
        name: deepcopy(rule)
        for name, rule in (contract.get('args') or {}).items()
        if bool(rule.get('semantic'))
    }
    role_schema = {
        role: {
            'status': 'specified' if role == 'source' else 'specified or unspecified',
            'quantifier': 'one or all' if role == 'source' else 'one',
            'reviewed_row_count': len(rows),
            'included_row_ids': [],
        }
        for role, rows in role_tables.items()
    }
    request: dict[str, Any] = {
        'request': 'resolve_task_intent',
        'stage': 'roles',
        'instruction': str(task or ''),
        'selected_action': action,
        'grounding_mode': grounding_mode,
        'selection_encoding': 'included_row_ids_v2',
        'role_requirements': {
            role: list(affordances)
            for role, affordances in requirements.items()
        },
        'role_tables': role_tables,
        'semantic_arg_schema': semantic_arg_schema,
        'output_schema': {
            'status': 'resolved or no_match',
            'roles': role_schema,
            'semantic_args': {
                name: f'value satisfying {rule}'
                for name, rule in semantic_arg_schema.items()
            },
            'summary': 'one short sentence',
        },
        'constraints': [
            'Review every row in every role table before selecting rows.',
            'Copy the exact table length into reviewed_row_count for each role.',
            'included_row_ids contains only integer row_id values copied from that role table; omitted rows are excluded.',
            'Use [] when no row is selected; never output type names or node ids instead of row_id values.',
            'Do not output a classifications array in this protocol version.',
            'Do not duplicate row ids and do not return ids outside the corresponding role table.',
            'Source quantifier all means every current instance of each included row; one means one instance of each included semantic row and the planner will split multiple rows into atomic tasks.',
            'Source is always specified and must include every source category requested by the instruction.',
            'Destination specified includes exactly one row; use unspecified with [] only when the instruction intentionally leaves per-source destinations abstract.',
            'Source and destination are independent roles, so the same category may be included in both.',
            'For broad category words, include only clear ordinary members of that category, not nearby containers or appliances.',
            'For an abstract destination, select the single most conventional compatible purpose-specific receptacle for the chosen sources.',
            'Return JSON only.',
        ],
    }
    forced_source_quantifier = contract.get('source_quantifier')
    if forced_source_quantifier:
        request['forced_source_quantifier'] = str(forced_source_quantifier)
    if semantic_correction:
        request['semantic_correction'] = deepcopy(semantic_correction)
    return request


def _validate_semantic_args(
    raw_args: Any,
    schema: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    if raw_args is None:
        raw_args = {}
    if not isinstance(raw_args, dict):
        return {}, ['semantic_args must be a JSON object']
    errors: list[str] = []
    unknown = sorted(str(key) for key in raw_args if key not in schema)
    if unknown:
        errors.append(f'semantic_args contains unsupported fields: {unknown}')
    normalized: dict[str, Any] = {}
    for name, rule in schema.items():
        if name not in raw_args:
            errors.append(f'semantic_args is missing required field {name!r}')
            continue
        value = raw_args[name]
        if rule.get('type') == 'enum':
            normalized_value = str(value or '').strip().lower()
            allowed = set(rule.get('values') or ())
            if normalized_value not in allowed:
                errors.append(
                    f'semantic_args.{name} must be one of {sorted(allowed)}'
                )
            else:
                normalized[name] = normalized_value
        else:
            errors.append(f'unsupported semantic argument schema for {name!r}')
    return normalized, errors


def _normalize_role_row_selection(
    raw_role: dict[str, Any],
    rows: list[dict[str, Any]],
    role: str,
) -> tuple[list[int], list[str], list[str]]:
    """Normalize sparse row-id selection while accepting the legacy full array."""

    errors: list[str] = []
    table_row_ids = [row.get('row_id') for row in rows]
    uses_row_ids = 'included_row_ids' in raw_role
    uses_legacy_classifications = 'classifications' in raw_role
    if uses_row_ids and uses_legacy_classifications:
        errors.append(
            f'roles.{role} must use included_row_ids or classifications, not both'
        )

    if uses_row_ids:
        reviewed_row_count = raw_role.get('reviewed_row_count')
        if (
            isinstance(reviewed_row_count, bool)
            or not isinstance(reviewed_row_count, int)
            or reviewed_row_count != len(rows)
        ):
            errors.append(
                f'roles.{role}.reviewed_row_count must be {len(rows)}'
            )
        raw_row_ids = raw_role.get('included_row_ids')
        if not isinstance(raw_row_ids, list):
            raw_row_ids = []
            errors.append(f'roles.{role}.included_row_ids must be a list')
        invalid_row_ids = [
            value
            for value in raw_row_ids
            if isinstance(value, bool) or not isinstance(value, int)
        ]
        if invalid_row_ids:
            errors.append(
                f'roles.{role}.included_row_ids must contain only integer row ids'
            )
        integer_row_ids = [
            value
            for value in raw_row_ids
            if isinstance(value, int) and not isinstance(value, bool)
        ]
        if len(integer_row_ids) != len(set(integer_row_ids)):
            errors.append(
                f'roles.{role}.included_row_ids must not contain duplicates'
            )
        unknown_row_ids = sorted({
            value for value in integer_row_ids if value not in table_row_ids
        })
        if unknown_row_ids:
            errors.append(
                f'roles.{role}.included_row_ids contains unknown row ids: '
                f'{unknown_row_ids}'
            )
        selected_row_ids = set(integer_row_ids).intersection(table_row_ids)
        included_row_ids = [
            row_id for row_id in table_row_ids if row_id in selected_row_ids
        ]
        normalized_classifications = [
            'included' if row_id in selected_row_ids else 'excluded'
            for row_id in table_row_ids
        ]
        return included_row_ids, normalized_classifications, errors

    classifications = raw_role.get('classifications')
    if not isinstance(classifications, list):
        classifications = []
        errors.append(f'roles.{role}.classifications must be a list')
    normalized_classifications = [
        str(value or '').strip().lower()
        for value in classifications
    ]
    if len(normalized_classifications) != len(rows):
        errors.append(
            f'roles.{role}.classifications length must be {len(rows)}, '
            f'got {len(normalized_classifications)}'
        )
    invalid_values = sorted({
        value for value in normalized_classifications
        if value not in {'included', 'excluded'}
    })
    if invalid_values:
        errors.append(
            f'roles.{role}.classifications contains invalid values: '
            f'{invalid_values}'
        )
    included_row_ids = [
        table_row_ids[index]
        for index, value in enumerate(normalized_classifications)
        if value == 'included' and index < len(rows)
    ]
    return included_row_ids, normalized_classifications, errors


def validate_semantic_goal_resolution(
    raw_result: Any,
    request_or_candidates: dict[str, Any] | list[str],
) -> dict[str, Any]:
    """Validate sparse row-id role selection or the legacy full classification."""

    if not isinstance(request_or_candidates, dict):
        return _legacy_validate_semantic_goal_resolution(
            raw_result,
            list(request_or_candidates),
        )
    request = request_or_candidates
    if not isinstance(raw_result, dict):
        return {'protocol_status': 'invalid', 'errors': ['response is not a JSON object']}
    errors: list[str] = []
    status = str(raw_result.get('status') or '').strip().lower()
    if status not in {'resolved', 'no_match'}:
        errors.append('status must be resolved or no_match')
    raw_roles = raw_result.get('roles')
    if not isinstance(raw_roles, dict):
        raw_roles = {}
        errors.append('roles must be a JSON object')
    expected_roles = set((request.get('role_tables') or {}).keys())
    unknown_roles = sorted(set(raw_roles) - expected_roles)
    missing_roles = sorted(expected_roles - set(raw_roles))
    if unknown_roles:
        errors.append(f'roles contains unsupported roles: {unknown_roles}')
    if missing_roles:
        errors.append(f'roles is missing required roles: {missing_roles}')

    normalized_roles: dict[str, dict[str, Any]] = {}
    forced_source_quantifier = str(
        request.get('forced_source_quantifier') or ''
    ).strip()
    for role in sorted(expected_roles):
        rows = list((request.get('role_tables') or {}).get(role) or [])
        raw_role = raw_roles.get(role)
        if not isinstance(raw_role, dict):
            continue
        role_status = str(raw_role.get('status') or '').strip().lower()
        allowed_statuses = (
            {'specified'} if role == 'source' else {'specified', 'unspecified'}
        )
        if role_status not in allowed_statuses:
            errors.append(
                f'roles.{role}.status must be one of {sorted(allowed_statuses)}'
            )
        quantifier = str(raw_role.get('quantifier') or '').strip().lower()
        allowed_quantifiers = {'one', 'all'} if role == 'source' else {'one'}
        if quantifier not in allowed_quantifiers:
            errors.append(
                f'roles.{role}.quantifier must be one of {sorted(allowed_quantifiers)}'
            )
        if (
            role == 'source'
            and forced_source_quantifier
            and quantifier != forced_source_quantifier
        ):
            errors.append(
                f'roles.source.quantifier must be {forced_source_quantifier!r} for this action'
            )
        (
            included_row_ids,
            normalized_classifications,
            selection_errors,
        ) = _normalize_role_row_selection(raw_role, rows, role)
        errors.extend(selection_errors)
        if status == 'resolved' and role == 'source' and not included_row_ids:
            errors.append('resolved status requires at least one included source row')
        if role == 'destination':
            if role_status == 'specified' and len(included_row_ids) != 1:
                errors.append('specified destination requires exactly one included row')
            if role_status == 'unspecified' and included_row_ids:
                errors.append(
                    'unspecified destination requires every row to be excluded'
                )
        if status == 'no_match' and role == 'source' and included_row_ids:
            errors.append(
                'no_match status requires every source row to be excluded'
            )
        normalized_roles[role] = {
            'status': role_status,
            'quantifier': quantifier,
            'classifications': normalized_classifications,
            'included_row_ids': included_row_ids,
        }

    semantic_args, semantic_arg_errors = _validate_semantic_args(
        raw_result.get('semantic_args'),
        request.get('semantic_arg_schema') or {},
    )
    errors.extend(semantic_arg_errors)
    if errors:
        return {'protocol_status': 'invalid', 'errors': errors}
    return {
        'protocol_status': 'valid',
        'status': status,
        'roles': normalized_roles,
        'semantic_args': semantic_args,
        'summary': str(raw_result.get('summary') or '').strip(),
    }


def call_local_qwen_for_semantic_goal_resolution(
    *,
    request: dict[str, Any],
    model_path: str | None,
    conv_mode: str,
    num_gpus: int,
    qwen_chat: Any,
    max_new_tokens: int,
    correction_errors: list[str] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Run either isolated intent stage and parse a JSON-only response."""

    stage = str(request.get('stage') or '')
    if diagnostics is not None:
        diagnostics.update({
            'attempted': True,
            'status': 'started',
            'stage': stage or 'initialization',
            'error_type': None,
            'error_message': None,
            'raw_response_preview': None,
            'response_characters': 0,
            'json_parse_status': 'not_attempted',
            'max_new_tokens': max_new_tokens,
        })
    system_prompt = (
        'You are an independent intent resolver for an embodied household robot. '
        'Read the complete public instruction. First choose only from the supplied '
        'interaction-action whitelist; then, in the roles stage, review every '
        'supplied row and select exact row_id values for each independent action role. '
        'Never invent actions, types, '
        'or nodes. Do not use hidden benchmark answers. Return JSON only.'
    )
    payload = deepcopy(request)
    if correction_errors:
        payload['protocol_correction'] = {
            'errors': list(correction_errors),
            'instruction': (
                'Return a fresh complete JSON object that fixes every protocol error.'
            ),
        }

    chat = qwen_chat
    owns_chat = chat is None
    close_chat = None
    original_max_new_tokens: Any = None
    overrides_max_new_tokens = False
    if owns_chat:
        try:
            from conceptgraph.vlm import build_vlm_chat, close_vlm_chat
            close_chat = close_vlm_chat
            chat = build_vlm_chat(
                backend='qwen',
                model_path=model_path,
                conv_mode=conv_mode,
                num_gpus=num_gpus,
            )
        except Exception as exc:
            if diagnostics is not None:
                diagnostics.update({
                    'status': 'failed',
                    'stage': 'initialization',
                    'error_type': type(exc).__name__,
                    'error_message': str(exc),
                })
            return None
    try:
        if hasattr(chat, 'max_new_tokens'):
            original_max_new_tokens = chat.max_new_tokens
            chat.max_new_tokens = max_new_tokens
            overrides_max_new_tokens = True
        if hasattr(chat, 'reset'):
            chat.reset()
        if hasattr(chat, 'messages'):
            chat.messages = [{'role': 'system', 'content': system_prompt}]
        response_text = chat(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as exc:
        if diagnostics is not None:
            diagnostics.update({
                'status': 'failed',
                'stage': 'model_call',
                'error_type': type(exc).__name__,
                'error_message': str(exc),
            })
        return None
    finally:
        if overrides_max_new_tokens:
            chat.max_new_tokens = original_max_new_tokens
        if owns_chat and close_chat is not None:
            close_chat(chat)
    response_string = str(response_text or '')
    if diagnostics is not None:
        diagnostics.update({
            'stage': 'json_parse',
            'raw_response_preview': (
                response_string
                if len(response_string) <= 8000
                else response_string[:8000] + '...'
            ),
            'response_characters': len(response_string),
        })
    parsed = parse_json_from_text(response_string)
    if diagnostics is not None:
        diagnostics.update({
            'status': 'success' if parsed is not None else 'failed',
            'stage': 'complete' if parsed is not None else 'json_parse',
            'json_parse_status': 'success' if parsed is not None else 'invalid_json',
        })
    return parsed


def call_local_qwen_for_task_graph(
    task: str,
    subgraph: dict,
    model_path: str | None,
    conv_mode: str,
    num_gpus: int,
    qwen_chat: Any | None = None,
    diagnostics: dict[str, Any] | None = None,
    max_new_tokens: int = 2048,
    agent_context: dict[str, Any] | None = None,
    scene_catalog_summary: list[dict[str, Any]] | None = None,
    resolved_semantic_goal: dict[str, Any] | None = None,
    placement_compatibility: dict[str, list[str]] | None = None,
    correction: dict[str, Any] | None = None,
    planning_mode: str = 'initial',
    execution_context: dict[str, Any] | None = None,
    runtime_constraints: dict[str, Any] | None = None,
) -> dict | None:
    if diagnostics is not None:
        diagnostics.update({
            'attempted': True,
            'status': 'started',
            'stage': 'initialization',
            'error_type': None,
            'error_message': None,
            'raw_response_preview': None,
            'response_characters': 0,
            'json_parse_status': 'not_attempted',
            'max_new_tokens': max_new_tokens,
        })
    system_prompt = (
        "You are a multi-agent robot task planner. Given a user task, a grounded scene "
        "subgraph, and the available agent state, "
        "decompose the task into small executable subtasks and explicit dependencies. "
        "Return only valid JSON. Do not write analysis or discuss the constraints. "
        "Keep reasoning_summary to at most 20 words. Use grounded node ids from the "
        "subgraph whenever possible and emit only required atomic interaction tasks."
    )

    user_payload = {
        'planning_mode': planning_mode,
        'task': task,
        'subgraph_summary': summarize_subgraph(subgraph),
        'agent_context': deepcopy(agent_context) if isinstance(agent_context, dict) else {},
        'scene_catalog': deepcopy(scene_catalog_summary) if isinstance(scene_catalog_summary, list) else [],
        'resolved_semantic_goal': deepcopy(resolved_semantic_goal or {}),
        'placement_compatibility': deepcopy(placement_compatibility or {}),
        'execution_action_contracts': semantic_action_contracts_for_prompt(),
        'output_schema': {
            'reasoning_summary': 'one sentence of at most 20 words; no deliberation',
            'subtasks': [
                {
                    'id': 'T1',
                    'name': 'short action name',
                    'description': 'what the robot should accomplish',
                    'action': 'pick/place/navigate/find/inspect/open/close/toggle_on/toggle_off/clean/slice/drop/push/pull/move_held/break/cook/fill/other',
                    'action_args': {
                        'moveMagnitude': 'required finite number in (0,1000] for push/pull',
                        'right': 'move_held relative displacement in [-0.5,0.5]',
                        'up': 'move_held relative displacement in [-0.5,0.5]',
                        'ahead': 'move_held relative displacement in [-0.5,0.5]',
                        'fillLiquid': 'water/coffee/wine for fill',
                    },
                    'grounding': {
                        'node_ids': [1, 2],
                        'source_node_ids': [1],
                        'destination_node_ids': [2],
                        'object_tags': ['chair'],
                        'source_object_tags': ['chair'],
                        'destination_object_tags': [],
                        'source_selector': {
                            'quantifier': 'one/all',
                            'object_types': ['exact object_type values from scene_catalog'],
                        },
                        'destination_selector': {
                            'quantifier': 'one',
                            'object_types': ['exact receptacle object_type from scene_catalog'],
                        },
                        'relation_texts': ['chair is near table'],
                        'status': 'grounded/partial/unresolved',
                        'missing_reason': 'why grounding is unresolved, if applicable',
                        'recovery': 'none/search_visible_scene/explore_and_update_scene_graph',
                    },
                    'depends_on': ['T0'],
                    'termination_check': 'how to know this subtask is finished',
                }
            ],
        },
        'constraints': [
            'Return JSON only.',
            'Every subtask id must be unique.',
            'Use depends_on to express prerequisites.',
            'Independent subtasks should have empty depends_on.',
            'The scene graph may be incomplete. Missing nodes mean unresolved grounding, not task failure.',
            'When a required object is absent from the subgraph but its exact type is present in scene_catalog, put that type in the required selector and do not add a find subtask; runtime binds and navigates to the instance.',
            'Only add a find/inspect/search subtask when no valid scene selector can be constructed for a required object.',
            'Use grounded node ids when available; do not fabricate node ids.',
            'Each place subtask must contain one source role and one destination receptacle; an explicit quantifier=all selector may represent multiple source instances.',
            'Each place destination_selector.object_types list must contain exactly one receptacle type; split tasks when different sources require different destinations.',
            'When the instruction asks for appropriate positions, split heterogeneous source categories into separate place tasks whenever one destination is not semantically suitable for every listed source type.',
            'For an explicit all/every/each goal, emit one place subtask with source_selector.quantifier=all and list every matching concrete scene object type. Otherwise emit one atomic place subtask per semantic source object.',
            'For every explicit all/every/each interaction goal, emit one selector subtask before runtime instance expansion. Do not repeat one model task per scene instance.',
            'When grounding_mode=subgraph, omit source_selector and destination_selector entirely; resolved role node ids are authoritative.',
            'A place subtask is an end-to-end executor interaction that acquires its source and deposits it; do not emit a separate pick subtask solely as a prerequisite for place.',
            'When grounding_mode=subgraph, copy every resolved role node into source_node_ids or destination_node_ids and split quantifier-one multi-source goals into one atomic task per source node.',
            'When the user explicitly names a place destination, preserve that destination exactly; ordinary household suitability applies only when the user leaves the destination implicit.',
            'Each find, inspect, and slice subtask must reference exactly one semantic target object.',
            'For slicing goals, emit one action="slice" subtask per food object. Do not emit separate find, navigate, or pickup subtasks for a knife; the executor deterministically acquires a cutting tool and opens its containing receptacle.',
            'Do not schedule multiple pickup subtasks before their corresponding placements when that would exceed an agent inventory capacity.',
            'Every open, close, toggle, clean, slice, pick, and place task whose type exists in scene_catalog must include its required source_selector; place must also include destination_selector.',
            'Use source_selector and destination_selector only for object interaction tasks.',
            'Set selector quantifier=all only when the instruction explicitly requests all/every/each matching source instance.',
            'Selector object_types must be exact object_type values present in scene_catalog; never emit a semantic group label when concrete scene types can be listed.',
            'For every interaction role, selector object_types must be present in scene_catalog and satisfy execution_action_contracts[action].roles.',
            'A visual subgraph caption never makes a type executable when it is absent from scene_catalog or lacks the required affordance.',
            'Use scene_catalog parent_locations as live scene facts when the instruction identifies a source surface or spatial region; prefer these facts over uncertain visual captions.',
            'When the instruction says central, select only types whose relevant parent_locations entry is marked relative_location=central; do not include types located only on peripheral instances.',
            'parent_locations describe current simulator containment and are not object IDs; do not copy positions into output selectors.',
            'When resolved_semantic_goal.required_source_types is non-empty, copy that complete exact set into the root source selector; do not omit or add types.',
            'Do not emit object IDs; deterministic runtime validation binds IDs after planning.',
            'For push and pull, emit action_args.moveMagnitude; use 200.0 when the instruction does not specify force.',
            'For move_held, emit only action_args.right/up/ahead and make at least one axis non-zero.',
            'For fill, emit action_args.fillLiquid using water, coffee, or wine.',
            'Drop and move_held require quantifier=one source_selector and either a matching prior pick dependency or an already-held matching object.',
            'Never emit forceAction, Teleport, TeleportFull, GetReachablePositions, SetObjectStates, Pass, or Done as planner actions.',
        ],
    }
    if planning_mode == 'runtime_replan':
        user_payload['execution_context'] = deepcopy(
            execution_context if isinstance(execution_context, dict) else {}
        )
        user_payload['runtime_constraints'] = deepcopy(
            runtime_constraints if isinstance(runtime_constraints, dict) else {}
        )
        user_payload['constraints'].extend([
            'This is runtime replanning after execution failed. Treat execution_context and runtime_constraints as authoritative current-state facts.',
            'Return a complete replacement plan for only the work that remains from the current world state, not a patch to the previous graph.',
            'Do not regenerate actions whose effects are listed as completed or already satisfied in execution_context.',
            'Respect current robot inventory and held-object ownership; do not pick up an object that is already held.',
            'Never select or depend on an object ID listed in runtime_constraints.excluded_object_ids.',
            'Do not repeat a failed action/target plan listed in execution_context unless supplied current-state evidence makes it newly executable.',
            'Preserve the public user goal while using only current scene catalogue types and executable action contracts.',
        ])
        system_prompt += (
            ' This request is runtime replanning. The execution report and current '
            'world state are authoritative. Generate a complete plan only for the '
            'remaining work and do not repeat already-satisfied effects.'
        )
    if correction:
        correction_violations = normalize_validation_violations(
            list(correction.get('violations') or []),
            stage='deterministic_validation',
        )
        user_payload['correction'] = {
            'attempt': int(correction.get('attempt') or 1),
            'max_attempts': int(correction.get('max_attempts') or 1),
            'violations': correction_violations,
            'summary': str(correction.get('summary') or correction_summary(correction_violations)),
            'instruction': (
                'Correct every listed error and return a complete replacement JSON '
                'plan. Do not repeat the invalid output.'
            ),
        }
        user_payload['constraints'].extend([
            'This is a corrective retry: do not copy any selector named by a violation into the replacement plan.',
            'For each violation, apply required_fix literally across the entire replacement plan, not only the named subtask.',
            'Never repeat an invalid_values entry in the same reported field or action/selector context.',
            'When an object_type is absent from scene_catalog, remove or replace every interaction selector that uses it.',
            'When an object_type lacks the affordance required by an action, do not schedule that action for that type.',
            'Scene-graph captions are visual hints only and never override exact scene_catalog object_type or affordance values.',
            'Before returning, check every interaction selector, action argument, and listed violation.',
        ])
        system_prompt += (
            ' This request is a corrective retry. The replacement must fix every '
            'reported validation error and must not repeat an invalid selector.'
        )

    chat = qwen_chat
    owns_chat = chat is None
    close_vlm_chat = None
    original_max_new_tokens: Any = None
    overrides_max_new_tokens = False
    if owns_chat:
        try:
            from conceptgraph.vlm import build_vlm_chat, close_vlm_chat
        except Exception as exc:
            if diagnostics is not None:
                diagnostics.update({
                    'status': 'failed',
                    'stage': 'import',
                    'error_type': type(exc).__name__,
                    'error_message': str(exc),
                })
            return None
    try:
        if chat is None:
            chat = build_vlm_chat(
                backend='qwen',
                model_path=model_path,
                conv_mode=conv_mode,
                num_gpus=num_gpus,
            )
        if hasattr(chat, 'max_new_tokens'):
            original_max_new_tokens = chat.max_new_tokens
            chat.max_new_tokens = max_new_tokens
            overrides_max_new_tokens = True
        if hasattr(chat, 'reset'):
            chat.reset()
        if hasattr(chat, 'messages'):
            chat.messages = [{'role': 'system', 'content': system_prompt}]
        response_text = chat(json.dumps(user_payload, ensure_ascii=False, indent=2))
    except Exception as exc:
        if diagnostics is not None:
            diagnostics.update({
                'status': 'failed',
                'stage': 'model_call',
                'error_type': type(exc).__name__,
                'error_message': str(exc),
            })
        return None
    finally:
        if overrides_max_new_tokens:
            chat.max_new_tokens = original_max_new_tokens
        if owns_chat and close_vlm_chat is not None:
            close_vlm_chat(chat)

    if diagnostics is not None:
        response_string = str(response_text)
        diagnostics.update({
            'stage': 'json_parse',
            'raw_response_preview': (
                response_string
                if len(response_string) <= 8000
                else response_string[:8000] + '...'
            ),
            'response_characters': len(response_string),
        })
    parsed = parse_json_from_text(str(response_text or ''))
    if diagnostics is not None:
        diagnostics.update({
            'status': 'success' if parsed is not None else 'failed',
            'stage': 'complete' if parsed is not None else 'json_parse',
            'json_parse_status': 'success' if parsed is not None else 'invalid_json',
        })
    return parsed


def semantic_action_contracts_for_critic() -> dict[str, dict[str, Any]]:
    """Return semantics only; execution preconditions belong to compiler/runtime."""

    contracts: dict[str, dict[str, Any]] = {}
    for action, contract in ACTION_CONTRACTS.items():
        item: dict[str, Any] = {
            'roles': list((contract.get('roles') or {}).keys()),
        }
        semantic_args = {
            name: {
                'values': list(rule.get('values') or ()),
            }
            for name, rule in (contract.get('args') or {}).items()
            if isinstance(rule, dict) and rule.get('semantic')
        }
        if semantic_args:
            item['semantic_args'] = semantic_args
        contracts[action] = item
    return contracts


def call_local_qwen_for_plan_critique(
    *,
    task: str,
    subgraph: dict[str, Any],
    candidate_plan: dict[str, Any],
    scene_catalog_summary: list[dict[str, Any]],
    resolved_semantic_goal: dict[str, Any] | None = None,
    placement_compatibility: dict[str, list[str]] | None = None,
    model_path: str | None,
    conv_mode: str,
    num_gpus: int,
    qwen_chat: Any | None = None,
    max_new_tokens: int = 768,
    diagnostics: dict[str, Any] | None = None,
    planning_mode: str = 'initial',
    execution_context: dict[str, Any] | None = None,
    runtime_constraints: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Ask a reset Qwen verifier session to judge plan/instruction semantics."""

    if diagnostics is not None:
        diagnostics.update({
            'attempted': True,
            'status': 'started',
            'stage': 'initialization',
            'error_type': None,
            'error_message': None,
            'raw_response_preview': None,
            'response_characters': 0,
            'json_parse_status': 'not_attempted',
            'max_new_tokens': max_new_tokens,
        })
    system_prompt = (
        'You are an independent semantic verifier for a robot task planner. '
        'Compare the public user instruction with the complete candidate plan. '
        'Reject plans that perform the wrong action, omit requested objects or actions, '
        'use a source or destination that contradicts the instruction, or declare '
        'completion without satisfying the instruction. Treat the scene graph as '
        'incomplete and do not invent benchmark answers. Deterministic selector and '
        'affordance validation is handled elsewhere. Return JSON only. '
        'The evidence_policy in the request is mandatory. Do not return invalid '
        'for an object merely missing from the incomplete subgraph, and do not '
        'use hidden benchmark answers. Use ordinary category-level household '
        'semantics when the instruction explicitly asks for appropriate positions. '
        'The semantic_action_contracts in the request describe semantics only. '
        'Never require a source or destination role that the candidate action '
        'contract does not declare. Never judge tool acquisition, held state, '
        'ancestor pickup steps, affordances, argument syntax or ranges, or runtime '
        'grounding; deterministic compiler/runtime own all of those checks.'
    )
    if planning_mode == 'runtime_replan':
        system_prompt += (
            ' This is runtime replanning. Judge whether the authoritative completed '
            'effects in execution_context together with the candidate remaining plan '
            'satisfy the public instruction. Do not require completed effects to be '
            'repeated in the candidate plan.'
        )
    user_payload = {
        'request': 'validate_candidate_plan',
        'planning_mode': planning_mode,
        'task': task,
        'subgraph_summary': summarize_subgraph(subgraph),
        'scene_catalog': deepcopy(scene_catalog_summary),
        'resolved_semantic_goal': deepcopy(resolved_semantic_goal or {}),
        'placement_compatibility': deepcopy(placement_compatibility or {}),
        'candidate_plan': semantic_critic_plan_projection(candidate_plan),
        'semantic_action_contracts': semantic_action_contracts_for_critic(),
        'evidence_policy': {
            'subgraph_is_incomplete': True,
            'catalog_type_means_object_exists_in_scene': True,
            'absence_from_subgraph_must_not_be_an_error': True,
            'commonsense_category_suitability_must_be_checked': True,
            'hidden_benchmark_ground_truth_must_not_be_used': True,
            'runtime_selector_binding_must_not_be_rejected': True,
        },
        'mandatory_decision_rules': [
            'Apply these rules before writing a verdict.',
            'Treat semantic_action_contracts as authoritative for task meaning only; never require a selector role absent from the contract for that action.',
            'Never require tool acquisition, a held object, an ancestor pick, an affordance, or an execution-only action argument; deterministic compiler/runtime own those checks.',
            'The clean action maps directly to AI2-THOR CleanObject and has only a dirtyable source role; wash/clean never requires a sink, water source, faucet, or destination_selector.',
            'Deterministic validation owns selector affordances, action arguments, and executable role validation; do not redefine those contracts.',
            'Treat resolved_semantic_goal as the resolver proposal that the candidate must follow, but compare that proposal with the public instruction.',
            'If the proposed action, source set, destination, quantifier, or semantic argument misunderstands the instruction, return code resolved_intent_mismatch and identify the affected semantic field.',
            'If the instruction names a spatial source such as central, compare source types with scene_catalog parent_locations and relative_location; reject omissions and types located only elsewhere.',
            'An exact selector type does not need node_ids or object IDs at planning time.',
            'Judge the executable source_selector and destination_selector; descriptive object_tags do not supply alternate destinations when the selector contains only one destination.',
            'For place instructions with an explicit destination, that user-named destination is authoritative. Do not reject the matching destination as unsuitable or replace it using ordinary household preferences.',
            'When the instruction asks for appropriate positions, use ordinary category-level suitability: food and utensils need not share one destination, and a single catch-all receptacle is invalid when it is unsuitable for one or more heterogeneous source types.',
            'Do not demand one hidden exact storage mapping when several destinations are ordinarily suitable, but every proposed source-to-destination pairing must itself be suitable.',
        ],
        'verdict_examples': [
            {
                'instruction': 'Clear a surface by placing items in appropriate positions.',
                'candidate_fact': 'place a perishable food item into a compatible cold-storage receptacle',
                'status': 'valid',
                'why': 'The proposed destination is ordinarily suitable and no supplied fact contradicts it.',
            },
            {
                'instruction': 'Clear a surface by placing items in appropriate positions.',
                'candidate_fact': 'use separate compatible destinations for food and utensils',
                'status': 'valid',
                'why': 'The plan uses category-appropriate destinations without relying on a hidden benchmark mapping.',
            },
            {
                'instruction': 'Clear a surface by placing items in appropriate positions.',
                'candidate_fact': 'one selector places heterogeneous food and utensils into one catch-all receptacle',
                'status': 'invalid',
                'why': 'A single catch-all destination is not an appropriate category-level placement for every heterogeneous source type.',
            },
            {
                'instruction': 'Clear a surface by placing items in appropriate positions.',
                'candidate_fact': 'place an item back onto the same surface',
                'status': 'invalid',
                'why': 'The destination directly contradicts clearing that surface.',
            },
        ],
        'output_schema': {
            'status': 'valid or invalid',
            'errors': [{
                'task_id': 'candidate subtask id, or null for a plan-wide error',
                'code': 'resolved_intent_mismatch when the locked resolver proposal is wrong; otherwise a short planner semantic error code',
                'field': 'action/grounding.source_selector/grounding.destination_selector/depends_on/or null',
                'message': 'concise actionable semantic error',
                'invalid_values': ['optional conflicting values'],
                'required_fix': 'specific correction required in the replacement plan',
            }],
        },
        'constraints': [
            'Judge only whether the candidate plan semantically satisfies the public task.',
            'A structurally executable plan is still invalid when its action, source, or destination contradicts the instruction.',
            'Use execution_action_contracts as the only authority for which object roles an action requires.',
            'Never demand a source_selector or destination_selector role that is absent from the action contract.',
            'For clean/wash, accept a complete source-only CleanObject task without a sink, water source, faucet, or destination.',
            'Do not require hidden benchmark subtasks or object mappings that are absent from the instruction and scene context.',
            'The scene graph is incomplete: absence of an object type from subgraph captions is not evidence that the object is absent.',
            'When an exact selector type exists in scene_catalog, do not reject it merely because the incomplete subgraph omitted it.',
            'Never reject a candidate because grounding.node_ids or concrete object IDs are empty; runtime binds exact selector types after planning.',
            'Never require the planner to add a node id for a type that has a valid exact selector.',
            'Do not require a hidden benchmark-specific source-to-destination mapping, but do check each proposed pairing for ordinary category-level suitability when the instruction says appropriate positions.',
            'Do not accept one catch-all destination for heterogeneous categories merely because it has the receptacle affordance.',
            'Example: accept a category-appropriate destination even when no hidden exact storage mapping was supplied.',
            'Example: for a task asking to clear a surface, placing an item back on that same surface directly contradicts the instruction and is invalid.',
            'Do not require containment or visibility evidence that is unavailable from the supplied summaries.',
            'When invalid, return at least one structured error that the planner can correct.',
            'When valid, return status valid and an empty errors list.',
        ],
    }
    if planning_mode == 'runtime_replan':
        user_payload['execution_context'] = deepcopy(
            execution_context if isinstance(execution_context, dict) else {}
        )
        user_payload['runtime_constraints'] = deepcopy(
            runtime_constraints if isinstance(runtime_constraints, dict) else {}
        )
        user_payload['constraints'].extend([
            'Judge the candidate as a replacement plan for remaining work from the supplied current state.',
            'Reject repeated completed effects, excluded object IDs, lost held-object ownership, or repetition of the failed plan without new evidence.',
        ])

    chat = qwen_chat
    owns_chat = chat is None
    close_vlm_chat = None
    original_max_new_tokens: Any = None
    overrides_max_new_tokens = False
    if owns_chat:
        try:
            from conceptgraph.vlm import build_vlm_chat, close_vlm_chat
        except Exception as exc:
            if diagnostics is not None:
                diagnostics.update({
                    'status': 'failed',
                    'stage': 'import',
                    'error_type': type(exc).__name__,
                    'error_message': str(exc),
                })
            return None
    try:
        if chat is None:
            chat = build_vlm_chat(
                backend='qwen',
                model_path=model_path,
                conv_mode=conv_mode,
                num_gpus=num_gpus,
            )
        if hasattr(chat, 'max_new_tokens'):
            original_max_new_tokens = chat.max_new_tokens
            chat.max_new_tokens = max_new_tokens
            overrides_max_new_tokens = True
        if hasattr(chat, 'reset'):
            chat.reset()
        if hasattr(chat, 'messages'):
            chat.messages = [{'role': 'system', 'content': system_prompt}]
        response_text = chat(json.dumps(user_payload, ensure_ascii=False, indent=2))
    except Exception as exc:
        if diagnostics is not None:
            diagnostics.update({
                'status': 'failed',
                'stage': 'model_call',
                'error_type': type(exc).__name__,
                'error_message': str(exc),
            })
        return None
    finally:
        if overrides_max_new_tokens:
            chat.max_new_tokens = original_max_new_tokens
        if owns_chat and close_vlm_chat is not None:
            close_vlm_chat(chat)

    response_string = str(response_text or '')
    if diagnostics is not None:
        diagnostics.update({
            'stage': 'json_parse',
            'raw_response_preview': (
                response_string
                if len(response_string) <= 8000
                else response_string[:8000] + '...'
            ),
            'response_characters': len(response_string),
        })
    parsed = parse_json_from_text(response_string)
    if diagnostics is not None:
        diagnostics.update({
            'status': 'success' if parsed is not None else 'failed',
            'stage': 'complete' if parsed is not None else 'json_parse',
            'json_parse_status': 'success' if parsed is not None else 'invalid_json',
        })
    return parsed


def validate_semantic_critique(raw_critique: Any) -> dict[str, Any]:
    """Validate the critic protocol without repairing or reinterpreting its answer."""

    if not isinstance(raw_critique, dict):
        return {
            'protocol_status': 'invalid',
            'status': None,
            'errors': ['semantic critic response must be a JSON object'],
        }
    status = str(raw_critique.get('status') or '').strip().lower()
    raw_errors = raw_critique.get('errors')
    if not isinstance(raw_errors, list) or any(not isinstance(item, dict) for item in raw_errors):
        return {
            'protocol_status': 'invalid',
            'status': status or None,
            'errors': ['semantic critic errors must be a list of structured objects'],
        }
    errors = normalize_validation_violations(raw_errors, stage='semantic_validation')
    if status not in {'valid', 'invalid'}:
        return {
            'protocol_status': 'invalid',
            'status': status or None,
            'errors': [f'semantic critic status must be valid or invalid, got {status!r}'],
        }
    if status == 'valid' and errors:
        return {
            'protocol_status': 'invalid',
            'status': status,
            'errors': ['semantic critic returned errors with valid status'],
        }
    if status == 'invalid' and not errors:
        return {
            'protocol_status': 'invalid',
            'status': status,
            'errors': ['semantic critic returned invalid status without errors'],
        }
    return {'protocol_status': 'valid', 'status': status, 'errors': errors}


SEMANTIC_GROUNDING_FIELDS = (
    'object_tags',
    'source_object_tags',
    'destination_object_tags',
    'source_selector',
    'destination_selector',
    'relation_texts',
)

INSTANCE_GROUNDING_FIELD_PATTERN = re.compile(
    r'(?:^|\.)(?:source_|destination_)?(?:node|object)_?ids?(?:$|\.)|objectid',
    flags=re.IGNORECASE,
)
INSTANCE_GROUNDING_TEXT_PATTERN = re.compile(
    r'\b(?:node|object)[ _-]?ids?\b|\b(?:scene|graph) node\b|\bconcrete object\b',
    flags=re.IGNORECASE,
)
SEMANTIC_SELECTOR_FIELD_PATTERN = re.compile(
    r'(?:^|\.)(source|destination)_selector(?:\.|$)',
    flags=re.IGNORECASE,
)


def semantic_action_args_for_critic(action: Any, value: Any) -> dict[str, Any]:
    """Project only arguments whose values change the public task meaning."""

    raw_args = value if isinstance(value, dict) else {}
    contract = ACTION_CONTRACTS.get(normalize_logical_action(action)) or {}
    return {
        name: deepcopy(raw_args[name])
        for name, rule in (contract.get('args') or {}).items()
        if isinstance(rule, dict) and rule.get('semantic') and name in raw_args
    }


def semantic_critic_plan_projection(candidate_plan: dict[str, Any]) -> dict[str, Any]:
    """Expose plan semantics to the critic without instance-grounding state."""

    raw_tasks = candidate_plan.get('flat_tasks')
    if not isinstance(raw_tasks, list):
        raw_tasks = candidate_plan.get('subtasks')
    projected_tasks: list[dict[str, Any]] = []
    for task in raw_tasks or []:
        if not isinstance(task, dict):
            continue
        grounding = task.get('grounding') if isinstance(task.get('grounding'), dict) else {}
        projected_grounding = {
            field: deepcopy(grounding[field])
            for field in SEMANTIC_GROUNDING_FIELDS
            if field in grounding
        }
        projected_tasks.append({
            'id': task.get('id'),
            'name': task.get('name'),
            'description': task.get('description'),
            'action': task.get('action'),
            'action_args': semantic_action_args_for_critic(
                task.get('action'),
                task.get('action_args'),
            ),
            'grounding': projected_grounding,
            'depends_on': deepcopy(task.get('depends_on') or []),
            'termination_check': task.get('termination_check'),
        })
    return {
        'task': candidate_plan.get('task'),
        'reasoning_summary': candidate_plan.get('reasoning_summary'),
        'subtasks': projected_tasks,
    }


def is_out_of_scope_semantic_grounding_error(violation: dict[str, Any]) -> bool:
    """Return true when a critic violation depends on runtime instance IDs."""

    field = str(violation.get('field') or '')
    if INSTANCE_GROUNDING_FIELD_PATTERN.search(field):
        return True
    text = ' '.join([
        str(violation.get('code') or ''),
        str(violation.get('message') or ''),
        str(violation.get('required_fix') or ''),
    ])
    return bool(INSTANCE_GROUNDING_TEXT_PATTERN.search(text))


def _semantic_candidate_tasks(
    candidate_plan: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(candidate_plan, dict):
        return {}
    raw_tasks = candidate_plan.get('flat_tasks')
    if not isinstance(raw_tasks, list):
        raw_tasks = candidate_plan.get('subtasks')
    return {
        str(task.get('id')): task
        for task in raw_tasks or []
        if isinstance(task, dict) and str(task.get('id') or '').strip()
    }


def _candidate_supplies_semantic_role(task: dict[str, Any], role: str) -> bool:
    grounding = task.get('grounding') if isinstance(task.get('grounding'), dict) else {}
    selector = grounding.get(f'{role}_selector')
    raw_types = selector.get('object_types') if isinstance(selector, dict) else []
    if isinstance(raw_types, str):
        raw_types = [raw_types]
    if isinstance(raw_types, list) and any(str(value).strip() for value in raw_types):
        return True
    raw_tags = grounding.get(f'{role}_object_tags')
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]
    return isinstance(raw_tags, list) and any(str(value).strip() for value in raw_tags)


def out_of_scope_action_contract_error(
    violation: dict[str, Any],
    candidate_tasks: dict[str, dict[str, Any]],
) -> tuple[str, str] | None:
    """Return action/role when a critic demands a role forbidden by the contract."""

    task_id = str(violation.get('task_id') or '').strip()
    if not task_id or task_id not in candidate_tasks:
        return None
    match = SEMANTIC_SELECTOR_FIELD_PATTERN.search(str(violation.get('field') or ''))
    if match is None:
        return None

    task = candidate_tasks[task_id]
    action = normalize_logical_action(task.get('action'))
    contract = ACTION_CONTRACTS.get(action)
    if contract is None:
        return None
    role = match.group(1).lower()
    if role in (contract.get('roles') or {}):
        return None
    if _candidate_supplies_semantic_role(task, role):
        return None
    return action, role


def out_of_scope_explicit_destination_error(
    violation: dict[str, Any],
    candidate_tasks: dict[str, dict[str, Any]],
    root_intent: dict[str, Any] | None,
) -> list[str] | None:
    """Return explicit destinations when the critic overrides a matching user goal."""


    if (
        isinstance(root_intent, dict)
        and root_intent.get('intent_resolution_source') == 'qwen_two_stage_intent'
    ):
        return None

    task_id = str(violation.get('task_id') or '').strip()
    if not task_id or task_id not in candidate_tasks:
        return None
    match = SEMANTIC_SELECTOR_FIELD_PATTERN.search(str(violation.get('field') or ''))
    if match is None or match.group(1).lower() != 'destination':
        return None

    task = candidate_tasks[task_id]
    if normalize_logical_action(task.get('action')) != 'place':
        return None
    roles = root_intent.get('roles') if isinstance(root_intent, dict) else {}
    authoritative = [
        str(value).strip()
        for value in (roles.get('destination') if isinstance(roles, dict) else []) or []
        if str(value).strip()
    ]
    if not authoritative:
        return None
    grounding = task.get('grounding') if isinstance(task.get('grounding'), dict) else {}
    selector = grounding.get('destination_selector')
    raw_types = selector.get('object_types') if isinstance(selector, dict) else []
    if isinstance(raw_types, str):
        raw_types = [raw_types]
    candidate_keys = {_semantic_key(value) for value in raw_types or [] if str(value).strip()}
    authoritative_keys = {_semantic_key(value) for value in authoritative}
    if not candidate_keys or candidate_keys != authoritative_keys:
        return None
    return authoritative


def out_of_scope_execution_contract_error(
    violation: dict[str, Any],
    candidate_tasks: dict[str, dict[str, Any]],
) -> str | None:
    """Return the action when a critic judges compiler/runtime-owned details."""

    task_id = str(violation.get('task_id') or '').strip()
    task = candidate_tasks.get(task_id)
    if task is None:
        return None
    action = normalize_logical_action(task.get('action'))
    contract = ACTION_CONTRACTS.get(action) or {}
    code = str(violation.get('code') or '').strip().lower()
    field = str(violation.get('field') or '').strip().lower()
    text = ' '.join([
        code,
        field,
        str(violation.get('message') or '').lower(),
        str(violation.get('required_fix') or '').lower(),
    ])

    if field.startswith('action_args'):
        semantic_args = {
            name: rule
            for name, rule in (contract.get('args') or {}).items()
            if isinstance(rule, dict) and rule.get('semantic')
        }
        field_name = field.rsplit('.', 1)[-1] if '.' in field else ''
        semantic_arg_mentioned = any(
            name.lower() in text
            or any(str(value).lower() in text for value in rule.get('values') or ())
            for name, rule in semantic_args.items()
        )
        if (
            field_name
            and field_name not in {name.lower() for name in semantic_args}
        ) or (not field_name and not semantic_arg_mentioned):
            return action

    if contract.get('required_tool') and (
        'tool' in text
        or 'held' in text
        or 'inventory' in text
        or 'pick up' in text
        or 'pickup' in text
    ):
        return action
    if contract.get('requires_held_source') and (
        'held' in text
        or 'inventory' in text
        or 'ancestor pick' in text
        or 'pick dependency' in text
        or 'pickup' in text
    ):
        return action
    if (
        'affordance' in text
        or code in {
            'invalid_affordance',
            'missing_affordance',
            'unsupported_affordance',
            'incompatible_affordance',
        }
    ):
        return action
    return None


def enforce_semantic_critic_scope(
    semantic_result: dict[str, Any],
    *,
    candidate_plan: dict[str, Any] | None = None,
    root_intent: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Remove critic claims owned by runtime grounding or execution contracts."""

    effective = deepcopy(semantic_result)
    if effective.get('protocol_status') != 'valid' or effective.get('status') != 'invalid':
        return effective, []

    candidate_tasks = _semantic_candidate_tasks(candidate_plan)
    retained: list[dict[str, Any]] = []
    discarded: list[dict[str, Any]] = []
    for violation in effective.get('errors') or []:
        if not isinstance(violation, dict):
            retained.append(violation)
            continue
        execution_action = None
        if is_out_of_scope_semantic_grounding_error(violation):
            replacement_code = 'out_of_scope_grounding_error'
            action_role = None
            explicit_destinations = None
        else:
            action_role = out_of_scope_action_contract_error(violation, candidate_tasks)
            explicit_destinations = out_of_scope_explicit_destination_error(
                violation,
                candidate_tasks,
                root_intent,
            )
            execution_action = out_of_scope_execution_contract_error(
                violation,
                candidate_tasks,
            )
            if action_role is not None:
                replacement_code = 'out_of_scope_action_contract_error'
            elif explicit_destinations is not None:
                replacement_code = 'out_of_scope_explicit_destination_error'
            elif execution_action is not None:
                replacement_code = 'out_of_scope_execution_error'
            else:
                retained.append(violation)
                continue
        annotated = deepcopy(violation)
        annotated['original_code'] = annotated.get('code')
        annotated['code'] = replacement_code
        annotated['discarded'] = True
        if action_role is not None:
            annotated['contract_action'] = action_role[0]
            annotated['unsupported_role'] = action_role[1]
        if explicit_destinations is not None:
            annotated['authoritative_destination_types'] = explicit_destinations
        if execution_action is not None:
            annotated['contract_action'] = execution_action
        discarded.append(annotated)

    effective['errors'] = retained
    if not retained:
        effective['status'] = 'valid'
    return effective, discarded


def semantic_violations_challenge_resolved_intent(
    violations: list[dict[str, Any]],
) -> bool:
    """Return true for a critic error that can only come from locked intent."""

    for violation in violations:
        if not isinstance(violation, dict):
            continue
        code = str(violation.get('code') or '').strip().lower()
        field = str(violation.get('field') or '').strip().lower()
        if code == 'resolved_intent_mismatch' and any(
            component in field
            for component in (
                'action',
                'source',
                'destination',
                'quantifier',
                'semantic_arg',
            )
        ):
            return True
        if (
            code == 'destination_suitability_mismatch'
            and 'destination' in field
        ):
            return True
    return False


def audit_plan_entity_coverage(task: str, subgraph: dict, task_graph: dict) -> dict[str, Any]:
    """Report entity loss without changing planner selection or task graph behavior."""

    task_spec = subgraph.get('task_spec') if isinstance(subgraph.get('task_spec'), dict) else {}
    required_entities = unique_preserve_order([
        str(entity)
        for key in ('target_objects', 'source_receptacles', 'destination_receptacles', 'landmarks')
        for entity in (task_spec.get(key) or [])
        if str(entity).strip()
    ])
    if not required_entities:
        required_entities = infer_task_object_tags(task, limit=12)

    plan_parts = []
    for subtask in task_graph.get('flat_tasks') or []:
        if not isinstance(subtask, dict):
            continue
        grounding = subtask.get('grounding') if isinstance(subtask.get('grounding'), dict) else {}
        plan_parts.extend([
            str(subtask.get('name') or ''),
            str(subtask.get('description') or ''),
            ' '.join(str(tag) for tag in grounding.get('object_tags') or []),
            ' '.join(str(tag) for tag in ((grounding.get('source_selector') or {}).get('object_types') or [])),
            ' '.join(str(tag) for tag in ((grounding.get('destination_selector') or {}).get('object_types') or [])),
            ' '.join(str(text) for text in grounding.get('relation_texts') or []),
        ])
    normalized_plan_text = re.sub(r'[^a-z0-9]+', ' ', ' '.join(plan_parts).lower())

    covered = []
    missing = []
    for entity in required_entities:
        entity_tokens = [
            token for token in tokenize(entity)
            if token not in STOPWORDS
        ]
        is_covered = bool(entity_tokens) and all(
            re.search(rf'\b{re.escape(token)}\b', normalized_plan_text)
            for token in entity_tokens
        )
        (covered if is_covered else missing).append(entity)
    return {
        'required_entities': required_entities,
        'covered_entities': covered,
        'missing_entities': missing,
        'complete': not missing,
        'enforced': False,
    }


def infer_actions(task: str) -> list[str]:
    tokens = tokenize(task)
    text = " ".join(tokens)
    actions = []
    if re.search(r"\b(?:turn|switch|toggle)\s+on\b", text):
        actions.append("toggle_on")
    if re.search(r"\b(?:turn|switch|toggle)\s+off\b", text):
        actions.append("toggle_off")
    for action, keywords in ACTION_KEYWORDS.items():
        if any(token in keywords for token in tokens):
            actions.append(action)
    return unique_preserve_order(actions)


def infer_keywords(task: str) -> list[str]:
    return unique_preserve_order([
        token for token in tokenize(task)
        if token not in STOPWORDS and len(token) > 1 and not token.isdigit()
    ])


def infer_task_object_tags(task: str, limit: int = 6) -> list[str]:
    """Semantic task terms to search for when the scene graph is incomplete."""

    action_words = set().union(*ACTION_KEYWORDS.values())
    return [
        token
        for token in infer_keywords(task)
        if token not in action_words
    ][:limit]


def _singular_word(value: str) -> str:
    word = str(value or '').strip().lower()
    if len(word) > 4 and word.endswith('ies'):
        return word[:-3] + 'y'
    if len(word) > 4 and word.endswith(('ches', 'shes', 'sses', 'xes', 'zes', 'oes')):
        return word[:-2]
    if len(word) > 3 and word.endswith('ves'):
        return word[:-3] + 'f'
    if len(word) > 3 and word.endswith('s') and not word.endswith('ss'):
        return word[:-1]
    return word


def _catalogue_type_match(task: str, object_type: str) -> tuple[str, int] | None:
    task_words = [_singular_word(word) for word in _type_words(task)]
    type_words = [_singular_word(word) for word in _type_words(object_type)]
    if not task_words or not type_words:
        return None
    width = len(type_words)
    if any(task_words[index:index + width] == type_words for index in range(len(task_words) - width + 1)):
        return ('exact', width)
    compact_type = ''.join(type_words)
    if compact_type in task_words:
        return ('exact', width)
    return None


def _match_catalogue_types(
    query_text: str,
    affordance: str | tuple[str, ...],
    scene_catalog_summary: list[dict[str, Any]],
) -> list[str]:
    matched_types: list[tuple[str, int]] = []
    required_affordances = _required_affordances(affordance)
    for entry in scene_catalog_summary:
        if not isinstance(entry, dict):
            continue
        object_type = str(entry.get('object_type') or '').strip()
        affordances = (
            entry.get('affordances')
            if isinstance(entry.get('affordances'), dict)
            else {}
        )
        match = _catalogue_type_match(query_text, object_type)
        if (
            object_type
            and match is not None
            and any(bool(affordances.get(field)) for field in required_affordances)
        ):
            matched_types.append((object_type, match[1]))
    if not matched_types:
        return []
    longest = max(width for _, width in matched_types)
    return [
        object_type
        for object_type, width in matched_types
        if width == longest
    ]


def _primary_root_action(task: str, subgraph: dict[str, Any]) -> str | None:
    inferred = infer_actions(task)
    task_spec = subgraph.get('task_spec') if isinstance(subgraph.get('task_spec'), dict) else {}
    declared = normalize_action_name(task_spec.get('task_type')) if task_spec.get('task_type') else None
    if declared and declared != 'other' and declared in inferred:
        return declared
    if 'place' in inferred:
        return 'place'
    return inferred[0] if inferred else None


def derive_root_intent(
    task: str,
    subgraph: dict[str, Any],
    scene_catalog_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    """Derive only validation facts that are explicit in the instruction/catalogue."""

    action = _primary_root_action(task, subgraph)
    quantifier = 'all' if re.search(r'\b(?:all|every|each)\b', task, re.IGNORECASE) else 'one'
    roles: dict[str, list[str]] = {}
    requirements = ACTION_ROLE_AFFORDANCES.get(str(action or '')) or {}
    task_spec = subgraph.get('task_spec') if isinstance(subgraph.get('task_spec'), dict) else {}
    for role, affordance in requirements.items():
        role_text = task
        if action == 'place':
            if role == 'destination':
                destination_entities = task_spec.get('destination_receptacles') or []
                if destination_entities:
                    role_text = ' '.join(str(value) for value in destination_entities)
            elif role == 'source':
                source_entities = task_spec.get('target_objects') or []
                destination_keys = {
                    _semantic_key(value)
                    for value in task_spec.get('destination_receptacles') or []
                }
                source_entities = [
                    value for value in source_entities
                    if _semantic_key(value) not in destination_keys
                ]
                if source_entities:
                    role_text = ' '.join(str(value) for value in source_entities)
        explicit_entities: list[str] = []
        if action == 'place':
            if role == 'destination':
                explicit_entities = [str(value) for value in task_spec.get('destination_receptacles') or []]
            elif role == 'source':
                explicit_entities = [str(value) for value in source_entities] if 'source_entities' in locals() else []

        def match_catalogue_types(query_text: str) -> list[str]:
            return _match_catalogue_types(
                query_text,
                affordance,
                scene_catalog_summary,
            )

        if explicit_entities:
            matching = []
            for entity in explicit_entities:
                matching.extend(match_catalogue_types(entity))
        else:
            matching = match_catalogue_types(role_text)
        roles[role] = unique_preserve_order(matching)
    return {
        'action': action,
        'explicit_actions': infer_actions(task),
        'quantifier': quantifier,
        'roles': roles,
        'catalogue_constrained': bool(scene_catalog_summary),
    }


def _validate_selector_for_role(
    *,
    task_id: str,
    grounding: dict[str, Any],
    role: str,
    affordance: str | tuple[str, ...],
    expected_quantifier: str | None,
    scene_catalog_summary: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    raw_selector = grounding.get(f'{role}_selector')
    if not isinstance(raw_selector, dict):
        return [], [f'{task_id}: missing {role}_selector']
    quantifier = str(raw_selector.get('quantifier') or '').strip().lower()
    if quantifier not in VALID_QUANTIFIERS:
        errors.append(f'{task_id}: invalid {role}_selector quantifier {quantifier!r}')
    elif expected_quantifier is not None and quantifier != expected_quantifier:
        errors.append(
            f'{task_id}: {role}_selector quantifier must be {expected_quantifier!r}, got {quantifier!r}'
        )
    raw_types = raw_selector.get('object_types') or []
    if isinstance(raw_types, str):
        raw_types = [raw_types]
    if not isinstance(raw_types, list) or not raw_types:
        return [], [*errors, f'{task_id}: missing {role}_selector object_types']
    if role == 'destination' and len(raw_types) != 1:
        errors.append(
            f'{task_id}: destination_selector must contain exactly one object_type; got {raw_types}'
        )

    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in scene_catalog_summary:
        if isinstance(entry, dict) and str(entry.get('object_type') or '').strip():
            by_key[_semantic_key(entry['object_type'])].append(entry)
    resolved: list[str] = []
    for raw_type in raw_types:
        candidates = by_key.get(_semantic_key(raw_type), [])
        if not candidates:
            errors.append(
                f'{task_id}: {role}_selector type {raw_type!r} is absent from the scene catalogue'
            )
            continue
        if len(candidates) != 1:
            errors.append(
                f'{task_id}: {role}_selector type {raw_type!r} is ambiguous in the scene catalogue'
            )
            continue
        entry = candidates[0]
        canonical = str(entry['object_type'])
        if str(raw_type).strip() != canonical:
            errors.append(
                f'{task_id}: {role}_selector type {raw_type!r} must use canonical type {canonical!r}'
            )
        affordances = entry.get('affordances') if isinstance(entry.get('affordances'), dict) else {}
        required_affordances = (affordance,) if isinstance(affordance, str) else tuple(affordance)
        if not any(bool(affordances.get(field)) for field in required_affordances):
            requirement = required_affordances[0] if len(required_affordances) == 1 else f"any of {list(required_affordances)}"
            errors.append(
                f'{task_id}: {role}_selector type {canonical!r} does not satisfy {requirement}'
            )
        if canonical not in resolved:
            resolved.append(canonical)
    return resolved, errors

def validate_planner_output(
    raw_plan: Any,
    *,
    task: str,
    subgraph: dict[str, Any],
    scene_catalog_summary: list[dict[str, Any]],
    agent_context: dict[str, Any] | None = None,
    planning_mode: str = 'initial',
    execution_context: dict[str, Any] | None = None,
    root_intent_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate model output before any sanitizer can repair or reinterpret it."""

    root_intent = (
        deepcopy(root_intent_override)
        if isinstance(root_intent_override, dict)
        else derive_root_intent(task, subgraph, scene_catalog_summary)
    )
    errors: list[str] = []
    if not isinstance(raw_plan, dict):
        return {
            'status': 'invalid',
            'errors': ['planner response is not a JSON object'],
            'root_intent': root_intent,
        }
    raw_subtasks = raw_plan.get('subtasks')
    if not isinstance(raw_subtasks, list) or not raw_subtasks:
        return {
            'status': 'invalid',
            'errors': ['planner response must contain a non-empty subtasks list'],
            'root_intent': root_intent,
        }
    non_objects = [index for index, value in enumerate(raw_subtasks) if not isinstance(value, dict)]
    if non_objects:
        errors.append(f'subtasks at indexes {non_objects} are not JSON objects')

    subtasks = [value for value in raw_subtasks if isinstance(value, dict)]
    allowed_actions = set(PLANNER_ACTIONS) | {'search', 'cut'}
    for index, subtask in enumerate(subtasks, start=1):
        action_value = normalize_logical_action(subtask.get('action'))
        if action_value not in allowed_actions:
            errors.append(f'subtask {index}: invalid action {action_value!r}')
            continue
        _, argument_errors = validate_action_args(action_value, subtask.get('action_args'))
        for argument_error in argument_errors:
            errors.append(
                f"{str(subtask.get('id') or f'subtask_{index}')}: {argument_error['message']}"
            )

    if scene_catalog_summary:
        for index, subtask in enumerate(subtasks, start=1):
            task_id = str(subtask.get('id') or f'subtask_{index}')
            action = normalize_action_name(subtask.get('action'))
            requirements = ACTION_ROLE_AFFORDANCES.get(action)
            if not requirements:
                continue
            grounding = subtask.get('grounding') if isinstance(subtask.get('grounding'), dict) else {}
            for role, affordance in requirements.items():
                contract = ACTION_CONTRACTS.get(action) or {}
                contract_quantifier = (
                    contract.get('source_quantifier')
                    if role == 'source'
                    else None
                )
                resolved, selector_errors = _validate_selector_for_role(
                    task_id=task_id,
                    grounding=grounding,
                    role=role,
                    affordance=affordance,
                    expected_quantifier=(
                        'one' if role == 'destination' else contract_quantifier
                    ),
                    scene_catalog_summary=scene_catalog_summary,
                )
                del resolved
                errors.extend(selector_errors)
            for role in ('source', 'destination'):
                if role in requirements:
                    continue
                selector = grounding.get(f'{role}_selector')
                raw_types = selector.get('object_types') if isinstance(selector, dict) else []
                if isinstance(raw_types, str):
                    raw_types = [raw_types]
                if isinstance(raw_types, list) and any(str(value).strip() for value in raw_types):
                    errors.append(
                        f'{task_id}: {role}_selector is not supported for action {action!r}'
                    )

    expected_action = root_intent.get('action')
    completed_actions = {
        normalize_action_name(task_item.get('action'))
        for task_item in (
            (execution_context or {}).get('completed_tasks') or []
        )
        if isinstance(task_item, dict) and task_item.get('action')
    }
    expected_action_already_satisfied = (
        planning_mode == 'runtime_replan'
        and expected_action in completed_actions
    )
    root_tasks = [
        subtask for subtask in subtasks
        if normalize_action_name(subtask.get('action')) == expected_action
    ] if expected_action else subtasks
    if expected_action and not root_tasks and not expected_action_already_satisfied:
        errors.append(f'root intent action {expected_action!r} is missing from subtasks')

    requirements = ACTION_ROLE_AFFORDANCES.get(str(expected_action or '')) or {}
    expected_roles = root_intent.get('roles') if isinstance(root_intent.get('roles'), dict) else {}
    if root_tasks and requirements:
        resolved_by_role: dict[str, list[str]] = defaultdict(list)
        for index, subtask in enumerate(root_tasks, start=1):
            task_id = str(subtask.get('id') or f'root_task_{index}')
            grounding = subtask.get('grounding') if isinstance(subtask.get('grounding'), dict) else {}
            for role, affordance in requirements.items():
                expected_types = list(expected_roles.get(role) or [])
                validate_unknown_all_source = (
                    role == 'source'
                    and root_intent.get('quantifier') == 'all'
                    and bool(scene_catalog_summary)
                )
                if not expected_types and not validate_unknown_all_source:
                    continue
                expected_quantifier = root_intent['quantifier'] if role == 'source' else 'one'
                resolved, selector_errors = _validate_selector_for_role(
                    task_id=task_id,
                    grounding=grounding,
                    role=role,
                    affordance=affordance,
                    expected_quantifier=expected_quantifier,
                    scene_catalog_summary=scene_catalog_summary,
                )
                resolved_by_role[role].extend(resolved)
                errors.extend(selector_errors)
        if (
            root_intent.get('quantifier') == 'one'
            and len(expected_roles.get('source') or []) > 1
            and 'source' in requirements
        ):
            selected_once: list[str] = []
            for index, subtask in enumerate(root_tasks, start=1):
                task_id = str(subtask.get('id') or f'root_task_{index}')
                grounding = (
                    subtask.get('grounding')
                    if isinstance(subtask.get('grounding'), dict)
                    else {}
                )
                selector = (
                    grounding.get('source_selector')
                    if isinstance(grounding.get('source_selector'), dict)
                    else {}
                )
                values = selector.get('object_types') or []
                if isinstance(values, str):
                    values = [values]
                values = [str(value) for value in values if str(value).strip()]
                if len(values) != 1:
                    errors.append(
                        f'{task_id}: quantifier one with multiple semantic source '
                        'types requires one atomic root task containing exactly one '
                        f'source_selector object_type; got {values}'
                    )
                selected_once.extend(values)
            duplicate_once = sorted({
                value for value in selected_once if selected_once.count(value) > 1
            })
            if duplicate_once:
                errors.append(
                    'root source_selector types must occur once across atomic '
                    f'quantifier-one tasks; duplicates {duplicate_once}'
                )
        expected_semantic_args = (
            root_intent.get('semantic_args')
            if isinstance(root_intent.get('semantic_args'), dict)
            else {}
        )
        if expected_semantic_args:
            for index, subtask in enumerate(root_tasks, start=1):
                task_id = str(subtask.get('id') or f'root_task_{index}')
                normalized_args, _ = validate_action_args(
                    expected_action,
                    subtask.get('action_args'),
                )
                for name, expected_value in expected_semantic_args.items():
                    if normalized_args.get(name) != expected_value:
                        errors.append(
                            f'{task_id}: action_args.{name} must preserve resolved '
                            f'semantic value {expected_value!r}'
                        )
        if root_intent.get('quantifier') == 'all' and 'source' in requirements:
            resolved_sources = resolved_by_role.get('source') or []
            duplicate_sources = sorted({
                value for value in resolved_sources if resolved_sources.count(value) > 1
            })
            if duplicate_sources:
                errors.append(
                    'root source_selector types must not overlap across all-selector '
                    f'tasks; duplicates {duplicate_sources}'
                )
        for role, expected_types_value in expected_roles.items():
            expected_types = set(expected_types_value or [])
            if not expected_types:
                continue
            resolved_types = set(resolved_by_role.get(role) or [])
            missing = sorted(expected_types - resolved_types)
            unexpected = sorted(resolved_types - expected_types)
            if missing:
                if role == 'source' and root_intent.get('semantic_goal_resolved'):
                    errors.append(
                        'root source_selector is missing semantic goal types '
                        f'{missing}; required exact types are {sorted(expected_types)}'
                    )
                else:
                    errors.append(f'root {role}_selector is missing catalogue types {missing}')
            if role == 'source' and unexpected and root_intent.get('semantic_goal_resolved'):
                errors.append(
                    'root source_selector contains unexpected semantic goal types '
                    f'{unexpected}; required exact types are {sorted(expected_types)}'
                )
            elif role != 'source' and unexpected:
                errors.append(f'root {role}_selector contains unrelated catalogue types {unexpected}')

    if (
        root_tasks
        and requirements
        and root_intent.get('grounding_mode') == 'subgraph'
    ):
        for index, subtask in enumerate(subtasks, start=1):
            task_id = str(subtask.get('id') or f'subtask_{index}')
            grounding = (
                subtask.get('grounding')
                if isinstance(subtask.get('grounding'), dict)
                else {}
            )
            for role in ('source', 'destination'):
                selector = grounding.get(f'{role}_selector')
                raw_types = (
                    selector.get('object_types')
                    if isinstance(selector, dict)
                    else []
                )
                if isinstance(raw_types, str):
                    raw_types = [raw_types]
                if isinstance(raw_types, list) and any(
                    str(value).strip() for value in raw_types
                ):
                    errors.append(
                        f'{task_id}: {role}_selector must be omitted in subgraph '
                        'grounding mode; use resolved role node ids'
                    )
        available_node_keys = {
            str(node.get('pruned_id'))
            for node in subgraph.get('nodes') or []
            if isinstance(node, dict) and node.get('pruned_id') is not None
        }
        expected_role_nodes = (
            root_intent.get('role_nodes')
            if isinstance(root_intent.get('role_nodes'), dict)
            else {}
        )
        expected_role_tags = (
            root_intent.get('role_tags')
            if isinstance(root_intent.get('role_tags'), dict)
            else {}
        )
        covered_node_keys: dict[str, list[str]] = defaultdict(list)
        for index, subtask in enumerate(root_tasks, start=1):
            task_id = str(subtask.get('id') or f'root_task_{index}')
            grounding = (
                subtask.get('grounding')
                if isinstance(subtask.get('grounding'), dict)
                else {}
            )
            raw_all_ids = grounding.get('node_ids') or []
            if not isinstance(raw_all_ids, list):
                raw_all_ids = []
            all_node_keys = [str(value) for value in raw_all_ids]
            unknown_node_keys = sorted(set(all_node_keys) - available_node_keys)
            if unknown_node_keys:
                errors.append(
                    f'{task_id}: grounding.node_ids contains nodes absent from '
                    f'the subgraph: {unknown_node_keys}'
                )
            for role in requirements:
                expected_keys = {
                    str(value) for value in expected_role_nodes.get(role) or []
                }
                if not expected_keys:
                    continue
                raw_role_ids = grounding.get(f'{role}_node_ids')
                explicit_role_ids = isinstance(raw_role_ids, list)
                role_node_keys = (
                    [str(value) for value in raw_role_ids]
                    if explicit_role_ids
                    else [
                        value for value in all_node_keys
                        if value in expected_keys
                    ]
                )
                unexpected_role_nodes = sorted(
                    set(role_node_keys) - expected_keys
                )
                if unexpected_role_nodes:
                    errors.append(
                        f'{task_id}: {role}_node_ids contains unexpected resolved '
                        f'nodes {unexpected_role_nodes}'
                    )
                covered_node_keys[role].extend(
                    value for value in role_node_keys if value in expected_keys
                )
                if (
                    role == 'source'
                    and root_intent.get('quantifier') == 'one'
                    and len(expected_keys) > 1
                    and len(set(role_node_keys).intersection(expected_keys)) != 1
                ):
                    errors.append(
                        f'{task_id}: quantifier one with multiple resolved source '
                        'nodes requires exactly one source node per atomic root task'
                    )
        for role in requirements:
            expected_keys = {
                str(value) for value in expected_role_nodes.get(role) or []
            }
            if not expected_keys:
                continue
            covered_keys = set(covered_node_keys.get(role) or [])
            missing_keys = sorted(expected_keys - covered_keys)
            unexpected_keys = sorted(covered_keys - expected_keys)
            if missing_keys:
                errors.append(
                    f'root {role} grounding is missing resolved subgraph nodes '
                    f'{missing_keys}'
                )
            if unexpected_keys:
                errors.append(
                    f'root {role} grounding contains unexpected subgraph nodes '
                    f'{unexpected_keys}'
                )
            duplicate_keys = sorted({
                value for value in covered_node_keys.get(role) or []
                if covered_node_keys[role].count(value) > 1
            })
            if role == 'source' and duplicate_keys:
                errors.append(
                    f'root {role} grounding repeats resolved subgraph nodes '
                    f'{duplicate_keys}'
                )
            expected_tags = {
                _semantic_key(value)
                for value in expected_role_tags.get(role) or []
                if str(value).strip()
            }
            if expected_tags and not covered_keys:
                errors.append(
                    f'root {role} grounding must reference the resolved node ids '
                    f'for tags {sorted(expected_tags)}'
                )

    tasks_by_id = {
        str(item.get('id') or f'subtask_{index}'): item
        for index, item in enumerate(subtasks, start=1)
    }
    held_type_keys: set[str] = set()
    for agent in (agent_context or {}).get('agents') or []:
        if not isinstance(agent, dict):
            continue
        for held in agent.get('inventory') or agent.get('held_objects') or []:
            if isinstance(held, dict):
                value = held.get('objectType') or held.get('type') or held.get('objectId')
            else:
                value = held
            if value:
                held_type_keys.add(_semantic_key(str(value).split('|', 1)[0]))

    def source_type_keys(item: dict[str, Any]) -> set[str]:
        grounding = item.get('grounding') if isinstance(item.get('grounding'), dict) else {}
        selector = grounding.get('source_selector') if isinstance(grounding.get('source_selector'), dict) else {}
        values = selector.get('object_types') or grounding.get('source_object_tags') or []
        if isinstance(values, str):
            values = [values]
        return {_semantic_key(value) for value in values if str(value).strip()}

    def destination_type_keys(item: dict[str, Any]) -> set[str]:
        grounding = item.get('grounding') if isinstance(item.get('grounding'), dict) else {}
        selector = grounding.get('destination_selector') if isinstance(grounding.get('destination_selector'), dict) else {}
        values = selector.get('object_types') or grounding.get('destination_object_tags') or []
        if isinstance(values, str):
            values = [values]
        return {_semantic_key(value) for value in values if str(value).strip()}

    def ancestor_ids(task_id: str) -> set[str]:
        ancestors: set[str] = set()
        pending = list(tasks_by_id.get(task_id, {}).get('depends_on') or [])
        while pending:
            dependency = str(pending.pop())
            if dependency in ancestors:
                continue
            ancestors.add(dependency)
            pending.extend(tasks_by_id.get(dependency, {}).get('depends_on') or [])
        return ancestors

    for task_id, subtask in tasks_by_id.items():
        action = normalize_logical_action(subtask.get('action'))
        intended = source_type_keys(subtask)
        requires_source_ready = bool(
            (ACTION_CONTRACTS.get(action) or {}).get('requires_held_source')
        )
        if not requires_source_ready:
            continue
        matching_pick = any(
            normalize_logical_action(tasks_by_id.get(dependency, {}).get('action')) == 'pick'
            and bool(intended & source_type_keys(tasks_by_id[dependency]))
            for dependency in ancestor_ids(task_id)
            if dependency in tasks_by_id
        )
        already_held = bool(intended & held_type_keys)
        if not intended or (not matching_pick and not already_held):
            errors.append(
                f'{task_id}: action {action!r} requires an already-held matching source '
                'or a matching ancestor pick dependency'
            )

    open_tasks_by_source: dict[str, list[str]] = defaultdict(list)
    place_tasks_by_destination: dict[str, list[str]] = defaultdict(list)
    for candidate_id, candidate in tasks_by_id.items():
        candidate_action = normalize_logical_action(candidate.get('action'))
        if candidate_action == 'open':
            for key in source_type_keys(candidate):
                open_tasks_by_source[key].append(candidate_id)
        elif candidate_action == 'place':
            for key in destination_type_keys(candidate):
                place_tasks_by_destination[key].append(candidate_id)

    for task_id, subtask in tasks_by_id.items():
        action = normalize_logical_action(subtask.get('action'))
        if action == 'place':
            ancestors = ancestor_ids(task_id)
            for destination in destination_type_keys(subtask):
                matching_open_tasks = open_tasks_by_source.get(destination) or []
                missing_open = [open_id for open_id in matching_open_tasks if open_id not in ancestors]
                if missing_open:
                    errors.append(
                        f'{task_id}: place destination requires matching open dependency {missing_open}'
                    )
        elif action == 'close':
            ancestors = ancestor_ids(task_id)
            for source in source_type_keys(subtask):
                missing_places = [
                    place_id
                    for place_id in place_tasks_by_destination.get(source, [])
                    if place_id not in ancestors
                ]
                if missing_places:
                    errors.append(
                        f'{task_id}: close action must depend on all placement tasks for {source!r}: {missing_places}'
                    )

    return {
        'status': 'valid' if not errors else 'invalid',
        'errors': unique_preserve_order(errors),
        'root_intent': root_intent,
    }


def normalize_action_name(action: str | None) -> str:
    action = normalize_logical_action(action)
    allowed = set(PLANNER_ACTIONS)
    return action if action in allowed else 'other'


def normalize_grounding_status(value: Any, *, has_nodes: bool) -> str:
    status = str(value or '').strip().lower()
    if status in {'grounded', 'partial', 'unresolved'}:
        return 'unresolved' if status == 'grounded' and not has_nodes else status
    return 'grounded' if has_nodes else 'unresolved'


def _type_words(value: Any) -> list[str]:
    separated = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', str(value or '').replace('_', ' '))
    return [
        word.lower()
        for word in re.findall(r'[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|[0-9]+', separated)
    ]


def _semantic_key(value: Any) -> str:
    return ''.join(_type_words(value))


def semantic_entity_catalog(subgraph: dict[str, Any]) -> dict[tuple[str, ...], str]:
    task_spec = subgraph.get('task_spec') if isinstance(subgraph.get('task_spec'), dict) else {}
    entities: list[str] = []
    for key in ('target_objects', 'source_receptacles', 'destination_receptacles', 'landmarks'):
        for value in task_spec.get(key) or []:
            if isinstance(value, str) and value.strip():
                entities.append(value.strip())
    for node in subgraph.get('nodes') or []:
        if not isinstance(node, dict):
            continue
        for key in ('object_tag', 'possible_tags'):
            values = node.get(key) or []
            if isinstance(values, str):
                values = [values]
            for value in values:
                if isinstance(value, str) and value.strip():
                    entities.append(value.strip())
    catalog: dict[tuple[str, ...], str] = {}
    for entity in unique_preserve_order(entities):
        tokens = tuple(_type_words(entity))
        if tokens:
            catalog.setdefault(tokens, entity)
    return catalog


def restore_compound_object_tags(
    tags: list[Any],
    catalog: dict[tuple[str, ...], str],
) -> list[str]:
    cleaned = [
        str(tag).strip()
        for tag in tags
        if str(tag).strip()
    ]
    restored: list[str] = []
    index = 0
    while index < len(cleaned):
        best_end = index + 1
        best_value = catalog.get(tuple(_type_words(cleaned[index])))
        combined: list[str] = []
        for end in range(index, min(len(cleaned), index + 4)):
            combined.extend(_type_words(cleaned[end]))
            candidate = catalog.get(tuple(combined))
            if candidate is not None:
                best_end = end + 1
                best_value = candidate
        if best_value is not None:
            restored.append(best_value)
            index = best_end
        else:
            restored.append(cleaned[index])
            index += 1
    return unique_preserve_order(restored)


def _task_spec_entities(subgraph: dict[str, Any], key: str) -> list[str]:
    task_spec = subgraph.get('task_spec') if isinstance(subgraph.get('task_spec'), dict) else {}
    return [str(value).strip() for value in task_spec.get(key) or [] if str(value).strip()]


def _role_tags_for_subtask(
    subtask_action: str,
    tags: list[str],
    grounding: dict[str, Any],
    subgraph: dict[str, Any],
) -> tuple[list[str], list[str]]:
    catalog = semantic_entity_catalog(subgraph)
    source = restore_compound_object_tags(grounding.get('source_object_tags') or [], catalog)
    destination = restore_compound_object_tags(grounding.get('destination_object_tags') or [], catalog)
    if subtask_action == 'place':
        destination_candidates = _task_spec_entities(subgraph, 'destination_receptacles')
        for tag in tags:
            if any(_semantic_key(tag) == _semantic_key(candidate) for candidate in destination_candidates):
                destination.append(tag)
        destination = unique_preserve_order(destination)
        if not destination:
            relation_text = ' '.join(str(value) for value in grounding.get('relation_texts') or [])
            for tag in tags:
                escaped = re.escape(str(tag))
                if re.search(rf'\b(?:on|onto|in|into|inside)\b.*\b{escaped}\b', relation_text, re.IGNORECASE):
                    destination.append(tag)
                    break
        if not source:
            destination_keys = {_semantic_key(tag) for tag in destination}
            source = [tag for tag in tags if _semantic_key(tag) not in destination_keys]
    elif not source:
        source = list(tags)
    return unique_preserve_order(source), unique_preserve_order(destination)


def _node_matches_entity(node: dict[str, Any], entity: str) -> bool:
    entity_key = _semantic_key(entity)
    if not entity_key:
        return False
    values: list[Any] = [node.get('object_tag'), node.get('caption')]
    values.extend(node.get('possible_tags') or [])
    return any(
        _semantic_key(value) == entity_key
        for value in values
        if value not in (None, '')
    )


def _role_node_ids(
    node_ids: list[int],
    node_lookup: dict[int, dict[str, Any]],
    entities: list[str],
) -> list[int]:
    return [
        node_id
        for node_id in node_ids
        if node_id in node_lookup
        and any(_node_matches_entity(node_lookup[node_id], entity) for entity in entities)
    ]


def normalize_scene_selector(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        return {"quantifier": "", "object_types": []}
    raw_types = value.get("object_types") or []
    if isinstance(raw_types, str):
        raw_types = [raw_types]
    object_types = unique_preserve_order([
        str(item) for item in raw_types if str(item).strip()
    ]) if isinstance(raw_types, list) else []
    return {
        "quantifier": str(value.get("quantifier") or "").strip().lower(),
        "object_types": object_types,
    }


def sanitize_subtasks(
    raw_plan: dict,
    subgraph: dict,
    task: str,
    *,
    allow_semantic_inference: bool = True,
) -> dict:
    node_lookup = build_node_lookup(subgraph)
    triples = build_triple_lookup(subgraph)
    entity_catalog = (
        semantic_entity_catalog(subgraph) if allow_semantic_inference else {}
    )
    raw_subtasks = list(raw_plan.get('subtasks') or [])
    if not raw_subtasks:
        raise TaskPlanningError(
            'Planner output has no subtasks; refusing to synthesize a fallback plan.',
            diagnostics={
                'task': task,
                'status': 'failed',
                'stage': 'sanitize_subtasks',
                'validation_errors': ['planner response must contain a non-empty subtasks list'],
            },
        )

    normalized = []
    seen_ids = set()
    for index, subtask in enumerate(raw_subtasks, start=1):
        if not isinstance(subtask, dict):
            continue
        task_id = str(subtask.get('id') or f'T{index}').strip() or f'T{index}'
        if task_id in seen_ids:
            task_id = f'T{index}'
        seen_ids.add(task_id)

        grounding = subtask.get('grounding') or {}
        raw_node_ids = grounding.get('node_ids') or []
        raw_source_node_ids = grounding.get('source_node_ids')
        raw_destination_node_ids = grounding.get('destination_node_ids')
        has_explicit_source_node_ids = isinstance(raw_source_node_ids, list)
        has_explicit_destination_node_ids = isinstance(raw_destination_node_ids, list)

        def valid_node_ids(values: Any) -> list[int]:
            if not isinstance(values, list):
                return []
            valid: list[int] = []
            for node_id in values:
                try:
                    node_id_int = int(node_id)
                except (TypeError, ValueError):
                    continue
                if node_id_int in node_lookup:
                    valid.append(node_id_int)
            return list(dict.fromkeys(valid))

        node_ids = valid_node_ids(raw_node_ids)
        explicit_source_node_ids = valid_node_ids(raw_source_node_ids)
        explicit_destination_node_ids = valid_node_ids(raw_destination_node_ids)
        node_ids = list(dict.fromkeys([
            *node_ids, *explicit_source_node_ids, *explicit_destination_node_ids,
        ]))

        raw_object_tags = grounding.get('object_tags') or []
        if isinstance(raw_object_tags, str):
            raw_object_tags = [raw_object_tags]
        object_tags = (
            restore_compound_object_tags(raw_object_tags, entity_catalog)
            if allow_semantic_inference
            else unique_preserve_order([
                str(value).strip()
                for value in raw_object_tags
                if str(value).strip()
            ])
        )
        if allow_semantic_inference and not object_tags and not node_ids:
            inferred_tags = infer_task_object_tags(
                ' '.join([
                    task,
                    str(subtask.get('name') or ''),
                    str(subtask.get('description') or ''),
                ])
            )
            object_tags = restore_compound_object_tags(inferred_tags, entity_catalog)
        relation_texts = unique_preserve_order([str(text) for text in grounding.get('relation_texts') or []])
        if allow_semantic_inference and not relation_texts and triples and node_ids:
            for triple in triples:
                if triple.get('source') in node_ids or triple.get('target') in node_ids:
                    relation_texts.append(str(triple.get('text', '')))
                if len(relation_texts) >= 4:
                    break

        action_name = normalize_action_name(subtask.get('action'))
        if allow_semantic_inference and action_name == 'other':
            semantic_actions = infer_actions(' '.join([
                str(subtask.get('name') or ''),
                str(subtask.get('description') or ''),
            ]))
            if 'slice' in semantic_actions:
                action_name = 'slice'
            elif 'find' in semantic_actions:
                action_name = 'find'
        action_args, action_arg_errors = validate_action_args(
            action_name, subtask.get('action_args')
        )
        if action_arg_errors:
            raise TaskPlanningError(
                f'Planner action arguments became invalid for {task_id}.',
                diagnostics={
                    'task': task,
                    'status': 'failed',
                    'stage': 'sanitize_subtasks',
                    'validation_errors': [item['message'] for item in action_arg_errors],
                },
            )
        if allow_semantic_inference:
            source_tags, destination_tags = _role_tags_for_subtask(
                action_name,
                object_tags,
                grounding,
                subgraph,
            )
        else:
            raw_source_tags = grounding.get('source_object_tags') or []
            raw_destination_tags = grounding.get('destination_object_tags') or []
            if isinstance(raw_source_tags, str):
                raw_source_tags = [raw_source_tags]
            if isinstance(raw_destination_tags, str):
                raw_destination_tags = [raw_destination_tags]
            source_tags = unique_preserve_order([
                str(value).strip()
                for value in raw_source_tags
                if str(value).strip()
            ])
            destination_tags = unique_preserve_order([
                str(value).strip()
                for value in raw_destination_tags
                if str(value).strip()
            ])
        source_selector = normalize_scene_selector(grounding.get('source_selector'))
        destination_selector = normalize_scene_selector(grounding.get('destination_selector'))
        if source_selector and source_selector.get('object_types'):
            source_tags = unique_preserve_order([*source_tags, *source_selector['object_types']])
        if destination_selector and destination_selector.get('object_types'):
            destination_tags = unique_preserve_order([*destination_tags, *destination_selector['object_types']])
        source_node_ids = (
            explicit_source_node_ids
            if has_explicit_source_node_ids
            else _role_node_ids(node_ids, node_lookup, source_tags)
        )
        destination_node_ids = (
            explicit_destination_node_ids
            if has_explicit_destination_node_ids
            else _role_node_ids(node_ids, node_lookup, destination_tags)
        )
        if action_name == 'place' and source_tags and destination_tags:
            if source_node_ids and destination_node_ids:
                grounding_status = 'grounded'
            elif source_node_ids or destination_node_ids:
                grounding_status = 'partial'
            else:
                grounding_status = 'unresolved'

        depends_on = unique_preserve_order([str(dep) for dep in subtask.get('depends_on') or []])
        depends_on = [dep for dep in depends_on if dep != task_id]
        grounding_status = normalize_grounding_status(grounding.get('status'), has_nodes=bool(node_ids))
        missing_reason = str(grounding.get('missing_reason') or '').strip()
        recovery = str(grounding.get('recovery') or '').strip()
        if grounding_status == 'unresolved':
            missing_reason = missing_reason or 'No valid scene-graph node is currently grounded for this subtask.'
            recovery = recovery or 'search_visible_scene'
        else:
            recovery = recovery or 'none'

        normalized.append({
            'id': task_id,
            'name': str(subtask.get('name') or f'Subtask {index}').strip(),
            'description': str(subtask.get('description') or '').strip(),
            'action': action_name,
            'action_args': action_args,
            'grounding': {
                'node_ids': node_ids,
                'object_tags': object_tags,
                'source_object_tags': source_tags,
                'destination_object_tags': destination_tags,
                'source_node_ids': source_node_ids,
                'destination_node_ids': destination_node_ids,
                'source_object_ids': list(grounding.get('source_object_ids') or []),
                'destination_object_ids': list(grounding.get('destination_object_ids') or []),
                'source_selector': source_selector,
                'destination_selector': destination_selector,
                'relation_texts': relation_texts,
                'status': grounding_status,
                'missing_reason': missing_reason,
                'recovery': recovery,
            },
            'depends_on': depends_on,
            'termination_check': str(subtask.get('termination_check') or '').strip(),
        })

    if not normalized:
        raise TaskPlanningError(
            'Planner output contains no valid subtask objects.',
            diagnostics={
                'task': task,
                'status': 'failed',
                'stage': 'sanitize_subtasks',
                'validation_errors': ['all planner subtasks were invalid'],
            },
        )

    valid_ids = {subtask['id'] for subtask in normalized}
    for subtask in normalized:
        subtask['depends_on'] = [dep for dep in subtask['depends_on'] if dep in valid_ids]

    normalized = break_dependency_cycles(normalized)
    return {
        'reasoning_summary': str(raw_plan.get('reasoning_summary') or ''),
        'planner_backend': str(raw_plan.get('planner_backend') or 'qwen_local'),
        'subtasks': normalized,
    }


def break_dependency_cycles(subtasks: list[dict]) -> list[dict]:
    id_to_task = {task['id']: task for task in subtasks}
    visiting = set()
    visited = set()

    def dfs(task_id: str):
        if task_id in visited:
            return
        if task_id in visiting:
            return
        visiting.add(task_id)
        task = id_to_task[task_id]
        cleaned = []
        for dep in task['depends_on']:
            if dep in visiting:
                continue
            dfs(dep)
            cleaned.append(dep)
        task['depends_on'] = cleaned
        visiting.remove(task_id)
        visited.add(task_id)

    for task in subtasks:
        dfs(task['id'])
    return subtasks


def topological_order(subtasks: list[dict]) -> list[str]:
    predecessors = {task['id']: set(task['depends_on']) for task in subtasks}
    successors = defaultdict(set)
    indegree = {}
    for task in subtasks:
        task_id = task['id']
        indegree[task_id] = len(predecessors[task_id])
        for dep in predecessors[task_id]:
            successors[dep].add(task_id)

    queue = deque(sorted([task_id for task_id, degree in indegree.items() if degree == 0]))
    order = []
    while queue:
        task_id = queue.popleft()
        order.append(task_id)
        for succ in sorted(successors.get(task_id, [])):
            indegree[succ] -= 1
            if indegree[succ] == 0:
                queue.append(succ)
    if len(order) != len(subtasks):
        remaining = [task['id'] for task in subtasks if task['id'] not in order]
        order.extend(sorted(remaining))
    return order


def build_chain_ids(subtasks: list[dict]) -> list[list[str]]:
    order = topological_order(subtasks)
    task_index = {task['id']: task for task in subtasks}
    predecessors = {task['id']: list(task['depends_on']) for task in subtasks}
    successors = defaultdict(list)
    for task in subtasks:
        for dep in task['depends_on']:
            successors[dep].append(task['id'])

    for task_id in successors:
        successors[task_id] = sorted(successors[task_id], key=lambda item: order.index(item))

    visited = set()
    chains = []

    def grow_chain(start_id: str) -> list[str]:
        chain = []
        current_id = start_id
        while current_id not in visited:
            chain.append(current_id)
            visited.add(current_id)
            succs = [succ for succ in successors.get(current_id, []) if succ not in visited]
            if len(succs) != 1:
                break
            next_id = succs[0]
            if len(predecessors.get(next_id, [])) != 1:
                break
            current_id = next_id
        return chain

    roots = [task_id for task_id in order if len(predecessors.get(task_id, [])) == 0]
    for root_id in roots:
        if root_id not in visited:
            chains.append(grow_chain(root_id))

    for task_id in order:
        if task_id not in visited:
            chains.append(grow_chain(task_id))

    return chains


def build_linked_node(task_id: str, task_index: dict[str, dict], next_node: dict | None) -> dict:
    task = deepcopy(task_index[task_id])
    task['next'] = next_node
    return task


def chain_to_linked_list(chain_task_ids: list[str], task_index: dict[str, dict]) -> dict | None:
    next_node = None
    for task_id in reversed(chain_task_ids):
        next_node = build_linked_node(task_id, task_index, next_node)
    return next_node


def build_dependency_edges(subtasks: list[dict]) -> list[dict]:
    edges = []
    for task in subtasks:
        for dep in task['depends_on']:
            edges.append({'from': dep, 'to': task['id']})
    return edges


def rebuild_task_graph_views(task_graph: dict[str, Any]) -> dict[str, Any]:
    """Rebuild every DAG-derived view from canonical ``flat_tasks``.

    Runtime compilers may clone or rewire flat tasks.  Keeping the derived edge,
    root, and chain views from the pre-compiled graph would leave consumers with
    mutually inconsistent representations.  This helper preserves graph-level
    metadata and task payloads while recomputing only those derived views.
    """

    rebuilt = deepcopy(task_graph)
    subtasks = [
        deepcopy(task)
        for task in rebuilt.get('flat_tasks') or []
        if isinstance(task, dict)
    ]
    order = topological_order(subtasks)
    task_index = {task['id']: deepcopy(task) for task in subtasks}
    chain_ids = build_chain_ids(subtasks)

    chains: list[dict[str, Any]] = []
    for chain_index, chain_task_ids in enumerate(chain_ids, start=1):
        chain_id = f'C{chain_index}'
        for task_id in chain_task_ids:
            task_index[task_id]['chain_id'] = chain_id
        head_id = chain_task_ids[0]
        chains.append({
            'chain_id': chain_id,
            'head_task_id': head_id,
            'head_depends_on': list(task_index[head_id].get('depends_on', [])),
            'task_ids': chain_task_ids,
            'linked_list': chain_to_linked_list(chain_task_ids, task_index),
        })

    flat_tasks = []
    for task_id in order:
        task = deepcopy(task_index[task_id])
        flat_tasks.append(task)

    rebuilt['flat_tasks'] = flat_tasks
    rebuilt['dependency_edges'] = build_dependency_edges(flat_tasks)
    rebuilt['root_task_ids'] = [
        task['id'] for task in flat_tasks if not task.get('depends_on')
    ]
    rebuilt['chains'] = chains
    return rebuilt


def _semantic_entity_tokens(value: Any) -> list[str]:
    return _type_words(value)


def _subtask_semantic_text(subtask: dict[str, Any]) -> str:
    grounding = subtask.get('grounding') if isinstance(subtask.get('grounding'), dict) else {}
    return ' '.join([
        str(subtask.get('name') or ''),
        str(subtask.get('description') or ''),
        str(subtask.get('termination_check') or ''),
        ' '.join(str(tag) for tag in grounding.get('object_tags') or []),
        ' '.join(str(tag) for tag in grounding.get('source_object_tags') or []),
        ' '.join(str(tag) for tag in grounding.get('destination_object_tags') or []),
        ' '.join(str(text) for text in grounding.get('relation_texts') or []),
    ])


def _mentions_semantic_entity(text: str, entity: str) -> bool:
    entity_tokens = _semantic_entity_tokens(entity)
    if not entity_tokens:
        return False
    normalized_text = re.sub(r'[^a-z0-9]+', ' ', str(text).lower())
    return all(
        re.search(rf'\b{re.escape(token)}\b', normalized_text)
        for token in entity_tokens
    )


def _requests_receptacle_left_open(task: str) -> bool:
    normalized = re.sub(r'\s+', ' ', task.lower())
    return bool(re.search(
        r"\b(?:leave|keep)\b.{0,40}\bopen\b"
        r"|\b(?:do not|don't|dont|never)\s+close\b"
        r"|\bwithout\s+clos(?:e|ing)\b",
        normalized,
    ))


def _next_generated_task_id(subtasks: list[dict[str, Any]]) -> str:
    existing_ids = {str(subtask.get('id') or '') for subtask in subtasks}
    numeric_ids = [
        int(match.group(1))
        for task_id in existing_ids
        if (match := re.fullmatch(r'T(\d+)', task_id))
    ]
    candidate = max(numeric_ids, default=0) + 1
    while f'T{candidate}' in existing_ids:
        candidate += 1
    return f'T{candidate}'


def _entity_span(text: str, entity: str) -> tuple[int, int] | None:
    normalized_text = re.sub(r'[^a-z0-9]+', ' ', str(text).lower()).strip()
    normalized_entity = re.sub(r'[^a-z0-9]+', ' ', str(entity).lower()).strip()
    if not normalized_text or not normalized_entity:
        return None
    match = re.search(rf'\b{re.escape(normalized_entity)}\b', normalized_text)
    return match.span() if match else None


def _placement_pairs(subtask: dict[str, Any]) -> list[tuple[str, str]]:
    """Extract unambiguous source/destination pairs from placement relations."""

    grounding = subtask.get('grounding') if isinstance(subtask.get('grounding'), dict) else {}
    tags = unique_preserve_order([
        str(tag).strip()
        for tag in grounding.get('object_tags') or []
        if str(tag).strip()
    ])
    if len(tags) < 3:
        return []

    pairs: list[tuple[str, str]] = []
    for relation in grounding.get('relation_texts') or []:
        normalized_relation = re.sub(r'[^a-z0-9]+', ' ', str(relation).lower()).strip()
        if not normalized_relation:
            continue
        spans = {
            tag: span
            for tag in tags
            if (span := _entity_span(normalized_relation, tag)) is not None
        }
        for source in tags:
            source_span = spans.get(source)
            if source_span is None:
                continue
            for destination in tags:
                if destination == source:
                    continue
                destination_span = spans.get(destination)
                if destination_span is None:
                    continue
                if source_span[1] <= destination_span[0]:
                    connector = normalized_relation[source_span[1]:destination_span[0]]
                    if re.search(r'\b(?:in|inside|into|on|onto)\b', connector):
                        pair = (source, destination)
                        if pair not in pairs:
                            pairs.append(pair)
                elif destination_span[1] <= source_span[0]:
                    connector = normalized_relation[destination_span[1]:source_span[0]]
                    if re.search(r'\b(?:contain|contains|holding|holds)\b', connector):
                        pair = (source, destination)
                        if pair not in pairs:
                            pairs.append(pair)
    return pairs


def split_multi_object_place_subtasks(
    normalized_plan: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Split unambiguous multi-object placements into relay-executable atomic tasks."""

    subtasks = [
        deepcopy(subtask)
        for subtask in normalized_plan.get('subtasks') or []
        if isinstance(subtask, dict)
    ]
    if not subtasks:
        return normalized_plan, []

    task_index = {str(subtask.get('id')): subtask for subtask in subtasks}
    replacements: dict[str, list[str]] = {}
    repaired_subtasks: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []

    for subtask in subtasks:
        task_id = str(subtask.get('id') or '')
        grounding = subtask.get('grounding') if isinstance(subtask.get('grounding'), dict) else {}
        source_selector = grounding.get('source_selector') if isinstance(grounding.get('source_selector'), dict) else {}
        if subtask.get('action') == 'place' and source_selector.get('quantifier') == 'all':
            repaired_subtasks.append(subtask)
            continue
        pairs = _placement_pairs(subtask) if subtask.get('action') == 'place' else []
        sources = unique_preserve_order([source for source, _ in pairs])
        destinations = unique_preserve_order([destination for _, destination in pairs])
        if len(sources) < 2 or len(destinations) != 1:
            repaired_subtasks.append(subtask)
            continue

        destination = destinations[0]
        source_pairs = [(source, target) for source, target in pairs if target == destination]
        if len(source_pairs) < 2:
            repaired_subtasks.append(subtask)
            continue

        generated_ids: list[str] = []
        placements: list[dict[str, str]] = []
        for pair_index, (source, target) in enumerate(source_pairs):
            atomic = deepcopy(subtask)
            if pair_index:
                atomic['id'] = _next_generated_task_id([*subtasks, *repaired_subtasks])
            atomic_id = str(atomic.get('id'))
            generated_ids.append(atomic_id)
            placements.append({'source': source, 'destination': target})

            original_dependencies = [
                str(dep)
                for dep in subtask.get('depends_on') or []
                if str(dep).strip()
            ]
            relevant_dependencies = [
                dep
                for dep in original_dependencies
                if dep in task_index
                and (
                    _mentions_semantic_entity(_subtask_semantic_text(task_index[dep]), source)
                    or _mentions_semantic_entity(_subtask_semantic_text(task_index[dep]), target)
                )
            ]
            atomic_grounding = deepcopy(
                atomic.get('grounding') if isinstance(atomic.get('grounding'), dict) else {}
            )
            atomic_grounding['object_tags'] = [source, target]
            atomic_grounding['source_object_tags'] = [source]
            atomic_grounding['destination_object_tags'] = [target]
            atomic_grounding['source_node_ids'] = []
            atomic_grounding['destination_node_ids'] = []
            atomic_grounding['relation_texts'] = [f'{source} is in {target}']
            atomic.update({
                'name': f'Place {source} in {target}',
                'description': f'Pick up the {source} and place it inside the {target}.',
                'grounding': atomic_grounding,
                'depends_on': relevant_dependencies or original_dependencies,
                'termination_check': f'{source} is confirmed to be inside {target}.',
            })
            repaired_subtasks.append(atomic)

        replacements[task_id] = generated_ids
        repairs.append({
            'kind': 'split_multi_object_place',
            'source_task_id': task_id,
            'task_ids': generated_ids,
            'placements': placements,
        })

    if not repairs:
        return normalized_plan, []

    for subtask in repaired_subtasks:
        expanded_dependencies: list[str] = []
        for dependency in subtask.get('depends_on') or []:
            expanded_dependencies.extend(replacements.get(str(dependency), [str(dependency)]))
        subtask['depends_on'] = unique_preserve_order(expanded_dependencies)

    repaired_plan = deepcopy(normalized_plan)
    repaired_plan['subtasks'] = break_dependency_cycles(repaired_subtasks)
    return repaired_plan, repairs



def split_multi_entity_find_subtasks(
    normalized_plan: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    subtasks = [deepcopy(item) for item in normalized_plan.get('subtasks') or [] if isinstance(item, dict)]
    if not subtasks:
        return normalized_plan, []

    replacements: dict[str, list[tuple[str, str]]] = {}
    repaired_subtasks: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    for subtask in subtasks:
        task_id = str(subtask.get('id') or '')
        grounding = subtask.get('grounding') if isinstance(subtask.get('grounding'), dict) else {}
        has_scene_selector = any(
            isinstance(grounding.get(f'{role}_selector'), dict)
            and bool(grounding[f'{role}_selector'].get('object_types') or [])
            for role in ('source', 'destination')
        )
        if subtask.get('action') in {'find', 'inspect'} and has_scene_selector:
            repaired_subtasks.append(subtask)
            continue

        entities = unique_preserve_order([str(tag) for tag in grounding.get('source_object_tags') or grounding.get('object_tags') or []])
        if subtask.get('action') not in {'find', 'inspect'} or len(entities) < 2:
            repaired_subtasks.append(subtask)
            continue

        generated: list[tuple[str, str]] = []
        for entity_index, entity in enumerate(entities):
            atomic = deepcopy(subtask)
            if entity_index:
                atomic['id'] = _next_generated_task_id([*subtasks, *repaired_subtasks])
            atomic_id = str(atomic.get('id'))
            generated.append((entity, atomic_id))
            atomic_grounding = deepcopy(grounding)
            atomic_grounding.update({
                'node_ids': [],
                'object_tags': [entity],
                'source_object_tags': [entity],
                'destination_object_tags': [],
                'source_node_ids': [],
                'destination_node_ids': [],
            })
            atomic.update({
                'name': f'Find {entity}',
                'description': f'Locate the {entity} and verify that it is actionable.',
                'grounding': atomic_grounding,
                'termination_check': f'{entity} is observed and individually grounded.',
            })
            repaired_subtasks.append(atomic)
        replacements[task_id] = generated
        repairs.append({'kind': 'split_multi_entity_find', 'source_task_id': task_id, 'tasks': [{'entity': entity, 'task_id': atomic_id} for entity, atomic_id in generated]})

    if not repairs:
        return normalized_plan, []

    for subtask in repaired_subtasks:
        expanded_dependencies: list[str] = []
        semantic_text = _subtask_semantic_text(subtask)
        for dependency in subtask.get('depends_on') or []:
            choices = replacements.get(str(dependency))
            if not choices:
                expanded_dependencies.append(str(dependency))
                continue
            matching = [atomic_id for entity, atomic_id in choices if _mentions_semantic_entity(semantic_text, entity)]
            expanded_dependencies.extend(matching or [atomic_id for _, atomic_id in choices])
        subtask['depends_on'] = unique_preserve_order(expanded_dependencies)

    repaired_plan = deepcopy(normalized_plan)
    repaired_plan['subtasks'] = break_dependency_cycles(repaired_subtasks)
    return repaired_plan, repairs


def remove_redundant_slice_tool_subtasks(
    normalized_plan: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Let the executor own cutting-tool acquisition for atomic slice goals."""

    subtasks = [deepcopy(item) for item in normalized_plan.get('subtasks') or [] if isinstance(item, dict)]
    if not any(item.get('action') == 'slice' for item in subtasks):
        return normalized_plan, []

    cutting_tool_keys = {_semantic_key('knife'), _semantic_key('butter knife')}
    removable: dict[str, dict[str, Any]] = {}
    for item in subtasks:
        if item.get('action') not in {'find', 'inspect', 'navigate', 'pick'}:
            continue
        grounding = item.get('grounding') if isinstance(item.get('grounding'), dict) else {}
        tags = unique_preserve_order([
            str(tag) for tag in grounding.get('source_object_tags') or grounding.get('object_tags') or []
        ])
        if tags and all(_semantic_key(tag) in cutting_tool_keys for tag in tags):
            removable[str(item.get('id'))] = item

    if not removable:
        return normalized_plan, []

    def expanded_dependencies(dependencies: list[Any]) -> list[str]:
        output: list[str] = []
        pending = [str(value) for value in dependencies]
        seen: set[str] = set()
        while pending:
            dependency = pending.pop(0)
            if dependency in seen:
                continue
            seen.add(dependency)
            removed = removable.get(dependency)
            if removed is None:
                output.append(dependency)
            else:
                pending.extend(str(value) for value in removed.get('depends_on') or [])
        return unique_preserve_order(output)

    kept = []
    for item in subtasks:
        if str(item.get('id')) in removable:
            continue
        item['depends_on'] = expanded_dependencies(item.get('depends_on') or [])
        kept.append(item)

    repaired_plan = deepcopy(normalized_plan)
    repaired_plan['subtasks'] = break_dependency_cycles(kept)
    repairs = [{
        'kind': 'remove_redundant_slice_tool_subtask',
        'task_id': task_id,
        'action': item.get('action'),
    } for task_id, item in removable.items()]
    return repaired_plan, repairs


def ensure_open_receptacles_closed(
    task: str,
    normalized_plan: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Append missing final close steps without altering existing planner steps.

    The repair is deliberately narrow: an open subtask, at least one related
    place subtask, and no related close subtask must all be present. Explicit
    instructions to leave a receptacle open disable the repair.
    """

    if _requests_receptacle_left_open(task):
        return normalized_plan, []

    subtasks = [
        deepcopy(subtask)
        for subtask in normalized_plan.get('subtasks') or []
        if isinstance(subtask, dict)
    ]
    open_tasks = [subtask for subtask in subtasks if subtask.get('action') == 'open']
    place_tasks = [subtask for subtask in subtasks if subtask.get('action') == 'place']
    close_tasks = [subtask for subtask in subtasks if subtask.get('action') == 'close']
    if not open_tasks or not place_tasks:
        return normalized_plan, []

    repairs: list[dict[str, Any]] = []
    repaired_targets: set[tuple[str, ...]] = set()
    for open_task in open_tasks:
        grounding = open_task.get('grounding') if isinstance(open_task.get('grounding'), dict) else {}
        tags = unique_preserve_order([
            str(tag)
            for tag in grounding.get('object_tags') or []
            if str(tag).strip()
        ])
        explicit_open_text = ' '.join([
            str(open_task.get('name') or ''),
            str(open_task.get('description') or ''),
            str(open_task.get('termination_check') or ''),
        ])
        target_tags = [tag for tag in tags if _mentions_semantic_entity(explicit_open_text, tag)]
        if not target_tags and len(tags) == 1:
            target_tags = tags

        for target_tag in target_tags:
            target_key = tuple(_semantic_entity_tokens(target_tag))
            if not target_key or target_key in repaired_targets:
                continue
            related_places = [
                place_task
                for place_task in place_tasks
                if _mentions_semantic_entity(_subtask_semantic_text(place_task), target_tag)
            ]
            if not related_places:
                continue
            if any(
                _mentions_semantic_entity(_subtask_semantic_text(close_task), target_tag)
                for close_task in close_tasks
            ):
                repaired_targets.add(target_key)
                continue

            close_id = _next_generated_task_id(subtasks)
            close_grounding = deepcopy(grounding)
            close_grounding['object_tags'] = [target_tag]
            close_grounding['relation_texts'] = unique_preserve_order([
                *[str(text) for text in close_grounding.get('relation_texts') or []],
                f'close {target_tag} after all related placements',
            ])
            dependency_ids = unique_preserve_order([
                str(place_task.get('id'))
                for place_task in related_places
                if str(place_task.get('id') or '').strip()
            ])
            close_task = {
                'id': close_id,
                'name': f'Close {target_tag}',
                'description': f'Close the {target_tag} after all objects have been placed inside.',
                'action': 'close',
                'grounding': close_grounding,
                'depends_on': dependency_ids,
                'termination_check': f'{target_tag} is closed.',
            }
            subtasks.append(close_task)
            close_tasks.append(close_task)
            repaired_targets.add(target_key)
            repairs.append({
                'kind': 'append_missing_close',
                'task_id': close_id,
                'target': target_tag,
                'depends_on': dependency_ids,
                'source_open_task_id': open_task.get('id'),
            })

    if not repairs:
        return normalized_plan, []
    repaired_plan = deepcopy(normalized_plan)
    repaired_plan['subtasks'] = break_dependency_cycles(subtasks)
    return repaired_plan, repairs


def assemble_task_graph(task: str, subgraph: dict, normalized_plan: dict) -> dict:
    subtasks = normalized_plan['subtasks']
    order = topological_order(subtasks)
    task_index = {task_item['id']: deepcopy(task_item) for task_item in subtasks}
    chains_ids = build_chain_ids(subtasks)

    chain_lookup = {}
    chains = []
    for chain_index, chain_task_ids in enumerate(chains_ids, start=1):
        chain_id = f'C{chain_index}'
        for task_id in chain_task_ids:
            chain_lookup[task_id] = chain_id
        head_id = chain_task_ids[0]
        chains.append({
            'chain_id': chain_id,
            'head_task_id': head_id,
            'head_depends_on': list(task_index[head_id].get('depends_on', [])),
            'task_ids': chain_task_ids,
            'linked_list': chain_to_linked_list(chain_task_ids, task_index),
        })

    flat_tasks = []
    for task_id in order:
        task_item = deepcopy(task_index[task_id])
        task_item['chain_id'] = chain_lookup.get(task_id)
        flat_tasks.append(task_item)

    return {
        'task': task,
        'planner_backend': normalized_plan['planner_backend'],
        'reasoning_summary': normalized_plan.get('reasoning_summary', ''),
        'subgraph_summary': summarize_subgraph(subgraph),
        'flat_tasks': flat_tasks,
        'dependency_edges': build_dependency_edges(subtasks),
        'root_task_ids': [task_item['id'] for task_item in flat_tasks if not task_item.get('depends_on')],
        'chains': chains,
    }


def _prepare_planner_candidate(
    raw_plan: dict[str, Any],
    *,
    task: str,
    subgraph: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Normalize one validated model response for compiler and runtime checks."""

    candidate = deepcopy(raw_plan)
    candidate['planner_backend'] = 'qwen_local'
    normalized_plan = sanitize_subtasks(
        candidate,
        subgraph,
        task,
        allow_semantic_inference=False,
    )
    normalized_plan, atomic_find_repairs = split_multi_entity_find_subtasks(normalized_plan)
    normalized_plan, atomic_place_repairs = split_multi_object_place_subtasks(normalized_plan)
    normalized_plan, slice_tool_repairs = remove_redundant_slice_tool_subtasks(normalized_plan)
    normalized_plan, completion_repairs = ensure_open_receptacles_closed(task, normalized_plan)
    return assemble_task_graph(task, subgraph, normalized_plan), {
        'atomic_find_repairs': atomic_find_repairs,
        'atomic_place_repairs': atomic_place_repairs,
        'slice_tool_repairs': slice_tool_repairs,
        'completion_repairs': completion_repairs,
    }


def _legacy_resolve_semantic_goal_for_planning(
    *,
    task: str,
    subgraph: dict[str, Any],
    scene_catalog_summary: list[dict[str, Any]],
    model_path: str | None,
    conv_mode: str,
    num_gpus: int,
    qwen_chat: Any,
    planning_max_new_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str | None]:
    """Resolve unresolved source entities before entering planner retries."""

    root_intent = derive_root_intent(task, subgraph, scene_catalog_summary)
    roles = (
        root_intent.get('roles')
        if isinstance(root_intent.get('roles'), dict)
        else {}
    )
    semantic_goal = {
        'quantifier': str(root_intent.get('quantifier') or 'one'),
        'required_source_types': list(roles.get('source') or []),
        'destination_types': list(roles.get('destination') or []),
        'resolution_source': 'deterministic_exact',
    }
    request = semantic_goal_resolution_request(
        task=task,
        subgraph=subgraph,
        root_intent=root_intent,
        scene_catalog_summary=scene_catalog_summary,
    )
    diagnostics: dict[str, Any] = {
        'attempted': request is not None,
        'status': 'not_required' if request is None else 'started',
        'stage': 'not_required' if request is None else 'semantic_goal_resolution',
        'reason': (
            'all required source types were matched exactly'
            if request is None
            else 'unresolved source entity'
        ),
        'model_path': str(model_path or ''),
        'max_attempts': 2,
        'attempt_count': 0,
        'reused_across_planning_attempts': request is not None,
        'request': deepcopy(request) if request is not None else None,
        'attempts': [],
        'result': None,
    }
    if request is None:
        diagnostics['result'] = deepcopy(semantic_goal)
        return root_intent, semantic_goal, diagnostics, None

    candidates = list(request.get('candidate_types') or [])
    if not candidates:
        error = 'semantic goal resolver has no executable source candidates'
        diagnostics.update({
            'status': 'failed',
            'stage': 'candidate_filtering',
            'error': error,
        })
        return root_intent, semantic_goal, diagnostics, error

    correction_errors: list[str] | None = None
    resolver_max_new_tokens = min(max(int(planning_max_new_tokens), 1), 1024)
    for attempt_number in range(1, 3):
        attempt_diagnostics: dict[str, Any] = {'attempt': attempt_number}
        raw_result = call_local_qwen_for_semantic_goal_resolution(
            request=request,
            model_path=model_path,
            conv_mode=conv_mode,
            num_gpus=num_gpus,
            qwen_chat=qwen_chat,
            max_new_tokens=resolver_max_new_tokens,
            correction_errors=correction_errors,
            diagnostics=attempt_diagnostics,
        )
        if raw_result is None:
            if attempt_diagnostics.get('stage') == 'model_call':
                protocol_errors = [
                    'resolver model call failed: '
                    f"{attempt_diagnostics.get('error_type')}: "
                    f"{attempt_diagnostics.get('error_message')}"
                ]
            elif attempt_diagnostics.get('json_parse_status') == 'invalid_json':
                protocol_errors = ['resolver response is not valid JSON']
            else:
                protocol_errors = [
                    'resolver failed during '
                    f"{attempt_diagnostics.get('stage') or 'unknown stage'}"
                ]
            validated = {'protocol_status': 'invalid', 'errors': protocol_errors}
        else:
            validated = validate_semantic_goal_resolution(raw_result, candidates)
            protocol_errors = list(validated.get('errors') or [])
        attempt_diagnostics['protocol_validation'] = deepcopy(validated)
        diagnostics['attempts'].append(deepcopy(attempt_diagnostics))
        diagnostics['attempt_count'] = attempt_number
        if validated.get('protocol_status') == 'valid':
            diagnostics['result'] = deepcopy(validated)
            if validated.get('status') == 'no_match':
                error = (
                    'semantic goal resolver found no matching executable scene type '
                    f"for {request.get('semantic_phrase')!r}"
                )
                diagnostics.update({
                    'status': 'failed',
                    'stage': 'no_match',
                    'error': error,
                })
                return root_intent, semantic_goal, diagnostics, error
            resolved_sources = unique_preserve_order([
                *[str(value) for value in request.get('already_resolved_types') or []],
                *[str(value) for value in validated.get('included_types') or []],
            ])
            root_intent = deepcopy(root_intent)
            root_intent.setdefault('roles', {})['source'] = resolved_sources
            root_intent['semantic_goal_resolved'] = True
            semantic_goal = {
                'semantic_phrase': str(request.get('semantic_phrase') or ''),
                'quantifier': str(request.get('quantifier') or 'one'),
                'required_source_types': resolved_sources,
                'destination_types': list(
                    (root_intent.get('roles') or {}).get('destination') or []
                ),
                'resolution_source': 'qwen_semantic_goal_resolver',
            }
            diagnostics.update({
                'status': 'success',
                'stage': 'complete',
                'result': deepcopy(validated),
                'effective_semantic_goal': deepcopy(semantic_goal),
            })
            return root_intent, semantic_goal, diagnostics, None
        correction_errors = protocol_errors

    error = (
        'semantic goal resolver failed protocol validation after 2 attempts: '
        + '; '.join(correction_errors or ['unknown resolver error'])
    )
    diagnostics.update({
        'status': 'failed',
        'stage': 'attempts_exhausted',
        'error': error,
    })
    return root_intent, semantic_goal, diagnostics, error


def resolve_semantic_goal_for_planning(
    *,
    task: str,
    subgraph: dict[str, Any],
    scene_catalog_summary: list[dict[str, Any]],
    model_path: str | None,
    conv_mode: str,
    num_gpus: int,
    qwen_chat: Any,
    planning_max_new_tokens: int,
    semantic_correction: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str | None]:
    """Resolve one root action and all required roles before planner retries."""

    root_intent: dict[str, Any] = {
        'action': None,
        'explicit_actions': [],
        'quantifier': 'one',
        'roles': {},
        'role_quantifiers': {},
        'role_statuses': {},
        'role_nodes': {},
        'role_tags': {},
        'grounding_mode': (
            'catalogue' if scene_catalog_summary else 'subgraph'
        ),
        'semantic_args': {},
        'catalogue_constrained': bool(scene_catalog_summary),
        'semantic_goal_resolved': False,
        'intent_resolution_source': 'qwen_two_stage_intent',
    }
    semantic_goal: dict[str, Any] = {
        'action': None,
        'quantifier': 'one',
        'required_source_types': [],
        'destination_types': [],
        'required_source_nodes': [],
        'destination_nodes': [],
        'resolution_source': 'qwen_two_stage_intent',
    }
    diagnostics: dict[str, Any] = {
        'attempted': True,
        'status': 'started',
        'stage': 'action',
        'reason': 'mandatory two-stage intent resolution',
        'model_path': str(model_path or ''),
        'max_attempts_per_stage': 2,
        'intent_resolution_source': 'qwen_two_stage_intent',
        'attempt_count': 0,
        'reused_across_planning_attempts': True,
        'request': {},
        'attempts': [],
        'result': None,
    }

    def run_stage(
        *,
        stage_name: str,
        request: dict[str, Any],
        validator: Any,
        max_new_tokens: int,
    ) -> tuple[dict[str, Any] | None, str | None]:
        correction_errors: list[str] | None = None
        for attempt_number in range(1, 3):
            attempt_diagnostics: dict[str, Any] = {
                'stage_name': stage_name,
                'attempt': attempt_number,
            }
            raw_result = call_local_qwen_for_semantic_goal_resolution(
                request=request,
                model_path=model_path,
                conv_mode=conv_mode,
                num_gpus=num_gpus,
                qwen_chat=qwen_chat,
                max_new_tokens=max_new_tokens,
                correction_errors=correction_errors,
                diagnostics=attempt_diagnostics,
            )
            if raw_result is None:
                if attempt_diagnostics.get('stage') == 'model_call':
                    protocol_errors = [
                        'resolver model call failed: '
                        f"{attempt_diagnostics.get('error_type')}: "
                        f"{attempt_diagnostics.get('error_message')}"
                    ]
                elif attempt_diagnostics.get('json_parse_status') == 'invalid_json':
                    protocol_errors = ['resolver response is not valid JSON']
                else:
                    protocol_errors = [
                        'resolver failed during '
                        f"{attempt_diagnostics.get('stage') or 'unknown stage'}"
                    ]
                validated = {
                    'protocol_status': 'invalid',
                    'errors': protocol_errors,
                }
            else:
                validated = validator(raw_result)
                protocol_errors = list(validated.get('errors') or [])
            attempt_diagnostics['protocol_validation'] = deepcopy(validated)
            diagnostics['attempts'].append(deepcopy(attempt_diagnostics))
            diagnostics['attempt_count'] += 1
            if validated.get('protocol_status') == 'valid':
                return validated, None
            correction_errors = protocol_errors
        return None, (
            f'{stage_name} intent resolver failed protocol validation after 2 attempts: '
            + '; '.join(correction_errors or ['unknown resolver error'])
        )

    action_request = action_intent_resolution_request(
        task=task,
        semantic_correction=semantic_correction,
    )
    diagnostics['request']['action'] = deepcopy(action_request)
    action_result, action_error = run_stage(
        stage_name='action',
        request=action_request,
        validator=validate_action_intent_resolution,
        max_new_tokens=min(max(int(planning_max_new_tokens), 1), 512),
    )
    if action_error:
        diagnostics.update({
            'status': 'failed',
            'stage': 'action_attempts_exhausted',
            'error': action_error,
        })
        return root_intent, semantic_goal, diagnostics, action_error
    assert action_result is not None
    if action_result.get('status') == 'no_match':
        error = 'action intent resolver found no supported interaction action'
        diagnostics.update({
            'status': 'failed',
            'stage': 'action_no_match',
            'error': error,
            'action_result': deepcopy(action_result),
        })
        return root_intent, semantic_goal, diagnostics, error
    action = str(action_result['action'])
    root_intent['action'] = action
    root_intent['explicit_actions'] = [action]
    semantic_goal['action'] = action

    roles_request = role_intent_resolution_request(
        task=task,
        subgraph=subgraph,
        action=action,
        scene_catalog_summary=scene_catalog_summary,
        semantic_correction=semantic_correction,
    )
    diagnostics['request']['roles'] = deepcopy(roles_request)
    empty_roles = [
        role
        for role, rows in (roles_request.get('role_tables') or {}).items()
        if not rows
    ]
    if empty_roles:
        error = (
            'intent resolver has no grounded candidates for required roles: '
            + ', '.join(sorted(empty_roles))
        )
        diagnostics.update({
            'status': 'failed',
            'stage': 'candidate_filtering',
            'error': error,
            'action_result': deepcopy(action_result),
        })
        return root_intent, semantic_goal, diagnostics, error

    role_result, role_error = run_stage(
        stage_name='roles',
        request=roles_request,
        validator=lambda raw: validate_semantic_goal_resolution(
            raw,
            roles_request,
        ),
        max_new_tokens=min(max(int(planning_max_new_tokens), 1), 2048),
    )
    if role_error:
        diagnostics.update({
            'status': 'failed',
            'stage': 'role_attempts_exhausted',
            'error': role_error,
            'action_result': deepcopy(action_result),
        })
        return root_intent, semantic_goal, diagnostics, role_error
    assert role_result is not None
    if role_result.get('status') == 'no_match':
        error = 'role intent resolver found no matching grounded source'
        diagnostics.update({
            'status': 'failed',
            'stage': 'role_no_match',
            'error': error,
            'action_result': deepcopy(action_result),
            'role_result': deepcopy(role_result),
        })
        return root_intent, semantic_goal, diagnostics, error

    grounding_mode = str(roles_request.get('grounding_mode') or 'catalogue')
    resolved_roles: dict[str, list[str]] = {}
    role_nodes: dict[str, list[Any]] = {}
    role_tags: dict[str, list[str]] = {}
    role_quantifiers: dict[str, str] = {}
    role_statuses: dict[str, str] = {}
    for role, role_value in (role_result.get('roles') or {}).items():
        table = list((roles_request.get('role_tables') or {}).get(role) or [])
        rows_by_id = {row.get('row_id'): row for row in table}
        selected_rows = [
            rows_by_id[row_id]
            for row_id in role_value.get('included_row_ids') or []
            if row_id in rows_by_id
        ]
        resolved_roles[role] = unique_preserve_order([
            str(row.get('object_type') or '')
            for row in selected_rows
            if str(row.get('object_type') or '').strip()
        ])
        role_nodes[role] = [
            row.get('node_id')
            for row in selected_rows
            if row.get('node_id') is not None
        ]
        role_tags[role] = unique_preserve_order([
            str(row.get('object_tag') or '')
            for row in selected_rows
            if str(row.get('object_tag') or '').strip()
        ])
        role_quantifiers[role] = str(role_value.get('quantifier') or 'one')
        role_statuses[role] = str(role_value.get('status') or 'specified')

    if grounding_mode == 'subgraph':
        overlap_nodes = sorted(
            set(role_nodes.get('source') or [])
            .intersection(role_nodes.get('destination') or []),
            key=str,
        )
        if overlap_nodes:
            error = (
                'resolved source and destination nodes overlap; one instance '
                f'cannot fill both roles: {overlap_nodes}'
            )
            diagnostics.update({
                'status': 'failed',
                'stage': 'overlapping_roles',
                'error': error,
                'action_result': deepcopy(action_result),
                'role_result': deepcopy(role_result),
            })
            return root_intent, semantic_goal, diagnostics, error

    source_quantifier = role_quantifiers.get('source', 'one')
    if source_quantifier == 'all':
        if grounding_mode == 'catalogue':
            overlap = sorted(
                set(resolved_roles.get('source') or [])
                .intersection(resolved_roles.get('destination') or [])
            )
        else:
            overlap = sorted(
                set(role_nodes.get('source') or [])
                .intersection(role_nodes.get('destination') or []),
                key=str,
            )
        if overlap:
            error = (
                'source quantifier all overlaps the selected destination; '
                'instance-level exclusion is not supported: '
                f'{overlap}'
            )
            diagnostics.update({
                'status': 'failed',
                'stage': 'overlapping_all_roles',
                'error': error,
                'action_result': deepcopy(action_result),
                'role_result': deepcopy(role_result),
            })
            return root_intent, semantic_goal, diagnostics, error

    root_intent.update({
        'quantifier': source_quantifier,
        'roles': resolved_roles,
        'role_quantifiers': role_quantifiers,
        'role_statuses': role_statuses,
        'role_nodes': role_nodes,
        'role_tags': role_tags,
        'grounding_mode': grounding_mode,
        'semantic_args': deepcopy(role_result.get('semantic_args') or {}),
        'semantic_goal_resolved': True,
    })
    semantic_goal = {
        'action': action,
        'quantifier': source_quantifier,
        'required_source_types': list(resolved_roles.get('source') or []),
        'destination_types': list(resolved_roles.get('destination') or []),
        'required_source_nodes': list(role_nodes.get('source') or []),
        'destination_nodes': list(role_nodes.get('destination') or []),
        'required_source_tags': list(role_tags.get('source') or []),
        'destination_tags': list(role_tags.get('destination') or []),
        'role_quantifiers': deepcopy(role_quantifiers),
        'role_statuses': deepcopy(role_statuses),
        'grounding_mode': grounding_mode,
        'semantic_args': deepcopy(root_intent['semantic_args']),
        'resolution_source': 'qwen_two_stage_intent',
    }
    diagnostics.update({
        'status': 'success',
        'stage': 'complete',
        'action_result': deepcopy(action_result),
        'role_result': deepcopy(role_result),
        'result': deepcopy(semantic_goal),
        'effective_semantic_goal': deepcopy(semantic_goal),
    })
    return root_intent, semantic_goal, diagnostics, None


def _decompose_task_to_graph_with_chat(
    task: str,
    subgraph: dict | Path,
    use_qwen: bool = True,
    qwen_model_path: str | None = DEFAULT_PLANNING_MODEL_PATH,
    qwen_conv_mode: str = 'v0_mmtag',
    qwen_num_gpus: int = 1,
    qwen_chat: Any | None = None,
    qwen_max_new_tokens: int = 2048,
    agent_context: dict[str, Any] | None = None,
    scene_catalog: list[dict[str, Any]] | None = None,
    planning_max_attempts: int = 3,
    planning_mode: str = 'initial',
    execution_context: dict[str, Any] | None = None,
    _intent_cycle: int = 1,
    _semantic_correction: dict[str, Any] | None = None,
    runtime_constraints: dict[str, Any] | None = None,
) -> dict:
    if isinstance(subgraph, Path):
        subgraph_data = load_subgraph(subgraph)
    else:
        subgraph_data = subgraph

    max_attempts = max(int(planning_max_attempts), 1)
    scene_catalog_summary = summarize_scene_catalogue(scene_catalog)
    qwen_diagnostics: dict[str, Any] = {
        'attempted': bool(use_qwen),
        'status': 'disabled' if not use_qwen else 'not_started',
        'stage': 'disabled' if not use_qwen else 'not_started',
        'error_type': None,
        'error_message': None,
        'raw_response_preview': None,
        'response_characters': 0,
        'json_parse_status': 'not_attempted',
        'max_new_tokens': qwen_max_new_tokens,
        'max_attempts': max_attempts,
        'attempt_count': 0,
        'selected_attempt': None,
        'attempts': [],
    }
    if not use_qwen:
        diagnostics = {
            'task': task,
            'status': 'failed',
            'stage': 'planning_disabled',
            'root_intent': derive_root_intent(task, subgraph_data, scene_catalog_summary),
            'scene_catalog': scene_catalog_summary,
            'qwen': qwen_diagnostics,
        }
        raise TaskPlanningError(
            'Planning model is disabled; heuristic task decomposition is not permitted.',
            code='planning_disabled',
            diagnostics=diagnostics,
        )

    (
        root_intent,
        resolved_semantic_goal,
        semantic_goal_diagnostics,
        semantic_goal_error,
    ) = resolve_semantic_goal_for_planning(
        task=task,
        subgraph=subgraph_data,
        scene_catalog_summary=scene_catalog_summary,
        model_path=qwen_model_path,
        conv_mode=qwen_conv_mode,
        num_gpus=qwen_num_gpus,
        qwen_chat=qwen_chat,
        planning_max_new_tokens=qwen_max_new_tokens,
        semantic_correction=_semantic_correction,
    )
    if semantic_goal_error is not None:
        qwen_diagnostics.update({
            'status': 'failed',
            'stage': 'semantic_goal_resolution',
        })
        raise TaskPlanningError(
            f'Semantic goal resolution failed: {semantic_goal_error}',
            code='semantic_goal_resolution',
            diagnostics={
                'task': task,
                'status': 'failed',
                'stage': 'semantic_goal_resolution',
                'root_intent': deepcopy(root_intent),
                'scene_catalog': scene_catalog_summary,
                'semantic_goal_resolution': deepcopy(semantic_goal_diagnostics),
                'qwen': deepcopy(qwen_diagnostics),
            },
        )
    prompt_scene_catalog = build_prompt_scene_catalogue(
        task=task,
        subgraph=subgraph_data,
        root_intent=root_intent,
        scene_catalog_summary=scene_catalog_summary,
    )
    prompt_placement_compatibility = placement_compatibility_for_prompt(
        root_intent,
        scene_catalog_summary,
    )

    raw_plan: dict[str, Any] | None = None
    accepted_task_graph: dict[str, Any] | None = None
    accepted_repairs: dict[str, list[dict[str, Any]]] | None = None
    accepted_validation: dict[str, Any] | None = None
    correction: dict[str, Any] | None = None
    for attempt_number in range(1, max_attempts + 1):
        attempt_diagnostics: dict[str, Any] = {
            'attempt': attempt_number,
            'prompt_kind': 'initial' if attempt_number == 1 else 'corrective_retry',
        }
        candidate_task_graph: dict[str, Any] | None = None
        candidate_repairs: dict[str, list[dict[str, Any]]] | None = None
        semantic_protocol_error: str | None = None
        candidate = call_local_qwen_for_task_graph(
            task=task,
            subgraph=subgraph_data,
            model_path=qwen_model_path,
            conv_mode=qwen_conv_mode,
            num_gpus=qwen_num_gpus,
            agent_context=agent_context,
            max_new_tokens=qwen_max_new_tokens,
            qwen_chat=qwen_chat,
            diagnostics=attempt_diagnostics,
            scene_catalog_summary=prompt_scene_catalog,
            resolved_semantic_goal=resolved_semantic_goal,
            placement_compatibility=prompt_placement_compatibility,
            correction=correction,
            planning_mode=planning_mode,
            execution_context=execution_context,
            runtime_constraints=runtime_constraints,
        )
        if candidate is None:
            if attempt_diagnostics.get('stage') == 'model_call':
                validation_errors = [
                    'planner model call failed: '
                    f"{attempt_diagnostics.get('error_type')}: {attempt_diagnostics.get('error_message')}"
                ]
            elif attempt_diagnostics.get('json_parse_status') == 'invalid_json':
                validation_errors = ['planner response is not valid JSON']
            else:
                validation_errors = [
                    f"planner attempt failed during {attempt_diagnostics.get('stage') or 'unknown stage'}"
                ]
            validation = {
                'status': 'invalid',
                'errors': validation_errors,
                'root_intent': deepcopy(root_intent),
            }
        else:
            validation = validate_planner_output(
                candidate,
                task=task,
                subgraph=subgraph_data,
                scene_catalog_summary=scene_catalog_summary,
                agent_context=agent_context,
                planning_mode=planning_mode,
                execution_context=execution_context,
                root_intent_override=root_intent,
            )
        violations = normalize_validation_violations(
            list(validation['errors']),
            stage='deterministic_validation',
        )
        candidate_subtasks = candidate.get('subtasks') if isinstance(candidate, dict) else None
        if isinstance(candidate_subtasks, list) and candidate_subtasks:
            try:
                candidate_task_graph, candidate_repairs = _prepare_planner_candidate(
                    candidate,
                    task=task,
                    subgraph=subgraph_data,
                )
            except TaskPlanningError as exc:
                attempt_diagnostics['candidate_preparation'] = exc.to_dict()
                violations.extend(normalize_validation_violations(
                    [f'candidate preparation failed: {exc}'],
                    stage='candidate_preparation',
                ))

        if candidate_task_graph is not None:
            post_sanitize_plan = {
                'subtasks': deepcopy(candidate_task_graph.get('flat_tasks') or []),
            }
            post_sanitize_validation = validate_planner_output(
                post_sanitize_plan,
                task=task,
                subgraph=subgraph_data,
                scene_catalog_summary=scene_catalog_summary,
                agent_context=agent_context,
                planning_mode=planning_mode,
                execution_context=execution_context,
                root_intent_override=root_intent,
            )
            attempt_diagnostics['post_sanitize_validation'] = deepcopy(
                post_sanitize_validation
            )
            post_sanitize_violations = normalize_validation_violations(
                list(post_sanitize_validation['errors']),
                stage='post_sanitize_validation',
            )
            existing_messages = {
                str(item.get('message') or '') for item in violations
            }
            violations.extend(
                item for item in post_sanitize_violations
                if str(item.get('message') or '') not in existing_messages
            )

        if (
            candidate_task_graph is not None
            and root_intent.get('grounding_mode') == 'subgraph'
        ):
            attempt_diagnostics['selector_expansion'] = {
                'status': 'skipped',
                'reason': 'subgraph node grounding does not use catalogue selectors',
                'events': [],
            }
        elif candidate_task_graph is not None:
            compiled_candidate = expand_scene_catalog_task_graph(
                candidate_task_graph,
                scene_catalog,
                authoritative_destination_types=(
                    (validation.get('root_intent') or {}).get('roles', {}).get('destination')
                    or []
                ),
            )
            selector_diagnostics = deepcopy(
                (compiled_candidate.get('planner_diagnostics') or {}).get(
                    'selector_expansion'
                ) or {}
            )
            attempt_diagnostics['selector_expansion'] = selector_diagnostics
            if selector_diagnostics.get('status') == 'rejected':
                selector_violations = selector_diagnostics.get('violations')
                violations.extend(normalize_validation_violations(
                    selector_violations
                    if isinstance(selector_violations, list) and selector_violations
                    else [
                        'selector compiler: '
                        + str(
                            selector_diagnostics.get('reason')
                            or 'selector expansion was rejected'
                        )
                    ],
                    stage='selector_expansion',
                ))
        else:
            attempt_diagnostics['selector_expansion'] = {
                'status': 'not_attempted',
                'reason': 'candidate could not be prepared for selector compilation',
                'events': [],
            }

        deterministic_violations = deepcopy(violations)
        if candidate is not None:
            semantic_candidate_plan = candidate_task_graph or candidate
            critic_scene_catalog = build_prompt_scene_catalogue(
                task=task,
                subgraph=subgraph_data,
                root_intent=root_intent,
                scene_catalog_summary=scene_catalog_summary,
                candidate_plan=semantic_candidate_plan,
            )
            semantic_diagnostics: dict[str, Any] = {}
            raw_critique = call_local_qwen_for_plan_critique(
                task=task,
                subgraph=subgraph_data,
                candidate_plan=semantic_candidate_plan,
                scene_catalog_summary=critic_scene_catalog,
                resolved_semantic_goal=resolved_semantic_goal,
                placement_compatibility=prompt_placement_compatibility,
                model_path=qwen_model_path,
                conv_mode=qwen_conv_mode,
                num_gpus=qwen_num_gpus,
                qwen_chat=qwen_chat,
                max_new_tokens=min(max(int(qwen_max_new_tokens), 1), 768),
                diagnostics=semantic_diagnostics,
                planning_mode=planning_mode,
                execution_context=execution_context,
                runtime_constraints=runtime_constraints,
            )
            if raw_critique is None:
                semantic_protocol_error = (
                    'semantic critic failed during '
                    f"{semantic_diagnostics.get('stage') or 'unknown stage'}: "
                    f"{semantic_diagnostics.get('error_type') or ''}: "
                    f"{semantic_diagnostics.get('error_message') or ''}"
                ).rstrip(': ')
                raw_semantic_result = {
                    'protocol_status': 'invalid',
                    'status': None,
                    'errors': [semantic_protocol_error],
                }
            else:
                raw_semantic_result = validate_semantic_critique(raw_critique)
                if raw_semantic_result['protocol_status'] != 'valid':
                    semantic_protocol_error = '; '.join(
                        str(item.get('message') or item) if isinstance(item, dict) else str(item)
                        for item in raw_semantic_result['errors']
                    )
            semantic_result, discarded_semantic_errors = enforce_semantic_critic_scope(
                raw_semantic_result,
                candidate_plan=semantic_candidate_plan,
                root_intent=validation.get('root_intent'),
            )
            semantic_diagnostics['verdict'] = deepcopy(raw_semantic_result)
            semantic_diagnostics['discarded_out_of_scope_errors'] = deepcopy(
                discarded_semantic_errors
            )
            semantic_diagnostics['effective_verdict'] = deepcopy(semantic_result)
            attempt_diagnostics['semantic_validation'] = semantic_diagnostics
            if semantic_protocol_error is None and semantic_result['status'] == 'invalid':
                semantic_violations = normalize_validation_violations(
                    list(semantic_result['errors']),
                    stage='semantic_validation',
                )
                if (
                    not deterministic_violations
                    and semantic_violations_challenge_resolved_intent(
                        semantic_violations
                    )
                ):
                    recovery_record = {
                        'trigger': 'semantic_critic_resolved_intent_mismatch',
                        'cycle': _intent_cycle,
                        'previous_root_intent': deepcopy(root_intent),
                        'previous_semantic_goal': deepcopy(
                            resolved_semantic_goal
                        ),
                        'critic_errors': deepcopy(semantic_violations),
                        'semantic_goal_resolution': deepcopy(
                            semantic_goal_diagnostics
                        ),
                        'planner_attempt': deepcopy(attempt_diagnostics),
                    }
                    if _intent_cycle >= 2:
                        raise TaskPlanningError(
                            'Semantic critic still rejects the re-resolved intent.',
                            code='semantic_intent_mismatch',
                            diagnostics={
                                'task': task,
                                'status': 'failed',
                                'stage': 'semantic_intent_mismatch',
                                'root_intent': deepcopy(root_intent),
                                'scene_catalog': scene_catalog_summary,
                                'semantic_goal_resolution': deepcopy(
                                    semantic_goal_diagnostics
                                ),
                                'semantic_validation': deepcopy(
                                    semantic_diagnostics
                                ),
                                'intent_recovery': {
                                    'status': 'failed_after_one_retry',
                                    **recovery_record,
                                },
                                'qwen': deepcopy(qwen_diagnostics),
                            },
                        )
                    semantic_correction = {
                        'reason': (
                            'The semantic critic found that the prior resolved '
                            'intent misunderstood the public instruction.'
                        ),
                        'previous_resolved_intent': deepcopy(root_intent),
                        'critic_errors': deepcopy(semantic_violations),
                        'instruction': (
                            'Re-read the original instruction and independently '
                            'choose both the action and all role row selections '
                            'again. Do not preserve the prior choice unless it is '
                            'actually supported by the instruction.'
                        ),
                    }
                    try:
                        recovered_graph = _decompose_task_to_graph_with_chat(
                            task=task,
                            subgraph=subgraph_data,
                            use_qwen=use_qwen,
                            qwen_model_path=qwen_model_path,
                            qwen_conv_mode=qwen_conv_mode,
                            qwen_num_gpus=qwen_num_gpus,
                            qwen_chat=qwen_chat,
                            qwen_max_new_tokens=qwen_max_new_tokens,
                            agent_context=agent_context,
                            scene_catalog=scene_catalog,
                            planning_max_attempts=planning_max_attempts,
                            planning_mode=planning_mode,
                            execution_context=execution_context,
                            runtime_constraints=runtime_constraints,
                            _intent_cycle=2,
                            _semantic_correction=semantic_correction,
                        )
                    except TaskPlanningError as exc:
                        exc.diagnostics['intent_recovery'] = {
                            'status': 'failed',
                            **recovery_record,
                            'second_cycle_failure': deepcopy(
                                exc.diagnostics
                            ),
                        }
                        raise
                    recovered_graph.setdefault(
                        'planner_diagnostics',
                        {},
                    )['intent_recovery'] = {
                        'status': 'recovered',
                        **recovery_record,
                        'replacement_root_intent': deepcopy(
                            (
                                recovered_graph.get('planner_diagnostics')
                                or {}
                            ).get('root_intent_validation', {}).get(
                                'root_intent'
                            )
                        ),
                    }
                    return recovered_graph
                for violation in semantic_violations:
                    if not violation['message'].startswith('semantic validation:'):
                        violation['message'] = f"semantic validation: {violation['message']}"
                violations.extend(semantic_violations)
        else:
            attempt_diagnostics['semantic_validation'] = {
                'attempted': False,
                'status': 'not_attempted',
                'stage': 'not_attempted',
                'reason': 'planner candidate is unavailable',
            }

        violations = normalize_validation_violations(
            violations,
            stage='deterministic_validation',
        )
        validation = {
            'status': 'valid' if not violations else 'invalid',
            'errors': [str(item['message']) for item in violations],
            'violations': deepcopy(violations),
            'root_intent': validation['root_intent'],
        }

        attempt_diagnostics['validation'] = deepcopy(validation)
        attempt_diagnostics['validation_status'] = validation['status']
        attempt_diagnostics['validation_errors'] = list(validation['errors'])
        qwen_diagnostics['attempts'].append(deepcopy(attempt_diagnostics))
        qwen_diagnostics['attempt_count'] = attempt_number
        if semantic_protocol_error is not None:
            qwen_diagnostics.update({
                'status': 'failed',
                'stage': 'semantic_validation',
                'selected_attempt': None,
            })
            raise TaskPlanningError(
                f'Semantic validation failed: {semantic_protocol_error}',
                diagnostics={
                    'task': task,
                    'status': 'failed',
                    'stage': 'semantic_validation',
                    'root_intent': deepcopy(validation.get('root_intent')),
                    'scene_catalog': scene_catalog_summary,
                    'semantic_goal_resolution': deepcopy(
                        semantic_goal_diagnostics
                    ),
                    'qwen': deepcopy(qwen_diagnostics),
                    'semantic_validation': deepcopy(
                        attempt_diagnostics.get('semantic_validation') or {}
                    ),
                },
            )

        if candidate_task_graph is not None and candidate_repairs is not None and validation['status'] == 'valid':
            raw_plan = candidate
            accepted_task_graph = candidate_task_graph
            accepted_repairs = candidate_repairs
            accepted_validation = validation
            qwen_diagnostics.update({
                key: deepcopy(value)
                for key, value in attempt_diagnostics.items()
                if key not in {'attempt', 'prompt_kind', 'validation'}
            })
            qwen_diagnostics.update({
                'status': 'success',
                'stage': 'complete',
                'selected_attempt': attempt_number,
            })
            break
        correction = {
            'attempt': attempt_number + 1,
            'max_attempts': max_attempts,
            'violations': deepcopy(validation['violations']),
            'summary': correction_summary(validation['violations']),
        }

    if raw_plan is None:
        last_attempt = qwen_diagnostics['attempts'][-1] if qwen_diagnostics['attempts'] else {}
        qwen_diagnostics.update({
            key: deepcopy(value)
            for key, value in last_attempt.items()
            if key not in {'attempt', 'prompt_kind', 'validation'}
        })
        qwen_diagnostics.update({
            'status': 'failed',
            'stage': 'attempts_exhausted',
            'selected_attempt': None,
        })
        diagnostics = {
            'task': task,
            'status': 'failed',
            'stage': 'planning_attempts_exhausted',
            'root_intent': (
                deepcopy(last_attempt.get('validation', {}).get('root_intent'))
                or deepcopy(root_intent)
            ),
            'scene_catalog': scene_catalog_summary,
            'semantic_goal_resolution': deepcopy(semantic_goal_diagnostics),
            'qwen': qwen_diagnostics,
        }
        raise TaskPlanningError(
            f'Qwen planning failed validation after {max_attempts} attempts.',
            diagnostics=diagnostics,
        )

    assert accepted_task_graph is not None
    assert accepted_repairs is not None
    task_graph = accepted_task_graph

    task_graph['planner_diagnostics'] = {
        'qwen': qwen_diagnostics,
        'semantic_goal_resolution': deepcopy(semantic_goal_diagnostics),
        'prompt_scene_catalog': deepcopy(prompt_scene_catalog),
        'fallback_used': False,
        'selected_backend': task_graph.get('planner_backend'),
        'root_intent_validation': accepted_validation,
        'entity_coverage': audit_plan_entity_coverage(task, subgraph_data, task_graph),
        'agent_context_provided': bool(agent_context),
        'atomic_find_repairs': accepted_repairs['atomic_find_repairs'],
        'atomic_place_repairs': accepted_repairs['atomic_place_repairs'],
        'slice_tool_repairs': accepted_repairs['slice_tool_repairs'],
        'completion_repairs': accepted_repairs['completion_repairs'],
        'planning_mode': planning_mode,
    }
    return task_graph


def decompose_task_to_graph(
    task: str,
    subgraph: dict | Path,
    use_qwen: bool = True,
    qwen_model_path: str | None = DEFAULT_PLANNING_MODEL_PATH,
    qwen_conv_mode: str = 'v0_mmtag',
    qwen_num_gpus: int = 1,
    qwen_chat: Any | None = None,
    qwen_max_new_tokens: int = 2048,
    agent_context: dict[str, Any] | None = None,
    scene_catalog: list[dict[str, Any]] | None = None,
    planning_max_attempts: int = 3,
    planning_mode: str = 'initial',
    execution_context: dict[str, Any] | None = None,
    runtime_constraints: dict[str, Any] | None = None,
) -> dict:
    """Build a plan while sharing one model instance across planner and critic sessions."""

    kwargs = {
        'task': task,
        'subgraph': subgraph,
        'use_qwen': use_qwen,
        'qwen_model_path': qwen_model_path,
        'qwen_conv_mode': qwen_conv_mode,
        'qwen_num_gpus': qwen_num_gpus,
        'qwen_max_new_tokens': qwen_max_new_tokens,
        'agent_context': agent_context,
        'scene_catalog': scene_catalog,
        'planning_max_attempts': planning_max_attempts,
        'planning_mode': planning_mode,
        'execution_context': execution_context,
        'runtime_constraints': runtime_constraints,
    }
    if not use_qwen or qwen_chat is not None:
        return _decompose_task_to_graph_with_chat(
            **kwargs,
            qwen_chat=qwen_chat,
        )

    try:
        from conceptgraph.vlm import build_vlm_chat, close_vlm_chat
        shared_chat = build_vlm_chat(
            backend='qwen',
            model_path=qwen_model_path,
            conv_mode=qwen_conv_mode,
            num_gpus=qwen_num_gpus,
        )
    except Exception as exc:
        initialization_error = exc

        class UnavailablePlanningChat:
            def __call__(self, prompt: str) -> str:
                del prompt
                raise exc

        return _decompose_task_to_graph_with_chat(
            **kwargs,
            qwen_chat=UnavailablePlanningChat(),
        )

    try:
        return _decompose_task_to_graph_with_chat(
            **kwargs,
            qwen_chat=shared_chat,
        )
    finally:
        close_vlm_chat(shared_chat)


def save_task_graph(task_graph: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(task_graph, f, indent=2, ensure_ascii=False)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Build a dependency-aware task graph from a task instruction and a task-relevant subgraph.'
    )
    parser.add_argument('--task', required=True, help='Natural-language task instruction.')
    parser.add_argument('--subgraph', type=Path, required=True, help='Path to task_relevant_subgraph.json or its parent directory.')
    parser.add_argument('--output', type=Path, default=None, help='Output JSON path. Defaults to <subgraph_dir>/task_graph.json')
    parser.add_argument(
        '--disable-qwen',
        action='store_true',
        help='Disable planning. Hybrid execution fails closed; no heuristic decomposition is used.',
    )
    parser.add_argument(
        '--qwen-model-path',
        default=DEFAULT_PLANNING_MODEL_PATH,
        help='Local planning model path (default: Qwen3.5-9B; override with PLANNING_MODEL_PATH).',
    )
    parser.add_argument('--qwen-conv-mode', default='v0_mmtag')
    parser.add_argument('--qwen-num-gpus', type=int, default=1)
    parser.add_argument('--planning-max-attempts', type=int, default=3)
    parser.add_argument('--planning-max-new-tokens', type=int, default=2048)
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    subgraph_path = resolve_subgraph_path(args.subgraph)
    output_path = args.output or subgraph_path.with_name('task_graph.json')
    task_graph = decompose_task_to_graph(
        task=args.task,
        subgraph=subgraph_path,
        use_qwen=not args.disable_qwen,
        qwen_model_path=args.qwen_model_path,
        qwen_conv_mode=args.qwen_conv_mode,
        qwen_num_gpus=args.qwen_num_gpus,
        planning_max_attempts=args.planning_max_attempts,
        qwen_max_new_tokens=args.planning_max_new_tokens,
    )
    save_task_graph(task_graph, output_path)

    print('Saved task graph')
    print(f"Planner backend: {task_graph['planner_backend']}")
    print(f"Chains: {len(task_graph['chains'])}")
    print(f"Flat tasks: {len(task_graph['flat_tasks'])}")
    print(f"Output: {output_path}")


if __name__ == '__main__':
    main()
