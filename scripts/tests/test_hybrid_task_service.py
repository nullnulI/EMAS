from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError


EMAS_ROOT = Path('/225010231/mwl/EMAS')
MODULE_PATH = EMAS_ROOT / 'scripts' / 'hybrid_decision_loop.py'


def install_stubs() -> None:
    planning = types.ModuleType('planning')
    datagen = types.ModuleType('planning.datagen')
    generation = types.ModuleType('planning.datagen.generation')

    def build_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--task', default='root task')
        parser.add_argument('--scene_name', default='train_3')
        parser.add_argument('--agentnum', type=int, default=2)
        parser.add_argument('--max-task-loops', dest='max_task_loops', type=int, default=1)
        parser.add_argument('--disable-qwen', action='store_true')
        parser.add_argument('--planning-max-attempts', dest='planning_max_attempts', type=int, default=3)
        parser.add_argument('--disable-skill-qwen', action='store_true')
        parser.add_argument('--skill-max-steps', dest='skill_max_steps', type=int, default=6)
        parser.add_argument('--skill-complete-on-execute', dest='skill_complete_on_execute', action='store_true')
        parser.add_argument('--max-task-retries', dest='max_task_retries', type=int, default=3)
        return parser

    generation.build_parser = build_parser
    generation.json_safe = lambda value: value
    datagen.generation = generation
    planning.datagen = datagen

    extract_subgraph = types.ModuleType('planning.extract_subgraph')
    extract_subgraph.parse_task_locally = lambda task: {}

    task_graph = types.ModuleType('planning.task_graph')

    class TaskPlanningError(RuntimeError):
        code = 'task_planning_failed'

        def __init__(self, message, *, diagnostics):
            super().__init__(message)
            self.diagnostics = diagnostics

        def to_dict(self):
            return {'code': self.code, 'message': str(self), 'diagnostics': self.diagnostics}

    task_graph.TaskPlanningError = TaskPlanningError
    task_graph.decompose_task_to_graph = lambda *args, **kwargs: {}
    task_graph.infer_task_object_tags = lambda task: []
    task_graph.save_task_graph = lambda *args, **kwargs: None
    task_graph.unique_preserve_order = lambda values: list(dict.fromkeys(values))

    def rebuild_task_graph_views(graph):
        rebuilt = deepcopy(graph)
        tasks = [deepcopy(task) for task in rebuilt.get('flat_tasks') or []]
        for task in tasks:
            task['depends_on'] = list(dict.fromkeys(
                str(value) for value in task.get('depends_on') or []
            ))
            task['chain_id'] = 'C1'
        rebuilt['flat_tasks'] = tasks
        rebuilt['dependency_edges'] = [
            {'from': dependency, 'to': task['id']}
            for task in tasks
            for dependency in task.get('depends_on') or []
        ]
        rebuilt['root_task_ids'] = [
            task['id'] for task in tasks if not task.get('depends_on')
        ]
        rebuilt['chains'] = ([{
            'chain_id': 'C1',
            'head_task_id': tasks[0]['id'],
            'head_depends_on': list(tasks[0].get('depends_on') or []),
            'task_ids': [task['id'] for task in tasks],
            'linked_list': deepcopy(tasks[0]),
        }] if tasks else [])
        return rebuilt

    task_graph.rebuild_task_graph_views = rebuild_task_graph_views

    allocation = types.ModuleType('planning.task_allocation')

    class TaskAllocationError(RuntimeError):
        code = 'task_allocation_failed'

        def __init__(self, message, *, diagnostics, code=None):
            super().__init__(message)
            self.code = code or type(self).code
            self.diagnostics = diagnostics

        def to_dict(self):
            return {'code': self.code, 'message': str(self), 'diagnostics': self.diagnostics}

    allocation.TaskAllocationError = TaskAllocationError
    allocation.build_execution_plan = lambda *args, **kwargs: {}
    allocation.first_execution_unit = lambda *args, **kwargs: []

    task_status = types.ModuleType('planning.utils.task_status')
    task_status.FAILURE = 'failure'
    task_status.SUCCESS = 'success'
    task_status.WAIT_RETRY = 'wait_retry'

    utils = types.ModuleType('planning.utils')
    utils.task_status = task_status

    memory = types.ModuleType('memory')
    graph_update = types.ModuleType('memory.graph_update')
    graph_update.diff_objects = lambda *args, **kwargs: []
    graph_update.infer_relation_deltas_from_metadata = lambda *args, **kwargs: []
    graph_update.update_scenegraph_files = lambda *args, **kwargs: {}
    memory.graph_update = graph_update

    sys.modules.update({
        'planning': planning,
        'planning.datagen': datagen,
        'planning.datagen.generation': generation,
        'planning.extract_subgraph': extract_subgraph,
        'planning.task_graph': task_graph,
        'planning.task_allocation': allocation,
        'planning.utils': utils,
        'planning.utils.task_status': task_status,
        'memory': memory,
        'memory.graph_update': graph_update,
    })


def load_module():
    install_stubs()
    spec = importlib.util.spec_from_file_location('hybrid_decision_loop_under_test', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode('utf-8')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TaskExecutionServiceAdapterTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def args(self):
        return argparse.Namespace(
            agentnum=3,
            task_service_url='http://127.0.0.1:18080/execute_task',
            task_service_timeout=12.0,
            task_service_dry_run=True,
            task_service_relay_strategy='rules',
            task_service_max_replan_steps=4,
            task_service_relay_agent_max_turns=5,
            task_service_max_actions=6,
        )

    def payload(self):
        return {
            'loop_index': 2,
            'task': 'open the fridge',
            'assignments': [
                {'agent_id': '1', 'subtask': {'id': 'T1', 'name': 'Open fridge', 'description': 'open the fridge', 'action': 'open'}},
            ],
            'agent_states': [{'agent_id': '0'}, {'agent_id': '1'}],
        }

    def test_only_structural_terminal_failure_requests_task_graph_replan(self):
        structural = {
            'status': self.module.FAILURE,
            'failure_code': 'no_valid_placement',
            'recommended_recovery': 'replan_task_graph',
        }
        retryable = {
            'status': self.module.WAIT_RETRY,
            'failure_code': 'target_not_visible',
            'recommended_recovery': 'retry_original_task',
        }
        self.assertTrue(
            self.module.execution_status_requires_task_graph_replan(structural)
        )
        self.assertFalse(
            self.module.execution_status_requires_task_graph_replan(retryable)
        )

    def test_replanning_context_preserves_report_and_current_inventory(self):
        graph = self.module.rebuild_task_graph({
            'task': 'place mug',
            'flat_tasks': [
                {'id': 'T1', 'action': 'pick', 'depends_on': []},
                {'id': 'T2', 'action': 'place', 'depends_on': ['T1']},
            ],
        })
        status = {
            'subtask_id': 'T2',
            'status': self.module.FAILURE,
            'failure_code': 'no_valid_placement',
            'recommended_recovery': 'replan_task_graph',
            'excluded_destination_object_ids': ['Cabinet|1'],
        }
        report = {'marker': 'complete-report', 'object_changes': []}
        context = self.module.build_replanning_execution_context(
            root_task='place mug', report=report, task_statuses=[status],
            task_graph=graph, completed={'T1'}, failed=set(),
            agent_states=[{
                'agent_id': '0',
                'inventory': [{'objectId': 'Mug|1', 'objectType': 'Mug'}],
            }],
        )

        self.assertEqual(context['execution_report']['marker'], 'complete-report')
        self.assertEqual(context['completed_tasks'][0]['id'], 'T1')
        self.assertEqual(
            context['agent_states'][0]['inventory'][0]['objectId'], 'Mug|1'
        )
        constraints = self.module.build_runtime_replanning_constraints(
            context,
            [{'objectId': 'Cabinet|2', 'objectType': 'Cabinet'}],
        )
        self.assertEqual(constraints['excluded_object_ids'], ['Cabinet|1'])
        self.assertEqual(constraints['held_objects'][0]['objectType'], 'Mug')

    def test_replanned_destination_binding_excludes_failed_instance(self):
        graph = self.module.rebuild_task_graph({
            'task': 'place mug in cabinet',
            'flat_tasks': [{
                'id': 'R1', 'action': 'place', 'depends_on': [],
                'grounding': {
                    'source_object_tags': ['Mug'],
                    'destination_selector': {
                        'quantifier': 'one', 'object_types': ['Cabinet'],
                    },
                },
            }],
        })
        catalog = [
            {'objectId': 'Cabinet|1', 'objectType': 'Cabinet', 'receptacle': True},
            {'objectId': 'Cabinet|2', 'objectType': 'Cabinet', 'receptacle': True},
        ]
        rebound = self.module.bind_replanned_destination_instances(
            graph, catalog, {'Cabinet|1'}
        )
        grounding = rebound['flat_tasks'][0]['grounding']
        self.assertEqual(grounding['destination_object_ids'], ['Cabinet|2'])
        self.assertEqual(
            grounding['excluded_destination_object_ids'], ['Cabinet|1']
        )

    def test_runtime_validation_rejects_pick_of_already_held_type(self):
        graph = self.module.rebuild_task_graph({
            'task': 'place mug',
            'flat_tasks': [{
                'id': 'R1', 'action': 'pick', 'depends_on': [],
                'grounding': {'source_object_tags': ['Mug']},
            }],
        })
        violations = self.module.validate_replanned_graph_runtime_constraints(
            graph,
            {
                'excluded_object_ids': [],
                'held_objects': [{'agent_id': '0', 'objectType': 'Mug'}],
            },
        )
        self.assertEqual(violations[0]['code'], 'duplicate_pick_of_held_object')

    def test_runtime_validation_rejects_incompatible_replanned_placement(self):
        graph = self.module.rebuild_task_graph({
            'task': 'place apple',
            'flat_tasks': [{
                'id': 'R1', 'action': 'place', 'depends_on': [],
                'grounding': {
                    'source_object_tags': ['Apple'],
                    'destination_object_tags': ['Drawer'],
                },
            }],
        })
        violations = self.module.validate_replanned_graph_runtime_constraints(
            graph, {'excluded_object_ids': [], 'held_objects': []}
        )
        self.assertEqual(violations[0]['code'], 'incompatible_receptacle')
        self.assertIn(
            'Fridge', violations[0]['invalid_values'][0]['compatible_receptacles']
        )

    def test_runtime_validation_preserves_explicit_user_destination(self):
        graph = self.module.rebuild_task_graph({
            'task': 'put all shakers in the fridge',
            'flat_tasks': [{
                'id': 'R1', 'action': 'place', 'depends_on': [],
                'grounding': {
                    'source_object_tags': ['SaltShaker'],
                    'destination_object_tags': ['Fridge'],
                },
            }],
        })
        graph['planner_diagnostics'] = {
            'root_intent_validation': {
                'root_intent': {
                    'action': 'place',
                    'roles': {
                        'source': ['SaltShaker'],
                        'destination': ['Fridge'],
                    },
                },
            },
        }

        violations = self.module.validate_replanned_graph_runtime_constraints(
            graph, {'excluded_object_ids': [], 'held_objects': []}
        )

        self.assertEqual(violations, [])

    def test_replan_wrapper_calls_existing_planning_with_execution_context(self):
        captured = {}
        old_extract = getattr(
            self.module.gen, 'extract_task_relevant_subgraph_to_file', None
        )
        self.module.gen.extract_task_relevant_subgraph_to_file = lambda **kwargs: {
            'nodes': [], 'seed_nodes': [],
        }
        old_decompose = self.module.decompose_task_to_graph
        old_selectors = self.module.apply_scene_catalog_selectors

        def fake_decompose(**kwargs):
            captured.update(kwargs)
            return self.module.rebuild_task_graph({
                'task': kwargs['task'],
                'flat_tasks': [{
                    'id': 'R1', 'action': 'place', 'depends_on': [],
                    'grounding': {
                        'source_object_tags': ['Mug'],
                        'destination_selector': {
                            'quantifier': 'one', 'object_types': ['Cabinet'],
                        },
                    },
                }],
            })

        self.module.decompose_task_to_graph = fake_decompose
        self.module.apply_scene_catalog_selectors = lambda graph, catalog: graph
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                result = self.module.replan_from_execution_report(
                    root_task='place mug in cabinet',
                    current_task_graph=self.module.rebuild_task_graph({
                        'task': 'place mug in cabinet',
                        'flat_tasks': [{
                            'id': 'T1', 'action': 'place', 'depends_on': [],
                        }],
                    }),
                    report={'marker': 'full-report', 'object_changes': []},
                    task_statuses=[{
                        'subtask_id': 'T1',
                        'status': self.module.FAILURE,
                        'failure_code': 'no_valid_placement',
                        'recommended_recovery': 'replan_task_graph',
                        'excluded_destination_object_ids': ['Cabinet|1'],
                    }],
                    completed=set(), failed=set(), current_scenegraph={},
                    agent_states=[{
                        'agent_id': '0',
                        'inventory': [{'objectId': 'Mug|1', 'objectType': 'Mug'}],
                    }],
                    planning_chat=object(), loop_index=2, replan_index=1,
                    loop_dir=Path(temp_dir),
                    args=argparse.Namespace(
                        _scene_object_catalog=[
                            {'objectId': 'Mug|1', 'objectType': 'Mug', 'pickupable': True},
                            {'objectId': 'Cabinet|1', 'objectType': 'Cabinet', 'receptacle': True},
                            {'objectId': 'Cabinet|2', 'objectType': 'Cabinet', 'receptacle': True},
                        ],
                        planning_model_path='/models/planner',
                        qwen_conv_mode='v0_mmtag', qwen_num_gpus=1,
                        planning_max_new_tokens=512,
                        task_graph_replan_max_attempts=2, agentnum=1,
                    ),
                )
        finally:
            if old_extract is None:
                delattr(self.module.gen, 'extract_task_relevant_subgraph_to_file')
            else:
                self.module.gen.extract_task_relevant_subgraph_to_file = old_extract
            self.module.decompose_task_to_graph = old_decompose
            self.module.apply_scene_catalog_selectors = old_selectors

        self.assertEqual(result['status'], 'success')
        self.assertEqual(captured['planning_mode'], 'runtime_replan')
        self.assertEqual(
            captured['execution_context']['execution_report']['marker'],
            'full-report',
        )
        grounding = result['task_graph']['flat_tasks'][0]['grounding']
        self.assertEqual(grounding['destination_object_ids'], ['Cabinet|2'])

    def test_exhaustive_failure_planning_explainer_is_explain_only(self):
        class FakePlanningChat:
            def __init__(self):
                self.messages = []
                self.reset_count = 0
                self.payload = None

            def reset(self):
                self.reset_count += 1

            def __call__(self, prompt):
                self.payload = json.loads(prompt)
                return json.dumps({
                    'status': 'impossible',
                    'reason_code': 'resource_unavailable',
                    'reason': 'No observed robot can hold or acquire a compatible knife.',
                    'evidence_refs': ['candidate_evidence'],
                })

        chat = FakePlanningChat()
        result = self.module.explain_exhaustive_runtime_failure(
            chat,
            parent_task='slice the lettuce',
            task_status={
                'subtask_id': 'T2',
                'failure_code': 'no_capable_executor',
                'reason': 'all candidates exhausted',
                'candidate_evidence': [{'robot_id': 0}, {'robot_id': 1}],
                'subtask': {'id': 'T2', 'action': 'slice'},
            },
            agent_states=[{'agent_id': '0', 'inventory': []}],
        )

        self.assertEqual(chat.reset_count, 1)
        self.assertEqual(chat.payload['request'], 'explain_terminal_execution_failure')
        self.assertNotIn('task_graph', chat.payload)
        self.assertEqual(result['status'], 'impossible')

    def test_task_service_summary_retains_exhaustive_candidate_evidence(self):
        summary = self.module.task_service_response_summary({
            'status': 'failed',
            'result': {
                'closed_loop_result': {
                    'status': 'needs_upstream_planning',
                    'failure_code': 'no_capable_executor',
                    'exhaustive': True,
                    'candidate_evidence': [{'robot_id': 0, 'executable': False}],
                    'resource_recovery_history': [{'resource_executor_robot_id': 1}],
                },
            },
        })

        closed_loop = summary['closed_loop_result']
        self.assertTrue(closed_loop['exhaustive'])
        self.assertEqual(closed_loop['candidate_evidence'][0]['robot_id'], 0)
        self.assertEqual(
            closed_loop['resource_recovery_history'][0]['resource_executor_robot_id'],
            1,
        )

    def test_parser_accepts_task_service(self):
        parser = self.module.build_parser()
        args = parser.parse_args(['--execution-mode', 'task_service', '--task-service-dry-run'])
        self.assertEqual(args.execution_mode, 'task_service')
        self.assertTrue(args.task_service_dry_run)
        self.assertEqual(args.task_service_url, 'http://127.0.0.1:18080/execute_task')

    def test_parser_rejects_legacy_execution_mode(self):
        parser = self.module.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(['--execution-mode', 'relay_service'])

    def test_parser_rejects_legacy_service_option(self):
        parser = self.module.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(['--relay-service-dry-run'])

    def test_parser_rejects_legacy_launcher_option(self):
        parser = self.module.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(['--relay-task-python', 'python'])

    def test_parser_accepts_task_service_autostart_args(self):
        parser = self.module.build_parser()
        args = parser.parse_args([
            '--execution-mode', 'task_service',
            '--task-service-autostart',
            '--task-service-python', '/opt/qwen/bin/python',
            '--task-service-model-path', '/models/qwen',
            '--task-service-device', 'cpu',
            '--task-service-port-base', '18100',
            '--receiver-port-base', '19100',
        ])
        self.assertEqual(args.execution_mode, 'task_service')
        self.assertTrue(args.task_service_autostart)
        self.assertEqual(args.task_service_python, '/opt/qwen/bin/python')
        self.assertEqual(args.task_service_model_path, '/models/qwen')
        self.assertEqual(args.task_service_device, 'cpu')
        self.assertEqual(args.task_service_port_base, 18100)
        self.assertEqual(args.receiver_port_base, 19100)

    def test_success_response_maps_to_success_report(self):
        response = {'status': 'success', 'task_id': 'relay-1', 'result': {'closed_loop_result': {'status': 'success'}}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(self.module, 'urlopen', return_value=FakeResponse(response)) as mocked:
            report = self.module.TaskExecutionServiceAdapter(self.args()).receive_execution_report(self.payload(), Path(tmp))

        self.assertEqual(report['execution']['completed_task_ids'], ['T1'])
        self.assertEqual(report['task_statuses'][0]['status'], 'success')
        request_payload = json.loads(mocked.call_args.args[0].data.decode('utf-8'))
        self.assertEqual(request_payload['task_id'], 'loop_2_T1')
        self.assertEqual(request_payload['primary_robot_id'], 1)
        self.assertNotIn('known_robot_ids', request_payload)
        self.assertTrue(request_payload['dry_run'])
        self.assertEqual(request_payload['relay_strategy'], 'rules')
        self.assertIn('open the fridge', request_payload['task'])
        self.assertEqual(request_payload['parent_task'], 'open the fridge')
        self.assertEqual(request_payload['subtask']['id'], 'T1')
        self.assertEqual(report['execution']['traces'][0]['executor'], 'task_execution_service')

    def test_success_response_that_drops_open_action_is_rejected(self):
        response = {
            'status': 'success',
            'task_id': 'relay-1',
            'task_normalization': {
                'intentSteps': [
                    {'order': 1, 'action': 'GotoObject', 'objectType': 'Fridge'},
                ],
            },
            'result': {'closed_loop_result': {'status': 'success'}},
        }
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            self.module, 'urlopen', return_value=FakeResponse(response)
        ):
            report = self.module.TaskExecutionServiceAdapter(
                self.args()
            ).receive_execution_report(self.payload(), Path(tmp))

        self.assertEqual(report['execution']['completed_task_ids'], [])
        self.assertEqual(report['task_statuses'][0]['status'], 'wait_retry')
        self.assertEqual(
            report['task_statuses'][0]['failure_code'], 'semantic_action_dropped'
        )


    def test_successful_goto_response_adds_object_change(self):
        response = {
            'status': 'success',
            'task_id': 'relay-1',
            'result': {
                'closed_loop_result': {'status': 'success', 'strategy': 'goto'},
                'goto_result_summary': {
                    'target': {
                        'object_id': 'Fridge|-02.10|+00.00|+01.07',
                        'object_type': 'Fridge',
                        'position': {'x': -2.1, 'y': 0.0, 'z': 1.07},
                    },
                    'goal_position': {'x': -1.25, 'y': 0.9, 'z': 1.0},
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp, patch.object(self.module, 'urlopen', return_value=FakeResponse(response)):
            report = self.module.TaskExecutionServiceAdapter(self.args()).receive_execution_report(self.payload(), Path(tmp))

        self.assertEqual(report['execution']['completed_task_ids'], ['T1'])
        self.assertEqual(report['object_changes'], [
            {
                'objectId': 'Fridge|-02.10|+00.00|+01.07',
                'objectType': 'Fridge',
                'source': 'task_execution_service_goto',
                'position': {'x': -2.1, 'y': 0.0, 'z': 1.07},
                'observer_goal_position': {'x': -1.25, 'y': 0.9, 'z': 1.0},
                'source_task_id': 'T1',
                'observer_robot_id': '1',
            }
        ])
        observed = self.module.adapter_object_changes_to_observed_nodes(report['object_changes'])
        self.assertEqual(observed[0]['object_tag'], 'Fridge')
        self.assertEqual(observed[0]['bbox_center'], [-2.1, 0.0, 1.07])
        self.assertEqual(observed[0]['last_observer_agent_id'], '1')
        self.assertEqual(observed[0]['last_observer_position'], [-1.25, 0.9, 1.0])
        self.assertEqual(observed[0]['source_task_id'], 'T1')

    def test_post_execution_states_replace_stale_payload_states(self):
        post_states = [
            {
                'agent_id': '0',
                'robot_id': 0,
                'agent': {'position': {'x': 0, 'y': 0.9, 'z': 1}},
                'visible_objects': [],
            },
            {
                'agent_id': '1',
                'robot_id': 1,
                'agent': {'position': {'x': 2, 'y': 0.9, 'z': 3}},
                'visible_objects': [{'objectId': 'Lettuce|1', 'objectType': 'Lettuce'}],
            },
            {
                'agent_id': '2',
                'robot_id': 2,
                'agent': {'position': {'x': 4, 'y': 0.9, 'z': 5}},
                'visible_objects': [],
            },
        ]
        response = {
            'status': 'success',
            'task_id': 'relay-1',
            'post_agent_states': post_states,
            'result': {'closed_loop_result': {'status': 'success'}},
        }
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            self.module, 'urlopen', return_value=FakeResponse(response)
        ):
            report = self.module.TaskExecutionServiceAdapter(self.args()).receive_execution_report(
                self.payload(), Path(tmp)
            )

        self.assertEqual(report['agent_states'], post_states)
        self.assertEqual(report['post_agent_states'], post_states)
        self.assertEqual(
            [state['robot_id'] for state in report['post_agent_states']],
            [0, 1, 2],
        )
        self.assertNotEqual(report['agent_states'], self.payload()['agent_states'])

    def test_completion_agent_comes_from_actual_relay_executor(self):
        response = {
            'status': 'success',
            'result': {
                'primary_robot_id': 1,
                'closed_loop_result': {'status': 'success'},
                'closed_loop_trace': [{'executor_robot_id': 0}],
            },
        }
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            self.module, 'urlopen', return_value=FakeResponse(response)
        ):
            report = self.module.TaskExecutionServiceAdapter(self.args()).receive_execution_report(
                self.payload(), Path(tmp)
            )

        self.assertEqual(report['task_statuses'][0]['agent_id'], '1')
        self.assertEqual(report['task_statuses'][0]['completion_agent_id'], '0')



    def test_execution_state_changes_add_object_changes(self):
        tomato_id = 'Tomato|-00.50|+01.00|+00.25'
        response = {
            'status': 'success',
            'task_id': 'relay-1',
            'result': {
                'closed_loop_result': {'status': 'success'},
                'execution_state_changes': {
                    'object_changes': [
                        {
                            'objectId': 'Fridge|-02.10|+00.00|+01.07',
                            'objectType': 'Fridge',
                            'source_action': 'OpenObject',
                            'action_index': 0,
                            'robot_id': 0,
                            'isOpen': True,
                            'state_changed': True,
                            'after': {'objectId': 'Fridge|-02.10|+00.00|+01.07', 'objectType': 'Fridge', 'isOpen': True},
                        },
                        {
                            'objectId': tomato_id,
                            'objectType': 'Tomato',
                            'source_action': 'PutObject',
                            'action_index': 1,
                            'robot_id': 0,
                            'position': {'x': 0.5, 'y': 0.95, 'z': -2.0},
                            'parentReceptacles': ['CounterTop|+00.50|+00.95|-02.00'],
                            'state_changed': True,
                            'after': {'objectId': tomato_id, 'objectType': 'Tomato', 'position': {'x': 0.5, 'y': 0.95, 'z': -2.0}},
                        },
                    ],
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp, patch.object(self.module, 'urlopen', return_value=FakeResponse(response)):
            report = self.module.TaskExecutionServiceAdapter(self.args()).receive_execution_report(self.payload(), Path(tmp))

        self.assertEqual(report['execution']['completed_task_ids'], ['T1'])
        self.assertEqual(len(report['object_changes']), 2)
        self.assertEqual(report['object_changes'][0]['source'], 'task_execution_service_execute_action')
        self.assertEqual(report['object_changes'][0]['source_action'], 'OpenObject')
        self.assertIs(report['object_changes'][0]['isOpen'], True)
        self.assertEqual(report['object_changes'][1]['source_action'], 'PutObject')
        self.assertEqual(report['object_changes'][1]['position'], {'x': 0.5, 'y': 0.95, 'z': -2.0})
        self.assertEqual(report['object_changes'][1]['parentReceptacles'], ['CounterTop|+00.50|+00.95|-02.00'])
        observed = self.module.adapter_object_changes_to_observed_nodes(report['object_changes'])
        self.assertEqual(observed[0]['last_reported_state']['source_action'], 'OpenObject')
        self.assertEqual(observed[1]['bbox_center'], [0.5, 0.95, -2.0])

    def test_scenegraph_update_prefers_json_object_relations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_graph = root / 'scene_graph.json'
            binary_relations = root / 'cfslam_scenegraph_edges.pkl'
            object_relations = root / 'cfslam_object_relations.json'
            scene_graph.write_text('[]', encoding='utf-8')
            binary_relations.write_bytes(b'\x80\x04]\x94.')
            object_relations.write_text('[]', encoding='utf-8')

            captured = {}

            def fake_update_scenegraph_files(**kwargs):
                captured.update(kwargs)
                return {
                    'scene_graph': str(scene_graph),
                    'relations': str(object_relations),
                }

            current = {
                'scene_graph': str(scene_graph),
                'relations': str(binary_relations),
                'object_relations': str(object_relations),
            }
            with patch.object(
                self.module,
                'update_scenegraph_files',
                side_effect=fake_update_scenegraph_files,
            ):
                self.module.update_scenegraph_from_report(
                    current,
                    {
                        'object_changes': [
                            {
                                'objectId': 'Cabinet|1',
                                'objectType': 'Cabinet',
                                'isOpen': True,
                                'before': {'isOpen': False},
                                'after': {'isOpen': True},
                            }
                        ]
                    },
                    root / 'updated',
                )

            self.assertEqual(captured['stored_relations_path'], object_relations)
            self.assertTrue(captured['append_observed_only'])
            self.assertEqual(captured['object_deltas'][0]['object_id'], 'Cabinet|1')
            self.assertIs(captured['object_deltas'][0]['changed_fields']['isOpen']['after'], True)

    def test_needs_upstream_planning_maps_to_wait_retry(self):
        response = {
            'status': 'needs_upstream_planning',
            'task_id': 'relay-1',
            'result': {'closed_loop_result': {'status': 'needs_upstream_planning', 'failure_code': 'target_not_visible', 'reason': 'Fridge is not visible'}},
        }
        with tempfile.TemporaryDirectory() as tmp, patch.object(self.module, 'urlopen', return_value=FakeResponse(response)):
            report = self.module.TaskExecutionServiceAdapter(self.args()).receive_execution_report(self.payload(), Path(tmp))

        self.assertEqual(report['execution']['completed_task_ids'], [])
        self.assertEqual(report['task_statuses'][0]['status'], 'wait_retry')
        self.assertEqual(report['task_statuses'][0]['failure_code'], 'target_not_visible')
        self.assertEqual(report['task_statuses'][0]['recommended_recovery'], 'retry_original_task')
        self.assertEqual(report['task_statuses'][0]['recovery_target_type'], 'Fridge')
        self.assertIn('target_not_visible', report['task_statuses'][0]['reason'])
        self.assertIn('Fridge is not visible', report['task_statuses'][0]['reason'])
        self.assertEqual(report['feedback'][0]['type'], 'task_service_not_completed')

    def test_robot_blocking_failure_requests_agent_switch(self):
        response = {
            'status': 'needs_upstream_planning',
            'task_id': 'relay-1',
            'result': {
                'closed_loop_result': {
                    'status': 'needs_upstream_planning',
                    'strategy': 'goto',
                    'failure_code': 'execution_failed_after_replans',
                    'reason': (
                        'navigation action execution failed after replanning; '
                        'Agent 1 is blocking Agent 0 from moving'
                    ),
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            self.module,
            'urlopen',
            return_value=FakeResponse(response),
        ):
            report = self.module.TaskExecutionServiceAdapter(self.args()).receive_execution_report(
                self.payload(),
                Path(tmp),
            )

        status = report['task_statuses'][0]
        self.assertTrue(status['recoverable'])
        self.assertEqual(status['recommended_recovery'], 'switch_agent')
        self.assertEqual(status['blocked_by_agent_id'], '1')
        self.assertEqual(status['blocked_agent_id'], '0')
        self.assertEqual(
            self.module.blocking_agent_pair(status['reason']),
            ('1', '0'),
        )

    def test_blocked_recovery_task_avoids_same_agent_on_retry(self):
        task = {
            'id': 'F_T1_01',
            'action': 'find',
            'grounding': {'object_tags': ['Bread']},
            'runtime': {'source_task_id': 'T1'},
            'depends_on': [],
        }
        graph = self.module.rebuild_task_graph({'task': 'find Bread', 'flat_tasks': [task]})
        updated = self.module.apply_execution_feedback_to_task_graph(
            graph,
            [{
                'subtask_id': 'F_T1_01',
                'agent_id': '0',
                'status': 'wait_retry',
                'failure_code': 'execution_failed_after_replans',
                'blocked_by_agent_id': '1',
                'blocked_agent_id': '0',
                'reason': 'Agent 1 is blocking Agent 0 from moving',
            }],
        )
        retry_task = updated['flat_tasks'][0]

        self.assertEqual(retry_task['runtime']['avoid_agent_ids'], ['0'])
        self.assertEqual(retry_task['runtime']['last_blocked_by_agent_id'], '1')

    def test_allocation_block_is_converted_to_runtime_replanning_facts(self):
        graph = self.module.rebuild_task_graph({
            'task': 'place mug',
            'flat_tasks': [{'id': 'T1', 'action': 'place', 'depends_on': []}],
        })
        plan = {
            'state': 'blocked',
            'task_graph_version': 2,
            'graph_fingerprint': 'sha256:graph',
            'diagnostics': {'remaining_task_ids': ['T1']},
            'blocking': {
                'code': 'no_eligible_agent',
                'task_ids': ['T1'],
                'agent_inventories': {'0': []},
                'conflicts': [{'task_id': 'T1'}],
                'recommended_recovery': 'replan_task_graph',
            },
        }
        statuses = self.module.allocation_blocking_statuses(plan, graph)
        self.assertEqual([item['subtask_id'] for item in statuses], ['T1'])
        self.assertEqual(statuses[0]['trigger_stage'], 'allocation')
        self.assertEqual(statuses[0]['recommended_recovery'], 'replan_task_graph')
        self.assertTrue(
            self.module.execution_status_requires_task_graph_replan(statuses[0])
        )

    def test_transport_error_maps_to_wait_retry(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(self.module, 'urlopen', side_effect=URLError('connection refused')):
            report = self.module.TaskExecutionServiceAdapter(self.args()).receive_execution_report(self.payload(), Path(tmp))

        self.assertEqual(report['task_statuses'][0]['status'], 'wait_retry')
        self.assertIn('connection refused', report['task_statuses'][0]['reason'])
        self.assertEqual(report['execution']['traces'][0]['task_service_response']['status'], 'failed')

    def test_retry_count_is_monotonic_and_exhausts_budget(self):
        completed, failed, retries = set(), set(), {}
        status = [{'subtask_id': 'T1', 'status': 'wait_retry', 'reason': 'not visible'}]

        first = self.module.apply_progress(status, completed, failed, retries, max_task_retries=3)
        second = self.module.apply_progress(status, completed, failed, retries, max_task_retries=3)
        third = self.module.apply_progress(status, completed, failed, retries, max_task_retries=3)

        self.assertEqual(first['wait_retry_this_loop'], ['T1'])
        self.assertEqual(second['wait_retry_this_loop'], ['T1'])
        self.assertEqual(third['failed_this_loop'], ['T1'])
        self.assertIn('T1', failed)
        self.assertNotIn('T1', retries)

    def test_runtime_failure_retries_original_task_without_inserting_find(self):
        task = {
            'id': 'T1',
            'action': 'place',
            'grounding': {'status': 'grounded', 'object_tags': ['bread']},
        }
        matched_subgraph = {'nodes': [{'object_tag': 'bread'}]}
        self.assertFalse(self.module.task_requires_runtime_find(task, matched_subgraph))

        graph = {'task': 'place bread', 'flat_tasks': [task]}
        updated = self.module.apply_execution_feedback_to_task_graph(
            graph,
            [{
                'subtask_id': 'T1',
                'status': 'wait_retry',
                'failure_code': 'target_not_visible',
                'reason': 'Bread is not visible',
            }],
        )

        self.assertEqual([item['id'] for item in updated['flat_tasks']], ['T1'])
        updated_task = updated['flat_tasks'][0]
        self.assertEqual(updated_task['grounding']['status'], 'grounded')
        self.assertNotIn('execution_status', updated_task['grounding'])
        self.assertEqual(updated_task['runtime']['retry_strategy'], 'retry_original_task')
        self.assertEqual(updated_task['runtime']['last_execution_failure_code'], 'target_not_visible')
        self.assertFalse(self.module.task_requires_runtime_find(updated_task, matched_subgraph))

    def test_runtime_retry_prefers_failed_assignment_agent_without_inserting_find(self):
        task = {
            'id': 'T1',
            'action': 'pick',
            'grounding': {'status': 'grounded', 'object_tags': ['Bread']},
            'depends_on': [],
        }
        graph = self.module.rebuild_task_graph({'task': 'pick bread', 'flat_tasks': [task]})
        updated = self.module.apply_execution_feedback_to_task_graph(
            graph,
            [{
                'subtask_id': 'T1',
                'agent_id': '1',
                'status': 'wait_retry',
                'failure_code': 'target_not_visible',
                'recovery_target_type': 'Bread',
                'reason': "target_not_visible: 'Bread' is not visible",
            }],
        )

        self.assertEqual([item['id'] for item in updated['flat_tasks']], ['T1'])
        updated_task = updated['flat_tasks'][0]
        self.assertEqual(updated_task['grounding']['execution_preferred_agent_id'], '1')
        self.assertEqual(updated_task['runtime']['last_recovery_target_type'], 'Bread')
        self.assertEqual(updated_task['runtime']['retry_strategy'], 'retry_original_task')

    def test_runtime_retry_records_failed_destination_without_inserting_find(self):
        task = {
            'id': 'T4',
            'action': 'place',
            'grounding': {
                'status': 'grounded',
                'object_tags': ['bread'],
                'relation_texts': ['in fridge'],
            },
            'depends_on': ['T1'],
        }
        graph = self.module.rebuild_task_graph({'task': 'place bread in fridge', 'flat_tasks': [task]})
        updated = self.module.apply_execution_feedback_to_task_graph(
            graph,
            [{
                'subtask_id': 'T4',
                'status': 'wait_retry',
                'failure_code': 'target_not_visible',
                'recovery_target_type': 'Fridge',
                'recovery_agent_id': '1',
                'reason': "target_not_visible: 'Fridge' is not visible",
            }],
        )

        self.assertEqual([item['id'] for item in updated['flat_tasks']], ['T4'])
        updated_task = updated['flat_tasks'][0]
        self.assertEqual(updated_task['runtime']['last_recovery_target_type'], 'Fridge')
        self.assertEqual(updated_task['grounding']['execution_preferred_agent_id'], '1')

    def test_recovery_find_keeps_source_tags_when_feedback_has_no_target_type(self):
        task = {
            'id': 'T1',
            'action': 'pick',
            'grounding': {'status': 'unresolved', 'object_tags': ['Bread']},
            'depends_on': [],
        }
        graph = self.module.rebuild_task_graph({'task': 'pick bread', 'flat_tasks': [task]})

        _, find_task, inserted = self.module.insert_find_before_task(
            graph,
            task,
            reason='No matching node was found.',
            max_find_insertions=2,
            completed=set(),
        )

        self.assertTrue(inserted)
        self.assertEqual(find_task['grounding']['object_tags'], ['Bread'])

    def test_recovery_target_prefers_structured_field_then_parses_legacy_reason(self):
        structured = {
            'status': 'needs_upstream_planning',
            'result': {
                'closed_loop_result': {
                    'failure_code': 'target_not_visible',
                    'recovery_target_type': 'Cabinet',
                    'reason': "'Fridge' is not visible",
                },
            },
        }
        legacy = {
            'status': 'needs_upstream_planning',
            'result': {
                'closed_loop_result': {
                    'failure_code': 'target_not_visible',
                    'reason': "'Fridge' is not visible to any robot",
                },
            },
        }

        self.assertEqual(self.module.task_service_response_recovery_target_type(structured), 'Cabinet')
        self.assertEqual(self.module.task_service_response_recovery_target_type(legacy), 'Fridge')

    def test_object_not_actionable_recovers_target_receptacle_with_holder_robot(self):
        response = {
            'status': 'needs_upstream_planning',
            'result': {
                'closed_loop_result': {
                    'failure_code': 'object_not_actionable',
                    'reason': (
                        "'Tomato' is visible, but robot 1: target receptacle "
                        "'Fridge' is not visible to robot 1"
                    ),
                },
            },
        }

        self.assertEqual(self.module.task_service_response_recovery_target_type(response), 'Fridge')
        self.assertEqual(self.module.task_service_response_recovery_agent_id(response), '1')

    def test_successful_find_releases_source_task_and_binds_object_id(self):
        graph = {
            'task': 'place bread',
            'flat_tasks': [
                {
                    'id': 'F_T1_01',
                    'action': 'find',
                    'grounding': {'object_tags': ['bread']},
                    'depends_on': [],
                    'runtime': {'source_task_id': 'T1'},
                },
                {
                    'id': 'T1',
                    'action': 'place',
                    'grounding': {
                        'object_tags': ['bread'],
                        'execution_status': 'unresolved',
                        'execution_failure_code': 'target_not_visible',
                    },
                    'depends_on': ['F_T1_01'],
                },
            ],
        }
        updated = self.module.apply_execution_feedback_to_task_graph(
            graph,
            [{'subtask_id': 'F_T1_01', 'status': 'success'}],
            [{'objectId': 'Bread|1', 'objectType': 'Bread', 'source': 'relay_find_discovery'}],
        )

        source = next(task for task in updated['flat_tasks'] if task['id'] == 'T1')
        self.assertEqual(source['grounding']['execution_status'], 'discovered')
        self.assertEqual(source['grounding']['object_ids'], ['Bread|1'])
        self.assertNotIn('execution_failure_code', source['grounding'])

    def test_discovered_objects_are_converted_to_memory_changes(self):
        response = {
            'status': 'success',
            'result': {
                'closed_loop_result': {'status': 'success'},
                'discovered_objects': [{
                    'objectId': 'Bread|1',
                    'objectType': 'Bread',
                    'position': {'x': 1, 'y': 2, 'z': 3},
                    'visible_by_agent_ids': [1],
                }],
            },
        }
        changes = self.module.task_service_response_object_changes(response)
        self.assertEqual(changes[0]['source'], 'relay_find_discovery')
        self.assertEqual(changes[0]['objectId'], 'Bread|1')
        self.assertEqual(changes[0]['visible_by_agent_ids'], [1])

    def test_allocation_blocking_budget_key_is_stable_and_version_scoped(self):
        plan = {
            'task_graph_version': 1,
            'graph_fingerprint': 'sha256:graph',
            'blocking': {'code': 'resource_deadlock', 'task_ids': ['T1']},
        }
        first = self.module.allocation_blocking_budget_key(plan)
        self.assertEqual(first, self.module.allocation_blocking_budget_key(plan))
        changed = dict(plan, task_graph_version=2)
        self.assertNotEqual(first, self.module.allocation_blocking_budget_key(changed))

    def test_duplicate_agent_map_is_rejected_instead_of_overwritten(self):
        with self.assertRaisesRegex(ValueError, 'more than one task'):
            self.module.assignments_to_agent_task_map([
                {'agent_id': '0', 'subtask': {'id': 'T1'}},
                {'agent_id': '0', 'subtask': {'id': 'T2'}},
            ])


class InitialFindOnlyRepairTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def find_only_graph(self):
        return {
            'task': 'close the fridge.',
            'reasoning_summary': 'planner inserted search because target was unresolved',
            'flat_tasks': [
                {
                    'id': 'T1',
                    'name': 'Find or verify target objects',
                    'description': 'Search the current scene for task-relevant objects before acting: fridge',
                    'action': 'find',
                    'grounding': {'status': 'unresolved', 'object_tags': ['fridge']},
                    'depends_on': [],
                }
            ],
        }

    def test_interaction_find_only_graph_preserves_original_task_after_find(self):
        repaired = self.module.repair_initial_find_only_task_graph(self.find_only_graph(), 'close the fridge.')
        tasks = repaired['flat_tasks']

        self.assertEqual([task['id'] for task in tasks], ['F_T1_01', 'T1'])
        self.assertEqual(tasks[0]['action'], 'find')
        self.assertEqual(tasks[1]['action'], 'close')
        self.assertEqual(tasks[1]['depends_on'], ['F_T1_01'])
        self.assertEqual(tasks[1]['description'], 'close the fridge.')
        self.assertTrue(tasks[1]['runtime']['recovered_from_initial_find_only'])

    def test_navigation_find_only_graph_is_not_repaired(self):
        graph = self.find_only_graph()
        repaired = self.module.repair_initial_find_only_task_graph(graph, 'go to fridge.')

        self.assertEqual(repaired, graph)


if __name__ == '__main__':
    unittest.main()


class Task6RoleGroundingTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_place_missing_destination_inserts_role_specific_find(self):
        task = {
            'id': 'T1',
            'action': 'place',
            'grounding': {'source_object_tags': ['vase'], 'source_object_ids': ['Vase|1'], 'destination_object_tags': ['table']},
            'depends_on': [],
        }
        self.assertEqual(self.module.missing_grounding_role(task, {}), ('destination', ['table']))
        graph = self.module.rebuild_task_graph({'task': 'put vase on table', 'flat_tasks': [task]})
        recovered, find_task, inserted = self.module.insert_find_before_task(
            graph,
            task,
            reason='destination is unresolved',
            max_find_insertions=2,
            completed=set(),
            grounding_role='destination',
            object_tags_override=['table'],
        )
        self.assertTrue(inserted)
        self.assertEqual(find_task['name'], 'Find table')
        self.assertEqual(find_task['grounding']['object_tags'], ['table'])
        self.assertEqual(find_task['runtime']['grounding_role'], 'destination')

    def test_semantic_destination_binding_propagates_to_all_places(self):
        tasks = [
            {'id': 'T1', 'action': 'place', 'grounding': {'source_object_tags': ['vase'], 'destination_object_tags': ['table']}},
            {'id': 'T2', 'action': 'place', 'grounding': {'source_object_tags': ['tissue box'], 'destination_object_tags': ['table']}},
            {'id': 'T3', 'action': 'place', 'grounding': {'source_object_tags': ['remote control'], 'destination_object_tags': ['table']}},
        ]
        graph = {'task': 'put objects on table', 'flat_tasks': tasks}
        changed = self.module.propagate_semantic_destination_binding(graph, {
            'status': 'resolved', 'semantic_input': 'table',
            'chosen_type': 'CoffeeTable', 'chosen_object_id': 'CoffeeTable|1',
        })
        self.assertTrue(changed)
        for task in tasks:
            self.assertEqual(task['grounding']['destination_object_ids'], ['CoffeeTable|1'])
            self.assertEqual(task['grounding']['destination_object_tags'], ['CoffeeTable'])
            self.assertEqual(task['grounding']['object_tags'][-1], 'CoffeeTable')

    def test_open_feedback_binds_task19_place_chain_and_final_close(self):
        drawer_b = 'Drawer|+00.81|+00.48|-01.16'
        open_task = {
            'id': 'T2', 'action': 'open', 'depends_on': [],
            'grounding': {
                'source_selector': {'quantifier': 'one', 'object_types': ['Drawer']},
                'source_object_tags': ['Drawer'], 'source_object_ids': [],
            },
        }
        tasks = [open_task]
        previous = 'T2'
        for index, object_type in enumerate(
            ['ButterKnife', 'Fork', 'Knife', 'Ladle', 'Spatula', 'Spoon'],
            start=1,
        ):
            task_id = f'T3_{index}'
            tasks.append({
                'id': task_id, 'action': 'place', 'depends_on': [previous],
                'grounding': {
                    'source_selector': {'quantifier': 'one', 'object_types': [object_type]},
                    'destination_selector': {'quantifier': 'one', 'object_types': ['Drawer']},
                    'source_object_tags': [object_type],
                    'source_object_ids': [f'{object_type}|1'],
                    'destination_object_tags': ['Drawer'],
                    'destination_object_ids': [],
                },
            })
            previous = task_id
        tasks.append({
            'id': 'T4', 'action': 'close', 'depends_on': [previous],
            'grounding': {
                'source_selector': {'quantifier': 'one', 'object_types': ['Drawer']},
                'source_object_tags': ['Drawer'], 'source_object_ids': [],
            },
        })
        graph = self.module.rebuild_task_graph({'task': 'put all silverware in one drawer', 'flat_tasks': tasks})

        updated = self.module.apply_execution_feedback_to_task_graph(
            graph,
            [{'subtask_id': 'T2', 'status': 'success'}],
            [{
                'source_task_id': 'T2', 'source_action': 'OpenObject',
                'objectId': drawer_b, 'objectType': 'Drawer', 'isOpen': True,
            }],
        )

        indexed = {task['id']: task for task in updated['flat_tasks']}
        self.assertEqual(indexed['T2']['grounding']['source_object_ids'], [drawer_b])
        for task_id in [f'T3_{index}' for index in range(1, 7)]:
            self.assertEqual(indexed[task_id]['grounding']['destination_object_ids'], [drawer_b])
            binding = indexed[task_id]['runtime']['selector_binding']
            self.assertEqual(binding['source_task_id'], 'T2')
            self.assertEqual(binding['source_action'], 'OpenObject')
            self.assertEqual(binding['source'], 'execution_feedback')
            self.assertEqual(binding['object_id'], drawer_b)
            self.assertEqual(binding['role'], 'destination')
        self.assertEqual(indexed['T4']['grounding']['source_object_ids'], [drawer_b])
        self.assertEqual(indexed['T4']['runtime']['selector_binding']['role'], 'source')

    def test_open_feedback_respects_explicit_ids_and_independent_open_branches(self):
        def drawer_grounding(role, object_ids=None, quantifier='one'):
            return {
                f'{role}_selector': {'quantifier': quantifier, 'object_types': ['Drawer']},
                f'{role}_object_tags': ['Drawer'],
                f'{role}_object_ids': list(object_ids or []),
            }

        tasks = [
            {'id': 'T1', 'action': 'open', 'depends_on': [], 'grounding': drawer_grounding('source')},
            {
                'id': 'T2', 'action': 'place', 'depends_on': ['T1'],
                'grounding': drawer_grounding('destination', ['Drawer|explicit']),
            },
            {'id': 'T3', 'action': 'close', 'depends_on': ['T2'], 'grounding': drawer_grounding('source')},
            {'id': 'T4', 'action': 'open', 'depends_on': [], 'grounding': drawer_grounding('source')},
            {
                'id': 'T5', 'action': 'place', 'depends_on': ['T4'],
                'grounding': drawer_grounding('destination'),
            },
            {'id': 'T6', 'action': 'open', 'depends_on': ['T1'], 'grounding': drawer_grounding('source')},
            {
                'id': 'T7', 'action': 'place', 'depends_on': ['T6'],
                'grounding': drawer_grounding('destination'),
            },
        ]
        graph = self.module.rebuild_task_graph({'task': 'independent drawers', 'flat_tasks': tasks})

        updated = self.module.apply_execution_feedback_to_task_graph(
            graph,
            [{'subtask_id': 'T1', 'status': 'success'}],
            [{
                'source_task_id': 'T1', 'source_action': 'OpenObject',
                'objectId': 'Drawer|goto', 'objectType': 'Drawer', 'isOpen': True,
            }],
        )

        indexed = {task['id']: task for task in updated['flat_tasks']}
        self.assertEqual(indexed['T1']['grounding']['source_object_ids'], ['Drawer|goto'])
        self.assertEqual(indexed['T2']['grounding']['destination_object_ids'], ['Drawer|explicit'])
        self.assertEqual(indexed['T3']['grounding']['source_object_ids'], [])
        self.assertEqual(indexed['T4']['grounding']['source_object_ids'], [])
        self.assertEqual(indexed['T5']['grounding']['destination_object_ids'], [])
        self.assertEqual(indexed['T6']['grounding']['source_object_ids'], [])
        self.assertEqual(indexed['T7']['grounding']['destination_object_ids'], [])

    def test_all_selector_expansion_is_not_rebound_by_open_feedback(self):
        task = {
            'id': 'T1', 'action': 'open', 'depends_on': [],
            'grounding': {
                'source_selector': {'quantifier': 'all', 'object_types': ['Drawer']},
                'source_object_tags': ['Drawer'], 'source_object_ids': ['Drawer|1'],
            },
        }
        graph = self.module.rebuild_task_graph({'task': 'open all drawers', 'flat_tasks': [task]})

        updated = self.module.apply_execution_feedback_to_task_graph(
            graph,
            [{'subtask_id': 'T1', 'status': 'success'}],
            [{
                'source_task_id': 'T1', 'source_action': 'OpenObject',
                'objectId': 'Drawer|1', 'objectType': 'Drawer', 'isOpen': True,
            }],
        )

        updated_task = updated['flat_tasks'][0]
        self.assertEqual(updated_task['grounding']['source_object_ids'], ['Drawer|1'])
        self.assertNotIn('selector_binding', updated_task.get('runtime', {}))

    def test_exhausted_placement_recovery_is_terminal_and_persisted(self):
        task = {
            'id': 'T1', 'action': 'place', 'depends_on': [],
            'grounding': {
                'source_object_tags': ['Mug'],
                'source_object_ids': ['Mug|1'],
                'destination_object_tags': ['Cabinet'],
                'destination_object_ids': ['Cabinet|failed'],
            },
        }
        graph = self.module.rebuild_task_graph({'task': 'store mug', 'flat_tasks': [task]})
        status = {
            'subtask_id': 'T1',
            'status': self.module.FAILURE,
            'failure_code': 'no_valid_placement',
            'reason': 'No valid positions to place object found',
            'excluded_destination_object_ids': ['Cabinet|failed'],
            'recovery_history': [{'strategy': 'try_alternate_receptacle_instance', 'status': 'failed'}],
        }

        updated = self.module.apply_execution_feedback_to_task_graph(graph, [status], [])
        runtime = updated['flat_tasks'][0]['runtime']
        self.assertEqual(runtime['retry_strategy'], 'replan_task_graph')
        self.assertEqual(runtime['excluded_destination_object_ids'], ['Cabinet|failed'])
        completed, failed, retries = set(), set(), {}
        delta = self.module.apply_progress([status], completed, failed, retries, max_task_retries=6)
        self.assertEqual(delta['failed_this_loop'], ['T1'])
        self.assertEqual(retries, {})


    def test_semantic_resolution_cannot_replace_execution_feedback_binding(self):
        task = {
            'id': 'T1', 'action': 'place',
            'grounding': {
                'source_object_tags': ['Fork'],
                'destination_object_tags': ['Drawer'],
                'destination_object_ids': ['Drawer|runtime'],
            },
            'runtime': {'selector_binding': {'source': 'execution_feedback'}},
        }
        graph = {'task': 'place fork in drawer', 'flat_tasks': [task]}

        changed = self.module.propagate_semantic_destination_binding(graph, {
            'status': 'resolved', 'semantic_input': 'Drawer',
            'chosen_type': 'Drawer', 'chosen_object_id': 'Drawer|other',
        })

        self.assertFalse(changed)
        self.assertEqual(task['grounding']['destination_object_ids'], ['Drawer|runtime'])
