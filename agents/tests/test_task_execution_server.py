import contextlib
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

import task_execution_server
from task_execution_server import TaskExecutionRuntimeConfig, TaskExecutionService, model_shard_error


class FakeEngine:
    def __init__(self):
        self.argv = None

    def parse_args(self, argv):
        self.argv = argv
        return Namespace()

    def run(self, args):
        assert getattr(args, "_qwen_backend") is not None
        print(json.dumps({"task_id": "task-1", "closed_loop_result": {"status": "success", "step_count": 2}}))
        return 0


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeNormalizerBackend:
    def __init__(self, arguments):
        self.arguments = arguments
        self.calls = []

    def generate_with_tools(self, messages, tools):
        self.calls.append({"messages": messages, "tools": tools})
        return json.dumps(
            {
                "name": "normalize_incoming_task",
                "arguments": self.arguments,
            }
        )


class SequentialToolBackend:
    def __init__(self, *arguments):
        self.arguments = list(arguments)
        self.calls = []

    def generate_with_tools(self, messages, tools):
        self.calls.append({"messages": messages, "tools": tools})
        index = min(len(self.calls) - 1, len(self.arguments) - 1)
        arguments = self.arguments[index] if self.arguments else {}
        return json.dumps({"name": "normalize_incoming_task", "arguments": arguments})


def service_config(temp_dir: str | Path = "/tmp") -> TaskExecutionRuntimeConfig:
    return TaskExecutionRuntimeConfig(
        receiver_url="http://127.0.0.1:19000",
        model_path="models/Qwen3.5-4B",
        device="cuda",
        device_map="auto",
        dtype="float16",
        max_new_tokens=64,
        temperature=0.1,
        send_timeout=60.0,
        output_dir=Path(temp_dir),
        max_replan_steps=10,
        relay_agent_max_turns=8,
        max_actions=8,
    )


class TaskExecutionServiceTest(unittest.TestCase):
    def test_safe_structured_actions_preserve_controller_arguments(self):
        cases = {
            "push": ("PushObject", {"moveMagnitude": 200.0}),
            "pull": ("PullObject", {"moveMagnitude": 100.0}),
            "move_held": (
                "MoveHeldObject",
                {"right": 0.1, "up": 0.0, "ahead": 0.0},
            ),
            "fill": ("FillObjectWithLiquid", {"fillLiquid": "water"}),
        }
        arguments = {
            "push": {"moveMagnitude": 200},
            "pull": {"moveMagnitude": 100},
            "move_held": {"right": 0.1},
            "fill": {"fillLiquid": "water"},
        }
        for logical_action, (native_action, expected_args) in cases.items():
            with self.subTest(action=logical_action):
                normalized = task_execution_server.normalize_structured_subtask(
                    {
                        "action": logical_action,
                        "action_args": arguments[logical_action],
                        "grounding": {
                            "source_object_tags": ["Mug"],
                            "source_object_ids": ["Mug|1"],
                        },
                    },
                    f"{logical_action} the mug",
                    ["Mug"],
                )
                self.assertIsNotNone(normalized)
                step = next(
                    item for item in normalized["intentSteps"]
                    if item["action"] == native_action
                )
                self.assertEqual(step["actionArgs"], expected_args)
                if logical_action == "move_held":
                    self.assertEqual(
                        [item["action"] for item in normalized["intentSteps"]],
                        ["MoveHeldObject"],
                    )

    def test_drop_structured_action_has_no_navigation_or_object_id(self):
        normalized = task_execution_server.normalize_structured_subtask(
            {
                "action": "drop",
                "grounding": {
                    "source_object_tags": ["Mug"],
                    "source_object_ids": ["Mug|1"],
                },
            },
            "drop the held mug",
            ["Mug"],
        )
        self.assertIsNotNone(normalized)
        self.assertEqual(
            normalized["intentSteps"],
            [{
                "order": 1,
                "action": "DropHandObject",
                "objectType": "Mug",
                "targetType": None,
            }],
        )
    def test_fetch_receiver_agent_states_returns_fresh_visible_state_and_partial_errors(self):
        receiver_state = {
            "sceneName": "FloorPlan1",
            "step": 42,
            "agent": {
                "position": {"x": 1.0, "y": 0.9, "z": 2.0},
                "rotation": {"x": 0, "y": 90, "z": 0},
                "horizon": 30,
            },
            "robots": [{
                "robot_id": 0,
                "name": "robot_0",
                "inventory": [{"objectId": "Bread|1", "objectType": "Bread"}],
                "last_action": "PickupObject",
                "last_success": True,
            }],
            "inventory": [{"objectId": "Bread|1", "objectType": "Bread"}],
            "objects": [
                {
                    "id": "Lettuce|1",
                    "type": "Lettuce",
                    "visible": True,
                    "position": {"x": 1.2, "y": 0.8, "z": 2.1},
                },
                {"objectId": "Tomato|1", "objectType": "Tomato", "visible": False},
            ],
        }
        with patch.object(
            task_execution_server,
            "urlopen",
            side_effect=[FakeResponse(receiver_state), URLError("robot 1 unavailable")],
        ):
            states, errors = task_execution_server.fetch_receiver_agent_states(
                "http://127.0.0.1:19000", [0, 1], 2.0
            )

        self.assertEqual(len(states), 1)
        self.assertEqual(states[0]["agent_id"], "0")
        self.assertEqual(states[0]["state_step"], 42)
        self.assertEqual(states[0]["inventoryObjects"][0]["objectType"], "Bread")
        self.assertEqual([item["objectType"] for item in states[0]["visible_objects"]], ["Lettuce"])
        self.assertEqual(
            [item["objectType"] for item in states[0]["scene_object_catalog"]],
            ["Lettuce", "Tomato"],
        )
        self.assertEqual(errors[0]["robot_id"], 1)

    def test_non_dry_run_response_includes_authoritative_post_states(self):
        engine = FakeEngine()
        backend = FakeNormalizerBackend({
            "normalized_task": "open the Fridge.",
            "intentSteps": [
                {"order": 1, "action": "OpenObject", "objectType": "Fridge", "targetType": None}
            ],
            "confidence": "high",
            "reason": "open requested receptacle",
        })
        post_states = [
            {"agent_id": "0", "state_step": 9, "visible_objects": []},
            {"agent_id": "1", "state_step": 9, "visible_objects": []},
        ]
        config = service_config()
        service = TaskExecutionService(engine, backend, config)
        with patch.object(
            task_execution_server, "fetch_receiver_state_object_types", return_value=(["Fridge"], None)
        ), patch.object(
            task_execution_server,
            "fetch_receiver_health",
            return_value={
                "reachable": True,
                "controller_ready": True,
                "robot_ids": [0, 1],
            },
        ), patch.object(
            task_execution_server, "fetch_receiver_agent_states", return_value=(post_states, [])
        ) as fetch_states:
            response = service.execute_task({
                "task": "open the fridge",
                "primary_robot_id": 0,
            })

        self.assertEqual(response["post_agent_states"], post_states)
        self.assertEqual(
            response["coordinator_agent_registry"]["robot_ids"],
            [0, 1],
        )
        self.assertEqual(
            response["coordinator_agent_registry"]["source"],
            "receiver_health",
        )
        fetch_states.assert_called_once_with(
            config.receiver_url,
            [0, 1],
            config.send_timeout,
        )

    def test_health_requires_backend_and_ready_receiver(self):
        old_fetch = task_execution_server.fetch_receiver_health
        task_execution_server.fetch_receiver_health = lambda receiver_url, timeout: {
            "reachable": True,
            "controller_ready": True,
            "robot_ids": [0, 1],
        }
        try:
            service = TaskExecutionService(FakeEngine(), FakeNormalizerBackend({}), service_config())
            health = service.health()
        finally:
            task_execution_server.fetch_receiver_health = old_fetch

        self.assertEqual(health["status"], "ready")
        self.assertTrue(health["backend_ready"])
        self.assertTrue(health["receiver"]["controller_ready"])
        self.assertEqual(health["receiver"]["robot_ids"], [0, 1])

    def test_parse_task_normalizer_ignores_tool_schema_block(self):
        output = (
            'system\n<tools>{"type":"function","function":{"name":"normalize_incoming_task",'
            '"parameters":{"type":"object"}}}</tools>\n'
            '<tool_call>{"name":"normalize_incoming_task","arguments":'
            '{"normalized_task":"look down.","intentSteps":[{"order":1,"action":"LookDown","objectType":null,"targetType":null}],"confidence":"high","reason":"preserve requested look action"}}}</tool_call>'
        )
        tool_call = task_execution_server.parse_task_normalizer_tool_call(output)
        self.assertEqual(tool_call["name"], "normalize_incoming_task")
        self.assertEqual(tool_call["arguments"]["normalized_task"], "look down.")
        self.assertEqual(tool_call["arguments"]["intentSteps"][0]["action"], "LookDown")

    def test_parse_task_normalizer_accepts_qwen_parameter_tool_call(self):
        output = (
            '<tool_call>\n<function=normalize_incoming_task>\n'
            '<parameter=normalized_task>\nOpen the fridge and look down.\n</parameter>\n'
            '<parameter=intentSteps>\n'
            '[{"order": 1, "action": "OpenObject", "objectType": "Fridge", "targetType": null}, '
            '{"order": 2, "action": "LookDown", "objectType": null, "targetType": null}]\n'
            '</parameter>\n'
            '<parameter=confidence>\nhigh\n</parameter>\n'
            '<parameter=reason>\nThe task explicitly requests these two actions.'
        )
        tool_call = task_execution_server.parse_task_normalizer_tool_call(output)
        self.assertEqual(tool_call["name"], "normalize_incoming_task")
        self.assertEqual(tool_call["arguments"]["normalized_task"], "Open the fridge and look down.")
        self.assertEqual(tool_call["arguments"]["confidence"], "high")
        self.assertEqual(tool_call["arguments"]["intentSteps"][1]["action"], "LookDown")

    def test_builds_closed_loop_relay_request(self):
        old_fetch = task_execution_server.fetch_receiver_state_object_types
        task_execution_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Bread", "CounterTop"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                backend = FakeNormalizerBackend(
                    {
                        "normalized_task": "put the Bread on the CounterTop.",
                        "intentSteps": [
                            {"order": 1, "action": "PutObject", "objectType": "Bread", "targetType": "CounterTop"},
                        ],
                        "confidence": "high",
                        "reason": "the task asks to put Bread on CounterTop",
                    }
                )
                service = TaskExecutionService(engine, backend, service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task(
                        {
                            "task_id": "task-1",
                            "task": "put the bread on the countertop",
                            "primary_robot_id": 0,
                            "dry_run": True,
                        }
                    )
        finally:
            task_execution_server.fetch_receiver_state_object_types = old_fetch

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["result"]["closed_loop_result"]["step_count"], 2)
        self.assertIn("--relay-mode", engine.argv)
        self.assertIn("--closed-loop-replan", engine.argv)
        self.assertIn("--save-raw-output", engine.argv)
        self.assertIn("http://127.0.0.1:19000/execute_actions", engine.argv)
        self.assertNotIn("--known-robot-ids", engine.argv)
        self.assertIn("--dry-run", engine.argv)

    def test_preserves_optional_upstream_context_in_task_intent(self):
        old_fetch = task_execution_server.fetch_receiver_state_object_types
        task_execution_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Fridge"], None)
        try:
            engine = FakeEngine()
            backend = FakeNormalizerBackend(
                {
                    "normalized_task": "open the Fridge.",
                    "intentSteps": [
                        {"order": 1, "action": "OpenObject", "objectType": "Fridge", "targetType": None},
                    ],
                    "confidence": "high",
                    "reason": "open requested receptacle",
                }
            )
            service = TaskExecutionService(engine, backend, service_config())
            service.execute_task(
                {
                    "task": "open the fridge",
                    "parent_task": "put bread in the fridge",
                    "subtask": {
                        "id": "T2",
                        "action": "open",
                        "grounding": {"object_tags": ["fridge"]},
                    },
                }
            )
        finally:
            task_execution_server.fetch_receiver_state_object_types = old_fetch

        intent_index = engine.argv.index("--task-intent-json") + 1
        intent_payload = json.loads(engine.argv[intent_index])
        self.assertEqual(intent_payload["parent_task"], "put bread in the fridge")
        self.assertEqual(intent_payload["upstream_subtask"]["id"], "T2")


    def test_normalizer_prompt_includes_recognized_action_sequence_constraint(self):
        messages = task_execution_server.task_normalizer_messages("close the fridge.", ["Fridge"])
        prompt = messages[0]["content"][0]["text"]
        self.assertIn("requested action names in this order", prompt)
        self.assertIn("CloseObject", prompt)
        self.assertIn("PickupObject tasks may add GotoObject for the same object", prompt)
        self.assertIn("PutObject/Place tasks are compound", prompt)

    def test_normalizes_planning_find_subtask_with_llm_tool(self):
        old_fetch = task_execution_server.fetch_receiver_state_object_types
        task_execution_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Fridge", "Cabinet"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                backend = FakeNormalizerBackend(
                    {
                        "normalized_task": "go to the Fridge.",
                        "intentSteps": [
                            {"order": 1, "action": "GotoObject", "objectType": "Fridge", "targetType": None},
                        ],
                        "confidence": "high",
                        "reason": "planning requested finding a fridge; execute as navigation",
                    }
                )
                service = TaskExecutionService(engine, backend, service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task(
                        {
                            "task_id": "find-fridge",
                            "task": "Find target for T1 Search/inspect the environment for unresolved task-relevant objects before executing T1: fridge find fridge",
                            "primary_robot_id": 0,
                        }
                    )
        finally:
            task_execution_server.fetch_receiver_state_object_types = old_fetch

        task_index = engine.argv.index("--task") + 1
        intent_index = engine.argv.index("--task-intent-json") + 1
        task_intent_payload = json.loads(engine.argv[intent_index])
        self.assertEqual(engine.argv[task_index], "go to the Fridge.")
        self.assertEqual(task_intent_payload["task_intent_source"], "qwen_normalizer_tool_call")
        self.assertEqual(task_intent_payload["task_intent"]["requestedAction"], "GotoObject")
        self.assertEqual(task_intent_payload["task_intent"]["intentSteps"][0]["action"], "GotoObject")
        self.assertTrue(response["task_normalization"]["used"])
        self.assertEqual(response["task_normalization"]["object_type"], "Fridge")
        self.assertEqual(response["status"], "success")

    def test_structured_find_subtask_bypasses_qwen_normalizer(self):
        old_fetch = task_execution_server.fetch_receiver_state_object_types
        task_execution_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (
            ["Lettuce", "Fridge"],
            None,
        )
        try:
            engine = FakeEngine()
            backend = FakeNormalizerBackend(
                {
                    "normalized_task": "look down.",
                    "intentSteps": [
                        {"order": 1, "action": "LookDown", "objectType": None, "targetType": None},
                    ],
                    "confidence": "high",
                    "reason": "unexpected fallback",
                }
            )
            service = TaskExecutionService(engine, backend, service_config())
            with contextlib.redirect_stdout(io.StringIO()):
                response = service.execute_task(
                    {
                        "task_id": "find-lettuce",
                        "task": "Find target for T2: lettuce",
                        "subtask": {
                            "id": "T2",
                            "action": "find",
                            "grounding": {"object_tags": ["lettuce"]},
                        },
                        "primary_robot_id": 0,
                    }
                )
        finally:
            task_execution_server.fetch_receiver_state_object_types = old_fetch

        task_index = engine.argv.index("--task") + 1
        intent_index = engine.argv.index("--task-intent-json") + 1
        task_intent_payload = json.loads(engine.argv[intent_index])
        self.assertEqual(backend.calls, [])
        self.assertEqual(engine.argv[task_index], "go to the Lettuce.")
        self.assertEqual(task_intent_payload["task_intent_source"], "upstream_structured_task")
        self.assertEqual(task_intent_payload["task_intent"]["requestedAction"], "GotoObject")
        self.assertEqual(response["task_normalization"]["source"], "upstream_structured_task")
        self.assertEqual(response["status"], "success")

    def test_pickup_task_allows_same_object_goto_helper(self):
        old_fetch = task_execution_server.fetch_receiver_state_object_types
        task_execution_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Bread"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                backend = FakeNormalizerBackend(
                    {
                        "normalized_task": "go to the Moon.",
                        "intentSteps": [
                            {"order": 1, "action": "GotoObject", "objectType": "Moon", "targetType": None},
                        ],
                        "confidence": "low",
                        "reason": "unexpected model fallback",
                    }
                )
                service = TaskExecutionService(engine, backend, service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task(
                        {
                            "task_id": "pickup-bread-with-goto",
                            "task": "Pick up the bread Grab the bread once it is located. pick bread",
                            "subtask": {
                                "id": "T2",
                                "name": "Pick up the bread",
                                "description": "Grab the bread once it is located.",
                                "action": "pick",
                                "grounding": {"object_tags": ["bread"]},
                            },
                        }
                    )
        finally:
            task_execution_server.fetch_receiver_state_object_types = old_fetch

        self.assertEqual(response["status"], "success")
        self.assertEqual(backend.calls, [])
        self.assertEqual(response["task_normalization"]["warnings"], [])
        self.assertEqual(response["task_normalization"]["source"], "upstream_structured_task")
        intent_index = engine.argv.index("--task-intent-json") + 1
        task_intent = json.loads(engine.argv[intent_index])["task_intent"]
        self.assertEqual(
            [step["action"] for step in task_intent["intentSteps"]],
            ["GotoObject", "PickupObject"],
        )

    def test_open_subtask_preserves_open_interaction(self):
        normalized = task_execution_server.normalize_structured_subtask(
            {
                "id": "T1",
                "name": "Open the drawer",
                "action": "open",
                "grounding": {"object_tags": ["drawer"]},
            },
            "Open all the drawers",
            ["Drawer"],
        )

        self.assertIsNotNone(normalized)
        self.assertEqual(
            [step["action"] for step in normalized["intentSteps"]],
            ["GotoObject", "OpenObject"],
        )
        self.assertEqual(normalized["normalized_task"], "go to the Drawer and open it.")

    def test_close_subtask_preserves_close_interaction(self):
        normalized = task_execution_server.normalize_structured_subtask(
            {
                "id": "T1",
                "action": "close",
                "grounding": {"object_tags": ["fridge"]},
            },
            "Close the fridge",
            ["Fridge"],
        )

        self.assertIsNotNone(normalized)
        self.assertEqual(
            [step["action"] for step in normalized["intentSteps"]],
            ["GotoObject", "CloseObject"],
        )

    def test_object_type_resolver_handles_format_plural_and_aliases(self):
        cases = [
            ("REMOTE_control", ["RemoteControl"], "RemoteControl", "exact"),
            ("tomatoes", ["Tomato"], "Tomato", "plural"),
            ("drawers", ["Drawer"], "Drawer", "plural"),
            ("computer", ["Laptop"], "Laptop", "alias"),
            ("laptop computer", ["Laptop"], "Laptop", "alias"),
            ("keys", ["KeyChain"], "KeyChain", "alias"),
            ("couch", ["Sofa"], "Sofa", "alias"),
            ("refrigerator", ["Fridge"], "Fridge", "alias"),
            ("counter", ["CounterTop"], "CounterTop", "alias"),
            ("kitchen-counter", ["CounterTop"], "CounterTop", "alias"),
            ("table", ["DiningTable", "Book"], "DiningTable", "alias"),
        ]
        for value, available, canonical, method in cases:
            with self.subTest(value=value):
                resolution = task_execution_server.resolve_object_type(value, available)
                self.assertEqual(resolution["canonical"], canonical)
                self.assertEqual(resolution["method"], method)

    def test_object_type_resolver_rejects_missing_and_ambiguous_aliases(self):
        missing = task_execution_server.resolve_object_type("computer", ["Book", "Sofa"])
        self.assertIsNone(missing["canonical"])
        self.assertEqual(missing["method"], "unresolved")

        ambiguous = task_execution_server.resolve_object_type("table", ["DiningTable", "CoffeeTable"])
        self.assertIsNone(ambiguous["canonical"])
        self.assertEqual(ambiguous["method"], "ambiguous")
        self.assertEqual(ambiguous["candidates"], ["DiningTable", "CoffeeTable"])

        exact_table = task_execution_server.resolve_object_type("table", ["Table", "DiningTable"])
        self.assertEqual(exact_table["canonical"], "Table")
        self.assertEqual(exact_table["method"], "exact")

    def test_object_type_resolver_keeps_distinct_canonical_types(self):
        for value, available in (("Knife", ["Knife", "ButterKnife"]), ("Cup", ["Cup", "Mug"])):
            with self.subTest(value=value):
                resolution = task_execution_server.resolve_object_type(value, available)
                self.assertEqual(resolution["canonical"], value)
                self.assertEqual(resolution["method"], "exact")

    def test_real_lettuce_other_payload_is_deterministic_pickup(self):
        normalized = task_execution_server.normalize_structured_subtask(
            {
                "id": "T2",
                "name": "Find and pick up the lettuce",
                "description": "Locate and retrieve the lettuce from its current location.",
                "action": "other",
                "grounding": {"object_tags": ["lettuce"], "status": "unresolved"},
            },
            "Find and pick up the lettuce Locate and retrieve the lettuce from its current location. other lettuce",
            ["Lettuce", "Tomato", "Fridge"],
            primary_task_text="Find and pick up the lettuce",
            action_coverage_text="Find and pick up the lettuce "
            "Locate and retrieve the lettuce from its current location.",
        )

        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["source"], "upstream_structured_task")
        self.assertEqual(normalized["confidence"], "high")
        self.assertEqual(
            [step["action"] for step in normalized["intentSteps"]],
            ["GotoObject", "PickupObject"],
        )
        self.assertIn(
            {
                "input": "lettuce",
                "canonical": "Lettuce",
                "method": "exact",
                "candidates": ["Lettuce"],
            },
            normalized["type_resolutions"],
        )

    def test_action_other_place_uses_compound_deterministic_intent(self):
        normalized = task_execution_server.normalize_structured_subtask(
            {
                "name": "Put the tomato in the fridge",
                "action": "other",
                "grounding": {"object_tags": ["tomato"]},
            },
            "Put the tomato in the fridge",
            ["Tomato", "Fridge"],
        )

        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["source"], "upstream_structured_task")
        self.assertEqual(
            [step["action"] for step in normalized["intentSteps"]],
            ["GotoObject", "PickupObject", "GotoObject", "OpenObject", "PutObject", "CloseObject"],
        )

    def test_slice_subtask_uses_supported_deterministic_intent(self):
        object_id = "Tomato|+00.10|+00.20|+00.30"
        normalized = task_execution_server.normalize_structured_subtask(
            {
                "name": "Slice the tomato",
                "action": "slice",
                "grounding": {
                    "source_object_tags": ["tomato"],
                    "source_object_ids": [object_id],
                },
            },
            "Slice the tomato",
            ["Tomato", "Knife"],
        )

        self.assertIsNotNone(normalized)
        steps = normalized["intentSteps"]
        self.assertEqual(
            [step["action"] for step in steps],
            ["GotoObject", "SliceObject"],
        )
        self.assertEqual([step["objectId"] for step in steps], [object_id, object_id])
        self.assertEqual(normalized["object_type"], "Tomato")

    def test_toggle_and_clean_subtasks_preserve_intent_priority_and_object_id(self):
        self.assertEqual(
            task_execution_server._extract_requested_action("Turn off the right stove knob"),
            "ToggleObjectOff",
        )
        self.assertEqual(
            task_execution_server._extract_requested_action("Turn on the left desk lamp"),
            "ToggleObjectOn",
        )
        cases = [
            ("toggle_on", "Turn on the stove knob", "StoveKnob", "ToggleObjectOn"),
            ("toggle_off", "Turn off the faucet", "Faucet", "ToggleObjectOff"),
            ("other", "Wash the bowl", "Bowl", "CleanObject"),
        ]
        for action, task, object_type, expected_action in cases:
            with self.subTest(action=action):
                object_id = f"{object_type}|+01.00|+02.00|+03.00"
                normalized = task_execution_server.normalize_structured_subtask(
                    {
                        "name": task,
                        "action": action,
                        "grounding": {
                            "source_object_tags": [object_type],
                            "source_object_ids": [object_id],
                        },
                    },
                    task,
                    [object_type],
                )

                self.assertIsNotNone(normalized)
                steps = normalized["intentSteps"]
                self.assertEqual(
                    [step["action"] for step in steps],
                    ["GotoObject", expected_action],
                )
                self.assertEqual([step["objectId"] for step in steps], [object_id, object_id])
                self.assertIn(expected_action, task_execution_server.SUPPORTED_NORMALIZED_ACTIONS)

    def test_clean_subtask_with_sink_context_skips_semantic_destination_lookup(self):
        object_id = "Bowl|+00.27|+01.10|-00.75"
        engine = FakeEngine()
        backend = FakeNormalizerBackend({})
        with patch.object(
            task_execution_server,
            "fetch_receiver_state_object_types",
            return_value=(["Bowl", "Sink"], None),
        ), patch.object(
            task_execution_server,
            "fetch_receiver_state",
            side_effect=AssertionError("clean should not resolve semantic destination"),
        ):
            service = TaskExecutionService(engine, backend, service_config())
            with contextlib.redirect_stdout(io.StringIO()):
                response = service.execute_task(
                    {
                        "task_id": "clean-bowl-sink-context",
                        "task": "Clean bowl Wash the bowl in the sink. clean Bowl sink Sink",
                        "subtask": {
                            "id": "T1",
                            "name": "Clean bowl",
                            "description": "Wash the bowl in the sink.",
                            "action": "clean",
                            "grounding": {
                                "object_tags": ["Bowl", "sink", "Sink"],
                                "source_object_tags": ["Bowl"],
                                "source_object_ids": [object_id],
                                "destination_object_tags": ["sink", "Sink"],
                                "destination_selector": {"quantifier": "one", "object_types": ["Sink"]},
                                "destination_object_ids": [],
                            },
                        },
                    }
                )

        self.assertEqual(response["status"], "success")
        self.assertEqual(backend.calls, [])
        task_intent_json = engine.argv[engine.argv.index("--task-intent-json") + 1]
        task_intent = json.loads(task_intent_json)["task_intent"]
        steps = task_intent["intentSteps"]
        self.assertEqual([step["action"] for step in steps], ["GotoObject", "CleanObject"])
        self.assertEqual([step["objectId"] for step in steps], [object_id, object_id])
        self.assertNotIn("failure_code", response)

    def test_computer_alias_bypasses_low_confidence_backend_at_service_boundary(self):
        engine = FakeEngine()
        backend = FakeNormalizerBackend(
            {
                "normalized_task": "find the computer",
                "intentSteps": [
                    {"order": 1, "action": "GotoObject", "objectType": "computer", "targetType": None},
                    {"order": 2, "action": "PickupObject", "objectType": "computer", "targetType": None},
                ],
                "confidence": "low",
                "reason": "unexpected model fallback",
            }
        )
        with patch.object(
            task_execution_server,
            "fetch_receiver_state_object_types",
            return_value=(["Laptop", "Book", "Sofa"], None),
        ):
            service = TaskExecutionService(engine, backend, service_config())
            with contextlib.redirect_stdout(io.StringIO()):
                response = service.execute_task(
                    {
                        "task_id": "find-pick-computer",
                        "task": "Find and pick up the computer other computer",
                        "subtask": {
                            "name": "Find and pick up the computer",
                            "action": "other",
                            "grounding": {"object_tags": ["computer"]},
                        },
                    }
                )

        self.assertEqual(response["status"], "success")
        self.assertEqual(backend.calls, [])
        self.assertEqual(response["task_normalization"]["source"], "upstream_structured_task")
        self.assertEqual(response["task_normalization"]["object_type"], "Laptop")
    def test_subtask_name_find_and_pickup_becomes_goto_then_pickup(self):
        old_fetch = task_execution_server.fetch_receiver_state_object_types
        task_execution_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Tomato", "Fridge"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                backend = FakeNormalizerBackend(
                    {
                        "normalized_task": "go to the Tomato and pick it up.",
                        "intentSteps": [
                            {"order": 1, "action": "GotoObject", "objectType": "Tomato", "targetType": None},
                            {"order": 2, "action": "PickupObject", "objectType": "Tomato", "targetType": None},
                        ],
                        "confidence": "low",
                        "reason": "unexpected model fallback",
                    }
                )
                service = TaskExecutionService(engine, backend, service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task(
                        {
                            "task_id": "find-pick-tomato",
                            "task": "Find and pick up tomato Locate and pick up the tomato. other tomato",
                            "subtask": {
                                "id": "T3",
                                "name": "Find and pick up tomato",
                                "description": "Locate and pick up the tomato.",
                                "action": "other",
                                "grounding": {
                                    "object_tags": ["tomato"],
                                    "status": "unresolved",
                                    "recovery": "search_visible_scene",
                                },
                            },
                            "primary_robot_id": 1,
                        }
                    )
        finally:
            task_execution_server.fetch_receiver_state_object_types = old_fetch

        task_index = engine.argv.index("--task") + 1
        intent_index = engine.argv.index("--task-intent-json") + 1
        task_intent_payload = json.loads(engine.argv[intent_index])
        task_intent = task_intent_payload["task_intent"]
        self.assertEqual(engine.argv[task_index], "go to the Tomato and pick it up.")
        self.assertEqual(
            task_intent["intentSteps"],
            [
                {"order": 1, "action": "GotoObject", "objectType": "Tomato", "targetType": None},
                {"order": 2, "action": "PickupObject", "objectType": "Tomato", "targetType": None},
            ],
        )
        self.assertEqual(response["task_normalization"]["primary_task_text"], "Find and pick up tomato")
        self.assertEqual(response["task_normalization"]["action_coverage_text"], "Find and pick up tomato Locate and pick up the tomato.")
        self.assertTrue(response["task_normalization"]["used_subtask_name"])
        self.assertEqual(response["task_normalization"]["subtask_context"]["action"], "other")
        self.assertEqual(response["task_normalization"]["source"], "upstream_structured_task")
        self.assertEqual(backend.calls, [])
        self.assertEqual(response["status"], "success")

    def test_structured_find_uses_pure_goto_even_when_name_mentions_pickup(self):
        old_fetch = task_execution_server.fetch_receiver_state_object_types
        task_execution_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Tomato"], None)
        try:
            engine = FakeEngine()
            backend = FakeNormalizerBackend(
                {
                    "normalized_task": "pick up the Tomato.",
                    "intentSteps": [
                        {"order": 1, "action": "PickupObject", "objectType": "Tomato", "targetType": None},
                    ],
                    "confidence": "high",
                    "reason": "badly collapsed find and pickup",
                }
            )
            service = TaskExecutionService(engine, backend, service_config())
            with contextlib.redirect_stdout(io.StringIO()):
                response = service.execute_task(
                    {
                        "task_id": "bad-find-pick",
                        "task": "Find and pick up tomato Locate and pick up the tomato. other tomato",
                        "subtask": {
                            "id": "T3",
                            "name": "Find and pick up tomato",
                            "action": "find",
                            "grounding": {"object_tags": ["tomato"]},
                        },
                        "primary_robot_id": 1,
                    }
                )
        finally:
            task_execution_server.fetch_receiver_state_object_types = old_fetch

        self.assertEqual(response["status"], "success")
        self.assertEqual(backend.calls, [])
        self.assertEqual(response["task_normalization"]["primary_task_text"], "Find and pick up tomato")
        self.assertTrue(response["task_normalization"]["used_subtask_name"])
        self.assertEqual(
            response["task_normalization"]["intentSteps"],
            [{"order": 1, "action": "GotoObject", "objectType": "Tomato", "targetType": None}],
        )

    def test_subtask_description_pickup_extends_search_name_coverage(self):
        old_fetch = task_execution_server.fetch_receiver_state_object_types
        task_execution_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Tomato"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                backend = FakeNormalizerBackend(
                    {
                        "normalized_task": "go to the Tomato and pick it up.",
                        "intentSteps": [
                            {"order": 1, "action": "GotoObject", "objectType": "Tomato", "targetType": None},
                            {"order": 2, "action": "PickupObject", "objectType": "Tomato", "targetType": None},
                        ],
                        "confidence": "high",
                        "reason": "description asks to locate and pick up tomato",
                    }
                )
                service = TaskExecutionService(engine, backend, service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task(
                        {
                            "task_id": "search-pick-tomato",
                            "task": "Search for tomato Locate and pick up the tomato. other tomato",
                            "subtask": {
                                "id": "T6",
                                "name": "Search for tomato",
                                "description": "Locate and pick up the tomato.",
                                "action": "other",
                                "grounding": {"object_tags": ["tomato"]},
                            },
                            "primary_robot_id": 1,
                        }
                    )
        finally:
            task_execution_server.fetch_receiver_state_object_types = old_fetch

        intent_index = engine.argv.index("--task-intent-json") + 1
        task_intent = json.loads(engine.argv[intent_index])["task_intent"]
        self.assertEqual(response["task_normalization"]["primary_task_text"], "Search for tomato")
        self.assertEqual(response["task_normalization"]["action_coverage_text"], "Search for tomato Locate and pick up the tomato.")
        self.assertEqual(
            task_intent["intentSteps"],
            [
                {"order": 1, "action": "GotoObject", "objectType": "Tomato", "targetType": None},
                {"order": 2, "action": "PickupObject", "objectType": "Tomato", "targetType": None},
            ],
        )
        self.assertEqual(response["task_normalization"]["source"], "upstream_structured_task")
        self.assertEqual(backend.calls, [])
        self.assertEqual(response["status"], "success")

    def test_subtask_description_pickup_bypasses_search_only_backend_output(self):
        old_fetch = task_execution_server.fetch_receiver_state_object_types
        task_execution_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Tomato"], None)
        try:
            engine = FakeEngine()
            backend = FakeNormalizerBackend(
                {
                    "normalized_task": "go to the Tomato.",
                    "intentSteps": [
                        {"order": 1, "action": "GotoObject", "objectType": "Tomato", "targetType": None},
                    ],
                    "confidence": "low",
                    "reason": "unexpected model fallback",
                }
            )
            service = TaskExecutionService(engine, backend, service_config())
            with contextlib.redirect_stdout(io.StringIO()):
                response = service.execute_task(
                    {
                        "task_id": "bad-search-pick",
                        "task": "Search for tomato Locate and pick up the tomato. other tomato",
                        "subtask": {
                            "id": "T6",
                            "name": "Search for tomato",
                            "description": "Locate and pick up the tomato.",
                            "action": "other",
                            "grounding": {"object_tags": ["tomato"]},
                        },
                        "primary_robot_id": 1,
                    }
                )
        finally:
            task_execution_server.fetch_receiver_state_object_types = old_fetch

        self.assertIsNotNone(engine.argv)
        self.assertEqual(response["status"], "success")
        self.assertEqual(backend.calls, [])
        self.assertEqual(response["task_normalization"]["primary_task_text"], "Search for tomato")
        self.assertEqual(response["task_normalization"]["action_coverage_text"], "Search for tomato Locate and pick up the tomato.")
        self.assertEqual(
            [step["action"] for step in response["task_normalization"]["intentSteps"]],
            ["GotoObject", "PickupObject"],
        )

    def test_pure_search_subtask_with_action_other_maps_to_goto(self):
        old_fetch = task_execution_server.fetch_receiver_state_object_types
        task_execution_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Fridge"], None)
        try:
            engine = FakeEngine()
            backend = FakeNormalizerBackend(
                {
                    "normalized_task": "look down.",
                    "intentSteps": [
                        {"order": 1, "action": "LookDown", "objectType": None, "targetType": None},
                    ],
                    "confidence": "high",
                    "reason": "unexpected fallback",
                }
            )
            service = TaskExecutionService(engine, backend, service_config())
            with contextlib.redirect_stdout(io.StringIO()):
                response = service.execute_task(
                    {
                        "task_id": "find-target-fridge",
                        "task": "Find target for T2 Search/inspect the environment for unresolved task-relevant objects before executing T2: fridge find fridge",
                        "subtask": {
                            "id": "F_T2_01",
                            "name": "Find target for T2",
                            "description": "Search/inspect the environment for unresolved task-relevant objects before executing T2: fridge",
                            "action": "other",
                            "grounding": {"object_tags": ["fridge"]},
                        },
                        "primary_robot_id": 0,
                    }
                )
        finally:
            task_execution_server.fetch_receiver_state_object_types = old_fetch

        task_index = engine.argv.index("--task") + 1
        intent_index = engine.argv.index("--task-intent-json") + 1
        task_intent_payload = json.loads(engine.argv[intent_index])
        self.assertEqual(backend.calls, [])
        self.assertEqual(engine.argv[task_index], "go to the Fridge.")
        self.assertEqual(task_intent_payload["task_intent_source"], "upstream_structured_task")
        self.assertEqual(response["task_normalization"]["source"], "upstream_structured_task")
        self.assertEqual(response["task_normalization"]["primary_task_text"], "Find target for T2")
        self.assertEqual(response["task_normalization"]["action_coverage_text"], "Find target for T2 Search/inspect the environment for unresolved task-relevant objects before executing T2: fridge")
        self.assertEqual(response["status"], "success")

    def test_normalizer_preserves_multi_step_task_intent(self):
        old_fetch = task_execution_server.fetch_receiver_state_object_types
        task_execution_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Tomato", "CounterTop"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                backend = FakeNormalizerBackend(
                    {
                        "normalized_task": "pick up the Tomato and put it on the CounterTop.",
                        "requestedAction": "PickupObject",
                        "requestedObjectType": "Tomato",
                        "requestedTargetType": "CounterTop",
                        "intentSteps": [
                            {"order": 1, "action": "PickupObject", "objectType": "Tomato", "targetType": None},
                            {"order": 2, "action": "PutObject", "objectType": "Tomato", "targetType": "CounterTop"},
                        ],
                        "confidence": "high",
                        "reason": "the task asks for pickup followed by placement",
                    }
                )
                service = TaskExecutionService(engine, backend, service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task(
                        {"task_id": "tomato-put", "task": "pick up the tomato and put it on the counter", "primary_robot_id": 0}
                    )
        finally:
            task_execution_server.fetch_receiver_state_object_types = old_fetch

        intent_index = engine.argv.index("--task-intent-json") + 1
        task_intent = json.loads(engine.argv[intent_index])["task_intent"]
        self.assertEqual(task_intent["requestedAction"], "PickupObject")
        self.assertEqual(task_intent["requestedTargetType"], "CounterTop")
        self.assertEqual(
            task_intent["intentSteps"],
            [
                {"order": 1, "action": "PickupObject", "objectType": "Tomato", "targetType": None},
                {"order": 2, "action": "PutObject", "objectType": "Tomato", "targetType": "CounterTop"},
            ],
        )
        self.assertEqual(response["task_normalization"]["intentSteps"][1]["action"], "PutObject")

    def test_normalizer_accepts_put_pronoun_with_distinct_target(self):
        old_fetch = task_execution_server.fetch_receiver_state_object_types
        task_execution_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["PaperTowelRoll", "Mug"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                backend = FakeNormalizerBackend(
                    {
                        "normalized_task": "pick up the PaperTowelRoll and put it in the Mug.",
                        "intentSteps": [
                            {"order": 1, "action": "PickupObject", "objectType": "PaperTowelRoll", "targetType": None},
                            {"order": 2, "action": "PutObject", "objectType": "PaperTowelRoll", "targetType": "Mug"},
                        ],
                        "confidence": "high",
                        "reason": "the task asks to pick up the paper towel roll and place it in the mug",
                    }
                )
                service = TaskExecutionService(engine, backend, service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task(
                        {
                            "task_id": "paper-mug",
                            "task": "pick up the paper towel roll and put it in the mug",
                            "primary_robot_id": 0,
                        }
                    )
        finally:
            task_execution_server.fetch_receiver_state_object_types = old_fetch

        self.assertEqual(response["status"], "success")
        self.assertTrue(response["task_normalization"]["used"])
        self.assertEqual(response["task_normalization"]["warnings"], [])
        intent_index = engine.argv.index("--task-intent-json") + 1
        task_intent = json.loads(engine.argv[intent_index])["task_intent"]
        self.assertEqual(task_intent["intentSteps"][1]["objectType"], "PaperTowelRoll")
        self.assertEqual(task_intent["intentSteps"][1]["targetType"], "Mug")

    def test_normalizer_uses_tool_retry_when_tool_arguments_are_empty(self):
        old_fetch = task_execution_server.fetch_receiver_state_object_types
        task_execution_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Fridge"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                service = TaskExecutionService(
                    engine,
                    SequentialToolBackend(
                        {},
                        {
                            "normalized_task": "go to the Fridge.",
                            "intentSteps": [
                                {"order": 1, "action": "GotoObject", "objectType": "Fridge", "targetType": None},
                            ],
                            "confidence": "high",
                            "reason": "retry normalized search to navigation",
                        },
                    ),
                    service_config(temp_dir),
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task({"task_id": "retry", "task": "find fridge"})
        finally:
            task_execution_server.fetch_receiver_state_object_types = old_fetch

        task_index = engine.argv.index("--task") + 1
        self.assertEqual(engine.argv[task_index], "go to the Fridge.")
        self.assertTrue(response["task_normalization"]["used"])
        self.assertEqual(response["task_normalization"]["source"], "qwen_tool_call_retry")
        self.assertIn("raw_tool_output_preview", response["task_normalization"])
        self.assertIn("raw_retry_tool_output_preview", response["task_normalization"])
        self.assertNotIn("raw_json_fallback_preview", response["task_normalization"])
        intent_index = engine.argv.index("--task-intent-json") + 1
        task_intent = json.loads(engine.argv[intent_index])["task_intent"]
        self.assertEqual(task_intent["requestedAction"], "GotoObject")
        self.assertEqual(task_intent["requestedObjectType"], "Fridge")


    def test_normalizer_failure_reason_includes_low_confidence_reason(self):
        response = task_execution_server.task_normalization_failure_response(
            "task-low-confidence",
            False,
            {
                "warnings": [],
                "confidence": "low",
                "reason": "The object 'Moon' is not present in the current AI2-THOR environment object types.",
            },
        )

        self.assertEqual(response["failure_code"], "task_normalization_failed")
        self.assertIn("confidence", response["reason"])
        self.assertIn("Moon", response["reason"])
        self.assertEqual(response["result"]["closed_loop_result"]["reason"], response["reason"])

    def test_normalizer_failure_does_not_call_runtime_or_legacy_parser(self):
        old_fetch = task_execution_server.fetch_receiver_state_object_types
        task_execution_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Fridge"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                service = TaskExecutionService(engine, SequentialToolBackend({}, {}), service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task({"task_id": "normalizer-fail", "task": "find fridge", "dry_run": True})
        finally:
            task_execution_server.fetch_receiver_state_object_types = old_fetch

        self.assertIsNone(engine.argv)
        self.assertEqual(response["status"], "needs_upstream_planning")
        self.assertEqual(response["failure_code"], "task_normalization_failed")
        self.assertEqual(response["dry_run"], True)
        self.assertEqual(response["result"]["closed_loop_result"]["failure_code"], "task_normalization_failed")
        self.assertIn("normalizer omitted usable intentSteps", response["reason"])
        self.assertNotEqual(response["task_normalization"].get("source"), "qwen_json_fallback")


    def test_normalizer_rejects_exploratory_actions_for_find_target_and_retries_goto(self):
        old_fetch = task_execution_server.fetch_receiver_state_object_types
        task_execution_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Fridge"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                service = TaskExecutionService(
                    engine,
                    SequentialToolBackend(
                        {
                            "normalized_task": "search for the Fridge.",
                            "intentSteps": [
                                {"order": 1, "action": "LookDown", "objectType": None, "targetType": None},
                                {"order": 2, "action": "MoveAhead", "objectType": None, "targetType": None},
                            ],
                            "confidence": "high",
                            "reason": "bad exploratory search plan",
                        },
                        {
                            "normalized_task": "go to the Fridge.",
                            "intentSteps": [
                                {"order": 1, "action": "GotoObject", "objectType": "Fridge", "targetType": None},
                            ],
                            "confidence": "high",
                            "reason": "find target is executable as navigation to Fridge",
                        },
                    ),
                    service_config(temp_dir),
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task({"task_id": "find-fridge", "task": "find fridge"})
        finally:
            task_execution_server.fetch_receiver_state_object_types = old_fetch

        intent_index = engine.argv.index("--task-intent-json") + 1
        task_intent = json.loads(engine.argv[intent_index])["task_intent"]
        task_index = engine.argv.index("--task") + 1
        self.assertEqual(engine.argv[task_index], "go to the Fridge.")
        self.assertEqual(response["task_normalization"]["source"], "qwen_tool_call_retry")
        self.assertEqual(task_intent["requestedAction"], "GotoObject")
        self.assertEqual(task_intent["requestedObjectType"], "Fridge")
        self.assertEqual(task_intent["intentSteps"], [
            {"order": 1, "action": "GotoObject", "objectType": "Fridge", "targetType": None}
        ])

    def test_normalizer_tool_retry_extracts_multi_step_open_close(self):
        old_fetch = task_execution_server.fetch_receiver_state_object_types
        task_execution_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Fridge"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                service = TaskExecutionService(
                    engine,
                    SequentialToolBackend(
                        {},
                        {
                            "normalized_task": "open the Fridge and close the Fridge.",
                            "intentSteps": [
                                {"order": 1, "action": "OpenObject", "objectType": "Fridge", "targetType": None},
                                {"order": 2, "action": "CloseObject", "objectType": "Fridge", "targetType": None},
                            ],
                            "confidence": "high",
                            "reason": "the task requests opening then closing the same object",
                        },
                    ),
                    service_config(temp_dir),
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task({"task_id": "open-close", "task": "open the fridge and close the fridge"})
        finally:
            task_execution_server.fetch_receiver_state_object_types = old_fetch

        intent_index = engine.argv.index("--task-intent-json") + 1
        task_intent = json.loads(engine.argv[intent_index])["task_intent"]
        self.assertEqual(response["task_normalization"]["source"], "qwen_tool_call_retry")
        self.assertEqual(task_intent["requestedAction"], "OpenObject")
        self.assertEqual(task_intent["intentSteps"][1]["action"], "CloseObject")

    def test_normalizer_tool_retry_extracts_multi_step_pickup_put(self):
        old_fetch = task_execution_server.fetch_receiver_state_object_types
        task_execution_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Tomato", "CounterTop"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                service = TaskExecutionService(
                    engine,
                    SequentialToolBackend(
                        {},
                        {
                            "normalized_task": "pick up the Tomato and put it on the CounterTop.",
                            "intentSteps": [
                                {"order": 1, "action": "PickupObject", "objectType": "Tomato", "targetType": None},
                                {"order": 2, "action": "PutObject", "objectType": "Tomato", "targetType": "CounterTop"},
                            ],
                            "confidence": "high",
                            "reason": "the task requests pickup followed by placement",
                        },
                    ),
                    service_config(temp_dir),
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task({"task_id": "pickup-put", "task": "pick up the tomato and put it on the counter"})
        finally:
            task_execution_server.fetch_receiver_state_object_types = old_fetch

        intent_index = engine.argv.index("--task-intent-json") + 1
        task_intent = json.loads(engine.argv[intent_index])["task_intent"]
        self.assertEqual(response["task_normalization"]["source"], "qwen_tool_call_retry")
        self.assertEqual(task_intent["requestedTargetType"], "CounterTop")
        self.assertEqual(task_intent["intentSteps"][0]["action"], "PickupObject")
        self.assertEqual(task_intent["intentSteps"][1]["action"], "PutObject")

    def test_normalizer_canonicalizes_action_only_task_text(self):
        old_fetch = task_execution_server.fetch_receiver_state_object_types
        task_execution_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Fridge"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                backend = FakeNormalizerBackend(
                    {
                        "normalized_task": "GotoObject",
                        "action": "GotoObject",
                        "object_type": "Fridge",
                        "target_type": "Fridge",
                        "confidence": "high",
                        "reason": "planning find request can be executed as navigation",
                    }
                )
                service = TaskExecutionService(engine, backend, service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task({"task_id": "action-only", "task": "find fridge"})
        finally:
            task_execution_server.fetch_receiver_state_object_types = old_fetch

        task_index = engine.argv.index("--task") + 1
        self.assertEqual(engine.argv[task_index], "go to the Fridge.")
        self.assertTrue(response["task_normalization"]["used"])
        self.assertEqual(response["task_normalization"]["normalized_task"], "go to the Fridge.")

    def test_normalizer_accepts_camel_case_and_numeric_confidence(self):
        old_fetch = task_execution_server.fetch_receiver_state_object_types
        task_execution_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Fridge"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                backend = FakeNormalizerBackend(
                    {
                        "normalizedTask": "go to the Fridge.",
                        "action": "GotoObject",
                        "objectType": "Fridge",
                        "targetType": None,
                        "confidence": 0.95,
                        "reason": "planning find request can be executed as navigation",
                    }
                )
                service = TaskExecutionService(engine, backend, service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task({"task_id": "numeric-confidence", "task": "find fridge"})
        finally:
            task_execution_server.fetch_receiver_state_object_types = old_fetch

        task_index = engine.argv.index("--task") + 1
        self.assertEqual(engine.argv[task_index], "go to the Fridge.")
        self.assertTrue(response["task_normalization"]["used"])
        self.assertEqual(response["task_normalization"]["confidence"], "high")
        self.assertEqual(response["task_normalization"]["object_type"], "Fridge")

    def test_normalizer_preserves_no_arg_look_down_step(self):
        old_fetch = task_execution_server.fetch_receiver_state_object_types
        task_execution_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Fridge"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                backend = FakeNormalizerBackend(
                    {
                        "normalized_task": "open the Fridge and look down.",
                        "intentSteps": [
                            {"order": 1, "action": "OpenObject", "objectType": "Fridge", "targetType": None},
                            {"order": 2, "action": "LookDown", "objectType": None, "targetType": None},
                        ],
                        "confidence": "high",
                        "reason": "the task requests opening the fridge and looking down",
                    }
                )
                service = TaskExecutionService(engine, backend, service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task({"task_id": "open-look-down", "task": "open the fridge and look down"})
        finally:
            task_execution_server.fetch_receiver_state_object_types = old_fetch

        intent_index = engine.argv.index("--task-intent-json") + 1
        task_intent = json.loads(engine.argv[intent_index])["task_intent"]
        self.assertTrue(response["task_normalization"]["used"])
        self.assertEqual(task_intent["intentSteps"][0]["action"], "OpenObject")
        self.assertEqual(task_intent["intentSteps"][1]["action"], "LookDown")
        self.assertIsNone(task_intent["intentSteps"][1]["objectType"])

    def test_normalizer_preserves_single_look_down_task(self):
        old_fetch = task_execution_server.fetch_receiver_state_object_types
        task_execution_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Fridge"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                backend = FakeNormalizerBackend(
                    {
                        "normalized_task": "look down.",
                        "intentSteps": [
                            {"order": 1, "action": "LookDown", "objectType": None, "targetType": None},
                        ],
                        "confidence": "high",
                        "reason": "the task requests looking down",
                    }
                )
                service = TaskExecutionService(engine, backend, service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task({"task_id": "look-down", "task": "look down"})
        finally:
            task_execution_server.fetch_receiver_state_object_types = old_fetch

        intent_index = engine.argv.index("--task-intent-json") + 1
        task_intent = json.loads(engine.argv[intent_index])["task_intent"]
        self.assertTrue(response["task_normalization"]["used"])
        self.assertEqual(task_intent["requestedAction"], "LookDown")
        self.assertEqual(task_intent["intentSteps"][0]["action"], "LookDown")

    def test_place_in_fridge_allows_put_helper_actions(self):
        old_fetch = task_execution_server.fetch_receiver_state_object_types
        task_execution_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Bread", "Fridge"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                backend = FakeNormalizerBackend(
                    {
                        "normalized_task": "put the Bread in the Fridge.",
                        "intentSteps": [
                            {"order": 1, "action": "GotoObject", "objectType": "Bread", "targetType": None},
                            {"order": 2, "action": "PickupObject", "objectType": "Bread", "targetType": None},
                            {"order": 3, "action": "GotoObject", "objectType": "Fridge", "targetType": None},
                            {"order": 4, "action": "OpenObject", "objectType": "Fridge", "targetType": None},
                            {"order": 5, "action": "PutObject", "objectType": "Bread", "targetType": "Fridge"},
                            {"order": 6, "action": "CloseObject", "objectType": "Fridge", "targetType": None},
                        ],
                        "confidence": "high",
                        "reason": "placing bread in fridge requires picking it up and opening the fridge",
                    }
                )
                service = TaskExecutionService(engine, backend, service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task(
                        {
                            "task_id": "place-bread-fridge",
                            "task": "Place the bread in the fridge Put the bread in the fridge. place bread in",
                            "subtask": {
                                "id": "T4",
                                "name": "Place the bread in the fridge",
                                "description": "Put the bread in the fridge.",
                                "action": "place",
                                "grounding": {"object_tags": ["bread"], "relation_texts": ["in"]},
                            },
                            "primary_robot_id": 1,
                        }
                    )
        finally:
            task_execution_server.fetch_receiver_state_object_types = old_fetch

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["task_normalization"]["warnings"], [])
        intent_index = engine.argv.index("--task-intent-json") + 1
        task_intent = json.loads(engine.argv[intent_index])["task_intent"]
        self.assertEqual(
            [step["action"] for step in task_intent["intentSteps"]],
            ["GotoObject", "PickupObject", "GotoObject", "OpenObject", "PutObject", "CloseObject"],
        )
        put_step = next(step for step in task_intent["intentSteps"] if step["action"] == "PutObject")
        self.assertEqual(put_step["objectType"], "Bread")
        self.assertEqual(put_step["targetType"], "Fridge")

    def test_place_in_fridge_repairs_destination_helpers_before_validation(self):
        old_fetch = task_execution_server.fetch_receiver_state_object_types
        task_execution_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Bread", "Fridge"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                backend = FakeNormalizerBackend(
                    {
                        "normalized_task": "put the Bread in the Fridge.",
                        "intentSteps": [
                            {"order": 1, "action": "GotoObject", "objectType": "Fridge", "targetType": None},
                            {"order": 2, "action": "OpenObject", "objectType": "Fridge", "targetType": None},
                            {"order": 3, "action": "PickupObject", "objectType": "Bread", "targetType": None},
                            {"order": 4, "action": "PutObject", "objectType": "Bread", "targetType": "Fridge"},
                        ],
                        "confidence": "high",
                        "reason": "badly prepared the receptacle before acquiring bread",
                    }
                )
                service = TaskExecutionService(engine, backend, service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task(
                        {
                            "task_id": "place-bread-fridge-bad-order",
                            "task": "Place the bread in the fridge Put the bread in the fridge. place bread in",
                            "subtask": {
                                "id": "T4",
                                "name": "Place the bread in the fridge",
                                "description": "Put the bread in the fridge.",
                                "action": "place",
                                "grounding": {"object_tags": ["bread"], "relation_texts": ["in"]},
                            },
                            "primary_robot_id": 1,
                        }
                    )
        finally:
            task_execution_server.fetch_receiver_state_object_types = old_fetch

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["task_normalization"]["warnings"], [])
        self.assertEqual(backend.calls, [])
        self.assertEqual(response["task_normalization"]["source"], "upstream_structured_task")
        intent_index = engine.argv.index("--task-intent-json") + 1
        task_intent = json.loads(engine.argv[intent_index])["task_intent"]
        self.assertEqual(
            task_intent["intentSteps"],
            [
                {"order": 1, "action": "GotoObject", "objectType": "Bread", "targetType": None},
                {"order": 2, "action": "PickupObject", "objectType": "Bread", "targetType": None},
                {"order": 3, "action": "GotoObject", "objectType": "Fridge", "targetType": None},
                {"order": 4, "action": "OpenObject", "objectType": "Fridge", "targetType": None},
                {"order": 5, "action": "PutObject", "objectType": "Bread", "targetType": "Fridge"},
                {"order": 6, "action": "CloseObject", "objectType": "Fridge", "targetType": None},
            ],
        )
        self.assertEqual(task_intent["requestedAction"], "GotoObject")
        self.assertEqual(task_intent["requestedObjectType"], "Bread")
        self.assertEqual(task_intent["requestedTargetType"], "Fridge")

    def test_place_in_fridge_repairs_goto_with_erroneous_target_type(self):
        old_fetch = task_execution_server.fetch_receiver_state_object_types
        task_execution_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Tomato", "Fridge"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                backend = FakeNormalizerBackend(
                    {
                        "normalized_task": "Place the Tomato in the Fridge.",
                        "intentSteps": [
                            {"order": 1, "action": "GotoObject", "objectType": "Tomato", "targetType": "Fridge"},
                            {"order": 2, "action": "PickupObject", "objectType": "Tomato", "targetType": None},
                            {"order": 3, "action": "GotoObject", "objectType": "Fridge", "targetType": None},
                            {"order": 4, "action": "PutObject", "objectType": "Tomato", "targetType": "Fridge"},
                        ],
                        "confidence": "high",
                        "reason": "badly put the placement target on a GotoObject helper",
                    }
                )
                service = TaskExecutionService(engine, backend, service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task(
                        {
                            "task_id": "place-tomato-fridge-goto-target",
                            "task": "Place Tomato in Fridge Place the tomato in the fridge. place tomato in",
                            "subtask": {
                                "id": "T6",
                                "name": "Place Tomato in Fridge",
                                "description": "Place the tomato in the fridge.",
                                "action": "place",
                            },
                        }
                    )
        finally:
            task_execution_server.fetch_receiver_state_object_types = old_fetch

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["task_normalization"]["warnings"], [])
        intent_index = engine.argv.index("--task-intent-json") + 1
        task_intent = json.loads(engine.argv[intent_index])["task_intent"]
        self.assertEqual(
            task_intent["intentSteps"],
            [
                {"order": 1, "action": "GotoObject", "objectType": "Tomato", "targetType": None},
                {"order": 2, "action": "PickupObject", "objectType": "Tomato", "targetType": None},
                {"order": 3, "action": "GotoObject", "objectType": "Fridge", "targetType": None},
                {"order": 4, "action": "OpenObject", "objectType": "Fridge", "targetType": None},
                {"order": 5, "action": "PutObject", "objectType": "Tomato", "targetType": "Fridge"},
                {"order": 6, "action": "CloseObject", "objectType": "Fridge", "targetType": None},
            ],
        )

    def test_structured_place_does_not_use_backend_missing_core_put_action(self):
        old_fetch = task_execution_server.fetch_receiver_state_object_types
        task_execution_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Bread", "Fridge"], None)
        try:
            engine = FakeEngine()
            backend = FakeNormalizerBackend(
                {
                    "normalized_task": "pick up the Bread and go to the Fridge.",
                    "intentSteps": [
                        {"order": 1, "action": "GotoObject", "objectType": "Bread", "targetType": None},
                        {"order": 2, "action": "PickupObject", "objectType": "Bread", "targetType": None},
                        {"order": 3, "action": "GotoObject", "objectType": "Fridge", "targetType": None},
                    ],
                    "confidence": "high",
                    "reason": "forgot the placement step",
                }
            )
            service = TaskExecutionService(engine, backend, service_config())
            with contextlib.redirect_stdout(io.StringIO()):
                response = service.execute_task(
                    {
                        "task_id": "missing-put",
                        "task": "Place the bread in the fridge Put the bread in the fridge. place bread in",
                        "subtask": {
                            "name": "Place the bread in the fridge",
                            "description": "Put the bread in the fridge.",
                            "action": "place",
                        },
                    }
                )
        finally:
            task_execution_server.fetch_receiver_state_object_types = old_fetch

        self.assertEqual(response["status"], "success")
        self.assertEqual(backend.calls, [])
        self.assertEqual(
            [step["action"] for step in response["task_normalization"]["intentSteps"]],
            ["GotoObject", "PickupObject", "GotoObject", "OpenObject", "PutObject", "CloseObject"],
        )

    def test_rejects_unrelated_helper_action_for_put_task(self):
        old_fetch = task_execution_server.fetch_receiver_state_object_types
        task_execution_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Tomato", "CounterTop", "Apple"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                backend = FakeNormalizerBackend(
                    {
                        "normalized_task": "pick up the Tomato and put it on the CounterTop.",
                        "intentSteps": [
                            {"order": 1, "action": "GotoObject", "objectType": "Apple", "targetType": None},
                            {"order": 2, "action": "PickupObject", "objectType": "Tomato", "targetType": None},
                            {"order": 3, "action": "PutObject", "objectType": "Tomato", "targetType": "CounterTop"},
                        ],
                        "confidence": "high",
                        "reason": "badly inserted unrelated navigation",
                    }
                )
                service = TaskExecutionService(engine, backend, service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task(
                        {"task_id": "bad-extra-goto", "task": "pick up the tomato and put it on the counter"}
                    )
        finally:
            task_execution_server.fetch_receiver_state_object_types = old_fetch

        self.assertIsNone(engine.argv)
        self.assertEqual(response["failure_code"], "task_normalization_failed")
        self.assertIn("added unrequested action", " ".join(response["task_normalization"]["warnings"]))

    def test_rejects_normalizer_that_changes_recognized_look_down_action(self):
        old_fetch = task_execution_server.fetch_receiver_state_object_types
        task_execution_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Fridge"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                backend = FakeNormalizerBackend(
                    {
                        "normalized_task": "open the Fridge and close the Fridge.",
                        "intentSteps": [
                            {"order": 1, "action": "OpenObject", "objectType": "Fridge", "targetType": None},
                            {"order": 2, "action": "CloseObject", "objectType": "Fridge", "targetType": None},
                        ],
                        "confidence": "high",
                        "reason": "bad rewrite",
                    }
                )
                service = TaskExecutionService(engine, backend, service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task({"task_id": "bad-look-down", "task": "open the fridge and look down"})
        finally:
            task_execution_server.fetch_receiver_state_object_types = old_fetch

        self.assertIsNone(engine.argv)
        self.assertEqual(response["status"], "needs_upstream_planning")
        self.assertEqual(response["failure_code"], "task_normalization_failed")
        self.assertFalse(response["task_normalization"]["used"])
        self.assertIn("changed recognized action LookDown to CloseObject", " ".join(response["task_normalization"]["warnings"]))

    def test_rejects_normalized_task_with_unknown_object_type(self):
        old_fetch = task_execution_server.fetch_receiver_state_object_types
        task_execution_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Fridge"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                backend = FakeNormalizerBackend(
                    {
                        "normalized_task": "go to the Moon.",
                        "action": "GotoObject",
                        "object_type": "Moon",
                        "target_type": None,
                        "confidence": "high",
                        "reason": "bad object",
                    }
                )
                service = TaskExecutionService(engine, backend, service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task({"task_id": "bad-target", "task": "find the moon"})
        finally:
            task_execution_server.fetch_receiver_state_object_types = old_fetch

        self.assertIsNone(engine.argv)
        self.assertEqual(response["status"], "needs_upstream_planning")
        self.assertEqual(response["failure_code"], "task_normalization_failed")
        self.assertFalse(response["task_normalization"]["used"])
        self.assertIn("requestedObjectType", " ".join(response["task_normalization"]["warnings"]))

    def test_model_shard_error_reports_missing_safetensors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "model.safetensors-00001-of-00002.safetensors"}}),
                encoding="utf-8",
            )
            error = model_shard_error(str(model_dir))
        self.assertIsNotNone(error)
        self.assertIn("missing safetensors shard", error)
        self.assertIn("/225010231/mwl/Linhao/models/Qwen3.5-4B", error)

    def test_rejects_upstream_known_robot_ids_without_invoking_runtime(self):
        engine = FakeEngine()
        service = TaskExecutionService(engine, FakeNormalizerBackend({}), service_config())

        with self.assertRaisesRegex(ValueError, "known_robot_ids"):
            service.execute_task({"task": "pick up bread", "known_robot_ids": [0, 1]})

        self.assertIsNone(engine.argv)

    def test_surfaces_engine_error_when_no_json_is_produced(self):
        class FailingEngine(FakeEngine):
            def run(self, args):
                print("error: receiver unavailable", file=__import__("sys").stderr)
                return 1

        old_fetch = task_execution_server.fetch_receiver_state_object_types
        task_execution_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Bread"], None)
        try:
            service = TaskExecutionService(FailingEngine(), FakeNormalizerBackend({
                    "normalized_task": "pick up the Bread.",
                    "intentSteps": [
                        {"order": 1, "action": "PickupObject", "objectType": "Bread", "targetType": None},
                    ],
                    "confidence": "high",
                    "reason": "the task asks to pick up Bread",
                }), service_config())
            with self.assertRaisesRegex(RuntimeError, "receiver unavailable"):
                service.execute_task({"task": "pick up bread"})
        finally:
            task_execution_server.fetch_receiver_state_object_types = old_fetch


    def test_semantic_table_selects_reachable_receptacle_and_preserves_object_id(self):
        state = {"objects": [
            {"objectId": "DiningTable|2", "objectType": "DiningTable", "receptacle": True, "distance": 2.0},
            {"objectId": "CoffeeTable|1", "objectType": "CoffeeTable", "receptacle": True, "distance": 1.0},
            {"objectId": "SideTable|0", "objectType": "SideTable", "receptacle": True, "distance": 0.5},
            {"objectId": "CoffeeTable|bad", "objectType": "CoffeeTable", "receptacle": False, "distance": 0.1},
        ]}

        def route(object_id):
            return {"status": "success", "actions": [1, 2]} if object_id == "CoffeeTable|1" else {"status": "success", "actions": [1, 2, 3]}

        resolution = task_execution_server.resolve_semantic_receptacle("table", state, route_planner=route)
        self.assertEqual(resolution["chosen_type"], "CoffeeTable")
        self.assertEqual(resolution["chosen_object_id"], "CoffeeTable|1")
        self.assertNotIn("CoffeeTable|bad", [item["objectId"] for item in resolution["candidates"]])
        self.assertIsNone(task_execution_server.resolve_object_type("table", ["DiningTable", "CoffeeTable"])["canonical"])

        subtask = {"action": "place", "name": "Put vase on table", "grounding": {"object_tags": ["vase", "table"], "source_object_tags": ["vase"], "destination_object_tags": ["table"]}}
        augmented, _ = task_execution_server.prepare_semantic_destination(subtask, state, route_planner=route)
        self.assertEqual(augmented["grounding"]["destination_object_ids"], ["CoffeeTable|1"])
        normalized = task_execution_server.normalize_structured_subtask(augmented, "Put the vase on the table", ["Vase", "DiningTable", "CoffeeTable", "SideTable"])
        self.assertIsNotNone(normalized)
        steps = normalized["intentSteps"]
        self.assertEqual({step.get("targetObjectId") for step in steps if step["action"] == "PutObject"}, {"CoffeeTable|1"})
        self.assertEqual({step.get("targetType") for step in steps if step["action"] == "PutObject"}, {"CoffeeTable"})
        self.assertNotIn("Table", [step.get("objectType") for step in steps] + [step.get("targetType") for step in steps])

    def test_semantic_table_rejects_when_no_candidate_is_reachable(self):
        state = {"objects": [{"objectId": "SideTable|0", "objectType": "SideTable", "receptacle": True}]}
        resolution = task_execution_server.resolve_semantic_receptacle("table", state, route_planner=lambda _: {"status": "failed"})
        self.assertEqual(resolution["status"], "unresolved")
        self.assertIsNone(resolution["chosen_object_id"])

    def test_place_subtask_with_unreachable_semantic_destination_still_fails(self):
        state = {
            "objects": [
                {"objectId": "SideTable|0", "objectType": "SideTable", "receptacle": True},
            ]
        }
        engine = FakeEngine()
        backend = FakeNormalizerBackend({})
        with patch.object(
            task_execution_server,
            "fetch_receiver_state_object_types",
            return_value=(["Vase", "SideTable"], None),
        ), patch.object(
            task_execution_server,
            "fetch_receiver_state",
            return_value=(state, None),
        ), patch.object(
            task_execution_server,
            "post_receiver_json",
            return_value={"status": "failed"},
        ):
            service = TaskExecutionService(engine, backend, service_config())
            response = service.execute_task(
                {
                    "task_id": "place-vase-unreachable-table",
                    "task": "Put vase on table",
                    "subtask": {
                        "id": "T1",
                        "name": "Put vase on table",
                        "action": "place",
                        "grounding": {
                            "object_tags": ["vase", "table"],
                            "source_object_tags": ["vase"],
                            "source_object_ids": ["Vase|+00.00|+00.00|+00.00"],
                            "destination_object_tags": ["table"],
                        },
                    },
                }
            )

        self.assertEqual(response["status"], "needs_upstream_planning")
        self.assertEqual(response["failure_code"], "semantic_target_unresolved")
        self.assertEqual(response["task_normalization"]["semantic_resolution"]["semantic_input"], "table")
