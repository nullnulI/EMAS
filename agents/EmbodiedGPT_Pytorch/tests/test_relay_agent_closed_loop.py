from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest

from demo import auto_scene_actions as module


class FakeJsonRelayBackend:
    def __init__(self, outputs: list[dict]):
        self.outputs = [json.dumps(output) for output in outputs]
        self.calls: list[dict] = []

    def generate_messages(self, messages, *, tools=None, deterministic=False):
        if not deterministic:
            raise AssertionError("closed-loop relay must use deterministic JSON generation")
        if not self.outputs:
            raise AssertionError("unexpected relay-agent turn")
        self.calls.append({"messages": messages, "tools": tools})
        return self.outputs.pop(0)


class RelayAgentClosedLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.originals = {
            "probe_scene": module.probe_scene,
            "post_json": module.post_json,
            "execute_actions_probe_scene": module.execute_actions_probe_scene,
            "generate_semantic_plan": module.generate_semantic_plan,
            "generate_task_intent_tool_call": module.generate_task_intent_tool_call,
            "send_actions": module.send_actions,
        }

    def tearDown(self) -> None:
        for name, value in self.originals.items():
            setattr(module, name, value)

    def test_structured_open_binds_exact_visible_drawer_without_vlm(self) -> None:
        drawer_a = {
            "id": "Drawer|A",
            "type": "Drawer",
            "visible": True,
            "openable": True,
            "isOpen": True,
        }
        drawer_b = {
            "id": "Drawer|B",
            "type": "Drawer",
            "visible": True,
            "openable": True,
            "isOpen": False,
        }
        step = {
            "action": "OpenObject",
            "objectType": "Drawer",
            "objectId": "Drawer|B",
        }
        args = argparse.Namespace(_task_intent_source="upstream_structured_task")
        observation = {"objects": [drawer_a, drawer_b]}

        action = module.deterministic_structured_interaction_action(args, step, observation)

        self.assertEqual(
            action,
            {"action": "OpenObject", "objectId": "Drawer|B", "forceAction": True},
        )
        self.assertFalse(module.step_already_satisfied(step, observation))
        self.assertTrue(
            module.step_already_satisfied(
                {**step, "objectId": "Drawer|A"},
                observation,
            )
        )

    def test_structured_open_rejects_invisible_exact_drawer(self) -> None:
        args = argparse.Namespace(_task_intent_source="upstream_structured_task")
        step = {
            "action": "OpenObject",
            "objectType": "Drawer",
            "objectId": "Drawer|B",
        }
        observation = {
            "objects": [
                {
                    "id": "Drawer|A",
                    "type": "Drawer",
                    "visible": True,
                    "openable": True,
                    "isOpen": False,
                },
                {
                    "id": "Drawer|B",
                    "type": "Drawer",
                    "visible": False,
                    "openable": True,
                    "isOpen": False,
                },
            ]
        }

        self.assertIsNone(
            module.deterministic_structured_interaction_action(args, step, observation)
        )

    def test_exact_open_postcondition_uses_target_instance(self) -> None:
        step = {
            "action": "OpenObject",
            "objectType": "Drawer",
            "objectId": "Drawer|B",
        }
        observation = {
            "objects": [
                {"id": "Drawer|A", "type": "Drawer", "isOpen": True},
                {"id": "Drawer|B", "type": "Drawer", "isOpen": False},
            ]
        }

        failure = module.interaction_state_postcondition_failure(step, observation)
        self.assertEqual(failure[0], "interaction_postcondition_failed")
        observation["objects"][1]["isOpen"] = True
        self.assertIsNone(module.interaction_state_postcondition_failure(step, observation))

    def test_goto_requires_and_confirms_exact_visible_object(self) -> None:
        args = argparse.Namespace(
            primary_robot_id=0,
            dry_run=False,
            goto_max_actions=None,
            goto_min_distance=None,
            goto_max_distance=None,
        )
        payload = module.goto_payload_for_navigation_task(
            args,
            "task-1",
            "Drawer",
            object_id="Drawer|B",
        )
        self.assertTrue(payload["require_target_visible"])
        self.assertEqual(payload["object_id"], "Drawer|B")
        failure = module.goto_postcondition_failure(
            {
                "status": "success",
                "post_target_visible": True,
                "post_target_object_id": "Drawer|A",
            },
            requested_object_id="Drawer|B",
            require_execution=True,
        )
        self.assertEqual(failure[0], "goto_target_object_mismatch")

    def test_pickup_and_put_remain_on_selected_peer(self) -> None:
        counter = {
            "id": "CounterTop|1",
            "type": "CounterTop",
            "visible": True,
            "receptacle": True,
        }
        tomato = {
            "id": "Tomato|2",
            "type": "Tomato",
            "visible": True,
            "pickupable": True,
        }
        primary_probe = {
            "selected_robot_id": 0,
            "robots": [{"robot_id": 0}, {"robot_id": 2}],
            "objects": [counter],
            "image_base64": "eA==",
        }
        peer_probe = {
            "robot_id": 2,
            "objects": [tomato, counter],
            "image_base64": "eA==",
        }
        task_intent = {
            "requestedAction": "PutObject",
            "requestedObjectType": "Tomato",
            "intentSteps": [
                {"order": 1, "action": "PickupObject", "objectType": "Tomato", "targetType": None},
                {
                    "order": 2,
                    "action": "PutObject",
                    "objectType": "Tomato",
                    "targetType": "CounterTop",
                },
            ],
        }
        tool_call = {"name": "extract_task_intent", "arguments": {"task": "put the tomato on the counter."}}
        sent: list[dict] = []
        held_state: dict[str, dict | None] = {"object": None}

        module.probe_scene = lambda *args, **kwargs: primary_probe
        module.execute_actions_probe_scene = lambda *args, **kwargs: {
            **peer_probe,
            "robot": {"held_object": held_state["object"]},
        }
        module.generate_task_intent_tool_call = lambda *args, **kwargs: (
            tool_call,
            task_intent,
            {"status": "ok", "warnings": []},
        )

        def generate_step_plan(args, image_path, objects, task_id):
            intent = args._task_intent
            step = intent["intentSteps"][0]
            return (
                "{}",
                {
                    "task": args.task,
                    "targetObjectType": step["objectType"],
                    "needsGrounding": True,
                    "observations": [],
                    "plan": [step],
                },
                None,
            )

        module.generate_semantic_plan = generate_step_plan

        def send_actions(url, payload, timeout):
            sent.append(payload)
            action = payload["actions"][0]["action"]
            held_object = (
                {"objectId": "Tomato|2", "objectType": "Tomato"}
                if action == "PickupObject"
                else None
            )
            held_state["object"] = held_object
            return json.dumps(
                {
                    "status": "success",
                    "robot_id": 2,
                    "robot": {"held_object": held_object},
                    "objects": [tomato, counter],
                    "image_base64": "eA==",
                    "results": [
                        {
                            "robot_id": 2,
                            "success": True,
                            "robot": {"held_object": held_object},
                        }
                    ],
                }
            )

        module.send_actions = send_actions

        relay_outputs = [
            {
                "name": "select_executor",
                "arguments": {"robot_id": 2, "reason": "robot 2 sees a pickupable tomato"},
            },
            {
                "name": "select_executor",
                "arguments": {"robot_id": 2, "reason": "robot 2 holds the tomato and sees the counter"},
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            args = module.parse_args(
                [
                    "--execute-actions-url",
                    "http://127.0.0.1:1/execute_actions",
                    "--task",
                    "put the tomato on the counter.",
                    "--task-id",
                    "task-1",
                    "--output-dir",
                    temp_dir,
                    "--relay-mode",
                    "--closed-loop-replan",
                ]
            )
            backend = FakeJsonRelayBackend(relay_outputs)
            args._qwen_backend = backend
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = module.run(args)

        output = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(output["closed_loop_result"], {"status": "success", "step_count": 2}, json.dumps(output, indent=2))
        self.assertEqual([payload["robot_id"] for payload in sent], [2, 2, 2])
        self.assertEqual(
            [payload["actions"][0]["action"] for payload in sent],
            ["PickupObject", "PutObject", "Done"],
        )
        self.assertIn("relay_explanation", output["closed_loop_trace"][0])
        self.assertIn("robot 0 cannot execute PickupObject Tomato", output["closed_loop_trace"][0]["relay_explanation"]["primary_inability_reason"])
        self.assertIn("coordination succeeded: robot_2 selected", output["closed_loop_trace"][0]["relay_explanation"]["coordination_explanation"])
        self.assertIn("relay_explanation", output["closed_loop_trace"][1])
        self.assertIn("robot 0 cannot execute PutObject Tomato -> CounterTop", output["closed_loop_trace"][1]["relay_explanation"]["primary_inability_reason"])
        stderr_text = stderr.getvalue()
        self.assertIn("[relay] primary cannot execute", stderr_text)
        self.assertIn("[relay] coordination succeeded: robot_2 selected", stderr_text)
        self.assertIn("[relay] candidates:", stderr_text)
        self.assertTrue(
            all(
                [tool["function"]["name"] for tool in call["tools"]]
                == ["select_executor", "report_failure"]
                for call in backend.calls
            )
        )

    def test_task7_failed_local_knife_navigation_hands_slice_to_holder(self) -> None:
        lettuce = {
            "id": "Lettuce|1",
            "objectId": "Lettuce|1",
            "type": "Lettuce",
            "objectType": "Lettuce",
            "visible": True,
            "sliceable": True,
        }
        knife = {
            "id": "Knife|1",
            "objectId": "Knife|1",
            "type": "Knife",
            "objectType": "Knife",
            "visible": True,
            "pickupable": True,
        }
        butter_knife = {
            "objectId": "ButterKnife|1",
            "objectType": "ButterKnife",
        }
        primary_probe = {
            "selected_robot_id": 1,
            "robots": [{"robot_id": 0}, {"robot_id": 1}],
            "objects": [lettuce, knife],
            "image_base64": "eA==",
            "robot": {"held_object": None},
        }
        holder_probe = {
            "robot_id": 0,
            "objects": [lettuce],
            "image_base64": "eA==",
            "robot": {"held_object": butter_knife},
            "held_object": butter_knife,
        }
        task_intent = {
            "requestedAction": "SliceObject",
            "requestedObjectType": "Lettuce",
            "intentSteps": [
                {"order": 1, "action": "SliceObject", "objectType": "Lettuce"},
            ],
        }
        module.probe_scene = lambda *args, **kwargs: (
            holder_probe if kwargs.get("primary_robot_id") == 0 else primary_probe
        )
        module.execute_actions_probe_scene = lambda *args, **kwargs: (
            holder_probe if kwargs.get("robot_id") == 0 else primary_probe
        )
        module.generate_task_intent_tool_call = lambda *args, **kwargs: (
            {"name": "extract_task_intent", "arguments": {"task": "slice the lettuce"}},
            task_intent,
            {"status": "ok", "warnings": []},
        )
        goto_calls: list[dict] = []

        def fake_post_json(url, payload, timeout):
            goto_calls.append(payload)
            if payload["robot_id"] == 1 and payload.get("object_id") == "Knife|1":
                return {
                    "status": "failed",
                    "error_code": "no_interactable_pose",
                    "error": "no interactable pose for Knife|1",
                }
            return {
                "status": "success",
                "robot_id": payload["robot_id"],
                "post_target_visible": True,
                "post_target_object_id": payload.get("object_id") or "Lettuce|1",
                "execute_result": {"status": "success", "results": []},
            }

        module.post_json = fake_post_json
        sent: list[dict] = []

        def fake_send_actions(url, payload, timeout):
            sent.append(payload)
            action = payload["actions"][0]["action"]
            return json.dumps({
                "status": "success",
                "robot_id": payload.get("robot_id"),
                "robot": {"held_object": butter_knife if payload.get("robot_id") == 0 else None},
                "held_object": butter_knife if payload.get("robot_id") == 0 else None,
                "objects": [lettuce],
                "image_base64": "eA==",
                "results": [{
                    "action": action,
                    "success": True,
                    "robot_id": payload.get("robot_id"),
                }],
            })

        module.send_actions = fake_send_actions
        backend = FakeJsonRelayBackend([
            {
                "name": "select_executor",
                "arguments": {
                    "robot_id": 0,
                    "reason": "robot 0 already holds ButterKnife",
                },
            },
        ])

        with tempfile.TemporaryDirectory() as temp_dir:
            args = module.parse_args([
                "--execute-actions-url", "http://127.0.0.1:1/execute_actions",
                "--task", "slice the lettuce",
                "--task-id", "task7-lettuce",
                "--output-dir", temp_dir,
                "--primary-robot-id", "1",
                "--known-robot-ids", "0,1",
                "--relay-mode",
                "--closed-loop-replan",
            ])
            args._qwen_backend = backend
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = module.run(args)

        self.assertTrue(stdout.getvalue(), stderr.getvalue())
        output = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(output["closed_loop_result"]["status"], "success")
        self.assertEqual(goto_calls[0]["robot_id"], 1)
        self.assertEqual(goto_calls[0]["object_id"], "Knife|1")
        self.assertTrue(any(call["robot_id"] == 0 for call in goto_calls[1:]))
        slice_payload = next(
            payload for payload in sent
            if payload["actions"][0]["action"] == "SliceObject"
        )
        self.assertEqual(slice_payload["robot_id"], 0)
        self.assertFalse(any(
            payload["actions"][0]["action"] == "PickupObject"
            and payload.get("robot_id") == 0
            for payload in sent
        ))
        recovery = next(
            trace for trace in output["closed_loop_trace"]
            if trace.get("resource_failure")
        )
        self.assertEqual(
            recovery["resource_failure"]["failure_code"],
            "resource_acquisition_failed",
        )
        self.assertEqual(recovery["relay_result"]["executor_robot_id"], 0)


if __name__ == "__main__":
    unittest.main()
